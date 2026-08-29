#!/usr/bin/env python3
"""Run deterministic, rotating benchmark target matrices without persisting secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import validate_result

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunCoordinate:
    """Expected identity and request accounting for one matrix run."""

    run_id: str
    label: str
    mode: str
    concurrency: int
    repeat: int
    execution_order: int
    requests: int


def expand(value: Any) -> Any:
    """Expand ${VARS}, rejecting unresolved values instead of silently drifting."""
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"unresolved environment variable in {value!r}")
        return expanded
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = expand(json.load(file))
    if config.get("schema_version") != 1:
        raise ValueError("matrix schema_version must be 1")
    required = {"model", "dataset", "targets", "concurrency", "repetitions"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing matrix fields: {', '.join(missing)}")
    labels = [target.get("label") for target in config["targets"]]
    if len(labels) != len(set(labels)) or any(not label for label in labels):
        raise ValueError("target labels must be present and unique")
    for target in config["targets"]:
        validate_admission(target)
    return config


def validate_admission(target: dict[str, Any]) -> None:
    if str(target.get("batching_enabled", "not_applicable")) != "true":
        return
    size = int(target.get("batch_max_size", 0))
    tenant = int(target.get("tenant_max_inflight", 0))
    global_limit = int(target.get("global_max_inflight", 0))
    if size < 2 or tenant < size or global_limit < size:
        raise ValueError(
            f"{target.get('label')}: batching admission limits must be at least batch_max_size"
        )


def rotated(targets: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
    offset = (repeat - 1) % len(targets)
    return targets[offset:] + targets[:offset]


def requests_for(config: dict[str, Any], concurrency: int) -> int:
    configured = config.get("requests", "auto")
    return max(200, concurrency * 20) if configured == "auto" else int(configured)


def add_optional(command: list[str], flag_name: str, value: Any) -> None:
    if value is not None and value != "":
        command.extend([flag_name, str(value)])


def build_command(
    config: dict[str, Any],
    target: dict[str, Any],
    mode: str,
    concurrency: int,
    repeat: int,
    order: int,
    output: Path,
    run_id: str,
) -> list[str]:
    generation = config.get("generation", {})
    command = [
        "go",
        "run",
        "./cmd/gateway-bench",
        "--base-url",
        str(target["base_url"]),
        "--endpoint",
        str(config.get("endpoint", "/v1/chat/completions")),
        "--model",
        str(config["model"]),
        "--dataset",
        str(resolve_path(str(config["dataset"]))),
        "--mode",
        mode,
        "--concurrency",
        str(concurrency),
        "--requests",
        str(requests_for(config, concurrency)),
        "--warmup",
        str(config.get("warmup", 20)),
        "--timeout",
        str(config.get("timeout", "120s")),
        "--label",
        str(target["label"]),
        "--output",
        str(output),
        "--temperature",
        str(generation.get("temperature", 0)),
        "--top-p",
        str(generation.get("top_p", 1)),
        "--max-tokens",
        str(generation.get("max_tokens", 128)),
        "--seed",
        str(generation.get("seed", 1)),
        "--run-id",
        run_id,
        "--repeat",
        str(repeat),
        "--execution-order",
        str(order),
        "--batching-enabled",
        str(target.get("batching_enabled", "not_applicable")),
        "--prefix-caching",
        str(config.get("prefix_caching", "unknown")),
        "--vllm-version",
        str(config.get("vllm_version", "unknown")),
    ]
    for flag_name, key in [
        ("--gateway-metrics-url", "gateway_metrics_url"),
        ("--vllm-metrics-url", "vllm_metrics_url"),
        ("--output-token-counter", "output_token_counter"),
        ("--batch-max-size", "batch_max_size"),
        ("--batch-max-wait-seconds", "batch_max_wait_seconds"),
        ("--tenant-max-inflight", "tenant_max_inflight"),
        ("--global-max-inflight", "global_max_inflight"),
        ("--gateway-git-sha", "gateway_git_sha"),
    ]:
        add_optional(command, flag_name, target.get(key, config.get(key)))
    for flag_name, key in [
        ("--sample-interval", "sample_interval"),
        ("--gpu-sample-interval", "gpu_sample_interval"),
        ("--vllm-config", "vllm_config"),
        ("--environment", "environment"),
    ]:
        value = config.get(key)
        if key in {"vllm_config", "environment"} and value:
            value = resolve_path(str(value))
        add_optional(command, flag_name, value)
    gateway_config = target.get("gateway_config")
    if gateway_config:
        add_optional(command, "--gateway-config", resolve_path(str(gateway_config)))
    return command


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BENCHMARK_ROOT / path


def safe_command(command: list[str]) -> str:
    # Credentials never enter argv. This representation is safe for stdout metadata.
    return shlex.join(command)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def referenced_input_hashes(config: dict[str, Any], target: dict[str, Any]) -> dict[str, str]:
    """Hash every file input whose contents can affect a retained result."""
    references = {
        "dataset": config.get("dataset"),
        "environment": config.get("environment"),
        "gateway_config": target.get("gateway_config"),
        "vllm_config": config.get("vllm_config"),
    }
    return {
        name: sha256_file(resolve_path(str(value))) for name, value in references.items() if value
    }


def resume_plan_sha256(
    command: list[str],
    expected: RunCoordinate,
    input_hashes: dict[str, str],
    credential_env_name: str | None,
) -> str:
    """Fingerprint the secret-free execution plan and content-addressed inputs."""
    plan = {
        "schema_version": 1,
        "command": command,
        "coordinate": asdict(expected),
        "input_sha256": input_hashes,
        "credential_env_name": credential_env_name,
    }
    normalized = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(normalized).hexdigest()


def planned_result_configuration(
    config: dict[str, Any],
    target: dict[str, Any],
    mode: str,
    concurrency: int,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    generation = config.get("generation", {})
    expected = {
        "base_url": str(target["base_url"]),
        "endpoint": str(config.get("endpoint", "/v1/chat/completions")),
        "model": str(config["model"]),
        "mode": mode,
        "concurrency": concurrency,
        "requests": requests_for(config, concurrency),
        "warmup": int(config.get("warmup", 20)),
        "dataset_sha256": input_hashes["dataset"],
        "temperature": generation.get("temperature", 0),
        "top_p": generation.get("top_p", 1),
        "max_tokens": int(generation.get("max_tokens", 128)),
        "seed": int(generation.get("seed", 1)),
        "n": 1,
        "stream": mode == "streaming",
        "stream_include_usage": mode == "streaming",
        "batching_enabled": str(target.get("batching_enabled", "not_applicable")),
        "prefix_caching": str(config.get("prefix_caching", "unknown")),
    }
    for field, convert in [
        ("batch_max_size", int),
        ("batch_max_wait_seconds", float),
        ("tenant_max_inflight", int),
        ("global_max_inflight", int),
    ]:
        value = target.get(field, config.get(field))
        if value is not None and value != "":
            expected[field] = convert(value)
    return expected


def execute(
    command: list[str],
    target: dict[str, Any],
    output: Path,
    dry_run: bool,
    plan_sha256: str,
) -> None:
    print(safe_command(command), flush=True)
    if dry_run:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("BENCH_AUTH_TOKEN", None)
    credential_name = target.get("auth_token_env")
    if credential_name:
        if credential_name not in os.environ:
            raise ValueError(f"required credential environment variable {credential_name} is unset")
        environment["BENCH_AUTH_TOKEN"] = os.environ[credential_name]
    started = datetime.now(UTC)
    completed = subprocess.run(
        command,
        cwd=BENCHMARK_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    (output.parent / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output.parent / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    metadata = {
        "started_utc": started.isoformat(),
        "ended_utc": datetime.now(UTC).isoformat(),
        "exit_code": completed.returncode,
        "command": command,
        "resume_plan_sha256": plan_sha256,
        "credential_source": "environment" if credential_name else "none",
    }
    (output.parent / "process.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(f"benchmark failed for {target['label']}; see {output.parent}")


def completed_run(
    output: Path,
    command: list[str],
    expected: RunCoordinate,
    expected_configuration: dict[str, Any],
    plan_sha256: str,
) -> bool:
    """Accept only a valid, successful result matching the exact planned run."""
    process = output.parent / "process.json"
    if not output.is_file() or not process.is_file():
        return False
    try:
        with output.open(encoding="utf-8") as file:
            result = json.load(file)
        with process.open(encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(result, dict) or not isinstance(metadata, dict):
        return False
    if validate_result.validate(result):
        return False
    result_metadata = result.get("metadata", {})
    configuration = result.get("configuration", {})
    summary = result.get("summary", {})
    per_request = result.get("per_request", [])
    validity = result.get("validity", {})
    return (
        metadata.get("exit_code") == 0
        and metadata.get("command") == command
        and metadata.get("resume_plan_sha256") == plan_sha256
        and result_metadata.get("run_id") == expected.run_id
        and result_metadata.get("label") == expected.label
        and result_metadata.get("repeat") == expected.repeat
        and result_metadata.get("execution_order") == expected.execution_order
        and configuration.get("mode") == expected.mode
        and configuration.get("concurrency") == expected.concurrency
        and configuration.get("requests") == expected.requests
        and all(
            configuration.get(field) == value for field, value in expected_configuration.items()
        )
        and summary.get("attempted") == expected.requests
        and summary.get("successful") == expected.requests
        and summary.get("failed") == 0
        and isinstance(per_request, list)
        and len(per_request) == expected.requests
        and validity.get("valid") is True
    )


def run_matrix(
    config: dict[str, Any],
    output_root: Path,
    selected: set[str],
    dry_run: bool,
    resume: bool = False,
) -> None:
    matrix_id = config.get("run_id") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    targets = [
        target for target in config["targets"] if not selected or target["label"] in selected
    ]
    if not targets:
        raise ValueError("no selected targets")
    modes = config.get("modes", ["non_streaming"])
    settling = float(config.get("settle_seconds", 5))
    first = True
    for mode in modes:
        if mode not in {"streaming", "non_streaming"}:
            raise ValueError(f"invalid mode {mode}")
        for concurrency in config["concurrency"]:
            for repeat in range(1, int(config["repetitions"]) + 1):
                for order, target in enumerate(rotated(targets, repeat), start=1):
                    run_id = f"{matrix_id}-{mode}-c{concurrency}-r{repeat}-{target['label']}"
                    request_count = requests_for(config, int(concurrency))
                    output = (
                        output_root
                        / matrix_id
                        / mode
                        / f"c{concurrency}"
                        / f"repeat-{repeat}"
                        / target["label"]
                        / "client-result.json"
                    )
                    command = build_command(
                        config, target, mode, int(concurrency), repeat, order, output, run_id
                    )
                    expected = RunCoordinate(
                        run_id=run_id,
                        label=str(target["label"]),
                        mode=mode,
                        concurrency=int(concurrency),
                        repeat=repeat,
                        execution_order=order,
                        requests=request_count,
                    )
                    input_hashes = referenced_input_hashes(config, target)
                    expected_configuration = planned_result_configuration(
                        config, target, mode, int(concurrency), input_hashes
                    )
                    plan_sha256 = resume_plan_sha256(
                        command,
                        expected,
                        input_hashes,
                        target.get("auth_token_env"),
                    )
                    if resume and completed_run(
                        output,
                        command,
                        expected,
                        expected_configuration,
                        plan_sha256,
                    ):
                        print(f"skipping completed run: {output}", flush=True)
                        continue
                    if not first and settling > 0 and not dry_run:
                        time.sleep(settling)
                    first = False
                    execute(command, target, output, dry_run, plan_sha256)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=BENCHMARK_ROOT / "results")
    parser.add_argument("--target", action="append", default=[], help="run only this label")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip complete, valid, successful runs matching the planned command",
    )
    arguments = parser.parse_args()
    try:
        config = load_config(arguments.config)
        run_matrix(
            config,
            arguments.output_root,
            set(arguments.target),
            arguments.dry_run,
            arguments.resume,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"run_matrix.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
