#!/usr/bin/env python3
"""Run an optional two-tenant interference scenario with separate client artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_matrix


def start_client(
    config: dict[str, Any],
    client: dict[str, Any],
    output_root: Path,
    scenario_id: str,
) -> tuple[subprocess.Popen[str], Any, Any]:
    client_config = dict(config)
    client_config["requests"] = int(client["requests"])
    client_config["warmup"] = int(client.get("warmup", config.get("warmup", 20)))
    target = dict(config["target"])
    target["label"] = client["label"]
    target["auth_token_env"] = client["auth_token_env"]
    output = output_root / scenario_id / client["label"] / "client-result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = run_matrix.build_command(
        client_config,
        target,
        "non_streaming",
        int(client["concurrency"]),
        1,
        1,
        output,
        f"{scenario_id}-{client['label']}",
    )
    environment = os.environ.copy()
    environment.pop("BENCH_AUTH_TOKEN", None)
    credential_name = client["auth_token_env"]
    if credential_name not in os.environ:
        raise ValueError(f"required credential environment variable {credential_name} is unset")
    environment["BENCH_AUTH_TOKEN"] = os.environ[credential_name]
    stdout = (output.parent / "stdout.log").open("w", encoding="utf-8")
    stderr = (output.parent / "stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=run_matrix.BENCHMARK_ROOT,
        env=environment,
        text=True,
        stdout=stdout,
        stderr=stderr,
    )
    (output.parent / "process.json").write_text(
        json.dumps(
            {
                "started_utc": datetime.now(UTC).isoformat(),
                "command": command,
                "credential_source": "environment",
                "scenario": "tenant_isolation",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return process, stdout, stderr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=run_matrix.BENCHMARK_ROOT / "results")
    arguments = parser.parse_args()
    processes: list[tuple[str, subprocess.Popen[str], Any, Any]] = []
    try:
        config = run_matrix.expand(json.loads(arguments.config.read_text(encoding="utf-8")))
        if config.get("schema_version") != 1 or len(config.get("clients", [])) != 2:
            raise ValueError("tenant scenario requires schema_version 1 and exactly two clients")
        run_matrix.validate_admission(config["target"])
        scenario_id = config.get("run_id") or datetime.now(UTC).strftime("tenant-%Y%m%dT%H%M%SZ")
        for index, client in enumerate(config["clients"]):
            if index:
                time.sleep(float(config.get("probe_start_delay_seconds", 1)))
            process, stdout, stderr = start_client(
                config, client, arguments.output_root, scenario_id
            )
            processes.append((client["label"], process, stdout, stderr))
        failed = []
        for label, process, stdout, stderr in processes:
            status = process.wait()
            stdout.close()
            stderr.close()
            if status:
                failed.append(f"{label} exit={status}")
        if failed:
            raise RuntimeError(", ".join(failed))
    except (OSError, ValueError, RuntimeError) as error:
        for _, process, stdout, stderr in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if not stdout.closed:
                stdout.close()
            if not stderr.closed:
                stderr.close()
        print(f"run_tenant_isolation.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
