#!/usr/bin/env python3
"""Validate raw repetitions and create transparent A/B/C CSV and comparison reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from validate_result import validate

ARMS = ("direct", "gateway_no_batch", "gateway_batch")
SAMPLE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')

CSV_FIELDS = [
    "run_id",
    "repeat",
    "target",
    "mode",
    "concurrency",
    "requests",
    "success",
    "error_rate",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "request_throughput_rps",
    "output_throughput_tps",
    "mean_batch_size",
    "batch_wait_p95_ms",
    "queue_wait_p95_ms",
    "valid",
    "invalid_reasons",
]


def discover(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        if item.is_dir():
            paths.update(item.rglob("client-result.json"))
        elif item.name == "client-result.json" or item.suffix == ".json":
            paths.add(item)
    return sorted(paths)


def optional(mapping: dict[str, Any] | None, *names: str) -> Any:
    value: Any = mapping
    for name in names:
        if not isinstance(value, dict):
            return None
        value = value.get(name)
    return value


def flatten(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    config = result["configuration"]
    metadata = result["metadata"]
    batch = optional(result, "metrics", "batch") or {}
    validity = result.get("validity", {})
    return {
        "run_id": metadata["run_id"],
        "repeat": metadata["repeat"],
        "target": metadata["label"],
        "mode": config["mode"],
        "concurrency": config["concurrency"],
        "requests": summary["attempted"],
        "success": summary["successful"],
        "error_rate": summary["error_rate"],
        "e2e_p50_ms": optional(summary, "e2e", "p50_ms"),
        "e2e_p95_ms": optional(summary, "e2e", "p95_ms"),
        "e2e_p99_ms": optional(summary, "e2e", "p99_ms"),
        "ttft_p50_ms": optional(summary, "ttft", "p50_ms"),
        "ttft_p95_ms": optional(summary, "ttft", "p95_ms"),
        "ttft_p99_ms": optional(summary, "ttft", "p99_ms"),
        "request_throughput_rps": summary["request_throughput_rps"],
        "output_throughput_tps": summary.get("output_throughput_tps"),
        "mean_batch_size": batch.get("mean_size"),
        "batch_wait_p95_ms": batch.get("wait_p95_ms"),
        "queue_wait_p95_ms": batch.get("queue_wait_p95_ms"),
        "valid": validity.get("valid", False),
        "invalid_reasons": ";".join(validity.get("reasons") or []),
    }


def parse_prometheus(path: Path) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    return parse_prometheus_text(path.read_text(encoding="utf-8"))


def parse_prometheus_text(
    raw: str,
) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    series: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for line in raw.splitlines():
        match = SAMPLE.match(line.strip())
        if not match:
            continue
        labels = tuple(
            sorted(
                (name, bytes(value, "utf-8").decode("unicode_escape"))
                for name, value in LABEL.findall(match.group(2) or "")
            )
        )
        series[(match.group(1), labels)] = float(match.group(3))
    return series


def counter_delta(
    before: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    after: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    name: str,
) -> dict[str, Any]:
    before_values = {labels: value for (metric, labels), value in before.items() if metric == name}
    after_values = {labels: value for (metric, labels), value in after.items() if metric == name}
    if not after_values:
        return {"missing": True, "reset": False, "value": None}
    reset = any(value < before_values.get(labels, 0) for labels, value in after_values.items())
    missing = any(labels not in after_values for labels in before_values)
    value = (
        None
        if reset or missing
        else sum(current - before_values.get(labels, 0) for labels, current in after_values.items())
    )
    return {"missing": missing, "reset": reset, "value": value}


def artifact_counter_deltas(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    artifacts = result.get("metric_artifacts", {})
    derived: dict[str, Any] = {}
    for source, name in [
        ("gateway", "gateway_observed_output_tokens_total"),
        ("vllm", "vllm:generation_tokens"),
    ]:
        before_name = artifacts.get(f"{source}_before")
        after_name = artifacts.get(f"{source}_after")
        if not before_name or not after_name:
            continue
        before_path = result_path.parent / before_name
        after_path = result_path.parent / after_name
        if before_path.exists() and after_path.exists():
            derived[name] = counter_delta(
                parse_prometheus(before_path), parse_prometheus(after_path), name
            )
    return derived


def comparison_errors(arms: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if set(arms) != set(ARMS):
        return [f"comparison requires {', '.join(ARMS)}"]
    for label, result in arms.items():
        if not result.get("validity", {}).get("valid"):
            errors.append(f"{label} run is invalid")
        if result.get("summary", {}).get("attempted", 0) == 0:
            errors.append(f"{label} has zero measured requests")
        if not result.get("metadata", {}).get("label"):
            errors.append(f"{label} is missing target metadata")
    configs = [arms[label]["configuration"] for label in ARMS]
    for field in [
        "dataset_sha256",
        "model",
        "workload_fingerprint",
        "vllm_config_fingerprint",
    ]:
        if len({config.get(field) for config in configs}) != 1:
            errors.append(f"different {field} across compared arms")
    off = arms["gateway_no_batch"]
    on = arms["gateway_batch"]
    off_sha = off.get("environment", {}).get("gateway_git_sha")
    on_sha = on.get("environment", {}).get("gateway_git_sha")
    if not off_sha or off_sha == "unknown" or off_sha != on_sha:
        errors.append("gateway OFF/ON Git SHA is missing or different")
    if off["configuration"].get("batching_enabled") != "false":
        errors.append("gateway_no_batch does not record batching_enabled=false")
    if on["configuration"].get("batching_enabled") != "true":
        errors.append("gateway_batch does not record batching_enabled=true")
    return errors


def value(result: dict[str, Any], metric: str) -> float | None:
    summary = result["summary"]
    if metric == "e2e_p95_ms":
        return optional(summary, "e2e", "p95_ms")
    return summary.get(metric)


def median_range(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def build_comparisons(
    results_with_paths: list[tuple[dict[str, Any], Path]],
) -> tuple[list[dict[str, Any]], list[str]]:
    groups: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    errors: list[str] = []
    for result, path in results_with_paths:
        for metric, delta in artifact_counter_deltas(result, path).items():
            if delta["reset"]:
                errors.append(f"{path}: invalid_counter_reset:{metric}")
        label = result["metadata"]["label"]
        if label in ARMS:
            key = (
                result["configuration"]["mode"],
                result["configuration"]["concurrency"],
                result["metadata"]["repeat"],
            )
            if label in groups[key]:
                errors.append(f"duplicate arm {label} for {key}")
            groups[key][label] = result
    per_repeat: list[dict[str, Any]] = []
    for key, arms in sorted(groups.items()):
        invalid = comparison_errors(arms)
        if invalid:
            errors.extend(f"{key}: {message}" for message in invalid)
            continue
        mode, concurrency, repeat = key
        item: dict[str, Any] = {
            "mode": mode,
            "concurrency": concurrency,
            "repeat": repeat,
            "targets": {},
            "gateway_overhead": {},
            "batching_effect": {},
        }
        for metric in [
            "e2e_p95_ms",
            "request_throughput_rps",
            "output_throughput_tps",
            "error_rate",
        ]:
            direct = value(arms["direct"], metric)
            off = value(arms["gateway_no_batch"], metric)
            on = value(arms["gateway_batch"], metric)
            item["targets"][metric] = {
                "direct": direct,
                "gateway_no_batch": off,
                "gateway_batch": on,
            }
            item["gateway_overhead"][metric] = (
                None if direct is None or off is None else off - direct
            )
            item["batching_effect"][metric] = None if off is None or on is None else on - off
            if metric in {"request_throughput_rps", "output_throughput_tps"}:
                item["batching_effect"][metric + "_relative_percent"] = (
                    None if off in {None, 0} or on is None else (on / off - 1) * 100
                )
        per_repeat.append(item)
    return per_repeat, errors


def aggregate_comparisons(per_repeat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in per_repeat:
        groups[(item["mode"], item["concurrency"])].append(item)
    aggregated = []
    for (mode, concurrency), items in sorted(groups.items()):
        output: dict[str, Any] = {
            "mode": mode,
            "concurrency": concurrency,
            "repeats": [item["repeat"] for item in items],
            "targets": {},
            "gateway_overhead": {},
            "batching_effect": {},
        }
        for section in ["targets", "gateway_overhead", "batching_effect"]:
            keys: set[str] = set()
            for item in items:
                keys.update(item[section])
            for key in sorted(keys):
                if section == "targets":
                    output[section][key] = {}
                    for arm in ARMS:
                        values = [
                            item[section][key][arm]
                            for item in items
                            if item[section][key][arm] is not None
                        ]
                        output[section][key][arm] = median_range(values)
                else:
                    values = [
                        item[section][key] for item in items if item[section][key] is not None
                    ]
                    output[section][key] = median_range(values)
        aggregated.append(output)
    return aggregated


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict):
        return f"{value['median']:.3f} [{value['min']:.3f}, {value['max']:.3f}]"
    return f"{value:.3f}"


def write_markdown(path: Path, aggregated: list[dict[str, Any]]) -> None:
    lines = [
        "# A/B/C comparison",
        "",
        "Values are median [min, max] across run-level repetitions. Differences are "
        "descriptive deltas, not automatic improvement claims.",
        "",
    ]
    for metric in ["e2e_p95_ms", "request_throughput_rps", "output_throughput_tps", "error_rate"]:
        lines.extend(
            [
                f"## {metric}",
                "",
                "| Mode | C | Direct | GW-off | GW-on | Gateway overhead (off-direct) | "
                "Batching effect (on-off) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in aggregated:
            targets = item["targets"][metric]
            lines.append(
                "| {mode} | {c} | {direct} | {off} | {on} | {overhead} | {effect} |".format(
                    mode=item["mode"],
                    c=item["concurrency"],
                    direct=format_value(targets["direct"]),
                    off=format_value(targets["gateway_no_batch"]),
                    on=format_value(targets["gateway_batch"]),
                    overhead=format_value(item["gateway_overhead"][metric]),
                    effect=format_value(item["batching_effect"][metric]),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    paths = discover(arguments.inputs)
    if not paths:
        print("summarize.py: no result files found", file=sys.stderr)
        return 1
    loaded: list[tuple[dict[str, Any], Path]] = []
    validation_errors: list[str] = []
    for path in paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            validation_errors.append(f"{path}: {error}")
            continue
        validation_errors.extend(f"{path}: {error}" for error in validate(result))
        loaded.append((result, path))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with (arguments.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(flatten(result) for result, _ in loaded)
    per_repeat, comparison_validation = build_comparisons(loaded)
    aggregated = aggregate_comparisons(per_repeat)
    derived_counters = {str(path): artifact_counter_deltas(result, path) for result, path in loaded}
    report = {
        "schema_version": 1,
        "per_repeat": per_repeat,
        "aggregated": aggregated,
        "artifact_counter_deltas": derived_counters,
        "validation_errors": validation_errors + comparison_validation,
    }
    (arguments.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_markdown(arguments.output_dir / "comparison.md", aggregated)
    all_errors = validation_errors + comparison_validation
    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        print("summarize.py: invalid comparisons were not calculated", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
