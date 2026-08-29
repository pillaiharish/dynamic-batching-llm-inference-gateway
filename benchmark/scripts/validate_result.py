#!/usr/bin/env python3
"""Lightweight schema-version-1 result validator with no third-party dependency."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def validate(result: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["root must be an object"]
    if result.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    for field in [
        "metadata",
        "configuration",
        "timing",
        "warmup",
        "summary",
        "per_request",
        "metrics",
        "metric_artifacts",
        "environment",
        "validity",
    ]:
        if field not in result:
            errors.append(f"missing {field}")
    required_object_fields = {
        "metadata": ["run_id", "timestamp_utc", "label", "repeat", "execution_order"],
        "timing": ["measured_start_utc", "measured_end_utc", "duration_seconds"],
        "warmup": ["attempted", "success", "failed"],
        "summary": [
            "attempted",
            "successful",
            "failed",
            "error_rate",
            "errors_by_category",
            "e2e",
            "ttft",
            "request_throughput_rps",
            "output_throughput_tps",
            "authoritative_token_coverage",
        ],
        "validity": ["valid", "reasons"],
    }
    for section, required in required_object_fields.items():
        value = result.get(section, {})
        if not isinstance(value, dict):
            errors.append(f"{section} must be an object")
            continue
        for field in required:
            if field not in value:
                errors.append(f"{section} missing {field}")
    for section in ["metrics", "metric_artifacts", "environment"]:
        if not isinstance(result.get(section, {}), dict):
            errors.append(f"{section} must be an object")
    configuration = result.get("configuration", {})
    if not isinstance(configuration, dict):
        return errors + ["configuration must be an object"]
    for field in [
        "base_url",
        "endpoint",
        "model",
        "mode",
        "concurrency",
        "requests",
        "warmup",
        "timeout_seconds",
        "dataset_sha256",
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "n",
        "stream",
        "stream_include_usage",
        "batching_enabled",
        "prefix_caching",
        "workload_fingerprint",
        "gateway_config_fingerprint",
        "vllm_config_fingerprint",
    ]:
        if field not in configuration:
            errors.append(f"configuration missing {field}")
    if configuration.get("mode") not in {"streaming", "non_streaming"}:
        errors.append("configuration.mode is invalid")
    per_request = result.get("per_request", [])
    if not isinstance(per_request, list):
        errors.append("per_request must be an array")
        per_request = []
    if configuration.get("mode") == "non_streaming":
        for index, request in enumerate(per_request):
            if not isinstance(request, dict):
                errors.append(f"per_request[{index}] must be an object")
            elif request.get("ttft_ms") is not None:
                errors.append(f"per_request[{index}] has TTFT in non-streaming mode")
    if not re.fullmatch(r"[0-9a-f]{64}", str(configuration.get("dataset_sha256", ""))):
        errors.append("dataset_sha256 must be a 64-character hex digest")
    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        return errors + ["summary must be an object"]
    attempted = summary.get("attempted")
    if isinstance(attempted, int) and attempted != len(per_request):
        errors.append("summary.attempted does not match per_request length")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    arguments = parser.parse_args()
    failed = False
    for path in arguments.results:
        try:
            with path.open(encoding="utf-8") as file:
                errors = validate(json.load(file))
        except (OSError, json.JSONDecodeError) as error:
            errors = [str(error)]
        if errors:
            failed = True
            for error in errors:
                print(f"{path}: {error}", file=sys.stderr)
        else:
            print(f"{path}: valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
