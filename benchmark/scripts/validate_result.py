#!/usr/bin/env python3
"""Lightweight schema-version-1 result validator with no third-party dependency."""

from __future__ import annotations

import argparse
import json
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
        "metric_artifacts",
        "environment",
        "validity",
    ]:
        if field not in result:
            errors.append(f"missing {field}")
    configuration = result.get("configuration", {})
    for field in [
        "model",
        "mode",
        "concurrency",
        "requests",
        "warmup",
        "dataset_sha256",
        "workload_fingerprint",
        "vllm_config_fingerprint",
    ]:
        if field not in configuration:
            errors.append(f"configuration missing {field}")
    if configuration.get("mode") not in {"streaming", "non_streaming"}:
        errors.append("configuration.mode is invalid")
    if configuration.get("mode") == "non_streaming":
        for index, request in enumerate(result.get("per_request", [])):
            if request.get("ttft_ms") is not None:
                errors.append(f"per_request[{index}] has TTFT in non-streaming mode")
    if len(configuration.get("dataset_sha256", "")) != 64:
        errors.append("dataset_sha256 must be a 64-character hex digest")
    summary = result.get("summary", {})
    attempted = summary.get("attempted")
    if isinstance(attempted, int) and attempted != len(result.get("per_request", [])):
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
