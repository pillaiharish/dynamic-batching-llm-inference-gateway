#!/usr/bin/env python3
"""Write a non-secret host/GPU environment record for a benchmark manifest."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def command(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            arguments, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    gpu: dict[str, Any] = {"available": False}
    if nvidia_smi:
        raw = command(
            nvidia_smi,
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        )
        if raw:
            devices = []
            for line in raw.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 4:
                    devices.append(
                        {
                            "index": fields[0],
                            "name": fields[1],
                            "memory_total_mib": fields[2],
                            "driver_version": fields[3],
                        }
                    )
            gpu = {"available": bool(devices), "devices": devices}
    return {
        "captured_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "go_version": command("go", "version"),
        "cuda_version": command("nvcc", "--version"),
        "gpu": gpu,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(collect(), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
