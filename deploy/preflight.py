#!/usr/bin/env python3
"""Capture a concise, secret-free experiment environment fingerprint."""

from __future__ import annotations

import json
import shutil
import subprocess


def run(*command: str) -> str | None:
    if not shutil.which(command[0]):
        return None
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    return (result.stdout or result.stderr).strip() if result.returncode == 0 else None


def main() -> None:
    fingerprint = {
        "git_sha": run("git", "rev-parse", "HEAD"),
        "go": run("go", "version"),
        "gpu": run(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,memory.used",
            "--format=csv,noheader",
        ),
        "gpu_processes": run(
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ),
        "python": run("python", "--version"),
        "pytorch": run("python", "-c", "import torch; print(torch.__version__)"),
        "transformers": run("python", "-c", "import transformers; print(transformers.__version__)"),
        "vllm": run("vllm", "--version"),
        "ports": run("ss", "-lntp"),
    }
    print(json.dumps(fingerprint, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
