#!/usr/bin/env python3
"""Terraform adapter for the official Vast.ai CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_instance_id(output: str) -> int:
    result = json.loads(output)
    instance_id = result.get("new_contract")
    if result.get("success") is not True or not isinstance(instance_id, int):
        raise ValueError(f"unexpected Vast create response: {result}")
    return instance_id


def create(state_file: Path) -> None:
    command = [
        "vastai",
        "create",
        "instance",
        os.environ["VAST_OFFER_ID"],
        "--raw",
        "--image",
        os.environ["VAST_IMAGE"],
        "--disk",
        os.environ["VAST_DISK_GB"],
        "--ssh",
        "--direct",
        "--cancel-unavail",
    ]
    if onstart := os.environ.get("VAST_ONSTART"):
        command.extend(["--onstart-cmd", onstart])
    if volume_id := os.environ.get("VAST_VOLUME_ID"):
        command.extend(["--link-volume", volume_id, "--mount-path", os.environ["VAST_MOUNT_PATH"]])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(f"{parse_instance_id(result.stdout)}\n", encoding="utf-8")


def destroy(state_file: Path) -> None:
    instance_id = int(state_file.read_text(encoding="utf-8").strip())
    subprocess.run(["vastai", "destroy", "instance", str(instance_id), "--raw"], check=True)
    state_file.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "destroy"))
    parser.add_argument("state_file", type=Path)
    args = parser.parse_args()
    try:
        globals()[args.action](args.state_file)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Vast lab {args.action} failed: {exc}") from exc


if __name__ == "__main__":
    main()
