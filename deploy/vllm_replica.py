#!/usr/bin/env python3
"""Start, inspect, and stop vLLM replicas as process groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def record_path(pid_dir: Path, name: str) -> Path:
    if not name.isalnum():
        raise ValueError("replica name must be alphanumeric")
    return pid_dir / f"{name}.json"


def alive(pid: int) -> bool:
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def process_identity(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-o", "lstart=", "-o", "command=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    identity = result.stdout.strip()
    if result.returncode != 0 or not identity:
        raise ProcessLookupError(pid)
    return hashlib.sha256(identity.encode()).hexdigest()


def verify_identity(record: dict) -> None:
    if process_identity(record["pid"]) != record.get("identity"):
        raise RuntimeError("stale/reused PID: process identity does not match lifecycle record")


def start(pid_dir: Path, name: str, port: int, command: list[str]) -> None:
    path = record_path(pid_dir, name)
    if not command or not 1 <= port <= 65535:
        raise ValueError("a command and valid port are required")
    if path.exists() and alive(json.loads(path.read_text())["pid"]):
        raise RuntimeError(f"replica {name} is already running")
    pid_dir.mkdir(parents=True, exist_ok=True)
    log = (pid_dir / f"{name}.log").open("ab")
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    record = {"name": name, "pid": process.pid, "pgid": process.pid, "port": port}
    time.sleep(0.1)
    if process.poll() is not None:
        raise RuntimeError(f"replica {name} exited during startup; inspect {log.name}")
    record["identity"] = process_identity(process.pid)
    path.write_text(json.dumps(record) + "\n")


def stop(pid_dir: Path, name: str, timeout: float) -> None:
    path = record_path(pid_dir, name)
    record = json.loads(path.read_text())
    verify_identity(record)
    if os.getpgid(record["pid"]) != record["pgid"] or record["pgid"] != record["pid"]:
        raise RuntimeError("recorded PID no longer owns the expected process group")
    os.killpg(record["pgid"], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and alive(record["pid"]):
        time.sleep(0.1)
    if alive(record["pid"]):
        raise RuntimeError("process group still exists; inspect it before any force-kill")
    path.unlink()


def status(pid_dir: Path, name: str) -> dict:
    record = json.loads(record_path(pid_dir, name).read_text())
    pid_alive = alive(record["pid"])
    identity_matches = pid_alive and process_identity(record["pid"]) == record.get("identity")
    return {
        **record,
        "alive": pid_alive and identity_matches,
        "identity_matches": identity_matches,
        "stale_or_reused": pid_alive and not identity_matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-dir", type=Path, default=Path(".run/vllm"))
    sub = parser.add_subparsers(dest="action", required=True)
    start_parser = sub.add_parser("start")
    start_parser.add_argument("name")
    start_parser.add_argument("port", type=int)
    start_parser.add_argument("command", nargs=argparse.REMAINDER)
    stop_parser = sub.add_parser("stop")
    stop_parser.add_argument("name")
    stop_parser.add_argument("--timeout", type=float, default=30)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("name")
    args = parser.parse_args()
    try:
        if args.action == "start":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            start(args.pid_dir, args.name, args.port, command)
        elif args.action == "stop":
            stop(args.pid_dir, args.name, args.timeout)
        else:
            result = status(args.pid_dir, args.name)
            print(json.dumps(result, sort_keys=True))
            raise SystemExit(0 if result["alive"] else 1)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, KeyError) as exc:
        raise SystemExit(f"lifecycle error: {exc}") from exc


if __name__ == "__main__":
    main()
