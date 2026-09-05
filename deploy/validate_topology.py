#!/usr/bin/env python3
"""Fail-closed validation for the documented H100 experiment topologies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(config: dict) -> None:
    required = {"topology", "replicas", "traffic", "gateway_backends", "aggregation"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing required fields: {', '.join(sorted(missing))}")
    topology = config["topology"]
    replicas = config["replicas"]
    traffic = config["traffic"]
    backends = config["gateway_backends"]
    if not isinstance(replicas, list) or not replicas or len(set(replicas)) != len(replicas):
        raise ValueError("replicas must be a non-empty unique list")
    if not set(traffic) <= set(replicas) or not set(backends) <= set(replicas):
        raise ValueError("traffic and gateway backends must name declared replicas")

    expected = {
        "T0": (1, ["A"], [], False),
        "T1": (1, ["A"], [], False),
        "T2": (2, ["A"], [], False),
        "T3-A": (2, ["A", "B"], [], False),
        "T3-B": (2, ["A", "B"], ["A", "B"], False),
        "T3-C": (2, ["A", "B"], ["A", "B"], True),
    }
    if topology not in expected:
        raise ValueError(f"unknown topology: {topology!r}")
    want = expected[topology]
    got = (len(replicas), traffic, backends, config["aggregation"])
    if got != want:
        raise ValueError(f"unsafe {topology} configuration: expected {want}, got {got}")
    if topology == "T3-A" and config.get("routing") != "deterministic_round_robin":
        raise ValueError("T3-A requires deterministic_round_robin routing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate(config)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid topology: {exc}") from exc
    print(f"valid topology: {config['topology']}")


if __name__ == "__main__":
    main()
