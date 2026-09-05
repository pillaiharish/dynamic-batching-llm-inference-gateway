#!/usr/bin/env python3
"""Build a repeated-text chat prompt with an exact templated token count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def token_count(encoded: object) -> int:
    """Count input IDs, never fields on a BatchEncoding-like mapping."""
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise TypeError("tokenizer output has no input_ids")
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        if len(input_ids) != 1:
            raise ValueError("expected one encoded conversation")
        input_ids = input_ids[0]
    return len(input_ids)


def build_exact_prompt(tokenizer: object, target: int, *, unit: str = "x ") -> str:
    if target <= 0:
        raise ValueError("target must be positive")

    def count(repetitions: int) -> int:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": unit * repetitions}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return token_count(encoded)

    low, high = 0, 1
    while count(high) < target:
        low, high = high, high * 2
    while low <= high:
        mid = (low + high) // 2
        measured = count(mid)
        if measured == target:
            return unit * mid
        if measured < target:
            low = mid + 1
        else:
            high = mid - 1
    raise ValueError(f"target token count {target} is not reachable with unit {unit!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer  # Optional runtime dependency; never needed by CI.

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prompt = build_exact_prompt(tokenizer, args.tokens)
    measured = token_count(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    args.output.write_text(
        json.dumps(
            {"model": args.model, "revision": args.revision, "tokens": measured, "prompt": prompt}
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
