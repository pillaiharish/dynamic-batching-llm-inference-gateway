#!/usr/bin/env python3
"""Build a repeated-text chat prompt with an exact templated token count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

QWEN3_V1_MODEL = "Qwen/Qwen3-8B"


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


def templated_token_count(
    tokenizer: object, prompt: str, *, chat_template_kwargs: dict[str, Any]
) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    return token_count(encoded)


def build_exact_prompt(
    tokenizer: object,
    target: int,
    *,
    chat_template_kwargs: dict[str, Any],
    unit: str = "x ",
) -> str:
    if target <= 0:
        raise ValueError("target must be positive")

    def count(repetitions: int) -> int:
        return templated_token_count(
            tokenizer,
            unit * repetitions,
            chat_template_kwargs=chat_template_kwargs,
        )

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
    parser.add_argument(
        "--chat-template-kwargs",
        required=True,
        help="explicit JSON object, e.g. '{\"enable_thinking\": false}'",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        chat_template_kwargs = json.loads(args.chat_template_kwargs)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --chat-template-kwargs JSON: {exc}")
    if not isinstance(chat_template_kwargs, dict):
        parser.error("--chat-template-kwargs must be a JSON object")
    if args.model == QWEN3_V1_MODEL and chat_template_kwargs.get("enable_thinking") is not False:
        parser.error("Qwen3 V1 requires enable_thinking=false")

    from transformers import AutoTokenizer  # Optional runtime dependency; never needed by CI.

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prompt = build_exact_prompt(
        tokenizer,
        args.tokens,
        chat_template_kwargs=chat_template_kwargs,
    )
    measured = templated_token_count(
        tokenizer,
        prompt,
        chat_template_kwargs=chat_template_kwargs,
    )
    args.output.write_text(
        json.dumps(
            {
                "chat_template_kwargs": chat_template_kwargs,
                "model": args.model,
                "prompt": prompt,
                "revision": args.revision,
                "tokens": measured,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
