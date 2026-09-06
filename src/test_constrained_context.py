"""Diagnostic test for constrained decoding token-context propagation in MLX.

This test runs a short constrained generation and verifies whether the logits
processor receives growing generated-token context across decoding steps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

from constraint import JudgementSetConstraint
from mlx_binding import JudgementSetLogitsProcessor, vocabulary_bytes

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures.jsonl"
DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode(tokenizer, text: str) -> list[int]:
    if getattr(tokenizer, "has_chat_template", False):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.encode(text)


def build_prompt(item: dict) -> str:
    return (
        "You are a careful modal semanticist.\n"
        "Return only the answer string and no explanation.\n\n"
        f"{item['source']}\n\n"
        f"{item['query']}\n"
        "Use exact format: w_i |= FORMULA iff TRUE/FALSE.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture-id", default=None)
    parser.add_argument("--max-steps", type=int, default=24)
    return parser.parse_args()


def pick_fixture(fixtures: list[dict], fixture_id: str | None) -> dict:
    if fixture_id is None:
        for item in fixtures:
            if item.get("present"):
                return item
        return fixtures[0]

    for item in fixtures:
        if item["id"] == fixture_id:
            return item
    raise ValueError(f"fixture not found: {fixture_id}")


def main() -> None:
    args = parse_args()

    fixtures = read_jsonl(FIXTURES)
    item = pick_fixture(fixtures, args.fixture_id)
    allowed_set = {row["gold"] for row in fixtures if row["present"] and row["gold"]}

    model, tokenizer = load(args.model)
    token_bytes = vocabulary_bytes(tokenizer)
    eos_ids = tokenizer.eos_token_ids

    prompt = build_prompt(item)
    p_tokens = encode(tokenizer, prompt)

    constraint = JudgementSetConstraint(allowed_set, token_bytes)
    processor = JudgementSetLogitsProcessor(
        constraint=constraint,
        prompt_length=len(p_tokens),
        prompt_tokens=p_tokens,
        eos_token_ids=eos_ids,
        allow_empty_output=not item["present"],
    )

    raw_token_seq_lens: list[int] = []
    raw_token_seq_first_tokens: list[int | None] = []

    def wrapped_processor(tokens: mx.array, logits: mx.array) -> mx.array:
        token_seq = tokens.tolist()
        if token_seq and isinstance(token_seq[0], list):
            token_seq = token_seq[0]
        token_seq = [int(token) for token in token_seq] if isinstance(token_seq, list) else [int(token_seq)]
        raw_token_seq_lens.append(len(token_seq))
        raw_token_seq_first_tokens.append(token_seq[0] if token_seq else None)
        return processor(tokens, logits)

    kwargs = {"max_tokens": args.max_steps, "logits_processors": [wrapped_processor]}
    generated_tokens: list[int] = []

    for step_index, (token, _) in enumerate(generate_step(mx.array(p_tokens), model, **kwargs), start=1):
        token = int(token)
        if token in eos_ids:
            break
        generated_tokens.append(token)
        if step_index >= args.max_steps:
            break

    trace = processor.trace
    generated_lens = [step.get("generated_len", -1) for step in trace]
    all_zero = len(generated_lens) > 0 and all(length == 0 for length in generated_lens)

    report = {
        "model": args.model,
        "fixture_id": item["id"],
        "present": item["present"],
        "prompt_length": len(p_tokens),
        "trace_calls": len(trace),
        "generated_token_count": len(generated_tokens),
        "raw_token_seq_lens_first10": raw_token_seq_lens[:10],
        "raw_token_seq_first_tokens_first10": raw_token_seq_first_tokens[:10],
        "generated_lens_first10": generated_lens[:10],
        "all_generated_len_zero": all_zero,
        "last_trace": trace[-1] if trace else None,
    }
    print(json.dumps(report, indent=2))

    if all_zero:
        raise SystemExit(
            "FAIL: logits processor did not receive growing generated-token context "
            "(generated_len=0 for all calls)."
        )

    print("PASS: logits processor received non-zero generated-token context.")


if __name__ == "__main__":
    main()