"""Diagnostic test for constrained decoding token-context propagation.

This test runs a short constrained generation and verifies whether the logits
processor receives growing generated-token context across decoding steps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from constraint import JudgementSetConstraint
from torch_binding import JudgementSetLogitsProcessor, vocabulary_bytes

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode(tokenizer, text: str) -> list[int]:
    if tokenizer.chat_template:
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    model.eval()

    token_bytes = vocabulary_bytes(tokenizer)
    eos_ids = tokenizer.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)

    prompt = build_prompt(item)
    p_tokens = encode(tokenizer, prompt)
    input_ids = torch.tensor([p_tokens], device=DEVICE)

    constraint = JudgementSetConstraint(allowed_set, token_bytes)
    processor = JudgementSetLogitsProcessor(
        constraint=constraint,
        prompt_length=len(p_tokens),
        eos_token_ids=eos_ids,
        allow_empty_output=not item["present"],
    )

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=args.max_steps,
            do_sample=False,
            logits_processor=LogitsProcessorList([processor]),
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id or eos_ids[0],
        )
    generated_tokens = generated[0, len(p_tokens):].tolist()

    trace = processor.trace
    generated_lens = [step.get("generated_len", -1) for step in trace]
    # Step 0 always sees generated_len=0 (nothing generated yet); the bug we
    # diagnose is generated_len staying at 0 on every subsequent step too.
    all_zero = len(generated_lens) > 1 and all(length == 0 for length in generated_lens[1:])

    report = {
        "model": args.model,
        "fixture_id": item["id"],
        "present": item["present"],
        "prompt_length": len(p_tokens),
        "trace_calls": len(trace),
        "generated_token_count": len(generated_tokens),
        "generated_lens_first10": generated_lens[:10],
        "all_generated_len_zero_after_step0": all_zero,
        "last_trace": trace[-1] if trace else None,
    }
    print(json.dumps(report, indent=2))

    if all_zero:
        raise SystemExit(
            "FAIL: logits processor did not receive growing generated-token context "
            "(generated_len=0 for all calls after step 0)."
        )

    print("PASS: logits processor received non-zero generated-token context.")


if __name__ == "__main__":
    main()