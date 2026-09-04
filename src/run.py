"""Run five-arm modal judgement generation on local MLX models."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

from constraint import JudgementSetConstraint
from mlx_binding import JudgementSetLogitsProcessor, vocabulary_bytes

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures.jsonl"
OUT = ROOT / "outputs" / "raw.jsonl"
RUN_META = ROOT / "outputs" / "run_meta.json"

MODELS = [
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
]

ARMS = ["naive", "prompted", "constrained", "prompt_tuned", "constrained_tuned"]


@dataclass(frozen=True)
class Row:
    fixture_id: str
    model: str
    arm: str


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


def build_prompt(item: dict, arm: str) -> str:
    base = (
        "You are a careful modal semanticist.\n"
        "Return only the answer string and no explanation.\n\n"
        f"{item['source']}\n\n"
        f"{item['query']}\n"
    )
    if arm in {"prompted", "constrained"}:
        base += "Use exact format: w_i |= FORMULA iff TRUE/FALSE.\n"
    if arm in {"prompt_tuned", "constrained_tuned"}:
        base += (
            "Strictly output one line in exact canonical format:\n"
            "w_i |= FORMULA iff TRUE.\n"
            "or\n"
            "w_i |= FORMULA iff FALSE.\n"
            "If out-of-language atom appears, output empty.\n"
        )
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-constrained", action="store_true")
    parser.add_argument("--write-meta-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixtures = read_jsonl(FIXTURES)
    if args.limit:
        fixtures = fixtures[: args.limit]

    active_arms = [arm for arm in ARMS if not (args.skip_constrained and "constrained" in arm)]

    RUN_META.parent.mkdir(parents=True, exist_ok=True)
    RUN_META.write_text(
        json.dumps(
            {
                "models": list(args.models),
                "all_arms": ARMS,
                "active_arms": active_arms,
                "skip_constrained": bool(args.skip_constrained),
                "fixture_count": len(fixtures),
                "overwrite": bool(args.overwrite),
                "limit": args.limit,
            },
            indent=2,
        )
    )

    if args.write_meta_only:
        print(f"wrote {RUN_META} (meta only)")
        return

    existing = set()
    if OUT.exists() and not args.overwrite:
        for row in read_jsonl(OUT):
            existing.add(Row(row["id"], row["model"], row["arm"]))

    if args.overwrite and OUT.exists():
        OUT.unlink()

    allowed_set = {row["gold"] for row in read_jsonl(FIXTURES) if row["present"] and row["gold"]}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as out:
        for model_name in args.models:
            model, tokenizer = load(model_name)
            token_bytes = vocabulary_bytes(tokenizer)
            eos_ids = tokenizer.eos_token_ids
            constraint = JudgementSetConstraint(allowed_set, token_bytes)

            for item in fixtures:
                for arm in active_arms:
                    key = Row(item["id"], model_name, arm)
                    if key in existing:
                        continue

                    prompt = build_prompt(item, arm)
                    p_tokens = encode(tokenizer, prompt)

                    processor = None
                    kwargs = {"max_tokens": 64}
                    if arm in {"constrained", "constrained_tuned"}:
                        processor = JudgementSetLogitsProcessor(
                            constraint=constraint,
                            prompt_length=len(p_tokens),
                            prompt_tokens=p_tokens,
                            eos_token_ids=eos_ids,
                            allow_empty_output=not item["present"],
                        )
                        kwargs["logits_processors"] = [processor]

                    start = perf_counter()
                    tokens = []
                    for token, _ in generate_step(mx.array(p_tokens), model, **kwargs):
                        token = int(token)
                        if token in eos_ids:
                            break
                        tokens.append(token)
                    elapsed = perf_counter() - start

                    output = tokenizer.decode(tokens, skip_special_tokens=True).strip()

                    if processor and processor.trace and all(step.get("generated_len", 0) == 0 for step in processor.trace):
                        raise RuntimeError(
                            "Constrained decoding aborted: MLX logits processor did not receive generated-token context "
                            "(generated_len=0 for all steps). Constrained-arm results would be invalid."
                        )

                    rec = {
                        "id": item["id"],
                        "present": item["present"],
                        "world": item["world"],
                        "formula": item["formula"],
                        "gold": item["gold"],
                        "model": model_name,
                        "arm": arm,
                        "output": output,
                        "elapsed_sec": elapsed,
                    }
                    if processor:
                        rec["constraint_elapsed_sec"] = processor.elapsed_ns / 1e9
                        rec["constraint_trace"] = processor.trace

                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"{model_name} {arm} {item['id']} -> {output!r}")


if __name__ == "__main__":
    main()