"""Run five-arm modal judgement generation on local HF models via CUDA/PyTorch."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from constraint import JudgementSetConstraint
from torch_binding import JudgementSetLogitsProcessor, vocabulary_bytes

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures.jsonl"
OUT = ROOT / "outputs" / "raw.jsonl"
RUN_META = ROOT / "outputs" / "run_meta.json"

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

ARMS = ["naive", "prompted", "constrained", "prompt_tuned", "constrained_tuned"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    if tokenizer.chat_template:
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
    parser.add_argument(
        "--allowed-set",
        choices=["gold", "product"],
        default="gold",
        help=(
            "How to build the constraint's admissible set. 'gold' (default, used in the "
            "2026-09 cached run) = the gold judgements of in-language fixtures; this admits "
            "only ONE truth value for (world, formula) pairs whose gold is constant across "
            "frames and valuations, and therefore leaks the answer on 25/80 fixtures. "
            "'product' = every world x formula x truth value (32 strings); leak-free."
        ),
    )
    return parser.parse_args()


def build_allowed_set(fixtures: list[dict], mode: str) -> set[str]:
    present = [row for row in fixtures if row["present"] and row["gold"]]
    if mode == "gold":
        return {row["gold"] for row in present}
    worlds = sorted({row["world"] for row in present})
    formulas = sorted({row["formula"] for row in present})
    return {f"{w} |= {f} iff {t}." for w in worlds for f in formulas for t in ("TRUE", "FALSE")}


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
                "backend": "torch",
                "device": DEVICE,
                "allowed_set": args.allowed_set,
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

    allowed_set = build_allowed_set(read_jsonl(FIXTURES), args.allowed_set)
    print(f"allowed set: mode={args.allowed_set} size={len(allowed_set)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as out:
        for model_name in args.models:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            ).to(DEVICE)
            model.eval()

            token_bytes = vocabulary_bytes(tokenizer)
            eos_ids = tokenizer.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
            constraint = JudgementSetConstraint(allowed_set, token_bytes)

            for item in fixtures:
                for arm in active_arms:
                    key = Row(item["id"], model_name, arm)
                    if key in existing:
                        continue

                    prompt = build_prompt(item, arm)
                    p_tokens = encode(tokenizer, prompt)
                    input_ids = torch.tensor([p_tokens], device=DEVICE)

                    processor = None
                    logits_processors = None
                    if arm in {"constrained", "constrained_tuned"}:
                        processor = JudgementSetLogitsProcessor(
                            constraint=constraint,
                            prompt_length=len(p_tokens),
                            eos_token_ids=eos_ids,
                            allow_empty_output=not item["present"],
                        )
                        logits_processors = LogitsProcessorList([processor])

                    start = perf_counter()
                    with torch.no_grad():
                        generated = model.generate(
                            input_ids,
                            max_new_tokens=64,
                            do_sample=False,
                            logits_processor=logits_processors,
                            eos_token_id=eos_ids,
                            pad_token_id=tokenizer.pad_token_id or eos_ids[0],
                        )
                    elapsed = perf_counter() - start

                    new_tokens = generated[0, len(p_tokens):].tolist()
                    output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                    if processor and processor.trace and all(step.get("generated_len", 0) == 0 for step in processor.trace[1:]):
                        raise RuntimeError(
                            "Constrained decoding aborted: logits processor did not receive generated-token context "
                            "past step 0. Constrained-arm results would be invalid."
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

            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()