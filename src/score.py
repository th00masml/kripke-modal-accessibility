"""Score modal-judgement outputs with migration classes and paired tests."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "raw.jsonl"
SCORED = ROOT / "outputs" / "scored.jsonl"
SUMMARY = ROOT / "outputs" / "summary.json"
FIXTURES = ROOT / "data" / "fixtures.jsonl"
RUN_META = ROOT / "outputs" / "run_meta.json"

ARMS = ["naive", "prompted", "constrained", "prompt_tuned", "constrained_tuned"]

CANONICAL_RE = re.compile(
    r"(w\d+)\s*\|=\s*([A-Za-z0-9_(),]+)\s*iff\s*(TRUE|FALSE)\.?",
    flags=re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_first_canonical(output: str) -> str | None:
    match = CANONICAL_RE.search(output)
    if not match:
        return None
    world, formula, truth = match.groups()
    return f"{world.lower()} |= {formula} iff {truth.upper()}."


def classify(row: dict, allowed: set[str]) -> tuple[str, bool, bool, bool, str | None]:
    output = (row.get("output") or "").strip()
    gold = row["gold"]
    present = row["present"]

    if output == "":
        if present:
            return "abstention", False, False, False, None
        return "correct_abstain", False, False, False, None

    parsed = parse_first_canonical(output)
    if parsed is None:
        return "invented_schema", False, False, True, None

    membership_violation = parsed not in allowed
    is_exact = bool(present and parsed == gold)

    if present:
        if is_exact:
            return "exact", True, membership_violation, False, parsed
        if not membership_violation:
            return "valid_wrong_judgement", False, False, False, parsed
        return "invented_schema", False, True, False, parsed

    if not membership_violation:
        return "valid_spurious_judgement", False, False, False, parsed
    return "invented_schema", False, True, False, parsed


def main() -> None:
    raw = read_jsonl(RAW)
    fixtures = read_jsonl(FIXTURES)
    run_meta = json.loads(RUN_META.read_text()) if RUN_META.exists() else {}
    allowed_mode = run_meta.get("allowed_set", "gold")
    if allowed_mode == "product":
        present = [row for row in fixtures if row["present"] and row["gold"]]
        worlds = sorted({row["world"] for row in present})
        formulas = sorted({row["formula"] for row in present})
        allowed = {f"{w} |= {f} iff {t}." for w in worlds for f in formulas for t in ("TRUE", "FALSE")}
    else:
        allowed = {row["gold"] for row in fixtures if row["present"] and row["gold"]}

    by_model_arm = defaultdict(Counter)
    diagnostics_by_model_arm = defaultdict(Counter)
    scored = []

    for row in raw:
        cls, hit, membership_violation, format_violation, parsed = classify(row, allowed)
        record = dict(row)
        record["class"] = cls
        record["is_exact"] = bool(hit)
        record["membership_violation"] = bool(membership_violation)
        record["format_violation"] = bool(format_violation)
        record["parsed_canonical"] = parsed
        scored.append(record)
        by_model_arm[(row["model"], row["arm"])][cls] += 1
        diagnostics_by_model_arm[(row["model"], row["arm"])]["membership_violations"] += int(membership_violation)
        diagnostics_by_model_arm[(row["model"], row["arm"])]["format_violations"] += int(format_violation)
        diagnostics_by_model_arm[(row["model"], row["arm"])]["canonical_found"] += int(parsed is not None)

    models_in_rows = sorted({row["model"] for row in raw})
    has_run_meta = bool(run_meta)
    configured_models = run_meta.get("models", [])
    if configured_models:
        models = sorted(configured_models)
    else:
        models = models_in_rows

    expected_rows_per_arm = int(run_meta.get("fixture_count", len(fixtures)))
    active_arms = run_meta.get("active_arms", ARMS)
    active_arms_set = set(active_arms)

    coverage = {}
    missing_arms = []
    non_present_arms = []
    for model in models:
        for arm in ARMS:
            observed = sum(1 for row in raw if row["model"] == model and row["arm"] == arm)
            if not has_run_meta and observed == 0:
                status = "unknown_missing_metadata"
                expected = expected_rows_per_arm
                reason = "run metadata unavailable; cannot distinguish skipped vs failed"
            elif arm not in active_arms_set:
                status = "skipped_by_config"
                expected = 0
                reason = "arm not scheduled in this run"
            else:
                expected = expected_rows_per_arm
                if observed == expected:
                    status = "present"
                    reason = ""
                elif observed == 0:
                    status = "missing_data"
                    reason = "scheduled arm produced no rows"
                else:
                    status = "partial_data"
                    reason = "scheduled arm produced incomplete rows"
            coverage_key = f"{model}::{arm}"
            coverage[coverage_key] = {
                "observed_rows": observed,
                "expected_rows": expected,
                "status": status,
                "reason": reason,
            }
            if status in {"missing_data", "partial_data"}:
                missing_arms.append(coverage_key)
            if status != "present":
                non_present_arms.append(coverage_key)

    SCORED.parent.mkdir(parents=True, exist_ok=True)
    with SCORED.open("w") as handle:
        for row in scored:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "total": len(scored),
        "by_model_arm": {f"{m}::{a}": dict(c) for (m, a), c in by_model_arm.items()},
        "diagnostics_by_model_arm": {f"{m}::{a}": dict(c) for (m, a), c in diagnostics_by_model_arm.items()},
        "coverage": coverage,
        "missing_arms": missing_arms,
        "non_present_arms": non_present_arms,
        "paired_mcnemar": {},
    }

    for model in models:
        prompted = {
            row["id"]: row for row in scored if row["model"] == model and row["arm"] == "prompted"
        }
        constrained = {
            row["id"]: row for row in scored if row["model"] == model and row["arm"] == "constrained"
        }
        ids = sorted(set(prompted) & set(constrained))
        b = c = 0
        pvalue = None
        test_status = "ok"
        reason = ""

        if not has_run_meta:
            test_status = "not_computed"
            reason = "run metadata unavailable"
        elif "prompted" not in active_arms_set or "constrained" not in active_arms_set:
            test_status = "not_computed"
            reason = "required arms not scheduled in this run"
        elif len(ids) == 0:
            test_status = "not_computed"
            reason = "no paired rows"
        else:
            for row_id in ids:
                p_ok = prompted[row_id]["is_exact"]
                q_ok = constrained[row_id]["is_exact"]
                if p_ok and not q_ok:
                    b += 1
                elif q_ok and not p_ok:
                    c += 1
            table = [[0, b], [c, 0]]
            pvalue = float(mcnemar(table, exact=True).pvalue) if (b + c) else 1.0

        summary["paired_mcnemar"][model] = {
            "paired_count": len(ids),
            "prompted_only_correct": b,
            "constrained_only_correct": c,
            "pvalue": pvalue,
            "status": test_status,
            "reason": reason,
        }

    with SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
