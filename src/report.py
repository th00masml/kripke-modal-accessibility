"""Render results markdown from summary json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "summary.json"
RESULTS = ROOT / "RESULTS.md"


def main() -> None:
    summary = json.loads(SUMMARY.read_text())

    lines = []
    lines.append("# Results - Kripke Modal Accessibility")
    lines.append("")
    lines.append("## Kripke framing")
    lines.append("")
    lines.append(
        "This benchmark compares formal output validity against actual modal-formula truth"
        " in Kripke models (R, V, target world)."
    )
    lines.append("")
    lines.append("## Run coverage")
    lines.append("")
    lines.append("| Model | Arm | observed_rows | expected_rows | status | reason |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key, row in sorted(summary.get("coverage", {}).items()):
        model, arm = key.split("::", 1)
        lines.append(
            f"| {model} | {arm} | {row.get('observed_rows', 0)} | {row.get('expected_rows', 0)} | {row.get('status', 'unknown')} | {row.get('reason', '')} |"
        )
    missing_arms = summary.get("missing_arms", [])
    non_present_arms = summary.get("non_present_arms", [])
    if non_present_arms:
        lines.append("")
        lines.append("Warning: run coverage is not complete (skipped, missing, partial, or metadata-unknown arms present); statistical comparisons may be invalid.")
    lines.append("")
    lines.append("## Class counts by model and arm")
    lines.append("")
    lines.append(
        "| Model | Arm | exact | abstention | correct_abstain | valid_wrong_judgement | valid_spurious_judgement | invented_schema | canonical_found | membership_violations | format_violations |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for key, stats in sorted(summary["by_model_arm"].items()):
        model, arm = key.split("::", 1)
        diagnostics = summary.get("diagnostics_by_model_arm", {}).get(key, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    arm,
                    str(stats.get("exact", 0)),
                    str(stats.get("abstention", 0)),
                    str(stats.get("correct_abstain", 0)),
                    str(stats.get("valid_wrong_judgement", 0)),
                    str(stats.get("valid_spurious_judgement", 0)),
                    str(stats.get("invented_schema", 0)),
                    str(diagnostics.get("canonical_found", 0)),
                    str(diagnostics.get("membership_violations", 0)),
                    str(diagnostics.get("format_violations", 0)),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Paired exact McNemar (prompted vs constrained)")
    lines.append("")
    lines.append("If constrained rows are absent in raw data (for example after --skip-constrained), McNemar is reported as n/a.")
    lines.append("")
    lines.append("| Model | paired_count | prompted-only correct | constrained-only correct | p-value | status | reason |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model, row in sorted(summary["paired_mcnemar"].items()):
        pvalue = row.get("pvalue")
        pvalue_text = f"{pvalue:.6g}" if pvalue is not None else "n/a"
        lines.append(
            f"| {model} | {row.get('paired_count', 0)} | {row['prompted_only_correct']} | {row['constrained_only_correct']} | {pvalue_text} | {row.get('status', 'unknown')} | {row.get('reason', '')} |"
        )

    RESULTS.write_text("\n".join(lines) + "\n")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
