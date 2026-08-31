"""Deterministic Kripke-model fixtures for modal judgement generation."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fixtures.jsonl"

WORLDS = ["w0", "w1", "w2", "w3"]


def f_atom(name: str):
    return ("atom", name)


def f_not(sub):
    return ("not", sub)


def f_box(sub):
    return ("box", sub)


def f_dia(sub):
    return ("dia", sub)


def f_and(left, right):
    return ("and", left, right)


def to_text(formula) -> str:
    tag = formula[0]
    if tag == "atom":
        return formula[1]
    if tag == "not":
        return f"NOT({to_text(formula[1])})"
    if tag == "box":
        return f"BOX({to_text(formula[1])})"
    if tag == "dia":
        return f"DIA({to_text(formula[1])})"
    if tag == "and":
        return f"AND({to_text(formula[1])},{to_text(formula[2])})"
    raise ValueError(f"Unknown formula tag: {tag}")


def eval_formula(formula, world: str, rel: dict[str, set[str]], val: dict[str, set[str]]) -> bool | None:
    tag = formula[0]
    if tag == "atom":
        atom = formula[1]
        if atom not in val:
            return None
        return world in val[atom]

    if tag == "not":
        inner = eval_formula(formula[1], world, rel, val)
        return None if inner is None else (not inner)

    if tag == "and":
        left = eval_formula(formula[1], world, rel, val)
        right = eval_formula(formula[2], world, rel, val)
        if left is None or right is None:
            return None
        return left and right

    if tag == "box":
        targets = rel[world]
        for target in targets:
            tv = eval_formula(formula[1], target, rel, val)
            if tv is None:
                return None
            if not tv:
                return False
        return True

    if tag == "dia":
        targets = rel[world]
        seen_true = False
        for target in targets:
            tv = eval_formula(formula[1], target, rel, val)
            if tv is None:
                return None
            if tv:
                seen_true = True
                break
        return seen_true

    raise ValueError(f"Unknown formula tag: {tag}")


def judgement(world: str, formula_text: str, truth: bool) -> str:
    return f"{world} |= {formula_text} iff {'TRUE' if truth else 'FALSE'}."


def frame_to_text(rel: dict[str, set[str]]) -> str:
    rows = []
    for w in WORLDS:
        targets = ", ".join(sorted(rel[w]))
        rows.append(f"{w} -> {{{targets}}}")
    return "\n".join(rows)


def valuation_to_text(val: dict[str, set[str]]) -> str:
    rows = []
    for atom in sorted(val):
        rows.append(f"{atom}: {{{', '.join(sorted(val[atom]))}}}")
    return "\n".join(rows)


def main() -> None:
    frames = [
        {
            "w0": {"w0", "w1"},
            "w1": {"w1", "w2"},
            "w2": {"w2", "w3"},
            "w3": {"w3"},
        },
        {
            "w0": {"w1", "w2"},
            "w1": {"w0"},
            "w2": {"w2"},
            "w3": {"w0", "w3"},
        },
        {
            "w0": {"w0", "w2"},
            "w1": {"w1"},
            "w2": {"w1", "w3"},
            "w3": {"w0", "w3"},
        },
        {
            "w0": {"w3"},
            "w1": {"w0", "w2"},
            "w2": {"w1"},
            "w3": {"w2", "w3"},
        },
        {
            "w0": {"w0", "w1", "w2", "w3"},
            "w1": {"w1", "w2"},
            "w2": {"w2"},
            "w3": {"w3", "w0"},
        },
    ]

    valuations = [
        {"p": {"w0", "w2"}, "q": {"w1", "w2"}},
        {"p": {"w1", "w3"}, "q": {"w0"}},
        {"p": {"w0", "w1", "w3"}, "q": {"w2", "w3"}},
        {"p": {"w2"}, "q": {"w0", "w1"}},
    ]

    formulas_present = [
        f_box(f_atom("p")),
        f_dia(f_atom("q")),
        f_not(f_box(f_not(f_atom("p")))),
        f_and(f_box(f_atom("p")), f_dia(f_atom("q"))),
    ]

    formulas_absent = [
        f_box(f_atom("r")),
        f_dia(f_atom("r")),
        f_not(f_atom("r")),
        f_and(f_atom("p"), f_atom("r")),
    ]

    rows = []
    idx = 1

    # 5 frames x 4 valuations x 4 formulas = 80 present fixtures.
    for fi, frame in enumerate(frames):
        for vi, valuation in enumerate(valuations):
            for qi, formula in enumerate(formulas_present):
                world = WORLDS[(fi + vi + qi) % len(WORLDS)]
                ftxt = to_text(formula)
                truth = eval_formula(formula, world, frame, valuation)
                assert truth is not None
                rows.append(
                    {
                        "id": f"kripke_{idx:04d}",
                        "present": True,
                        "world": world,
                        "formula": ftxt,
                        "source": (
                            "KRIPKE MODEL\n"
                            f"Worlds: {', '.join(WORLDS)}\n"
                            "Accessibility relation R:\n"
                            f"{frame_to_text(frame)}\n\n"
                            "Valuation V:\n"
                            f"{valuation_to_text(valuation)}\n\n"
                            f"Target world: {world}\n"
                            f"Target formula: {ftxt}"
                        ),
                        "query": "Output exactly one canonical judgement in the form: w_i |= FORMULA iff TRUE/FALSE.",
                        "gold": judgement(world, ftxt, bool(truth)),
                    }
                )
                idx += 1

    # 5 frames x 4 valuations x 4 formulas = 80 absent fixtures.
    for fi, frame in enumerate(frames):
        for vi, valuation in enumerate(valuations):
            for qi, formula in enumerate(formulas_absent):
                world = WORLDS[(fi + vi + qi) % len(WORLDS)]
                ftxt = to_text(formula)
                rows.append(
                    {
                        "id": f"kripke_{idx:04d}",
                        "present": False,
                        "world": world,
                        "formula": ftxt,
                        "source": (
                            "KRIPKE MODEL\n"
                            f"Worlds: {', '.join(WORLDS)}\n"
                            "Accessibility relation R:\n"
                            f"{frame_to_text(frame)}\n\n"
                            "Valuation V:\n"
                            f"{valuation_to_text(valuation)}\n\n"
                            f"Target world: {world}\n"
                            f"Target formula: {ftxt}\n"
                            "Note: atom r is outside the configured language."
                        ),
                        "query": "Output exactly one canonical judgement in the form: w_i |= FORMULA iff TRUE/FALSE. If the formula uses out-of-language atoms, output empty.",
                        "gold": "",
                    }
                )
                idx += 1

    assert len(rows) == 160

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} fixtures")


if __name__ == "__main__":
    main()
