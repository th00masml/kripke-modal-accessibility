"""Recompute every number and figure in the preprint from outputs/scored.jsonl.
Run from src/:  python analysis_paper.py
Writes outputs/paper_stats.json and paper/figs/*.pdf. Needs numpy, matplotlib.
Exploratory re-analysis; the pipeline's own summary is outputs/summary.json."""
from __future__ import annotations

import collections
import json
import re
from math import comb
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
SHORT = {MODELS[0]: "0.5B", MODELS[1]: "1.5B"}
ARMS = ["naive", "prompted", "prompt_tuned", "constrained", "constrained_tuned"]

fixtures = [json.loads(l) for l in (ROOT / "data" / "fixtures.jsonl").open()]
allowed = {f["gold"] for f in fixtures if f["present"]}


def flip(s: str) -> str:
    return s.replace("iff TRUE.", "iff XX.").replace("iff FALSE.", "iff TRUE.").replace("iff XX.", "iff FALSE.")


present_ids = [f["id"] for f in fixtures if f["present"]]
absent_ids = [f["id"] for f in fixtures if not f["present"]]
forced = [f["id"] for f in fixtures if f["present"] and flip(f["gold"]) not in allowed]
free = [i for i in present_ids if i not in set(forced)]
gold = {f["id"]: f["gold"] for f in fixtures}
formula_of = {f["id"]: f["formula"] for f in fixtures}

rows = [json.loads(l) for l in (OUT / "scored.jsonl").open()]
R: dict[tuple[str, str], dict[str, dict]] = collections.defaultdict(dict)
for r in rows:
    R[(r["model"], r["arm"])][r["id"]] = r

# Lenient parser: tolerates w_0, spaces after commas, DIAMOND for DIA, missing '=' in '|='.
LEN = re.compile(r"(w)_?(\d+)\s*\|=?\s*([A-Za-z0-9_(), ]+?)\s*iff\s*(TRUE|FALSE)\.?", re.I)


def lenient(out: str) -> str | None:
    m = LEN.search(out or "")
    if not m:
        return None
    _, d, f, t = m.groups()
    f = f.replace(" ", "").replace("DIAMONDS", "DIA").replace("DIAMOND", "DIA")
    return f"w{d} |= {f} iff {t.upper()}."


def wf(s: str | None) -> str | None:
    return None if s is None else s.split(" iff")[0]


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(v for v in pmf if v <= pmf[k] + 1e-12))


def boot_ci(hits: list[int], n=2000, seed=0):
    rng = np.random.default_rng(seed)
    h = np.array(hits)
    return np.percentile([rng.choice(h, len(h)).mean() for _ in range(n)], [2.5, 97.5]).round(3).tolist()


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n, k = a_only + b_only, min(a_only, b_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return a_only, b_only, p


STATS: dict = {
    "allowed_set_size": len(allowed),
    "forced_fixtures": len(forced),
    "free_fixtures": len(free),
    "forced_pairs": sorted({(f["world"], f["formula"], gold[f["id"]].split("iff ")[1]) for f in fixtures if f["id"] in set(forced)}),
    "per_model": {},
}

print(f"allowed set = {len(allowed)} strings; forced = {len(forced)}, free = {len(free)}")

for m in MODELS:
    S: dict = {}
    for arm in ARMS:
        a = R[(m, arm)]
        pres = [a[i] for i in present_ids]
        strict = [bool(r["is_exact"]) for r in pres]
        len_parsed = [lenient(r["output"]) for r in pres]
        len_exact = [p == r["gold"] for p, r in zip(len_parsed, pres)]
        wf_right = [p is not None and wf(p) == wf(r["gold"]) for p, r in zip(len_parsed, pres)]
        truth_given_wf = [e for e, w in zip(len_exact, wf_right) if w]
        free_hits = [lenient(a[i]["output"]) == gold[i] for i in free]
        forced_hits = [lenient(a[i]["output"]) == gold[i] for i in forced]
        absent_rows = [a[i] for i in absent_ids]
        truth_pred = collections.Counter((p or "none").split("iff ")[-1].rstrip(".") for p in len_parsed)
        per_formula = collections.defaultdict(lambda: [0, 0])
        for r, e in zip(pres, len_exact):
            per_formula[r["formula"]][0] += int(e)
            per_formula[r["formula"]][1] += 1
        S[arm] = {
            "strict_exact": sum(strict), "strict_ci": boot_ci([int(x) for x in strict]),
            "lenient_parsed": sum(p is not None for p in len_parsed),
            "lenient_exact": sum(len_exact), "lenient_ci": boot_ci([int(x) for x in len_exact]),
            "world_formula_right": sum(wf_right),
            "truth_right_given_wf": [sum(truth_given_wf), len(truth_given_wf)],
            "free_exact": sum(free_hits), "free_n": len(free), "free_binom_p_vs_chance": binom_two_sided(sum(free_hits), len(free)),
            "forced_exact": sum(forced_hits), "forced_n": len(forced),
            "truth_pred_dist": dict(truth_pred),
            "per_formula_lenient": {k: v for k, v in per_formula.items()},
            "absent_empty": sum(1 for r in absent_rows if (r["output"] or "").strip() == ""),
            "absent_mentions_r": sum(1 for r in absent_rows if re.search(r"\(r\)|\br\b", r["output"] or "")),
            "absent_canonical_in_language": sum(1 for r in absent_rows if r["class"] == "valid_spurious_judgement"),
            "mean_elapsed_sec": float(np.mean([r["elapsed_sec"] for r in a.values()])),
            "class_counts": dict(collections.Counter(r["class"] for r in a.values())),
        }
        print(f"{SHORT[m]} {arm:17} strict {sum(strict):2d}/80 lenient {sum(len_exact):2d}/80 wf {sum(wf_right):2d} "
              f"free {sum(free_hits):2d}/{len(free)} (p={S[arm]['free_binom_p_vs_chance']:.2f}) forced {sum(forced_hits):2d}/{len(forced)} "
              f"truth {dict(truth_pred)} empty-on-absent {S[arm]['absent_empty']}")
    # paired tests on the free set: constrained vs prompted (lenient) and vs naive (lenient)
    c_free = [R[(m, "constrained")][i]["is_exact"] for i in free]
    for other in ["prompted", "naive"]:
        o_free = [lenient(R[(m, other)][i]["output"]) == gold[i] for i in free]
        ao, bo, p = mcnemar_exact(o_free, c_free)
        S[f"mcnemar_free_{other}_lenient_vs_constrained"] = {f"{other}_only": ao, "constrained_only": bo, "p": p}
        print(f"  free-set McNemar {other}(lenient) vs constrained: {ao} vs {bo}, p={p:.3f}")
    ct, c = R[(m, "constrained_tuned")], R[(m, "constrained")]
    S["constrained_vs_tuned"] = {"outputs_differ": sum(1 for i in c if c[i]["output"] != ct[i]["output"]),
                                 "exact_differ": sum(1 for i in c if c[i]["is_exact"] != ct[i]["is_exact"])}
    STATS["per_model"][m] = S

json.dump(STATS, (OUT / "paper_stats.json").open("w"), indent=1, default=float)

# ---------- figures ----------
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
CLASSES = ["exact", "valid_wrong_judgement", "invented_schema", "valid_spurious_judgement", "abstention", "correct_abstain"]
CCOL = {"exact": "#2a6fb0", "valid_wrong_judgement": "#9ecae1", "invented_schema": "#bbbbbb",
        "valid_spurious_judgement": "#e08a2e", "abstention": "#555555", "correct_abstain": "#31a354"}
LAB = {"exact": "exact", "valid_wrong_judgement": "valid, wrong truth value", "invented_schema": "not canonical (strict)",
       "valid_spurious_judgement": "valid, spurious (out-of-language)", "abstention": "empty on in-language", "correct_abstain": "correct abstain"}

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
for ax, m in zip(axes, MODELS):
    bottoms = np.zeros(len(ARMS))
    for cl in CLASSES:
        vals = np.array([STATS["per_model"][m][arm]["class_counts"].get(cl, 0) for arm in ARMS])
        ax.bar(range(len(ARMS)), vals, bottom=bottoms, color=CCOL[cl], label=LAB[cl], width=0.7)
        bottoms += vals
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(["naive", "prompted", "prompt\ntuned", "constr.", "constr.\ntuned"], fontsize=7)
    ax.set_title(f"Qwen2.5-{SHORT[m]}-Instruct", fontsize=9); ax.set_ylim(0, 160)
axes[0].set_ylabel("fixtures (of 160)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(); fig.savefig(FIGS / "class_counts.pdf", bbox_inches="tight")

# forced vs free, strict constrained and lenient prompt arms
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
arms_show = ["naive", "prompted", "prompt_tuned", "constrained"]
for ax, m in zip(axes, MODELS):
    S = STATS["per_model"][m]
    x = np.arange(len(arms_show)); w = 0.36
    ax.bar(x - w / 2, [S[a]["forced_exact"] / S[a]["forced_n"] for a in arms_show], w, color="#e08a2e", label="forced (n=25): only one truth value in the trie")
    ax.bar(x + w / 2, [S[a]["free_exact"] / S[a]["free_n"] for a in arms_show], w, color="#2a6fb0", label="free (n=55): both truth values in the trie")
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([a.replace("_", "\n") for a in arms_show], fontsize=7)
    ax.set_title(f"Qwen2.5-{SHORT[m]}-Instruct", fontsize=9); ax.set_ylim(0, 1.05)
axes[0].set_ylabel("semantic accuracy (lenient parse)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=1, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(); fig.savefig(FIGS / "forced_vs_free.pdf", bbox_inches="tight")

# truth-value distribution on free fixtures, constrained arm
fig, ax = plt.subplots(figsize=(4.2, 2.2))
labels, tv, fv = [], [], []
gold_free = collections.Counter(gold[i].split("iff ")[1].rstrip(".") for i in free)
labels.append("gold"); tv.append(gold_free["TRUE"]); fv.append(gold_free["FALSE"])
for m in MODELS:
    for arm in ["prompted", "constrained"]:
        cnt = collections.Counter((lenient(R[(m, arm)][i]["output"]) or "none").split("iff ")[-1].rstrip(".") for i in free)
        labels.append(f"{SHORT[m]}\n{arm}"); tv.append(cnt["TRUE"]); fv.append(cnt["FALSE"])
x = np.arange(len(labels))
ax.bar(x, tv, color="#2a6fb0", label="TRUE"); ax.bar(x, fv, bottom=tv, color="#e08a2e", label="FALSE")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7); ax.set_ylabel("free fixtures (n=55)")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
fig.tight_layout(); fig.savefig(FIGS / "truth_value_bias.pdf", bbox_inches="tight")
print("wrote", OUT / "paper_stats.json", "and figures")
