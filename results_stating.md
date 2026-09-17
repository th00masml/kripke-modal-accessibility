# Results - Kripke Modal Accessibility

## Kripke framing

This benchmark compares formal output validity against actual modal-formula truth in Kripke models (R, V, target world).

## Run coverage

| Model | Arm | observed_rows | expected_rows | status | reason |
|---|---:|---:|---:|---:|---:|
| Qwen/Qwen2.5-0.5B-Instruct | constrained | 160 | 160 | present |  |
| Qwen/Qwen2.5-0.5B-Instruct | constrained_tuned | 160 | 160 | present |  |
| Qwen/Qwen2.5-0.5B-Instruct | naive | 160 | 160 | present |  |
| Qwen/Qwen2.5-0.5B-Instruct | prompt_tuned | 160 | 160 | present |  |
| Qwen/Qwen2.5-0.5B-Instruct | prompted | 160 | 160 | present |  |
| Qwen/Qwen2.5-1.5B-Instruct | constrained | 160 | 160 | present |  |
| Qwen/Qwen2.5-1.5B-Instruct | constrained_tuned | 160 | 160 | present |  |
| Qwen/Qwen2.5-1.5B-Instruct | naive | 160 | 160 | present |  |
| Qwen/Qwen2.5-1.5B-Instruct | prompt_tuned | 160 | 160 | present |  |
| Qwen/Qwen2.5-1.5B-Instruct | prompted | 160 | 160 | present |  |

## Class counts by model and arm

| Model | Arm | exact | abstention | correct_abstain | valid_wrong_judgement | valid_spurious_judgement | invented_schema | canonical_found | membership_violations | format_violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2.5-0.5B-Instruct | constrained | 51 | 0 | 0 | 29 | 80 | 0 | 160 | 0 | 0 |
| Qwen/Qwen2.5-0.5B-Instruct | constrained_tuned | 51 | 0 | 0 | 29 | 80 | 0 | 160 | 0 | 0 |
| Qwen/Qwen2.5-0.5B-Instruct | naive | 0 | 0 | 0 | 0 | 0 | 160 | 0 | 0 | 160 |
| Qwen/Qwen2.5-0.5B-Instruct | prompt_tuned | 19 | 0 | 0 | 17 | 0 | 124 | 143 | 107 | 17 |
| Qwen/Qwen2.5-0.5B-Instruct | prompted | 26 | 0 | 0 | 19 | 0 | 115 | 159 | 114 | 1 |
| Qwen/Qwen2.5-1.5B-Instruct | constrained | 52 | 0 | 0 | 28 | 80 | 0 | 160 | 0 | 0 |
| Qwen/Qwen2.5-1.5B-Instruct | constrained_tuned | 52 | 0 | 0 | 28 | 80 | 0 | 160 | 0 | 0 |
| Qwen/Qwen2.5-1.5B-Instruct | naive | 25 | 0 | 0 | 24 | 0 | 111 | 105 | 56 | 55 |
| Qwen/Qwen2.5-1.5B-Instruct | prompt_tuned | 0 | 0 | 0 | 0 | 0 | 160 | 0 | 0 | 160 |
| Qwen/Qwen2.5-1.5B-Instruct | prompted | 8 | 0 | 0 | 6 | 0 | 146 | 56 | 42 | 104 |

## Paired exact McNemar (prompted vs constrained)

If constrained rows are absent in raw data (for example after --skip-constrained), McNemar is reported as n/a.

| Model | paired_count | prompted-only correct | constrained-only correct | p-value | status | reason |
|---|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2.5-0.5B-Instruct | 160 | 0 | 25 | 5.96046e-08 | ok |  |
| Qwen/Qwen2.5-1.5B-Instruct | 160 | 0 | 44 | 1.13687e-13 | ok |  |

## Re-analysis (2026-09-17): the McNemar result above should not be read as a semantic gain

Everything in this section is recomputed from `outputs/scored.jsonl` by
`src/analysis_paper.py`, which writes `outputs/paper_stats.json` and the
figures in `paper/figs/`. Full write-up in `paper/main.tex`.

**1. The constraint set leaks the answer on 25/80 in-language fixtures.**
The admissible set was built from the gold judgements (27 distinct strings).
For 5 of the 16 (world, formula) pairs only one truth value ever occurs in
gold, so the trie admits exactly one complete string for those fixtures:

| world | formula | only admissible value | fixtures |
|---|---|---|---:|
| w1 | AND(BOX(p),DIA(q)) | FALSE | 5 |
| w1 | NOT(BOX(NOT(p))) | TRUE | 5 |
| w2 | AND(BOX(p),DIA(q)) | FALSE | 5 |
| w2 | BOX(p) | FALSE | 5 |
| w2 | NOT(BOX(NOT(p))) | TRUE | 5 |

Constrained arms score 25/25 on these ("forced") and **26/55 (0.5B) and
27/55 (1.5B) on the other 55 ("free")**, binomial p vs 0.5 = 0.79 and 1.00.
Of the 51-52 exact matches credited to constrained decoding, 25 come from
the constraint set and the rest are chance.

Fix: `run.py --allowed-set product` builds the full world x formula x truth
set (32 strings). **Not re-run**; the cached run used `gold`
(`outputs/run_meta.json` now records this).

**2. The truth value is a constant per model.** On the 55 free fixtures the
0.5B model emits TRUE 55/55 under the constraint (80/80 in-language under
`prompted`); the 1.5B model emits FALSE 51/55 (73/80 prompted, 75/80 naive).
Per-formula accuracy is the gold base rate of that constant
(NOT(BOX(NOT(p))) is TRUE 18/20 -> 0.5B scores 18/20; AND(...) is FALSE
18/20 -> 1.5B scores 16/20).

**3. Strict parsing scored formatting as semantics.** A lenient parser
(`w_0` -> `w0`, spaces inside formulas removed, `DIAMOND` -> `DIA`,
missing `=` in `|=`) recovers a canonical judgement from 78-80/80
in-language outputs of the 1.5B prompt arms. Under it:

| model | arm | strict exact | lenient exact | world+formula right | truth right given W+F |
|---|---|---:|---:|---:|---|
| 0.5B | prompted | 26 | 28 | 70 | 28/70 |
| 0.5B | constrained | 51 | 51 | 80 | 51/80 |
| 1.5B | naive | 25 | 42 | 79 | 42/79 |
| 1.5B | prompted | 8 | 40 | 78 | 40/78 |
| 1.5B | prompt_tuned | 0 | 25 | 73 | 25/73 |
| 1.5B | constrained | 52 | 52 | 80 | 52/80 |

Free-set McNemar, lenient prompted vs constrained: 1.5B 2 vs 2 (p=1.0);
0.5B 0 vs 8 (p=0.008, driven by the 0.5B model dropping the world or
mangling the formula without the constraint, not by truth values).

**4. Nobody ever abstains.** 0 empty outputs in 1,600. Constrained arms
turn every out-of-language fixture into a fluent in-language judgement about
p/q at the target world; unconstrained arms at least mention `r` (54-80/80).

**Bottom line.** The constraint removed all format and membership
violations at ~0.05-0.08 s/item. It did not change the truth value the
model was going to emit. The "prompted vs constrained" McNemar test above
compares a coin that always lands in the admissible set against outputs
discarded for an underscore.

## Not tested

- Leak-free admissible set (`--allowed-set product`): implemented, not run.
- Balanced truth values per formula, few-shot, chain-of-thought, models > 1.5B.
