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
