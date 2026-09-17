# Problem: Kripke Modal Accessibility and Possible-World Evaluation

Goal: test whether local LLMs can generate correct Kripke-style modal semantic judgements (accessibility relation plus truth of formulas in possible worlds).

Canonical output:

w_i |= FORMULA iff TRUE.

or

w_i |= FORMULA iff FALSE.

## Membership predicate

A non-empty output is structurally valid iff it is one of the canonical judgements in the configured modal language.

Empty output is always allowed.

## What we measure

- Formal correctness (whether output belongs to the closed set of canonical judgements)
- Semantic correctness (whether judgement matches formula truth at the target Kripke world)

## Migration classes

- valid_wrong_judgement: output is canonical, but has the wrong truth value, formula, or world
- valid_spurious_judgement: output is canonical even though fixture contains an out-of-language atom and should have empty output

## Setup

- 160 deterministic fixtures
- 80 in-language fixtures (atoms p, q)
- 80 out-of-language fixtures (atom r; correct response is empty)
- 5 arms: naive, prompted, constrained, prompt_tuned, constrained_tuned
- 2 local HF models (Qwen2.5-0.5B/1.5B-Instruct) served via CUDA/PyTorch

## Research question

Does decode-time constraint improve formal membership while shifting failures into semantic error classes that are invisible to parser-only or membership-only checks?
