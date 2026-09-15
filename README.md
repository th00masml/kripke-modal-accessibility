# Kripke Modal Accessibility Benchmark

This project targets Kripke modal semantics: accessibility relations, possible worlds, and truth of modal formulas.

Answer format:

w_i |= FORMULA iff TRUE.

or

w_i |= FORMULA iff FALSE.

## Quick start

1. Create environment

python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

2. Full pipeline

./progress.sh

This runs all five arms, including constrained and constrained_tuned, on a
CUDA GPU via transformers. transformers'' `LogitsProcessor` receives the full
`input_ids` sequence (prompt + generated-so-far) on every decoding step, so
trie-state advancement for constrained decoding has valid token-prefix
context - unlike the earlier MLX-based path (see below).

3. Artifacts

- outputs/raw.jsonl
- outputs/scored.jsonl
- outputs/summary.json
- RESULTS.md

## Why this is interesting

This benchmark probes the gap between formal output validity and true semantic correctness in Kripke models.

## Backend history

This project originally targeted Apple MLX (`mlx-lm`) for local inference.
On that backend, generated-token context was not reliably exposed to the
logits processor across decoding steps, which invalidated trie-state
advancement for constrained arms; runs were limited to `--skip-constrained`.
The project has since moved to a CUDA/PyTorch (`transformers`) backend,
which resolves this because `LogitsProcessor.__call__` always receives the
full `input_ids` tensor.

## Constrained context diagnostic test

To verify whether the logits processor receives growing generated-token
context (required by trie-state constrained decoding), run:

.venv/bin/python src/test_constrained_context.py

The script returns PASS only when processor trace shows non-zero generated
prefix length on at least one call after step 0; otherwise it fails with an
explicit error.