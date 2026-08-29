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

This currently runs unconstrained arms only (`--skip-constrained`) until
backend token-context support is available for valid constrained decoding.

Important: when `--skip-constrained` is used, constrained and
constrained_tuned are omitted from outputs and reports. Missing rows mean
"not run", not a measured zero.
For publication use, treat this as partial coverage and report it explicitly.

3. Artifacts

- outputs/raw.jsonl
- outputs/scored.jsonl
- outputs/summary.json
- RESULTS.md

## Why this is interesting

This benchmark probes the gap between formal output validity and true semantic correctness in Kripke models.

## Current limitation

Constrained arms require stateful token-prefix visibility in the logits processor.
With the current MLX generation path used here, generated-token context may not
be exposed to the processor, which invalidates trie-state advancement.
The runner therefore aborts constrained runs in that state instead of emitting
misleading constrained metrics.

## Constrained context diagnostic test

To verify whether the MLX logits processor receives growing generated-token
context (required by trie-state constrained decoding), run:

.venv/bin/python src/test_constrained_context.py

The script returns PASS only when processor trace shows non-zero generated
prefix length on at least one call; otherwise it fails with an explicit error.