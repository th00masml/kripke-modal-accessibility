#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv/bin/python. Create the virtualenv and install requirements first."
  exit 1
fi

"$PY" src/fixtures.py
"$PY" src/constraint.py
"$PY" src/run.py
"$PY" src/score.py
"$PY" src/report.py

echo "done: outputs/raw.jsonl outputs/scored.jsonl outputs/summary.json RESULTS.md"