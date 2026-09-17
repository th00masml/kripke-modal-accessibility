#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

./progress.sh

PY=".venv/bin/python"
"$PY" - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("outputs/summary.json").read_text())
non_present = summary.get("non_present_arms", [])
if non_present:
    print("ERROR: publication gate failed; run does not have full arm coverage:")
    for arm in non_present:
        print(f" - {arm}")
    raise SystemExit(2)

print("publication gate passed: full scheduled-arm coverage")
PY
