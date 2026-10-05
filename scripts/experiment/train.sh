#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${1:-}}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "usage: CONFIG_PATH=config.json train.sh  (or train.sh config.json)" >&2
  exit 64
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "training config is missing: $CONFIG_PATH" >&2
  exit 66
fi

CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
LOG_FILE="$("$PYTHON_BIN" - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))
run_root = Path(config.get("run_root") or "runs")
if not run_root.is_absolute():
    run_root = (Path(sys.argv[2]) / run_root).resolve()
print(run_root / config["run_id"] / "logs" / "runner.log")
PY
)"
mkdir -p "$(dirname "$LOG_FILE")"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -u -m runners.evolve_stream --config "$CONFIG_PATH" 2>&1 | tee -a "$LOG_FILE"
