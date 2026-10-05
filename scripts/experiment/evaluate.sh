#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${1:-}}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "usage: evaluate.sh CONFIG [SPLIT] [VARIANT] [CHECKPOINT]" >&2
  exit 64
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
fi
SPLIT="${SPLIT:-${2:-validation}}"
VARIANT="${VARIANT:-${3:-all}}"
CHECKPOINT="${CHECKPOINT:-${4:-latest}}"
WORKERS="${WORKERS:-4}"
MAX_TASKS="${MAX_TASKS:-}"
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

args=(
  "$PYTHON_BIN" -u -m runners.evaluate_frozen
  --config "$CONFIG_PATH" --split "$SPLIT" --variant "$VARIANT"
  --checkpoint "$CHECKPOINT" --workers "$WORKERS"
)
if [[ -n "$MAX_TASKS" ]]; then
  args+=(--max-tasks "$MAX_TASKS")
fi
if [[ -n "${EVAL_NAME:-}" ]]; then
  args+=(--name "$EVAL_NAME")
fi
cd "$PROJECT_ROOT"
exec "${args[@]}"
