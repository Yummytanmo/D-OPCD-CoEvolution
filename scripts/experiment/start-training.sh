#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
config="${1:?usage: start-training.sh CONFIG}"
readarray -t values < <(.venv/bin/python - "$config" <<'PY'
import json, sys
from pathlib import Path
c = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(c["run_id"])
print(c.get("run_root", "runs"))
PY
)
log_directory="${values[1]}/${values[0]}/logs"
pid_file="$log_directory/training.pid"
session_file="$log_directory/tmux-session"
session="evolution-${values[0]}"
mkdir -p "$log_directory"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "training already running run_id=${values[0]} tmux=$session"
  exit 0
fi

tmux new-session -d -s "$session" \
  "exec ./scripts/experiment/train.sh '$config'"
pid="$(tmux display-message -p -t "$session:0.0" '#{pane_pid}')"
printf '%s\n' "$pid" >"$pid_file"
printf '%s\n' "$session" >"$session_file"
echo "training started run_id=${values[0]} tmux=$session pid=$pid log=$log_directory/runner.log"
