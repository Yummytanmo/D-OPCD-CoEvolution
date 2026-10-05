#!/usr/bin/env bash
set -euo pipefail
umask 022

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DOPCD_LAUNCHER_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "D-OPCD environment is missing: ${PYTHON}" >&2
  exit 1
fi

exec "${PYTHON}" "${PROJECT_ROOT}/baselines/launch.py" "$@"

