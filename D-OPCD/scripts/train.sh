#!/usr/bin/env bash
set -euo pipefail
umask 022

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER_PYTHON="${DOPCD_LAUNCHER_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

if [[ ! -x "${LAUNCHER_PYTHON}" ]]; then
  echo "D-OPCD uv environment is missing: ${LAUNCHER_PYTHON}" >&2
  echo "Run: cd ${PROJECT_ROOT} && uv sync" >&2
  exit 1
fi

CONFIG=""
previous=""
for argument in "$@"; do
  if [[ "${previous}" == "--config" ]]; then
    CONFIG="${argument}"
    break
  fi
  previous="${argument}"
done
if [[ -z "${CONFIG}" ]]; then
  echo "Training launcher requires --config CONFIG" >&2
  exit 2
fi
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${PROJECT_ROOT}/${CONFIG#./}"
fi

RUN_ID="$("${LAUNCHER_PYTHON}" -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["run_id"])' \
  "${CONFIG}")"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Unsafe or missing run_id in ${CONFIG}: ${RUN_ID}" >&2
  exit 2
fi

RUN_ROOT="${DOPCD_RESULTS_ROOT:-${PROJECT_ROOT}/results}/runs/${RUN_ID}"
BOOTSTRAP_LOG="${RUN_ROOT}/metadata/launcher-execution.log"
mkdir -p "${RUN_ROOT}/metadata"
touch "${BOOTSTRAP_LOG}"
chmod 0644 "${BOOTSTRAP_LOG}"
cd "${PROJECT_ROOT}"

set +e
"${LAUNCHER_PYTHON}" "${PROJECT_ROOT}/scripts/launch_training.py" "$@" \
  2>&1 | tee -a "${BOOTSTRAP_LOG}"
launcher_status=${PIPESTATUS[0]}
set -e
exit "${launcher_status}"
