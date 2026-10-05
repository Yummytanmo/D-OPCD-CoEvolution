#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "${Z_IMAGE_PYTHON:-}" ]]; then
  PYTHON_BIN="${Z_IMAGE_PYTHON}"
elif [[ -n "${GENERATOR_PYTHON:-}" ]]; then
  PYTHON_BIN="${GENERATOR_PYTHON}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi

export Z_IMAGE_MODEL_PATH="${Z_IMAGE_MODEL_PATH:?Set Z_IMAGE_MODEL_PATH to the local generator checkpoint}"
export Z_IMAGE_LORA_PATH="${Z_IMAGE_LORA_PATH:-}"
export Z_IMAGE_LORA_SCALE="${Z_IMAGE_LORA_SCALE:-1.0}"
export Z_IMAGE_NUM_GPUS="${Z_IMAGE_NUM_GPUS:-1}"
export Z_IMAGE_HOST="${Z_IMAGE_HOST:-0.0.0.0}"
export Z_IMAGE_PORT="${Z_IMAGE_PORT:-8001}"
export Z_IMAGE_RESOLUTION="${Z_IMAGE_RESOLUTION:-1024}"
export Z_IMAGE_STEPS="${Z_IMAGE_STEPS:-9}"
export Z_IMAGE_CFG_SCALE="${Z_IMAGE_CFG_SCALE:-0.0}"
export Z_IMAGE_TIMEOUT_SECONDS="${Z_IMAGE_TIMEOUT_SECONDS:-600}"
export Z_IMAGE_SERVICE_CONFIG_PATH="${Z_IMAGE_SERVICE_CONFIG_PATH:-${PROJECT_ROOT}/configs/generator.z-image.json}"
export Z_IMAGE_BACKGROUND_LOAD_LOG="${Z_IMAGE_BACKGROUND_LOAD_LOG:-${PROJECT_ROOT}/evaluation/logs/generator/z-image/background-load-$(date -u +%Y%m%d-%H%M%S)-$$.jsonl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Z-Image startup failed: Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${Z_IMAGE_MODEL_PATH}/model_index.json" ]]; then
  echo "Z-Image startup failed: model is unavailable: ${Z_IMAGE_MODEL_PATH}" >&2
  exit 1
fi
if [[ -n "${Z_IMAGE_LORA_PATH}" && ! -f "${Z_IMAGE_LORA_PATH}" ]]; then
  echo "Z-Image startup failed: LoRA is unavailable: ${Z_IMAGE_LORA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${Z_IMAGE_SERVICE_CONFIG_PATH}" ]]; then
  echo "Z-Image startup failed: service config is unavailable: ${Z_IMAGE_SERVICE_CONFIG_PATH}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
echo "Starting Z-Image model=${Z_IMAGE_MODEL_PATH} lora=${Z_IMAGE_LORA_PATH:-none} lora_scale=${Z_IMAGE_LORA_SCALE} gpus=${Z_IMAGE_NUM_GPUS} endpoint=http://${Z_IMAGE_HOST}:${Z_IMAGE_PORT}/generate"

set +e
"${PYTHON_BIN}" -u scripts/generator/z_image_background_load.py \
  --config "${Z_IMAGE_SERVICE_CONFIG_PATH}" \
  --service-url "http://127.0.0.1:${Z_IMAGE_PORT}" \
  --log-file "${Z_IMAGE_BACKGROUND_LOAD_LOG}" \
  --check
BACKGROUND_LOAD_CHECK_CODE=$?
set -e

if [[ "${BACKGROUND_LOAD_CHECK_CODE}" -eq 3 ]]; then
  echo "Z-Image background load disabled"
  exec "${PYTHON_BIN}" -u agent/server/z_image.py
fi
if [[ "${BACKGROUND_LOAD_CHECK_CODE}" -ne 0 ]]; then
  echo "Z-Image startup failed: background-load validation exit_code=${BACKGROUND_LOAD_CHECK_CODE}" >&2
  exit "${BACKGROUND_LOAD_CHECK_CODE}"
fi

SERVER_PID=""
BACKGROUND_LOAD_PID=""
cleanup() {
  for child_pid in "${BACKGROUND_LOAD_PID}" "${SERVER_PID}"; do
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
      kill "${child_pid}" 2>/dev/null || true
    fi
  done
  for child_pid in "${BACKGROUND_LOAD_PID}" "${SERVER_PID}"; do
    if [[ -n "${child_pid}" ]]; then
      wait "${child_pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${PYTHON_BIN}" -u agent/server/z_image.py &
SERVER_PID=$!
echo "Starting local Z-Image background load log=${Z_IMAGE_BACKGROUND_LOAD_LOG}"
"${PYTHON_BIN}" -u scripts/generator/z_image_background_load.py \
  --config "${Z_IMAGE_SERVICE_CONFIG_PATH}" \
  --service-url "http://127.0.0.1:${Z_IMAGE_PORT}" \
  --log-file "${Z_IMAGE_BACKGROUND_LOAD_LOG}" &
BACKGROUND_LOAD_PID=$!

set +e
wait -n "${SERVER_PID}" "${BACKGROUND_LOAD_PID}"
CHILD_EXIT_CODE=$?
set -e
if kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "Z-Image background load exited unexpectedly exit_code=${CHILD_EXIT_CODE}" >&2
  if [[ "${CHILD_EXIT_CODE}" -eq 0 ]]; then
    CHILD_EXIT_CODE=1
  fi
fi
exit "${CHILD_EXIT_CODE}"
