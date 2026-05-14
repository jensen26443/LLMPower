#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-experiment_results/prefill_token_power_modeling/default_350w_0_20000}"
INPUT_LENGTHS="${INPUT_LENGTHS:-default}"
REPEATS="${REPEATS:-}"
POWER="${POWER:-350}"
POLY_DEGREE="${POLY_DEGREE:-2}"
SKIP_SET_POWER="${SKIP_SET_POWER:-false}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
POWER_SETTLE_SEC="${POWER_SETTLE_SEC:-20}"
MEASUREMENT_MODE="${MEASUREMENT_MODE:-block}"
BLOCK_TARGET_WINDOW_SEC="${BLOCK_TARGET_WINDOW_SEC:-2}"
BLOCK_MIN_REQUESTS="${BLOCK_MIN_REQUESTS:-}"
BLOCK_MAX_REQUESTS="${BLOCK_MAX_REQUESTS:-}"
BLOCK_WARMUP_REQUESTS="${BLOCK_WARMUP_REQUESTS:-1}"
BLOCK_COOLDOWN_SEC="${BLOCK_COOLDOWN_SEC:-1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${DEVICE_INDEX}"
fi

echo "=========================================="
echo "Prefill Token-Power Modeling"
echo "=========================================="
echo "Output dir: ${OUTPUT_DIR}"
echo "Input lengths: ${INPUT_LENGTHS}"
if [[ -n "${REPEATS}" ]]; then
  echo "Repeats: ${REPEATS}"
else
  echo "Repeats: zone default (0-512=5, 513-3000=3, >3000=2)"
fi
echo "Power cap: ${POWER}W"
echo "Polynomial degree: ${POLY_DEGREE}"
echo "Skip set power: ${SKIP_SET_POWER}"
echo "Device index: ${DEVICE_INDEX}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Power settle sec: ${POWER_SETTLE_SEC}"
echo "Measurement mode: ${MEASUREMENT_MODE}"
echo "Block target window sec: ${BLOCK_TARGET_WINDOW_SEC}"
if [[ -n "${BLOCK_MIN_REQUESTS}" || -n "${BLOCK_MAX_REQUESTS}" ]]; then
  echo "Block min/max requests: ${BLOCK_MIN_REQUESTS:-auto}/${BLOCK_MAX_REQUESTS:-auto}"
else
  echo "Block min/max requests: zone default (0-512=30/100, 513-3000=10/30, >3000=3/5)"
fi
echo "Block warmup requests: ${BLOCK_WARMUP_REQUESTS}"
echo "Block cooldown sec: ${BLOCK_COOLDOWN_SEC}"
echo

RUN_ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --input-lengths "${INPUT_LENGTHS}"
  --power "${POWER}"
  --device-index "${DEVICE_INDEX}"
  --power-settle-sec "${POWER_SETTLE_SEC}"
  --measurement-mode "${MEASUREMENT_MODE}"
  --block-target-window-sec "${BLOCK_TARGET_WINDOW_SEC}"
  --block-warmup-requests "${BLOCK_WARMUP_REQUESTS}"
  --block-cooldown-sec "${BLOCK_COOLDOWN_SEC}"
)

if [[ -n "${BLOCK_MIN_REQUESTS}" ]]; then
  RUN_ARGS+=(--block-min-requests "${BLOCK_MIN_REQUESTS}")
fi

if [[ -n "${BLOCK_MAX_REQUESTS}" ]]; then
  RUN_ARGS+=(--block-max-requests "${BLOCK_MAX_REQUESTS}")
fi

if [[ -n "${REPEATS}" ]]; then
  RUN_ARGS+=(--repeats "${REPEATS}")
fi

if [[ "${SKIP_SET_POWER}" == "true" ]]; then
  RUN_ARGS+=(--skip-set-power)
fi

python run_prefill_token_power_modeling.py "${RUN_ARGS[@]}"

python analyze_prefill_token_power_modeling.py \
  --input-dir "${OUTPUT_DIR}" \
  --poly-degree "${POLY_DEGREE}"

echo
echo "Done. Images: ${OUTPUT_DIR}/images"
