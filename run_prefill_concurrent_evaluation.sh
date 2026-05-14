#!/usr/bin/env bash

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-experiment_results/prefill_concurrent_evaluation/prefill_strategy_gpu0_r50x3}"
MODEL_PATH="${MODEL_PATH:-./Qwen2.5-7B-Instruct-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-7B-Instruct-AWQ}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./Qwen2.5-7B-Instruct-AWQ}"
SHAREGPT_DIR="${SHAREGPT_DIR:-./input/ShareGPT}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
REPEATS_PER_BATCH=50
FULL_REPEATS=3
WARMUP_BATCHES=2
MONITOR_WARMUP_BATCHES=1
INTER_BATCH_SEC=0.3
QUEUE_SEED=20260401
SAMPLING_SEED=20260401
TTFT_THRESHOLD_PCT=5.0
SKIP_SET_POWER=false
DEVICE_INDEX=0
STRATEGY_NAMES="baseline_350w,prefill_token_fit,prefill_token_fit_plus25w,prefill_manual_buckets"

REPEATS_PER_BATCH="${REPEATS_PER_BATCH_OVERRIDE:-${REPEATS_PER_BATCH}}"
FULL_REPEATS="${FULL_REPEATS_OVERRIDE:-${FULL_REPEATS}}"
WARMUP_BATCHES="${WARMUP_BATCHES_OVERRIDE:-${WARMUP_BATCHES}}"
MONITOR_WARMUP_BATCHES="${MONITOR_WARMUP_BATCHES_OVERRIDE:-${MONITOR_WARMUP_BATCHES}}"
INTER_BATCH_SEC="${INTER_BATCH_SEC_OVERRIDE:-${INTER_BATCH_SEC}}"
QUEUE_SEED="${QUEUE_SEED_OVERRIDE:-${QUEUE_SEED}}"
SAMPLING_SEED="${SAMPLING_SEED_OVERRIDE:-${SAMPLING_SEED}}"
TTFT_THRESHOLD_PCT="${TTFT_THRESHOLD_PCT_OVERRIDE:-${TTFT_THRESHOLD_PCT}}"
SKIP_SET_POWER="${SKIP_SET_POWER_OVERRIDE:-${SKIP_SET_POWER}}"
DEVICE_INDEX="${DEVICE_INDEX_OVERRIDE:-${DEVICE_INDEX}}"
STRATEGY_NAMES="${STRATEGY_NAMES_OVERRIDE:-${STRATEGY_NAMES}}"

RUN_CMD=(
  python run_prefill_concurrent_evaluation.py
  --output-dir "$OUTPUT_DIR"
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --tokenizer-path "$TOKENIZER_PATH"
  --sharegpt-dir "$SHAREGPT_DIR"
  --base-url "$BASE_URL"
  --repeats-per-batch "$REPEATS_PER_BATCH"
  --full-repeats "$FULL_REPEATS"
  --warmup-batches "$WARMUP_BATCHES"
  --monitor-warmup-batches "$MONITOR_WARMUP_BATCHES"
  --inter-batch-sec "$INTER_BATCH_SEC"
  --queue-seed "$QUEUE_SEED"
  --sampling-seed "$SAMPLING_SEED"
  --device-index "$DEVICE_INDEX"
  --strategy-names "$STRATEGY_NAMES"
)

if [[ "${SUDO_PASSWORD:-}" != "" ]]; then
  RUN_CMD+=(--sudo-password "$SUDO_PASSWORD")
fi

if [[ "$SKIP_SET_POWER" == "true" ]]; then
  RUN_CMD+=(--skip-set-power)
fi

echo "步骤1: 运行并发 prefill-only 实验..."
"${RUN_CMD[@]}"

echo "步骤2: 分析实验结果..."
python analyze_prefill_concurrent_evaluation.py \
  --result-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR/images" \
  --ttft-threshold-pct "$TTFT_THRESHOLD_PCT"
