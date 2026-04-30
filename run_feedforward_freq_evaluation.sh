#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results_freq/feedforward_freq_evaluation_full}"
MODEL_PATH="${MODEL_PATH:-./Qwen2.5-7B-Instruct-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-7B-Instruct-AWQ}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./Qwen2.5-7B-Instruct-AWQ}"
SHAREGPT_DIR="${SHAREGPT_DIR:-./filtered_prompts}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
OUTPUT_LENGTHS="${OUTPUT_LENGTHS:-100,200}"
REPEATS_PER_BATCH="${REPEATS_PER_BATCH:-10}"
FULL_REPEATS="${FULL_REPEATS:-3}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
MONITOR_WARMUP_BATCHES="${MONITOR_WARMUP_BATCHES:-1}"
INTER_BATCH_SEC="${INTER_BATCH_SEC:-0.8}"
QUEUE_SEED="${QUEUE_SEED:-20260329}"
SAMPLING_SEED="${SAMPLING_SEED:-20260329}"
STRATEGY_NAMES="${STRATEGY_NAMES:-baseline_350w,ff_v2_recommended,ff_v3_freq_recommended}"
PREFILL_RECOMMENDATION_PATH="${PREFILL_RECOMMENDATION_PATH:-results_freq/prefill_freq_sweep_full/images/prefill_freq_recommendation.json}"
DECODE_RECOMMENDATION_PATH="${DECODE_RECOMMENDATION_PATH:-results_freq/decode_freq_sweep_full/images/decode_freq_recommendation.json}"

python run_feedforward_freq_evaluation.py \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --sharegpt-dir "${SHAREGPT_DIR}" \
  --base-url "${BASE_URL}" \
  --output-lengths "${OUTPUT_LENGTHS}" \
  --repeats-per-batch "${REPEATS_PER_BATCH}" \
  --full-repeats "${FULL_REPEATS}" \
  --warmup-batches "${WARMUP_BATCHES}" \
  --monitor-warmup-batches "${MONITOR_WARMUP_BATCHES}" \
  --inter-batch-sec "${INTER_BATCH_SEC}" \
  --queue-seed "${QUEUE_SEED}" \
  --sampling-seed "${SAMPLING_SEED}" \
  --strategy-names "${STRATEGY_NAMES}" \
  --prefill-recommendation-path "${PREFILL_RECOMMENDATION_PATH}" \
  --decode-recommendation-path "${DECODE_RECOMMENDATION_PATH}"

python analyze_feedforward_freq_evaluation.py \
  --result-dir "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}/images"
