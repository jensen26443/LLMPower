#!/bin/bash
#
# Decode 阶段固定 power-cap batch sweep。
#

set -euo pipefail

SUDO_PASSWORD="${SUDO_PASSWORD:-}"
OUTPUT_DIR="experiment_results/decode_power_cap_batch_sweep/smoke_q64_out100"
MODEL_PATH="./Qwen2.5-7B-Instruct-AWQ"
TOKENIZER_PATH="./Qwen2.5-7B-Instruct-AWQ"
SERVED_MODEL_NAME="Qwen2.5-7B-Instruct-AWQ"
SHAREGPT_DIR="./input/ShareGPT"
BASE_URL="http://localhost:8000/v1"
QUERY_COUNTS="64"
OUTPUT_LENGTHS="100"
DECODE_POWER_CAPS="150,170,190,210,230,250,300"
REPEATS_PER_CAP=3
WARMUP_BATCHES=1
MONITOR_WARMUP_BATCHES=1
INTER_BATCH_SEC=0.3
QUEUE_SEED=20260426
SAMPLING_SEED=20260426
SKIP_SET_POWER=false

echo "=========================================="
echo "Decode Power-Cap Batch Sweep"
echo "=========================================="
echo "输出目录: ${OUTPUT_DIR}"
echo "Query Counts: ${QUERY_COUNTS}"
echo "Output Lengths: ${OUTPUT_LENGTHS}"
echo "Decode Power Caps: ${DECODE_POWER_CAPS}"
echo "Repeats Per Cap: ${REPEATS_PER_CAP}"
echo ""

mkdir -p "${OUTPUT_DIR}"

CMD=(
    python run_decode_power_cap_batch_sweep.py
    --output-dir "${OUTPUT_DIR}"
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --sharegpt-dir "${SHAREGPT_DIR}"
    --base-url "${BASE_URL}"
    --query-counts "${QUERY_COUNTS}"
    --output-lengths "${OUTPUT_LENGTHS}"
    --decode-power-caps "${DECODE_POWER_CAPS}"
    --repeats-per-cap "${REPEATS_PER_CAP}"
    --warmup-batches "${WARMUP_BATCHES}"
    --monitor-warmup-batches "${MONITOR_WARMUP_BATCHES}"
    --inter-batch-sec "${INTER_BATCH_SEC}"
    --queue-seed "${QUEUE_SEED}"
    --sampling-seed "${SAMPLING_SEED}"
)

if [ -n "${SUDO_PASSWORD}" ]; then
    CMD+=(--sudo-password "${SUDO_PASSWORD}")
fi

if [ "${SKIP_SET_POWER}" = true ]; then
    CMD+=(--skip-set-power)
fi

echo "步骤1: 运行 decode power-cap sweep..."
"${CMD[@]}"

echo ""
echo "步骤2: 分析 sweep 结果..."
python analyze_decode_power_cap_batch_sweep.py \
    --result-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/images"

echo ""
echo "实验数据目录: ${OUTPUT_DIR}"
echo "图表目录: ${OUTPUT_DIR}/images"
