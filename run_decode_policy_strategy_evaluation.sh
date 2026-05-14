#!/bin/bash
#
# Decode policy strategy evaluation for paper figures.
#

set -euo pipefail

SUDO_PASSWORD="${SUDO_PASSWORD:-}"
OUTPUT_DIR="experiment_results/decode_strategy/strategy_evaluation_policy_retry"
MODEL_PATH="./Qwen2.5-7B-Instruct-AWQ"
TOKENIZER_PATH="./Qwen2.5-7B-Instruct-AWQ"
SHAREGPT_DIR="./input/ShareGPT"
SERVED_MODEL_NAME="Qwen2.5-7B-Instruct-AWQ"
BASE_URL="http://localhost:8000/v1"
REPEATS_PER_BATCH=50
FULL_REPEATS=3
CONCURRENCY_VALUES="8 16"
PROMPT_TOKEN_COUNT=1
SAMPLING_SEED=20260329
WARMUP_BATCHES=3
MONITOR_WARMUP_BATCHES=1
SKIP_SET_POWER=false
ONLY_STRATEGY=""

echo "=========================================="
echo "Decode policy strategy evaluation"
echo "=========================================="
echo ""
echo "Config:"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Model path: ${MODEL_PATH}"
echo "  Tokenizer path: ${TOKENIZER_PATH}"
echo "  ShareGPT dir: ${SHAREGPT_DIR}"
echo "  Served model name: ${SERVED_MODEL_NAME}"
echo "  Base URL: ${BASE_URL}"
echo "  Repeats per config: ${REPEATS_PER_BATCH}"
echo "  Full repeats: ${FULL_REPEATS}"
echo "  Number of query values: ${CONCURRENCY_VALUES}"
echo "  Prompt token count: ${PROMPT_TOKEN_COUNT}"
echo "  Sampling seed: ${SAMPLING_SEED}"
echo "  Warmup batches: ${WARMUP_BATCHES}"
echo "  Monitor warmup batches: ${MONITOR_WARMUP_BATCHES}"
echo "  Skip power setting: ${SKIP_SET_POWER}"
if [ -n "${ONLY_STRATEGY}" ]; then
    echo "  Only strategy: ${ONLY_STRATEGY}"
fi
echo ""
echo "Please make sure local vLLM service is available at ${BASE_URL} before running."
echo ""

mkdir -p "${OUTPUT_DIR}"

for CONCURRENCY in ${CONCURRENCY_VALUES}; do
    CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_q${CONCURRENCY}"
    mkdir -p "${CURRENT_OUTPUT_DIR}"

    CMD=(
        python run_decode_strategy_evaluation.py
        --output-dir "${CURRENT_OUTPUT_DIR}"
        --model-path "${MODEL_PATH}"
        --tokenizer-path "${TOKENIZER_PATH}"
        --sharegpt-dir "${SHAREGPT_DIR}"
        --served-model-name "${SERVED_MODEL_NAME}"
        --base-url "${BASE_URL}"
        --repeats-per-batch "${REPEATS_PER_BATCH}"
        --full-repeats "${FULL_REPEATS}"
        --concurrency "${CONCURRENCY}"
        --prompt-token-count "${PROMPT_TOKEN_COUNT}"
        --sampling-seed "${SAMPLING_SEED}"
        --warmup-batches "${WARMUP_BATCHES}"
        --monitor-warmup-batches "${MONITOR_WARMUP_BATCHES}"
        --strategy-set decode_policy_eval
    )

    if [ -n "${SUDO_PASSWORD}" ]; then
        CMD+=(--sudo-password "${SUDO_PASSWORD}")
    fi

    if [ "${SKIP_SET_POWER}" = true ]; then
        CMD+=(--skip-set-power)
    fi

    if [ -n "${ONLY_STRATEGY}" ]; then
        CMD+=(--only-strategy "${ONLY_STRATEGY}")
    fi

    echo "Step 1: run decode policy evaluation (query=${CONCURRENCY})..."
    "${CMD[@]}"

    echo ""
    echo "Step 2: analyze results (query=${CONCURRENCY})..."
    python analyze_decode_strategy_evaluation.py \
        --result-dir "${CURRENT_OUTPUT_DIR}" \
        --output-dir "${CURRENT_OUTPUT_DIR}/images"
done

MERGED_OUTPUT_DIR="${OUTPUT_DIR}_merged"
MERGED_RESULT_DIRS=""
for CONCURRENCY in ${CONCURRENCY_VALUES}; do
    CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_q${CONCURRENCY}"
    if [ -n "${MERGED_RESULT_DIRS}" ]; then
        MERGED_RESULT_DIRS="${MERGED_RESULT_DIRS},${CURRENT_OUTPUT_DIR}"
    else
        MERGED_RESULT_DIRS="${CURRENT_OUTPUT_DIR}"
    fi
done

mkdir -p "${MERGED_OUTPUT_DIR}"
echo ""
echo "Step 3: analyze merged results..."
python analyze_decode_strategy_evaluation.py \
    --result-dirs "${MERGED_RESULT_DIRS}" \
    --output-dir "${MERGED_OUTPUT_DIR}/images"

echo ""
echo "=========================================="
echo "Done"
echo "=========================================="
echo ""
echo "Data dirs: ${OUTPUT_DIR}_q*"
echo "Merged figures: ${MERGED_OUTPUT_DIR}/images"
echo ""
