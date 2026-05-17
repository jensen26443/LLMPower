#!/bin/bash
#
# 解码阶段功率策略评估实验 - 批量运行脚本
#

set -euo pipefail

# 配置参数
SUDO_PASSWORD="${SUDO_PASSWORD:-}"
OUTPUT_DIR="results_decode/strategy_evaluation_full"
MODEL_PATH="./Qwen2.5-7B-Instruct-AWQ"
TOKENIZER_PATH="./Qwen2.5-7B-Instruct-AWQ"
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
echo "解码阶段功率策略评估实验"
echo "=========================================="
echo ""
echo "配置参数:"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  模型路径: ${MODEL_PATH}"
echo "  Tokenizer 路径: ${TOKENIZER_PATH}"
echo "  服务模型名: ${SERVED_MODEL_NAME}"
echo "  服务地址: ${BASE_URL}"
echo "  每配置批次数: ${REPEATS_PER_BATCH}"
echo "  Full Repeats: ${FULL_REPEATS}"
echo "  并发数列表: ${CONCURRENCY_VALUES}"
echo "  Prompt Token 数: ${PROMPT_TOKEN_COUNT}"
echo "  Sampling Seed: ${SAMPLING_SEED}"
echo "  Warmup Batches: ${WARMUP_BATCHES}"
echo "  Monitor Warmup Batches: ${MONITOR_WARMUP_BATCHES}"
echo "  跳过功率设置: ${SKIP_SET_POWER}"
if [ -n "${ONLY_STRATEGY}" ]; then
    echo "  仅运行策略: ${ONLY_STRATEGY}"
fi
echo ""
echo "注意: 请先确认本地 vLLM 服务已启动并可访问 ${BASE_URL}"
echo ""

mkdir -p "${OUTPUT_DIR}"

for CONCURRENCY in ${CONCURRENCY_VALUES}; do
    CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_c${CONCURRENCY}"
    mkdir -p "${CURRENT_OUTPUT_DIR}"

    CMD=(
        python run_decode_strategy_evaluation.py
        --output-dir "${CURRENT_OUTPUT_DIR}"
        --model-path "${MODEL_PATH}"
        --tokenizer-path "${TOKENIZER_PATH}"
        --served-model-name "${SERVED_MODEL_NAME}"
        --base-url "${BASE_URL}"
        --repeats-per-batch "${REPEATS_PER_BATCH}"
        --full-repeats "${FULL_REPEATS}"
        --concurrency "${CONCURRENCY}"
        --prompt-token-count "${PROMPT_TOKEN_COUNT}"
        --sampling-seed "${SAMPLING_SEED}"
        --warmup-batches "${WARMUP_BATCHES}"
        --monitor-warmup-batches "${MONITOR_WARMUP_BATCHES}"
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

    echo "步骤1: 运行解码阶段功率策略评估实验 (concurrency=${CONCURRENCY})..."
    "${CMD[@]}"

    echo ""
    echo "步骤2: 分析实验结果 (concurrency=${CONCURRENCY})..."
    python analyze_decode_strategy_evaluation.py \
        --result-dir "${CURRENT_OUTPUT_DIR}" \
        --output-dir "${CURRENT_OUTPUT_DIR}/images"
done

MERGED_OUTPUT_DIR="${OUTPUT_DIR}_merged"
MERGED_RESULT_DIRS=""
for CONCURRENCY in ${CONCURRENCY_VALUES}; do
    CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_c${CONCURRENCY}"
    if [ -n "${MERGED_RESULT_DIRS}" ]; then
        MERGED_RESULT_DIRS="${MERGED_RESULT_DIRS},${CURRENT_OUTPUT_DIR}"
    else
        MERGED_RESULT_DIRS="${CURRENT_OUTPUT_DIR}"
    fi
done

mkdir -p "${MERGED_OUTPUT_DIR}"
echo ""
echo "步骤3: 分析合并结果..."
python analyze_decode_strategy_evaluation.py \
    --result-dirs "${MERGED_RESULT_DIRS}" \
    --output-dir "${MERGED_OUTPUT_DIR}/images"

echo ""
echo "=========================================="
echo "全部完成！"
echo "=========================================="
echo ""
echo "实验数据目录前缀: ${OUTPUT_DIR}_c*"
echo "合并图表目录: ${MERGED_OUTPUT_DIR}/images"
echo ""
