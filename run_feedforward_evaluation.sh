#!/bin/bash
#
# 前馈控制与基线方案对比评估 - 批量运行脚本
#

set -euo pipefail

SUDO_PASSWORD="${SUDO_PASSWORD:-}"
OUTPUT_DIR="results_decode/feedforward_evaluation_full"
MODEL_PATH="./Qwen2.5-7B-Instruct-AWQ"
TOKENIZER_PATH="./Qwen2.5-7B-Instruct-AWQ"
SERVED_MODEL_NAME="Qwen2.5-7B-Instruct-AWQ"
SHAREGPT_DIR="./input/ShareGPT"
BASE_URL="http://localhost:8000/v1"
OUTPUT_LENGTHS="100,200"
REPEATS_PER_BATCH=3
FULL_REPEATS=1
WARMUP_BATCHES=2
MONITOR_WARMUP_BATCHES=1
INTER_BATCH_SEC=0.3
QUEUE_SEED=20260401
SAMPLING_SEED=20260401
SKIP_SET_POWER=false
ONLY_STRATEGY=""

echo "=========================================="
echo "前馈控制与基线方案对比评估"
echo "=========================================="
echo ""
echo "配置参数:"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  模型路径: ${MODEL_PATH}"
echo "  Tokenizer 路径: ${TOKENIZER_PATH}"
echo "  服务模型名: ${SERVED_MODEL_NAME}"
echo "  ShareGPT 路径: ${SHAREGPT_DIR}"
echo "  服务地址: ${BASE_URL}"
echo "  输出长度: ${OUTPUT_LENGTHS}"
echo "  每配置批次数: ${REPEATS_PER_BATCH}"
echo "  Full Repeats: ${FULL_REPEATS}"
echo "  Warmup Batches: ${WARMUP_BATCHES}"
echo "  Monitor Warmup Batches: ${MONITOR_WARMUP_BATCHES}"
echo "  Inter Batch Interval: ${INTER_BATCH_SEC}"
echo "  Queue Seed: ${QUEUE_SEED}"
echo "  Sampling Seed: ${SAMPLING_SEED}"
echo "  跳过功率设置: ${SKIP_SET_POWER}"
if [ -n "${ONLY_STRATEGY}" ]; then
    echo "  仅运行策略: ${ONLY_STRATEGY}"
fi
echo ""
echo "注意: 请先确认本地 vLLM 服务已启动并可访问 ${BASE_URL}"
echo ""

mkdir -p "${OUTPUT_DIR}"

CMD=(
    python run_feedforward_evaluation.py
    --output-dir "${OUTPUT_DIR}"
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --sharegpt-dir "${SHAREGPT_DIR}"
    --base-url "${BASE_URL}"
    --output-lengths "${OUTPUT_LENGTHS}"
    --repeats-per-batch "${REPEATS_PER_BATCH}"
    --full-repeats "${FULL_REPEATS}"
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

if [ -n "${ONLY_STRATEGY}" ]; then
    CMD+=(--only-strategy "${ONLY_STRATEGY}")
fi

echo "步骤1: 运行前馈控制实验..."
"${CMD[@]}"

echo ""
echo "步骤2: 分析实验结果..."
python analyze_feedforward_evaluation.py \
    --result-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/images"

echo ""
echo "=========================================="
echo "全部完成！"
echo "=========================================="
echo ""
echo "实验数据目录: ${OUTPUT_DIR}"
echo "图表目录: ${OUTPUT_DIR}/images"
echo ""
