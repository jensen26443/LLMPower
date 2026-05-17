#!/bin/bash
#
# 前馈 + PID 控制对比评估 - 批量运行脚本
#

set -euo pipefail

SUDO_PASSWORD="${SUDO_PASSWORD:-}"
OUTPUT_DIR="results_decode/feedforward_pid_evaluation_full"
MODEL_PATH="./Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
TOKENIZER_PATH="./Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
SERVED_MODEL_NAME="Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
SHAREGPT_DIR="./input/ShareGPT"
BASE_URL="http://localhost:8000/v1"
OUTPUT_LENGTHS="100,200"
REPEATS_PER_BATCH=10
FULL_REPEATS=3
WARMUP_BATCHES=2
MONITOR_WARMUP_BATCHES=1
INTER_BATCH_SEC=0.3
QUEUE_SEED=20260401
SAMPLING_SEED=20260401
PID_TARGETS_PATH="./feedforward_pid_targets.json"
SKIP_SET_POWER=false
STRATEGY_NAMES="baseline_350w,ff_v2_recommended,ff_v2_pid_stable"

echo "=========================================="
echo "前馈 + PID 控制对比评估"
echo "=========================================="
echo ""
echo "配置参数:"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  输出长度: ${OUTPUT_LENGTHS}"
echo "  每配置批次数: ${REPEATS_PER_BATCH}"
echo "  Full Repeats: ${FULL_REPEATS}"
echo "  PID Targets: ${PID_TARGETS_PATH}"
echo "  策略: ${STRATEGY_NAMES}"
echo ""

mkdir -p "${OUTPUT_DIR}"

CMD=(
    python run_feedforward_pid_evaluation.py
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
    --pid-targets-path "${PID_TARGETS_PATH}"
    --strategy-names "${STRATEGY_NAMES}"
)

if [ -n "${SUDO_PASSWORD}" ]; then
    CMD+=(--sudo-password "${SUDO_PASSWORD}")
fi

if [ "${SKIP_SET_POWER}" = true ]; then
    CMD+=(--skip-set-power)
fi

echo "步骤1: 运行前馈 + PID 实验..."
"${CMD[@]}"

echo ""
echo "步骤2: 分析实验结果..."
python analyze_feedforward_pid_evaluation.py \
    --result-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/images"

echo ""
echo "=========================================="
echo "全部完成！"
echo "=========================================="
echo "实验数据目录: ${OUTPUT_DIR}"
echo "图表目录: ${OUTPUT_DIR}/images"
