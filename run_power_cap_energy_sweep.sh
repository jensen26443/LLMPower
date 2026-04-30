#!/bin/bash
#
# 固定完整请求的 power-cap 能量扫描批量脚本。
#

set -euo pipefail

SUDO_PASSWORD="${SUDO_PASSWORD:-}"
OUTPUT_DIR="experiment_results/power_cap_energy_sweep/q64_out100"
MODEL_PATH="./Qwen2.5-7B-Instruct-AWQ"
TOKENIZER_PATH="./Qwen2.5-7B-Instruct-AWQ"
SERVED_MODEL_NAME="Qwen2.5-7B-Instruct-AWQ"
SHAREGPT_DIR="./input/ShareGPT"
BASE_URL="http://localhost:8000/v1"
QUERY_COUNTS="64"
OUTPUT_LENGTHS="100"
POWER_CAPS="150,170,190,210,230,250,275,300,350"
REPEATS_PER_POWER_CAP=5
WARMUP_BATCHES=1
INTER_BATCH_SEC=0.3
QUEUE_SEED=20260425
SAMPLING_SEED=20260425
SKIP_SET_POWER=false

echo "=========================================="
echo "固定完整请求 Power-Cap 能量扫描"
echo "=========================================="
echo ""
echo "配置参数:"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  模型路径: ${MODEL_PATH}"
echo "  Tokenizer 路径: ${TOKENIZER_PATH}"
echo "  服务模型名: ${SERVED_MODEL_NAME}"
echo "  ShareGPT 路径: ${SHAREGPT_DIR}"
echo "  服务地址: ${BASE_URL}"
echo "  Query Counts: ${QUERY_COUNTS}"
echo "  Output Lengths: ${OUTPUT_LENGTHS}"
echo "  Power Caps: ${POWER_CAPS}"
echo "  每个 Power Cap 批次数: ${REPEATS_PER_POWER_CAP}"
echo "  Warmup Batches: ${WARMUP_BATCHES}"
echo "  Inter Batch Interval: ${INTER_BATCH_SEC}"
echo "  Queue Seed: ${QUEUE_SEED}"
echo "  Sampling Seed: ${SAMPLING_SEED}"
echo "  跳过功率设置: ${SKIP_SET_POWER}"
echo ""
echo "注意: 请先确认本地 vLLM 服务已启动并可访问 ${BASE_URL}"
echo ""

mkdir -p "${OUTPUT_DIR}"

CMD=(
    python run_power_cap_energy_sweep.py
    --output-dir "${OUTPUT_DIR}"
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --sharegpt-dir "${SHAREGPT_DIR}"
    --base-url "${BASE_URL}"
    --query-counts "${QUERY_COUNTS}"
    --output-lengths "${OUTPUT_LENGTHS}"
    --power-caps "${POWER_CAPS}"
    --repeats-per-power-cap "${REPEATS_PER_POWER_CAP}"
    --warmup-batches "${WARMUP_BATCHES}"
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

echo "步骤1: 运行 power-cap 能量扫描..."
"${CMD[@]}"

echo ""
echo "步骤2: 分析实验结果..."
python analyze_power_cap_energy_sweep.py \
    --result-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/images"

echo ""
echo "=========================================="
echo "全部完成"
echo "=========================================="
echo ""
echo "实验数据目录: ${OUTPUT_DIR}"
echo "图表目录: ${OUTPUT_DIR}/images"
