#!/bin/bash
#
# 解码阶段离线建模实验 - 批量运行脚本
#

# 配置参数
SUDO_PASSWORD="123456"
POWER_LIMIT=350
REPEATS=5
BATCH_SIZES="1,2,4,6,8,12,16,24,32,40,48,50,56,60,64"
OUTPUT_LENGTHS="10,20,40,50,75,100,150,200,300"
OUTPUT_DIR="results_decode/decode_modeling"
MODEL_PATH="./Qwen2.5-7B-Instruct-AWQ"
SERVED_MODEL_NAME="Qwen2.5-7B-Instruct-AWQ"
TOKENIZER_PATH="./Qwen2.5-7B-Instruct-AWQ"
BASE_URL="http://localhost:8000/v1"
PROMPT_TOKEN_COUNT=1
INTER_BATCH_SEC=0.8
IDLE_BASELINE_SEC=2.0
TIME_PADDING_MS=20.0
GPU_MEMORY_UTILIZATION=0.85
ENABLE_CHUNKED_PREFILL=true
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=64
QUEUE_SEED=20260329
SAMPLING_SEED=20260329

# 运行模式
# false: 复用外部已启动的 vLLM 服务（推荐）
# true: 由 run_decode_modeling.py 自动启动和关闭服务
START_SERVER=false

# 是否跳过功率设置
SKIP_SET_POWER=false

echo "=========================================="
echo "解码阶段离线建模实验"
echo "=========================================="
echo ""
echo "配置参数:"
echo "  功率限制: ${POWER_LIMIT}W"
echo "  重复次数: ${REPEATS}"
echo "  Batch Sizes: ${BATCH_SIZES}"
echo "  输出长度: ${OUTPUT_LENGTHS}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  模型路径: ${MODEL_PATH}"
echo "  服务模型名: ${SERVED_MODEL_NAME}"
echo "  服务地址: ${BASE_URL}"
echo "  Prompt Token 数: ${PROMPT_TOKEN_COUNT}"
echo "  Chunked Prefill: ${ENABLE_CHUNKED_PREFILL}"
echo "  max_num_batched_tokens: ${MAX_NUM_BATCHED_TOKENS}"
echo "  max_num_seqs: ${MAX_NUM_SEQS}"
echo "  Queue Seed: ${QUEUE_SEED}"
echo "  Sampling Seed: ${SAMPLING_SEED}"
echo "  自动启动服务: ${START_SERVER}"
echo "  跳过功率设置: ${SKIP_SET_POWER}"
echo ""

if [ "${START_SERVER}" = false ]; then
    echo "注意: 当前脚本默认不会自动启动 vLLM 服务。"
    echo "请先在另一个终端执行: bash start_vllm_server.sh"
    echo ""
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
    python run_decode_modeling.py
    --power "${POWER_LIMIT}"
    --batch-sizes "${BATCH_SIZES}"
    --output-lengths "${OUTPUT_LENGTHS}"
    --repeats "${REPEATS}"
    --output-dir "${OUTPUT_DIR}"
    --model-path "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --base-url "${BASE_URL}"
    --prompt-token-count "${PROMPT_TOKEN_COUNT}"
    --inter-batch-sec "${INTER_BATCH_SEC}"
    --idle-baseline-sec "${IDLE_BASELINE_SEC}"
    --time-padding-ms "${TIME_PADDING_MS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --queue-seed "${QUEUE_SEED}"
    --sampling-seed "${SAMPLING_SEED}"
)

if [ -n "${SUDO_PASSWORD}" ]; then
    CMD+=(--sudo-password "${SUDO_PASSWORD}")
fi

if [ "${SKIP_SET_POWER}" = true ]; then
    CMD+=(--skip-set-power)
fi

if [ "${START_SERVER}" = true ]; then
    CMD+=(--start-server)
fi

if [ "${ENABLE_CHUNKED_PREFILL}" = true ]; then
    CMD+=(--enable-chunked-prefill)
fi

echo "步骤1: 运行解码阶段建模实验..."
"${CMD[@]}"

if [ $? -ne 0 ]; then
    echo "错误: 实验运行失败"
    exit 1
fi

echo ""
echo "步骤2: 分析实验结果..."
python analyze_decode_modeling.py \
    --result-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/images"

if [ $? -ne 0 ]; then
    echo "警告: 结果分析失败，但实验数据已保存"
else
    echo ""
    echo "分析完成！"
fi

echo ""
echo "=========================================="
echo "全部完成！"
echo "=========================================="
echo ""
echo "实验数据目录: ${OUTPUT_DIR}"
echo "图表目录: ${OUTPUT_DIR}/images"
echo ""
