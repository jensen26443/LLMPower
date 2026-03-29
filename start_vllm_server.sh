#!/bin/bash
# 启动 vLLM OpenAI 兼容 API 服务

MODEL_PATH="./Qwen2.5-7B-Instruct-AWQ"
MODEL_NAME="Qwen2.5-7B-Instruct-AWQ"
HOST="0.0.0.0"
PORT="8000"
GPU_MEM_UTIL="0.85"
MAX_NUM_BATCHED_TOKENS="2048"
MAX_NUM_SEQS="64"
ENABLE_CHUNKED_PREFILL="true"

echo "=========================================="
echo "启动 vLLM OpenAI 兼容 API 服务"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "模型名称: $MODEL_NAME"
echo "监听地址: $HOST:$PORT"
echo "GPU显存利用率: $GPU_MEM_UTIL"
echo "Chunked Prefill: $ENABLE_CHUNKED_PREFILL"
echo "max_num_batched_tokens: $MAX_NUM_BATCHED_TOKENS"
echo "max_num_seqs: $MAX_NUM_SEQS"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --quantization awq \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-chunked-prefill \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --no-enable-prefix-caching
