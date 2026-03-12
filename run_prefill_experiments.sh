#!/bin/bash
#
# 预填充阶段离线建模实验 - 批量运行脚本
#

# 配置参数
SUDO_PASSWORD="123456"
POWER_LIMIT=350
REPEATS=20
INPUT_LENGTHS="dense"  # dense=密集1-3000(100点), sparse=稀疏, 或自定义列表
OUTPUT_DIR="results/prefill_modeling"
MODEL_PATH="./Qwen2.5-7B-Instruct-AWQ"

echo "=========================================="
echo "预填充阶段离线建模实验"
echo "=========================================="
echo ""
echo "配置参数:"
echo "  功率限制: ${POWER_LIMIT}W"
echo "  重复次数: ${REPEATS}"
echo "  输入长度: ${INPUT_LENGTHS}"
if [ "${INPUT_LENGTHS}" = "dense" ]; then
    echo "    (密集模式: 1-3000 tokens, 约100个采样点)"
elif [ "${INPUT_LENGTHS}" = "sparse" ]; then
    echo "    (稀疏模式: 2的幂次)"
fi
echo "  输出目录: ${OUTPUT_DIR}"
echo "  模型路径: ${MODEL_PATH}"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 步骤1: 运行预填充建模实验
echo "步骤1: 运行预填充建模实验..."
python run_prefill_modeling.py \
    --power "${POWER_LIMIT}" \
    --input-lengths "${INPUT_LENGTHS}" \
    --repeats "${REPEATS}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --sudo-password "${SUDO_PASSWORD}"

if [ $? -ne 0 ]; then
    echo "错误: 实验运行失败"
    exit 1
fi

echo ""
echo "实验运行完成！"
echo ""

# 步骤2: 分析结果
echo "步骤2: 分析实验结果..."
python analyze_prefill_modeling.py \
    --input-dir "${OUTPUT_DIR}"

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
