#!/bin/bash

# RTX 4080 功率档位列表 (实际最高功耗约230W，只测试150W和200W)
POWERS=(200 150)
# 并发度列表 (按idea.md要求：8, 16, 32
CONCURRENCIES=(8 32 64 128)
# 每个实验重复次数
REPEAT=3
# 请求数量
REQUEST_COUNT=50
# 最大生成token数量
MAX_TOKENS=100
# 负载类型
LOAD_TYPE="mixed"
# 结果保存目录
OUTPUT_DIR="results"
# 模型路径 (可选，不设置则使用默认值)
MODEL_PATH=""
# 是否跳过功率设置 (设置为true则手动设置功率后运行)
SKIP_SET_POWER=false
# sudo密码 (用于自动设置功率限制，不设置则手动输入)
SUDO_PASSWORD="123456"

echo "=========================================="
echo "  LLM功率控制实验 - RTX 4080"
echo "=========================================="
echo "测试功率: ${POWERS[*]}"
echo "测试并发度: ${CONCURRENCIES[*]}"
echo "重复次数: $REPEAT"
echo "请求数量: $REQUEST_COUNT"
echo "最大生成Token: $MAX_TOKENS"
echo "负载类型: $LOAD_TYPE"
echo "结果目录: $OUTPUT_DIR"
echo "跳过功率设置: $SKIP_SET_POWER"
if [ -n "$MODEL_PATH" ]; then
    echo "模型路径: $MODEL_PATH"
fi
if [ -n "$SUDO_PASSWORD" ]; then
    echo "自动sudo密码: 已配置"
fi
echo "=========================================="

mkdir -p $OUTPUT_DIR

# 显示GPU功率信息
echo ""
echo "检测GPU功率配置..."
python run_experiment.py --power 240 --show-power-info 2>/dev/null || echo "无法获取GPU信息"

# 每个实验重复多次
for repeat in $(seq 1 $REPEAT); do
    echo ""
    echo "=========================================="
    echo "  第 $repeat 轮实验"
    echo "=========================================="
    for power in "${POWERS[@]}"; do
        for concurrency in "${CONCURRENCIES[@]}"; do
            echo ""
            echo "----------------------------------------"
            echo "配置: 功率 ${power}W, 并发度 ${concurrency}, 第 ${repeat} 次"
            echo "----------------------------------------"

            # 构建命令参数
            CMD_ARGS="--power $power --concurrency $concurrency --count $REQUEST_COUNT --load-type $LOAD_TYPE --output-dir $OUTPUT_DIR --max-tokens $MAX_TOKENS"

            if [ "$SKIP_SET_POWER" = true ]; then
                CMD_ARGS="$CMD_ARGS --skip-set-power"
            fi

            if [ -n "$MODEL_PATH" ]; then
                CMD_ARGS="$CMD_ARGS --model-path $MODEL_PATH"
            fi

            if [ -n "$SUDO_PASSWORD" ]; then
                CMD_ARGS="$CMD_ARGS --sudo-password $SUDO_PASSWORD"
            fi

            # 运行实验
            python run_experiment.py $CMD_ARGS

            # 检查实验是否成功
            if [ $? -ne 0 ]; then
                echo "实验失败，跳过当前配置"
                continue
            fi

            # 冷却GPU
            echo ""
            echo "等待GPU冷却90秒..."
            sleep 90
        done
    done
done

echo ""
echo "=========================================="
echo "  所有固定功率实验完成！"
echo "=========================================="
echo "结果已保存到 $OUTPUT_DIR 目录"
echo "接下来可以运行: python analyze_results.py"
echo ""
