#!/bin/bash

# idea2.md 静态功率封顶总体对比实验
# 使用不同数量的查询而不是并发度

# RTX 4080 功率档位列表
POWERS=(350 300 250 200 175 150)
# 查询数量列表 (按idea2.md要求：8, 32, 64, 128)
QUERY_COUNTS=(8 32 64 128)
# 固定并发度为1
CONCURRENCY=1
# 每个实验重复次数
REPEAT=3
# 最大生成token数量
MAX_TOKENS=100
# 负载类型
LOAD_TYPE="mixed"
# 结果保存目录
OUTPUT_DIR="./results0/data"
# 模型路径 (可选，不设置则使用默认值)
MODEL_PATH=""
# 是否跳过功率设置 (设置为true则手动设置功率后运行)
SKIP_SET_POWER=false
# sudo密码 (用于自动设置功率限制，不设置则手动输入)
SUDO_PASSWORD="123456"

echo "=========================================="
echo "  静态功率封顶总体对比实验"
echo "  idea2.md - Query Count Variation"
echo "=========================================="
echo "测试功率: ${POWERS[*]}"
echo "测试查询数量: ${QUERY_COUNTS[*]}"
echo "固定并发度: $CONCURRENCY"
echo "重复次数: $REPEAT"
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
        for query_count in "${QUERY_COUNTS[@]}"; do
            echo ""
            echo "----------------------------------------"
            echo "配置: 功率 ${power}W, 查询数量 ${query_count}, 第 ${repeat} 次"
            echo "----------------------------------------"

            # 构建命令参数
            CMD_ARGS="--power $power --concurrency $CONCURRENCY --count $query_count --load-type $LOAD_TYPE --output-dir $OUTPUT_DIR --max-tokens $MAX_TOKENS"

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
echo "  所有查询数量实验完成！"
echo "=========================================="
echo "数据已保存到 $OUTPUT_DIR 目录"
echo "接下来可以运行: python analyze_gpu_power.py"
echo "图表将保存到 ./results0/img/ 目录"
echo ""
