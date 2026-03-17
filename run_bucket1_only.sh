#!/bin/bash
# 单独运行bucket1策略的脚本

SUDO_PASSWORD="123456"
OUTPUT_DIR="./results1/bucket1_temp"
REPEATS_PER_PROMPT=10
FULL_REPEATS=1

echo "=== 开始运行bucket1策略 ==="

# 确保目录存在
mkdir -p $OUTPUT_DIR/data $OUTPUT_DIR/img

# 先设置功率到165W（第一个需要的功率）
echo "设置功率到165W..."
echo $SUDO_PASSWORD | sudo -S nvidia-smi -i 0 -pl 165
sleep 5

# 运行实验，使用 --skip-set-power，我们手动在需要时切换功率
# 注意：由于bucket1需要不同功率，我们还是让代码尝试设置
echo "运行实验..."
python3 run_strategy_evaluation.py \
    --output-dir $OUTPUT_DIR \
    --only-strategy bucket1 \
    --repeats-per-prompt $REPEATS_PER_PROMPT \
    --full-repeats $FULL_REPEATS \
    --sudo-password $SUDO_PASSWORD

echo "=== 完成 ==="
