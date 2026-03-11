#!/bin/bash
# 打包代码到服务器（排除大文件）

set -e

PACKAGE_NAME="vllm_experiment_code_$(date +%Y%m%d).tar.gz"

echo "=== LLM功率控制实验 - 代码打包工具 ==="
echo ""

# 检查文件
echo "检查必要文件..."
required_files=(
    "run_experiment.py"
    "llm_inference.py"
    "power_control.py"
    "monitor.py"
    "load_generator.py"
    "analyze_results.py"
    "requirements.txt"
    "run_all_fixed_experiments.sh"
    "CLAUDE.md"
)

missing_count=0
for f in "${required_files[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  ⚠️  缺失: $f"
        missing_count=$((missing_count + 1))
    else
        echo "  ✓ $f"
    fi
done

if [ $missing_count -gt 0 ]; then
    echo ""
    echo "警告: 缺失 $missing_count 个文件"
    read -p "继续打包? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "打包代码文件..."
tar -czf $PACKAGE_NAME \
    --exclude='models/*' \
    --exclude='results/*' \
    --exclude='error/*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.claude' \
    --exclude='.git' \
    --exclude='*.tar.gz' \
    *.py *.sh *.txt *.md \
    docs/ summary/ 2>/dev/null || true

echo ""
echo "✓ 已创建: $PACKAGE_NAME"
echo "  大小: $(du -h $PACKAGE_NAME | cut -f1)"
echo ""
echo "=== 下一步 ==="
echo "1. 传输到服务器:"
echo "   scp $PACKAGE_NAME user@server-ip:/path/to/destination/"
echo ""
echo "2. 在服务器上解压:"
echo "   ssh user@server-ip"
echo "   cd /path/to/destination"
echo "   tar -xzf $PACKAGE_NAME"
echo ""
echo "3. 查看迁移指南:"
echo "   cat docs/migration_guide.md"
echo ""
