#!/usr/bin/env python
"""
自动适配代码到服务器环境
运行在服务器端
"""
import shutil
import os
import subprocess
import sys

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(text):
    print(f"\n► {text}")

def print_success(text):
    print(f"  ✓ {text}")

def print_warning(text):
    print(f"  ⚠️  {text}")

def print_error(text):
    print(f"  ✗ {text}")

def check_nvidia_smi():
    """检查nvidia-smi是否可用"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--help"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except:
        return False

def get_gpu_memory():
    """获取GPU显存"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10
        )
        mem_mb = float(result.stdout.strip().split('\n')[0])
        return mem_mb / 1024
    except Exception as e:
        print_warning(f"无法获取显存: {e}")
        return None

def get_power_limit():
    """获取当前功率限制"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-d", "POWER"],
            capture_output=True, text=True, check=True, timeout=10
        )
        import re
        match = re.search(r"Power Limit\s+:\s+(\d+\.?\d*)\s+W", result.stdout)
        if match:
            return float(match.group(1))
    except Exception as e:
        print_warning(f"无法获取功率限制: {e}")
    return None

def test_sudo_nvidia_smi():
    """测试sudo nvidia-smi是否可用"""
    try:
        result = subprocess.run(
            ["sudo", "-n", "nvidia-smi", "-pl", "0"],
            capture_output=True, text=True, timeout=10
        )
        # 如果返回0或关于无效功率的错误，说明sudo可用
        if result.returncode == 0 or "Invalid power limit" in result.stderr:
            return True
        return False
    except:
        return False

def adapt_power_control():
    """适配功率控制模块"""
    print_step("适配 power_control.py")

    if not os.path.exists('power_control.py'):
        print_error("找不到 power_control.py")
        return False

    if os.path.exists('power_control.py.wsl2'):
        print_success("已备份过，跳过")
        return True

    # 备份
    shutil.copy('power_control.py', 'power_control.py.wsl2')
    print_success("已备份到 power_control.py.wsl2")

    # 读取并修改
    with open('power_control.py', 'r') as f:
        content = f.read()

    # 替换nvidia-smi路径
    original_content = content
    content = content.replace('/usr/lib/wsl/lib/nvidia-smi', 'nvidia-smi')

    # 简化sudo密码处理
    if 'sudo_password="1234"' in content:
        content = content.replace(
            'def set_power_cap(watts, device_index=0, sudo_password="1234"):',
            'def set_power_cap(watts, device_index=0):'
        )
        # 简化subprocess调用
        import re
        content = re.sub(
            r'\["sudo", "-S", "/usr/lib/wsl/lib/nvidia-smi"(.*?)\],\s*input=sudo_password.*?text=True,',
            r'["sudo", "nvidia-smi"\1],',
            content,
            flags=re.DOTALL
        )
        # 再次确保nvidia-smi路径正确
        content = content.replace('/usr/lib/wsl/lib/nvidia-smi', 'nvidia-smi')

    if content != original_content:
        with open('power_control.py', 'w') as f:
            f.write(content)
        print_success("已修改 power_control.py")
    else:
        print_success("无需修改")

    return True

def suggest_gpu_config():
    """根据显存建议配置"""
    print_step("检测GPU配置")

    gpu_mem_gb = get_gpu_memory()
    if gpu_mem_gb:
        print(f"  GPU显存: {gpu_mem_gb:.1f}GB")

        if gpu_mem_gb >= 24:
            suggested = 0.95
            reason = "大显存，可以充分利用"
        elif gpu_mem_gb >= 16:
            suggested = 0.90
            reason = "中等显存，留有一定余量"
        elif gpu_mem_gb >= 12:
            suggested = 0.85
            reason = "12GB显存，与原配置相同"
        else:
            suggested = 0.70
            reason = "显存较小，需要降低占用"

        print(f"  建议 gpu_memory_utilization = {suggested} ({reason})")
        return suggested
    return None

def update_llm_inference(suggested_utilization):
    """更新llm_inference.py的显存配置"""
    if suggested_utilization is None:
        return

    print_step("可选：更新 llm_inference.py 显存配置")

    if not os.path.exists('llm_inference.py'):
        print_warning("找不到 llm_inference.py")
        return

    with open('llm_inference.py', 'r') as f:
        content = f.read()

    import re
    match = re.search(r'gpu_memory_utilization\s*=\s*([0-9.]+)', content)
    if match:
        current = float(match.group(1))
        if current != suggested_utilization:
            print(f"  当前值: {current}")
            print(f"  建议值: {suggested_utilization}")

            response = input(f"  更新为 {suggested_utilization}? (y/N): ").strip().lower()
            if response == 'y':
                content = re.sub(
                    r'gpu_memory_utilization\s*=\s*[0-9.]+',
                    f'gpu_memory_utilization={suggested_utilization}',
                    content
                )
                with open('llm_inference.py', 'w') as f:
                    f.write(content)
                print_success("已更新")
        else:
            print_success("当前值已是建议值")

def check_sudo():
    """检查sudo权限"""
    print_step("检查sudo权限")

    has_sudo = test_sudo_nvidia_smi()
    if has_sudo:
        print_success("sudo nvidia-smi 可用")
        print("  可以自动设置功率限制")
    else:
        print_warning("sudo nvidia-smi 不可用或需要密码")
        print("  建议:")
        print("    1. 手动设置功率: sudo nvidia-smi -pl <watts>")
        print("    2. 运行实验时添加 --skip-set-power 参数")

def create_test_script():
    """创建测试脚本"""
    print_step("创建测试脚本")

    script_content = '''#!/bin/bash
echo "=== 环境测试脚本 ==="
echo ""

echo "1. 检查nvidia-smi..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ nvidia-smi 可用"
    nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
else
    echo "   ✗ nvidia-smi 不可用"
fi

echo ""
echo "2. 检查Python环境..."
python --version 2>/dev/null || python3 --version

echo ""
echo "3. 检查vLLM..."
python -c "import vllm; print(f'   ✓ vLLM {vllm.__version__}')" 2>/dev/null || echo "   ✗ vLLM未安装"

echo ""
echo "4. 检查模型文件..."
if [ -d "models/Qwen2.5-7B-Instruct-GPTQ-Int4" ]; then
    echo "   ✓ 模型目录存在 (Qwen2.5-7B)"
    ls -lh models/Qwen2.5-7B-Instruct-GPTQ-Int4/ | head -5
elif [ -d "models/Qwen2-7B-Instruct-GPTQ-Int4" ]; then
    echo "   ✓ 模型目录存在 (Qwen2-7B)"
    ls -lh models/Qwen2-7B-Instruct-GPTQ-Int4/ | head -5
else
    echo "   ✗ 模型目录不存在"
    echo "   请从HuggingFace下载或从本地传输"
    echo "   推荐模型: Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
fi

echo ""
echo "测试完成！"
'''

    with open('test_server_env.sh', 'w') as f:
        f.write(script_content)

    os.chmod('test_server_env.sh', 0o755)
    print_success("已创建 test_server_env.sh")

def main():
    print_header("LLM功率控制实验 - 服务器环境适配工具")

    # 检查是否在服务器上
    if os.path.exists('/usr/lib/wsl/lib/nvidia-smi'):
        print_warning("检测到WSL2环境，此工具用于服务器")
        response = input("继续运行? (y/N): ").strip().lower()
        if response != 'y':
            return

    # 检查nvidia-smi
    print_step("检查GPU环境")
    if check_nvidia_smi():
        print_success("nvidia-smi 可用")

        power_limit = get_power_limit()
        if power_limit:
            print(f"  当前功率限制: {power_limit}W")
    else:
        print_error("nvidia-smi 不可用")
        print("  请确认服务器有GPU且驱动已安装")
        return

    # 适配功率控制
    if not adapt_power_control():
        return

    # 检查sudo
    check_sudo()

    # 建议配置
    suggested_util = suggest_gpu_config()

    # 更新llm_inference
    if suggested_util:
        update_llm_inference(suggested_util)

    # 创建测试脚本
    create_test_script()

    print_header("适配完成！")
    print("\n下一步:")
    print("  1. 运行测试: ./test_server_env.sh")
    print("  2. 确保模型已准备")
    print("  3. 运行轻量测试: python test_env.py")
    print("  4. 运行完整实验: bash run_all_fixed_experiments.sh")
    print("\n详细文档: docs/migration_guide.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
        sys.exit(1)
