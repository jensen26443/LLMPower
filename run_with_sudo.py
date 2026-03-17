#!/usr/bin/env python3
"""
使用正确的sudo密码方式运行bucket1实验
"""
import subprocess
import sys
import os

# 修改power_control.py临时使用环境变量中的密码
power_control_code = '''
import subprocess
import re
import os

def is_wsl2():
    return os.path.exists('/usr/lib/wsl/lib/nvidia-smi')

def get_nvidia_smi_path():
    if is_wsl2():
        return '/usr/lib/wsl/lib/nvidia-smi'
    return 'nvidia-smi'

def get_power_cap(device_index=0):
    try:
        nvidia_smi = get_nvidia_smi_path()
        result = subprocess.run(
            [nvidia_smi, "-i", str(device_index), "-q", "-d", "POWER"],
            check=True, capture_output=True, text=True
        )
        match = re.search(r"Power Limit\\s+:\\s+(\\d+\\.?\\d*)\\s+W", result.stdout)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception as e:
        print(f"获取功率限制失败: {e}")
        return 0.0

def set_power_cap(watts, device_index=0, sudo_password=None):
    try:
        nvidia_smi = get_nvidia_smi_path()
        cmd = ["sudo", "-S", nvidia_smi, "-i", str(device_index), "-pl", str(watts)]

        if sudo_password:
            result = subprocess.run(
                cmd,
                input=sudo_password + "\\n",
                text=True,
                capture_output=True,
                check=True
            )
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except Exception as e:
        print(f"设置功率限制失败: {e}")
        return False
'''

# 临时修改power_control.py
with open('power_control.py', 'r') as f:
    original_code = f.read()

with open('power_control.py', 'w') as f:
    f.write(power_control_code)

try:
    # 现在运行实验
    from run_strategy_evaluation import run_strategy_evaluation

    print("开始运行bucket1实验...")
    result = run_strategy_evaluation(
        output_dir="./results1/bucket1_temp",
        only_strategy="bucket1",
        repeats_per_prompt=10,
        full_repeats=1,
        sudo_password="123456",
        skip_set_power=False
    )
    print("实验完成!")

finally:
    # 恢复原始文件
    with open('power_control.py', 'w') as f:
        f.write(original_code)
    print("已恢复power_control.py")
