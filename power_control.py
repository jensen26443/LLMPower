import os
import re
import subprocess
import threading
import time


class SudoKeepAlive:
    """维持 sudo ticket，避免长实验中反复输入密码。"""

    def __init__(self, interval_sec=60.0):
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = None

    def start(self, sudo_password=None):
        if not prime_sudo_credentials(sudo_password=sudo_password):
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self):
        while not self._stop_event.wait(self.interval_sec):
            refresh_sudo_credentials(non_interactive=True)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

def is_wsl2():
    """检测是否为WSL2环境"""
    return os.path.exists('/usr/lib/wsl/lib/nvidia-smi')

def get_nvidia_smi_path():
    """获取nvidia-smi路径，自动适配WSL2和服务器环境"""
    if is_wsl2():
        return '/usr/lib/wsl/lib/nvidia-smi'
    return 'nvidia-smi'

def get_power_cap(device_index=0):
    """获取当前GPU功率限制，单位W，使用nvidia-smi命令行兼容性更好"""
    try:
        nvidia_smi = get_nvidia_smi_path()
        result = subprocess.run(
            [nvidia_smi, "-i", str(device_index), "-q", "-d", "POWER"],
            check=True, capture_output=True, text=True
        )
        match = re.search(r"Power Limit\s+:\s+(\d+\.?\d*)\s+W", result.stdout)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception as e:
        print(f"获取功率限制失败: {e}")
        return 0.0

def get_default_power_limit(device_index=0):
    """获取GPU默认功率限制，单位W"""
    try:
        nvidia_smi = get_nvidia_smi_path()
        result = subprocess.run(
            [nvidia_smi, "-i", str(device_index), "-q", "-d", "POWER"],
            check=True, capture_output=True, text=True
        )
        match = re.search(r"Default Power Limit\s+:\s+(\d+\.?\d*)\s+W", result.stdout)
        if match:
            return float(match.group(1))
        return None
    except Exception as e:
        print(f"获取默认功率限制失败: {e}")
        return None

def get_max_power_limit(device_index=0):
    """获取GPU最大功率限制，单位W"""
    try:
        nvidia_smi = get_nvidia_smi_path()
        result = subprocess.run(
            [nvidia_smi, "-i", str(device_index), "-q", "-d", "POWER"],
            check=True, capture_output=True, text=True
        )
        match = re.search(r"Max Power Limit\s+:\s+(\d+\.?\d*)\s+W", result.stdout)
        if match:
            return float(match.group(1))
        return None
    except Exception as e:
        print(f"获取最大功率限制失败: {e}")
        return None

def _run_sudo_command(args, sudo_password=None, non_interactive=False):
    cmd = ["sudo"]
    if non_interactive:
        cmd.append("-n")
    elif sudo_password:
        cmd.append("-S")
    cmd.extend(args)
    kwargs = {
        "check": True,
        "capture_output": True,
        "text": True,
    }
    if sudo_password and not non_interactive:
        kwargs["input"] = sudo_password + "\n"
    return subprocess.run(cmd, **kwargs)


def prime_sudo_credentials(sudo_password=None):
    """在实验开始前建立 sudo ticket，只需交互一次。"""
    try:
        _run_sudo_command(["-v"], sudo_password=sudo_password)
        return True
    except subprocess.CalledProcessError as e:
        print(f"初始化 sudo 凭证失败: {e.stderr}")
        return False


def refresh_sudo_credentials(non_interactive=True):
    """刷新 sudo ticket，失败时返回 False，由调用方决定是否处理。"""
    try:
        _run_sudo_command(["-v"], non_interactive=non_interactive)
        return True
    except subprocess.CalledProcessError:
        return False


def set_power_cap(watts, device_index=0, sudo_password=None):
    """设置GPU功率限制，需要sudo权限，单位W

    自动适配WSL2和服务器环境：
    - WSL2: 使用sudo密码
    - 服务器: 使用系统sudo配置（免密或需要手动输入）
    - 提供sudo_password时，所有环境都使用密码输入
    """
    try:
        nvidia_smi = get_nvidia_smi_path()
        _run_sudo_command(
            [nvidia_smi, "-i", str(device_index), "-pl", str(watts)],
            sudo_password=sudo_password,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"设置功率限制失败: {e.stderr}")
        return False

def get_current_power(device_index=0):
    """获取当前GPU实时功率，单位W，使用nvidia-smi命令行"""
    try:
        nvidia_smi = get_nvidia_smi_path()
        result = subprocess.run(
            [nvidia_smi, "-i", str(device_index), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"获取实时功率失败: {e}")
        return 0.0

def get_gpu_name(device_index=0):
    """获取GPU型号名称"""
    try:
        nvidia_smi = get_nvidia_smi_path()
        result = subprocess.run(
            [nvidia_smi, "-i", str(device_index), "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"获取GPU型号失败: {e}")
        return None

def suggest_power_range(device_index=0):
    """根据GPU型号建议功率范围"""
    gpu_name = get_gpu_name(device_index)
    max_power = get_max_power_limit(device_index)
    default_power = get_default_power_limit(device_index)

    if gpu_name and "4080" in gpu_name:
        # RTX 4080
        return {
            "min": 150,
            "max": 350 if max_power is None else min(350, int(max_power)),
            "default": 300 if default_power is None else int(default_power),
            "steps": [350, 300, 250, 200, 150]
        }
    elif gpu_name and "5070" in gpu_name:
        # RTX 5070Ti
        return {
            "min": 80,
            "max": 240 if max_power is None else min(240, int(max_power)),
            "default": 140 if default_power is None else int(default_power),
            "steps": [80, 100, 120, 140, 160, 180, 200, 220]
        }
    else:
        # 通用配置
        min_p = 100
        max_p = 300 if max_power is None else int(max_power)
        default_p = 200 if default_power is None else int(default_power)
        step = 30
        steps = list(range(min_p, max_p + 1, step))
        return {
            "min": min_p,
            "max": max_p,
            "default": default_p,
            "steps": steps
        }

if __name__ == "__main__":
    print(f"环境: {'WSL2' if is_wsl2() else 'Linux服务器'}")
    print(f"GPU型号: {get_gpu_name() or '未知'}")
    print(f"当前功率限制: {get_power_cap()}W")
    print(f"当前实时功率: {get_current_power()}W")

    default_power = get_default_power_limit()
    if default_power:
        print(f"默认功率限制: {default_power}W")

    max_power = get_max_power_limit()
    if max_power:
        print(f"最大功率限制: {max_power}W")

    power_range = suggest_power_range()
    print(f"\n建议功率范围: {power_range['min']}W - {power_range['max']}W")
    print(f"建议功率档位: {power_range['steps']}")
