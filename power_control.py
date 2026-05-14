import os
import re
import subprocess
import threading
import time
from typing import Dict, List, Optional


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
    """获取当前GPU功率限制，单位W，使用 nvidia-smi query 避免误匹配默认功率。"""
    try:
        nvidia_smi = get_nvidia_smi_path()
        result = subprocess.run(
            [
                nvidia_smi,
                "-i",
                str(device_index),
                "--query-gpu=power.limit",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True
        )
        match = re.search(r"(\d+\.?\d*)", result.stdout)
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


def parse_supported_clock_pairs(raw_output: str) -> List[Dict[str, int]]:
    """解析 `nvidia-smi --query-supported-clocks=memory,graphics` 输出。"""
    pairs: List[Dict[str, int]] = []
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("memory"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            memory_mhz = int(float(parts[0]))
            graphics_mhz = int(float(parts[1]))
        except ValueError:
            continue
        pairs.append({
            "memory_mhz": memory_mhz,
            "graphics_mhz": graphics_mhz,
        })
    return pairs


def sample_clock_profile_pairs(pairs: List[Dict[str, int]],
                               count: int = 6,
                               min_sm_mhz: int = 1000,
                               min_mem_mhz: int = 5000) -> List[Dict[str, int]]:
    """按最低 `SM/MEM` 频率阈值过滤后，再按分位点抽样受支持的组合。"""
    if not pairs:
        return []
    ordered = sorted(
        {
            (int(item["memory_mhz"]), int(item["graphics_mhz"]))
            for item in pairs
            if int(item["graphics_mhz"]) >= int(min_sm_mhz)
            and int(item["memory_mhz"]) >= int(min_mem_mhz)
        }
    )
    if not ordered:
        return []
    if count >= len(ordered):
        return [
            {"memory_mhz": memory_mhz, "graphics_mhz": graphics_mhz}
            for memory_mhz, graphics_mhz in ordered
        ]

    selected_indices = set()
    max_index = len(ordered) - 1
    for bucket_index in range(count):
        ratio = bucket_index / max(1, count - 1)
        selected_indices.add(int(round(ratio * max_index)))
    sampled = [ordered[index] for index in sorted(selected_indices)]
    return [
        {"memory_mhz": memory_mhz, "graphics_mhz": graphics_mhz}
        for memory_mhz, graphics_mhz in sampled
    ]


def query_supported_clock_pairs(device_index=0) -> List[Dict[str, int]]:
    """查询 GPU 支持的 `(memory, graphics)` 频率组合。"""
    try:
        nvidia_smi = get_nvidia_smi_path()
        result = subprocess.run(
            [
                nvidia_smi,
                "-i",
                str(device_index),
                "--query-supported-clocks=memory,graphics",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return parse_supported_clock_pairs(result.stdout)
    except Exception as e:
        print(f"查询支持频率失败: {e}")
        return []


def probe_clock_capabilities(device_index=0,
                             sample_count: int = 6,
                             min_sm_mhz: int = 1000,
                             min_mem_mhz: int = 5000) -> Dict:
    """探测 GPU 锁频能力和抽样频率档位。"""
    support_pairs = query_supported_clock_pairs(device_index=device_index)
    sampled_pairs = sample_clock_profile_pairs(
        support_pairs,
        count=sample_count,
        min_sm_mhz=min_sm_mhz,
        min_mem_mhz=min_mem_mhz,
    )
    return {
        "device_index": int(device_index),
        "supports_gpu_clock_lock": bool(support_pairs),
        "supports_memory_clock_lock": bool(support_pairs),
        "supported_clock_pairs": support_pairs,
        "sampled_clock_pairs": sampled_pairs,
        "min_sm_mhz_filter": int(min_sm_mhz),
        "min_mem_mhz_filter": int(min_mem_mhz),
        "sample_count": int(sample_count),
    }


def build_hardware_profile_commands(power_w: Optional[int] = None,
                                    sm_mhz: Optional[int] = None,
                                    mem_mhz: Optional[int] = None,
                                    device_index: int = 0) -> List[List[str]]:
    nvidia_smi = get_nvidia_smi_path()
    commands: List[List[str]] = []
    if sm_mhz is not None:
        commands.append([nvidia_smi, "-i", str(device_index), "-lgc", f"{int(sm_mhz)},{int(sm_mhz)}"])
    if mem_mhz is not None:
        commands.append([nvidia_smi, "-i", str(device_index), "-lmc", f"{int(mem_mhz)},{int(mem_mhz)}"])
    if power_w is not None:
        commands.append([nvidia_smi, "-i", str(device_index), "-pl", str(int(power_w))])
    return commands


def build_reset_clock_commands(device_index: int = 0) -> List[List[str]]:
    nvidia_smi = get_nvidia_smi_path()
    return [
        [nvidia_smi, "-i", str(device_index), "-rgc"],
        [nvidia_smi, "-i", str(device_index), "-rmc"],
    ]

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


def set_gpu_clocks(sm_mhz: Optional[int] = None,
                   mem_mhz: Optional[int] = None,
                   device_index: int = 0,
                   sudo_password: Optional[str] = None) -> bool:
    """设置 GPU SM / 显存频率。"""
    commands = build_hardware_profile_commands(
        power_w=None,
        sm_mhz=sm_mhz,
        mem_mhz=mem_mhz,
        device_index=device_index,
    )
    try:
        for command in commands:
            _run_sudo_command(command, sudo_password=sudo_password)
        return True
    except subprocess.CalledProcessError as e:
        print(f"设置 GPU 频率失败: {e.stderr}")
        return False


def reset_gpu_clocks(device_index: int = 0, sudo_password: Optional[str] = None) -> bool:
    """恢复 GPU SM / 显存频率。"""
    try:
        for command in build_reset_clock_commands(device_index=device_index):
            _run_sudo_command(command, sudo_password=sudo_password)
        return True
    except subprocess.CalledProcessError as e:
        print(f"恢复 GPU 频率失败: {e.stderr}")
        return False


def apply_hardware_profile(power_w: Optional[int] = None,
                           sm_mhz: Optional[int] = None,
                           mem_mhz: Optional[int] = None,
                           device_index: int = 0,
                           sudo_password: Optional[str] = None) -> bool:
    """统一设置频率和功率上限。先锁频，再设功率。"""
    try:
        for command in build_hardware_profile_commands(
            power_w=power_w,
            sm_mhz=sm_mhz,
            mem_mhz=mem_mhz,
            device_index=device_index,
        ):
            _run_sudo_command(command, sudo_password=sudo_password)
        return True
    except subprocess.CalledProcessError as e:
        print(f"应用硬件 profile 失败: {e.stderr}")
        return False


def reset_hardware_profile(default_power_w: Optional[int] = None,
                           device_index: int = 0,
                           sudo_password: Optional[str] = None) -> bool:
    """恢复默认频率和功率上限。"""
    ok = reset_gpu_clocks(device_index=device_index, sudo_password=sudo_password)
    if default_power_w is not None:
        ok = set_power_cap(default_power_w, device_index=device_index, sudo_password=sudo_password) and ok
    return ok

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
