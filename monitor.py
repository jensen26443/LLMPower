import subprocess
import time
import threading
import csv
from datetime import datetime
from typing import List, Dict
import re

try:
    import pynvml
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False

class PowerMonitor:
    """后台功率采样器。

    实验脚本会在每个 batch 的推理窗口前启动监测，并在推理结束后停止。
    采样结果用于计算真实平均功率和能耗，而不是直接用 power cap 估算。
    """

    def __init__(self, device_index: int = 0, sample_interval: float = 0.1):
        self.device_index = device_index
        self.sample_interval = sample_interval
        self.running = False
        self.power_data: List[Dict] = []
        self.thread = None
        self._backend = "pynvml" if HAS_PYNVML else "nvidia-smi"
        self._nvml_initialized = False
        self._nvml_handle = None

    def _init_backend(self):
        if self._backend != "pynvml" or self._nvml_initialized:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._nvml_initialized = True
        except Exception as e:
            print(f"pynvml 初始化失败，回退 nvidia-smi: {e}")
            self._backend = "nvidia-smi"

    def _shutdown_backend(self):
        if self._backend == "pynvml" and self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
            self._nvml_handle = None

    def _get_gpu_stats(self):
        """获取GPU当前统计信息。优先使用 pynvml，失败回退 nvidia-smi。"""
        if self._backend == "pynvml":
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                temperature = pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(
                    self._nvml_handle,
                    pynvml.NVML_CLOCK_GRAPHICS,
                )
                memory_clock = pynvml.nvmlDeviceGetClockInfo(
                    self._nvml_handle,
                    pynvml.NVML_CLOCK_MEM,
                )
                return (
                    power_mw / 1000.0,
                    memory.used / (1024 ** 3),
                    int(temperature),
                    float(graphics_clock),
                    float(memory_clock),
                )
            except Exception as e:
                print(f"pynvml 读取失败，回退 nvidia-smi: {e}")
                self._backend = "nvidia-smi"

        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.device_index),
                 "--query-gpu=power.draw,memory.used,temperature.gpu,clocks.current.graphics,clocks.current.memory",
                 "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True
            )
            parts = result.stdout.strip().split(', ')
            power = float(parts[0])
            memory_used = float(parts[1]) / 1024  # 转换为GB
            temperature = int(parts[2])
            graphics_clock = float(parts[3])
            memory_clock = float(parts[4])
            return power, memory_used, temperature, graphics_clock, memory_clock
        except Exception as e:
            print(f"获取GPU状态失败: {e}")
            return 0.0, 0.0, 0, 0.0, 0.0

    def _monitor_loop(self):
        while self.running:
            power, memory_used, temperature, graphics_clock, memory_clock = self._get_gpu_stats()
            # 在采样数据获取后打时间戳，减少命令调用耗时带来的时间偏移
            timestamp = time.time()

            # 每条采样保留功率、显存、温度和频率，后续分析可同时解释能耗和硬件状态。
            self.power_data.append({
                "timestamp": timestamp,
                "power_w": power,
                "memory_gb": memory_used,
                "temperature_c": temperature,
                "graphics_clock_mhz": graphics_clock,
                "memory_clock_mhz": memory_clock,
            })
            time.sleep(self.sample_interval)

    def start(self):
        """开始监测"""
        self._init_backend()
        self.running = True
        self.power_data = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> List[Dict]:
        """停止监测，返回监测数据"""
        self.running = False
        if self.thread:
            self.thread.join()
        self._shutdown_backend()
        return self.power_data

    def calculate_total_energy(self) -> float:
        """计算总能耗，单位焦耳"""
        if len(self.power_data) < 2:
            return 0.0

        total_energy = 0.0
        for i in range(1, len(self.power_data)):
            # 使用相邻采样点的梯形积分近似能耗：J = W * s。
            dt = self.power_data[i]["timestamp"] - self.power_data[i-1]["timestamp"]
            avg_power = (self.power_data[i]["power_w"] + self.power_data[i-1]["power_w"]) / 2
            total_energy += avg_power * dt  # J = W * s

        return total_energy

    def save_to_csv(self, filename: str):
        """保存监测数据到CSV"""
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "power_w",
                    "memory_gb",
                    "temperature_c",
                    "graphics_clock_mhz",
                    "memory_clock_mhz",
                ],
            )
            writer.writeheader()
            writer.writerows(self.power_data)

if __name__ == "__main__":
    # 测试监测模块
    monitor = PowerMonitor()
    monitor.start()
    print("开始监测3秒...")
    time.sleep(3)
    data = monitor.stop()
    energy = monitor.calculate_total_energy()
    print(f"监测到{len(data)}条数据，总能耗: {energy:.2f}J")
    monitor.save_to_csv("test_monitor.csv")
    print("数据已保存到test_monitor.csv")
