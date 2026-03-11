import subprocess
import time
import threading
import csv
from datetime import datetime
from typing import List, Dict
import re

class PowerMonitor:
    def __init__(self, device_index: int = 0, sample_interval: float = 0.1):
        self.device_index = device_index
        self.sample_interval = sample_interval
        self.running = False
        self.power_data: List[Dict] = []
        self.thread = None

    def _get_gpu_stats(self):
        """获取GPU当前统计信息，使用nvidia-smi命令行"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.device_index),
                 "--query-gpu=power.draw,memory.used,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True
            )
            parts = result.stdout.strip().split(', ')
            power = float(parts[0])
            memory_used = float(parts[1]) / 1024  # 转换为GB
            temperature = int(parts[2])
            return power, memory_used, temperature
        except Exception as e:
            print(f"获取GPU状态失败: {e}")
            return 0.0, 0.0, 0

    def _monitor_loop(self):
        while self.running:
            timestamp = time.time()
            power, memory_used, temperature = self._get_gpu_stats()

            self.power_data.append({
                "timestamp": timestamp,
                "power_w": power,
                "memory_gb": memory_used,
                "temperature_c": temperature
            })
            time.sleep(self.sample_interval)

    def start(self):
        """开始监测"""
        self.running = True
        self.power_data = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> List[Dict]:
        """停止监测，返回监测数据"""
        self.running = False
        if self.thread:
            self.thread.join()
        return self.power_data

    def calculate_total_energy(self) -> float:
        """计算总能耗，单位焦耳"""
        if len(self.power_data) < 2:
            return 0.0

        total_energy = 0.0
        for i in range(1, len(self.power_data)):
            dt = self.power_data[i]["timestamp"] - self.power_data[i-1]["timestamp"]
            avg_power = (self.power_data[i]["power_w"] + self.power_data[i-1]["power_w"]) / 2
            total_energy += avg_power * dt  # J = W * s

        return total_energy

    def save_to_csv(self, filename: str):
        """保存监测数据到CSV"""
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "power_w", "memory_gb", "temperature_c"])
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
