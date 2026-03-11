import vllm
import pynvml
import pandas as pd
import matplotlib.pyplot as plt

print(f"vLLM版本: {vllm.__version__}")
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
print(f"GPU数量: {device_count}")
for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(handle)
    print(f"GPU {i}: {name}")
pynvml.nvmlShutdown()
print("环境测试通过!")