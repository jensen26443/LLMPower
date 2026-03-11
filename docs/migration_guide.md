# 迁移到实验室GPU服务器指南

## 前置检查

在迁移前，请确认实验室服务器的以下信息：

1. **GPU型号**：`nvidia-smi`
2. **GPU显存**：确认是否≥12GB
3. **Python版本**：建议3.10-3.12
4. **CUDA版本**：与vLLM 0.17.0兼容
5. **是否有sudo权限**：用于设置功率限制

---

## 迁移步骤

### 1. 代码迁移（排除大文件）

在本地WSL2执行：

```bash
# 创建打包脚本
cat > package_for_server.sh << 'EOF'
#!/bin/bash
# 打包代码到服务器（排除大文件）

PACKAGE_NAME="vllm_experiment_code_$(date +%Y%m%d).tar.gz"

echo "打包代码文件..."
tar -czf $PACKAGE_NAME \
    --exclude='models/*' \
    --exclude='results/*' \
    --exclude='error/*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.claude' \
    *.py *.sh *.txt *.md \
    docs/ summary/

echo "已创建: $PACKAGE_NAME"
echo "大小: $(du -h $PACKAGE_NAME | cut -f1)"
EOF

chmod +x package_for_server.sh
./package_for_server.sh
```

### 2. 传输到服务器

```bash
# 使用scp传输（替换为你的服务器地址）
scp vllm_experiment_code_*.tar.gz user@server-ip:/path/to/destination/

# 或使用rsync
rsync -avz --exclude='models/' --exclude='results/' ./ user@server-ip:/path/to/vllm/
```

### 3. 在服务器上解压

```bash
ssh user@server-ip
cd /path/to/destination/
tar -xzf vllm_experiment_code_*.tar.gz
cd vllm
```

### 4. 环境配置

```bash
# 创建conda环境
conda create -n vllm python=3.12 -y
conda activate vllm

# 安装依赖
pip install -r requirements.txt

# 验证vLLM安装
python -c "import vllm; print(f'vLLM版本: {vllm.__version__}')"
```

### 5. 模型准备（二选一）

**方案A：从HuggingFace下载（推荐）**

```bash
# 在服务器上创建models目录
mkdir -p models
cd models

# 使用huggingface-cli下载（需要先安装huggingface_hub）
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2-7B-Instruct-GPTQ-Int4',
    local_dir='./Qwen2-7B-Instruct-GPTQ-Int4',
    local_dir_use_symlinks=False
)
"
```

**方案B：从本地传输（如果网络慢）**

```bash
# 本地打包模型（约4-5GB）
cd /home/jensen/vllm/models
tar -czf Qwen2-7B-GPTQ.tar.gz Qwen2-7B-Instruct-GPTQ-Int4/

# 传输到服务器
scp Qwen2-7B-GPTQ.tar.gz user@server-ip:/path/to/vllm/models/

# 服务器解压
cd /path/to/vllm/models
tar -xzf Qwen2-7B-GPTQ.tar.gz
```

### 6. 适配功率控制模块

服务器上通常是原生Linux，需要修改`power_control.py`：

```bash
# 备份原文件
cp power_control.py power_control.py.wsl2

# 编辑power_control.py，修改set_power_cap函数
# 把/usr/lib/wsl/lib/nvidia-smi 改为 nvidia-smi
# 去掉sudo密码输入（如果服务器NOPASSWD sudo或用其他方式）
```

或者使用我提供的`power_control_server.py`（见下文）。

### 7. 调整显存占用

根据服务器显存调整`llm_inference.py`中的`gpu_memory_utilization`：

```python
# 例如：24GB显存可以设为0.95
gpu_memory_utilization=0.95
```

### 8. 测试环境

```bash
# 测试GPU
nvidia-smi

# 测试功率控制
python power_control.py

# 测试推理（轻量测试）
python test_env.py
```

---

## 需要修改的关键文件

### power_control.py - 服务器版本

创建适用于服务器的版本：

```python
# power_control_server.py
import subprocess
import re
import os

def get_power_cap(device_index=0):
    """获取当前GPU功率限制，单位W"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(device_index), "-q", "-d", "POWER"],
            check=True, capture_output=True, text=True
        )
        match = re.search(r"Power Limit\s+:\s+(\d+\.?\d*)\s+W", result.stdout)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception as e:
        print(f"获取功率限制失败: {e}")
        return 0.0

def set_power_cap(watts, device_index=0):
    """设置GPU功率限制，需要sudo权限"""
    try:
        # 方案1: 使用sudo（如果配置了NOPASSWD）
        result = subprocess.run(
            ["sudo", "nvidia-smi", "-i", str(device_index), "-pl", str(watts)],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"设置功率限制失败: {e.stderr}")
        print("提示: 可以使用--skip-set-power参数，手动设置功率后运行")
        return False

def get_current_power(device_index=0):
    """获取当前GPU实时功率，单位W"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(device_index), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"获取实时功率失败: {e}")
        return 0.0

if __name__ == "__main__":
    print(f"当前功率限制: {get_power_cap()}W")
    print(f"当前实时功率: {get_current_power()}W")
```

### 快速适配脚本

创建`adapt_to_server.py`自动适配：

```python
#!/usr/bin/env python
"""
自动适配代码到服务器环境
"""
import shutil
import os

def adapt_power_control():
    """适配功率控制模块"""
    print("适配 power_control.py...")

    if os.path.exists('power_control.py.wsl2'):
        print("  已备份，跳过")
        return

    # 备份
    shutil.copy('power_control.py', 'power_control.py.wsl2')

    # 读取并修改
    with open('power_control.py', 'r') as f:
        content = f.read()

    # 替换nvidia-smi路径
    content = content.replace('/usr/lib/wsl/lib/nvidia-smi', 'nvidia-smi')

    # 简化sudo密码处理
    content = content.replace(
        'def set_power_cap(watts, device_index=0, sudo_password="1234"):',
        'def set_power_cap(watts, device_index=0):'
    )
    content = content.replace(
        '["sudo", "-S", "/usr/lib/wsl/lib/nvidia-smi", "-i", str(device_index), "-pl", str(watts)],\n            input=sudo_password + "\\n",\n            text=True,',
        '["sudo", "nvidia-smi", "-i", str(device_index), "-pl", str(watts)],'
    )

    with open('power_control.py', 'w') as f:
        f.write(content)

    print("  完成")

def suggest_gpu_memory():
    """根据显存建议gpu_memory_utilization"""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        total_mem_mb = float(result.stdout.strip())
        total_mem_gb = total_mem_mb / 1024

        if total_mem_gb >= 24:
            suggested = 0.95
        elif total_mem_gb >= 16:
            suggested = 0.90
        elif total_mem_gb >= 12:
            suggested = 0.85
        else:
            suggested = 0.70

        print(f"\n检测到GPU显存: {total_mem_gb:.1f}GB")
        print(f"建议 gpu_memory_utilization = {suggested}")
        print(f"请编辑 llm_inference.py 第8行修改")

    except Exception as e:
        print(f"\n无法检测显存: {e}")

if __name__ == "__main__":
    print("=== 服务器环境适配工具 ===\n")
    adapt_power_control()
    suggest_gpu_memory()
    print("\n适配完成！")
```

---

## 服务器运行实验

### 方式1：手动设置功率（推荐，不需要sudo）

```bash
# 1. 手动设置GPU功率（在服务器终端）
sudo nvidia-smi -pl 140  # 设置为140W

# 2. 验证
nvidia-smi -q -d POWER

# 3. 运行实验（使用--skip-set-power）
bash run_all_fixed_experiments.sh
```

### 方式2：自动设置功率（需要sudo权限）

```bash
# 如果服务器允许sudo nvidia-smi -pl
# 直接运行即可
bash run_all_fixed_experiments.sh
```

### 修改功率档位

编辑`run_all_fixed_experiments.sh`，根据服务器GPU支持的功率范围调整：

```bash
# 例如：A10显卡可能支持100-300W
POWERS=(100 150 200 250 300)
```

---

## 结果回传

实验完成后，把结果传回本地分析：

```bash
# 在服务器上打包结果
cd /path/to/vllm
tar -czf results_$(date +%Y%m%d).tar.gz results/

# 传回本地
scp user@server-ip:/path/to/vllm/results_*.tar.gz /home/jensen/vllm/

# 本地解压分析
cd /home/jensen/vllm
tar -xzf results_*.tar.gz
python analyze_results.py
```

---

## 常见问题

### Q: 服务器没有sudo权限怎么办？
A: 使用`--skip-set-power`参数，找管理员帮忙设置功率，或者只在默认功率下测试。

### Q: 显存不够怎么办？
A: 降低`llm_inference.py`中的`gpu_memory_utilization`，或者使用更小的模型。

### Q: vLLM版本不兼容？
A: 确认服务器CUDA版本，可能需要调整vLLM版本。

### Q: 模型下载慢？
A: 使用hf-mirror.com镜像，或者从本地传输模型文件。

---

## 检查清单

- [ ] 代码已传输到服务器
- [ ] conda环境已创建
- [ ] 依赖已安装
- [ ] 模型已准备
- [ ] power_control.py已适配
- [ ] 显存参数已调整
- [ ] 单次测试通过
- [ ] 批量实验已运行
- [ ] 结果已回传分析
