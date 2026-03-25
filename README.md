# LLM Power Control

用于研究 vLLM 推理过程中功率限制、时延和能耗关系的实验平台。项目围绕固定功率实验、预填充阶段建模、解码阶段建模和策略评估展开，支持本地脚本实验和 vLLM OpenAI 服务模式测量。

## 项目结构

- `power_control.py`：GPU 功率限制设置与读取
- `llm_inference.py`：vLLM 推理封装，记录 TTFT、TPOT/TBT、E2E
- `load_generator.py`：负载生成与 token 长度控制
- `monitor.py`：功率、显存、温度采样
- `run_experiment.py`：固定功率实验入口
- `run_prefill_modeling.py`：预填充阶段离线建模
- `run_decode_modeling.py`：解码阶段离线建模
- `run_strategy_evaluation.py`：动态功率策略评估
- `analyze_*.py`：结果分析与绘图脚本

常用结果目录：

- `results/`：固定功率实验
- `results0/`：查询数量相关实验
- `results1/`：策略评估实验
- `results_decode/decode_modeling/`：解码阶段建模实验

## 环境要求

- Python 3.12
- vLLM 0.17.0
- NVIDIA GPU
- `nvidia-smi` 可用

安装依赖：

```bash
pip install -r requirements.txt
```

检查环境：

```bash
python test_env.py
python run_experiment.py --power 240 --show-power-info
```

## 常用运行方式

固定功率实验：

```bash
python run_experiment.py --power 240 --load-type mixed --count 100 --concurrency 1
```

启动 vLLM 服务：

```bash
bash start_vllm_server.sh
```

TTFT/TPOT/E2E 测试：

```bash
python test_ttft_openai.py --warmup 5 --num-prompts 20 --input-len 32 --max-tokens 50
```

预填充阶段建模：

```bash
bash run_prefill_experiments.sh
```

解码阶段建模：

```bash
python run_decode_modeling.py --skip-set-power
```

或使用批量脚本：

```bash
bash run_decode_experiments.sh
```

策略评估：

```bash
python run_strategy_evaluation.py
```

## 结果分析

固定功率实验分析：

```bash
python analyze_results.py
```

预填充阶段分析：

```bash
python analyze_prefill_modeling.py
```

解码阶段分析：

```bash
python analyze_decode_modeling.py \
  --result-dir results_decode/decode_modeling \
  --output-dir results_decode/decode_modeling/images
```

## 当前实验重点

- 固定功率下的时延、吞吐和能耗对比
- 预填充阶段建模：`P_prefill = f(C)`，`TTFT = g(C)`
- 解码阶段建模：`P_decoding = f(B, KV)`，`TBT = g(B, KV)`
- 动态功率调节策略与基线对比

## 说明

- 不建议提交大型模型文件和实验结果文件
- 涉及功率限制时，请确认当前环境具备对应权限
- 解码阶段建模默认输出目录为 `results_decode/decode_modeling/`
