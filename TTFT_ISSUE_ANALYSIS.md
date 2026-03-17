
# TTFT 测量问题分析

## 问题描述

实验数据中 TTFT（首个词元生成延迟）和 TBT（词元间延迟）几乎相等，TTFT 只比 TBT 多几毫秒，这不符合正常的 LLM 推理行为。

## 正常的 LLM 推理延迟模式

```
请求 → [预填充阶段] → [解码阶段1] → [解码阶段2] → ... → [解码阶段N] → 完成
     ↑               ↑                ↑                ↑                ↑
     0ms          TTFT          TTFT+TBT1     TTFT+TBT1+TBT2      E2E

预期关系：
- TTFT 主要包含：预填充时间 + 第一个token解码时间
- TBT 主要包含：后续每个token的解码时间
- 通常 TTFT &gt;&gt; TBT（因为预填充需要处理整个输入序列）
```

## 当前实现的问题

### 代码位置：`llm_inference.py:36-71`

**当前测量逻辑：**

```python
# 第一步：用 max_tokens=1 测量 TTFT
temp_params = SamplingParams(max_tokens=1)
ttft_start = time.time()
llm.generate([prompt], temp_params)  # ← 问题在这里！
ttft_end = time.time()
ttft = (ttft_end - ttft_start) * 1000

# 第二步：完整生成
full_start = time.time()
outputs = llm.generate([prompt], sampling_params)  # max_tokens=100
full_end = time.time()
e2e = (full_end - full_start) * 1000

# 第三步：计算 TBT
avg_tbt = (e2e - ttft) / (token_count - 1)
```

### 问题根源

1. **`llm.generate()` 是非流式的**：它返回的是**完整生成结束**的时间，不是第一个token生成的时间

2. **用 `max_tokens=1` 测量的时间包含**：
   - 预填充时间（处理输入）
   - 1个token解码时间
   - vLLM 的各种开销（初始化、后处理等）
   - **但这不是真实的 TTFT！**

3. **完整生成时**：
   - 同样包含预填充 + 100个token解码
   - 但因为缓存或其他原因，两次调用的开销被分摊了

4. **结果**：
   - TTFT ≈ 预填充 + 1个token解码 + 开销
   - E2E ≈ 预填充 + 100个token解码 + 开销
   - TBT = (E2E - TTFT) / 99 ≈ (99个token解码) / 99 ≈ 1个token解码时间
   - 所以 TTFT ≈ TBT + 预填充 + 开销... 等等，为什么数据中 TTFT ≈ TBT？

### 数据分析

从 `results0/data/350W_mixed_8q_1773727571_inference.csv`：

| 指标 | 值 |
|------|-----|
| TTFT | 70-86ms |
| TBT | 66-66.8ms |
| E2E | ~6663ms |
| token_count | 100 |

计算：
- E2E - TTFT = 6663 - 75 ≈ 6588ms
- 6588ms / 99 ≈ 66.5ms = TBT ✓

这说明计算逻辑本身是对的，但 TTFT 的测量值不对！

## 正确的测量方法

### 使用 vLLM 流式输出

```python
from vllm import LLM, SamplingParams

llm = LLM(model=...)
sampling_params = SamplingParams(max_tokens=100)

first_token_time = None
full_start = time.time()
token_times = []

# 使用流式生成
for output in llm.generate([prompt], sampling_params, stream=True):
    current_time = time.time()
    if first_token_time is None:
        first_token_time = current_time  # ← 真实的 TTFT！
    token_times.append(current_time)

full_end = time.time()

# 计算正确的指标
ttft = (first_token_time - full_start) * 1000
e2e = (full_end - full_start) * 1000

# 计算每个 TBT
tbts = []
for i in range(1, len(token_times)):
    tbt = (token_times[i] - token_times[i-1]) * 1000
    tbts.append(tbt)

avg_tbt = sum(tbts) / len(tbts) if tbts else 0
```

### 预期的正常结果

| 指标 | 预期值（估算） |
|------|---------------|
| TTFT | 200-500ms（取决于输入长度） |
| TBT | 30-100ms |
| 关系 | TTFT &gt;&gt; TBT |

## 修复方案

1. 修改 `llm_inference.py` 中的 `infer()` 方法，使用流式输出
2. 保持向后兼容性（可选参数选择测量方法）
3. 重新运行实验收集正确的数据

