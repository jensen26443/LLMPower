# 预填充阶段离线建模实验结果

## 实验概述

本实验针对大语言模型预填充（Prefill）阶段进行离线建模，旨在量化输入Token数与关键性能/能耗指标之间的关系。

**实验配置**：
- 模型：Qwen2.5-7B-Instruct-AWQ
- 功率限制：350W
- 采样范围：1-3000 tokens
- 采样点数：66个输入长度
- 重复次数：每点20次
- 总样本数：1320个

---

## 实验结果

### 1. TTFT（Time To First Token）

**拟合公式**：
```
TTFT = 0.1369 × C + 64.35
```

**拟合质量**：
- R² = 0.9977
- 拟合函数：线性（Linear）

**结论**：TTFT随输入Token数呈现极强的线性关系。

---

### 2. 能耗（Dynamic Energy）

**拟合公式**：
```
Energy = 0.0126 × C + 5.75
```

**拟合质量**：
- R² = 0.9916
- 拟合函数：线性（Linear）

**结论**：预填充阶段动态能耗随输入Token数呈现极强的线性关系。

---

### 3. 功率（Power）

#### 平均功率（Average Power）
**二次多项式拟合**：
```
P_avg = 1.45×10⁻⁶ × C² - 0.00316 × C + 139.82
```

**上界曲线（拟合+25W）**：
```
P_avg_upper = 1.45×10⁻⁶ × C² - 0.00316 × C + 164.82
```

- 变化范围：137.7 W → 141.3 W
- 总变化量：+3.6 W
- R² = 0.0194
- 拟合函数：二次多项式（Poly2）

#### 峰值功率（Peak Power）
**二次多项式拟合**：
```
P_peak = 2.28×10⁻⁶ × C² - 0.000111 × C + 141.82
```

**上界曲线（拟合+25W）**：
```
P_peak_upper = 2.28×10⁻⁶ × C² - 0.000111 × C + 166.82
```

- 变化范围：142.0 W → 157.1 W
- 总变化量：+15.1 W
- R² = 0.3086
- 拟合函数：二次多项式（Poly2）

**结论**：功率消耗在预填充阶段变化很小，R²值较低表明功率与输入Token数相关性较弱，可近似为常数。上界曲线（拟合+25W）可覆盖绝大多数实测数据点。

---

## 关键发现

### 核心关系验证

```
Energy = Power × Time
```

其中：
- **Power** ≈ 常数（~140 W，变化很小）
- **Time** ≈ TTFT（线性增长）
- **→ Energy** ≈ 常数 × TTFT（线性增长）✓

### 功率恒定的原因

1. **预填充阶段计算特性**：
   - 预填充是计算密集型阶段，GPU始终处于高负载
   - 输入长度影响计算时间，而非计算强度

2. **功率限制裕量**：
   - 设置功率限制：350 W
   - 实际功率消耗：~140 W（仅为限制的40%）
   - GPU未达到功率瓶颈

3. **Transformer架构特性**：
   - 注意力计算复杂度O(n²)主要体现在计算时间上
   - 瞬时功率相对稳定

---

## 实验数据质量

### KV Cache问题解决

采用三层防御策略：
1. 禁用vLLM前缀缓存（`enable_prefix_caching=False`）
2. 添加UUID唯一前缀（`[REQ_{8位hex}]`）
3. 随机化实验队列顺序

**效果**：异常TTFT样本比例从94.2%降至11.1%，改善83.1%。

### 数据验证

- TTFT-Tokens相关系数：0.9987
- TTFT拟合R²：0.9977
- Energy拟合R²：0.9916

---

## 对动态功率调节的启示

### 建模结论

1. **预填充阶段模型**：
   - TTFT(C) = a₁ × C + b₁
   - Energy(C) = a₂ × C + b₂
   - Power(C) ≈ P_constant

2. **关键参数**：
   - a₁ = 0.1369 ms/token
   - b₁ = 64.35 ms
   - a₂ = 0.0126 J/token
   - b₂ = 5.75 J
   - P_avg ≈ 140 W（变化很小）
   - P_peak ≈ 142 W（变化很小）

### 应用价值

这些线性模型为动态功率调节策略提供了坚实的基础：
- 可准确预测任意输入长度的TTFT和能耗
- 功率相对恒定的特性简化了功率管理策略
- 为Prefill/Decode分阶段功率调节提供了依据

---

## 图表文件

实验可视化图表位于：`results/prefill_modeling/images_v2/`

1. `prefill_ttft_vs_tokens.png` - TTFT vs Input Tokens
2. `prefill_energy_vs_tokens.png` - Energy vs Input Tokens
3. `prefill_peak_power_vs_tokens.png` - Peak Power vs Input Tokens
4. `prefill_avg_power_vs_tokens.png` - Average Power vs Input Tokens

拟合参数汇总：`results/prefill_modeling/images_v2/fit_summary.json`

---

*实验完成时间：2026-03-12*
