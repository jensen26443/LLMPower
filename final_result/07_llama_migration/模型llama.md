# Meta-Llama-3.1-8B-Instruct-AWQ-INT4 实验结果总结

## 实验概述

本次实验将前馈 + PID 功率控制方法迁移到 `Meta-Llama-3.1-8B-Instruct-AWQ-INT4` 模型上，并与 `baseline_350w` 进行对比。实验结果目录为：

`experiment_results/feedforward/llama_pid_guard_out100_r50x3/`

实验使用 vLLM OpenAI API 服务，模型与 tokenizer 均来自本地目录：

`./Meta-Llama-3.1-8B-Instruct-AWQ-INT4`

对比策略如下：

| 策略 | 含义 |
|---|---|
| `baseline_350w` | 固定 350 W 功率上限的基线策略 |
| `ff_decode_tbt_guarded_pid` | 前馈 decode 功率推荐 + TBT guard + PID 反馈修正 |

本次实验中 `skip_set_power=false`，因此实验过程实际启用了 GPU 功率上限调整。

## 实验参数

| 参数 | 值 |
|---|---:|
| 模型 | `Meta-Llama-3.1-8B-Instruct-AWQ-INT4` |
| 输出长度 | 100 tokens |
| query count | 8, 16, 32, 64, 96, 128 |
| 每批重复次数 | 50 |
| full repeat | 3 |
| warmup batches | 2 |
| monitor warmup batches | 1 |
| queue seed | 20260401 |
| sampling seed | 20260401 |
| ShareGPT 数据目录 | `./input/ShareGPT` |
| decode 推荐表 | `experiment_results/decode_power_cap_batch_sweep/multi_load_tbt5_out100/images/decode_bucket_recommendations_q64_q96_205w_q128_210w_override.json` |
| PID target 文件 | `feedforward_pid_targets.json` |

PID 关键参数如下：

| 参数 | 值 |
|---|---:|
| `kp_prefill` | 0.1 |
| `kp_decode` | 0.45 |
| `ki` | 0.0 |
| `kd` | 0.0 |
| `pid_interval_sec` | 2.0 |
| `pid_delta_limit_w` | 20.0 |
| `pid_max_step_w` | 10.0 |
| `pid_min_power_change_w` | 5.0 |
| `pid_deadband_ms` | 1.0 |
| `pid_tbt_budget_ratio` | 1.05 |
| `pid_ttft_budget_ratio` | 1.05 |
| `pid_power_max_w` | 350 |

## 主要实验结果

相对 `baseline_350w`，前馈 + PID 策略的几何平均结果为：

| 指标 | 结果 |
|---|---:|
| 能耗节省 | 13.29% |
| TBT 增加 | 5.34% |
| TTFT 增加 | 5.19% |
| E2E 延迟增加 | 5.39% |

按 query count 分组的主要结果如下：

| Query Count | Baseline 能耗 (J) | 前馈+PID 能耗 (J) | 能耗节省 | Baseline 平均功率 (W) | 前馈+PID 平均功率 (W) | 平均功率降低 | TBT 增加 | TTFT 增加 | E2E 增加 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1449.99 | 1280.72 | 11.67% | 214.29 | 180.36 | 15.84% | 5.05% | 4.89% | 5.05% |
| 16 | 1499.02 | 1307.63 | 12.77% | 217.51 | 180.06 | 17.22% | 5.36% | 6.15% | 5.39% |
| 32 | 1564.38 | 1342.09 | 14.21% | 224.10 | 180.96 | 19.25% | 6.62% | 4.67% | 6.50% |
| 64 | 1676.59 | 1487.08 | 11.30% | 225.51 | 191.80 | 14.95% | 4.70% | 4.22% | 4.66% |
| 96 | 3743.64 | 3218.67 | 14.02% | 245.54 | 200.78 | 18.23% | 5.25% | 5.29% | 5.26% |
| 128 | 4507.21 | 3800.66 | 15.68% | 256.49 | 205.44 | 19.90% | 5.08% | 6.01% | 5.47% |
| GEOMEAN | - | - | 13.29% | - | - | - | 5.34% | 5.19% | 5.39% |

从整体平均绝对值看，`ff_decode_tbt_guarded_pid` 将平均功率从 230.57 W 降到 189.90 W，平均能耗从 2406.81 J 降到 2072.81 J。代价是平均 TBT 从 68.17 ms 增加到 71.81 ms，平均 TTFT 从 1701.22 ms 增加到 1796.52 ms，平均 E2E 从 8450.02 ms 增加到 8905.02 ms。

## 结果分析

本次 Llama 8B AWQ-INT4 实验表明，前馈 + PID 策略在全部 query count 上均取得了正向能耗收益，能耗节省范围为 11.30% 到 15.68%。其中 q128 的节能幅度最高，达到 15.68%；q64 的节能幅度最低，为 11.30%。

延迟方面，TBT、TTFT 和 E2E 均有约 4% 到 7% 的增加。几何平均下，TBT 增加 5.34%，TTFT 增加 5.19%，E2E 增加 5.39%，基本贴近 PID 配置中 1.05 的预算约束。这说明 PID 反馈没有完全消除延迟损失，但将延迟增加控制在约 5% 左右，同时换取了约 13% 的整体能耗下降。

功率方面，前馈 + PID 策略在所有 query count 上都降低了平均功率，降幅为 14.95% 到 19.90%。随着 query count 增大，baseline 平均功率从 214.29 W 上升到 256.49 W，而前馈 + PID 策略的平均功率主要维持在 180 W 到 205 W 区间，说明 decode 阶段功率推荐和 PID 修正对高并发负载仍然有效。

## 图表输出

本次实验已生成无误差棒的 2x2 四联柱状图：

- `experiment_results/feedforward/llama_pid_guard_out100_r50x3/images_bar/llama_pid_guard_out100_metrics_2x2.png`
- `experiment_results/feedforward/llama_pid_guard_out100_r50x3/images_bar/llama_pid_guard_out100_metrics_2x2.pdf`
- `experiment_results/feedforward/llama_pid_guard_out100_r50x3/images_bar/llama_pid_guard_out100_metrics_2x2.svg`

对应 summary CSV：

`experiment_results/feedforward/llama_pid_guard_out100_r50x3/images_bar/llama_pid_guard_out100_metrics_2x2_summary.csv`

## 结论

在 `Meta-Llama-3.1-8B-Instruct-AWQ-INT4` 上，前馈 + PID 控制策略可以稳定降低推理能耗。相比 350 W baseline，整体几何平均节能 13.29%，平均功率降低约 40.67 W，同时将 TBT、TTFT 和 E2E 的增长控制在约 5% 左右。该结果说明该策略可以从 Qwen2.5-7B-Instruct-AWQ 迁移到相近规模的 Llama 8B AWQ-INT4 模型，并保持较好的节能效果。
