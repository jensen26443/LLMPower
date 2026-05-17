# 前馈+PID 策略评估

## 数据来源与计算口径

本评估汇总纯前馈策略和前馈+PID策略在 output length 100 与 200 下的实验结果。所有指标均相对各自实验目录中的 `baseline_350w` 计算，不跨实验目录混用 baseline。

数据来源：

- 纯前馈 out100：`experiment_results/feedforward/final_guarded_out100_r50x3/`
- 纯前馈 out200：`experiment_results/feedforward/final_guarded_out200_r50x3_retry/`
- 前馈+PID out100：`experiment_results/feedforward/pid_guard_energy_first_out100_r50x3/`
- 前馈+PID out200：`experiment_results/feedforward/pid_guard_energy_first_out200_r50x3/images_q64_ttft_outliers_removed/`

图表与汇总表输出：

- 带误差棒版本：`experiment_results/feedforward/paper_figures_ff_vs_pid_separate/`
- 无误差棒版本：`experiment_results/feedforward/paper_figures_ff_vs_pid_separate_no_errorbars/`

计算公式：

- `Energy Saving = (1 - strategy_energy / baseline_energy) * 100`
- `Latency Increase = (strategy_latency / baseline_latency - 1) * 100`

其中延迟指标包括 TBT、TTFT 和 E2E。每个 query count 使用 3 次 full repeat 的平均值；GEOMEAN 使用各 repeat 内相对 baseline 的比值做几何平均后再转为百分比。

## 总体结论

前馈+PID策略在两个输出长度下都能保持约 10% 到 11% 的节能收益，同时把 TBT 与 E2E 增幅控制在约 4% 左右。相比纯前馈策略，前馈+PID的主要作用不是进一步提高节能，而是略微降低 TBT 与 E2E 的性能代价。

从 GEOMEAN 看：

- out100 下，前馈+PID节能为 10.34%，比纯前馈低 0.31 个百分点；TBT、TTFT、E2E 增幅分别降低 0.10、0.49、0.20 个百分点。
- out200 下，前馈+PID节能为 11.43%，比纯前馈低 0.46 个百分点；TBT 与 E2E 增幅分别降低 0.11、0.12 个百分点，但 TTFT 增幅高 0.40 个百分点。
- 前馈+PID在高负载 q=128 下仍有明显节能：out100 为 13.22%，out200 为 11.47%。

因此，当前数据支持这样的表述：前馈+PID在牺牲少量节能收益的情况下，能让 decode 相关延迟指标更稳健，尤其是 TBT 和 E2E；但对 TTFT 的改善不稳定，out200 下还略差于纯前馈。

## GEOMEAN 汇总

| 策略 | Output length | Energy Saving (%) | TBT Increase (%) | TTFT Increase (%) | E2E Increase (%) |
|:--|--:|--:|--:|--:|--:|
| 纯前馈 | 100 | 10.64 | 4.11 | 4.43 | 4.30 |
| 前馈+PID | 100 | 10.34 | 4.02 | 3.94 | 4.11 |
| 纯前馈 | 200 | 11.89 | 4.05 | 4.11 | 4.17 |
| 前馈+PID | 200 | 11.43 | 3.94 | 4.51 | 4.04 |

## 前馈+PID 相对纯前馈的 GEOMEAN 差异

数值为“前馈+PID - 纯前馈”，单位为百分点。节能差异越高越好；延迟增幅差异越低越好。

| Output length | Energy Saving 差异 | TBT Increase 差异 | TTFT Increase 差异 | E2E Increase 差异 |
|--:|--:|--:|--:|--:|
| 100 | -0.31 | -0.10 | -0.49 | -0.20 |
| 200 | -0.46 | -0.11 | +0.40 | -0.12 |

## 前馈+PID out100 主要数据

| Query count | Energy Saving (%) | TBT Increase (%) | TTFT Increase (%) | E2E Increase (%) |
|--:|--:|--:|--:|--:|
| 8 | 8.66 | 3.90 | -1.41 | 3.74 |
| 16 | 9.49 | 4.07 | 5.43 | 4.12 |
| 32 | 10.66 | 4.45 | 6.22 | 4.55 |
| 64 | 8.69 | 3.55 | 4.28 | 3.60 |
| 96 | 11.21 | 3.97 | 4.30 | 4.08 |
| 128 | 13.22 | 4.16 | 5.13 | 4.56 |
| GEOMEAN | 10.34 | 4.02 | 3.94 | 4.11 |

out100 下，前馈+PID的节能范围为 8.66% 到 13.22%，最高出现在 q=128。TBT 增幅范围为 3.55% 到 4.45%，E2E 增幅范围为 3.60% 到 4.56%，整体较稳定。TTFT 在 q=8 为 -1.41%，说明该点相对 baseline 没有 TTFT 损失，但 q=32 达到 6.22%，TTFT 波动仍然存在。

## 前馈+PID out200 主要数据

| Query count | Energy Saving (%) | TBT Increase (%) | TTFT Increase (%) | E2E Increase (%) |
|--:|--:|--:|--:|--:|
| 8 | 10.58 | 4.04 | 4.06 | 4.04 |
| 16 | 11.45 | 4.10 | 5.18 | 4.12 |
| 32 | 13.44 | 4.66 | 4.85 | 4.67 |
| 64 | 10.58 | 3.66 | 4.26 | 3.68 |
| 96 | 11.05 | 3.77 | 4.47 | 3.97 |
| 128 | 11.47 | 3.40 | 4.41 | 3.79 |
| GEOMEAN | 11.43 | 3.94 | 4.51 | 4.04 |

out200 下，前馈+PID的节能范围为 10.58% 到 13.44%，最高出现在 q=32。TBT 增幅范围为 3.40% 到 4.66%，q=128 的 TBT 增幅最低。E2E 增幅范围为 3.68% 到 4.67%，整体仍接近 4%。该组数据使用了 q=64 TTFT 长尾去除后的聚合结果，因此更适合作为论文展示口径。

## 论文表述建议

可以将前馈+PID策略描述为一种以稳定延迟约束为目标的补偿机制：它在保留大部分纯前馈节能收益的同时，进一步压低 TBT 和 E2E 的相对增幅。当前实验中，前馈+PID在 out100 和 out200 下分别取得 10.34% 和 11.43% 的 GEOMEAN 节能，TBT 增幅分别为 4.02% 和 3.94%，E2E 增幅分别为 4.11% 和 4.04%。这说明策略能够在约 4% 的端到端延迟代价内提供约 10% 以上的能耗下降。

需要注意的是，PID并未显著提升节能收益，且 TTFT 指标仍受实验波动影响。若正文强调 PID 的贡献，建议重点放在“延迟稳定性”和“TBT/E2E 代价降低”，而不是“节能率提升”。
