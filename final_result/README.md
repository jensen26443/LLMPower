# final_result 结果整理说明

本目录按结题检查需要整理了 7 类正式实验结果。所有文件均为 copy，原始实验目录保持不变。

## 目录结构

| 目录 | 实验 | 主要内容 |
|---|---|---|
| `00_overview/` | 总览材料 | `代码内容整理.md`、`实验参数表格.md` |
| `01_prefill_token_power_modeling/` | 预填充阶段 Token-Power 建模实验 | 建模结果目录、raw/aggregated CSV、fit json、建模图片和报告 |
| `02_prefill_strategy_evaluation/` | 预填充阶段策略评估 | prefill 策略原始/聚合结果、分析图、最终版论文图 |
| `03_decode_modeling/` | 解码阶段建模 | merged filtered 数据、paper figures、TBT batch 建模结果 |
| `04_decode_strategy_evaluation/` | 解码阶段策略评估 | q=8/q=16 原始结果、merged 图表、最终版 image_bar 论文图 |
| `05_pure_feedforward_controller/` | 纯前馈控制器实验 | out100/out200 正式结果、聚合 CSV、最终版纯前馈 2x2 图 |
| `06_feedforward_pid_evaluation/` | 前馈+PID 策略评估 | PID out100/out200 正式结果、q64 TTFT outlier removed 口径、最终版 PID 2x2 图 |
| `07_llama_migration/` | 迁移到 Llama 模型的实验结果 | Llama 8B AWQ-INT4 正式结果、聚合 CSV、最终版 2x2 图 |

## 来源路径

### 1. 预填充阶段 Token-Power 建模实验

- 文档：`预填充阶段 Token-Power 建模实验总结.md`
- 来源目录：`experiment_results/prefill_token_power_modeling/gpu1_350w_block_nocache_fixed_0_20000/`
- 复制到：`final_result/01_prefill_token_power_modeling/`

### 2. 预填充阶段策略评估

- 文档：`预填充阶段策略评估.md`
- 来源目录：`experiment_results/prefill_concurrent_evaluation/prefill_strategy_gpu0_r50x3/`
- 最终图来源：`experiment_results/prefill_concurrent_evaluation/prefill_strategy_gpu0_r50x3/images_paper_energy_ttft_only_image_bar/`
- 复制到：`final_result/02_prefill_strategy_evaluation/`
- 最终图副本：`final_result/02_prefill_strategy_evaluation/final_paper_figures/`

### 3. 解码阶段建模

- 文档：`解码阶段建模.md`
- 来源目录：`experiment_results/decode_modeling/decode_modeling/merged_filtered/`
- 复制到：`final_result/03_decode_modeling/`

### 4. 解码阶段策略评估

- 文档：`解码阶段策略评估.md`
- 来源目录：
  - `experiment_results/decode_strategy/strategy_evaluation_policy_retry_q8/`
  - `experiment_results/decode_strategy/strategy_evaluation_policy_retry_q16/`
  - `experiment_results/decode_strategy/strategy_evaluation_policy_retry_merged/`
- 最终图来源：`experiment_results/decode_strategy/strategy_evaluation_policy_retry_merged/images_image_bar/`
- 复制到：`final_result/04_decode_strategy_evaluation/`
- 最终图副本：`final_result/04_decode_strategy_evaluation/final_paper_figures/`

### 5. 纯前馈控制器实验

- 文档：`纯前馈控制器.md`
- 来源目录：
  - `experiment_results/feedforward/final_guarded_out100_r50x3/`
  - `experiment_results/feedforward/final_guarded_out200_r50x3_retry/`
- 最终图来源：`experiment_results/feedforward/paper_figures_ff_vs_pid_separate_no_errorbars_image_bar/`
- 复制到：`final_result/05_pure_feedforward_controller/`
- 最终图副本：`final_result/05_pure_feedforward_controller/final_paper_figures/`

### 6. 前馈+PID 策略评估

- 文档：`前馈+pid策略评估.md`、`参数pid参数.md`
- 来源目录：
  - `experiment_results/feedforward/pid_guard_energy_first_out100_r50x3/`
  - `experiment_results/feedforward/pid_guard_energy_first_out200_r50x3/`
- out200 论文口径：`images_q64_ttft_outliers_removed/`
- 最终图来源：`experiment_results/feedforward/paper_figures_ff_vs_pid_separate_no_errorbars_image_bar/`
- 复制到：`final_result/06_feedforward_pid_evaluation/`
- 最终图副本：`final_result/06_feedforward_pid_evaluation/final_paper_figures/`

### 7. 迁移到 Llama 模型的实验结果

- 文档：`模型llama.md`
- 来源目录：`experiment_results/feedforward/llama_pid_guard_out100_r50x3/`
- 最终图来源：`experiment_results/feedforward/llama_pid_guard_out100_r50x3/images_bar/`
- 复制到：`final_result/07_llama_migration/`
- 最终图副本：`final_result/07_llama_migration/final_paper_figures/`

## 使用建议

结题展示时优先打开：

1. `00_overview/代码内容整理.md`
2. 各实验目录下对应 Markdown 总结文件
3. 各实验目录下的 `final_paper_figures/`
4. 需要追溯数据时再打开对应实验目录中的 `*_aggregated.csv`、`*_raw.csv`、`*_metadata.json`
