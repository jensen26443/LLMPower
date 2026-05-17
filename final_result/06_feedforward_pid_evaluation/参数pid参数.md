# PID 参数表

## 数据来源

本表根据前馈 + PID 正式实验整理，覆盖两个输出长度：

| 实验 | 数据目录 | Metadata | Output length |
|---|---|---|---:|
| PID out100 | `experiment_results/feedforward/pid_guard_energy_first_out100_r50x3/` | `feedforward_eval_1777714017_metadata.json` | 100 |
| PID out200 | `experiment_results/feedforward/pid_guard_energy_first_out200_r50x3/` | `feedforward_eval_1777865373_metadata.json` | 200 |

两个实验使用相同的 PID 配置，策略为 `ff_decode_tbt_guarded_pid`，对照组为 `baseline_350w`。每个输出长度包含 6 个 query count 桶，每个配置 `repeats_per_batch=50`，`full_repeats=3`，因此每个输出长度每个策略共 900 个样本。

## PID 控制器参数

| 参数 | 取值 | 含义 |
|---|---:|---|
| `kp_prefill` | 0.1 | Prefill 阶段 TTFT 误差的比例增益 |
| `kp_decode` | 0.45 | Decode 阶段 TBT 误差的比例增益 |
| `ki` | 0.005 | 积分项增益 |
| `kd` | 0.001 | 微分项增益 |
| `pid_interval_sec` | 2.0 | Decode PID 两次反馈判断之间的最小时间间隔 |
| `pid_delta_limit_w` | 20.0 W | PID 对前馈功率的最大正向补偿量 |
| `pid_max_step_w` | 10.0 W | 单次实际功率调整的最大步长 |
| `pid_min_power_change_w` | 5.0 W | 小于该阈值的功率变化不下发 |
| `pid_deadband_ms` | 1.0 ms | Decode TBT 误差死区，误差不超过该值时不升功率 |
| `pid_min_decode_samples` | 4 | Decode PID 反馈前至少需要的有效样本数 |
| `pid_initial_skip_windows` | 1 | Decode 阶段跳过的初始反馈窗口数 |
| `pid_decode_confirm_windows` | 2 | Decode TBT 超预算后，需要连续确认的窗口数 |
| `pid_decay_step_w` | 5.0 W | TBT 回到预算内后，decode 补偿功率的衰减步长 |
| `pid_feedback_window_samples` | 32 | Decode TBT 反馈窗口保留的最近样本数 |
| `pid_ewma_alpha` | 1.0 | TBT 反馈 EWMA 系数；1.0 表示直接使用当前窗口 |
| `pid_power_max_w` | 350 W | PID 补偿后的最高功率上限 |
| `pid_ttft_budget_ratio` | 1.05 | TTFT 目标预算倍率 |
| `pid_tbt_budget_ratio` | 1.05 | TBT 目标预算倍率 |

## PID 更新公式

PID 状态更新使用如下形式：

```text
error = actual_value - target_value
integral = clamp(integral + error, -5000, 5000)
derivative = error - prev_error
raw_delta = kp * error + ki * integral + kd * derivative
delta_w = clamp(raw_delta, 0, pid_delta_limit_w)
```

本实验中 `ki=0`、`kd=0`，因此实际主要是比例控制：

```text
prefill_delta_w = clamp(0.1  * TTFT_error_ms, 0, 20)
decode_delta_w  = clamp(0.45 * TBT_error_ms,  0, 20)
```

PID 只进行正向补偿：当实际延迟低于目标时，`delta_w` 不会变成负数，因此不会在前馈功率桶基础上继续主动降功率。

## 目标值生成规则

PID 目标文件为 `feedforward_pid_targets.json`。代码优先使用纯前馈实验指标作为目标基准，目标值为：

```text
TTFT target = TTFT_ff * 1.05
TBT target  = TBT_ff  * 1.05
```

也就是说，PID guard 的目标不是回到 350W baseline 延迟，而是允许相对纯前馈指标再放宽 5%，用于避免明显超预算时补功率。

## out100 控制目标

| Query count | TTFT_ff (ms) | TTFT target (ms) | TBT_ff (ms) | TBT target (ms) |
|---:|---:|---:|---:|---:|
| 8 | 208.03 | 218.43 | 70.50 | 74.02 |
| 16 | 267.14 | 280.50 | 71.21 | 74.77 |
| 32 | 434.83 | 456.57 | 71.51 | 75.09 |
| 64 | 583.77 | 612.96 | 72.43 | 76.05 |
| 96 | 3632.17 | 3813.78 | 75.93 | 79.73 |
| 128 | 5723.20 | 6009.36 | 81.17 | 85.23 |

## out200 控制目标

| Query count | TTFT_ff (ms) | TTFT target (ms) | TBT_ff (ms) | TBT target (ms) |
|---:|---:|---:|---:|---:|
| 8 | 203.32 | 213.48 | 70.65 | 74.18 |
| 16 | 263.25 | 276.42 | 71.23 | 74.79 |
| 32 | 438.98 | 460.93 | 71.37 | 74.94 |
| 64 | 548.51 | 575.94 | 72.24 | 75.85 |
| 96 | 6058.33 | 6361.25 | 73.85 | 77.54 |
| 128 | 9380.75 | 9849.79 | 76.88 | 80.73 |

## 前馈基准功率桶

PID guard 叠加在纯前馈功率桶之上。两个输出长度使用相同的 decode 基准桶：

| Query count | Decode base power |
|---:|---:|
| 8 | 190W |
| 16 | 190W |
| 32 | 190W |
| 64 | 205W |
| 96 | 205W |
| 128 | 210W |

Prefill 阶段使用前馈分桶作为基准功率，PID 只在该基准上做正向补偿。

## 实际 PID 行为汇总

下表为 3 次 full repeat 聚合后的 PID 行为均值。

### out100

| Query count | Decode scheme | PID updates | Prefill delta (W) | Decode delta (W) | Prefill error (ms) | Decode error (ms) | Feedback TBT (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | `fixed_decode_190+pid_guard` | 1.00 | 0.30 | 0.00 | -19.44 | -3.45 | 70.57 |
| 16 | `fixed_decode_190+pid_guard` | 1.00 | 0.42 | 0.00 | -13.27 | -3.72 | 71.05 |
| 32 | `fixed_decode_190+pid_guard` | 1.00 | 1.21 | 0.00 | -19.45 | -4.00 | 71.09 |
| 64 | `fixed_decode_205+pid_guard` | 1.02 | 0.65 | 0.03 | -52.33 | -3.16 | 72.90 |
| 96 | `fixed_decode_205+pid_guard` | 1.01 | 0.80 | 0.00 | -213.50 | -9.88 | 69.85 |
| 128 | `fixed_decode_210+pid_guard` | 1.03 | 0.00 | 0.03 | -313.02 | -9.45 | 75.78 |

### out200

| Query count | Decode scheme | PID updates | Prefill delta (W) | Decode delta (W) | Prefill error (ms) | Decode error (ms) | Feedback TBT (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | `fixed_decode_190+pid_guard` | 1.00 | 0.82 | 0.00 | -8.49 | -3.55 | 70.63 |
| 16 | `fixed_decode_190+pid_guard` | 1.00 | 0.68 | 0.00 | -8.45 | -3.35 | 71.44 |
| 32 | `fixed_decode_190+pid_guard` | 1.00 | 0.90 | 0.00 | -17.39 | -3.69 | 71.25 |
| 64 | `fixed_decode_205+pid_guard` | 1.79 | 1.60 | 0.38 | 1.16 | 1.31 | 77.16 |
| 96 | `fixed_decode_205+pid_guard` | 2.67 | 1.32 | 0.00 | -316.45 | -7.13 | 70.41 |
| 128 | `fixed_decode_210+pid_guard` | 5.07 | 0.11 | 0.00 | -491.01 | -12.99 | 67.74 |

## 说明

1. out100 下 decode 反馈大多低于 TBT target，因此 decode delta 基本为 0，PID 主要表现为轻微 prefill 补偿。
2. out200 下 q=64 出现正向 decode error，触发了更明显的 decode 补偿；q=96 和 q=128 的 update 次数较多，但反馈 TBT 低于目标，因此没有形成正向 decode delta。
3. 当前 PID 参数更偏向 guard 机制，而不是持续主动调节机制：只有在延迟超过目标并经过确认窗口后才补功率，延迟回到目标内则保持或衰减补偿。

