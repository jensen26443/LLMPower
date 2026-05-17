# Decode Strategy Evaluation Report

- Source CSVs: `experiment_results/decode_strategy/strategy_evaluation_policy_retry_q8/decode_strategy_eval_1777989123_aggregated.csv`, `experiment_results/decode_strategy/strategy_evaluation_policy_retry_q16/decode_strategy_eval_1778006288_aggregated.csv`
- Output lengths: [8, 16, 32, 64, 96, 128]
- Concurrency values: [8, 16]

## GEOMEAN Relative Metrics

| Strategy | Energy Saving (%) | TBT Loss (%) |
|---|---:|---:|
| 150/151/191/210/220W | 12.04 | 5.64 |
| 150/170/200/220/230W | 9.05 | 4.01 |
| 190/205/210W | 9.62 | 3.18 |