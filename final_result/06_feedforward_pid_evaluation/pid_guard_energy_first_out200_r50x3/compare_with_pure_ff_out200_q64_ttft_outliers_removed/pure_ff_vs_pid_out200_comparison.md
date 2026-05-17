# Pure FF vs FF + PID Output200 Comparison

- Pure FF source: `experiment_results/feedforward/final_guarded_out200_r50x3_retry/feedforward_eval_1777603885_aggregated.csv`
- FF + PID source: `experiment_results/feedforward/pid_guard_energy_first_out200_r50x3/images_q64_ttft_outliers_removed/feedforward_eval_1777865373_aggregated_q64_ttft_le900.csv`
- PID side uses q=64 TTFT outlier-filtered data: remove `q=64 and avg_ttft_ms > 900ms` batch rows.

## Geomean Summary

| Method | Energy Saving | TTFT Increase | TBT Increase | E2E Increase |
|---|---:|---:|---:|---:|
| FF + PID | 11.43% | 4.50% | 3.94% | 4.04% |
| Pure FF | 11.89% | 4.11% | 4.05% | 4.17% |

## Per Query Count

| Method | q | Energy Saving | TTFT Increase | TBT Increase | E2E Increase |
|---|---:|---:|---:|---:|---:|
| FF + PID | 8 | 10.58% | 3.89% | 4.04% | 4.04% |
| Pure FF | 8 | 10.91% | 5.08% | 4.16% | 4.17% |
| FF + PID | 16 | 11.45% | 5.17% | 4.10% | 4.12% |
| Pure FF | 16 | 11.60% | 2.78% | 4.24% | 4.21% |
| FF + PID | 32 | 13.44% | 4.84% | 4.66% | 4.67% |
| Pure FF | 32 | 13.93% | 3.87% | 4.81% | 4.78% |
| FF + PID | 64 | 10.58% | 4.26% | 3.66% | 3.68% |
| Pure FF | 64 | 10.69% | 3.20% | 3.65% | 3.64% |
| FF + PID | 96 | 11.05% | 4.46% | 3.77% | 3.97% |
| Pure FF | 96 | 11.62% | 4.92% | 3.72% | 4.07% |
| FF + PID | 128 | 11.47% | 4.41% | 3.40% | 3.79% |
| Pure FF | 128 | 12.53% | 4.83% | 3.73% | 4.15% |