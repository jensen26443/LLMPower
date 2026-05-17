# Decode Phase Modeling Report
## Summary
- Config count: 135
- Batch size range: 1 - 64
- Target output token range: 10 - 300
- Median config-level TTFT P50: 232.76 ms
- Median config-level TBT P50: 68.17 ms
- Median config-level E2E P50: 5282.18 ms
## Key Observations
- Highest decode power: 233.36 W (batch=60, target_output=300, normalized_kv_blocks=19)
- Highest TBT: 74.27 ms (batch=64, target_output=10, normalized_kv_blocks=1)
- Lowest TBT: 66.59 ms (batch=1, target_output=10, normalized_kv_blocks=1)
- Highest TBT P95: 82.38 ms (batch=64, target_output=10, normalized_kv_blocks=1)
## Output Files
- decode_power_heatmap.png
- decode_tbt_heatmap.png
- decode_tbt_by_batch.png
- decode_tbt_p50_by_batch.png
- decode_tbt_p95_by_batch.png
- decode_tbt_p99_by_batch.png
- decode_power_by_kv.png
- aggregated csv now includes mean/p50/p95/p99 for TTFT/TBT/E2E