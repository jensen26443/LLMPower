# Prefill Token-Power Modeling Report

- Raw data: `experiment_results/prefill_token_power_modeling/gpu1_350w_block_nocache_fixed_0_20000/prefill_token_power_modeling_1777904889_raw.csv`
- Valid samples: 156
- Target token range: 1 - 3000
- Actual token range: 1 - 3000
- Polynomial degree: 2
- Non-power fit mode: linear
- Front-only max token: 3000.0
- Power fit: `y = 0.0357623x +184.6368`, R2=0.8320
- Active power fit: `y = 0.0242677x +214.5802`, R2=0.6919
- Energy-per-request fit: `y = 0.0389343x +9.0671`, R2=0.9945

- Primary power modeling metric: `avg_active_power_w`, which filters idle valleys inside repeated-request blocks. `avg_power_w` remains the full-window average for energy accounting. `median_power_w` and `p95_power_w` are provided for robustness; `peak_power_w` is diagnostic only because short prefill windows are sensitive to sampling phase. Power figures use a linear fit in front-only mode; otherwise they use a segmented fit with a linear front and logarithmic saturated tail. Equations are kept in this report and `fit_results.json` instead of being drawn inside the axes. `first_ttft_ms` captures the first request in each block; `avg_ttft_ms` is the block steady-state mean. TTFT and energy figures use linear fits in both full-range and front-only outputs. Energy plots should use per-request normalization in block mode.

Generated figures follow `image.md`: white background, full frame, inward ticks, minor ticks, Times/SimSun font preference, and 600 dpi export.
