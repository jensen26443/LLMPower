# Prefill Token-Power Modeling Report

- Raw data: `experiment_results/prefill_token_power_modeling/gpu1_350w_block_nocache_fixed_0_20000/prefill_token_power_modeling_1777904889_raw.csv`
- Valid samples: 174
- Target token range: 1 - 20000
- Actual token range: 1 - 20000
- Polynomial degree: 2
- Non-power fit mode: linear
- Front-only max token: none
- Power fit: `x <= 3000: y = 0.0357623x +184.6368; x > 3000: y = 9.56693ln(x / 3000) +278.3868`, combined R2=0.9198, front R2=0.8320, tail R2=0.8213
- Active power fit: `x <= 3000: y = 0.0242677x +214.5802; x > 3000: y = 9.73005ln(x / 3000) +278.1580`, combined R2=0.8493, front R2=0.6919, tail R2=0.8165
- Energy-per-request fit: `y = 0.0512112x -2.4559`, R2=0.9945

- Primary power modeling metric: `avg_active_power_w`, which filters idle valleys inside repeated-request blocks. `avg_power_w` remains the full-window average for energy accounting. `median_power_w` and `p95_power_w` are provided for robustness; `peak_power_w` is diagnostic only because short prefill windows are sensitive to sampling phase. Power figures use a linear fit in front-only mode; otherwise they use a segmented fit with a linear front and logarithmic saturated tail. Equations are kept in this report and `fit_results.json` instead of being drawn inside the axes. `first_ttft_ms` captures the first request in each block; `avg_ttft_ms` is the block steady-state mean. TTFT and energy figures use linear fits in both full-range and front-only outputs. Energy plots should use per-request normalization in block mode.

Generated figures follow `image.md`: white background, full frame, inward ticks, minor ticks, Times/SimSun font preference, and 600 dpi export.
