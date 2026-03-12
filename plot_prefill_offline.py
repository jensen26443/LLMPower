#!/usr/bin/env python3
"""Prefill 离线建模结果绘图脚本（独立版）

输入：run_prefill_modeling.py 产出的 *_raw.csv 与 *_aggregated.csv
输出：
- 输入 token 数 vs Prefill 功率/能耗散点图 + 拟合曲线
- 输入 token 数 vs TTFT 散点图 + 拟合曲线
"""

import argparse
import glob
import json
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


plt.style.use("seaborn-v0_8-whitegrid")


def linear_func(x, a, b):
    return a * x + b


def log_func(x, a, b):
    return a * np.log(x + 1) + b


def sqrt_func(x, a, b):
    return a * np.sqrt(x) + b


def poly2_func(x, a, b, c):
    return a * x ** 2 + b * x + c


FIT_FUNCTIONS = {
    "linear": (linear_func, [1.0, 0.0]),
    "log": (log_func, [1.0, 0.0]),
    "sqrt": (sqrt_func, [1.0, 0.0]),
    "poly2": (poly2_func, [0.0, 1.0, 0.0]),
}


def find_latest_csv(input_dir: str, suffix: str) -> Optional[str]:
    files = glob.glob(os.path.join(input_dir, f"*{suffix}"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def fit_best_curve(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray], float]:
    if not HAS_SCIPY:
        return None, None, float("-inf")

    best_name, best_params, best_r2 = None, None, float("-inf")
    for name, (func, p0) in FIT_FUNCTIONS.items():
        try:
            params, _ = curve_fit(func, x, y, p0=p0, maxfev=20000)
            y_hat = func(x, *params)
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if np.isfinite(r2) and r2 > best_r2:
                best_name, best_params, best_r2 = name, params, r2
        except Exception:
            continue

    return best_name, best_params, best_r2


def plot_metric(raw_df: pd.DataFrame,
                agg_df: pd.DataFrame,
                y_raw_col: str,
                y_agg_col: str,
                y_std_col: str,
                y_label: str,
                title: str,
                output_path: str) -> Dict:
    x_raw = raw_df["actual_tokens"].values if "actual_tokens" in raw_df.columns else raw_df["target_tokens"].values
    x_agg = agg_df["avg_actual_tokens"].values if "avg_actual_tokens" in agg_df.columns else agg_df["target_tokens"].values

    y_raw = raw_df[y_raw_col].values
    y_agg = agg_df[y_agg_col].values
    y_std = agg_df[y_std_col].values if y_std_col in agg_df.columns else np.zeros_like(y_agg)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(x_raw, y_raw, alpha=0.25, s=16, label="Raw samples")
    ax.errorbar(x_agg, y_agg, yerr=y_std, fmt="o", capsize=4, markersize=6, label="Mean ± Std")

    fit_name, fit_params, fit_r2 = fit_best_curve(x_agg, y_agg)
    if fit_name is not None:
        func = FIT_FUNCTIONS[fit_name][0]
        x_smooth = np.linspace(np.min(x_agg), np.max(x_agg), 300)
        y_smooth = func(x_smooth, *fit_params)
        ax.plot(x_smooth, y_smooth, "r--", linewidth=2, label=f"{fit_name} fit (R²={fit_r2:.4f})")

    ax.set_xlabel("Input Tokens")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close(fig)

    return {
        "fit_function": fit_name,
        "fit_params": fit_params.tolist() if fit_params is not None else None,
        "r2": fit_r2 if fit_name is not None else None,
        "output": output_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefill 离线建模绘图")
    parser.add_argument("--input-dir", default="results/prefill_modeling", help="实验结果目录")
    parser.add_argument("--output-dir", default=None, help="图片输出目录，默认 input-dir/images_v2")
    parser.add_argument("--raw-file", default=None, help="指定 raw csv")
    parser.add_argument("--agg-file", default=None, help="指定 aggregated csv")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "images_v2")
    os.makedirs(output_dir, exist_ok=True)

    raw_file = args.raw_file or find_latest_csv(args.input_dir, "_raw.csv")
    agg_file = args.agg_file or find_latest_csv(args.input_dir, "_aggregated.csv")

    if raw_file is None or agg_file is None:
        raise FileNotFoundError("未找到 raw/aggregated csv，请先运行 run_prefill_modeling.py")

    raw_df = pd.read_csv(raw_file)
    agg_df = pd.read_csv(agg_file)

    summary = {
        "raw_file": raw_file,
        "agg_file": agg_file,
        "plots": {
            "power": plot_metric(
                raw_df, agg_df,
                y_raw_col="avg_power_w", y_agg_col="avg_power_w", y_std_col="std_power_w",
                y_label="Prefill Average Power (W)",
                title="Input Tokens vs Prefill Power",
                output_path=os.path.join(output_dir, "prefill_power_vs_tokens.png"),
            ),
            "energy": plot_metric(
                raw_df, agg_df,
                y_raw_col="total_energy_j", y_agg_col="avg_energy_j", y_std_col="std_energy_j",
                y_label="Prefill Energy (J)",
                title="Input Tokens vs Prefill Energy",
                output_path=os.path.join(output_dir, "prefill_energy_vs_tokens.png"),
            ),
            "ttft": plot_metric(
                raw_df, agg_df,
                y_raw_col="ttft_ms", y_agg_col="avg_ttft_ms", y_std_col="std_ttft_ms",
                y_label="TTFT (ms)",
                title="Input Tokens vs TTFT",
                output_path=os.path.join(output_dir, "prefill_ttft_vs_tokens.png"),
            ),
        },
    }

    summary_path = os.path.join(output_dir, "fit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"raw: {raw_file}")
    print(f"agg: {agg_file}")
    print(f"images & summary saved to: {output_dir}")


if __name__ == "__main__":
    main()
