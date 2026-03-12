#!/usr/bin/env python3
"""Standalone plotting script for prefill offline modeling results."""

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
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def linear_func(x, a, b):
    return a * x + b


def log_func(x, a, b):
    return a * np.log(x + 1) + b


def sqrt_func(x, a, b):
    return a * np.sqrt(x) + b


def poly2_func(x, a, b, c):
    return a * x**2 + b * x + c


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


def pick_first_existing(df: pd.DataFrame, candidates) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of the candidate columns exist: {candidates}")


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


def plot_metric(
    raw_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    y_raw_col: str,
    y_agg_col: str,
    y_std_col: str,
    y_label: str,
    title: str,
    output_path: str,
) -> Dict:
    x_raw_col = pick_first_existing(raw_df, ["actual_tokens", "actual_input_tokens", "target_tokens", "input_tokens"])
    x_agg_col = pick_first_existing(agg_df, ["avg_actual_tokens", "target_tokens", "input_tokens"])

    x_raw = raw_df[x_raw_col].values
    x_agg = agg_df[x_agg_col].values
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
        ax.plot(x_smooth, y_smooth, "r--", linewidth=2, label=f"{fit_name} fit (R^2={fit_r2:.4f})")

    max_x = np.max(x_agg)
    if max_x <= 500:
        xticks = np.arange(0, max_x + 100, 100)
    elif max_x <= 1000:
        xticks = np.arange(0, max_x + 250, 250)
    else:
        xticks = np.arange(0, max_x + 500, 500)

    ax.set_xticks(xticks)
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
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
    parser = argparse.ArgumentParser(description="Prefill offline modeling plotting")
    parser.add_argument("--input-dir", default="results/prefill_modeling", help="Experiment result directory")
    parser.add_argument("--output-dir", default=None, help="Image output directory, default: input-dir/images_v2")
    parser.add_argument("--raw-file", default=None, help="Explicit raw csv path")
    parser.add_argument("--agg-file", default=None, help="Explicit aggregated csv path")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "images_v2")
    os.makedirs(output_dir, exist_ok=True)

    raw_file = args.raw_file or find_latest_csv(args.input_dir, "_raw.csv")
    agg_file = args.agg_file or find_latest_csv(args.input_dir, "_aggregated.csv")

    if raw_file is None or agg_file is None:
        raise FileNotFoundError("Could not find raw/aggregated csv files. Run run_prefill_modeling.py first.")

    raw_df = pd.read_csv(raw_file)
    agg_df = pd.read_csv(agg_file)

    energy_raw_col = "dynamic_energy_j" if "dynamic_energy_j" in raw_df.columns else "total_energy_j"
    energy_agg_col = "avg_dynamic_energy_j" if "avg_dynamic_energy_j" in agg_df.columns else "avg_energy_j"
    energy_std_col = "std_dynamic_energy_j" if "std_dynamic_energy_j" in agg_df.columns else "std_energy_j"

    summary = {
        "raw_file": raw_file,
        "agg_file": agg_file,
        "metric_columns": {
            "energy_raw_col": energy_raw_col,
            "energy_agg_col": energy_agg_col,
            "energy_std_col": energy_std_col,
        },
        "plots": {
            "ttft": plot_metric(
                raw_df,
                agg_df,
                y_raw_col="ttft_ms",
                y_agg_col="avg_ttft_ms",
                y_std_col="std_ttft_ms",
                y_label="TTFT (ms)",
                title="Input Tokens vs TTFT",
                output_path=os.path.join(output_dir, "prefill_ttft_vs_tokens.png"),
            ),
            "energy": plot_metric(
                raw_df,
                agg_df,
                y_raw_col=energy_raw_col,
                y_agg_col=energy_agg_col,
                y_std_col=energy_std_col,
                y_label="Prefill Dynamic Energy (J)" if energy_raw_col == "dynamic_energy_j" else "Prefill Energy (J)",
                title="Input Tokens vs Prefill Dynamic Energy"
                if energy_raw_col == "dynamic_energy_j"
                else "Input Tokens vs Prefill Energy",
                output_path=os.path.join(output_dir, "prefill_energy_vs_tokens.png"),
            ),
            "peak_power": plot_metric(
                raw_df,
                agg_df,
                y_raw_col="peak_power_w",
                y_agg_col="peak_power_w",
                y_std_col="std_power_w",
                y_label="Prefill Peak Power (W)",
                title="Input Tokens vs Prefill Peak Power",
                output_path=os.path.join(output_dir, "prefill_peak_power_vs_tokens.png"),
            ),
            "avg_power": plot_metric(
                raw_df,
                agg_df,
                y_raw_col="avg_power_w",
                y_agg_col="avg_power_w",
                y_std_col="std_power_w",
                y_label="Prefill Average Power (W)",
                title="Input Tokens vs Prefill Average Power",
                output_path=os.path.join(output_dir, "prefill_avg_power_vs_tokens.png"),
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
