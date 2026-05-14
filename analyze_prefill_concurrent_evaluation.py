#!/usr/bin/env python3
"""
并发 prefill-only 评估结果分析脚本。
"""
import argparse
import glob
import json
import math
import os
from typing import Dict, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

PLOT_DPI = 600
USE_PLOT_TITLES = False
LINE_WIDTH = 1.5
AXIS_LINE_WIDTH = 1.0
MARKER_SIZE = 42
PAPER_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#7f7f7f",
]

def pick_available_font(preferred_fonts: Sequence[str], fallback: str) -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            return font_name
    return fallback


EN_FONT = pick_available_font(["Times New Roman", "Liberation Serif"], "DejaVu Serif")
ZH_FONT = pick_available_font(["SimSun", "Noto Serif CJK SC", "Source Han Serif SC"], "DejaVu Sans")

sns.set_style("white")
plt.rcParams["font.family"] = [EN_FONT, ZH_FONT]
plt.rcParams["font.sans-serif"] = [ZH_FONT, "DejaVu Sans"]
plt.rcParams["font.serif"] = [EN_FONT, "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = AXIS_LINE_WIDTH
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False
plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8

PREFERRED_STRATEGY_ORDER = [
    "baseline_350w",
    "prefill_manual_buckets",
    "prefill_token_fit",
    "prefill_token_fit_plus25w",
    "prefill_170w",
    "prefill_200w",
    "prefill_220w",
    "prefill_260w",
]

STRATEGY_POWER_MAP = {
    "baseline_350w": 350,
    "prefill_170w": 170,
    "prefill_200w": 200,
    "prefill_220w": 220,
    "prefill_260w": 260,
}


def linear_func(x, a, b):
    return a * x + b


def log_func(x, a, b):
    return a * np.log(x + 1) + b


def sqrt_func(x, a, b):
    return a * np.sqrt(x) + b


FIT_FUNCTIONS = {
    "linear": linear_func,
    "log": log_func,
    "sqrt": sqrt_func,
}


def write_json_file(file_path: str, payload: Dict):
    with open(file_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def load_latest_aggregated_csv(result_dir: str) -> Optional[pd.DataFrame]:
    pattern = os.path.join(result_dir, "*_aggregated.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"没有找到 aggregated 文件: {pattern}")
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"加载文件: {latest_file}")
    return pd.read_csv(latest_file)


def load_latest_raw_csv(result_dir: str) -> Optional[pd.DataFrame]:
    pattern = os.path.join(result_dir, "*_raw.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"没有找到 raw 文件: {pattern}")
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"加载原始文件: {latest_file}")
    return pd.read_csv(latest_file)


def load_csvs(result_dirs: Sequence[str], loader) -> Optional[pd.DataFrame]:
    frames = []
    for result_dir in result_dirs:
        df = loader(result_dir)
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def get_plot_strategy_order(df: pd.DataFrame):
    present = set(df["strategy"].unique())
    ordered = [name for name in PREFERRED_STRATEGY_ORDER if name in present]
    remainder = sorted(name for name in present if name not in ordered)
    return ordered + remainder


def get_strategy_palette(strategy_order: Sequence[str]) -> Dict[str, str]:
    return {
        strategy: PAPER_COLORS[index % len(PAPER_COLORS)]
        for index, strategy in enumerate(strategy_order)
    }


def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "power_limit" not in normalized.columns:
        normalized["power_limit"] = normalized["strategy"].map(STRATEGY_POWER_MAP)
    if "actual_input_tokens" not in normalized.columns:
        normalized["actual_input_tokens"] = normalized["target_input_tokens"]
    if "avg_e2e_ms" not in normalized.columns:
        normalized["avg_e2e_ms"] = normalized["avg_ttft_ms"]
    if "total_energy_j" not in normalized.columns and "avg_energy_j" in normalized.columns:
        normalized["total_energy_j"] = normalized["avg_energy_j"]
    return normalized


def aggregate_across_full_repeats(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_input_df(df)
    return (
        df.groupby(
            ["strategy", "query_count", "target_input_tokens", "power_limit"],
            as_index=False,
        )
        .agg(
            avg_actual_input_tokens=("actual_input_tokens", "mean"),
            avg_ttft_ms=("avg_ttft_ms", "mean"),
            avg_e2e_ms=("avg_e2e_ms", "mean"),
            avg_energy_j=("avg_energy_j", "mean"),
            avg_power_w=("avg_power_w", "mean"),
        )
        .sort_values(["strategy", "query_count"])
    )


def aggregate_raw_for_fit(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_input_df(df)
    return (
        df.groupby(["strategy", "query_count", "target_input_tokens"], as_index=False)
        .agg(
            avg_actual_input_tokens=("actual_input_tokens", "mean"),
            avg_ttft_ms=("avg_ttft_ms", "mean"),
            avg_e2e_ms=("avg_e2e_ms", "mean"),
            avg_energy_j=("total_energy_j", "mean"),
            avg_power_w=("avg_power_w", "mean"),
            std_ttft_ms=("avg_ttft_ms", "std"),
            std_energy_j=("total_energy_j", "std"),
            std_power_w=("avg_power_w", "std"),
        )
        .fillna(0.0)
        .sort_values(["strategy", "query_count"])
    )


def aggregate_raw_to_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_input_df(df)
    return (
        df.groupby(
            ["strategy", "query_count", "target_input_tokens", "power_limit"],
            as_index=False,
        )
        .agg(
            avg_actual_input_tokens=("actual_input_tokens", "mean"),
            avg_ttft_ms=("avg_ttft_ms", "mean"),
            avg_e2e_ms=("avg_e2e_ms", "mean"),
            avg_energy_j=("total_energy_j", "mean"),
            avg_power_w=("avg_power_w", "mean"),
            num_samples=("batch_repeat", "count"),
        )
        .sort_values(["strategy", "query_count"])
    )


def geometric_mean(values):
    if not values:
        return 0.0
    positive = [float(value) for value in values if float(value) > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def annotate_suspicious_batches(df: pd.DataFrame) -> pd.DataFrame:
    annotated = normalize_input_df(df)
    annotated["group_median_ttft_ms"] = np.nan
    annotated["group_mad_ttft_ms"] = np.nan
    annotated["ttft_deviation_ratio"] = np.nan
    annotated["is_suspicious"] = False
    annotated["suspicious_reason"] = ""

    for (_, _), group in annotated.groupby(["strategy", "query_count"]):
        median_ttft = float(group["avg_ttft_ms"].median())
        mad_ttft = float(np.median(np.abs(group["avg_ttft_ms"] - median_ttft)))
        ttft_threshold = max(6.0 * mad_ttft, median_ttft * 0.35, 120.0)

        reasons_by_index: Dict[int, list[str]] = {}
        for index, row in group.iterrows():
            reasons = []
            deviation = float(abs(row["avg_ttft_ms"] - median_ttft))
            ratio = float(row["avg_ttft_ms"] / median_ttft) if median_ttft > 0 else 1.0

            if deviation > ttft_threshold and (ratio > 1.5 or ratio < 0.6):
                reasons.append("ttft_outlier_vs_group")
            if median_ttft > 0 and row["avg_ttft_ms"] > median_ttft * 3.0:
                reasons.append("extreme_high_ttft")

            annotated.at[index, "group_median_ttft_ms"] = median_ttft
            annotated.at[index, "group_mad_ttft_ms"] = mad_ttft
            annotated.at[index, "ttft_deviation_ratio"] = ratio
            if reasons:
                reasons_by_index[index] = reasons

        for index, reasons in reasons_by_index.items():
            annotated.at[index, "is_suspicious"] = True
            annotated.at[index, "suspicious_reason"] = ",".join(reasons)

    return annotated


def compute_relative_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_input_df(df)
    if df["strategy"].nunique() <= 1 or "baseline_350w" not in set(df["strategy"].unique()):
        return pd.DataFrame(columns=["strategy", "metric", "query_count", "target_input_tokens", "label", "value"])
    baseline_df = (
        df[df["strategy"] == "baseline_350w"]
        .set_index(["query_count", "target_input_tokens"])
        .sort_index()
    )
    rows = []
    for strategy in [name for name in get_plot_strategy_order(df) if name != "baseline_350w"]:
        strategy_df = (
            df[df["strategy"] == strategy]
            .set_index(["query_count", "target_input_tokens"])
            .sort_index()
        )
        shared_keys = sorted(set(strategy_df.index.tolist()) & set(baseline_df.index.tolist()))
        energy_ratios = []
        ttft_ratios = []
        for query_count, target_input_tokens in shared_keys:
            baseline_row = baseline_df.loc[(query_count, target_input_tokens)]
            row = strategy_df.loc[(query_count, target_input_tokens)]
            energy_ratio = row["avg_energy_j"] / baseline_row["avg_energy_j"]
            ttft_ratio = row["avg_ttft_ms"] / baseline_row["avg_ttft_ms"]
            energy_ratios.append(energy_ratio)
            ttft_ratios.append(ttft_ratio)
            rows.extend([
                {
                    "strategy": strategy,
                    "metric": "energy_saving_pct",
                    "query_count": int(query_count),
                    "target_input_tokens": int(target_input_tokens),
                    "label": f"{int(query_count)}/{int(target_input_tokens)}",
                    "value": (1.0 - energy_ratio) * 100.0,
                },
                {
                    "strategy": strategy,
                    "metric": "ttft_increase_pct",
                    "query_count": int(query_count),
                    "target_input_tokens": int(target_input_tokens),
                    "label": f"{int(query_count)}/{int(target_input_tokens)}",
                    "value": (ttft_ratio - 1.0) * 100.0,
                },
            ])
        rows.extend([
            {
                "strategy": strategy,
                "metric": "energy_saving_pct",
                "query_count": None,
                "target_input_tokens": None,
                "label": "GEOMEAN",
                "value": (1.0 - geometric_mean(energy_ratios)) * 100.0,
            },
            {
                "strategy": strategy,
                "metric": "ttft_increase_pct",
                "query_count": None,
                "target_input_tokens": None,
                "label": "GEOMEAN",
                "value": (geometric_mean(ttft_ratios) - 1.0) * 100.0,
            },
        ])
    return pd.DataFrame(rows)


def fit_curve(x: np.ndarray, y: np.ndarray):
    if not HAS_SCIPY or len(x) < 3:
        return None, None, None, None
    best = None
    for name, func in FIT_FUNCTIONS.items():
        try:
            params, _ = curve_fit(func, x, y, maxfev=10000)
            pred = func(x, *params)
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
            if best is None or r2 > best[2]:
                best = (name, params, r2, func)
        except Exception:
            continue
    return best


def recommend_prefill_buckets(df: pd.DataFrame, ttft_threshold_pct: float = 5.0) -> Dict[int, Dict]:
    df = normalize_input_df(df)
    agg_df = aggregate_across_full_repeats(df)
    baseline_df = agg_df[agg_df["strategy"] == "baseline_350w"].set_index("query_count")
    recommendations = {}

    for query_count in sorted(agg_df["query_count"].unique()):
        if query_count not in baseline_df.index:
            continue
        baseline_row = baseline_df.loc[query_count]
        candidates = []
        for _, row in agg_df[(agg_df["query_count"] == query_count) & (agg_df["strategy"] != "baseline_350w")].iterrows():
            ttft_increase_pct = (row["avg_ttft_ms"] / baseline_row["avg_ttft_ms"] - 1.0) * 100.0
            energy_saving_pct = (1.0 - row["avg_energy_j"] / baseline_row["avg_energy_j"]) * 100.0
            candidates.append({
                "strategy": row["strategy"],
                "recommended_power": int(row["power_limit"]),
                "target_input_tokens": int(row["target_input_tokens"]),
                "ttft_increase_pct": float(ttft_increase_pct),
                "energy_saving_pct": float(energy_saving_pct),
            })
        acceptable = [item for item in candidates if item["ttft_increase_pct"] <= ttft_threshold_pct]
        if acceptable:
            best = max(
                acceptable,
                key=lambda item: (item["energy_saving_pct"], -item["ttft_increase_pct"], -item["recommended_power"]),
            )
            best["status"] = "ok"
        else:
            best = {
                "strategy": None,
                "recommended_power": None,
                "target_input_tokens": int(baseline_row["target_input_tokens"]),
                "ttft_increase_pct": float(min(item["ttft_increase_pct"] for item in candidates)) if candidates else None,
                "energy_saving_pct": float(max(item["energy_saving_pct"] for item in candidates)) if candidates else None,
                "status": "unsatisfied",
            }
        recommendations[int(query_count)] = best
    return recommendations


def apply_paper_style(ax,
                      y_zero_floor: bool = True,
                      legend: bool = False,
                      rotate_xticks: bool = False):
    ax.set_title("")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LINE_WIDTH)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=4.0,
        width=AXIS_LINE_WIDTH,
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=2.0,
        width=0.8,
        top=False,
        right=False,
    )
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    if rotate_xticks:
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_ha("right")

    if y_zero_floor:
        bottom, top = ax.get_ylim()
        if bottom >= 0:
            ax.set_ylim(0, top * 1.05 if top > 0 else 1.0)

    if legend:
        legend_obj = ax.legend(frameon=False)
        if legend_obj is not None:
            for line in legend_obj.get_lines():
                line.set_linewidth(LINE_WIDTH)


def save_paper_figure(fig, output_path: str):
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metric(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, output_path: str):
    plot_df = df.copy()
    plot_df["label"] = plot_df.apply(
        lambda row: f"{int(row['query_count'])}/{int(row['target_input_tokens'])}",
        axis=1,
    )
    order = plot_df.sort_values(["query_count"])["label"].drop_duplicates().tolist()
    plot_df["label"] = pd.Categorical(plot_df["label"], categories=order, ordered=True)
    strategy_order = get_plot_strategy_order(plot_df)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.barplot(
        data=plot_df.sort_values("label"),
        x="label",
        y=metric_col,
        hue="strategy",
        hue_order=strategy_order,
        palette=get_strategy_palette(strategy_order),
        ax=ax,
    )
    ax.set_xlabel("Query Count / Target Input Tokens")
    ax.set_ylabel(ylabel)
    apply_paper_style(ax, y_zero_floor=True, legend=True, rotate_xticks=True)
    save_paper_figure(fig, output_path)


def plot_tokens_scatter(raw_df: pd.DataFrame,
                        filtered_df: pd.DataFrame,
                        metric_col: str,
                        title: str,
                        ylabel: str,
                        output_path: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if raw_df is not None and not raw_df.empty:
        ax.scatter(
            raw_df["target_input_tokens"],
            raw_df[metric_col],
            alpha=0.25,
            s=MARKER_SIZE,
            color="#7f7f7f",
            label="Raw",
        )
    if filtered_df is not None and not filtered_df.empty:
        ax.scatter(
            filtered_df["target_input_tokens"],
            filtered_df[metric_col],
            alpha=0.85,
            s=MARKER_SIZE,
            color=PAPER_COLORS[0],
            label="Filtered",
        )
        if HAS_SCIPY and len(filtered_df) >= 3:
            x = filtered_df["target_input_tokens"].to_numpy(dtype=float)
            y = filtered_df[metric_col].to_numpy(dtype=float)
            fit = fit_curve(x, y)
            if fit[0] is not None:
                _, params, _, func = fit
                x_smooth = np.linspace(x.min(), x.max(), 200)
                y_smooth = func(x_smooth, *params)
                ax.plot(x_smooth, y_smooth, color="black", linewidth=LINE_WIDTH, label="Fit")
    ax.set_xlabel("Target Input Tokens")
    ax.set_ylabel(ylabel)
    apply_paper_style(ax, y_zero_floor=False, legend=True)
    save_paper_figure(fig, output_path)


def plot_query_label_scatter(df: pd.DataFrame,
                             metric_col: str,
                             title: str,
                             ylabel: str,
                             output_path: str,
                             color: str = "tab:blue"):
    plot_df = df.copy()
    plot_df["label"] = plot_df.apply(
        lambda row: f"{int(row['query_count'])}/{int(row['target_input_tokens'])}",
        axis=1,
    )
    order = (
        plot_df[["query_count", "label"]]
        .drop_duplicates()
        .sort_values(["query_count"])["label"]
        .tolist()
    )
    x_positions = {label: idx for idx, label in enumerate(order)}

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(
        plot_df["label"].map(x_positions),
        plot_df[metric_col],
        alpha=0.8,
        s=MARKER_SIZE,
        color=color,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Query Count / Target Input Tokens")
    ax.set_ylabel(ylabel)
    apply_paper_style(ax, y_zero_floor=False, rotate_xticks=True)
    save_paper_figure(fig, output_path)


def plot_query_label_overlay_scatter(raw_df: pd.DataFrame,
                                     filtered_df: pd.DataFrame,
                                     metric_col: str,
                                     title: str,
                                     ylabel: str,
                                     output_path: str):
    combined = pd.concat(
        [
            raw_df.assign(series="Raw"),
            filtered_df.assign(series="Filtered"),
        ],
        ignore_index=True,
    )
    combined["label"] = combined.apply(
        lambda row: f"{int(row['query_count'])}/{int(row['target_input_tokens'])}",
        axis=1,
    )
    order = (
        combined[["query_count", "label"]]
        .drop_duplicates()
        .sort_values(["query_count"])["label"]
        .tolist()
    )
    x_positions = {label: idx for idx, label in enumerate(order)}

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    raw_points = combined[combined["series"] == "Raw"]
    filtered_points = combined[combined["series"] == "Filtered"]
    ax.scatter(
        raw_points["label"].map(x_positions),
        raw_points[metric_col],
        alpha=0.25,
        s=MARKER_SIZE,
        color="#7f7f7f",
        label="Raw",
    )
    ax.scatter(
        filtered_points["label"].map(x_positions),
        filtered_points[metric_col],
        alpha=0.85,
        s=MARKER_SIZE,
        color=PAPER_COLORS[0],
        label="Filtered",
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Query Count / Target Input Tokens")
    ax.set_ylabel(ylabel)
    apply_paper_style(ax, y_zero_floor=False, legend=True, rotate_xticks=True)
    save_paper_figure(fig, output_path)


def plot_relative_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: str):
    metric_df = df[df["metric"] == metric].copy()
    order = (
        metric_df[metric_df["label"] != "GEOMEAN"]
        .sort_values(["query_count"])
        ["label"]
        .drop_duplicates()
        .tolist()
    )
    order.append("GEOMEAN")
    metric_df["label"] = pd.Categorical(metric_df["label"], categories=order, ordered=True)
    strategy_order = get_plot_strategy_order(metric_df)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    sns.barplot(
        data=metric_df.sort_values("label"),
        x="label",
        y="value",
        hue="strategy",
        hue_order=strategy_order,
        palette=get_strategy_palette(strategy_order),
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=AXIS_LINE_WIDTH)
    ax.set_xlabel("Query Count / Target Input Tokens")
    ax.set_ylabel(ylabel)
    apply_paper_style(ax, y_zero_floor=False, legend=True, rotate_xticks=True)
    save_paper_figure(fig, output_path)


def plot_fit_curves(raw_df: pd.DataFrame, output_dir: str) -> Dict[str, Dict]:
    fit_results = {}
    metric_configs = [
        ("avg_ttft_ms", "TTFT", "prefill_concurrent_ttft_fit.png"),
        ("avg_energy_j", "Energy (J)", "prefill_concurrent_energy_fit.png"),
        ("avg_power_w", "Power (W)", "prefill_concurrent_power_fit.png"),
    ]

    for metric_col, ylabel, filename in metric_configs:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for strategy in get_plot_strategy_order(raw_df):
            strategy_df = raw_df[raw_df["strategy"] == strategy].sort_values("target_input_tokens")
            x = strategy_df["target_input_tokens"].to_numpy(dtype=float)
            y = strategy_df[metric_col].to_numpy(dtype=float)
            ax.scatter(x, y, alpha=0.45, s=28, label=f"{strategy} raw")
            fit = fit_curve(x, y)
            if fit[0] is not None:
                name, params, r2, func = fit
                x_smooth = np.linspace(x.min(), x.max(), 200)
                y_smooth = func(x_smooth, *params)
                ax.plot(x_smooth, y_smooth, linewidth=LINE_WIDTH, label=f"{strategy} {name} R²={r2:.3f}")
                fit_results.setdefault(strategy, {})[metric_col] = {
                    "function": name,
                    "params": [float(item) for item in params],
                    "r2": float(r2),
                }
        ax.set_xlabel("Target Input Tokens")
        ax.set_ylabel(ylabel)
        apply_paper_style(ax, y_zero_floor=False, legend=True)
        save_paper_figure(fig, os.path.join(output_dir, filename))
    return fit_results


def write_report(agg_df: pd.DataFrame,
                 fit_results: Dict,
                 recommendations: Dict[int, Dict],
                 output_path: str,
                 suspicious_count: int = 0):
    relative_df = compute_relative_plot_df(agg_df)
    lines = [
        "# Concurrent Prefill-Only Evaluation Report",
        "",
        "## Summary",
        f"- Strategy count: {agg_df['strategy'].nunique()}",
        f"- Query counts: {sorted(agg_df['query_count'].unique().tolist())}",
        f"- Suspicious batches filtered: {suspicious_count}",
        "",
        "## Relative To Baseline 350W",
    ]

    for strategy in [name for name in get_plot_strategy_order(agg_df) if name != "baseline_350w"]:
        strategy_rows = relative_df[relative_df["strategy"] == strategy]
        energy_rows = strategy_rows[strategy_rows["metric"] == "energy_saving_pct"]
        ttft_rows = strategy_rows[strategy_rows["metric"] == "ttft_increase_pct"]
        lines.append(
            f"- {strategy}: mean energy saving={energy_rows[energy_rows['label'] != 'GEOMEAN']['value'].mean():.2f}%, "
            f"mean TTFT increase={ttft_rows[ttft_rows['label'] != 'GEOMEAN']['value'].mean():.2f}%"
        )

    lines.extend([
        "",
        "## Recommended Prefill Buckets",
    ])
    for query_count, item in recommendations.items():
        if item["status"] == "ok":
            lines.append(
                f"- q={query_count}, C={item['target_input_tokens']}: "
                f"recommend {item['recommended_power']}W "
                f"(strategy={item['strategy']}, TTFT {item['ttft_increase_pct']:.2f}%, Energy {item['energy_saving_pct']:.2f}%)"
            )
        else:
            lines.append(
                f"- q={query_count}, C={item['target_input_tokens']}: "
                f"unsatisfied within TTFT<={5.0:.1f}% "
                f"(best TTFT {item['ttft_increase_pct']:.2f}%, best Energy {item['energy_saving_pct']:.2f}%)"
            )

    lines.extend([
        "",
        "## Fit Results",
        json.dumps(fit_results, ensure_ascii=False, indent=2),
    ])

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def generate_outputs(agg_df: pd.DataFrame,
                     output_dir: str,
                     raw_df: Optional[pd.DataFrame] = None,
                     ttft_threshold_pct: float = 5.0) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    suspicious_df = pd.DataFrame()
    filtered_raw_df = None

    if raw_df is not None and not raw_df.empty:
        annotated_raw_df = annotate_suspicious_batches(raw_df)
        suspicious_df = annotated_raw_df[annotated_raw_df["is_suspicious"]].copy()
        filtered_raw_df = annotated_raw_df[~annotated_raw_df["is_suspicious"]].copy()
        plot_df = aggregate_raw_to_plot_df(filtered_raw_df)
        fit_df = aggregate_raw_for_fit(filtered_raw_df)
    else:
        plot_df = aggregate_across_full_repeats(agg_df)
        fit_df = aggregate_raw_for_fit(agg_df)

    relative_df = compute_relative_plot_df(plot_df)
    single_strategy_mode = plot_df["strategy"].nunique() == 1
    recommendations = None if single_strategy_mode else recommend_prefill_buckets(plot_df, ttft_threshold_pct=ttft_threshold_pct)

    outputs = {
        "ttft": os.path.join(output_dir, "prefill_concurrent_ttft.png"),
        "energy": os.path.join(output_dir, "prefill_concurrent_energy.png"),
        "power": os.path.join(output_dir, "prefill_concurrent_power.png"),
        "report": os.path.join(output_dir, "prefill_concurrent_report.md"),
        "suspicious_batches": os.path.join(output_dir, "prefill_concurrent_suspicious_batches.csv"),
        "filtered_raw": os.path.join(output_dir, "prefill_concurrent_filtered_raw.csv"),
        "filtered_aggregated": os.path.join(output_dir, "prefill_concurrent_filtered_aggregated.csv"),
        "power_tokens_scatter": os.path.join(output_dir, "prefill_concurrent_power_vs_tokens_scatter.png"),
        "ttft_tokens_scatter": os.path.join(output_dir, "prefill_concurrent_ttft_vs_tokens_scatter.png"),
        "power_query_scatter": os.path.join(output_dir, "prefill_concurrent_power_vs_query_token_scatter.png"),
        "ttft_query_scatter": os.path.join(output_dir, "prefill_concurrent_ttft_vs_query_token_scatter.png"),
        "power_query_scatter_filtered": os.path.join(output_dir, "prefill_concurrent_power_vs_query_token_scatter_filtered.png"),
        "ttft_query_scatter_filtered": os.path.join(output_dir, "prefill_concurrent_ttft_vs_query_token_scatter_filtered.png"),
    }
    if not single_strategy_mode:
        outputs["energy_saving"] = os.path.join(output_dir, "prefill_concurrent_energy_saving.png")
        outputs["ttft_increase"] = os.path.join(output_dir, "prefill_concurrent_ttft_increase.png")
        outputs["recommendation_json"] = os.path.join(output_dir, "prefill_concurrent_recommendation.json")

    plot_metric(plot_df, "avg_ttft_ms", "Concurrent Prefill TTFT", "TTFT (ms)", outputs["ttft"])
    plot_metric(plot_df, "avg_energy_j", "Concurrent Prefill Energy", "Energy (J)", outputs["energy"])
    plot_metric(plot_df, "avg_power_w", "Concurrent Prefill Power", "Power (W)", outputs["power"])
    if raw_df is not None and filtered_raw_df is not None and not filtered_raw_df.empty:
        plot_tokens_scatter(raw_df, filtered_raw_df, "avg_power_w", "Prefill Power vs Target Input Tokens", "Power (W)", outputs["power_tokens_scatter"])
        plot_tokens_scatter(raw_df, filtered_raw_df, "avg_ttft_ms", "Prefill TTFT vs Target Input Tokens", "TTFT (ms)", outputs["ttft_tokens_scatter"])
        plot_query_label_overlay_scatter(raw_df, filtered_raw_df, "avg_power_w", "Prefill Power vs Query Count / Target Input Tokens", "Power (W)", outputs["power_query_scatter"])
        plot_query_label_overlay_scatter(raw_df, filtered_raw_df, "avg_ttft_ms", "Prefill TTFT vs Query Count / Target Input Tokens", "TTFT (ms)", outputs["ttft_query_scatter"])
        plot_query_label_scatter(filtered_raw_df, "avg_power_w", "Prefill Power vs Query Count / Target Input Tokens (Filtered)", "Power (W)", outputs["power_query_scatter_filtered"])
        plot_query_label_scatter(filtered_raw_df, "avg_ttft_ms", "Prefill TTFT vs Query Count / Target Input Tokens (Filtered)", "TTFT (ms)", outputs["ttft_query_scatter_filtered"])
    if not single_strategy_mode:
        plot_relative_metric(relative_df, "energy_saving_pct", "Energy Saving Relative to Baseline", "Energy Saving (%)", outputs["energy_saving"])
        plot_relative_metric(relative_df, "ttft_increase_pct", "TTFT Increase Relative to Baseline", "TTFT Increase (%)", outputs["ttft_increase"])
    fit_results = plot_fit_curves(fit_df, output_dir)
    if suspicious_df.empty:
        pd.DataFrame(columns=["strategy", "query_count", "batch_repeat", "avg_ttft_ms", "avg_power_w", "total_energy_j", "suspicious_reason"]).to_csv(outputs["suspicious_batches"], index=False)
    else:
        suspicious_df.to_csv(outputs["suspicious_batches"], index=False)

    if filtered_raw_df is not None:
        filtered_raw_df.to_csv(outputs["filtered_raw"], index=False)
        plot_df.to_csv(outputs["filtered_aggregated"], index=False)
    else:
        pd.DataFrame(columns=[]).to_csv(outputs["filtered_raw"], index=False)
        plot_df.to_csv(outputs["filtered_aggregated"], index=False)

    if single_strategy_mode:
        with open(outputs["report"], "w", encoding="utf-8") as file_obj:
            file_obj.write(
                "\n".join(
                    [
                        "# Concurrent Prefill-Only Evaluation Report",
                        "",
                        "## Summary",
                        f"- Strategy count: {plot_df['strategy'].nunique()}",
                        f"- Query counts: {sorted(plot_df['query_count'].unique().tolist())}",
                        f"- Suspicious batches filtered: {int(len(suspicious_df))}",
                        "",
                        "## Modeling Focus",
                        "- This run contains only baseline_350w and is intended for prefill modeling.",
                        "- Main relationships are captured by TTFT/Power scatter plots and fitted curves.",
                        "",
                        "## Fit Results",
                        json.dumps(fit_results, ensure_ascii=False, indent=2),
                    ]
                )
            )
    else:
        write_json_file(outputs["recommendation_json"], recommendations)
        write_report(
            plot_df,
            fit_results,
            recommendations,
            outputs["report"],
            suspicious_count=int(len(suspicious_df)),
        )
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze concurrent prefill-only evaluation.")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--result-dirs", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ttft-threshold-pct", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dirs = []
    if args.result_dirs:
        result_dirs.extend([item.strip() for item in args.result_dirs.split(",") if item.strip()])
    if args.result_dir:
        result_dirs.append(args.result_dir)
    if not result_dirs:
        raise ValueError("Please provide --result-dir or --result-dirs")

    agg_df = load_csvs(result_dirs, load_latest_aggregated_csv)
    raw_df = load_csvs(result_dirs, load_latest_raw_csv)
    if agg_df is None or agg_df.empty:
        raise RuntimeError("No aggregated data found")
    outputs = generate_outputs(
        agg_df,
        args.output_dir,
        raw_df=raw_df,
        ttft_threshold_pct=args.ttft_threshold_pct,
    )
    print("生成文件:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
