#!/usr/bin/env python3
"""
Generate image_bar-style decode strategy figures for retry q8/q16 results.
"""
from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


ROOT = Path("experiment_results/decode_strategy")
Q8_CSV = ROOT / "strategy_evaluation_policy_retry_q8/decode_strategy_eval_1777989123_aggregated.csv"
Q16_CSV = ROOT / "strategy_evaluation_policy_retry_q16/decode_strategy_eval_1778006288_aggregated.csv"
DEFAULT_OUTPUT_DIR = ROOT / "strategy_evaluation_policy_retry_merged/images_image_bar"

BASELINE = "baseline_350w"
STRATEGY_ORDER = [
    BASELINE,
    "scheme1_fit_curve",
    "scheme2_fit_plus",
    "scheme3_kv_guided",
]
RELATIVE_STRATEGY_ORDER = [strategy for strategy in STRATEGY_ORDER if strategy != BASELINE]
STRATEGY_LABELS = {
    BASELINE: "350W",
    "scheme1_fit_curve": "150/151/191/210/220W",
    "scheme2_fit_plus": "150/170/200/220/230W",
    "scheme3_kv_guided": "190/205/210W",
}
STRATEGY_COLORS = {
    BASELINE: "#7F7F7F",
    "scheme1_fit_curve": "#4C78A8",
    "scheme2_fit_plus": "#F58518",
    "scheme3_kv_guided": "#54A24B",
}
STRATEGY_HATCHES = {
    BASELINE: ".",
    "scheme1_fit_curve": "///",
    "scheme2_fit_plus": "\\\\",
    "scheme3_kv_guided": "xx",
}

ABSOLUTE_METRICS = [
    ("avg_tbt_ms", "Average TBT (ms)", "decode_strategy_tbt"),
    ("avg_ttft_ms", "Average TTFT (ms)", "decode_strategy_ttft"),
    ("avg_e2e_ms", "Average E2E (ms)", "decode_strategy_e2e"),
    ("avg_energy_j", "Average Energy (J)", "decode_strategy_energy"),
    ("avg_power_w", "Average Power (W)", "decode_strategy_power"),
]
RELATIVE_METRICS = [
    ("energy_saving_pct", "Energy Saving (%)", "decode_strategy_energy_saving"),
    ("tbt_loss_pct", "TBT Loss (%)", "decode_strategy_tbt_loss"),
]

AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 9
LEGEND_LABEL_SIZE = 8
PLOT_DPI = 600


def configure_matplotlib() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_candidates = [
        name
        for name in ["Times New Roman", "SimSun", "Songti SC", "Noto Serif CJK SC", "DejaVu Serif"]
        if name in available_fonts
    ]
    if not font_candidates:
        font_candidates = ["DejaVu Serif"]

    plt.rcParams.update(
        {
            "figure.dpi": PLOT_DPI,
            "savefig.dpi": PLOT_DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.1,
            "font.family": font_candidates,
            "font.serif": font_candidates,
            "axes.unicode_minus": False,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_LABEL_SIZE,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_decode_results(csv_paths: Iterable[Path]) -> pd.DataFrame:
    """读取 q=8/q=16 的聚合结果，并验证绘图需要的核心列。"""
    frames = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    required_columns = {
        "full_repeat",
        "strategy",
        "output_length",
        "concurrency",
        "power_limit",
        "avg_ttft_ms",
        "avg_tbt_ms",
        "avg_e2e_ms",
        "avg_energy_j",
        "avg_power_w",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Decode CSV missing columns: {sorted(missing_columns)}")
    return df


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["strategy", "concurrency", "output_length"], as_index=False)
        .agg(
            power_limit=("power_limit", "mean"),
            avg_ttft_ms=("avg_ttft_ms", "mean"),
            avg_tbt_ms=("avg_tbt_ms", "mean"),
            avg_e2e_ms=("avg_e2e_ms", "mean"),
            avg_energy_j=("avg_energy_j", "mean"),
            avg_power_w=("avg_power_w", "mean"),
        )
    )
    summary["strategy"] = pd.Categorical(summary["strategy"], categories=STRATEGY_ORDER, ordered=True)
    return summary.sort_values(["strategy", "concurrency", "output_length"]).reset_index(drop=True)


def geometric_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if float(value) > 0]
    if not values:
        return float("nan")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def category_labels(summary: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in (
        summary[["concurrency", "output_length"]]
        .drop_duplicates()
        .sort_values(["concurrency", "output_length"])
        .iterrows()
    ):
        labels.append(f"{int(row['concurrency'])}/{int(row['output_length'])}")
    return labels


def x_positions_for_labels(labels: list[str], include_geomean: bool = False) -> np.ndarray:
    count = len(labels) + (1 if include_geomean else 0)
    return np.arange(count, dtype=float)


def compute_relative_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """按同一 concurrency/output_length 下的 baseline 计算节能率和 TBT 增幅。"""
    baseline = summary[summary["strategy"] == BASELINE].set_index(["concurrency", "output_length"])
    rows: list[dict[str, object]] = []
    for strategy in RELATIVE_STRATEGY_ORDER:
        strategy_df = summary[summary["strategy"] == strategy].set_index(["concurrency", "output_length"])
        tbt_ratios = []
        energy_ratios = []
        for key in sorted(set(strategy_df.index.tolist()) & set(baseline.index.tolist())):
            baseline_row = baseline.loc[key]
            row = strategy_df.loc[key]
            tbt_ratio = float(row["avg_tbt_ms"] / baseline_row["avg_tbt_ms"])
            energy_ratio = float(row["avg_energy_j"] / baseline_row["avg_energy_j"])
            tbt_ratios.append(tbt_ratio)
            energy_ratios.append(energy_ratio)
            concurrency, output_length = key
            label = f"{int(concurrency)}/{int(output_length)}"
            rows.append(
                {
                    "strategy": strategy,
                    "metric": "tbt_loss_pct",
                    "label": label,
                    "concurrency": int(concurrency),
                    "output_length": int(output_length),
                    "value": (tbt_ratio - 1.0) * 100.0,
                }
            )
            rows.append(
                {
                    "strategy": strategy,
                    "metric": "energy_saving_pct",
                    "label": label,
                    "concurrency": int(concurrency),
                    "output_length": int(output_length),
                    "value": (1.0 - energy_ratio) * 100.0,
                }
            )

        rows.append(
            {
                "strategy": strategy,
                "metric": "tbt_loss_pct",
                "label": "GEOMEAN",
                "concurrency": 10**9,
                "output_length": 10**9,
                "value": (geometric_mean(tbt_ratios) - 1.0) * 100.0,
            }
        )
        rows.append(
            {
                "strategy": strategy,
                "metric": "energy_saving_pct",
                "label": "GEOMEAN",
                "concurrency": 10**9,
                "output_length": 10**9,
                "value": (1.0 - geometric_mean(energy_ratios)) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes, values: np.ndarray, y_zero_floor: bool = True) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.22)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("black")
    ax.tick_params(axis="x", which="major", length=0, width=0, top=False)
    ax.tick_params(axis="x", which="minor", length=0, width=0, top=False)
    ax.tick_params(axis="y", which="major", direction="in", length=4.5, width=1.0, right=False)
    ax.tick_params(axis="y", which="minor", direction="in", length=2.5, width=0.8, right=False)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    if y_zero_floor and data_min >= -0.1:
        bottom = 0.0
        top = data_max * 1.1 if data_max > 0 else 1.0
    elif data_min < 0:
        bottom = data_min * 1.12
        top = data_max * 1.1 if data_max > 0 else 0.5
        ax.axhline(0.0, color="black", linewidth=0.75, zorder=1)
    else:
        bottom = 0.0
        top = data_max * 1.1 if data_max > 0 else 1.0
    if top <= bottom:
        top = bottom + 1.0
    ax.set_ylim(bottom, top)


def legend_handles(strategies: list[str]) -> list[Patch]:
    return [
        Patch(
            facecolor=STRATEGY_COLORS[strategy],
            edgecolor="black",
            hatch=STRATEGY_HATCHES[strategy],
            label=STRATEGY_LABELS[strategy],
            linewidth=0.85,
        )
        for strategy in strategies
    ]


def add_legend(ax: plt.Axes, strategies: list[str], ncol: int = 2) -> None:
    ax.legend(
        handles=legend_handles(strategies),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        borderaxespad=0.2,
        columnspacing=0.9,
        handlelength=2.3,
        handleheight=1.05,
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / f"{stem}.png",
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.svg",
    ]
    for output_path in outputs:
        fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def plot_absolute_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_stem: str,
    output_dir: Path,
) -> list[Path]:
    configure_matplotlib()
    labels = category_labels(summary)
    x_positions = x_positions_for_labels(labels)
    width = 0.18
    offsets = (np.arange(len(STRATEGY_ORDER)) - (len(STRATEGY_ORDER) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(7.2, 3.45), constrained_layout=True)
    all_values = []

    for strategy_idx, strategy in enumerate(STRATEGY_ORDER):
        strategy_df = summary[summary["strategy"] == strategy].copy()
        strategy_df["label"] = strategy_df.apply(
            lambda row: f"{int(row['concurrency'])}/{int(row['output_length'])}",
            axis=1,
        )
        values = strategy_df.set_index("label").loc[labels][metric].to_numpy(dtype=float)
        all_values.extend(values.tolist())
        bars = ax.bar(
            x_positions + offsets[strategy_idx],
            values,
            width=width,
            color=STRATEGY_COLORS[strategy],
            edgecolor="black",
            linewidth=0.85,
            zorder=3,
        )
        for bar in bars:
            bar.set_hatch(STRATEGY_HATCHES[strategy])

    ax.set_xlim(x_positions[0] - 0.65, x_positions[-1] + 0.65)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Number of Query / Length of Generation")
    ax.set_ylabel(ylabel)
    add_legend(ax, STRATEGY_ORDER, ncol=4)
    style_axis(ax, np.array(all_values, dtype=float), y_zero_floor=True)
    return save_figure(fig, output_dir, output_stem)


def plot_relative_metric(
    relative: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_stem: str,
    output_dir: Path,
) -> list[Path]:
    """绘制 decode 策略相对指标图，legend/hatch/GEOMEAN 均按最终论文版样式处理。"""
    configure_matplotlib()
    metric_df = relative[relative["metric"] == metric].copy()
    labels = (
        metric_df[metric_df["label"] != "GEOMEAN"]
        .drop_duplicates(["label", "concurrency", "output_length"])
        .sort_values(["concurrency", "output_length"])["label"]
        .tolist()
    )
    labels.append("GEOMEAN")
    x_positions = x_positions_for_labels(labels[:-1], include_geomean=True)
    width = 0.22
    offsets = (np.arange(len(RELATIVE_STRATEGY_ORDER)) - (len(RELATIVE_STRATEGY_ORDER) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(7.2, 3.45), constrained_layout=True)
    all_values = []

    for strategy_idx, strategy in enumerate(RELATIVE_STRATEGY_ORDER):
        strategy_df = metric_df[metric_df["strategy"] == strategy].set_index("label")
        values = strategy_df.loc[labels]["value"].to_numpy(dtype=float)
        all_values.extend(values.tolist())
        bars = ax.bar(
            x_positions + offsets[strategy_idx],
            values,
            width=width,
            color=STRATEGY_COLORS[strategy],
            edgecolor="black",
            linewidth=0.85,
            zorder=3,
        )
        for bar in bars:
            bar.set_hatch(STRATEGY_HATCHES[strategy])

    ax.set_xlim(x_positions[0] - 0.65, x_positions[-1] + 0.65)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Number of Query / Length of Generation")
    ax.set_ylabel(ylabel)
    add_legend(ax, RELATIVE_STRATEGY_ORDER, ncol=3)
    style_axis(ax, np.array(all_values, dtype=float), y_zero_floor=(metric == "energy_saving_pct"))
    return save_figure(fig, output_dir, output_stem)


def write_report(summary: pd.DataFrame, relative: pd.DataFrame, output_dir: Path) -> Path:
    report_path = output_dir / "decode_strategy_report.md"
    lines = [
        "# Decode Strategy Evaluation Report",
        "",
        f"- Source CSVs: `{Q8_CSV}`, `{Q16_CSV}`",
        f"- Output lengths: {sorted(summary['output_length'].unique().tolist())}",
        f"- Concurrency values: {sorted(summary['concurrency'].unique().tolist())}",
        "",
        "## GEOMEAN Relative Metrics",
        "",
        "| Strategy | Energy Saving (%) | TBT Loss (%) |",
        "|---|---:|---:|",
    ]
    for strategy in RELATIVE_STRATEGY_ORDER:
        energy = relative[
            (relative["strategy"] == strategy)
            & (relative["metric"] == "energy_saving_pct")
            & (relative["label"] == "GEOMEAN")
        ]["value"].iloc[0]
        tbt = relative[
            (relative["strategy"] == strategy)
            & (relative["metric"] == "tbt_loss_pct")
            & (relative["label"] == "GEOMEAN")
        ]["value"].iloc[0]
        lines.append(f"| {STRATEGY_LABELS[strategy]} | {energy:.2f} | {tbt:.2f} |")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_all(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    df = load_decode_results([Q8_CSV, Q16_CSV])
    summary = aggregate_metrics(df)
    relative = compute_relative_summary(summary)
    outputs: list[Path] = []
    for metric, ylabel, stem in ABSOLUTE_METRICS:
        outputs.extend(plot_absolute_metric(summary, metric, ylabel, stem, output_dir))
    for metric, ylabel, stem in RELATIVE_METRICS:
        outputs.extend(plot_relative_metric(relative, metric, ylabel, stem, output_dir))
    summary.to_csv(output_dir / "decode_strategy_summary.csv", index=False)
    relative.to_csv(output_dir / "decode_strategy_relative_summary.csv", index=False)
    write_report(summary, relative, output_dir)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_all(args.output_dir)
    print(f"Wrote figures to: {args.output_dir}")
    for output in outputs:
        print(f"Wrote figure: {output}")


if __name__ == "__main__":
    main()
