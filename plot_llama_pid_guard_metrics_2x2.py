#!/usr/bin/env python3
"""
Generate a paper-style 2x2 bar figure for the Llama feedforward+PID result.
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
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


ROOT = Path("experiment_results/feedforward")
DEFAULT_RESULT_DIR = ROOT / "llama_pid_guard_out100_r50x3"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_DIR / "images_bar"
BASELINE = "baseline_350w"
STRATEGY = "ff_decode_tbt_guarded_pid"
OUTPUT_LENGTH = 100
QUERY_COUNTS = [8, 16, 32, 64, 96, 128]
FIGURE_STEM = "llama_pid_guard_out100_metrics_2x2"

AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 10
PANEL_LABEL_SIZE = 12

METRICS = [
    ("energy_saving_pct", "(a)", "Energy Saving (%)"),
    ("tbt_increase_pct", "(b)", "TBT Increase (%)"),
    ("ttft_increase_pct", "(c)", "TTFT Increase (%)"),
    ("e2e_increase_pct", "(d)", "E2E Increase (%)"),
]


def configure_matplotlib() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_candidates = [
        name
        for name in [
            "Times New Roman",
            "SimSun",
            "Songti SC",
            "Noto Serif CJK SC",
            "DejaVu Serif",
        ]
        if name in available_fonts
    ]
    if not font_candidates:
        font_candidates = ["DejaVu Serif"]

    plt.rcParams.update(
        {
            "figure.dpi": 600,
            "savefig.dpi": 600,
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
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def latest_aggregated_csv(result_dir: Path) -> Path:
    files = sorted(result_dir.glob("*_aggregated.csv"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No aggregated CSV found under {result_dir}")
    return files[-1]


def geometric_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values or any(value <= 0 for value in values):
        return float("nan")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def validate_input(df: pd.DataFrame, source: Path | None = None) -> None:
    required_columns = {
        "full_repeat",
        "strategy",
        "query_count",
        "output_length",
        "avg_ttft_ms",
        "avg_tbt_ms",
        "avg_e2e_ms",
        "avg_energy_j",
    }
    missing_columns = required_columns - set(df.columns)
    location = str(source) if source is not None else "input dataframe"
    if missing_columns:
        raise ValueError(f"{location} missing columns: {sorted(missing_columns)}")

    subset = df[
        (df["strategy"].isin([BASELINE, STRATEGY]))
        & (df["output_length"].astype(int) == OUTPUT_LENGTH)
    ].copy()
    strategies = set(subset["strategy"].unique())
    missing_strategies = {BASELINE, STRATEGY} - strategies
    if missing_strategies:
        raise ValueError(f"{location} missing strategies: {sorted(missing_strategies)}")

    for strategy in [BASELINE, STRATEGY]:
        q_values = sorted(subset[subset["strategy"] == strategy]["query_count"].astype(int).unique())
        if q_values != QUERY_COUNTS:
            raise ValueError(f"{location} {strategy} query counts are {q_values}; expected {QUERY_COUNTS}")

    repeat_values = sorted(subset["full_repeat"].astype(int).unique())
    if repeat_values != [1, 2, 3]:
        raise ValueError(f"{location} full repeats are {repeat_values}; expected [1, 2, 3]")


def compute_metric_summary(df: pd.DataFrame, source: Path | None = None) -> pd.DataFrame:
    validate_input(df, source)
    subset = df[
        (df["strategy"].isin([BASELINE, STRATEGY]))
        & (df["output_length"].astype(int) == OUTPUT_LENGTH)
    ].copy()
    subset["full_repeat"] = subset["full_repeat"].astype(int)
    subset["query_count"] = subset["query_count"].astype(int)

    baseline = subset[subset["strategy"] == BASELINE].set_index(["full_repeat", "query_count"])
    strategy = subset[subset["strategy"] == STRATEGY].set_index(["full_repeat", "query_count"])
    repeat_rows: list[dict[str, object]] = []

    for repeat in [1, 2, 3]:
        energy_ratios = []
        tbt_ratios = []
        ttft_ratios = []
        e2e_ratios = []
        for query_count in QUERY_COUNTS:
            baseline_row = baseline.loc[(repeat, query_count)]
            strategy_row = strategy.loc[(repeat, query_count)]
            energy_ratio = strategy_row["avg_energy_j"] / baseline_row["avg_energy_j"]
            tbt_ratio = strategy_row["avg_tbt_ms"] / baseline_row["avg_tbt_ms"]
            ttft_ratio = strategy_row["avg_ttft_ms"] / baseline_row["avg_ttft_ms"]
            e2e_ratio = strategy_row["avg_e2e_ms"] / baseline_row["avg_e2e_ms"]

            energy_ratios.append(energy_ratio)
            tbt_ratios.append(tbt_ratio)
            ttft_ratios.append(ttft_ratio)
            e2e_ratios.append(e2e_ratio)
            repeat_rows.append(
                {
                    "full_repeat": repeat,
                    "query_label": str(query_count),
                    "query_sort": QUERY_COUNTS.index(query_count),
                    "energy_saving_pct": (1.0 - energy_ratio) * 100.0,
                    "tbt_increase_pct": (tbt_ratio - 1.0) * 100.0,
                    "ttft_increase_pct": (ttft_ratio - 1.0) * 100.0,
                    "e2e_increase_pct": (e2e_ratio - 1.0) * 100.0,
                }
            )

        repeat_rows.append(
            {
                "full_repeat": repeat,
                "query_label": "GEOMEAN",
                "query_sort": len(QUERY_COUNTS),
                "energy_saving_pct": (1.0 - geometric_mean(energy_ratios)) * 100.0,
                "tbt_increase_pct": (geometric_mean(tbt_ratios) - 1.0) * 100.0,
                "ttft_increase_pct": (geometric_mean(ttft_ratios) - 1.0) * 100.0,
                "e2e_increase_pct": (geometric_mean(e2e_ratios) - 1.0) * 100.0,
            }
        )

    repeat_df = pd.DataFrame(repeat_rows)
    summary = (
        repeat_df.groupby(["query_label", "query_sort"], sort=False)[
            ["energy_saving_pct", "tbt_increase_pct", "ttft_increase_pct", "e2e_increase_pct"]
        ]
        .mean()
        .reset_index()
        .sort_values("query_sort")
        .drop(columns=["query_sort"])
        .reset_index(drop=True)
    )
    return summary


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.22)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("black")
    ax.tick_params(axis="both", which="major", direction="in", length=4.5, width=1.0, top=False, right=False)
    ax.tick_params(axis="y", which="minor", direction="in", length=2.5, width=0.8, right=False)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))


def set_bar_ylim(ax: plt.Axes, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    if data_min < 0:
        bottom = data_min * 1.18
        top = data_max * 1.18 if data_max > 0 else 0.5
        ax.axhline(0, color="black", linewidth=0.8)
    else:
        bottom = 0.0
        top = data_max * 1.18 if data_max > 0 else 1.0
    if top <= bottom:
        top = bottom + 1.0
    ax.set_ylim(bottom=bottom, top=top)


def plot_metrics_2x2(
    summary: pd.DataFrame,
    output_dir: Path,
    figure_stem: str = FIGURE_STEM,
) -> list[Path]:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    x_positions = np.array([0, 1, 2, 3, 4, 5, 6.65], dtype=float)
    x_labels = ["8", "16", "32", "64", "96", "128", "GEOMEAN"]
    query_color = "#4C78A8"
    geomean_color = "#7F7F7F"
    colors = [query_color] * 6 + [geomean_color]
    hatches = ["/"] * 6 + ["x"]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.05), constrained_layout=True)
    fig.supxlabel("Query Count", fontsize=AXIS_LABEL_SIZE)
    axes_flat = axes.ravel()
    for ax, (metric, panel_label, ylabel) in zip(axes_flat, METRICS):
        values = summary[metric].to_numpy(dtype=float)
        bars = ax.bar(
            x_positions,
            values,
            width=0.62,
            color=colors,
            edgecolor="black",
            linewidth=0.85,
            zorder=3,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.axvline(5.82, color="#B0B0B0", linewidth=0.65, linestyle="--", alpha=0.55, zorder=1)
        ax.set_xlim(-0.6, 7.2)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(ylabel)
        ax.text(
            0.02,
            0.96,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=PANEL_LABEL_SIZE,
        )
        style_axis(ax)
        set_bar_ylim(ax, values)

    outputs = [
        output_dir / f"{figure_stem}.png",
        output_dir / f"{figure_stem}.pdf",
        output_dir / f"{figure_stem}.svg",
    ]
    for path in outputs:
        fig.savefig(path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return outputs


def write_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{FIGURE_STEM}_summary.csv"
    summary.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--csv", type=Path, default=None, help="Aggregated CSV path. Defaults to latest in input dir.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv if args.csv is not None else latest_aggregated_csv(args.input_dir)
    df = pd.read_csv(csv_path)
    summary = compute_metric_summary(df, csv_path)
    outputs = plot_metrics_2x2(summary, args.output_dir)
    summary_path = write_summary(summary, args.output_dir)
    print(f"Read: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    for path in outputs:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
