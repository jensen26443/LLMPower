#!/usr/bin/env python3
"""
Generate paper-ready 2x2 metric figures for pure feedforward and feedforward+PID.
"""
from __future__ import annotations

import math
import os
import tempfile
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


ROOT = Path("experiment_results/feedforward")
OUTPUT_DIR = ROOT / "paper_figures_ff_vs_pid_separate_bigger"
NO_ERROR_OUTPUT_DIR = ROOT / "paper_figures_ff_vs_pid_separate_no_errorbars_bigger"
QUERY_COUNTS = [8, 16, 32, 64, 96, 128]
BASELINE = "baseline_350w"
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 12
PANEL_LABEL_SIZE = 14

METRICS = [
    ("energy_saving_pct", "Energy Saving (%)"),
    ("tbt_increase_pct", "TBT Increase (%)"),
    ("ttft_increase_pct", "TTFT Increase (%)"),
    ("e2e_increase_pct", "E2E Increase (%)"),
]


@dataclass(frozen=True)
class FigureSpec:
    key: str
    result_dir: Path
    strategy: str
    output_length: int
    color: str
    csv_name: str | None = None


FIGURES = [
    FigureSpec(
        key="pure_ff_out100",
        result_dir=ROOT / "final_guarded_out100_r50x3",
        strategy="ff_decode_tbt_guarded",
        output_length=100,
        color="#4C78A8",
    ),
    FigureSpec(
        key="pure_ff_out200",
        result_dir=ROOT / "final_guarded_out200_r50x3_retry",
        strategy="ff_decode_tbt_guarded",
        output_length=200,
        color="#4C78A8",
    ),
    FigureSpec(
        key="ff_pid_out100",
        result_dir=ROOT / "pid_guard_energy_first_out100_r50x3",
        strategy="ff_decode_tbt_guarded_pid",
        output_length=100,
        color="#D65F5F",
    ),
    FigureSpec(
        key="ff_pid_out200",
        result_dir=ROOT / "pid_guard_energy_first_out200_r50x3/images_q64_ttft_outliers_removed",
        strategy="ff_decode_tbt_guarded_pid",
        output_length=200,
        color="#D65F5F",
        csv_name="feedforward_eval_1777865373_aggregated_q64_ttft_le900.csv",
    ),
]


def configure_matplotlib() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_candidates = [
        name
        for name in ["Times New Roman", "SimSun", "Noto Serif CJK SC", "Source Han Serif SC", "DejaVu Serif"]
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
            "axes.linewidth": 0.8,
            "font.family": font_candidates,
            "font.serif": font_candidates,
            "axes.unicode_minus": False,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def latest_aggregated_csv(result_dir: Path, csv_name: str | None = None) -> Path:
    if csv_name is not None:
        path = result_dir / csv_name
        if not path.exists():
            raise FileNotFoundError(f"Expected aggregated CSV not found: {path}")
        return path
    files = sorted(result_dir.glob("*_aggregated.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No aggregated CSV found under {result_dir}")
    return files[-1]


def geometric_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values or any(value <= 0 for value in values):
        return float("nan")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def validate_input(df: pd.DataFrame, spec: FigureSpec, source: Path) -> None:
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
    if missing_columns:
        raise ValueError(f"{source} missing columns: {sorted(missing_columns)}")

    strategies = set(df["strategy"].unique())
    expected_strategies = {BASELINE, spec.strategy}
    if not expected_strategies.issubset(strategies):
        raise ValueError(
            f"{source} missing strategies {sorted(expected_strategies - strategies)}; "
            f"found {sorted(strategies)}"
        )

    subset = df[
        (df["strategy"].isin(expected_strategies))
        & (df["output_length"].astype(int) == spec.output_length)
    ]
    for strategy in expected_strategies:
        q_values = sorted(subset[subset["strategy"] == strategy]["query_count"].astype(int).unique())
        if q_values != QUERY_COUNTS:
            raise ValueError(f"{source} {strategy} query counts are {q_values}, expected {QUERY_COUNTS}")
    repeat_values = sorted(subset["full_repeat"].astype(int).unique())
    if repeat_values != [1, 2, 3]:
        raise ValueError(f"{source} full repeats are {repeat_values}, expected [1, 2, 3]")


def compute_metrics(df: pd.DataFrame, spec: FigureSpec, source: Path) -> pd.DataFrame:
    validate_input(df, spec, source)
    subset = df[
        (df["strategy"].isin([BASELINE, spec.strategy]))
        & (df["output_length"].astype(int) == spec.output_length)
    ].copy()
    subset["full_repeat"] = subset["full_repeat"].astype(int)
    subset["query_count"] = subset["query_count"].astype(int)

    baseline = subset[subset["strategy"] == BASELINE].set_index(["full_repeat", "query_count"])
    strategy = subset[subset["strategy"] == spec.strategy].set_index(["full_repeat", "query_count"])
    rows: list[dict[str, object]] = []

    for repeat in [1, 2, 3]:
        energy_ratios = []
        ttft_ratios = []
        tbt_ratios = []
        e2e_ratios = []
        for query_count in QUERY_COUNTS:
            baseline_row = baseline.loc[(repeat, query_count)]
            strategy_row = strategy.loc[(repeat, query_count)]
            energy_ratio = strategy_row["avg_energy_j"] / baseline_row["avg_energy_j"]
            ttft_ratio = strategy_row["avg_ttft_ms"] / baseline_row["avg_ttft_ms"]
            tbt_ratio = strategy_row["avg_tbt_ms"] / baseline_row["avg_tbt_ms"]
            e2e_ratio = strategy_row["avg_e2e_ms"] / baseline_row["avg_e2e_ms"]
            energy_ratios.append(energy_ratio)
            ttft_ratios.append(ttft_ratio)
            tbt_ratios.append(tbt_ratio)
            e2e_ratios.append(e2e_ratio)
            rows.append(
                {
                    "figure": spec.key,
                    "strategy": spec.strategy,
                    "output_length": spec.output_length,
                    "full_repeat": repeat,
                    "query_count": query_count,
                    "query_label": str(query_count),
                    "energy_saving_pct": (1.0 - energy_ratio) * 100.0,
                    "tbt_increase_pct": (tbt_ratio - 1.0) * 100.0,
                    "ttft_increase_pct": (ttft_ratio - 1.0) * 100.0,
                    "e2e_increase_pct": (e2e_ratio - 1.0) * 100.0,
                }
            )

        rows.append(
            {
                "figure": spec.key,
                "strategy": spec.strategy,
                "output_length": spec.output_length,
                "full_repeat": repeat,
                "query_count": np.nan,
                "query_label": "GEOMEAN",
                "energy_saving_pct": (1.0 - geometric_mean(energy_ratios)) * 100.0,
                "tbt_increase_pct": (geometric_mean(tbt_ratios) - 1.0) * 100.0,
                "ttft_increase_pct": (geometric_mean(ttft_ratios) - 1.0) * 100.0,
                "e2e_increase_pct": (geometric_mean(e2e_ratios) - 1.0) * 100.0,
            }
        )

    repeat_df = pd.DataFrame(rows)
    summary = (
        repeat_df.groupby(["figure", "strategy", "output_length", "query_label"], sort=False)
        .agg(
            query_count=("query_count", "first"),
            energy_saving_pct=("energy_saving_pct", "mean"),
            energy_saving_std=("energy_saving_pct", "std"),
            tbt_increase_pct=("tbt_increase_pct", "mean"),
            tbt_increase_std=("tbt_increase_pct", "std"),
            ttft_increase_pct=("ttft_increase_pct", "mean"),
            ttft_increase_std=("ttft_increase_pct", "std"),
            e2e_increase_pct=("e2e_increase_pct", "mean"),
            e2e_increase_std=("e2e_increase_pct", "std"),
        )
        .reset_index()
    )
    geomean_mask = summary["query_label"] == "GEOMEAN"
    for column in ["energy_saving_std", "tbt_increase_std", "ttft_increase_std", "e2e_increase_std"]:
        summary.loc[geomean_mask, column] = np.nan
    return summary


def clean_axis(ax: plt.Axes, values: np.ndarray, errors: np.ndarray) -> None:
    finite_values = values[np.isfinite(values)]
    finite_errors = errors[np.isfinite(errors)]
    if finite_values.size == 0:
        return
    if finite_errors.size == finite_values.size:
        lower_data = finite_values - finite_errors
        upper_data = finite_values + finite_errors
    else:
        lower_data = finite_values
        upper_data = finite_values
    data_min = float(np.nanmin(lower_data))
    data_max = float(np.nanmax(upper_data))
    span = data_max - data_min
    if span <= 1e-9:
        span = max(abs(data_max), 1.0) * 0.1
    pad = max(span * 0.12, 0.5)
    lower = data_min - pad
    upper = data_max + pad
    if data_min >= 0 and lower < data_min * 0.55:
        lower = max(0.0, data_min - pad)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", direction="in", length=3.5, width=0.75, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="both", which="minor", direction="in", length=2.0, width=0.6)
    ax.tick_params(top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.grid(False)


def plot_figure(
    summary: pd.DataFrame,
    spec: FigureSpec,
    output_dir: Path,
    show_error_bars: bool = True,
    split_geomean_label: bool = False,
) -> None:
    labels = [str(q) for q in QUERY_COUNTS] + ["GEOMEAN"]
    display_labels = [str(q) for q in QUERY_COUNTS] + (["GEO\nMEAN"] if split_geomean_label else ["GEOMEAN"])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), constrained_layout=False)
    axes_flat = axes.ravel()

    for idx, (metric, ylabel) in enumerate(METRICS):
        ax = axes_flat[idx]
        metric_std = metric.replace("_pct", "_std")
        plot_df = summary.set_index("query_label").loc[labels].reset_index()
        values = plot_df[metric].to_numpy(dtype=float)
        errors = plot_df[metric_std].to_numpy(dtype=float)
        yerr = np.where(np.isfinite(errors), errors, 0.0)
        bar_kwargs = {
            "width": 0.62,
            "color": spec.color,
            "edgecolor": "black",
            "linewidth": 0.45,
            "zorder": 2,
        }
        axis_errors = yerr if show_error_bars else np.zeros_like(values)
        if show_error_bars:
            bar_kwargs["yerr"] = yerr
            bar_kwargs["error_kw"] = {
                "elinewidth": 0.65,
                "ecolor": "black",
                "capsize": 2.5,
                "capthick": 0.65,
            }
        ax.bar(x, values, **bar_kwargs)
        ax.axhline(0.0, color="black", linewidth=0.6, zorder=1)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, fontsize=TICK_LABEL_SIZE, rotation=0)
        minor_ticks = (x[:-1] + x[1:]) / 2.0
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xlim(x[0] - 0.6, x[-1] + 0.65)
        ax.set_xlabel("Query Count", fontsize=AXIS_LABEL_SIZE)
        ax.text(
            0.02,
            0.96,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="normal",
        )
        clean_axis(ax, values, axis_errors)

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.985, wspace=0.28, hspace=0.43)
    png_path = output_dir / f"{spec.key}_metrics_2x2.png"
    pdf_path = output_dir / f"{spec.key}_metrics_2x2.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Separate Strategy Metrics Summary",
        "",
        "Values are relative to the baseline_350w rows from the same experiment directory.",
        "",
    ]
    for figure in summary["figure"].drop_duplicates():
        fig_df = summary[summary["figure"] == figure].copy()
        lines.extend([f"## {figure}", ""])
        table = fig_df[
            [
                "query_label",
                "energy_saving_pct",
                "tbt_increase_pct",
                "ttft_increase_pct",
                "e2e_increase_pct",
            ]
        ].copy()
        for column in table.columns[1:]:
            table[column] = table[column].map(lambda value: f"{value:.2f}")
        lines.append(table.to_markdown(index=False))
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures and summary tables.",
    )
    parser.add_argument(
        "--no-error-bars",
        action="store_true",
        help="Draw bars without full-repeat standard deviation error bars.",
    )
    parser.add_argument(
        "--split-geomean-label",
        action="store_true",
        help="Display GEOMEAN as a two-line tick label with extra spacing before it.",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.no_error_bars:
        return NO_ERROR_OUTPUT_DIR
    return OUTPUT_DIR


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in FIGURES:
        csv_path = latest_aggregated_csv(spec.result_dir, spec.csv_name)
        df = pd.read_csv(csv_path)
        summary = compute_metrics(df, spec, csv_path)
        plot_figure(
            summary,
            spec,
            output_dir,
            show_error_bars=not args.no_error_bars,
            split_geomean_label=args.split_geomean_label,
        )
        summaries.append(summary)

    all_summary = pd.concat(summaries, ignore_index=True)
    csv_path = output_dir / "separate_strategy_metrics_summary.csv"
    md_path = output_dir / "separate_strategy_metrics_summary.md"
    all_summary.to_csv(csv_path, index=False)
    write_markdown(all_summary, md_path)
    print(f"Wrote figures and summaries to {output_dir}")


if __name__ == "__main__":
    main()
