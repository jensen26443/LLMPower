#!/usr/bin/env python3
"""
Regenerate the prefill two-strategy energy/TTFT bar charts in image_bar style.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


RESULT_DIR = Path("experiment_results/prefill_concurrent_evaluation/prefill_strategy_gpu0_r50x3")
SOURCE_IMAGE_DIR = RESULT_DIR / "images_paper_energy_ttft_only"
DEFAULT_SUMMARY_CSV = SOURCE_IMAGE_DIR / "prefill_energy_saving_ttft_increase_summary_two_strategies.csv"
DEFAULT_OUTPUT_DIR = RESULT_DIR / "images_paper_energy_ttft_only_image_bar"

STRATEGY_ORDER = ["200/220/260W", "Token fit"]
LABEL_ORDER = ["8/225", "16/504", "32/1581", "64/2175", "103/6053", "112/11106", "119/20295", "GEOMEAN"]
X_POSITIONS = np.array([0, 1, 2, 3, 4, 5, 6, 7.65], dtype=float)
SERIES_COLORS = {
    "200/220/260W": "#4C78A8",
    "Token fit": "#F58518",
}
SERIES_HATCHES = {
    "200/220/260W": "/",
    "Token fit": "\\",
}
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 10
LEGEND_LABEL_SIZE = 9


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
            "legend.fontsize": LEGEND_LABEL_SIZE,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"strategy", "label", "energy_saving_pct", "ttft_increase_pct"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")

    missing_strategies = [strategy for strategy in STRATEGY_ORDER if strategy not in set(df["strategy"])]
    if missing_strategies:
        raise ValueError(f"{path} missing strategies: {missing_strategies}")

    missing_labels = [label for label in LABEL_ORDER if label not in set(df["label"])]
    if missing_labels:
        raise ValueError(f"{path} missing labels: {missing_labels}")

    df = df.copy()
    df["strategy"] = pd.Categorical(df["strategy"], categories=STRATEGY_ORDER, ordered=True)
    df["label"] = pd.Categorical(df["label"], categories=LABEL_ORDER, ordered=True)
    return df.sort_values(["label", "strategy"]).reset_index(drop=True)


def style_axis(ax: plt.Axes, values: np.ndarray) -> None:
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

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    if data_min < 0:
        bottom = data_min * 1.18
        top = data_max * 1.18 if data_max > 0 else 0.5
        ax.axhline(0.0, color="black", linewidth=0.75, zorder=1)
    else:
        bottom = 0.0
        top = data_max * 1.18 if data_max > 0 else 1.0
    if top <= bottom:
        top = bottom + 1.0
    ax.set_ylim(bottom, top)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_stem: str,
    output_dir: Path,
) -> list[Path]:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    width = 0.32
    offsets = np.array([-width / 2, width / 2])
    fig, ax = plt.subplots(figsize=(6.9, 2.85), constrained_layout=True)
    all_values = []

    for strategy_idx, strategy in enumerate(STRATEGY_ORDER):
        strategy_df = summary[summary["strategy"] == strategy].set_index("label").loc[LABEL_ORDER]
        values = strategy_df[metric].to_numpy(dtype=float)
        all_values.extend(values.tolist())
        colors = [SERIES_COLORS[strategy]] * len(LABEL_ORDER)
        bars = ax.bar(
            X_POSITIONS + offsets[strategy_idx],
            values,
            width=width,
            color=colors,
            edgecolor="black",
            linewidth=0.85,
            zorder=3,
        )
        for bar in bars:
            bar.set_hatch(SERIES_HATCHES[strategy])

    ax.axvline(7.0, color="#B0B0B0", linewidth=0.65, linestyle="--", alpha=0.55, zorder=1)
    ax.set_xlim(-0.6, 8.25)
    ax.set_xticks(X_POSITIONS)
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_xlabel("Query Count / Target Input Tokens")
    ax.set_ylabel(ylabel)

    legend_handles = [
        Patch(
            facecolor=SERIES_COLORS[strategy],
            edgecolor="black",
            hatch=SERIES_HATCHES[strategy],
            label=strategy,
            linewidth=0.85,
        )
        for strategy in STRATEGY_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=2,
        frameon=False,
        borderaxespad=0.2,
        columnspacing=1.0,
        handlelength=2.4,
        handleheight=1.1,
    )
    style_axis(ax, np.array(all_values, dtype=float))

    outputs = [
        output_dir / f"{output_stem}.png",
        output_dir / f"{output_stem}.pdf",
        output_dir / f"{output_stem}.svg",
    ]
    for output_path in outputs:
        fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary_csv)
    outputs = []
    outputs.extend(
        plot_metric(
            summary,
            "energy_saving_pct",
            "Energy Saving (%)",
            "prefill_energy_saving_two_strategies",
            args.output_dir,
        )
    )
    outputs.extend(
        plot_metric(
            summary,
            "ttft_increase_pct",
            "TTFT Increase (%)",
            "prefill_ttft_increase_two_strategies",
            args.output_dir,
        )
    )
    summary.to_csv(args.output_dir / "prefill_energy_saving_ttft_increase_summary_two_strategies.csv", index=False)
    print(f"Read: {args.summary_csv}")
    print(f"Wrote figures to: {args.output_dir}")
    for output_path in outputs:
        print(f"Wrote figure: {output_path}")


if __name__ == "__main__":
    main()
