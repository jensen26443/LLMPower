#!/usr/bin/env python3
"""Generate paper-style decode modeling figures from merged filtered data."""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


DEFAULT_DATA_DIR = Path(
    "experiment_results/decode_modeling/decode_modeling/merged_filtered"
)
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "paper_figures"

FIG_DPI = 600
FONT_SIZE = 9
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 8
LEGEND_SIZE = 7
LINE_WIDTH = 1.5
MARKER_SIZE = 4.0
SPINE_WIDTH = 1.0

OUTPUT_TOKEN_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
    "#bcbd22",
]
BATCH_PALETTE = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#17becf",
    "#9edae5",
    "#7f7f7f",
]


def choose_font(candidates: Iterable[str]) -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def configure_matplotlib() -> None:
    english_font = choose_font(["Times New Roman", "Times", "Nimbus Roman"])
    chinese_font = choose_font(["SimSun", "Noto Serif CJK SC", "Source Han Serif SC"])
    mpl.rcParams.update(
        {
            "font.family": [english_font, chinese_font, "DejaVu Sans"],
            "font.size": FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": FIG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agg_path = data_dir / "decode_modeling_merged_filtered_aggregated.csv"
    raw_path = data_dir / "decode_modeling_merged_filtered_raw.csv"
    missing = [str(path) for path in (agg_path, raw_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required CSV file(s): " + ", ".join(missing))

    agg_df = pd.read_csv(agg_path)
    raw_df = pd.read_csv(raw_path)
    agg_df["normalized_kv_blocks"] = (
        agg_df["avg_normalized_kv_blocks"].round().astype(int)
    )
    return agg_df.sort_values(["batch_size", "target_output_tokens"]), raw_df


def aggregate_raw_by_kv(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        raw_df.groupby(["batch_size", "normalized_kv_blocks"], as_index=False)
        .agg(
            avg_power_w=("avg_power_w", "mean"),
            avg_tbt_ms=("avg_tbt_ms", "mean"),
        )
        .sort_values(["batch_size", "normalized_kv_blocks"])
    )
    return grouped


def style_axes(ax: plt.Axes, use_minor: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("black")
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=False,
        right=False,
        length=4,
        width=0.9,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=False,
        right=False,
        length=2,
        width=0.7,
    )
    if use_minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(False)


def style_colorbar(cbar: mpl.colorbar.Colorbar) -> None:
    cbar.ax.tick_params(which="major", direction="in", length=4, width=0.8)
    cbar.ax.tick_params(which="minor", direction="in", length=2, width=0.6)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    cbar.outline.set_linewidth(0.8)


def apply_y_limits(ax: plt.Axes, values: pd.Series) -> None:
    ymin = float(values.min())
    ymax = float(values.max())
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0) * 0.1
    lower = ymin - 0.08 * span
    upper = ymax + 0.10 * span
    if lower > 0 and ymin / max(ymax, 1e-9) < 0.25:
        lower = 0.0
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_origin_cmap(colors: Tuple[str, str, str], name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, colors)


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    cmap: mpl.colors.Colormap,
    colorbar_label: str,
) -> None:
    pivot = (
        df.pivot_table(
            index="normalized_kv_blocks",
            columns="batch_size",
            values=value_col,
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(4.9, 3.35), constrained_layout=True)
    mesh = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap=cmap)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Normalized KV Blocks")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(item)) for item in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(item)) for item in pivot.index])

    finite_values = pivot.values[np.isfinite(pivot.values)]
    annotate = finite_values.size <= 150
    if annotate:
        threshold = (float(np.nanmin(finite_values)) + float(np.nanmax(finite_values))) / 2
        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                value = pivot.iloc[row_idx, col_idx]
                if pd.isna(value):
                    continue
                color = "white" if value > threshold else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color=color,
                )

    style_axes(ax, use_minor=False)
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.tick_params(which="minor", length=1.5)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label(colorbar_label, fontsize=AXIS_LABEL_SIZE)
    style_colorbar(cbar)
    save_figure(fig, output_path)


def plot_metric_by_batch(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 3.25))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.97)
    output_tokens = sorted(df["target_output_tokens"].unique())
    for idx, output_tokens_value in enumerate(output_tokens):
        subset = df[df["target_output_tokens"] == output_tokens_value]
        ax.plot(
            subset["batch_size"],
            subset[metric_col],
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=OUTPUT_TOKEN_PALETTE[idx % len(OUTPUT_TOKEN_PALETTE)],
            label=str(int(output_tokens_value)),
        )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(df["batch_size"].min()) - 1.5, float(df["batch_size"].max()) + 1.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    apply_y_limits(ax, df[metric_col])
    style_axes(ax)
    legend = ax.legend(
        title="Output Tokens",
        title_fontsize=LEGEND_SIZE,
        frameon=False,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.4,
    )
    for line in legend.get_lines():
        line.set_linewidth(LINE_WIDTH)
    save_figure(fig, output_path)


def plot_power_by_kv(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 3.35), constrained_layout=True)
    batch_sizes = sorted(df["batch_size"].unique())
    norm = mpl.colors.Normalize(vmin=min(batch_sizes), vmax=max(batch_sizes))
    cmap = mpl.colormaps.get_cmap("viridis")
    for idx, batch_size in enumerate(batch_sizes):
        subset = df[df["batch_size"] == batch_size]
        ax.plot(
            subset["normalized_kv_blocks"],
            subset["avg_power_w"],
            marker="o",
            markersize=3.4,
            linewidth=1.25,
            color=cmap(norm(batch_size)),
        )

    ax.set_xlabel("Normalized KV Blocks")
    ax.set_ylabel("Average Decode Power (W)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    apply_y_limits(ax, df["avg_power_w"])
    style_axes(ax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.025)
    cbar.set_label("Batch Size", fontsize=AXIS_LABEL_SIZE)
    style_colorbar(cbar)
    save_figure(fig, output_path)


def generate_summary(agg_df: pd.DataFrame, output_dir: Path) -> None:
    summary_rows = [
        ("config_count", len(agg_df)),
        ("batch_min", int(agg_df["batch_size"].min())),
        ("batch_max", int(agg_df["batch_size"].max())),
        ("target_output_min", int(agg_df["target_output_tokens"].min())),
        ("target_output_max", int(agg_df["target_output_tokens"].max())),
        ("avg_power_min_w", round(float(agg_df["avg_power_w"].min()), 3)),
        ("avg_power_max_w", round(float(agg_df["avg_power_w"].max()), 3)),
        ("avg_tbt_min_ms", round(float(agg_df["avg_tbt_ms"].min()), 3)),
        ("avg_tbt_max_ms", round(float(agg_df["avg_tbt_ms"].max()), 3)),
        ("p95_tbt_max_ms", round(float(agg_df["p95_tbt_ms"].max()), 3)),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary_df.to_csv(output_dir / "decode_modeling_paper_summary.csv", index=False)

    highest_power = agg_df.loc[agg_df["avg_power_w"].idxmax()]
    highest_tbt = agg_df.loc[agg_df["avg_tbt_ms"].idxmax()]
    lines = [
        "# Decode Modeling Paper Figures Summary",
        "",
        f"- Config count: {len(agg_df)}",
        (
            f"- Batch size range: {int(agg_df['batch_size'].min())}"
            f"-{int(agg_df['batch_size'].max())}"
        ),
        (
            f"- Target output token range: {int(agg_df['target_output_tokens'].min())}"
            f"-{int(agg_df['target_output_tokens'].max())}"
        ),
        (
            f"- Average decode power range: {agg_df['avg_power_w'].min():.2f}"
            f"-{agg_df['avg_power_w'].max():.2f} W"
        ),
        (
            f"- Average TBT range: {agg_df['avg_tbt_ms'].min():.2f}"
            f"-{agg_df['avg_tbt_ms'].max():.2f} ms"
        ),
        (
            f"- Highest power: {highest_power['avg_power_w']:.2f} W "
            f"(batch={int(highest_power['batch_size'])}, "
            f"output={int(highest_power['target_output_tokens'])}, "
            f"KV={int(highest_power['normalized_kv_blocks'])})"
        ),
        (
            f"- Highest average TBT: {highest_tbt['avg_tbt_ms']:.2f} ms "
            f"(batch={int(highest_tbt['batch_size'])}, "
            f"output={int(highest_tbt['target_output_tokens'])}, "
            f"KV={int(highest_tbt['normalized_kv_blocks'])})"
        ),
    ]
    (output_dir / "decode_modeling_paper_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def generate_figures(data_dir: Path, output_dir: Path) -> Dict[str, Path]:
    configure_matplotlib()
    agg_df, raw_df = load_data(data_dir)
    kv_df = aggregate_raw_by_kv(raw_df)
    output_dir.mkdir(parents=True, exist_ok=True)

    red_cmap = make_origin_cmap(("#fff7ec", "#fdbb84", "#b30000"), "paper_red")
    blue_cmap = make_origin_cmap(("#f7fbff", "#9ecae1", "#08519c"), "paper_blue")

    figure_stems = {
        "power_heatmap": output_dir / "decode_power_heatmap_paper",
        "tbt_heatmap": output_dir / "decode_tbt_heatmap_paper",
        "tbt_line": output_dir / "decode_tbt_by_batch_paper",
        "tbt_p50_line": output_dir / "decode_tbt_p50_by_batch_paper",
        "tbt_p95_line": output_dir / "decode_tbt_p95_by_batch_paper",
        "tbt_p99_line": output_dir / "decode_tbt_p99_by_batch_paper",
        "power_line": output_dir / "decode_power_by_kv_paper",
    }

    plot_heatmap(
        kv_df,
        value_col="avg_power_w",
        output_path=figure_stems["power_heatmap"],
        cmap=red_cmap,
        colorbar_label="Average Decode Power (W)",
    )
    plot_heatmap(
        kv_df,
        value_col="avg_tbt_ms",
        output_path=figure_stems["tbt_heatmap"],
        cmap=blue_cmap,
        colorbar_label="Average TBT (ms)",
    )
    plot_metric_by_batch(
        agg_df,
        metric_col="avg_tbt_ms",
        ylabel="Average TBT (ms)",
        output_path=figure_stems["tbt_line"],
    )
    plot_metric_by_batch(
        agg_df,
        metric_col="p50_tbt_ms",
        ylabel="TBT P50 (ms)",
        output_path=figure_stems["tbt_p50_line"],
    )
    plot_metric_by_batch(
        agg_df,
        metric_col="p95_tbt_ms",
        ylabel="TBT P95 (ms)",
        output_path=figure_stems["tbt_p95_line"],
    )
    plot_metric_by_batch(
        agg_df,
        metric_col="p99_tbt_ms",
        ylabel="TBT P99 (ms)",
        output_path=figure_stems["tbt_p99_line"],
    )
    plot_power_by_kv(kv_df, figure_stems["power_line"])
    generate_summary(agg_df, output_dir)
    return figure_stems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-style decode modeling figures."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    outputs = generate_figures(args.data_dir, args.output_dir)
    print(f"Paper figures written to: {args.output_dir}")
    for name, stem in outputs.items():
        print(f"- {name}: {stem.with_suffix('.png')} / {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
