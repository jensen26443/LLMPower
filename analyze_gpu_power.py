#!/usr/bin/env python3
"""Analyze static GPU power-cap experiments and regenerate paper-style figures."""

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator, MultipleLocator


DEFAULT_DATA_DIR = "experiment_results/legacy/results0/data"
DEFAULT_OUTPUT_DIR = "experiment_results/legacy/results0/img_bigger"
DPI = 600
FIGSIZE_2X2 = (7.2, 5.8)
FIGSIZE_1X2 = (7.2, 2.9)
FIGSIZE_SINGLE = (3.6, 2.9)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
MARKERS = ["o", "^", "s", "D", "v", "P"]


def query_count_label(query_count):
    return str(int(query_count))


def get_available_font_family(candidates):
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return candidates[-1]


def configure_origin_style():
    serif_font = get_available_font_family(["Times New Roman", "DejaVu Serif"])
    cjk_font = get_available_font_family(["SimSun", "Noto Serif CJK SC", "DejaVu Sans"])
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": [serif_font, cjk_font],
        "font.serif": [serif_font, cjk_font],
        "font.sans-serif": [cjk_font, serif_font],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "axes.linewidth": 1.2,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.2,
    })


def apply_axis_style(ax, x_values=None, y_values=None, y_zero=True):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        bottom=True,
        left=True,
        top=False,
        right=False,
        length=4.8,
        width=1.0,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        bottom=True,
        left=True,
        top=False,
        right=False,
        length=2.4,
        width=0.8,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(False)
    ax.set_facecolor("white")

    if x_values is not None and len(x_values) > 0:
        x_values = np.asarray(sorted(set(x_values)), dtype=float)
        x_pad = max((float(x_values.max()) - float(x_values.min())) * 0.035, 1.0)
        ax.set_xlim(float(x_values.min()) - x_pad, float(x_values.max()) + x_pad)
        ax.set_xticks(x_values)
    if y_values is not None and len(y_values) > 0:
        max_y = float(np.nanmax(y_values))
        if y_zero:
            ax.set_ylim(0.0, max(1.0, max_y * 1.10))
        else:
            min_y = float(np.nanmin(y_values))
            pad = max((max_y - min_y) * 0.10, max_y * 0.02, 1.0)
            ax.set_ylim(max(0.0, min_y - pad), max_y + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def parse_filename(filename):
    match = re.search(r"(\d+)W_mixed_(\d+)[qc]_", filename)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def load_all_data(directory):
    rows = []
    data_dir = Path(directory)
    csv_files = sorted(data_dir.glob("*metadata.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metadata CSV files found in {data_dir}")

    for file_path in csv_files:
        power_cap, query_count = parse_filename(file_path.name)
        if power_cap is None:
            continue
        frame = pd.read_csv(file_path)
        frame["parsed_power_cap"] = power_cap
        frame["parsed_query_count"] = query_count
        frame["source_file"] = file_path.name
        rows.append(frame)

    if not rows:
        raise ValueError(f"No parseable metadata CSV files found in {data_dir}")

    data = pd.concat(rows, ignore_index=True)
    data["energy_per_token_j"] = data["total_energy_j"] / data["total_tokens"].clip(lower=1)
    data["edp_j_s"] = data["total_energy_j"] * data["total_time_s"]
    return data


def build_summary(df):
    metrics = [
        "total_energy_j",
        "total_time_s",
        "avg_ttft_ms",
        "avg_tbt_ms",
        "avg_e2e_ms",
        "throughput_tps",
        "energy_per_token_j",
        "edp_j_s",
    ]
    grouped = df.groupby(["parsed_power_cap", "parsed_query_count"], as_index=False)
    summary = grouped.agg(
        repeat_count=("experiment_id", "count"),
        total_tokens=("total_tokens", "mean"),
        **{metric: (metric, "mean") for metric in metrics},
        **{f"{metric}_std": (metric, "std") for metric in metrics},
    ).fillna(0.0)
    summary["total_energy_kj"] = summary["total_energy_j"] / 1000.0
    summary["total_energy_kj_std"] = summary["total_energy_j_std"] / 1000.0
    summary["edp_1e8_j_s"] = summary["edp_j_s"] / 1e8
    summary["edp_1e8_j_s_std"] = summary["edp_j_s_std"] / 1e8
    return summary.sort_values(["parsed_query_count", "parsed_power_cap"])


def save_figure(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def add_query_count_legend(ax, location="upper right", columns=2, anchor=None):
    kwargs = {}
    if anchor is not None:
        kwargs["bbox_to_anchor"] = anchor
    ax.legend(
        title="Query count",
        frameon=False,
        loc=location,
        ncol=columns,
        handlelength=1.8,
        columnspacing=1.0,
        borderaxespad=0.35,
        **kwargs,
    )


def apply_decimal_y_axis(ax, nbins=7, decimals=1):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))


def apply_integer_y_axis(ax, nbins=6):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, integer=True))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))


def apply_edp_axis(ax):
    ax.set_yscale("linear")
    ax.set_ylim(-0.05, 2.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def plot_line_series(ax, data, x_metric, y_metric, index, query_count, linewidth=1.5, markersize=4.2):
    ax.plot(
        data[x_metric],
        data[y_metric],
        marker=MARKERS[index % len(MARKERS)],
        color=COLORS[index % len(COLORS)],
        label=query_count_label(query_count),
        linewidth=linewidth,
        markersize=markersize,
    )


def plot_metric_grid(summary, output_dir, power_caps, query_counts):
    metrics = [
        ("total_energy_kj", "Energy (kJ)", True, False),
        ("total_time_s", "E2E latency (s)", True, False),
        ("avg_ttft_ms", "TTFT (ms)", False, True),
        ("avg_tbt_ms", "TBT (ms)", False, False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2X2)
    for ax, (metric, ylabel, y_zero, show_legend) in zip(axes.ravel(), metrics):
        for index, query_count in enumerate(query_counts):
            data = summary[summary["parsed_query_count"] == query_count].sort_values("parsed_power_cap")
            plot_line_series(ax, data, "parsed_power_cap", metric, index, query_count)
        ax.set_xlabel("Power cap (W)")
        ax.set_ylabel(ylabel)
        apply_axis_style(ax, power_caps, summary[metric], y_zero=y_zero)
        if metric == "avg_ttft_ms":
            apply_integer_y_axis(ax)
        if show_legend:
            add_query_count_legend(ax, location="upper right", columns=2)
    save_figure(fig, output_dir / "1_performance_metrics_comparison.png")


def plot_heatmaps(summary, output_dir):
    metrics = [
        ("total_energy_kj", "Energy (kJ)", ".1f"),
        ("total_time_s", "E2E latency (s)", ".1f"),
        ("avg_ttft_ms", "TTFT (ms)", ".1f"),
        ("avg_tbt_ms", "TBT (ms)", ".2f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2X2)
    for ax, (metric, title, fmt) in zip(axes.ravel(), metrics):
        pivot = summary.pivot(index="parsed_power_cap", columns="parsed_query_count", values=metric)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=fmt,
            cmap="RdYlGn_r",
            linewidths=0.35,
            linecolor="white",
            cbar_kws={"label": title},
            ax=ax,
        )
        ax.set_xlabel("Query count")
        ax.set_ylabel("Power cap (W)")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            bottom=True,
            left=True,
            top=False,
            right=False,
            length=4.8,
            width=1.0,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            bottom=True,
            left=True,
            top=False,
            right=False,
            length=2.4,
            width=0.8,
        )
    save_figure(fig, output_dir / "2_performance_heatmaps.png")


def plot_efficiency(summary, output_dir, power_caps, query_counts):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_1X2)
    panels = [
        ("throughput_tps", "Throughput (tokens/s)", False, True),
        ("edp_1e8_j_s", r"EDP ($10^8$ J$\cdot$s)", False, False),
    ]
    for ax, (metric, ylabel, y_zero, show_legend) in zip(axes, panels):
        for index, query_count in enumerate(query_counts):
            data = summary[summary["parsed_query_count"] == query_count].sort_values("parsed_power_cap")
            if metric == "edp_1e8_j_s":
                plot_line_series(
                    ax,
                    data,
                    "parsed_power_cap",
                    metric,
                    index,
                    query_count,
                    linewidth=1.05,
                    markersize=3.5,
                )
            else:
                plot_line_series(ax, data, "parsed_power_cap", metric, index, query_count)
        ax.set_xlabel("Power cap (W)")
        ax.set_ylabel(ylabel)
        apply_axis_style(ax, power_caps, summary[metric], y_zero=y_zero)
        if metric == "throughput_tps":
            apply_decimal_y_axis(ax, nbins=7, decimals=1)
        if metric == "edp_1e8_j_s":
            apply_edp_axis(ax)
        if show_legend:
            add_query_count_legend(ax, location="lower right", columns=2)
    save_figure(fig, output_dir / "3_efficiency_analysis.png")


def plot_edp(summary, output_dir, power_caps, query_counts):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for index, query_count in enumerate(query_counts):
        data = summary[summary["parsed_query_count"] == query_count].sort_values("parsed_power_cap")
        plot_line_series(
            ax,
            data,
            "parsed_power_cap",
            "edp_1e8_j_s",
            index,
            query_count,
            linewidth=1.05,
            markersize=3.5,
        )
    ax.set_xlabel("Power cap (W)")
    ax.set_ylabel(r"EDP ($10^8$ J$\cdot$s)")
    apply_axis_style(ax, power_caps, summary["edp_1e8_j_s"], y_zero=False)
    apply_edp_axis(ax)
    add_query_count_legend(ax, location="lower center", columns=4, anchor=(0.58, 0.11))
    save_figure(fig, output_dir / "4_edp_analysis.png")


def plot_bar_comparison(summary, output_dir, power_caps, query_counts):
    metrics = [
        ("total_energy_kj", "Energy (kJ)", True),
        ("total_time_s", "E2E latency (s)", True),
        ("avg_ttft_ms", "TTFT (ms)", False),
        ("avg_tbt_ms", "TBT (ms)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2X2)
    width = 0.16
    x = np.arange(len(power_caps))
    for ax, (metric, ylabel, y_zero) in zip(axes.ravel(), metrics):
        for index, query_count in enumerate(query_counts):
            data = summary[summary["parsed_query_count"] == query_count].set_index("parsed_power_cap")
            values = [data.loc[power, metric] if power in data.index else np.nan for power in power_caps]
            offset = (index - (len(query_counts) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=COLORS[index % len(COLORS)],
                label=query_count_label(query_count),
                edgecolor="black",
                linewidth=0.35,
            )
        ax.set_xlabel("Power cap (W)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(power_caps)
        apply_axis_style(ax, None, summary[metric], y_zero=y_zero)
        if metric == "avg_ttft_ms":
            apply_integer_y_axis(ax)
        if metric == "avg_ttft_ms":
            add_query_count_legend(ax, location="upper right", columns=2)
    save_figure(fig, output_dir / "5_metrics_bar_comparison.png")


def plot_configuration_space(summary, output_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    scatter = ax.scatter(
        summary["parsed_power_cap"],
        summary["parsed_query_count"],
        c=summary["energy_per_token_j"],
        s=38,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.35,
    )
    ax.set_xlabel("Power cap (W)")
    ax.set_ylabel("Query count")
    apply_axis_style(ax, summary["parsed_power_cap"], summary["parsed_query_count"], y_zero=False)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Energy per token (J/token)")
    save_figure(fig, output_dir / "6_configuration_space.png")

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    scatter = ax.scatter(
        summary["parsed_power_cap"],
        summary["parsed_query_count"],
        c=summary["energy_per_token_j"],
        s=38,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.35,
    )
    ax.set_xlabel("Power cap (W)")
    ax.set_ylabel("Query count")
    apply_axis_style(ax, summary["parsed_power_cap"], summary["parsed_query_count"], y_zero=False)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Energy per token (J/token)")
    save_figure(fig, output_dir / "6_3d_configuration_space.png")


def create_comprehensive_plots(df, output_dir=DEFAULT_OUTPUT_DIR):
    configure_origin_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = build_summary(df)
    power_caps = sorted(summary["parsed_power_cap"].unique())
    query_counts = sorted(summary["parsed_query_count"].unique())

    plot_metric_grid(summary, output_path, power_caps, query_counts)
    plot_heatmaps(summary, output_path)
    plot_efficiency(summary, output_path, power_caps, query_counts)
    plot_edp(summary, output_path, power_caps, query_counts)
    plot_bar_comparison(summary, output_path, power_caps, query_counts)
    plot_configuration_space(summary, output_path)

    display_cols = [
        "parsed_power_cap",
        "parsed_query_count",
        "repeat_count",
        "total_energy_kj",
        "total_time_s",
        "avg_ttft_ms",
        "avg_tbt_ms",
        "throughput_tps",
        "energy_per_token_j",
        "edp_j_s",
    ]
    summary[display_cols].rename(columns={
        "parsed_power_cap": "Power(W)",
        "parsed_query_count": "Query Count",
        "repeat_count": "Repeats",
        "total_energy_kj": "Energy(kJ)",
        "total_time_s": "E2E Latency(s)",
        "avg_ttft_ms": "TTFT(ms)",
        "avg_tbt_ms": "TBT(ms)",
        "throughput_tps": "Throughput(tokens/s)",
        "energy_per_token_j": "Energy per Token(J/token)",
        "edp_j_s": "EDP(J*s)",
    }).to_csv(output_path / "7_performance_summary.csv", index=False)
    summary.to_csv(output_path / "7_performance_summary_with_std.csv", index=False)

    print(f"Loaded {len(df)} runs across {len(summary)} configurations.")
    print(f"Power caps: {[int(x) for x in power_caps]}")
    print(f"Query counts: {[int(x) for x in query_counts]}")
    print(f"Saved figures and summaries to {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Plot static GPU power-cap experiment results.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = load_all_data(args.data_dir)
    create_comprehensive_plots(df, args.output_dir)


if __name__ == "__main__":
    main()
