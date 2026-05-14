#!/usr/bin/env python3
"""
解码阶段功率策略评估结果分析脚本
"""
import glob
import math
import os
from typing import Dict, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

PLOT_DPI = 600
FONT_SIZE = 9
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 8
LEGEND_SIZE = 7
AXIS_LINE_WIDTH = 1.0
STRATEGY_LABEL_SIZE = AXIS_LABEL_SIZE


def pick_available_font(candidates, fallback="DejaVu Sans"):
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return fallback


EN_FONT = pick_available_font(["Times New Roman", "Times", "Nimbus Roman"], "DejaVu Serif")
ZH_FONT = pick_available_font(["SimSun", "Noto Serif CJK SC", "Source Han Serif SC"])
mpl.rcParams.update({
    "font.family": [EN_FONT, ZH_FONT, "DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": PLOT_DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
sns.set_theme(style="white")

PREFERRED_STRATEGY_ORDER = [
    "scheme1_fit_curve",
    "scheme2_fit_plus",
    "scheme3_kv_guided",
    "scheme1_fit_bucket",
    "scheme2_fit_plus20",
    "scheme3_balanced_v2",
    "scheme4_latency_v2",
    "baseline_350w",
]

PAPER_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#7f7f7f",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
]

STRATEGY_DISPLAY_LABELS = {
    "scheme1_fit_curve": "150/151/191/210/220W",
    "scheme2_fit_plus": "150/170/200/220/230W",
    "scheme3_kv_guided": "190/205/210W",
    "baseline_350w": "350W",
}


def load_latest_aggregated_csv(result_dir: str) -> Optional[pd.DataFrame]:
    pattern = os.path.join(result_dir, "*_aggregated.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"没有找到 aggregated 文件: {pattern}")
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"加载文件: {latest_file}")
    return pd.read_csv(latest_file)


def load_aggregated_csvs(result_dirs) -> Optional[pd.DataFrame]:
    frames = []
    for result_dir in result_dirs:
        df = load_latest_aggregated_csv(result_dir)
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def aggregate_across_full_repeats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["strategy", "output_length", "concurrency", "power_limit"], as_index=False)
        .agg(
            avg_ttft_ms=("avg_ttft_ms", "mean"),
            avg_tbt_ms=("avg_tbt_ms", "mean"),
            avg_e2e_ms=("avg_e2e_ms", "mean"),
            avg_energy_j=("avg_energy_j", "mean"),
            avg_power_w=("avg_power_w", "mean"),
        )
        .sort_values(["strategy", "output_length"])
    )
    return grouped


def get_strategy_palette(strategy_order):
    return {
        strategy: PAPER_COLORS[index % len(PAPER_COLORS)]
        for index, strategy in enumerate(strategy_order)
    }


def get_strategy_display_label(strategy: str) -> str:
    return STRATEGY_DISPLAY_LABELS.get(strategy, strategy)


def apply_strategy_legend_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    display_labels = [get_strategy_display_label(label) for label in labels]
    return ax.legend(
        handles,
        display_labels,
        frameon=False,
        ncol=1,
        fontsize=STRATEGY_LABEL_SIZE,
    )


def get_query_length_order(df: pd.DataFrame):
    return [
        f"{int(row['concurrency'])}/{int(row['output_length'])}"
        for _, row in (
            df[["concurrency", "output_length"]]
            .drop_duplicates()
            .sort_values(["concurrency", "output_length"])
            .iterrows()
        )
    ]


def apply_paper_style(ax, y_zero_floor: bool = False, legend: bool = False, rotate_xticks: bool = False):
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
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=2.0,
        width=0.8,
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    if rotate_xticks:
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")

    if y_zero_floor:
        bottom, top = ax.get_ylim()
        if bottom >= 0:
            ax.set_ylim(0, top * 1.05 if top > 0 else 1.0)

    if legend:
        legend_obj = apply_strategy_legend_labels(ax)
        if legend_obj is not None:
            for text in legend_obj.get_texts():
                text.set_fontsize(STRATEGY_LABEL_SIZE)


def save_paper_figure(fig, output_path: str):
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metric(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, output_path: str):
    plot_df = df.copy()
    plot_df["query_length_label"] = plot_df.apply(
        lambda row: f"{int(row['concurrency'])}/{int(row['output_length'])}",
        axis=1,
    )
    label_order = get_query_length_order(plot_df)
    plot_df["query_length_label"] = pd.Categorical(
        plot_df["query_length_label"],
        categories=label_order,
        ordered=True,
    )
    hue_order = get_plot_strategy_order(df)
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    sns.barplot(
        data=plot_df.sort_values("query_length_label"),
        x="query_length_label",
        y=metric_col,
        hue="strategy",
        hue_order=hue_order,
        palette=get_strategy_palette(hue_order),
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_xlabel("Number of Query / Length of Generation")
    ax.set_ylabel(ylabel)
    apply_paper_style(ax, y_zero_floor=True, legend=True, rotate_xticks=True)
    save_paper_figure(fig, output_path)


def geometric_mean(values):
    if not values:
        return 0.0
    positive = [float(value) for value in values if float(value) > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def compute_relative_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["strategy", "output_length", "concurrency", "power_limit"]
    if df.duplicated(subset=key_cols).any():
        df = aggregate_across_full_repeats(df)

    baseline_df = (
        df[df["strategy"] == "baseline_350w"]
        .set_index(["concurrency", "output_length"])
        .sort_index()
    )
    rows = []
    strategies = sorted(strategy for strategy in df["strategy"].unique() if strategy != "baseline_350w")

    for strategy in strategies:
        strategy_df = (
            df[df["strategy"] == strategy]
            .set_index(["concurrency", "output_length"])
            .sort_index()
        )
        shared_keys = sorted(set(strategy_df.index.tolist()) & set(baseline_df.index.tolist()))
        tbt_ratios = []
        energy_ratios = []
        for concurrency, output_length in shared_keys:
            baseline_row = baseline_df.loc[(concurrency, output_length)]
            row = strategy_df.loc[(concurrency, output_length)]
            tbt_ratio = row["avg_tbt_ms"] / baseline_row["avg_tbt_ms"]
            energy_ratio = row["avg_energy_j"] / baseline_row["avg_energy_j"]
            tbt_ratios.append(tbt_ratio)
            energy_ratios.append(energy_ratio)
            label = f"{int(concurrency)}/{int(output_length)}"
            rows.append({
                "strategy": strategy,
                "metric": "tbt_loss_pct",
                "query_length_label": label,
                "concurrency": int(concurrency),
                "output_length": int(output_length),
                "value": (tbt_ratio - 1.0) * 100.0,
            })
            rows.append({
                "strategy": strategy,
                "metric": "energy_saving_pct",
                "query_length_label": label,
                "concurrency": int(concurrency),
                "output_length": int(output_length),
                "value": (1.0 - energy_ratio) * 100.0,
            })

        rows.append({
            "strategy": strategy,
            "metric": "tbt_loss_pct",
            "query_length_label": "GEOMEAN",
            "concurrency": None,
            "output_length": None,
            "value": (geometric_mean(tbt_ratios) - 1.0) * 100.0,
        })
        rows.append({
            "strategy": strategy,
            "metric": "energy_saving_pct",
            "query_length_label": "GEOMEAN",
            "concurrency": None,
            "output_length": None,
            "value": (1.0 - geometric_mean(energy_ratios)) * 100.0,
        })

    return pd.DataFrame(rows)


def get_plot_strategy_order(df: pd.DataFrame):
    present = set(df["strategy"].unique())
    ordered = [name for name in PREFERRED_STRATEGY_ORDER if name in present]
    remainder = sorted(name for name in present if name not in ordered)
    return ordered + remainder


def plot_relative_metric(df: pd.DataFrame,
                         metric: str,
                         title: str,
                         ylabel: str,
                         output_path: str,
                         ax=None,
                         save: bool = True):
    metric_df = df[df["metric"] == metric].copy()
    hue_order = get_plot_strategy_order(metric_df)
    ordered_labels = [
        row["query_length_label"]
        for _, row in (
            metric_df[metric_df["query_length_label"] != "GEOMEAN"]
            .sort_values(["concurrency", "output_length"])
            .drop_duplicates(["query_length_label"])
            .iterrows()
        )
    ]
    ordered_labels.append("GEOMEAN")
    metric_df["query_length_label"] = pd.Categorical(
        metric_df["query_length_label"],
        categories=ordered_labels,
        ordered=True,
    )
    metric_df["display_label"] = metric_df["query_length_label"].astype(str).replace({
        "GEOMEAN": "GEO\nMEAN",
    })
    display_order = ["GEO\nMEAN" if label == "GEOMEAN" else label for label in ordered_labels]
    metric_df["display_label"] = pd.Categorical(
        metric_df["display_label"],
        categories=display_order,
        ordered=True,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(9.6, 4.4))
    else:
        fig = ax.figure
    sns.barplot(
        data=metric_df.sort_values("query_length_label"),
        x="display_label",
        y="value",
        hue="strategy",
        hue_order=hue_order,
        palette=get_strategy_palette(hue_order),
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
    )
    if metric != "energy_saving_pct":
        ax.axhline(0.0, color="black", linewidth=AXIS_LINE_WIDTH)
    ax.set_xlabel("Number of Query / Length of Generation")
    ax.set_ylabel(ylabel)
    apply_paper_style(ax, y_zero_floor=False, legend=True, rotate_xticks=True)
    if metric == "energy_saving_pct":
        max_value = float(metric_df["value"].max()) if not metric_df.empty else 0.0
        ax.set_ylim(0.0, max_value * 1.1 if max_value > 0 else 1.0)
    if save:
        save_paper_figure(fig, output_path)
    return ax


def write_report(df: pd.DataFrame, output_path: str):
    baseline_df = df[df["strategy"] == "baseline_350w"].set_index(["concurrency", "output_length"])
    lines = [
        "# Decode Strategy Evaluation Report",
        "",
        "## Summary",
        f"- Strategy count: {df['strategy'].nunique()}",
        f"- Output lengths: {sorted(df['output_length'].unique().tolist())}",
        f"- Concurrency values: {sorted(df['concurrency'].unique().tolist())}",
        "",
        "## Relative To Baseline 350W",
    ]

    for strategy in sorted(df["strategy"].unique()):
        if strategy == "baseline_350w":
            continue
        strategy_df = df[df["strategy"] == strategy].set_index(["concurrency", "output_length"])
        shared_keys = sorted(set(strategy_df.index.tolist()) & set(baseline_df.index.tolist()))
        if not shared_keys:
            continue

        energy_saving = []
        tbt_loss = []
        for key in shared_keys:
            baseline_row = baseline_df.loc[key]
            row = strategy_df.loc[key]
            if baseline_row["avg_energy_j"] > 0:
                energy_saving.append((baseline_row["avg_energy_j"] - row["avg_energy_j"]) / baseline_row["avg_energy_j"] * 100.0)
            if baseline_row["avg_tbt_ms"] > 0:
                tbt_loss.append((row["avg_tbt_ms"] - baseline_row["avg_tbt_ms"]) / baseline_row["avg_tbt_ms"] * 100.0)

        lines.append(
            f"- {strategy}: mean energy saving={sum(energy_saving)/len(energy_saving):.2f}%, "
            f"mean TBT change={sum(tbt_loss)/len(tbt_loss):.2f}%"
        )

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def generate_outputs(agg_df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    plot_df = aggregate_across_full_repeats(agg_df)
    relative_df = compute_relative_plot_df(plot_df)

    outputs = {
        "tbt": os.path.join(output_dir, "decode_strategy_tbt.png"),
        "ttft": os.path.join(output_dir, "decode_strategy_ttft.png"),
        "e2e": os.path.join(output_dir, "decode_strategy_e2e.png"),
        "energy": os.path.join(output_dir, "decode_strategy_energy.png"),
        "power": os.path.join(output_dir, "decode_strategy_power.png"),
        "energy_saving": os.path.join(output_dir, "decode_strategy_energy_saving.png"),
        "tbt_loss": os.path.join(output_dir, "decode_strategy_tbt_loss.png"),
        "report": os.path.join(output_dir, "decode_strategy_report.md"),
    }

    plot_metric(plot_df, "avg_tbt_ms", "Strategy vs Decode TBT", "Average TBT (ms)", outputs["tbt"])
    plot_metric(plot_df, "avg_ttft_ms", "Strategy vs TTFT", "Average TTFT (ms)", outputs["ttft"])
    plot_metric(plot_df, "avg_e2e_ms", "Strategy vs E2E", "Average E2E (ms)", outputs["e2e"])
    plot_metric(plot_df, "avg_energy_j", "Strategy vs Energy", "Average Energy (J)", outputs["energy"])
    plot_metric(plot_df, "avg_power_w", "Strategy vs Decode Power", "Average Power (W)", outputs["power"])
    plot_relative_metric(
        relative_df,
        "energy_saving_pct",
        "Strategy vs Energy Saving",
        "Energy Saving (%)",
        outputs["energy_saving"],
    )
    plot_relative_metric(
        relative_df,
        "tbt_loss_pct",
        "Strategy vs TBT Loss",
        "TBT Loss (%)",
        outputs["tbt_loss"],
    )
    write_report(plot_df, outputs["report"])
    return outputs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="解码阶段功率策略评估结果分析")
    parser.add_argument("--result-dir", type=str, default="results_decode/strategy_evaluation")
    parser.add_argument("--result-dirs", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results_decode/strategy_evaluation/images")
    args = parser.parse_args()

    if args.result_dirs:
        result_dirs = [item.strip() for item in args.result_dirs.split(",") if item.strip()]
        agg_df = load_aggregated_csvs(result_dirs)
    else:
        agg_df = load_latest_aggregated_csv(args.result_dir)
    if agg_df is None or agg_df.empty:
        raise SystemExit(1)

    outputs = generate_outputs(agg_df, args.output_dir)
    print("分析完成，输出文件：")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
