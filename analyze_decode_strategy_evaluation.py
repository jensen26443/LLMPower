#!/usr/bin/env python3
"""
解码阶段功率策略评估结果分析脚本
"""
import glob
import math
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PREFERRED_STRATEGY_ORDER = [
    "scheme1_fit_bucket",
    "scheme2_fit_plus20",
    "scheme3_balanced_v2",
    "scheme4_latency_v2",
    "baseline_350w",
]


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


def plot_metric(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, output_path: str):
    hue_order = get_plot_strategy_order(df)
    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=df,
        x="output_length",
        y=metric_col,
        hue="strategy",
        hue_order=hue_order,
        marker="o",
        linewidth=2,
    )
    plt.title(title, fontsize=14, pad=16)
    plt.xlabel("Output Length (tokens)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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


def plot_relative_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: str):
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

    plt.figure(figsize=(13, 7))
    sns.barplot(
        data=metric_df.sort_values("query_length_label"),
        x="query_length_label",
        y="value",
        hue="strategy",
        hue_order=hue_order,
    )
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title(title, fontsize=14, pad=16)
    plt.xlabel("Number of Query / Length of Generation", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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
