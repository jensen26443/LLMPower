#!/usr/bin/env python3
"""
前馈 + PID 控制评估结果分析脚本。
"""
import argparse
import glob
import math
import os
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PREFERRED_STRATEGY_ORDER = [
    "ff_v2_recommended",
    "ff_v2_pid_stable",
    "ff_v2_pid",
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


def load_aggregated_csvs(result_dirs: Sequence[str]) -> Optional[pd.DataFrame]:
    frames = []
    for result_dir in result_dirs:
        df = load_latest_aggregated_csv(result_dir)
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


def aggregate_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for source, target in [
        ("pid_update_count", "avg_pid_update_count"),
        ("pid_prefill_delta_w", "avg_pid_prefill_delta_w"),
        ("pid_decode_delta_w", "avg_pid_decode_delta_w"),
        ("power_change_count", "avg_power_change_count"),
    ]:
        if target not in df.columns and source in df.columns:
            df[target] = df[source]
    if "avg_power_change_count" not in df.columns:
        df["avg_power_change_count"] = 0.0
    grouped = (
        df.groupby(["strategy", "query_count", "output_length"], as_index=False)
        .agg(
            avg_ttft_ms=("avg_ttft_ms", "mean"),
            avg_tbt_ms=("avg_tbt_ms", "mean"),
            avg_e2e_ms=("avg_e2e_ms", "mean"),
            avg_energy_j=("avg_energy_j", "mean"),
            avg_power_w=("avg_power_w", "mean"),
            avg_power_change_count=("avg_power_change_count", "mean"),
            avg_pid_update_count=("avg_pid_update_count", "mean"),
            avg_pid_prefill_delta_w=("avg_pid_prefill_delta_w", "mean"),
            avg_pid_decode_delta_w=("avg_pid_decode_delta_w", "mean"),
        )
        .sort_values(["strategy", "query_count", "output_length"])
    )
    grouped["query_length_label"] = grouped.apply(
        lambda row: f"{int(row['query_count'])}/{int(row['output_length'])}",
        axis=1,
    )
    return grouped


def plot_metric(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, output_path: str):
    hue_order = get_plot_strategy_order(df)
    ordered_labels = (
        df.sort_values(["query_count", "output_length"])["query_length_label"]
        .drop_duplicates()
        .tolist()
    )
    plot_df = df.copy()
    plot_df["query_length_label"] = pd.Categorical(plot_df["query_length_label"], categories=ordered_labels, ordered=True)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df.sort_values("query_length_label"), x="query_length_label", y=metric_col, hue="strategy", hue_order=hue_order)
    plt.title(title, fontsize=14, pad=16)
    plt.xlabel("Number of Query / Length of Generation", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=25)
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
    baseline_df = df[df["strategy"] == "baseline_350w"].set_index(["query_count", "output_length"]).sort_index()
    rows = []
    for strategy in [name for name in get_plot_strategy_order(df) if name != "baseline_350w"]:
        strategy_df = df[df["strategy"] == strategy].set_index(["query_count", "output_length"]).sort_index()
        shared_keys = sorted(set(strategy_df.index.tolist()) & set(baseline_df.index.tolist()))
        ttft_ratios = []
        tbt_ratios = []
        e2e_ratios = []
        energy_ratios = []
        for query_count, output_length in shared_keys:
            baseline_row = baseline_df.loc[(query_count, output_length)]
            row = strategy_df.loc[(query_count, output_length)]
            ttft_ratio = row["avg_ttft_ms"] / baseline_row["avg_ttft_ms"]
            tbt_ratio = row["avg_tbt_ms"] / baseline_row["avg_tbt_ms"]
            e2e_ratio = row["avg_e2e_ms"] / baseline_row["avg_e2e_ms"]
            energy_ratio = row["avg_energy_j"] / baseline_row["avg_energy_j"]
            ttft_ratios.append(ttft_ratio)
            tbt_ratios.append(tbt_ratio)
            e2e_ratios.append(e2e_ratio)
            energy_ratios.append(energy_ratio)
            label = f"{int(query_count)}/{int(output_length)}"
            rows.extend([
                {"strategy": strategy, "metric": "energy_saving_pct", "query_length_label": label, "query_count": int(query_count), "output_length": int(output_length), "value": (1.0 - energy_ratio) * 100.0},
                {"strategy": strategy, "metric": "ttft_increase_pct", "query_length_label": label, "query_count": int(query_count), "output_length": int(output_length), "value": (ttft_ratio - 1.0) * 100.0},
                {"strategy": strategy, "metric": "tbt_increase_pct", "query_length_label": label, "query_count": int(query_count), "output_length": int(output_length), "value": (tbt_ratio - 1.0) * 100.0},
                {"strategy": strategy, "metric": "e2e_increase_pct", "query_length_label": label, "query_count": int(query_count), "output_length": int(output_length), "value": (e2e_ratio - 1.0) * 100.0},
            ])
        rows.extend([
            {"strategy": strategy, "metric": "energy_saving_pct", "query_length_label": "GEOMEAN", "query_count": None, "output_length": None, "value": (1.0 - geometric_mean(energy_ratios)) * 100.0},
            {"strategy": strategy, "metric": "ttft_increase_pct", "query_length_label": "GEOMEAN", "query_count": None, "output_length": None, "value": (geometric_mean(ttft_ratios) - 1.0) * 100.0},
            {"strategy": strategy, "metric": "tbt_increase_pct", "query_length_label": "GEOMEAN", "query_count": None, "output_length": None, "value": (geometric_mean(tbt_ratios) - 1.0) * 100.0},
            {"strategy": strategy, "metric": "e2e_increase_pct", "query_length_label": "GEOMEAN", "query_count": None, "output_length": None, "value": (geometric_mean(e2e_ratios) - 1.0) * 100.0},
        ])
    return pd.DataFrame(rows)


def plot_relative_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: str):
    metric_df = df[df["metric"] == metric].copy()
    hue_order = get_plot_strategy_order(metric_df)
    ordered_labels = [
        row["query_length_label"]
        for _, row in (
            metric_df[metric_df["query_length_label"] != "GEOMEAN"]
            .sort_values(["query_count", "output_length"])
            .drop_duplicates(["query_length_label"])
            .iterrows()
        )
    ]
    ordered_labels.append("GEOMEAN")
    metric_df["query_length_label"] = pd.Categorical(metric_df["query_length_label"], categories=ordered_labels, ordered=True)
    plt.figure(figsize=(13, 7))
    sns.barplot(data=metric_df.sort_values("query_length_label"), x="query_length_label", y="value", hue="strategy", hue_order=hue_order)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title(title, fontsize=14, pad=16)
    plt.xlabel("Number of Query / Length of Generation", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_report(df: pd.DataFrame, output_path: str):
    baseline_df = df[df["strategy"] == "baseline_350w"].set_index(["query_count", "output_length"])
    lines = [
        "# Feedforward PID Evaluation Report",
        "",
        "## Summary",
        f"- Strategy count: {df['strategy'].nunique()}",
        f"- Query counts: {sorted(df['query_count'].unique().tolist())}",
        f"- Output lengths: {sorted(df['output_length'].unique().tolist())}",
        "",
        "## Relative To Baseline 350W",
    ]
    for strategy in get_plot_strategy_order(df):
        if strategy == "baseline_350w":
            continue
        strategy_df = df[df["strategy"] == strategy].set_index(["query_count", "output_length"])
        shared_keys = sorted(set(strategy_df.index.tolist()) & set(baseline_df.index.tolist()))
        energy_saving = []
        ttft_increase = []
        tbt_increase = []
        for key in shared_keys:
            baseline_row = baseline_df.loc[key]
            row = strategy_df.loc[key]
            energy_saving.append((1.0 - row["avg_energy_j"] / baseline_row["avg_energy_j"]) * 100.0)
            ttft_increase.append((row["avg_ttft_ms"] / baseline_row["avg_ttft_ms"] - 1.0) * 100.0)
            tbt_increase.append((row["avg_tbt_ms"] / baseline_row["avg_tbt_ms"] - 1.0) * 100.0)
        lines.append(
            f"- {strategy}: geomean energy saving={(1.0 - geometric_mean([1.0 - x / 100.0 for x in energy_saving])) * 100.0:.2f}%"
        )
        lines.append(
            f"  mean energy saving={sum(energy_saving)/len(energy_saving):.2f}%, "
            f"mean TTFT increase={sum(ttft_increase)/len(ttft_increase):.2f}%, "
            f"mean TBT increase={sum(tbt_increase)/len(tbt_increase):.2f}%"
        )
    pid_names = [name for name in get_plot_strategy_order(df) if name.startswith("ff_v2_pid")]
    if pid_names:
        lines.extend(["", "## PID Behavior"])
    ff_df = df[df["strategy"] == "ff_v2_recommended"].set_index(["query_count", "output_length"])
    for pid_name in pid_names:
        pid_df = df[df["strategy"] == pid_name]
        lines.extend([
            f"- {pid_name}: mean PID updates={pid_df['avg_pid_update_count'].mean():.2f}, "
            f"mean power changes={pid_df['avg_power_change_count'].mean():.2f}, "
            f"mean prefill delta={pid_df['avg_pid_prefill_delta_w'].mean():.2f} W, "
            f"mean decode delta={pid_df['avg_pid_decode_delta_w'].mean():.2f} W",
        ])
        if not ff_df.empty:
            pid_idx = pid_df.set_index(["query_count", "output_length"])
            shared_keys = sorted(set(pid_idx.index.tolist()) & set(ff_df.index.tolist()))
            if shared_keys:
                energy_ratios = []
                tbt_ratios = []
                for key in shared_keys:
                    ff_row = ff_df.loc[key]
                    pid_row = pid_idx.loc[key]
                    energy_ratios.append(pid_row["avg_energy_j"] / ff_row["avg_energy_j"])
                    tbt_ratios.append(pid_row["avg_tbt_ms"] / ff_row["avg_tbt_ms"])
                lines.append(
                    f"  relative to ff_v2_recommended: geomean energy change="
                    f"{(geometric_mean(energy_ratios) - 1.0) * 100.0:.2f}%, "
                    f"geomean TBT change={(geometric_mean(tbt_ratios) - 1.0) * 100.0:.2f}%"
                )
    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def generate_outputs(agg_df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    plot_df = aggregate_for_plot(agg_df)
    relative_df = compute_relative_plot_df(plot_df)
    outputs = {
        "ttft": os.path.join(output_dir, "feedforward_pid_ttft.png"),
        "tbt": os.path.join(output_dir, "feedforward_pid_tbt.png"),
        "e2e": os.path.join(output_dir, "feedforward_pid_e2e.png"),
        "energy": os.path.join(output_dir, "feedforward_pid_energy.png"),
        "power": os.path.join(output_dir, "feedforward_pid_power.png"),
        "energy_saving": os.path.join(output_dir, "feedforward_pid_energy_saving.png"),
        "ttft_increase": os.path.join(output_dir, "feedforward_pid_ttft_increase.png"),
        "tbt_increase": os.path.join(output_dir, "feedforward_pid_tbt_increase.png"),
        "e2e_increase": os.path.join(output_dir, "feedforward_pid_e2e_increase.png"),
        "pid_delta": os.path.join(output_dir, "feedforward_pid_delta.png"),
        "pid_updates": os.path.join(output_dir, "feedforward_pid_updates.png"),
        "power_changes": os.path.join(output_dir, "feedforward_pid_power_changes.png"),
        "report": os.path.join(output_dir, "feedforward_pid_report.md"),
    }
    plot_metric(plot_df, "avg_ttft_ms", "TTFT by Query Count / Output Length", "TTFT (ms)", outputs["ttft"])
    plot_metric(plot_df, "avg_tbt_ms", "TBT by Query Count / Output Length", "TBT (ms)", outputs["tbt"])
    plot_metric(plot_df, "avg_e2e_ms", "E2E by Query Count / Output Length", "E2E (ms)", outputs["e2e"])
    plot_metric(plot_df, "avg_energy_j", "Energy by Query Count / Output Length", "Energy (J)", outputs["energy"])
    plot_metric(plot_df, "avg_power_w", "Power by Query Count / Output Length", "Power (W)", outputs["power"])
    plot_relative_metric(relative_df, "energy_saving_pct", "Energy Saving Relative to Baseline", "Energy Saving (%)", outputs["energy_saving"])
    plot_relative_metric(relative_df, "ttft_increase_pct", "TTFT Increase Relative to Baseline", "TTFT Increase (%)", outputs["ttft_increase"])
    plot_relative_metric(relative_df, "tbt_increase_pct", "TBT Increase Relative to Baseline", "TBT Increase (%)", outputs["tbt_increase"])
    plot_relative_metric(relative_df, "e2e_increase_pct", "E2E Increase Relative to Baseline", "E2E Increase (%)", outputs["e2e_increase"])
    plot_metric(plot_df, "avg_pid_decode_delta_w", "Mean Decode PID Delta", "PID Delta (W)", outputs["pid_delta"])
    plot_metric(plot_df, "avg_pid_update_count", "Mean PID Update Count", "Update Count", outputs["pid_updates"])
    plot_metric(plot_df, "avg_power_change_count", "Mean Power Change Count", "Power Changes", outputs["power_changes"])
    write_report(plot_df, outputs["report"])
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze feedforward PID evaluation results.")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--result-dirs", default=None)
    parser.add_argument("--output-dir", required=True)
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
    agg_df = load_aggregated_csvs(result_dirs)
    if agg_df is None or agg_df.empty:
        raise RuntimeError("No aggregated data found")
    outputs = generate_outputs(agg_df, args.output_dir)
    print("生成文件:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
