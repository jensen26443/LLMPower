#!/usr/bin/env python3
"""
解码阶段离线建模结果分析脚本

生成：
- batch size / normalized KV blocks vs decoding power 热力图
- batch size / normalized KV blocks vs TBT 热力图
- batch size vs TBT 趋势图
- normalized KV blocks vs decoding power 趋势图
"""
import glob
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_latest_csv(result_dir: str, suffix: str) -> Optional[pd.DataFrame]:
    pattern = os.path.join(result_dir, f"*_{suffix}.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"没有找到 {suffix} 文件: {pattern}")
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"加载文件: {latest_file}")
    return pd.read_csv(latest_file)


def load_latest_result_set(result_dir: str) -> Dict[str, Optional[pd.DataFrame]]:
    pattern = os.path.join(result_dir, "*_aggregated.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"没有找到 aggregated 文件: {pattern}")
        return {"aggregated": None, "raw": None}

    latest_agg_file = max(files, key=os.path.getctime)
    raw_file = latest_agg_file.replace("_aggregated.csv", "_raw.csv")

    print(f"加载文件: {latest_agg_file}")
    agg_df = pd.read_csv(latest_agg_file)

    if not os.path.exists(raw_file):
        print(f"没有找到匹配的 raw 文件: {raw_file}")
        return {"aggregated": agg_df, "raw": None}

    print(f"加载文件: {raw_file}")
    raw_df = pd.read_csv(raw_file)
    return {"aggregated": agg_df, "raw": raw_df}


def load_all_result_sets(result_dir: str) -> Dict[str, Optional[pd.DataFrame]]:
    pattern = os.path.join(result_dir, "*_aggregated.csv")
    agg_files = sorted(glob.glob(pattern))
    if not agg_files:
        print(f"没有找到 aggregated 文件: {pattern}")
        return {"aggregated": None, "raw": None}

    agg_frames = []
    raw_frames = []
    for agg_file in agg_files:
        raw_file = agg_file.replace("_aggregated.csv", "_raw.csv")
        if not os.path.exists(raw_file):
            print(f"跳过，没有找到匹配的 raw 文件: {raw_file}")
            continue

        print(f"加载文件: {agg_file}")
        agg_frames.append(pd.read_csv(agg_file))
        print(f"加载文件: {raw_file}")
        raw_frames.append(pd.read_csv(raw_file))

    if not agg_frames or not raw_frames:
        return {"aggregated": None, "raw": None}

    return {
        "aggregated": pd.concat(agg_frames, ignore_index=True),
        "raw": pd.concat(raw_frames, ignore_index=True),
    }


def load_result_sets_from_dirs(result_dirs: List[str], merge_all_runs: bool) -> Dict[str, Optional[pd.DataFrame]]:
    agg_frames = []
    raw_frames = []

    for result_dir in result_dirs:
        result_set = load_all_result_sets(result_dir) if merge_all_runs else load_latest_result_set(result_dir)
        agg_df = result_set["aggregated"]
        raw_df = result_set["raw"]

        if agg_df is None or agg_df.empty or raw_df is None or raw_df.empty:
            continue

        agg_frames.append(agg_df)
        raw_frames.append(raw_df)

    if not agg_frames or not raw_frames:
        return {"aggregated": None, "raw": None}

    return {
        "aggregated": pd.concat(agg_frames, ignore_index=True),
        "raw": pd.concat(raw_frames, ignore_index=True),
    }


def prepare_dataframe(agg_df: pd.DataFrame) -> pd.DataFrame:
    df = agg_df.copy()
    df["normalized_kv_blocks"] = df["avg_normalized_kv_blocks"].round().astype(int)
    return df.sort_values(["batch_size", "target_output_tokens"])


def aggregate_raw_by_kv(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        raw_df.groupby(["batch_size", "normalized_kv_blocks"], as_index=False)
        .agg(
            count=("index", "count"),
            avg_power_w=("avg_power_w", "mean"),
            std_power_w=("avg_power_w", "std"),
            peak_power_w=("peak_power_w", "mean"),
            avg_energy_j=("total_energy_j", "mean"),
            avg_dynamic_power_w=("dynamic_power_w", "mean"),
            avg_dynamic_energy_j=("dynamic_energy_j", "mean"),
            avg_ttft_ms=("avg_ttft_ms", "mean"),
            avg_tbt_ms=("avg_tbt_ms", "mean"),
            avg_e2e_ms=("avg_e2e_ms", "mean"),
            avg_generated_tokens=("avg_generated_tokens", "mean"),
            avg_context_total_tokens=("context_total_tokens", "mean"),
            avg_approx_kv_pressure=("approx_kv_pressure", "mean"),
            avg_idle_baseline_w=("idle_baseline_w", "mean"),
        )
        .sort_values(["batch_size", "normalized_kv_blocks"])
    )
    grouped["std_power_w"] = grouped["std_power_w"].fillna(0.0)
    return grouped


def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, output_path: str, cmap: str):
    pivot = df.pivot_table(
        index="normalized_kv_blocks",
        columns="batch_size",
        values=value_col,
        aggfunc="mean",
    ).sort_index()
    plt.figure(figsize=(11, 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap=cmap)
    plt.title(title, fontsize=14, pad=16)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Normalized KV Blocks per Request", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tbt_by_batch(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=df,
        x="batch_size",
        y="avg_tbt_ms",
        hue="target_output_tokens",
        marker="o",
        palette="tab10",
    )
    plt.title("Batch Size vs Decode TBT", fontsize=14, pad=16)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Average TBT (ms)", fontsize=12)
    plt.legend(title="Target Output Tokens")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_power_by_kv(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=df,
        x="normalized_kv_blocks",
        y="avg_power_w",
        hue="batch_size",
        marker="o",
        palette="viridis",
    )
    plt.title("Normalized KV Blocks vs Decode Power", fontsize=14, pad=16)
    plt.xlabel("Normalized KV Blocks per Request", fontsize=12)
    plt.ylabel("Average Decode Power (W)", fontsize=12)
    plt.legend(title="Batch Size")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_report(df: pd.DataFrame, output_path: str):
    highest_power_row = df.loc[df["avg_power_w"].idxmax()]
    highest_tbt_row = df.loc[df["avg_tbt_ms"].idxmax()]
    lowest_tbt_row = df.loc[df["avg_tbt_ms"].idxmin()]
    highest_p95_tbt_row = df.loc[df["p95_tbt_ms"].idxmax()] if "p95_tbt_ms" in df.columns else highest_tbt_row
    median_ttft = df["p50_ttft_ms"].median() if "p50_ttft_ms" in df.columns else df["avg_ttft_ms"].median()
    median_tbt = df["p50_tbt_ms"].median() if "p50_tbt_ms" in df.columns else df["avg_tbt_ms"].median()
    median_e2e = df["p50_e2e_ms"].median() if "p50_e2e_ms" in df.columns else df["avg_e2e_ms"].median()

    report_lines = [
        "# Decode Phase Modeling Report",
        "",
        "## Summary",
        f"- Config count: {len(df)}",
        f"- Batch size range: {int(df['batch_size'].min())} - {int(df['batch_size'].max())}",
        f"- Target output token range: {int(df['target_output_tokens'].min())} - {int(df['target_output_tokens'].max())}",
        f"- Median config-level TTFT P50: {median_ttft:.2f} ms",
        f"- Median config-level TBT P50: {median_tbt:.2f} ms",
        f"- Median config-level E2E P50: {median_e2e:.2f} ms",
        "",
        "## Key Observations",
        (
            f"- Highest decode power: {highest_power_row['avg_power_w']:.2f} W "
            f"(batch={int(highest_power_row['batch_size'])}, "
            f"target_output={int(highest_power_row['target_output_tokens'])}, "
            f"normalized_kv_blocks={int(highest_power_row['normalized_kv_blocks'])})"
        ),
        (
            f"- Highest TBT: {highest_tbt_row['avg_tbt_ms']:.2f} ms "
            f"(batch={int(highest_tbt_row['batch_size'])}, "
            f"target_output={int(highest_tbt_row['target_output_tokens'])}, "
            f"normalized_kv_blocks={int(highest_tbt_row['normalized_kv_blocks'])})"
        ),
        (
            f"- Lowest TBT: {lowest_tbt_row['avg_tbt_ms']:.2f} ms "
            f"(batch={int(lowest_tbt_row['batch_size'])}, "
            f"target_output={int(lowest_tbt_row['target_output_tokens'])}, "
            f"normalized_kv_blocks={int(lowest_tbt_row['normalized_kv_blocks'])})"
        ),
        (
            f"- Highest TBT P95: {highest_p95_tbt_row['p95_tbt_ms']:.2f} ms "
            f"(batch={int(highest_p95_tbt_row['batch_size'])}, "
            f"target_output={int(highest_p95_tbt_row['target_output_tokens'])}, "
            f"normalized_kv_blocks={int(highest_p95_tbt_row['normalized_kv_blocks'])})"
        ) if "p95_tbt_ms" in df.columns else "",
        "",
        "## Output Files",
        "- decode_power_heatmap.png",
        "- decode_tbt_heatmap.png",
        "- decode_tbt_by_batch.png",
        "- decode_power_by_kv.png",
        "- aggregated csv now includes mean/p50/p95/p99 for TTFT/TBT/E2E",
    ]

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(line for line in report_lines if line))


def generate_visualizations(agg_df: pd.DataFrame, raw_df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    agg_plot_df = prepare_dataframe(agg_df)
    kv_plot_df = aggregate_raw_by_kv(raw_df)

    outputs = {
        "power_heatmap": os.path.join(output_dir, "decode_power_heatmap.png"),
        "tbt_heatmap": os.path.join(output_dir, "decode_tbt_heatmap.png"),
        "tbt_line": os.path.join(output_dir, "decode_tbt_by_batch.png"),
        "power_line": os.path.join(output_dir, "decode_power_by_kv.png"),
        "report": os.path.join(output_dir, "decode_modeling_report.md"),
    }

    plot_heatmap(
        kv_plot_df,
        value_col="avg_power_w",
        title="Batch Size / KV Pressure vs Decode Power",
        output_path=outputs["power_heatmap"],
        cmap="YlOrRd",
    )
    plot_heatmap(
        kv_plot_df,
        value_col="avg_tbt_ms",
        title="Batch Size / KV Pressure vs Decode TBT",
        output_path=outputs["tbt_heatmap"],
        cmap="Blues",
    )
    plot_tbt_by_batch(agg_plot_df, outputs["tbt_line"])
    plot_power_by_kv(kv_plot_df, outputs["power_line"])
    generate_report(agg_plot_df, outputs["report"])
    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="解码阶段离线建模结果分析")
    parser.add_argument("--result-dir", type=str, default="results_decode/decode_modeling",
                        help="实验结果目录")
    parser.add_argument("--result-dirs", type=str, default=None,
                        help="多个实验结果目录，逗号分隔；设置后会覆盖 --result-dir")
    parser.add_argument("--output-dir", type=str, default="results_decode/decode_modeling/images",
                        help="图表输出目录")
    parser.add_argument("--merge-all-runs", action="store_true",
                        help="合并目录下所有实验结果文件后再绘图")
    args = parser.parse_args()

    result_dirs = (
        [item.strip() for item in args.result_dirs.split(",") if item.strip()]
        if args.result_dirs
        else [args.result_dir]
    )
    result_set = load_result_sets_from_dirs(result_dirs, args.merge_all_runs)

    agg_df = result_set["aggregated"]
    raw_df = result_set["raw"]
    if agg_df is None or agg_df.empty or raw_df is None or raw_df.empty:
        raise SystemExit(1)

    outputs = generate_visualizations(agg_df, raw_df, args.output_dir)
    print("分析完成，输出文件：")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
