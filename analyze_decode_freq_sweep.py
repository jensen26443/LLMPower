#!/usr/bin/env python3
"""
Decode 频率扫频结果分析。
"""
import argparse
import glob
import json
import math
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_latest_csv(result_dir: str, suffix: str) -> Optional[pd.DataFrame]:
    pattern = os.path.join(result_dir, f"*_{suffix}.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"加载文件: {latest_file}")
    return pd.read_csv(latest_file)


def geometric_mean(values):
    positive = [float(value) for value in values if float(value) > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def annotate_suspicious_batches(df: pd.DataFrame) -> pd.DataFrame:
    annotated = df.copy()
    annotated["is_suspicious"] = False
    annotated["suspicious_reason"] = ""
    for (_, _, _, _), group in annotated.groupby(["strategy", "bucket_name", "clock_profile_name", "query_count"]):
        median = float(group["avg_tbt_ms"].median())
        mad = float(np.median(np.abs(group["avg_tbt_ms"] - median)))
        threshold = max(6.0 * mad, median * 0.35, 5.0)
        for index, row in group.iterrows():
            deviation = abs(float(row["avg_tbt_ms"]) - median)
            ratio = float(row["avg_tbt_ms"]) / median if median > 0 else 1.0
            reasons = []
            if deviation > threshold and (ratio > 1.5 or ratio < 0.6):
                reasons.append("tbt_outlier_vs_group")
            if median > 0 and float(row["avg_tbt_ms"]) > median * 3.0:
                reasons.append("extreme_high_tbt")
            if reasons:
                annotated.at[index, "is_suspicious"] = True
                annotated.at[index, "suspicious_reason"] = ",".join(reasons)
    return annotated


def aggregate_filtered(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "strategy",
                "bucket_name",
                "query_count",
                "target_input_tokens",
                "output_length",
                "prefill_power_limit",
                "decode_power_scheme",
                "clock_profile_name",
                "sm_clock_mhz",
                "mem_clock_mhz",
            ],
            as_index=False,
        )
        .agg(
            num_samples=("batch_repeat", "count"),
            avg_ttft_ms=("avg_ttft_ms", "mean"),
            avg_tbt_ms=("avg_tbt_ms", "mean"),
            avg_e2e_ms=("avg_e2e_ms", "mean"),
            avg_energy_j=("total_energy_j", "mean"),
            avg_power_w=("avg_power_w", "mean"),
        )
        .sort_values(["strategy", "bucket_name", "clock_profile_name"])
    )


def compute_relative_rows(filtered_agg: pd.DataFrame) -> pd.DataFrame:
    baseline_df = (
        filtered_agg[filtered_agg["strategy"] == "baseline_decode_profile"]
        .set_index("bucket_name")
        .sort_index()
    )
    rows = []
    sweep_df = filtered_agg[filtered_agg["strategy"] == "decode_freq_sweep"]
    for _, row in sweep_df.iterrows():
        bucket_name = row["bucket_name"]
        if bucket_name not in baseline_df.index:
            continue
        baseline = baseline_df.loc[bucket_name]
        energy_saving = (1.0 - float(row["avg_energy_j"]) / float(baseline["avg_energy_j"])) * 100.0
        tbt_increase = (float(row["avg_tbt_ms"]) / float(baseline["avg_tbt_ms"]) - 1.0) * 100.0
        rows.append({
            **row.to_dict(),
            "energy_saving_pct": energy_saving,
            "tbt_increase_pct": tbt_increase,
        })
    return pd.DataFrame(rows)


def build_recommendation(relative_df: pd.DataFrame) -> Dict:
    buckets = {}
    for bucket_name, group in relative_df.groupby("bucket_name"):
        candidates = []
        for (_, sm_clock_mhz, mem_clock_mhz), profile_group in group.groupby(
            ["clock_profile_name", "sm_clock_mhz", "mem_clock_mhz"]
        ):
            max_tbt = float(profile_group["tbt_increase_pct"].max())
            energy_saving = geometric_mean(
                [1.0 + float(value) / 100.0 for value in profile_group["energy_saving_pct"]]
            )
            candidates.append({
                "clock_profile_name": profile_group["clock_profile_name"].iloc[0],
                "sm_clock_mhz": int(profile_group["sm_clock_mhz"].iloc[0]),
                "mem_clock_mhz": int(profile_group["mem_clock_mhz"].iloc[0]),
                "base_power_w": int(profile_group["decode_power_scheme"].iloc[0].split("/")[-1]),
                "max_tbt_increase_pct": max_tbt,
                "geomean_energy_saving_pct": (energy_saving - 1.0) * 100.0,
            })
        valid = [item for item in candidates if item["max_tbt_increase_pct"] <= 3.0]
        if valid:
            best = max(valid, key=lambda item: (item["geomean_energy_saving_pct"], -item["sm_clock_mhz"], -item["mem_clock_mhz"]))
            status = "ok"
        else:
            best = max(candidates, key=lambda item: item["geomean_energy_saving_pct"]) if candidates else None
            status = "unsatisfied"
        buckets[bucket_name] = {
            "status": status,
            **(best or {
                "clock_profile_name": None,
                "sm_clock_mhz": None,
                "mem_clock_mhz": None,
                "base_power_w": None,
                "max_tbt_increase_pct": None,
                "geomean_energy_saving_pct": None,
            }),
        }
    return {
        "status": "ok",
        "constraint": {"tbt_increase_pct_max": 3.0},
        "buckets": buckets,
    }


def plot_relative_metric(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, output_path: str):
    if df.empty:
        return
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x="bucket_name", y=metric_col, hue="clock_profile_name")
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title(title)
    plt.xlabel("Decode Bucket")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_report(filtered_agg: pd.DataFrame, relative_df: pd.DataFrame, suspicious_count: int, output_path: str):
    recommendation = build_recommendation(relative_df)
    lines = [
        "# Decode Frequency Sweep Report",
        "",
        f"- Suspicious batch count: {suspicious_count}",
        f"- Filtered sample count: {len(filtered_agg)}",
        "",
        "## Bucket Recommendation",
    ]
    for bucket_name, payload in recommendation["buckets"].items():
        lines.append(
            f"- {bucket_name}: status={payload['status']}, base_power={payload['base_power_w']}W, "
            f"sm={payload['sm_clock_mhz']}, mem={payload['mem_clock_mhz']}, "
            f"max_tbt_increase={payload['max_tbt_increase_pct']}, "
            f"geomean_energy_saving={payload['geomean_energy_saving_pct']}"
        )
    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def generate_outputs(result_dir: str, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    raw_df = load_latest_csv(result_dir, "raw")
    if raw_df is None or raw_df.empty:
        raise RuntimeError("No raw data found")

    annotated = annotate_suspicious_batches(raw_df)
    suspicious = annotated[annotated["is_suspicious"]].copy()
    filtered_raw = annotated[~annotated["is_suspicious"]].copy()
    filtered_agg = aggregate_filtered(filtered_raw)
    relative_df = compute_relative_rows(filtered_agg)
    recommendation = build_recommendation(relative_df)

    outputs = {
        "suspicious": os.path.join(output_dir, "decode_freq_sweep_suspicious_batches.csv"),
        "filtered_raw": os.path.join(output_dir, "decode_freq_sweep_filtered_raw.csv"),
        "filtered_agg": os.path.join(output_dir, "decode_freq_sweep_filtered_aggregated.csv"),
        "tbt_increase": os.path.join(output_dir, "decode_freq_sweep_tbt_increase.png"),
        "energy_saving": os.path.join(output_dir, "decode_freq_sweep_energy_saving.png"),
        "recommendation": os.path.join(output_dir, "decode_freq_recommendation.json"),
        "report": os.path.join(output_dir, "decode_freq_sweep_report.md"),
    }

    suspicious.to_csv(outputs["suspicious"], index=False)
    filtered_raw.to_csv(outputs["filtered_raw"], index=False)
    filtered_agg.to_csv(outputs["filtered_agg"], index=False)
    with open(outputs["recommendation"], "w", encoding="utf-8") as file_obj:
        json.dump(recommendation, file_obj, ensure_ascii=False, indent=2)
    plot_relative_metric(
        relative_df,
        "tbt_increase_pct",
        "Decode TBT Increase by Clock Profile",
        "TBT Increase (%)",
        outputs["tbt_increase"],
    )
    plot_relative_metric(
        relative_df,
        "energy_saving_pct",
        "Decode Energy Saving by Clock Profile",
        "Energy Saving (%)",
        outputs["energy_saving"],
    )
    write_report(filtered_agg, relative_df, len(suspicious), outputs["report"])
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Analyze decode frequency sweep results.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = generate_outputs(args.result_dir, args.output_dir)
    print("生成文件:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
