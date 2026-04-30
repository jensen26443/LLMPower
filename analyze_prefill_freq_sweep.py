#!/usr/bin/env python3
"""
Prefill 频率扫频结果分析。
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
    for (_, _, _, _), group in annotated.groupby(["strategy", "query_count", "power_limit", "clock_profile_name"]):
        median = float(group["avg_ttft_ms"].median())
        mad = float(np.median(np.abs(group["avg_ttft_ms"] - median)))
        threshold = max(6.0 * mad, median * 0.35, 120.0)
        for index, row in group.iterrows():
            deviation = abs(float(row["avg_ttft_ms"]) - median)
            ratio = float(row["avg_ttft_ms"]) / median if median > 0 else 1.0
            reasons = []
            if deviation > threshold and (ratio > 1.5 or ratio < 0.6):
                reasons.append("ttft_outlier_vs_group")
            if median > 0 and float(row["avg_ttft_ms"]) > median * 3.0:
                reasons.append("extreme_high_ttft")
            if reasons:
                annotated.at[index, "is_suspicious"] = True
                annotated.at[index, "suspicious_reason"] = ",".join(reasons)
    return annotated


def aggregate_filtered(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "strategy",
                "query_count",
                "target_input_tokens",
                "power_limit",
                "clock_profile_name",
                "sm_clock_mhz",
                "mem_clock_mhz",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            num_samples=("batch_repeat", "count"),
            avg_actual_input_tokens=("actual_input_tokens", "mean"),
            avg_ttft_ms=("avg_ttft_ms", "mean"),
            avg_e2e_ms=("avg_e2e_ms", "mean"),
            avg_energy_j=("total_energy_j", "mean"),
            avg_power_w=("avg_power_w", "mean"),
        )
        .sort_values(["strategy", "query_count", "power_limit", "clock_profile_name"])
    )


def get_prefill_bucket_name(power_limit: int) -> str:
    return f"prefill_{int(power_limit)}w"


def compute_relative_rows(filtered_agg: pd.DataFrame) -> pd.DataFrame:
    baseline_df = (
        filtered_agg[filtered_agg["strategy"] == "baseline_prefill_profile"]
        .set_index("query_count")
        .sort_index()
    )
    rows = []
    sweep_df = filtered_agg[filtered_agg["strategy"] == "prefill_freq_sweep"]
    for _, row in sweep_df.iterrows():
        query_count = int(row["query_count"])
        if query_count not in baseline_df.index:
            continue
        baseline = baseline_df.loc[query_count]
        energy_saving = (1.0 - float(row["avg_energy_j"]) / float(baseline["avg_energy_j"])) * 100.0
        ttft_increase = (float(row["avg_ttft_ms"]) / float(baseline["avg_ttft_ms"]) - 1.0) * 100.0
        rows.append({
            **row.to_dict(),
            "prefill_bucket_name": get_prefill_bucket_name(int(row["power_limit"])),
            "energy_saving_pct": energy_saving,
            "ttft_increase_pct": ttft_increase,
        })
    if not rows:
        return pd.DataFrame(
            columns=[
                *filtered_agg.columns.tolist(),
                "prefill_bucket_name",
                "energy_saving_pct",
                "ttft_increase_pct",
            ]
        )
    return pd.DataFrame(rows)


def build_recommendation(relative_df: pd.DataFrame) -> Dict:
    if relative_df.empty or "power_limit" not in relative_df.columns:
        return {
            "status": "empty",
            "constraint": {"ttft_increase_pct_max": 5.0},
            "buckets": {},
        }
    buckets = {}
    for power_limit, group in relative_df.groupby("power_limit"):
        group = group[group["clock_profile_name"] != "baseline_default"].copy()
        if group.empty:
            continue
        candidates = []
        for (_, sm_clock_mhz, mem_clock_mhz), profile_group in group.groupby(
            ["clock_profile_name", "sm_clock_mhz", "mem_clock_mhz"]
        ):
            max_ttft = float(profile_group["ttft_increase_pct"].max())
            energy_saving = geometric_mean(
                [1.0 + float(value) / 100.0 for value in profile_group["energy_saving_pct"]]
            )
            candidates.append({
                "clock_profile_name": profile_group["clock_profile_name"].iloc[0],
                "sm_clock_mhz": int(profile_group["sm_clock_mhz"].iloc[0]),
                "mem_clock_mhz": int(profile_group["mem_clock_mhz"].iloc[0]),
                "max_ttft_increase_pct": max_ttft,
                "geomean_energy_saving_pct": (energy_saving - 1.0) * 100.0,
            })
        valid = [item for item in candidates if item["max_ttft_increase_pct"] <= 5.0]
        if valid:
            best = max(valid, key=lambda item: (item["geomean_energy_saving_pct"], -item["sm_clock_mhz"], -item["mem_clock_mhz"]))
            status = "ok"
        else:
            best = max(candidates, key=lambda item: item["geomean_energy_saving_pct"]) if candidates else None
            status = "unsatisfied"
        bucket_name = get_prefill_bucket_name(int(power_limit))
        buckets[bucket_name] = {
            "status": status,
            "power_w": int(power_limit),
            **(best or {
                "clock_profile_name": None,
                "sm_clock_mhz": None,
                "mem_clock_mhz": None,
                "max_ttft_increase_pct": None,
                "geomean_energy_saving_pct": None,
            }),
        }
    return {
        "status": "ok",
        "constraint": {"ttft_increase_pct_max": 5.0},
        "buckets": buckets,
    }


def build_same_power_dynamic_summary(filtered_agg: pd.DataFrame, recommendation: Dict) -> Dict[str, Dict]:
    summaries: Dict[str, Dict] = {}
    sweep_df = filtered_agg[filtered_agg["strategy"] == "prefill_freq_sweep"].copy()
    for bucket_name, payload in recommendation.get("buckets", {}).items():
        power_w = int(payload["power_w"])
        if payload.get("clock_profile_name") is None:
            continue
        dynamic_df = sweep_df[
            (sweep_df["power_limit"] == power_w)
            & (sweep_df["clock_profile_name"] == "baseline_default")
        ]
        locked_df = sweep_df[
            (sweep_df["power_limit"] == power_w)
            & (sweep_df["clock_profile_name"] == payload["clock_profile_name"])
        ]
        if dynamic_df.empty or locked_df.empty:
            continue
        merged = locked_df.merge(
            dynamic_df[["query_count", "avg_ttft_ms", "avg_energy_j"]],
            on="query_count",
            how="inner",
            suffixes=("_locked", "_dynamic"),
        )
        if merged.empty:
            continue
        ttft_changes = [
            float(row["avg_ttft_ms_locked"]) / float(row["avg_ttft_ms_dynamic"]) - 1.0
            for _, row in merged.iterrows()
            if float(row["avg_ttft_ms_dynamic"]) > 0
        ]
        energy_changes = [
            1.0 - float(row["avg_energy_j_locked"]) / float(row["avg_energy_j_dynamic"])
            for _, row in merged.iterrows()
            if float(row["avg_energy_j_dynamic"]) > 0
        ]
        summaries[bucket_name] = {
            "dynamic_profile_name": "baseline_default",
            "geomean_ttft_change_pct_vs_same_power_dynamic": (geometric_mean([1.0 + value for value in ttft_changes]) - 1.0) * 100.0 if ttft_changes else None,
            "geomean_energy_saving_pct_vs_same_power_dynamic": (geometric_mean([1.0 + value for value in energy_changes]) - 1.0) * 100.0 if energy_changes else None,
        }
    return summaries


def plot_relative_metric(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, output_path: str):
    if df.empty:
        return
    plot_df = df.copy()
    plot_df["query_label"] = plot_df.apply(lambda row: f"{int(row['query_count'])}/{int(row['target_input_tokens'])}", axis=1)
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=plot_df,
        x="query_label",
        y=metric_col,
        hue="clock_profile_name",
    )
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title(title)
    plt.xlabel("Query Count / Target Input Tokens")
    plt.ylabel(ylabel)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_report(filtered_agg: pd.DataFrame, relative_df: pd.DataFrame, suspicious_count: int, output_path: str):
    lines = [
        "# Prefill Frequency Sweep Report",
        "",
        f"- Suspicious batch count: {suspicious_count}",
        f"- Filtered sample count: {len(filtered_agg)}",
        "",
        "## Bucket Recommendation",
    ]
    recommendation = build_recommendation(relative_df)
    same_power_dynamic = build_same_power_dynamic_summary(filtered_agg, recommendation)
    for bucket_name, payload in recommendation["buckets"].items():
        lines.append(
            f"- {bucket_name}: status={payload['status']}, power={payload['power_w']}W, "
            f"sm={payload['sm_clock_mhz']}, mem={payload['mem_clock_mhz']}, "
            f"max_ttft_increase={payload['max_ttft_increase_pct']}, "
            f"geomean_energy_saving={payload['geomean_energy_saving_pct']}"
        )
    if same_power_dynamic:
        lines.extend(["", "## Relative To Same-Power Dynamic"])
        for bucket_name, payload in same_power_dynamic.items():
            lines.append(
                f"- {bucket_name}: dynamic=baseline_default, "
                f"geomean_ttft_change={payload['geomean_ttft_change_pct_vs_same_power_dynamic']}, "
                f"geomean_energy_saving={payload['geomean_energy_saving_pct_vs_same_power_dynamic']}"
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
        "suspicious": os.path.join(output_dir, "prefill_freq_sweep_suspicious_batches.csv"),
        "filtered_raw": os.path.join(output_dir, "prefill_freq_sweep_filtered_raw.csv"),
        "filtered_agg": os.path.join(output_dir, "prefill_freq_sweep_filtered_aggregated.csv"),
        "ttft_increase": os.path.join(output_dir, "prefill_freq_sweep_ttft_increase.png"),
        "energy_saving": os.path.join(output_dir, "prefill_freq_sweep_energy_saving.png"),
        "recommendation": os.path.join(output_dir, "prefill_freq_recommendation.json"),
        "report": os.path.join(output_dir, "prefill_freq_sweep_report.md"),
    }

    suspicious.to_csv(outputs["suspicious"], index=False)
    filtered_raw.to_csv(outputs["filtered_raw"], index=False)
    filtered_agg.to_csv(outputs["filtered_agg"], index=False)
    with open(outputs["recommendation"], "w", encoding="utf-8") as file_obj:
        json.dump(recommendation, file_obj, ensure_ascii=False, indent=2)
    plot_relative_metric(
        relative_df,
        "ttft_increase_pct",
        "Prefill TTFT Increase by Clock Profile",
        "TTFT Increase (%)",
        outputs["ttft_increase"],
    )
    plot_relative_metric(
        relative_df,
        "energy_saving_pct",
        "Prefill Energy Saving by Clock Profile",
        "Energy Saving (%)",
        outputs["energy_saving"],
    )
    write_report(filtered_agg, relative_df, len(suspicious), outputs["report"])
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Analyze prefill frequency sweep results.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = generate_outputs(args.result_dir, args.output_dir)
    print("生成文件:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
