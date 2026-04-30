#!/usr/bin/env python3
"""
Power-cap 能量扫描分析脚本。
"""
import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


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
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def fit_energy_curve(df: pd.DataFrame) -> Dict:
    points = (
        df[["power_cap_w", "energy_per_output_token_j"]]
        .dropna()
        .drop_duplicates("power_cap_w")
        .sort_values("power_cap_w")
    )
    powers = points["power_cap_w"].astype(float).to_numpy()
    energies = points["energy_per_output_token_j"].astype(float).to_numpy()
    if len(points) < 3:
        return {
            "a": None,
            "b": None,
            "c": None,
            "raw_recommended_power_cap_w": None,
            "recommended_power_cap_w": None,
            "is_valid": False,
            "reason": "need at least 3 unique power caps",
        }

    a, b, c = np.polyfit(powers, energies, 2)
    if not all(math.isfinite(value) for value in [a, b, c]) or a <= 0:
        return {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "raw_recommended_power_cap_w": None,
            "recommended_power_cap_w": None,
            "is_valid": False,
            "reason": "non-convex quadratic fit",
        }

    raw_power = -float(b) / (2.0 * float(a))
    recommended = round(clamp(raw_power, float(powers.min()), float(powers.max())), 6)
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "raw_recommended_power_cap_w": float(raw_power),
        "recommended_power_cap_w": float(recommended),
        "is_valid": True,
        "reason": "ok",
    }


def build_recommendations(df: pd.DataFrame) -> List[Dict]:
    rows = []
    for (query_count, output_length), group in df.groupby(["query_count", "output_length"]):
        sorted_group = group.sort_values(["energy_per_output_token_j", "power_cap_w"])
        best = sorted_group.iloc[0]
        fit = fit_energy_curve(group)
        rows.append({
            "query_count": int(query_count),
            "output_length": int(output_length),
            "target_input_tokens": int(round(float(best.get("target_input_tokens", 0)))),
            "actual_input_tokens": float(group["actual_input_tokens"].mean()) if "actual_input_tokens" in group else 0.0,
            "measured_best_power_cap_w": int(best["power_cap_w"]),
            "measured_best_energy_per_output_token_j": float(best["energy_per_output_token_j"]),
            "fit_recommended_power_cap_w": fit["recommended_power_cap_w"],
            "fit_raw_recommended_power_cap_w": fit["raw_recommended_power_cap_w"],
            "fit_valid": bool(fit["is_valid"]),
            "fit_reason": fit["reason"],
            "fit_a": fit["a"],
            "fit_b": fit["b"],
            "fit_c": fit["c"],
        })
    return sorted(rows, key=lambda row: (row["query_count"], row["output_length"]))


def build_transfer_table(df: pd.DataFrame, recommendations: List[Dict]) -> pd.DataFrame:
    rows = []
    grouped = {
        (int(query_count), int(output_length)): group.copy()
        for (query_count, output_length), group in df.groupby(["query_count", "output_length"])
    }
    for source in recommendations:
        source_power = source["measured_best_power_cap_w"]
        for target_key, target_df in grouped.items():
            target_best = target_df.sort_values(["energy_per_output_token_j", "power_cap_w"]).iloc[0]
            target_df = target_df.copy()
            target_df["distance"] = (target_df["power_cap_w"] - source_power).abs()
            transferred = target_df.sort_values(["distance", "power_cap_w"]).iloc[0]
            best_energy = float(target_best["energy_per_output_token_j"])
            transfer_energy = float(transferred["energy_per_output_token_j"])
            rows.append({
                "source_query_count": source["query_count"],
                "source_output_length": source["output_length"],
                "source_measured_best_power_cap_w": source_power,
                "target_query_count": int(target_key[0]),
                "target_output_length": int(target_key[1]),
                "target_measured_best_power_cap_w": int(target_best["power_cap_w"]),
                "transferred_power_cap_w": int(transferred["power_cap_w"]),
                "target_best_energy_per_output_token_j": best_energy,
                "transferred_energy_per_output_token_j": transfer_energy,
                "energy_over_best_pct": ((transfer_energy / best_energy) - 1.0) * 100.0 if best_energy > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def plot_energy_curve(df: pd.DataFrame, output_path: str):
    plot_df = df.copy()
    plot_df["load"] = plot_df.apply(
        lambda row: f"q={int(row['query_count'])}, out={int(row['output_length'])}",
        axis=1,
    )
    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=plot_df.sort_values("power_cap_w"),
        x="power_cap_w",
        y="energy_per_output_token_j",
        hue="load",
        marker="o",
    )
    plt.title("Energy per Output Token by Power Cap", fontsize=14, pad=16)
    plt.xlabel("Power Cap (W)")
    plt.ylabel("Energy per Output Token (J/token)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fit_curve(df: pd.DataFrame, recommendations: List[Dict], output_path: str):
    plt.figure(figsize=(11, 7))
    for recommendation in recommendations:
        query_count = recommendation["query_count"]
        output_length = recommendation["output_length"]
        group = df[
            (df["query_count"] == query_count)
            & (df["output_length"] == output_length)
        ].sort_values("power_cap_w")
        label = f"q={query_count}, out={output_length}"
        plt.scatter(group["power_cap_w"], group["energy_per_output_token_j"], label=f"{label} measured")
        if recommendation["fit_valid"]:
            powers = np.linspace(float(group["power_cap_w"].min()), float(group["power_cap_w"].max()), 100)
            a = float(recommendation["fit_a"])
            b = float(recommendation["fit_b"])
            c = float(recommendation["fit_c"])
            plt.plot(powers, a * powers ** 2 + b * powers + c, linestyle="--", label=f"{label} fit")
            plt.axvline(
                float(recommendation["fit_recommended_power_cap_w"]),
                color="gray",
                linewidth=0.8,
                alpha=0.35,
            )
    plt.title("Quadratic Fit for Energy per Output Token", fontsize=14, pad=16)
    plt.xlabel("Power Cap (W)")
    plt.ylabel("Energy per Output Token (J/token)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_report(recommendations: List[Dict], transfer_df: pd.DataFrame, output_path: str):
    lines = [
        "# Power-Cap Energy Sweep Report",
        "",
        "## Best Power Cap By Load",
        "",
        "| Query Count | Output Length | Measured Best W | J/output token | Fit Recommended W | Fit Valid |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in recommendations:
        fit_power = row["fit_recommended_power_cap_w"]
        fit_power_text = f"{fit_power:.1f}" if fit_power is not None else "N/A"
        lines.append(
            f"| {row['query_count']} | {row['output_length']} | "
            f"{row['measured_best_power_cap_w']} | "
            f"{row['measured_best_energy_per_output_token_j']:.6f} | "
            f"{fit_power_text} | {row['fit_valid']} |"
        )

    if not transfer_df.empty:
        mean_overhead = transfer_df["energy_over_best_pct"].mean()
        max_overhead = transfer_df["energy_over_best_pct"].max()
        lines.extend([
            "",
            "## Transfer Summary",
            f"- Mean energy overhead versus each target measured optimum: {mean_overhead:.2f}%",
            f"- Max energy overhead versus each target measured optimum: {max_overhead:.2f}%",
        ])

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def generate_outputs(agg_df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    recommendations = build_recommendations(agg_df)
    transfer_df = build_transfer_table(agg_df, recommendations)

    outputs = {
        "energy_curve": os.path.join(output_dir, "power_cap_energy_curve.png"),
        "fit_curve": os.path.join(output_dir, "power_cap_quadratic_fit.png"),
        "best_table": os.path.join(output_dir, "power_cap_best_table.csv"),
        "transfer_table": os.path.join(output_dir, "power_cap_transfer_table.csv"),
        "recommendations": os.path.join(output_dir, "power_cap_recommendations.json"),
        "report": os.path.join(output_dir, "power_cap_energy_report.md"),
    }
    plot_energy_curve(agg_df, outputs["energy_curve"])
    plot_fit_curve(agg_df, recommendations, outputs["fit_curve"])
    pd.DataFrame(recommendations).to_csv(outputs["best_table"], index=False)
    transfer_df.to_csv(outputs["transfer_table"], index=False)
    with open(outputs["recommendations"], "w", encoding="utf-8") as file_obj:
        json.dump({"loads": recommendations}, file_obj, ensure_ascii=False, indent=2)
    write_report(recommendations, transfer_df, outputs["report"])
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze power-cap energy sweep results.")
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
