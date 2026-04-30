#!/usr/bin/env python3
"""
Analyze decode fixed power-cap batch sweep results.
"""
import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
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


def round_float(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def build_best_power_table(df: pd.DataFrame,
                           min_power_cap: int = 150,
                           plateau_threshold_pct: float = 1.0) -> List[Dict]:
    rows = []
    for (query_count, output_length), group in df.groupby(["query_count", "output_length"]):
        group = group.sort_values("decode_power_cap_w").copy()
        best = group.sort_values(["energy_per_output_token_j", "decode_power_cap_w"]).iloc[0]
        best_power = int(best["decode_power_cap_w"])
        best_energy = float(best["energy_per_output_token_j"])
        higher = group[group["decode_power_cap_w"] > best_power].sort_values("decode_power_cap_w")
        if higher.empty:
            plateau_status = "no_higher_neighbor"
            next_energy_gain_pct = None
        else:
            next_row = higher.iloc[0]
            next_energy = float(next_row["energy_per_output_token_j"])
            next_energy_gain_pct = ((next_energy / best_energy) - 1.0) * 100.0 if best_energy > 0 else 0.0
            plateau_status = "plateau_near_best" if next_energy_gain_pct <= plateau_threshold_pct else "clear_best"
        boundary_status = "interior"
        if best_power <= int(min_power_cap):
            boundary_status = "lower_bound"
        elif best_power >= int(group["decode_power_cap_w"].max()):
            boundary_status = "upper_bound"
        rows.append({
            "query_count": int(query_count),
            "output_length": int(output_length),
            "target_input_tokens": int(round(float(best.get("target_input_tokens", 0)))),
            "actual_input_tokens": float(group["actual_input_tokens"].mean()) if "actual_input_tokens" in group else 0.0,
            "first_kvb": float(group["first_kvb"].mean()) if "first_kvb" in group else 0.0,
            "best_decode_power_cap_w": best_power,
            "best_energy_per_output_token_j": best_energy,
            "best_avg_tbt_ms": float(best.get("avg_tbt_ms", 0.0)),
            "best_avg_e2e_ms": float(best.get("avg_e2e_ms", 0.0)),
            "boundary_status": boundary_status,
            "plateau_status": plateau_status,
            "next_higher_energy_over_best_pct": next_energy_gain_pct,
        })
    return sorted(rows, key=lambda row: (row["query_count"], row["output_length"]))


def choose_balanced_row(group: pd.DataFrame, best_energy: float, energy_slack_pct: float) -> pd.Series:
    candidates = group[
        group["energy_per_output_token_j"] <= best_energy * (1.0 + float(energy_slack_pct) / 100.0)
    ].copy()
    if candidates.empty:
        candidates = group.copy()
    return candidates.sort_values(["avg_tbt_ms", "avg_e2e_ms", "decode_power_cap_w"]).iloc[0]


def choose_tbt_guarded_row(group: pd.DataFrame,
                           tbt_baseline_power_cap: int,
                           max_tbt_increase_pct: float) -> pd.Series:
    baseline_candidates = group[group["decode_power_cap_w"] == int(tbt_baseline_power_cap)]
    if baseline_candidates.empty:
        baseline = group.sort_values(["avg_tbt_ms", "decode_power_cap_w"]).iloc[0]
    else:
        baseline = baseline_candidates.iloc[0]
    max_tbt = float(baseline["avg_tbt_ms"]) * (1.0 + float(max_tbt_increase_pct) / 100.0)
    candidates = group[group["avg_tbt_ms"] <= max_tbt].copy()
    if candidates.empty:
        candidates = group.sort_values(["avg_tbt_ms", "energy_per_output_token_j", "decode_power_cap_w"]).head(1)
    return candidates.sort_values(["energy_per_output_token_j", "decode_power_cap_w"]).iloc[0]


def build_bucket_recommendations(df: pd.DataFrame,
                                 min_power_cap: int = 150,
                                 energy_slack_pct: float = 3.0,
                                 tbt_baseline_power_cap: int = 350,
                                 max_tbt_increase_pct: float = 5.0,
                                 plateau_threshold_pct: float = 1.0) -> Dict:
    profiles = {
        "decode_energy_saving": {"loads": []},
        "decode_balanced": {"loads": []},
        "decode_tbt_guarded": {"loads": []},
    }
    best_rows = build_best_power_table(
        df,
        min_power_cap=min_power_cap,
        plateau_threshold_pct=plateau_threshold_pct,
    )
    best_by_key = {(row["query_count"], row["output_length"]): row for row in best_rows}

    for (query_count, output_length), group in df.groupby(["query_count", "output_length"]):
        group = group.sort_values("decode_power_cap_w").copy()
        best_summary = best_by_key[(int(query_count), int(output_length))]
        best_row = group[group["decode_power_cap_w"] == best_summary["best_decode_power_cap_w"]].iloc[0]
        balanced = choose_balanced_row(
            group,
            best_energy=float(best_summary["best_energy_per_output_token_j"]),
            energy_slack_pct=energy_slack_pct,
        )
        tbt_guarded = choose_tbt_guarded_row(
            group,
            tbt_baseline_power_cap=tbt_baseline_power_cap,
            max_tbt_increase_pct=max_tbt_increase_pct,
        )
        for profile_name, row in [
            ("decode_energy_saving", best_row),
            ("decode_balanced", balanced),
            ("decode_tbt_guarded", tbt_guarded),
        ]:
            baseline_rows = group[group["decode_power_cap_w"] == int(tbt_baseline_power_cap)]
            if baseline_rows.empty:
                baseline_row = group.sort_values(["avg_tbt_ms", "decode_power_cap_w"]).iloc[0]
            else:
                baseline_row = baseline_rows.iloc[0]
            baseline_tbt = float(baseline_row["avg_tbt_ms"])
            tbt_increase = (
                (float(row["avg_tbt_ms"]) / baseline_tbt) - 1.0
            ) * 100.0 if baseline_tbt > 0 else 0.0
            energy_over_best = (
                (float(row["energy_per_output_token_j"]) / float(best_summary["best_energy_per_output_token_j"])) - 1.0
            ) * 100.0
            profiles[profile_name]["loads"].append({
                "query_count": int(query_count),
                "output_length": int(output_length),
                "target_input_tokens": int(round(float(row.get("target_input_tokens", 0)))),
                "actual_input_tokens": float(group["actual_input_tokens"].mean()) if "actual_input_tokens" in group else 0.0,
                "first_kvb": float(group["first_kvb"].mean()) if "first_kvb" in group else 0.0,
                "decode_power_cap_w": int(row["decode_power_cap_w"]),
                "energy_per_output_token_j": float(row["energy_per_output_token_j"]),
                "energy_over_best_pct": round_float(energy_over_best),
                "avg_tbt_ms": float(row["avg_tbt_ms"]),
                "avg_e2e_ms": float(row["avg_e2e_ms"]),
                "tbt_baseline_power_cap_w": int(baseline_row["decode_power_cap_w"]),
                "baseline_avg_tbt_ms": baseline_tbt,
                "tbt_increase_over_baseline_pct": round_float(tbt_increase),
                "tbt_guard_met": bool(tbt_increase <= float(max_tbt_increase_pct)),
                "boundary_status": best_summary["boundary_status"],
                "plateau_status": best_summary["plateau_status"],
            })
    return {
        "schema_version": 1,
        "selection": {
            "min_power_cap_w": int(min_power_cap),
            "energy_slack_pct": float(energy_slack_pct),
            "tbt_baseline_power_cap_w": int(tbt_baseline_power_cap),
            "max_tbt_increase_pct": float(max_tbt_increase_pct),
            "plateau_threshold_pct": float(plateau_threshold_pct),
        },
        "profiles": profiles,
    }


def build_marginal_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (query_count, output_length), group in df.groupby(["query_count", "output_length"]):
        ordered = group.sort_values("decode_power_cap_w")
        previous = None
        for _, row in ordered.iterrows():
            if previous is not None:
                rows.append({
                    "query_count": int(query_count),
                    "output_length": int(output_length),
                    "lower_power_cap_w": int(previous["decode_power_cap_w"]),
                    "higher_power_cap_w": int(row["decode_power_cap_w"]),
                    "lower_energy_per_output_token_j": float(previous["energy_per_output_token_j"]),
                    "higher_energy_per_output_token_j": float(row["energy_per_output_token_j"]),
                    "energy_increase_pct": (
                        (float(row["energy_per_output_token_j"]) / float(previous["energy_per_output_token_j"])) - 1.0
                    ) * 100.0 if float(previous["energy_per_output_token_j"]) > 0 else 0.0,
                })
            previous = row
    return pd.DataFrame(rows)


def plot_energy_curve(df: pd.DataFrame, output_path: str):
    plot_df = df.copy()
    plot_df["load"] = plot_df.apply(
        lambda row: f"q={int(row['query_count'])}, out={int(row['output_length'])}",
        axis=1,
    )
    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=plot_df.sort_values("decode_power_cap_w"),
        x="decode_power_cap_w",
        y="energy_per_output_token_j",
        hue="load",
        marker="o",
    )
    plt.title("Decode Energy per Output Token by Power Cap")
    plt.xlabel("Decode Power Cap (W)")
    plt.ylabel("Energy per Output Token (J/token)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_marginal_curve(marginal_df: pd.DataFrame, output_path: str):
    if marginal_df.empty:
        return
    plot_df = marginal_df.copy()
    plot_df["load"] = plot_df.apply(
        lambda row: f"q={int(row['query_count'])}, out={int(row['output_length'])}",
        axis=1,
    )
    plot_df["edge"] = plot_df.apply(
        lambda row: f"{int(row['lower_power_cap_w'])}->{int(row['higher_power_cap_w'])}",
        axis=1,
    )
    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df, x="edge", y="energy_increase_pct", hue="load")
    plt.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    plt.title("Adjacent Decode Power Marginal Energy Increase")
    plt.xlabel("Adjacent Power Caps (W)")
    plt.ylabel("Energy Increase (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_report(best_rows: List[Dict], output_path: str):
    lines = [
        "# Decode Power-Cap Batch Sweep Report",
        "",
        "## Best Decode Power By Load",
        "",
        "| Query Count | Output Length | Best Decode W | J/output token | Boundary | Plateau |",
        "|---:|---:|---:|---:|:---|:---|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['query_count']} | {row['output_length']} | "
            f"{row['best_decode_power_cap_w']} | "
            f"{row['best_energy_per_output_token_j']:.6f} | "
            f"{row['boundary_status']} | {row['plateau_status']} |"
        )
    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def generate_outputs(agg_df: pd.DataFrame,
                     output_dir: str,
                     min_power_cap: int = 150,
                     energy_slack_pct: float = 3.0,
                     tbt_baseline_power_cap: int = 350,
                     max_tbt_increase_pct: float = 5.0,
                     plateau_threshold_pct: float = 1.0) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    best_rows = build_best_power_table(
        agg_df,
        min_power_cap=min_power_cap,
        plateau_threshold_pct=plateau_threshold_pct,
    )
    recommendations = build_bucket_recommendations(
        agg_df,
        min_power_cap=min_power_cap,
        energy_slack_pct=energy_slack_pct,
        tbt_baseline_power_cap=tbt_baseline_power_cap,
        max_tbt_increase_pct=max_tbt_increase_pct,
        plateau_threshold_pct=plateau_threshold_pct,
    )
    marginal_df = build_marginal_table(agg_df)
    outputs = {
        "best_table": os.path.join(output_dir, "best_decode_power_by_load.csv"),
        "bucket_recommendations": os.path.join(output_dir, "decode_bucket_recommendations.json"),
        "marginal_table": os.path.join(output_dir, "decode_power_marginal_table.csv"),
        "energy_curve": os.path.join(output_dir, "decode_power_energy_curve.png"),
        "marginal_curve": os.path.join(output_dir, "decode_power_marginal_curve.png"),
        "report": os.path.join(output_dir, "decode_power_cap_batch_sweep_report.md"),
    }
    pd.DataFrame(best_rows).to_csv(outputs["best_table"], index=False)
    marginal_df.to_csv(outputs["marginal_table"], index=False)
    with open(outputs["bucket_recommendations"], "w", encoding="utf-8") as file_obj:
        json.dump(recommendations, file_obj, ensure_ascii=False, indent=2)
    plot_energy_curve(agg_df, outputs["energy_curve"])
    plot_marginal_curve(marginal_df, outputs["marginal_curve"])
    write_report(best_rows, outputs["report"])
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze decode fixed power-cap batch sweep.")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--result-dirs", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-power-cap", type=int, default=150)
    parser.add_argument("--energy-slack-pct", type=float, default=3.0)
    parser.add_argument("--tbt-baseline-power-cap", type=int, default=350)
    parser.add_argument("--max-tbt-increase-pct", type=float, default=5.0)
    parser.add_argument("--plateau-threshold-pct", type=float, default=1.0)
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
    outputs = generate_outputs(
        agg_df,
        args.output_dir,
        min_power_cap=args.min_power_cap,
        energy_slack_pct=args.energy_slack_pct,
        tbt_baseline_power_cap=args.tbt_baseline_power_cap,
        max_tbt_increase_pct=args.max_tbt_increase_pct,
        plateau_threshold_pct=args.plateau_threshold_pct,
    )
    print("生成文件:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
