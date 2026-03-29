#!/usr/bin/env python3
"""
TBT 与 Batch Size 线性建模脚本。

输出：
- 基于 raw repeat 级样本的整体拟合图与残差图
- 基于 raw repeat 级样本的分段拟合图与残差图
- 拟合系数、R^2、MAE、MAPE 的 Markdown 汇总
"""
import glob
import os
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OVERALL_TARGETS = [20, 40, 50, 75, 100, 150, 200, 300]
SHORT_TARGETS = [10, 20]
REGULAR_TARGETS = [40, 50, 75, 100, 150, 200, 300]


def load_latest_raw_csv(result_dir: str) -> pd.DataFrame:
    pattern = os.path.join(result_dir, "*_raw.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"没有找到 raw 文件: {pattern}")

    latest_file = max(files, key=os.path.getctime)
    print(f"加载文件: {latest_file}")
    return pd.read_csv(latest_file)


def filter_targets(df: pd.DataFrame, targets: Iterable[int]) -> pd.DataFrame:
    targets = list(targets)
    filtered = df[df["target_output_tokens"].isin(targets)].copy()
    if filtered.empty:
        raise ValueError(f"目标输出长度没有匹配数据: {targets}")
    return filtered.sort_values(["target_output_tokens", "batch_size"])


def add_repeat_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    if "ttft_ratio" not in enriched.columns:
        if "avg_ttft_ms" in enriched.columns and "avg_e2e_ms" in enriched.columns:
            denominator = enriched["avg_e2e_ms"].replace(0, np.nan)
            enriched["ttft_ratio"] = (enriched["avg_ttft_ms"] / denominator).fillna(0.0)
        else:
            enriched["ttft_ratio"] = 0.0

    if "stream_chunk_ratio" not in enriched.columns:
        if "avg_stream_chunk_count" in enriched.columns and "avg_generated_tokens" in enriched.columns:
            denominator = enriched["avg_generated_tokens"].replace(0, np.nan)
            enriched["stream_chunk_ratio"] = (enriched["avg_stream_chunk_count"] / denominator).fillna(0.0)
        else:
            enriched["stream_chunk_ratio"] = np.nan

    return enriched


def annotate_suspicious_repeats(df: pd.DataFrame) -> pd.DataFrame:
    annotated = add_repeat_diagnostics(df)
    annotated["group_median_tbt_ms"] = np.nan
    annotated["group_mad_tbt_ms"] = np.nan
    annotated["tbt_deviation_ratio"] = np.nan
    annotated["suspicious_reason"] = ""
    annotated["is_suspicious"] = False

    for (_, _), group in annotated.groupby(["batch_size", "target_output_tokens"]):
        median_tbt = float(group["avg_tbt_ms"].median())
        mad_tbt = float(np.median(np.abs(group["avg_tbt_ms"] - median_tbt)))

        reasons_by_index: Dict[int, List[str]] = {}
        for index, row in group.iterrows():
            reasons: List[str] = []
            deviation_ratio = (row["avg_tbt_ms"] / median_tbt) if median_tbt > 0 else 1.0

            if median_tbt > 0 and deviation_ratio < 0.85:
                reasons.append("low_tbt_vs_group")
            if median_tbt > 0 and deviation_ratio > 1.15:
                reasons.append("high_tbt_vs_group")
            if row["ttft_ratio"] > 0.60:
                reasons.append("high_ttft_ratio")
            if not np.isnan(row["stream_chunk_ratio"]) and row["stream_chunk_ratio"] < 0.50:
                reasons.append("low_stream_chunk_ratio")

            annotated.at[index, "group_median_tbt_ms"] = median_tbt
            annotated.at[index, "group_mad_tbt_ms"] = mad_tbt
            annotated.at[index, "tbt_deviation_ratio"] = deviation_ratio
            if reasons:
                reasons_by_index[index] = reasons

        for index, reasons in reasons_by_index.items():
            annotated.at[index, "is_suspicious"] = True
            annotated.at[index, "suspicious_reason"] = ",".join(reasons)

    return annotated


def fit_linear_model(df: pd.DataFrame) -> Dict[str, object]:
    x = df["batch_size"].to_numpy(dtype=float)
    y = df["avg_tbt_ms"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    predictions = slope * x + intercept
    residuals = y - predictions

    ss_res = float(np.sum((y - predictions) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(residuals)))
    non_zero_mask = y != 0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(residuals[non_zero_mask] / y[non_zero_mask])) * 100.0)
    else:
        mape = 0.0

    fitted_df = df.copy()
    fitted_df["predicted_tbt_ms"] = predictions
    fitted_df["residual_tbt_ms"] = residuals

    return {
        "sample_count": int(len(df)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "mae": mae,
        "mape": mape,
        "fitted_df": fitted_df,
    }


def plot_fit(
    df: pd.DataFrame,
    slope: float,
    intercept: float,
    title: str,
    output_path: str,
):
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=df,
        x="batch_size",
        y="avg_tbt_ms",
        hue="target_output_tokens",
        palette="tab10",
        s=80,
    )
    suspicious_df = df[df["is_suspicious"]] if "is_suspicious" in df.columns else df.iloc[0:0]
    if not suspicious_df.empty:
        plt.scatter(
            suspicious_df["batch_size"],
            suspicious_df["avg_tbt_ms"],
            s=220,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="Suspicious repeat",
        )

    x_values = np.array(sorted(df["batch_size"].unique()), dtype=float)
    y_values = slope * x_values + intercept
    plt.plot(x_values, y_values, color="black", linewidth=2, label="Linear fit")

    plt.title(title, fontsize=14, pad=16)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Average TBT (ms)", fontsize=12)
    plt.legend(title="Target Output Tokens")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_segmented_fit(
    short_df: pd.DataFrame,
    short_model: Dict[str, object],
    regular_df: pd.DataFrame,
    regular_model: Dict[str, object],
    output_path: str,
):
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=short_df,
        x="batch_size",
        y="avg_tbt_ms",
        hue="target_output_tokens",
        palette="tab10",
        s=80,
        marker="o",
    )
    sns.scatterplot(
        data=regular_df,
        x="batch_size",
        y="avg_tbt_ms",
        hue="target_output_tokens",
        palette="tab10",
        s=80,
        marker="X",
        legend=False,
    )
    suspicious_df = pd.concat(
        [
            short_df[short_df["is_suspicious"]] if "is_suspicious" in short_df.columns else short_df.iloc[0:0],
            regular_df[regular_df["is_suspicious"]] if "is_suspicious" in regular_df.columns else regular_df.iloc[0:0],
        ],
        ignore_index=True,
    )
    if not suspicious_df.empty:
        plt.scatter(
            suspicious_df["batch_size"],
            suspicious_df["avg_tbt_ms"],
            s=220,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="Suspicious repeat",
        )

    short_x = np.array(sorted(short_df["batch_size"].unique()), dtype=float)
    short_y = short_model["slope"] * short_x + short_model["intercept"]
    plt.plot(short_x, short_y, color="black", linewidth=2, label="Short-output fit")

    regular_x = np.array(sorted(regular_df["batch_size"].unique()), dtype=float)
    regular_y = regular_model["slope"] * regular_x + regular_model["intercept"]
    plt.plot(regular_x, regular_y, color="dimgray", linewidth=2, linestyle="--", label="Regular-output fit")

    plt.title("Segmented TBT vs Batch Size Linear Fits", fontsize=14, pad=16)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Average TBT (ms)", fontsize=12)
    plt.legend(title="Target Output Tokens / Fit")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals(
    df: pd.DataFrame,
    title: str,
    output_path: str,
):
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=df,
        x="batch_size",
        y="residual_tbt_ms",
        hue="target_output_tokens",
        palette="tab10",
        s=80,
    )
    suspicious_df = df[df["is_suspicious"]] if "is_suspicious" in df.columns else df.iloc[0:0]
    if not suspicious_df.empty:
        plt.scatter(
            suspicious_df["batch_size"],
            suspicious_df["residual_tbt_ms"],
            s=220,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="Suspicious repeat",
        )
    plt.axhline(0.0, color="black", linewidth=1.5, linestyle="--")
    plt.title(title, fontsize=14, pad=16)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Residual (ms)", fontsize=12)
    plt.legend(title="Target Output Tokens")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_segmented_residuals(
    short_df: pd.DataFrame,
    regular_df: pd.DataFrame,
    output_path: str,
):
    combined = pd.concat([short_df, regular_df], ignore_index=True)
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=combined,
        x="batch_size",
        y="residual_tbt_ms",
        hue="segment",
        style="segment",
        s=80,
        palette={"short": "tab:red", "regular": "tab:blue"},
    )
    suspicious_df = combined[combined["is_suspicious"]] if "is_suspicious" in combined.columns else combined.iloc[0:0]
    if not suspicious_df.empty:
        plt.scatter(
            suspicious_df["batch_size"],
            suspicious_df["residual_tbt_ms"],
            s=220,
            facecolors="none",
            edgecolors="black",
            linewidths=2,
            label="Suspicious repeat",
        )
    plt.axhline(0.0, color="black", linewidth=1.5, linestyle="--")
    plt.title("Segmented TBT Residuals", fontsize=14, pad=16)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Residual (ms)", fontsize=12)
    plt.legend(title="Segment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_summary(
    output_path: str,
    suspicious_df: pd.DataFrame,
    overall_targets: List[int],
    overall_model: Dict[str, object],
    short_targets: List[int],
    short_model: Dict[str, object],
    regular_targets: List[int],
    regular_model: Dict[str, object],
):
    lines = [
        "# TBT Batch Modeling Summary",
        "",
        "## Overall Model",
        f"- targets = {overall_targets}",
        f"- sample_count = {overall_model['sample_count']}",
        f"- slope = {overall_model['slope']:.6f}",
        f"- intercept = {overall_model['intercept']:.6f}",
        f"- R^2 = {overall_model['r_squared']:.6f}",
        f"- MAE = {overall_model['mae']:.6f}",
        f"- MAPE = {overall_model['mape']:.6f}%",
        "",
        "## Segmented Model",
        "",
        "### Short Output Segment",
        f"- targets = {short_targets}",
        f"- sample_count = {short_model['sample_count']}",
        f"- slope = {short_model['slope']:.6f}",
        f"- intercept = {short_model['intercept']:.6f}",
        f"- R^2 = {short_model['r_squared']:.6f}",
        f"- MAE = {short_model['mae']:.6f}",
        f"- MAPE = {short_model['mape']:.6f}%",
        "",
        "### Regular Output Segment",
        f"- targets = {regular_targets}",
        f"- sample_count = {regular_model['sample_count']}",
        f"- slope = {regular_model['slope']:.6f}",
        f"- intercept = {regular_model['intercept']:.6f}",
        f"- R^2 = {regular_model['r_squared']:.6f}",
        f"- MAE = {regular_model['mae']:.6f}",
        f"- MAPE = {regular_model['mape']:.6f}%",
    ]

    lines.extend(["", "## Suspicious Repeats"])
    if suspicious_df.empty:
        lines.append("- none")
    else:
        for _, row in suspicious_df.sort_values(["batch_size", "target_output_tokens", "repeat_id"]).iterrows():
            lines.append(
                "- "
                f"batch={int(row['batch_size'])}, "
                f"target={int(row['target_output_tokens'])}, "
                f"repeat={int(row['repeat_id'])}, "
                f"avg_tbt_ms={row['avg_tbt_ms']:.6f}, "
                f"ttft_ratio={row['ttft_ratio']:.6f}, "
                f"reasons={row['suspicious_reason']}"
            )

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def write_anomalies_csv(output_path: str, suspicious_df: pd.DataFrame):
    columns = [
        "repeat_id",
        "batch_size",
        "target_output_tokens",
        "avg_tbt_ms",
        "avg_ttft_ms",
        "avg_e2e_ms",
        "avg_generated_tokens",
        "avg_stream_chunk_count",
        "ttft_ratio",
        "stream_chunk_ratio",
        "group_median_tbt_ms",
        "tbt_deviation_ratio",
        "suspicious_reason",
    ]
    available_columns = [column for column in columns if column in suspicious_df.columns]
    suspicious_df.to_csv(output_path, index=False, columns=available_columns)


def generate_modeling_outputs(
    raw_df: pd.DataFrame,
    output_dir: str,
    overall_targets: Optional[List[int]] = None,
    short_targets: Optional[List[int]] = None,
    regular_targets: Optional[List[int]] = None,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    overall_targets = overall_targets or OVERALL_TARGETS
    short_targets = short_targets or SHORT_TARGETS
    regular_targets = regular_targets or REGULAR_TARGETS

    annotated_df = annotate_suspicious_repeats(raw_df)
    overall_df = filter_targets(annotated_df, overall_targets)
    short_df = filter_targets(annotated_df, short_targets)
    regular_df = filter_targets(annotated_df, regular_targets)

    overall_model = fit_linear_model(overall_df)
    short_model = fit_linear_model(short_df)
    regular_model = fit_linear_model(regular_df)

    short_residual_df = short_model["fitted_df"].copy()
    short_residual_df["segment"] = "short"
    regular_residual_df = regular_model["fitted_df"].copy()
    regular_residual_df["segment"] = "regular"

    outputs = {
        "summary": os.path.join(output_dir, "tbt_batch_modeling_summary.md"),
        "anomalies": os.path.join(output_dir, "tbt_batch_suspicious_repeats.csv"),
        "overall_fit_plot": os.path.join(output_dir, "tbt_batch_overall_fit.png"),
        "overall_residual_plot": os.path.join(output_dir, "tbt_batch_overall_residuals.png"),
        "segmented_fit_plot": os.path.join(output_dir, "tbt_batch_segmented_fit.png"),
        "segmented_residual_plot": os.path.join(output_dir, "tbt_batch_segmented_residuals.png"),
    }

    plot_fit(
        overall_df,
        overall_model["slope"],
        overall_model["intercept"],
        title="Overall TBT vs Batch Size Linear Fit",
        output_path=outputs["overall_fit_plot"],
    )
    plot_residuals(
        overall_model["fitted_df"],
        title="Overall TBT Residuals",
        output_path=outputs["overall_residual_plot"],
    )
    plot_segmented_fit(
        short_df,
        short_model,
        regular_df,
        regular_model,
        output_path=outputs["segmented_fit_plot"],
    )
    plot_segmented_residuals(
        short_residual_df,
        regular_residual_df,
        output_path=outputs["segmented_residual_plot"],
    )
    write_summary(
        outputs["summary"],
        annotated_df[annotated_df["is_suspicious"]].copy(),
        overall_targets,
        overall_model,
        short_targets,
        short_model,
        regular_targets,
        regular_model,
    )
    write_anomalies_csv(outputs["anomalies"], annotated_df[annotated_df["is_suspicious"]].copy())
    return outputs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TBT 与 Batch Size 线性建模")
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results_decode/decode_modeling",
        help="实验结果目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_decode/decode_modeling/modeling",
        help="建模输出目录",
    )
    args = parser.parse_args()

    raw_df = load_latest_raw_csv(args.result_dir)
    outputs = generate_modeling_outputs(raw_df, args.output_dir)

    print("建模完成，输出文件：")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
