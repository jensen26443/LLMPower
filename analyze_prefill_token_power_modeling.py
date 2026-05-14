#!/usr/bin/env python3
"""Analyze prefill token-power modeling results and generate paper-style figures."""

import argparse
import glob
import json
import os
from typing import Dict, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


PAPER_COLORS = {
    "raw": "#4C78A8",
    "mean": "#1F4E79",
    "fit": "#C44E52",
    "energy": "#DD8452",
    "ttft": "#55A868",
    "peak": "#8172B2",
}


def get_available_font_family(candidates):
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return candidates[-1]


def find_latest_file(input_dir: str, suffix: str) -> str:
    files = glob.glob(os.path.join(input_dir, f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No '*{suffix}' file found in {input_dir}")
    return max(files, key=os.path.getctime)


def apply_origin_style(ax) -> None:
    """Apply image.md style: clean Origin-like scientific plot."""
    serif_font = get_available_font_family(["Times New Roman", "DejaVu Serif"])
    cjk_font = get_available_font_family(["SimSun", "Noto Serif CJK SC", "DejaVu Sans"])
    plt.rcParams.update({
        "font.family": [serif_font, cjk_font],
        "font.serif": [serif_font, cjk_font],
        "font.sans-serif": [cjk_font, serif_font],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
    })
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
    ax.tick_params(axis="both", which="both", direction="in", top=False, right=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(False)
    ax.set_facecolor("white")


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 2) -> Dict:
    """Fit a polynomial and return coefficients, R2, and a formula string."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < degree + 1:
        raise ValueError(f"Need at least {degree + 1} finite points for degree {degree} fit")
    coefficients = np.polyfit(x, y, degree)
    y_pred = np.polyval(coefficients, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    terms = []
    for index, coeff in enumerate(coefficients):
        power = degree - index
        if power == 0:
            terms.append(f"{coeff:+.4f}")
        elif power == 1:
            terms.append(f"{coeff:+.6g}x")
        else:
            terms.append(f"{coeff:+.6g}x^{power}")
    formula = "y = " + " ".join(terms).lstrip("+")
    return {
        "degree": degree,
        "coefficients": coefficients.tolist(),
        "r2": r2,
        "formula": formula,
    }


def fit_linear(x: np.ndarray, y: np.ndarray) -> Dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        raise ValueError("Need at least 2 finite points for linear fit")
    coefficients = np.polyfit(x, y, 1)
    y_pred = np.polyval(coefficients, x)
    r2 = r2_score(y, y_pred)
    slope, intercept = coefficients
    return {
        "type": "linear",
        "coefficients": coefficients.tolist(),
        "r2": r2,
        "formula": f"y = {slope:.6g}x {intercept:+.4f}",
    }


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def fit_segmented_power(x: np.ndarray, y: np.ndarray, breakpoint: float = 3000.0) -> Dict:
    """Fit prefill power with a linear front and logarithmic saturated tail."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        raise ValueError("Need at least 5 finite points for segmented power fit")

    front_mask = x <= breakpoint
    tail_mask = x > breakpoint
    if np.count_nonzero(front_mask) < 2:
        raise ValueError("Need at least 2 front-section points for linear power fit")
    if np.count_nonzero(tail_mask) < 2:
        raise ValueError("Need at least 2 tail-section points for logarithmic power fit")

    front_x = x[front_mask]
    front_y = y[front_mask]
    tail_x = x[tail_mask]
    tail_y = y[tail_mask]

    front_coefficients = np.polyfit(front_x, front_y, 1)
    tail_log_x = np.log(tail_x / float(breakpoint))
    tail_coefficients = np.polyfit(tail_log_x, tail_y, 1)

    y_pred = np.empty_like(y, dtype=float)
    y_pred[front_mask] = np.polyval(front_coefficients, front_x)
    y_pred[tail_mask] = np.polyval(tail_coefficients, tail_log_x)
    combined_r2 = r2_score(y, y_pred)
    front_r2 = r2_score(front_y, np.polyval(front_coefficients, front_x))
    tail_r2 = r2_score(tail_y, np.polyval(tail_coefficients, tail_log_x))

    m_front, k_front = front_coefficients
    m, k = tail_coefficients
    formula = (
        f"x <= {breakpoint:.0f}: y = {m_front:.6g}x {k_front:+.4f}; "
        f"x > {breakpoint:.0f}: y = {m:.6g}ln(x / {breakpoint:.0f}) {k:+.4f}"
    )
    return {
        "type": "segmented_power",
        "breakpoint": float(breakpoint),
        "front": {
            "model": "linear",
            "coefficients": front_coefficients.tolist(),
            "r2": front_r2,
        },
        "tail": {
            "model": "log",
            "coefficients": tail_coefficients.tolist(),
            "r2": tail_r2,
        },
        "combined_r2": combined_r2,
        "formula": formula,
    }


def evaluate_segmented_power(fit: Dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    breakpoint = float(fit["breakpoint"])
    y = np.empty_like(x, dtype=float)
    front_mask = x <= breakpoint
    y[front_mask] = np.polyval(np.asarray(fit["front"]["coefficients"]), x[front_mask])
    tail_x = np.maximum(x[~front_mask], breakpoint)
    y[~front_mask] = np.polyval(
        np.asarray(fit["tail"]["coefficients"]),
        np.log(tail_x / breakpoint),
    )
    return y


def format_power_fit_for_report(fit: Dict) -> str:
    if fit.get("type") == "segmented_power":
        return (
            f"`{fit['formula']}`, combined R2={fit['combined_r2']:.4f}, "
            f"front R2={fit['front']['r2']:.4f}, tail R2={fit['tail']['r2']:.4f}"
        )
    return f"`{fit['formula']}`, R2={fit['r2']:.4f}"


def aggregate_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = raw_df[raw_df["status"].fillna("ok") == "ok"].copy()
    if ok_df.empty:
        ok_df = raw_df.copy()
    if "median_power_w" not in ok_df:
        ok_df["median_power_w"] = ok_df["avg_power_w"]
    if "p95_power_w" not in ok_df:
        ok_df["p95_power_w"] = ok_df["peak_power_w"]
    if "active_avg_power_w" not in ok_df:
        ok_df["active_avg_power_w"] = ok_df["p95_power_w"]
    if "active_median_power_w" not in ok_df:
        ok_df["active_median_power_w"] = ok_df["active_avg_power_w"]
    if "active_p95_power_w" not in ok_df:
        ok_df["active_p95_power_w"] = ok_df["p95_power_w"]
    if "active_sample_fraction" not in ok_df:
        ok_df["active_sample_fraction"] = 1.0
    if "energy_per_request_j" not in ok_df:
        denom = ok_df["block_request_count"] if "block_request_count" in ok_df else 1
        if not isinstance(denom, pd.Series):
            denom = pd.Series([denom] * len(ok_df), index=ok_df.index)
        ok_df["energy_per_request_j"] = ok_df["energy_j"] / denom.clip(lower=1)
    if "dynamic_energy_per_request_j" not in ok_df:
        denom = ok_df["block_request_count"] if "block_request_count" in ok_df else 1
        if not isinstance(denom, pd.Series):
            denom = pd.Series([denom] * len(ok_df), index=ok_df.index)
        ok_df["dynamic_energy_per_request_j"] = ok_df["dynamic_energy_j"] / denom.clip(lower=1)
    grouped = (
        ok_df.groupby("target_input_tokens", as_index=False)
        .agg(
            avg_actual_input_tokens=("actual_input_tokens", "mean"),
            num_samples=("actual_input_tokens", "count"),
            avg_duration_ms=("duration_ms", "mean"),
            std_duration_ms=("duration_ms", "std"),
            avg_ttft_ms=("ttft_ms", "mean"),
            std_ttft_ms=("ttft_ms", "std"),
            first_ttft_ms=("first_ttft_ms", "mean") if "first_ttft_ms" in ok_df.columns else ("ttft_ms", "mean"),
            avg_power_w=("avg_power_w", "mean"),
            std_power_w=("avg_power_w", "std"),
            median_power_w=("median_power_w", "mean"),
            p95_power_w=("p95_power_w", "mean"),
            peak_power_w=("peak_power_w", "max"),
            min_power_w=("min_power_w", "min"),
            avg_active_power_w=("active_avg_power_w", "mean"),
            std_active_power_w=("active_avg_power_w", "std"),
            active_median_power_w=("active_median_power_w", "mean"),
            active_p95_power_w=("active_p95_power_w", "mean"),
            avg_active_sample_fraction=("active_sample_fraction", "mean"),
            avg_energy_j=("energy_j", "mean"),
            avg_energy_per_request_j=("energy_per_request_j", "mean"),
            std_energy_j=("energy_j", "std"),
            avg_dynamic_power_w=("dynamic_power_w", "mean"),
            std_dynamic_power_w=("dynamic_power_w", "std"),
            avg_dynamic_energy_j=("dynamic_energy_j", "mean"),
            avg_dynamic_energy_per_request_j=("dynamic_energy_per_request_j", "mean"),
            std_dynamic_energy_j=("dynamic_energy_j", "std"),
            avg_idle_baseline_w=("idle_baseline_w", "mean"),
        )
        .fillna(0.0)
        .sort_values("target_input_tokens")
    )
    return grouped


def setup_axes_limits(ax, x_values, y_values, x_max: Optional[float] = None, y_zero: bool = True) -> None:
    max_x = float(x_max) if x_max is not None else float(np.nanmax(x_values))
    min_y = float(np.nanmin(y_values))
    max_y = float(np.nanmax(y_values))
    ax.set_xlim(0, max(1.0, max_x))
    if y_zero:
        ax.set_ylim(0, max(1.0, max_y * 1.1))
    else:
        span = max(1.0, max_y - min_y)
        ax.set_ylim(max(0.0, min_y - span * 0.12), max_y + span * 0.12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def save_figure(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_power_scatter(raw_df: pd.DataFrame, output_path: str, x_max: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    apply_origin_style(ax)
    ax.scatter(
        raw_df["target_input_tokens"],
        raw_df["avg_power_w"],
        s=10,
        alpha=0.55,
        color=PAPER_COLORS["raw"],
        edgecolors="none",
        label="Raw data",
    )
    ax.set_xlabel("Input tokens")
    ax.set_ylabel("Average power (W)")
    setup_axes_limits(ax, raw_df["target_input_tokens"], raw_df["avg_power_w"], x_max, y_zero=False)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, output_path)


def plot_metric_with_fit(raw_df: pd.DataFrame,
                         agg_df: pd.DataFrame,
                         raw_metric: str,
                         agg_metric: str,
                         std_metric: str,
                         ylabel: str,
                         title: str,
                         output_path: str,
                         degree: int = 2,
                         color: str = PAPER_COLORS["raw"],
                         x_max: Optional[float] = None,
                         fit_mode: str = "polynomial",
                         y_zero: bool = True) -> Dict:
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    apply_origin_style(ax)

    x_raw = raw_df["target_input_tokens"].to_numpy(dtype=float)
    y_raw = raw_df[raw_metric].to_numpy(dtype=float)
    x_agg = agg_df["target_input_tokens"].to_numpy(dtype=float)
    y_agg = agg_df[agg_metric].to_numpy(dtype=float)
    y_err = agg_df[std_metric].to_numpy(dtype=float) if std_metric in agg_df else None

    ax.scatter(x_raw, y_raw, s=9, alpha=0.35, color=color, edgecolors="none", label="Raw data")
    ax.errorbar(
        x_agg,
        y_agg,
        yerr=y_err,
        fmt="o",
        markersize=3.5,
        capsize=2.5,
        elinewidth=0.9,
        color=PAPER_COLORS["mean"],
        label="Mean +/- SD",
    )

    max_x = float(x_max) if x_max is not None else float(np.nanmax(x_agg))
    if fit_mode == "segmented_power":
        try:
            fit = fit_segmented_power(x_agg, y_agg)
        except ValueError:
            fit = fit_polynomial(x_agg, y_agg, degree=degree)
            fit["type"] = "polynomial_fallback"
            x_fit = np.linspace(0, max_x, 300)
            y_fit = np.polyval(np.asarray(fit["coefficients"]), x_fit)
            ax.plot(x_fit, y_fit, color=PAPER_COLORS["fit"], linewidth=1.5, label=f"Poly{degree} fit")
        else:
            breakpoint = float(fit["breakpoint"])
            min_x = max(0.0, float(np.nanmin(x_agg)))
            front_end = min(max_x, breakpoint)
            if front_end > min_x:
                x_front = np.linspace(min_x, front_end, 180)
                y_front = evaluate_segmented_power(fit, x_front)
                ax.plot(
                    x_front,
                    y_front,
                    color=PAPER_COLORS["fit"],
                    linewidth=1.5,
                    label="Front linear fit",
                )
            if max_x > breakpoint:
                x_tail = np.linspace(breakpoint, max_x, 180)
                y_tail = evaluate_segmented_power(fit, x_tail)
                ax.plot(
                    x_tail,
                    y_tail,
                    color=PAPER_COLORS["energy"],
                    linewidth=1.5,
                    linestyle="--",
                    label="Tail log fit",
                )
    elif fit_mode == "linear":
        fit = fit_linear(x_agg, y_agg)
        x_fit = np.linspace(float(np.nanmin(x_agg)), max_x, 300)
        y_fit = np.polyval(np.asarray(fit["coefficients"]), x_fit)
        ax.plot(x_fit, y_fit, color=PAPER_COLORS["fit"], linewidth=1.5, label="Linear fit")
    else:
        fit = fit_polynomial(x_agg, y_agg, degree=degree)
        x_fit = np.linspace(0, max_x, 300)
        y_fit = np.polyval(np.asarray(fit["coefficients"]), x_fit)
        ax.plot(x_fit, y_fit, color=PAPER_COLORS["fit"], linewidth=1.5, label=f"Poly{degree} fit")
        ax.text(
            0.03,
            0.95,
            f"{fit['formula']}\n$R^2$ = {fit['r2']:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "0.75", "linewidth": 0.5, "alpha": 0.85},
        )

    ax.set_xlabel("Input tokens")
    ax.set_ylabel(ylabel)
    setup_axes_limits(ax, x_raw, y_raw, x_max, y_zero=y_zero)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, output_path)
    return fit


def load_raw(input_dir: str) -> pd.DataFrame:
    raw_path = find_latest_file(input_dir, "_raw.csv")
    return pd.read_csv(raw_path)


def generate_analysis_outputs(input_dir: str,
                              output_dir: Optional[str] = None,
                              poly_degree: int = 2,
                              x_max: Optional[float] = None,
                              front_only_max: Optional[float] = None) -> Dict:
    if output_dir is None:
        output_dir = os.path.join(input_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    raw_path = find_latest_file(input_dir, "_raw.csv")
    raw_df = pd.read_csv(raw_path)
    raw_df = raw_df.copy()
    raw_df["status"] = raw_df.get("status", "ok")
    raw_ok = raw_df[raw_df["status"].fillna("ok") == "ok"].copy()
    if raw_ok.empty:
        raw_ok = raw_df.copy()
    if front_only_max is not None:
        raw_ok = raw_ok[raw_ok["target_input_tokens"] <= float(front_only_max)].copy()
        if raw_ok.empty:
            raise ValueError(f"No valid rows with target_input_tokens <= {front_only_max}")
    if "median_power_w" not in raw_ok:
        raw_ok["median_power_w"] = raw_ok["avg_power_w"]
    if "p95_power_w" not in raw_ok:
        raw_ok["p95_power_w"] = raw_ok["peak_power_w"]
    if "active_avg_power_w" not in raw_ok:
        raw_ok["active_avg_power_w"] = raw_ok["p95_power_w"]
    if "active_median_power_w" not in raw_ok:
        raw_ok["active_median_power_w"] = raw_ok["active_avg_power_w"]
    if "active_p95_power_w" not in raw_ok:
        raw_ok["active_p95_power_w"] = raw_ok["p95_power_w"]
    if "active_sample_fraction" not in raw_ok:
        raw_ok["active_sample_fraction"] = 1.0
    if "first_ttft_ms" not in raw_ok:
        raw_ok["first_ttft_ms"] = raw_ok["ttft_ms"]
    if "energy_per_request_j" not in raw_ok:
        denom = raw_ok["block_request_count"] if "block_request_count" in raw_ok else 1
        if not isinstance(denom, pd.Series):
            denom = pd.Series([denom] * len(raw_ok), index=raw_ok.index)
        raw_ok["energy_per_request_j"] = raw_ok["energy_j"] / denom.clip(lower=1)
    if "dynamic_energy_per_request_j" not in raw_ok:
        denom = raw_ok["block_request_count"] if "block_request_count" in raw_ok else 1
        if not isinstance(denom, pd.Series):
            denom = pd.Series([denom] * len(raw_ok), index=raw_ok.index)
        raw_ok["dynamic_energy_per_request_j"] = raw_ok["dynamic_energy_j"] / denom.clip(lower=1)
    agg_df = aggregate_raw(raw_ok)

    aggregated_csv = os.path.join(output_dir, "prefill_token_power_aggregated.csv")
    raw_ok.to_csv(os.path.join(output_dir, "prefill_token_power_filtered_raw.csv"), index=False)
    agg_df.to_csv(aggregated_csv, index=False)

    if x_max is None and front_only_max is not None:
        x_max = float(front_only_max)
    if x_max is None:
        x_max = max(20000.0, float(raw_ok["target_input_tokens"].max()))
    power_fit_mode = "linear" if front_only_max is not None else "segmented_power"
    non_power_fit_mode = "linear"

    power_scatter = os.path.join(output_dir, "prefill_power_raw_scatter.png")
    power_polyfit = os.path.join(output_dir, "prefill_power_polyfit.png")
    peak_power = os.path.join(output_dir, "prefill_peak_power_vs_tokens.png")
    p95_power = os.path.join(output_dir, "prefill_p95_power_vs_tokens.png")
    median_power = os.path.join(output_dir, "prefill_median_power_vs_tokens.png")
    active_power = os.path.join(output_dir, "prefill_active_power_polyfit.png")
    ttft = os.path.join(output_dir, "prefill_ttft_vs_tokens.png")
    first_ttft = os.path.join(output_dir, "prefill_first_ttft_vs_tokens.png")
    energy = os.path.join(output_dir, "prefill_energy_vs_tokens.png")

    plot_power_scatter(raw_ok, power_scatter, x_max=x_max)
    fit_results = {
        "avg_power_w": plot_metric_with_fit(
            raw_ok,
            agg_df,
            "avg_power_w",
            "avg_power_w",
            "std_power_w",
            "Average power (W)",
            "Prefill Power Polynomial Fit",
            power_polyfit,
            degree=poly_degree,
            color=PAPER_COLORS["raw"],
            x_max=x_max,
            fit_mode=power_fit_mode,
            y_zero=False,
        ),
        "peak_power_w": plot_metric_with_fit(
            raw_ok,
            agg_df,
            "peak_power_w",
            "peak_power_w",
            "std_power_w",
            "Peak power (W)",
            "Prefill Peak Power vs Input Tokens",
            peak_power,
            degree=poly_degree,
            color=PAPER_COLORS["peak"],
            x_max=x_max,
            fit_mode=power_fit_mode,
            y_zero=False,
        ),
        "p95_power_w": plot_metric_with_fit(
            raw_ok,
            agg_df,
            "p95_power_w",
            "p95_power_w",
            "std_power_w",
            "P95 power (W)",
            "Prefill P95 Power vs Input Tokens",
            p95_power,
            degree=poly_degree,
            color=PAPER_COLORS["peak"],
            x_max=x_max,
            fit_mode=power_fit_mode,
            y_zero=False,
        ),
        "median_power_w": plot_metric_with_fit(
            raw_ok,
            agg_df,
            "median_power_w",
            "median_power_w",
            "std_power_w",
            "Median power (W)",
            "Prefill Median Power vs Input Tokens",
            median_power,
            degree=poly_degree,
            color=PAPER_COLORS["raw"],
            x_max=x_max,
            fit_mode=power_fit_mode,
            y_zero=False,
        ),
        "avg_active_power_w": plot_metric_with_fit(
            raw_ok,
            agg_df,
            "active_avg_power_w",
            "avg_active_power_w",
            "std_active_power_w",
            "Active average power (W)",
            "Prefill Active Power Polynomial Fit",
            active_power,
            degree=poly_degree,
            color=PAPER_COLORS["raw"],
            x_max=x_max,
            fit_mode=power_fit_mode,
            y_zero=False,
        ),
        "avg_ttft_ms": plot_metric_with_fit(
            raw_ok,
            agg_df,
            "ttft_ms",
            "avg_ttft_ms",
            "std_ttft_ms",
            "TTFT (ms)",
            "Prefill Time vs Input Tokens",
            ttft,
            degree=poly_degree,
            color=PAPER_COLORS["ttft"],
            x_max=x_max,
            fit_mode=non_power_fit_mode,
        ),
        "first_ttft_ms": plot_metric_with_fit(
            raw_ok,
            agg_df,
            "first_ttft_ms",
            "first_ttft_ms",
            "std_ttft_ms",
            "First-request TTFT (ms)",
            "Prefill First-request TTFT vs Input Tokens",
            first_ttft,
            degree=poly_degree,
            color=PAPER_COLORS["ttft"],
            x_max=x_max,
            fit_mode=non_power_fit_mode,
        ),
        "avg_energy_j": plot_metric_with_fit(
            raw_ok,
            agg_df,
            "energy_j",
            "avg_energy_j",
            "std_energy_j",
            "Energy (J)",
            "Prefill Block Energy vs Input Tokens",
            energy,
            degree=poly_degree,
            color=PAPER_COLORS["energy"],
            x_max=x_max,
            fit_mode=non_power_fit_mode,
        ),
        "avg_energy_per_request_j": plot_metric_with_fit(
            raw_ok,
            agg_df,
            "energy_per_request_j",
            "avg_energy_per_request_j",
            "std_energy_j",
            "Energy per request (J)",
            "Prefill Energy per Request vs Input Tokens",
            os.path.join(output_dir, "prefill_energy_per_request_vs_tokens.png"),
            degree=poly_degree,
            color=PAPER_COLORS["energy"],
            x_max=x_max,
            fit_mode=non_power_fit_mode,
        ),
    }

    fit_results_json = os.path.join(output_dir, "fit_results.json")
    with open(fit_results_json, "w", encoding="utf-8") as file_obj:
        json.dump(fit_results, file_obj, indent=2, ensure_ascii=False)

    report = os.path.join(output_dir, "prefill_modeling_report.md")
    with open(report, "w", encoding="utf-8") as file_obj:
        power_fit = fit_results["avg_power_w"]
        active_power_fit = fit_results["avg_active_power_w"]
        file_obj.write(
            "# Prefill Token-Power Modeling Report\n\n"
            f"- Raw data: `{raw_path}`\n"
            f"- Valid samples: {len(raw_ok)}\n"
            f"- Target token range: {raw_ok['target_input_tokens'].min():.0f} - {raw_ok['target_input_tokens'].max():.0f}\n"
            f"- Actual token range: {raw_ok['actual_input_tokens'].min():.0f} - {raw_ok['actual_input_tokens'].max():.0f}\n"
            f"- Polynomial degree: {poly_degree}\n"
            f"- Non-power fit mode: linear\n"
            f"- Front-only max token: {front_only_max if front_only_max is not None else 'none'}\n"
            f"- Power fit: {format_power_fit_for_report(power_fit)}\n"
            f"- Active power fit: {format_power_fit_for_report(active_power_fit)}\n"
            f"- Energy-per-request fit: `{fit_results['avg_energy_per_request_j']['formula']}`, R2={fit_results['avg_energy_per_request_j']['r2']:.4f}\n\n"
            "- Primary power modeling metric: `avg_active_power_w`, which filters idle valleys "
            "inside repeated-request blocks. `avg_power_w` remains the full-window average for "
            "energy accounting. `median_power_w` and `p95_power_w` are provided for robustness; "
            "`peak_power_w` is diagnostic only because short prefill windows are sensitive to "
            "sampling phase. Power figures use a linear fit in front-only mode; otherwise they use "
            "a segmented fit with a linear front and logarithmic saturated tail. Equations are kept in this report "
            "and `fit_results.json` instead of being drawn inside the axes. `first_ttft_ms` captures the first "
            "request in each block; `avg_ttft_ms` is the block steady-state mean. TTFT and energy "
            "figures use linear fits in both full-range and front-only outputs. Energy plots should "
            "use per-request normalization in block mode.\n\n"
            "Generated figures follow `image.md`: white background, full frame, inward ticks, "
            "minor ticks, Times/SimSun font preference, and 600 dpi export.\n"
        )

    return {
        "raw_path": raw_path,
        "aggregated_csv": aggregated_csv,
        "power_scatter": power_scatter,
        "power_polyfit": power_polyfit,
        "peak_power": peak_power,
        "p95_power": p95_power,
        "median_power": median_power,
        "active_power": active_power,
        "ttft": ttft,
        "first_ttft": first_ttft,
        "energy": energy,
        "energy_per_request": os.path.join(output_dir, "prefill_energy_per_request_vs_tokens.png"),
        "fit_results_json": fit_results_json,
        "report": report,
        "fit_results": fit_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prefill token-power modeling results.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--poly-degree", type=int, default=2)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--front-only-max", type=float, default=None,
                        help="Only analyze rows with target_input_tokens <= this value and use linear power fits.")
    args = parser.parse_args()

    outputs = generate_analysis_outputs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        poly_degree=args.poly_degree,
        x_max=args.x_max,
        front_only_max=args.front_only_max,
    )
    print(f"分析完成: {outputs['report']}")


if __name__ == "__main__":
    main()
