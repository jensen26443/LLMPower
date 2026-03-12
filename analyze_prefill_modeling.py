#!/usr/bin/env python3
"""
预填充阶段离线建模结果分析脚本

生成散点图 + 拟合曲线：
- 输入 token 数 vs Prefill 功率
- 输入 token 数 vs Prefill 能耗
- 输入 token 数 vs TTFT
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# 尝试导入scipy，如果没有则提供基本替代
try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy未安装，将跳过曲线拟合，只绘制散点图")

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 拟合函数定义
def linear_func(x, a, b):
    """线性函数: y = a*x + b"""
    return a * x + b


def log_func(x, a, b):
    """对数函数: y = a*log(x) + b"""
    return a * np.log(x + 1) + b


def sqrt_func(x, a, b):
    """平方根函数: y = a*sqrt(x) + b"""
    return a * np.sqrt(x) + b


def poly2_func(x, a, b, c):
    """二次多项式: y = a*x² + b*x + c"""
    return a * x**2 + b * x + c


def power_func(x, a, b):
    """幂函数: y = a*x^b"""
    return a * np.power(x, b)


FIT_FUNCTIONS = {
    'linear': (linear_func, 2, r'$y = a x + b$'),
    'log': (log_func, 2, r'$y = a \log(x) + b$'),
    'sqrt': (sqrt_func, 2, r'$y = a \sqrt{x} + b$'),
    'poly2': (poly2_func, 3, r'$y = a x^2 + b x + c$'),
    'power': (power_func, 2, r'$y = a x^b$'),
}


def load_latest_results(result_dir: str = "results/prefill_modeling") -> Optional[pd.DataFrame]:
    """加载最新的实验结果"""
    if not os.path.exists(result_dir):
        print(f"结果目录 {result_dir} 不存在")
        return None

    # 查找最新的聚合文件
    agg_files = glob.glob(f"{result_dir}/*_aggregated.csv")
    if not agg_files:
        print("没有找到聚合结果文件")
        return None

    latest_file = max(agg_files, key=os.path.getctime)
    print(f"加载文件: {latest_file}")

    df = pd.read_csv(latest_file)
    return df


def load_raw_results(result_dir: str = "results/prefill_modeling") -> Optional[pd.DataFrame]:
    """加载原始实验结果（用于散点图）"""
    if not os.path.exists(result_dir):
        print(f"结果目录 {result_dir} 不存在")
        return None

    raw_files = glob.glob(f"{result_dir}/*_raw.csv")
    if not raw_files:
        print("没有找到原始结果文件")
        return None

    latest_file = max(raw_files, key=os.path.getctime)
    print(f"加载原始数据: {latest_file}")

    df = pd.read_csv(latest_file)
    return df


def fit_curve(x: np.ndarray, y: np.ndarray, func_name: str) -> Tuple[Optional[np.ndarray], float, str]:
    """拟合曲线

    Returns:
        (params, r_squared, formula)
    """
    if not HAS_SCIPY:
        return None, -float('inf'), ""

    func, param_count, formula_template = FIT_FUNCTIONS[func_name]

    try:
        # 初始猜测
        if func_name == 'linear':
            p0 = [1.0, 0.0]
        elif func_name == 'log':
            p0 = [1.0, 0.0]
        elif func_name == 'sqrt':
            p0 = [1.0, 0.0]
        elif func_name == 'poly2':
            p0 = [0.01, 1.0, 0.0]
        elif func_name == 'power':
            p0 = [1.0, 0.5]

        params, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)

        # 计算R²
        y_pred = func(x, *params)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # 生成公式字符串
        if func_name == 'linear':
            formula = rf'$y = {params[0]:.4f}x {params[1]:+.4f}$'
        elif func_name == 'log':
            formula = rf'$y = {params[0]:.4f} \log(x) {params[1]:+.4f}$'
        elif func_name == 'sqrt':
            formula = rf'$y = {params[0]:.4f} \sqrt{{x}} {params[1]:+.4f}$'
        elif func_name == 'poly2':
            formula = rf'$y = {params[0]:.6f}x^2 {params[1]:+.4f}x {params[2]:+.4f}$'
        elif func_name == 'power':
            formula = rf'$y = {params[0]:.4f} x^{{{params[1]:.4f}}}$'

        return params, r_squared, formula

    except Exception as e:
        print(f"拟合 {func_name} 失败: {e}")
        return None, -float('inf'), ""


def find_best_fit(x: np.ndarray, y: np.ndarray) -> Tuple[str, np.ndarray, float, str]:
    """尝试多种拟合函数，找到最佳的

    Returns:
        (best_func_name, best_params, best_r2, best_formula)
    """
    if not HAS_SCIPY:
        return None, None, -float('inf'), ""

    best_r2 = -float('inf')
    best_func = None
    best_params = None
    best_formula = ""

    for func_name in FIT_FUNCTIONS.keys():
        params, r2, formula = fit_curve(x, y, func_name)
        if params is not None and r2 > best_r2:
            best_r2 = r2
            best_func = func_name
            best_params = params
            best_formula = formula

    return best_func, best_params, best_r2, best_formula


def plot_with_fit(ax, x_raw: np.ndarray, y_raw: np.ndarray,
                  x_agg: np.ndarray, y_agg: np.ndarray, y_err: np.ndarray,
                  ylabel: str, title: str, color: str = 'blue'):
    """绘制带拟合曲线的图"""
    # 原始数据散点
    ax.scatter(x_raw, y_raw, alpha=0.3, s=20, color=color, label='Raw Data')

    # 聚合数据带误差棒
    ax.errorbar(x_agg, y_agg, yerr=y_err, fmt='o', color=color,
               markersize=8, capsize=5, label='Mean ± Std Dev')

    # 拟合曲线（使用聚合数据）
    best_func, best_params, best_r2, best_formula = None, None, -float('inf'), ""
    if HAS_SCIPY:
        best_func, best_params, best_r2, best_formula = find_best_fit(x_agg, y_agg)

        if best_func is not None:
            func = FIT_FUNCTIONS[best_func][0]
            x_smooth = np.linspace(max(0, x_agg.min() - 50), x_agg.max() + 50, 100)
            y_smooth = func(x_smooth, *best_params)
            ax.plot(x_smooth, y_smooth, '--', color='red', linewidth=2,
                   label=f'Fit Curve ({best_func})\n$R^2={best_r2:.4f}$\n{best_formula}')
    else:
        ax.text(0.02, 0.98, 'scipy not installed, skip curve fitting',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 线性横坐标，设置刻度为0,250,500...
    max_x = x_agg.max()
    if max_x <= 500:
        xticks = np.arange(0, max_x + 100, 100)
    elif max_x <= 1000:
        xticks = np.arange(0, max_x + 250, 250)
    else:
        xticks = np.arange(0, max_x + 500, 500)
    ax.set_xticks(xticks)
    ax.set_xlabel('Input Tokens', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    return best_func, best_params, best_r2, best_formula


def generate_visualizations(raw_df: pd.DataFrame, agg_df: pd.DataFrame,
                            output_dir: str = "results/prefill_modeling/images"):
    """生成所有可视化图表"""
    os.makedirs(output_dir, exist_ok=True)

    if len(agg_df) == 0:
        print("没有数据可生成图表")
        return {}

    # 优先使用实际token数，如果有的话（兼容新旧格式）
    if 'actual_tokens' in raw_df.columns:
        x_raw = raw_df['actual_tokens'].values
    elif 'actual_input_tokens' in raw_df.columns:
        x_raw = raw_df['actual_input_tokens'].values
    elif 'target_tokens' in raw_df.columns:
        x_raw = raw_df['target_tokens'].values
    else:
        x_raw = raw_df['input_tokens'].values

    if 'avg_actual_tokens' in agg_df.columns:
        x_agg = agg_df['avg_actual_tokens'].values
    elif 'target_tokens' in agg_df.columns:
        x_agg = agg_df['target_tokens'].values
    else:
        x_agg = agg_df['input_tokens'].values

    fit_results = {}

    # 1. Input tokens vs Prefill power
    fig, ax = plt.subplots(figsize=(12, 7))
    func, params, r2, formula = plot_with_fit(
        ax, x_raw, raw_df['avg_power_w'].values,
        x_agg, agg_df['avg_power_w'].values, agg_df['std_power_w'].values,
        'Average Power (W)', 'Input Tokens vs Prefill Stage Power',
        color='tab:blue'
    )
    fit_results['power'] = {'function': func, 'params': params.tolist() if params is not None else None,
                            'r2': r2, 'formula': formula}
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prefill_power_vs_tokens.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Input tokens vs Prefill energy
    fig, ax = plt.subplots(figsize=(12, 7))
    func, params, r2, formula = plot_with_fit(
        ax, x_raw, raw_df['total_energy_j'].values,
        x_agg, agg_df['avg_energy_j'].values, agg_df['std_energy_j'].values,
        'Total Energy (J)', 'Input Tokens vs Prefill Stage Energy',
        color='tab:orange'
    )
    fit_results['energy'] = {'function': func, 'params': params.tolist() if params is not None else None,
                             'r2': r2, 'formula': formula}
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prefill_energy_vs_tokens.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Input tokens vs TTFT
    fig, ax = plt.subplots(figsize=(12, 7))
    func, params, r2, formula = plot_with_fit(
        ax, x_raw, raw_df['ttft_ms'].values,
        x_agg, agg_df['avg_ttft_ms'].values, agg_df['std_ttft_ms'].values,
        'TTFT (ms)', 'Input Tokens vs Time to First Token (TTFT)',
        color='tab:green'
    )
    fit_results['ttft'] = {'function': func, 'params': params.tolist() if params is not None else None,
                           'r2': r2, 'formula': formula}
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prefill_ttft_vs_tokens.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Input tokens vs Peak power (额外图表)
    if 'peak_power_w' in agg_df.columns:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(x_raw, raw_df['peak_power_w'].values, alpha=0.3, s=20, color='tab:red', label='Raw Data')
        ax.errorbar(x_agg, agg_df['peak_power_w'].values, fmt='o', color='tab:red',
                   markersize=8, capsize=5, label='Mean ± Std Dev')
        ax.set_xlabel('Input Tokens', fontsize=12)
        ax.set_ylabel('Peak Power (W)', fontsize=12)
        ax.set_title('Input Tokens vs Prefill Stage Peak Power', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prefill_peak_power_vs_tokens.png", dpi=300, bbox_inches='tight')
        plt.close()

    return fit_results


def generate_report(agg_df: pd.DataFrame, fit_results: Dict,
                    output_dir: str = "results/prefill_modeling/images"):
    """生成分析报告"""
    os.makedirs(output_dir, exist_ok=True)

    if len(agg_df) == 0:
        report = "# Prefill Phase Modeling Report\n\nNo experimental data found"
    else:
        # 兼容新旧数据格式
        if 'input_tokens' in agg_df.columns:
            tokens_col = 'input_tokens'
        else:
            tokens_col = 'target_tokens'

        power_result = fit_results.get('power', {})
        energy_result = fit_results.get('energy', {})
        ttft_result = fit_results.get('ttft', {})

        power_trend = 'increases' if agg_df['avg_power_w'].iloc[-1] > agg_df['avg_power_w'].iloc[0] else 'decreases'
        ttft_trend = 'increases' if agg_df['avg_ttft_ms'].iloc[-1] > agg_df['avg_ttft_ms'].iloc[0] else 'decreases'
        energy_trend = 'increases' if agg_df['avg_energy_j'].iloc[-1] > agg_df['avg_energy_j'].iloc[0] else 'decreases'

        report = f"""# Prefill Phase Offline Modeling Report

## Experiment Overview
- Test input lengths: {', '.join(map(str, sorted(agg_df[tokens_col].unique())))} tokens
- Repeats per length: {agg_df['count'].iloc[0]} times
- Total experiments: {len(agg_df) * agg_df['count'].iloc[0]}

## Fitting Results Summary

### 1. Power Model P_prefill = f(C)
- Best fit function: {power_result.get('function', 'N/A')}
- Fit formula: {power_result.get('formula', 'N/A')}
- R² value: {power_result.get('r2', 'N/A'):.4f}

### 2. Energy Model E_prefill = f(C)
- Best fit function: {energy_result.get('function', 'N/A')}
- Fit formula: {energy_result.get('formula', 'N/A')}
- R² value: {energy_result.get('r2', 'N/A'):.4f}

### 3. TTFT Model TTFT = g(C)
- Best fit function: {ttft_result.get('function', 'N/A')}
- Fit formula: {ttft_result.get('formula', 'N/A')}
- R² value: {ttft_result.get('r2', 'N/A'):.4f}

## Key Observations
1. **Power vs input length**: Prefill power {power_trend} with input token count
2. **TTFT vs input length**: Time-to-first-token {ttft_trend} with input token count
3. **Energy trend**: Total energy {energy_trend} with input token count

## Chart Description
Generated charts:
- `prefill_power_vs_tokens.png`: Input tokens vs average power (with fit curve)
- `prefill_energy_vs_tokens.png`: Input tokens vs total energy (with fit curve)
- `prefill_ttft_vs_tokens.png`: Input tokens vs TTFT (with fit curve)

All charts use linear x-axis, showing raw data scatter, mean ± std dev error bars, and best fit curve.
"""

    with open(f"{output_dir}/prefill_modeling_report.md", 'w', encoding='utf-8') as f:
        f.write(report)

    # 保存拟合参数
    with open(f"{output_dir}/fit_results.json", 'w', encoding='utf-8') as f:
        json.dump(fit_results, f, indent=2, ensure_ascii=False)

    if len(agg_df) > 0:
        agg_df.to_csv(f"{output_dir}/aggregated_results.csv", index=False, encoding='utf-8-sig')
        print(f"Aggregated results saved to {output_dir}/aggregated_results.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prefill Phase Modeling Result Analysis")
    parser.add_argument("--input-dir", type=str, default="results/prefill_modeling",
                       help="Experiment result directory")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Chart output directory (default: input-dir/images)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"{args.input_dir}/images"

    raw_df = load_raw_results(args.input_dir)
    agg_df = load_latest_results(args.input_dir)

    if raw_df is not None and agg_df is not None:
        fit_results = generate_visualizations(raw_df, agg_df, args.output_dir)
        generate_report(agg_df, fit_results, args.output_dir)
        print(f"分析完成，结果已保存到 {args.output_dir}")
    else:
        print("无法加载实验结果，请先运行实验")
