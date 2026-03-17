#!/usr/bin/env python3
"""
预填充阶段功率分配策略评估结果分析与可视化

生成6种对比分析图表：
1. 能耗对比图
2. TTFT对比图
3. 能耗节省率图
4. TTFT损失率图
5. EDP对比图
6. 策略综合对比雷达图
"""
import argparse
import csv
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Tuple
from collections import defaultdict

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 策略显示名称
STRATEGY_NAMES = {
    'linear_165w': 'Linear 165W',
    'linear_185w': 'Linear 185W',
    'bucket1': 'Bucket 1',
    'bucket2': 'Bucket 2',
    'baseline_350w': 'Baseline 350W',
}

# 策略颜色
STRATEGY_COLORS = {
    'linear_165w': '#2ecc71',
    'linear_185w': '#3498db',
    'bucket1': '#e74c3c',
    'bucket2': '#f39c12',
    'baseline_350w': '#95a5a6',
}

# 子集显示顺序（最后加MEAN）
SUBSET_ORDER = [
    'subset_8', 'subset_16', 'subset_32', 'subset_64',
    'subset_103', 'subset_112', 'subset_119', 'MEAN'
]

# 子集标签
SUBSET_LABELS = {
    'subset_8': '8\n(225)',
    'subset_16': '16\n(504)',
    'subset_32': '32\n(1581)',
    'subset_64': '64\n(2175)',
    'subset_103': '103\n(6053)',
    'subset_112': '112\n(11106)',
    'subset_119': '119\n(20295)',
    'MEAN': 'MEAN',
}


def load_results(data_dir: str) -> Tuple[List[Dict], List[Dict], Dict]:
    """加载实验结果数据"""
    # 查找最新的结果文件
    raw_files = sorted(glob.glob(os.path.join(data_dir, "*_raw.csv")))
    agg_files = sorted(glob.glob(os.path.join(data_dir, "*_aggregated.csv")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "*_metadata.json")))

    if not raw_files or not agg_files:
        raise FileNotFoundError("未找到结果文件")

    raw_file = raw_files[-1]
    agg_file = agg_files[-1]
    meta_file = meta_files[-1] if meta_files else None

    print(f"加载原始数据: {raw_file}")
    print(f"加载聚合数据: {agg_file}")

    raw_results = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数值类型
            for key in ['full_repeat', 'subset_target_tokens', 'subset_num_prompts',
                       'power_limit', 'prompt_idx', 'prompt_repeat', 'prompt_tokens']:
                row[key] = int(float(row[key]))
            for key in ['ttft_ms', 'inference_duration_s', 'avg_power_w',
                       'total_energy_j', 'dynamic_power_w', 'dynamic_energy_j',
                       'idle_baseline_w']:
                row[key] = float(row[key])
            raw_results.append(row)

    agg_results = []
    with open(agg_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ['full_repeat', 'subset_target_tokens', 'subset_num_prompts',
                       'power_limit', 'num_samples']:
                row[key] = int(float(row[key]))
            for key in ['avg_ttft_ms', 'std_ttft_ms', 'avg_energy_j', 'std_energy_j',
                       'avg_power_w', 'std_power_w', 'avg_dynamic_energy_j',
                       'std_dynamic_energy_j', 'avg_dynamic_power_w',
                       'std_dynamic_power_w', 'total_energy_j_sum', 'idle_baseline_w']:
                row[key] = float(row[key])
            agg_results.append(row)

    metadata = {}
    if meta_file:
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    return raw_results, agg_results, metadata


def compute_summary_stats(agg_results: List[Dict]) -> Dict:
    """计算汇总统计数据（跨所有full_repeat取平均）"""
    key_func = lambda r: (r["strategy"], r["subset"])
    grouped = defaultdict(list)
    for r in agg_results:
        grouped[key_func(r)].append(r)

    summary = {}
    for (strategy, subset), group in grouped.items():
        summary[(strategy, subset)] = {
            "avg_ttft_ms": np.mean([r["avg_ttft_ms"] for r in group]),
            "std_ttft_ms": np.mean([r["std_ttft_ms"] for r in group]),
            "avg_energy_j": np.mean([r["avg_energy_j"] for r in group]),
            "std_energy_j": np.mean([r["std_energy_j"] for r in group]),
            "avg_power_w": np.mean([r["avg_power_w"] for r in group]),
            "total_energy_j_sum": np.mean([r["total_energy_j_sum"] for r in group]),
            "num_samples": group[0]["num_samples"],
            "power_limit": group[0]["power_limit"],
        }

    # 计算MEAN值（所有子集的平均）
    actual_subsets = [s for s in SUBSET_ORDER if s != 'MEAN']
    for strategy in STRATEGY_NAMES.keys():
        energies = []
        ttfts = []
        powers = []
        total_energies = []
        for subset in actual_subsets:
            key = (strategy, subset)
            if key in summary:
                energies.append(summary[key]["avg_energy_j"])
                ttfts.append(summary[key]["avg_ttft_ms"])
                powers.append(summary[key]["avg_power_w"])
                total_energies.append(summary[key]["total_energy_j_sum"])

        if energies:
            summary[(strategy, 'MEAN')] = {
                "avg_ttft_ms": np.mean(ttfts),
                "std_ttft_ms": np.std(ttfts) if len(ttfts) > 1 else 0,
                "avg_energy_j": np.mean(energies),
                "std_energy_j": np.std(energies) if len(energies) > 1 else 0,
                "avg_power_w": np.mean(powers),
                "total_energy_j_sum": np.mean(total_energies),
                "num_samples": -1,
                "power_limit": -1,
            }

    return summary


def compute_metrics_vs_baseline(summary: Dict) -> Dict:
    """计算相对于Baseline的指标"""
    metrics = {}

    for subset in SUBSET_ORDER:
        baseline_key = ("baseline_350w", subset)
        if baseline_key not in summary:
            continue

        baseline_energy = summary[baseline_key]["total_energy_j_sum"]
        baseline_ttft = summary[baseline_key]["avg_ttft_ms"]

        for strategy in STRATEGY_NAMES.keys():
            key = (strategy, subset)
            if key not in summary:
                continue

            energy = summary[key]["total_energy_j_sum"]
            ttft = summary[key]["avg_ttft_ms"]

            energy_saving = 100 * (baseline_energy - energy) / baseline_energy if baseline_energy > 0 else 0
            ttft_loss = 100 * (ttft - baseline_ttft) / baseline_ttft if baseline_ttft > 0 else 0
            edp = energy * ttft / 1000.0  # Energy-Delay Product (J*ms)
            baseline_edp = baseline_energy * baseline_ttft / 1000.0
            edp_improvement = 100 * (baseline_edp - edp) / baseline_edp if baseline_edp > 0 else 0

            metrics[(strategy, subset)] = {
                "energy_saving_pct": energy_saving,
                "ttft_loss_pct": ttft_loss,
                "edp": edp,
                "edp_improvement_pct": edp_improvement,
            }

    return metrics


def plot_energy_comparison(summary: Dict, output_file: str):
    """绘制能耗对比图"""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(SUBSET_ORDER))
    width = 0.16

    strategies = [s for s in STRATEGY_NAMES.keys() if s != 'baseline_350w'] + ['baseline_350w']

    for i, strategy in enumerate(strategies):
        energies = []
        for subset in SUBSET_ORDER:
            key = (strategy, subset)
            energies.append(summary.get(key, {}).get("total_energy_j_sum", 0))

        ax.bar(x + i * width, energies, width,
               label=STRATEGY_NAMES[strategy],
               color=STRATEGY_COLORS[strategy],
               alpha=0.85)

    ax.set_xlabel('Subset (Number of Prompts, Total Tokens)', fontsize=16)
    ax.set_ylabel('Total Energy Consumption (J)', fontsize=16)
    # ax.set_title('Energy Consumption Comparison by Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([SUBSET_LABELS[s] for s in SUBSET_ORDER], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()


def plot_ttft_comparison(summary: Dict, output_file: str):
    """绘制TTFT对比图"""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(SUBSET_ORDER))
    width = 0.16

    strategies = [s for s in STRATEGY_NAMES.keys() if s != 'baseline_350w'] + ['baseline_350w']

    for i, strategy in enumerate(strategies):
        ttfts = []
        for subset in SUBSET_ORDER:
            key = (strategy, subset)
            ttfts.append(summary.get(key, {}).get("avg_ttft_ms", 0))

        ax.bar(x + i * width, ttfts, width,
               label=STRATEGY_NAMES[strategy],
               color=STRATEGY_COLORS[strategy],
               alpha=0.85)

    ax.set_xlabel('Subset (Number of Prompts, Total Tokens)', fontsize=16)
    ax.set_ylabel('Average TTFT (ms)', fontsize=16)
    # ax.set_title('TTFT Comparison by Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([SUBSET_LABELS[s] for s in SUBSET_ORDER], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()


def plot_energy_saving_rate(metrics: Dict, output_file: str):
    """绘制能耗节省率图"""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(SUBSET_ORDER))
    width = 0.2

    strategies = [s for s in STRATEGY_NAMES.keys() if s != 'baseline_350w']

    for i, strategy in enumerate(strategies):
        savings = []
        for subset in SUBSET_ORDER:
            key = (strategy, subset)
            savings.append(metrics.get(key, {}).get("energy_saving_pct", 0))

        ax.bar(x + i * width, savings, width,
               label=STRATEGY_NAMES[strategy],
               color=STRATEGY_COLORS[strategy],
               alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Subset (Number of Prompts, Total Tokens)', fontsize=20)
    ax.set_ylabel('Energy Saving Rate (%)', fontsize=20)
    # ax.set_title('Energy Saving Rate vs Baseline 350W', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([SUBSET_LABELS[s] for s in SUBSET_ORDER], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()


def plot_ttft_loss_rate(metrics: Dict, output_file: str):
    """绘制TTFT损失率图"""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(SUBSET_ORDER))
    width = 0.2

    strategies = [s for s in STRATEGY_NAMES.keys() if s != 'baseline_350w']

    for i, strategy in enumerate(strategies):
        losses = []
        for subset in SUBSET_ORDER:
            key = (strategy, subset)
            losses.append(metrics.get(key, {}).get("ttft_loss_pct", 0))

        ax.bar(x + i * width, losses, width,
               label=STRATEGY_NAMES[strategy],
               color=STRATEGY_COLORS[strategy],
               alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Subset (Number of Prompts, Total Tokens)', fontsize=20)
    ax.set_ylabel('TTFT Increase Rate (%)', fontsize=20)
    # ax.set_title('TTFT Loss Rate vs Baseline 350W', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([SUBSET_LABELS[s] for s in SUBSET_ORDER], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()


def plot_edp_comparison(metrics: Dict, summary: Dict, output_file: str):
    """绘制EDP对比图"""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(SUBSET_ORDER))
    width = 0.16

    strategies = [s for s in STRATEGY_NAMES.keys() if s != 'baseline_350w'] + ['baseline_350w']

    for i, strategy in enumerate(strategies):
        edps = []
        for subset in SUBSET_ORDER:
            key = (strategy, subset)
            if strategy == 'baseline_350w':
                # 直接计算Baseline的EDP
                s = summary.get(key, {})
                edp = s.get("total_energy_j_sum", 0) * s.get("avg_ttft_ms", 0) / 1000.0
                edps.append(edp)
            else:
                edps.append(metrics.get(key, {}).get("edp", 0))

        ax.bar(x + i * width, edps, width,
               label=STRATEGY_NAMES[strategy],
               color=STRATEGY_COLORS[strategy],
               alpha=0.85)

    ax.set_xlabel('Subset (Number of Prompts, Total Tokens)', fontsize=16)
    ax.set_ylabel('EDP (J·ms)', fontsize=16)
    # ax.set_title('Energy-Delay Product (EDP) Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([SUBSET_LABELS[s] for s in SUBSET_ORDER], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()


def plot_radar_comparison(metrics: Dict, summary: Dict, output_file: str):
    """绘制雷达对比图"""
    # 计算所有子集的平均指标（排除MEAN）
    actual_subsets = [s for s in SUBSET_ORDER if s != 'MEAN']
    strategy_avg = defaultdict(lambda: {
        "energy_saving": [],
        "ttft_loss": [],
        "edp_improvement": [],
        "avg_power": [],
    })

    for subset in actual_subsets:
        baseline_key = ("baseline_350w", subset)
        if baseline_key not in summary:
            continue

        baseline_power = summary[baseline_key]["avg_power_w"]

        for strategy in STRATEGY_NAMES.keys():
            if strategy == 'baseline_350w':
                continue
            key = (strategy, subset)
            if key not in metrics or key not in summary:
                continue

            m = metrics[key]
            s = summary[key]

            strategy_avg[strategy]["energy_saving"].append(m["energy_saving_pct"])
            strategy_avg[strategy]["ttft_loss"].append(-m["ttft_loss_pct"])  # 负的损失就是收益
            strategy_avg[strategy]["edp_improvement"].append(m["edp_improvement_pct"])
            # 功率降低百分比
            power_reduction = 100 * (baseline_power - s["avg_power_w"]) / baseline_power if baseline_power > 0 else 0
            strategy_avg[strategy]["avg_power"].append(power_reduction)

    # 计算平均值
    radar_data = {}
    for strategy in strategy_avg.keys():
        data = strategy_avg[strategy]
        radar_data[strategy] = {
            "Energy Saving": np.mean(data["energy_saving"]),
            "TTFT Impact": np.mean(data["ttft_loss"]),
            "EDP Improvement": np.mean(data["edp_improvement"]),
            "Power Reduction": np.mean(data["avg_power"]),
        }

    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

    categories = list(radar_data[list(radar_data.keys())[0]].keys())
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    strategies = [s for s in STRATEGY_NAMES.keys() if s != 'baseline_350w']

    for strategy in strategies:
        values = list(radar_data[strategy].values())
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2,
               label=STRATEGY_NAMES[strategy],
               color=STRATEGY_COLORS[strategy],
               alpha=0.85)
        ax.fill(angles, values, alpha=0.15,
               color=STRATEGY_COLORS[strategy])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=15)
    ax.tick_params(axis='y', labelsize=12)
    # ax.set_title('Strategy Comparison Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()


def generate_markdown_report(summary: Dict, metrics: Dict, metadata: Dict,
                             output_file: str, img_dir: str):
    """生成Markdown格式分析报告"""
    # 计算平均指标（排除MEAN子集）
    actual_subsets = [s for s in SUBSET_ORDER if s != 'MEAN']
    overall = defaultdict(lambda: {"energy": [], "ttft": [], "edp_imp": []})

    for subset in actual_subsets:
        for strategy in STRATEGY_NAMES.keys():
            if strategy == 'baseline_350w':
                continue
            key = (strategy, subset)
            if key not in metrics or key not in summary:
                continue
            overall[strategy]["energy"].append(metrics[key]["energy_saving_pct"])
            overall[strategy]["ttft"].append(metrics[key]["ttft_loss_pct"])
            overall[strategy]["edp_imp"].append(metrics[key]["edp_improvement_pct"])

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 预填充阶段功率分配策略评估报告\n\n")

        f.write("## 实验概述\n\n")
        f.write(f"- 实验时间: {metadata.get('timestamp', 'N/A')}\n")
        f.write(f"- 完整重复次数: {metadata.get('full_repeats', 3)}\n")
        f.write(f"- 每prompt重复次数: {metadata.get('repeats_per_prompt', 100)}\n\n")

        f.write("## 测试策略\n\n")
        f.write("| 策略 | 描述 |\n")
        f.write("|------|------|\n")
        f.write("| Linear 165W | 固定功率 165W |\n")
        f.write("| Linear 185W | 固定功率 185W |\n")
        f.write("| Bucket 1 | 粗粒度分桶1: ≤1584→210W, ≤2176→230W, ≤6054→250W, ≤11107→270W, >11107→290W |\n")
        f.write("| Bucket 2 | 粗粒度分桶2: ≤1584→180W, ≤2176→210W, ≤6054→230W, ≤11107→240W, >11107→260W |\n")
        f.write("| Baseline 350W | 固定功率 350W（基准） |\n\n")

        f.write("## 测试子集\n\n")
        f.write("| 子集 | Prompt数量 | 总Token数 |\n")
        f.write("|------|-----------|----------|\n")
        for subset in metadata.get("subsets", []):
            f.write(f"| {subset['name']} | {subset['num_prompts']} | {subset['target_tokens']} |\n")
        f.write("\n")

        f.write("## 综合评估结果\n\n")
        f.write("| 策略 | 平均能耗节省 | 平均TTFT增加 | 平均EDP改善 |\n")
        f.write("|------|------------|------------|-----------|\n")
        for strategy in sorted(overall.keys()):
            avg_energy = np.mean(overall[strategy]["energy"])
            avg_ttft = np.mean(overall[strategy]["ttft"])
            avg_edp = np.mean(overall[strategy]["edp_imp"])
            f.write(f"| {STRATEGY_NAMES[strategy]} | {avg_energy:.2f}% | {avg_ttft:.2f}% | {avg_edp:.2f}% |\n")
        f.write("\n")

        f.write("## 详细图表\n\n")
        f.write("### 1. 能耗对比\n\n")
        f.write("![Energy Comparison](img/01_energy_comparison.png)\n\n")

        f.write("### 2. TTFT对比\n\n")
        f.write("![TTFT Comparison](img/02_ttft_comparison.png)\n\n")

        f.write("### 3. 能耗节省率\n\n")
        f.write("![Energy Saving Rate](img/03_energy_saving_rate.png)\n\n")

        f.write("### 4. TTFT损失率\n\n")
        f.write("![TTFT Loss Rate](img/04_ttft_loss_rate.png)\n\n")

        f.write("### 5. EDP对比\n\n")
        f.write("![EDP Comparison](img/05_edp_comparison.png)\n\n")

        f.write("### 6. 综合雷达图\n\n")
        f.write("![Radar Comparison](img/06_radar_comparison.png)\n\n")

        f.write("## 结论\n\n")
        f.write("（请根据实际实验结果填写结论）\n")

    print(f"已保存报告: {output_file}")


def analyze_strategy_evaluation(output_dir: str = "./results1"):
    """完整分析流程"""
    data_dir = os.path.join(output_dir, "data")
    img_dir = os.path.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    # 加载数据
    raw_results, agg_results, metadata = load_results(data_dir)

    # 计算统计数据
    summary = compute_summary_stats(agg_results)
    metrics = compute_metrics_vs_baseline(summary)

    # 生成图表
    print("\n生成图表...")
    plot_energy_comparison(summary, os.path.join(img_dir, "01_energy_comparison.png"))
    plot_ttft_comparison(summary, os.path.join(img_dir, "02_ttft_comparison.png"))
    plot_energy_saving_rate(metrics, os.path.join(img_dir, "03_energy_saving_rate.png"))
    plot_ttft_loss_rate(metrics, os.path.join(img_dir, "04_ttft_loss_rate.png"))
    plot_edp_comparison(metrics, summary, os.path.join(img_dir, "05_edp_comparison.png"))
    plot_radar_comparison(metrics, summary, os.path.join(img_dir, "06_radar_comparison.png"))

    # 生成报告
    print("\n生成分析报告...")
    report_file = os.path.join(output_dir, "strategy_evaluation_report.md")
    generate_markdown_report(summary, metrics, metadata, report_file, img_dir)

    print(f"\n分析完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="策略评估结果分析")
    parser.add_argument("--output-dir", type=str, default="./results1",
                       help="结果目录")
    args = parser.parse_args()

    analyze_strategy_evaluation(args.output_dir)
