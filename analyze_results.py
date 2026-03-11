import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_all_results(result_dir: str = "results") -> pd.DataFrame:
    """加载所有实验结果"""
    all_metadata = []
    if not os.path.exists(result_dir):
        print(f"结果目录 {result_dir} 不存在")
        return pd.DataFrame()

    for filename in os.listdir(result_dir):
        if filename.endswith("_metadata.csv"):
            df = pd.read_csv(f"{result_dir}/{filename}")
            all_metadata.append(df)

    if not all_metadata:
        print("没有找到实验结果")
        return pd.DataFrame()

    return pd.concat(all_metadata, ignore_index=True)

def generate_visualizations(df: pd.DataFrame, output_dir: str = "results/images"):
    """生成所有可视化图表"""
    os.makedirs(output_dir, exist_ok=True)

    if len(df) == 0:
        print("没有数据可生成图表")
        return

    # # 过滤异常值（EDP大于1e8的为实验异常结果）
    # abnormal_count = len(df[df['edp'] > 1e8])
    # df = df[df['edp'] <= 1e8].copy()
    # print(f"已过滤 {abnormal_count} 个异常EDP值")

    # 1. Power vs Average Latency
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="power_cap_w", y="avg_e2e_ms", marker='o', linewidth=2, markersize=8, errorbar='sd')
    plt.xlabel("Power Limit (W)", fontsize=12)
    plt.ylabel("Average End-to-End Latency (ms)", fontsize=12)
    plt.title("Power Limit vs Inference Latency", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/power_vs_latency.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Power vs Total Energy Consumption
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="power_cap_w", y="total_energy_j", marker='o', linewidth=2, markersize=8, color='orange', errorbar='sd')
    plt.xlabel("Power Limit (W)", fontsize=12)
    plt.ylabel("Total Energy Consumption (J)", fontsize=12)
    plt.title("Power Limit vs Total Energy", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/power_vs_energy.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Power vs Energy-Delay Product (EDP)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="power_cap_w", y="edp", marker='o', linewidth=2, markersize=8, color='green', errorbar='sd')
    plt.xlabel("Power Limit (W)", fontsize=12)
    plt.ylabel("Energy-Delay Product (EDP)", fontsize=12)
    plt.title("Power Limit vs Energy-Delay Product", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/power_vs_edp.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Power vs Throughput
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="power_cap_w", y="throughput_tps", marker='o', linewidth=2, markersize=8, color='red', errorbar='sd')
    plt.xlabel("Power Limit (W)", fontsize=12)
    plt.ylabel("Throughput (tokens/s)", fontsize=12)
    plt.title("Power Limit vs Inference Throughput", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/power_vs_throughput.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. EDP Comparison by Concurrency Level
    if df["concurrency"].nunique() > 1:
        plt.figure(figsize=(12, 6))
        df["concurrency"] = df["concurrency"].astype(str)
        if df["power_cap_w"].nunique() == 1:
            sns.barplot(data=df, x="concurrency", y="edp", palette="viridis", errorbar='sd')
            plt.xlabel("Concurrency Level", fontsize=12)
        else:
            sns.barplot(data=df, x="power_cap_w", y="edp", hue="concurrency", palette="viridis", errorbar='sd')
            plt.xlabel("Power Limit (W)", fontsize=12)
            plt.legend(title="Concurrency")
        plt.ylabel("Energy-Delay Product (EDP)", fontsize=12)
        plt.title("EDP Comparison by Concurrency Level", fontsize=14, pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(f"{output_dir}/edp_by_concurrency.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 6. Power vs Time to First Token (TTFT)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="power_cap_w", y="avg_ttft_ms", marker='o', linewidth=2, markersize=8, color='purple', errorbar='sd')
    plt.xlabel("Power Limit (W)", fontsize=12)
    plt.ylabel("Average Time to First Token (ms)", fontsize=12)
    plt.title("Power Limit vs Time to First Token", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/power_vs_ttft.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Power vs Time Between Tokens (TBT)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="power_cap_w", y="avg_tbt_ms", marker='o', linewidth=2, markersize=8, color='brown', errorbar='sd')
    plt.xlabel("Power Limit (W)", fontsize=12)
    plt.ylabel("Average Time Between Tokens (ms)", fontsize=12)
    plt.title("Power Limit vs Time Between Tokens", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/power_vs_tbt.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Energy Consumption by Concurrency Level
    target_concurrencies = [8, 32, 64, 128]
    df_filtered = df[df["concurrency"].isin(target_concurrencies)].copy()
    if len(df_filtered) > 0:
        df_filtered["concurrency"] = df_filtered["concurrency"].astype(str)
        df_filtered = df_filtered.sort_values("power_cap_w", ascending=False)
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="power_cap_w", y="total_energy_j",
                     hue="concurrency", marker='o', linewidth=2, markersize=8,
                     palette="viridis", errorbar='sd')
        plt.xlabel("Power Limit (W)", fontsize=12)
        plt.ylabel("Total Energy Consumption (J)", fontsize=12)
        plt.title("Energy Consumption by Power Limit and Concurrency", fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Concurrency")
        plt.savefig(f"{output_dir}/energy_by_concurrency.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 9. E2E Latency by Concurrency Level
    if len(df_filtered) > 0:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="power_cap_w", y="avg_e2e_ms",
                     hue="concurrency", marker='o', linewidth=2, markersize=8,
                     palette="viridis", errorbar='sd')
        plt.xlabel("Power Limit (W)", fontsize=12)
        plt.ylabel("Average End-to-End Latency (ms)", fontsize=12)
        plt.title("E2E Latency by Power Limit and Concurrency", fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Concurrency")
        plt.savefig(f"{output_dir}/e2e_latency_by_concurrency.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 10. TTFT by Concurrency Level
    if len(df_filtered) > 0:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="power_cap_w", y="avg_ttft_ms",
                     hue="concurrency", marker='o', linewidth=2, markersize=8,
                     palette="viridis", errorbar='sd')
        plt.xlabel("Power Limit (W)", fontsize=12)
        plt.ylabel("Average Time to First Token (ms)", fontsize=12)
        plt.title("TTFT by Power Limit and Concurrency", fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Concurrency")
        plt.savefig(f"{output_dir}/ttft_by_concurrency.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 11. TBT by Concurrency Level
    if len(df_filtered) > 0:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_filtered, x="power_cap_w", y="avg_tbt_ms",
                     hue="concurrency", marker='o', linewidth=2, markersize=8,
                     palette="viridis", errorbar='sd')
        plt.xlabel("Power Limit (W)", fontsize=12)
        plt.ylabel("Average Time Between Tokens (ms)", fontsize=12)
        plt.title("TBT by Power Limit and Concurrency", fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Concurrency")
        plt.savefig(f"{output_dir}/tbt_by_concurrency.png", dpi=300, bbox_inches='tight')
        plt.close()

def generate_report(df: pd.DataFrame, output_dir: str = "results/images"):
    """生成实验结果报告"""
    os.makedirs(output_dir, exist_ok=True)

    if len(df) == 0:
        report = "# 实验结果报告\n\n没有找到实验数据"
    else:
        report = f"""# 实验结果报告

## 实验概述
- 总实验次数: {len(df)}
- 测试功率范围: {df['power_cap_w'].min()}W ~ {df['power_cap_w'].max()}W
- 测试并发度: {', '.join(map(str, sorted(df['concurrency'].unique())))}
- 负载类型: {', '.join(map(str, sorted(df['load_type'].unique())))}

## 关键结果
1. **最优EDP功率点**: {df.loc[df['edp'].idxmin(), 'power_cap_w']}W，EDP值: {df['edp'].min():.2f}
2. **最高吞吐率**: {df['throughput_tps'].max():.2f} tokens/s，对应功率: {df.loc[df['throughput_tps'].idxmax(), 'power_cap_w']}W
3. **最低延迟**: {df['avg_e2e_ms'].min():.2f}ms，对应功率: {df.loc[df['avg_e2e_ms'].idxmin(), 'power_cap_w']}W
4. **最低能耗**: {df['total_energy_j'].min():.2f}J，对应功率: {df.loc[df['total_energy_j'].idxmin(), 'power_cap_w']}W
5. **最优TTFT**: {df['avg_ttft_ms'].min():.2f}ms，对应功率: {df.loc[df['avg_ttft_ms'].idxmin(), 'power_cap_w']}W

## 优化效果
相对最大功耗({df['power_cap_w'].max()}W)下的EDP:
- 最优功率点EDP降低: {(1 - df['edp'].min() / df.loc[df['power_cap_w'] == df['power_cap_w'].max(), 'edp'].mean()) * 100:.1f}%

## 实验建议
根据实验结果，推荐在实际部署中采用{df.loc[df['edp'].idxmin(), 'power_cap_w']}W的功率限制，可在损失少量性能的情况下显著降低能耗，获得最优的综合能效比。
"""

    with open(f"{output_dir}/report.md", 'w', encoding='utf-8') as f:
        f.write(report)

    if len(df) > 0:
        df.to_csv(f"{output_dir}/all_results.csv", index=False, encoding='utf-8-sig')
        print(f"所有结果已汇总保存到 {output_dir}/all_results.csv")

if __name__ == "__main__":
    df = load_all_results()
    generate_visualizations(df)
    generate_report(df)
    print("分析完成，结果已保存到results/images目录")
