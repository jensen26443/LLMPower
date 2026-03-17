import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 设置样式
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

# 配色方案
colors = {
    'energy': '#FF6B6B',      # 珊瑚红
    'e2e': '#4ECDC4',         # 青绿色
    'ttft': '#45B7D1',        # 天蓝色
    'tbt': '#96CEB4',         # 薄荷绿
    'throughput': '#FFEAA7',  # 暖黄色
    'edp': '#DDA0DD'          # 梅花紫
}

def parse_filename(filename):
    """从文件名解析功耗和查询数量: 150W_mixed_8q_xxx.csv -> (150, 8)
    同时兼容旧格式: 150W_mixed_8c_xxx.csv -> (150, 8)
    """
    # 优先匹配新格式 (q = query count)
    pattern_q = r'(\d+)W_mixed_(\d+)q_'
    match = re.search(pattern_q, filename)
    if match:
        return int(match.group(1)), int(match.group(2))

    # 兼容旧格式 (c = concurrency)
    pattern_c = r'(\d+)W_mixed_(\d+)c_'
    match = re.search(pattern_c, filename)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None

def load_all_data(directory='.'):
    """加载所有metadata.csv文件"""
    all_data = []
    path = Path(directory)
    csv_files = list(path.glob('*metadata.csv'))

    print(f"Found {len(csv_files)} data files:")
    loaded_configs = set()

    for f in csv_files:
        print(f"  - {f.name}")
        power_cap, query_count = parse_filename(f.name)
        if power_cap is None:
            print(f"    Warning: cannot parse filename parameters")
            continue

        df = pd.read_csv(f)

        config_key = (power_cap, query_count)
        if config_key in loaded_configs:
            print(f"    Skipping duplicate (already loaded {power_cap}W {query_count}q)")
            continue

        df['parsed_power_cap'] = power_cap
        df['parsed_query_count'] = query_count
        df['source_file'] = f.name
        all_data.append(df)
        loaded_configs.add(config_key)
        print(f"    Loaded: {power_cap}W, Query Count {query_count}")

    print(f"\nTotal loaded configurations: {len(loaded_configs)}")
    print(f"Loaded configs: {sorted(loaded_configs)}")

    if not all_data:
        raise ValueError("No valid data files found")

    return pd.concat(all_data, ignore_index=True)

def create_comprehensive_plots(df, output_dir='./results0/img'):
    """生成全面的性能分析图表"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取唯一值
    power_caps = sorted(df['parsed_power_cap'].unique())
    query_counts = sorted(df['parsed_query_count'].unique())

    print(f"\nData Overview: Power Caps {power_caps}W | Query counts: {query_counts} | Samples: {len(df)}")

    # 汇总数据
    summary = df.groupby(['parsed_power_cap', 'parsed_query_count']).agg({
        'total_energy_j': 'mean',
        'total_time_s': 'mean',
        'avg_ttft_ms': 'mean',
        'avg_tbt_ms': 'mean',
        'throughput_tps': 'mean'
    }).reset_index()

    # Convert energy to KJ
    summary['total_energy_kj'] = summary['total_energy_j'] / 1000

    # 重新计算EDP: 总能耗(J) * 总时间(s)
    summary['edp'] = summary['total_energy_j'] * summary['total_time_s']

    metrics_to_plot = {
        'total_energy_kj': ('Energy (KJ)', colors['energy']),
        'total_time_s': ('Total E2E Time (s)', colors['e2e']),
        'avg_ttft_ms': ('Time to First Token TTFT (ms)', colors['ttft']),
        'avg_tbt_ms': ('Time Between Tokens TBT (ms)', colors['tbt'])
    }

    # 1. Core Metrics Trend Plots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (metric, (title, color)) in enumerate(metrics_to_plot.items()):
        ax = axes[idx // 2, idx % 2]
        for qc in query_counts:
            qc_data = summary[summary['parsed_query_count'] == qc].sort_values('parsed_power_cap')
            ax.plot(qc_data['parsed_power_cap'], qc_data[metric],
                   marker='o', linewidth=2.5, markersize=8, label=f'{qc}', alpha=0.8)

        ax.set_xlabel('Power Cap (W)', fontsize=14, fontweight='bold')
        ax.set_ylabel(title, fontsize=14, fontweight='bold')
        ax.legend(title='Query Count', loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12, title_fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path / '1_performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: 1_performance_metrics_comparison.png")
    plt.close()

    # 2. Heatmap Matrix
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for idx, (metric, (title, _)) in enumerate(metrics_to_plot.items()):
        ax = axes[idx // 2, idx % 2]
        if metric == 'total_energy_kj':
            pivot = summary.pivot(index='parsed_power_cap', columns='parsed_query_count', values='total_energy_kj')
        else:
            pivot = summary.pivot(index='parsed_power_cap', columns='parsed_query_count', values=metric)
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax,
                   cbar_kws={'label': title}, linewidths=0.5, linecolor='white')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Query Count', fontsize=14, fontweight='bold')
        ax.set_ylabel('Power Cap (W)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / '2_performance_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: 2_performance_heatmaps.png")
    plt.close()

    # 3. Energy Efficiency and Throughput Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Energy per Token
    summary['energy_per_token'] = summary['total_energy_j'] / (summary['parsed_query_count'] * 100)
    for qc in query_counts:
        qc_data = summary[summary['parsed_query_count'] == qc].sort_values('parsed_power_cap')
        ax1.plot(qc_data['parsed_power_cap'], qc_data['energy_per_token'],
                marker='s', linewidth=2.5, markersize=8, label=f'{qc}')
    ax1.set_xlabel('Power Cap (W)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Energy per Token (J/token)', fontsize=14, fontweight='bold')
    ax1.legend(title='Query Count', fontsize=12, title_fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Throughput
    for qc in query_counts:
        qc_data = summary[summary['parsed_query_count'] == qc].sort_values('parsed_power_cap')
        ax2.plot(qc_data['parsed_power_cap'], qc_data['throughput_tps'],
                marker='^', linewidth=2.5, markersize=8, label=f'{qc}')
    ax2.set_xlabel('Power Cap (W)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Throughput (tokens/sec)', fontsize=14, fontweight='bold')
    ax2.set_title('Throughput Analysis', fontsize=13, fontweight='bold')
    ax2.legend(title='Query Count', fontsize=12, title_fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / '3_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: 3_efficiency_analysis.png")
    plt.close()

    # 4. EDP (Energy-Delay Product) Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    for qc in query_counts:
        qc_data = summary[summary['parsed_query_count'] == qc].sort_values('parsed_power_cap')
        ax.plot(qc_data['parsed_power_cap'], qc_data['edp']/1e3,
               marker='D', linewidth=2.5, markersize=9, label=f'{qc}')
    ax.set_xlabel('Power Cap (W)', fontsize=14, fontweight='bold')
    ax.set_ylabel('EDP (x10³ J·s)', fontsize=14, fontweight='bold')
    ax.legend(title='Query Count', fontsize=12, title_fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path / '4_edp_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: 4_edp_analysis.png")
    plt.close()

    # 5. Bar Chart Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (metric, (title, color)) in enumerate(metrics_to_plot.items()):
        ax = axes[idx // 2, idx % 2]
        x = np.arange(len(power_caps))
        width = 0.15

        for i, qc in enumerate(query_counts):
            values = []
            for power in power_caps:
                val = summary[(summary['parsed_power_cap'] == power) &
                             (summary['parsed_query_count'] == qc)][metric].values
                values.append(val[0] if len(val) > 0 else 0)
            ax.bar(x + i*width, values, width, label=f'{qc}', alpha=0.85, color=plt.cm.Set2(i))

        ax.set_xlabel('Power Cap (W)', fontsize=14, fontweight='bold')
        ax.set_ylabel(title, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * (len(query_counts)-1) / 2)
        ax.set_xticklabels(power_caps, fontsize=12)
        ax.legend(title='Query Count', fontsize=12, title_fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / '5_metrics_bar_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: 5_metrics_bar_comparison.png")
    plt.close()

    # 6. 3D Configuration Space
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = summary['parsed_power_cap'].values
    y = summary['parsed_query_count'].values
    z = summary['total_energy_kj'].values
    c = summary['throughput_tps'].values

    scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=100, alpha=0.8,
                        edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Power Cap (W)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Query Count', fontsize=14, fontweight='bold')
    ax.set_zlabel('Total Energy (KJ)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15)
    cbar.set_label('Throughput (TPS)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path / '6_3d_configuration_space.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: 6_3d_configuration_space.png")
    plt.close()

    # 7. Save Summary Data Table
    display_df = summary[['parsed_power_cap', 'parsed_query_count', 'total_energy_kj',
                         'total_time_s', 'avg_ttft_ms', 'avg_tbt_ms', 'throughput_tps', 'edp']]
    display_df.columns = ['Power(W)', 'Query Count', 'Energy(KJ)', 'Total E2E(s)', 'TTFT(ms)', 'TBT(ms)', 'Throughput(TPS)', 'EDP']
    display_df.to_csv(output_path / '7_performance_summary.csv', index=False)
    print(f"\n{'='*60}\nPerformance Data Summary\n{'='*60}")
    print(display_df.to_string(index=False))
    print(f"\n✅ Summary data saved to: {output_path / '7_performance_summary.csv'}")

if __name__ == "__main__":
    DATA_DIRECTORY = "./results0/data"

    try:
        df = load_all_data(DATA_DIRECTORY)
        create_comprehensive_plots(df)
        print(f"\n{'='*60}\n All plots generated successfully! Output directory: ./results0/img/\n{'='*60}")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
