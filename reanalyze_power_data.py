#!/usr/bin/env python3
"""
重新分析已有的功率数据，修复0值问题
"""
import os
import csv
import json
import pandas as pd
from statistics import mean
from typing import List, Dict


def reanalyze_power_timeline(results: List[Dict], power_data: List[Dict]) -> List[Dict]:
    """重新分析功率时间线，使用更宽松的匹配策略"""

    for result in results:
        start_time = result["inference_start"]
        end_time = result["inference_end"]

        # 提取这个时间范围内的功率数据
        relevant_powers = []
        for pd in power_data:
            t = pd["timestamp"]
            if start_time <= t <= end_time:
                relevant_powers.append(pd["power_w"])

        # 计算统计量 - 降低要求，只要有数据就用
        if len(relevant_powers) >= 1:
            avg_power = mean(relevant_powers)
            peak_power = max(relevant_powers)
            min_power = min(relevant_powers)

            # 计算能耗：梯形积分
            energy = 0.0
            for i in range(1, len(power_data)):
                t_prev = power_data[i-1]["timestamp"]
                t_curr = power_data[i]["timestamp"]
                if t_curr < start_time:
                    continue
                if t_prev > end_time:
                    break
                # 时间区间与推理窗口的交集
                t_start = max(t_prev, start_time)
                t_end = min(t_curr, end_time)
                dt = t_end - t_start
                if dt > 0:
                    p_prev = power_data[i-1]["power_w"]
                    p_curr = power_data[i]["power_w"]
                    avg_p = (p_prev + p_curr) / 2
                    energy += avg_p * dt

        else:
            # 如果没有精确匹配的点，找最近的点
            closest_before = None
            closest_after = None
            for pd in power_data:
                t = pd["timestamp"]
                if t < start_time:
                    closest_before = pd
                elif t > end_time and closest_after is None:
                    closest_after = pd
                    break

            # 使用邻近点估算
            if closest_before is not None and closest_after is not None:
                # 线性插值
                time_ratio = (start_time - closest_before["timestamp"]) / (closest_after["timestamp"] - closest_before["timestamp"])
                p_start = closest_before["power_w"] + time_ratio * (closest_after["power_w"] - closest_before["power_w"])

                time_ratio = (end_time - closest_before["timestamp"]) / (closest_after["timestamp"] - closest_before["timestamp"])
                p_end = closest_before["power_w"] + time_ratio * (closest_after["power_w"] - closest_before["power_w"])

                avg_power = (p_start + p_end) / 2
                peak_power = max(closest_before["power_w"], closest_after["power_w"])
                min_power = min(closest_before["power_w"], closest_after["power_w"])
                energy = avg_power * (end_time - start_time)
            elif closest_before is not None:
                # 只用前一个点
                avg_power = closest_before["power_w"]
                peak_power = closest_before["power_w"]
                min_power = closest_before["power_w"]
                energy = avg_power * (end_time - start_time)
            elif closest_after is not None:
                # 只用后一个点
                avg_power = closest_after["power_w"]
                peak_power = closest_after["power_w"]
                min_power = closest_after["power_w"]
                energy = avg_power * (end_time - start_time)
            else:
                # 完全没有数据
                avg_power = 0.0
                peak_power = 0.0
                min_power = 0.0
                energy = 0.0

        # 更新结果
        result["avg_power_w"] = avg_power
        result["peak_power_w"] = peak_power
        result["min_power_w"] = min_power
        result["total_energy_j"] = energy

    return results


def aggregate_results(results: List[Dict]) -> List[Dict]:
    """按目标token数聚合结果"""
    from collections import defaultdict
    import statistics

    grouped = defaultdict(list)
    for r in results:
        grouped[r["target_tokens"]].append(r)

    aggregated = []
    for target_count in sorted(grouped.keys()):
        group = grouped[target_count]
        powers = [r["avg_power_w"] for r in group if r["avg_power_w"] > 0]
        energies = [r["total_energy_j"] for r in group if r["total_energy_j"] > 0]
        ttfts = [r["ttft_ms"] for r in group]
        actual_tokens_list = [r["actual_tokens"] for r in group]
        peaks = [r["peak_power_w"] for r in group if r["peak_power_w"] > 0]

        agg_row = {
            "target_tokens": target_count,
            "avg_actual_tokens": statistics.mean(actual_tokens_list) if len(actual_tokens_list) > 0 else target_count,
            "count": len(group),
            "avg_power_w": statistics.mean(powers) if len(powers) > 0 else 0,
            "std_power_w": statistics.stdev(powers) if len(powers) > 1 else 0,
            "peak_power_w": statistics.mean(peaks) if len(peaks) > 0 else 0,
            "avg_energy_j": statistics.mean(energies) if len(energies) > 0 else 0,
            "std_energy_j": statistics.stdev(energies) if len(energies) > 1 else 0,
            "avg_ttft_ms": statistics.mean(ttfts) if len(ttfts) > 0 else 0,
            "std_ttft_ms": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
        }
        aggregated.append(agg_row)

    return aggregated


def main():
    # 找到最新的实验数据
    import glob
    result_dir = "results/prefill_modeling"

    raw_files = glob.glob(f"{result_dir}/*_raw.csv")
    timeline_files = glob.glob(f"{result_dir}/*_power_timeline.csv")

    if not raw_files or not timeline_files:
        print("没有找到实验数据文件")
        return

    # 找到匹配的文件对
    latest_raw = max(raw_files, key=os.path.getctime)
    base_name = os.path.basename(latest_raw).replace("_raw.csv", "")
    timeline_file = f"{result_dir}/{base_name}_power_timeline.csv"

    if not os.path.exists(timeline_file):
        print(f"找不到对应的时间线文件: {timeline_file}")
        return

    print(f"重新分析数据:")
    print(f"  原始数据: {latest_raw}")
    print(f"  功率时间线: {timeline_file}")

    # 读取数据
    raw_df = pd.read_csv(latest_raw)
    results = raw_df.to_dict('records')

    # 读取功率时间线
    power_data = []
    with open(timeline_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            power_data.append({
                "timestamp": float(row["timestamp"]),
                "power_w": float(row["power_w"]),
                "memory_gb": float(row["memory_gb"]),
                "temperature_c": float(row["temperature_c"])
            })

    # 重新分析
    print(f"\n开始重新分析 {len(results)} 条数据...")
    results = reanalyze_power_timeline(results, power_data)

    # 统计0值
    zero_power = sum(1 for r in results if r["avg_power_w"] == 0)
    print(f"  0值功率数据: {zero_power}/{len(results)} ({zero_power/len(results)*100:.1f}%)")

    # 保存新的原始数据
    output_raw = latest_raw.replace("_raw.csv", "_raw_reanalyzed.csv")
    pd.DataFrame(results).to_csv(output_raw, index=False, encoding='utf-8')
    print(f"\n保存重新分析的原始数据: {output_raw}")

    # 重新聚合
    aggregated = aggregate_results(results)
    output_agg = latest_raw.replace("_raw.csv", "_aggregated_reanalyzed.csv")
    pd.DataFrame(aggregated).to_csv(output_agg, index=False, encoding='utf-8')
    print(f"保存重新分析的聚合数据: {output_agg}")

    # 备份原文件并替换
    import shutil
    backup_raw = latest_raw.replace("_raw.csv", "_raw_old.csv")
    backup_agg = latest_raw.replace("_raw.csv", "_aggregated_old.csv")

    shutil.copy2(latest_raw, backup_raw)
    shutil.copy2(latest_raw.replace("_raw.csv", "_aggregated.csv"), backup_agg)

    shutil.copy2(output_raw, latest_raw)
    shutil.copy2(output_agg, latest_raw.replace("_raw.csv", "_aggregated.csv"))

    print(f"\n已更新原始数据和聚合数据")
    print(f"备份文件:")
    print(f"  {backup_raw}")
    print(f"  {backup_agg}")


if __name__ == "__main__":
    main()
