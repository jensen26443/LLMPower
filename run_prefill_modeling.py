#!/usr/bin/env python3
"""
预填充阶段离线建模实验脚本

用于拟合：
P_prefill = f(C)
TTFT = g(C)
其中 C 表示输入 token 数

新版本：使用连续推理 + 时间线分析，获得更准确的功率数据
"""
import argparse
import csv
import os
import time
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

from power_control import set_power_cap, get_power_cap
from llm_inference import LLMInferencer
from load_generator import LoadGenerator
from monitor import PowerMonitor


def run_prefill_experiment(input_token_counts: List[int],
                           repeats: int = 20,
                           power_cap: int = 350,
                           output_dir: str = "results/prefill_modeling",
                           model_path: str = None,
                           tokenizer_path: str = "./Qwen2.5-7B-Instruct-AWQ",
                           sharegpt_dir: str = "./input/ShareGPT",
                           time_padding_ms: float = 40.0,
                           sudo_password: str = None,
                           skip_set_power: bool = False):
    """运行预填充阶段建模实验（连续推理版本）

    Args:
        input_token_counts: 输入token数列表
        repeats: 每个点重复次数
        power_cap: 功率限制(W)
        output_dir: 结果输出目录
        model_path: 模型路径
        tokenizer_path: Qwen2.5分词器路径
        sharegpt_dir: ShareGPT数据集目录
        time_padding_ms: 推理窗口前后补偿时间（毫秒），用于覆盖功率传感器上报延迟
        sudo_password: sudo密码
        skip_set_power: 跳过设置功率步骤
    """
    os.makedirs(output_dir, exist_ok=True)

    # 设置功率限制
    if not skip_set_power:
        print(f"设置功率限制为 {power_cap}W")
        if not set_power_cap(power_cap, sudo_password=sudo_password):
            print("设置功率失败，实验终止")
            return None
    else:
        print(f"跳过功率设置，使用当前系统功率限制")

    actual_power_cap = get_power_cap()
    print(f"实际功率限制: {actual_power_cap}W")
    print("等待功率稳定20秒...")
    time.sleep(20)

    # 初始化组件
    if model_path:
        inferencer = LLMInferencer(model_name=model_path)
    else:
        inferencer = LLMInferencer()
    load_generator = LoadGenerator(sharegpt_dir=sharegpt_dir, tokenizer_name=tokenizer_path)

    # 预热GPU
    print("预热GPU...")
    warmup_prompt = load_generator.generate_prompt_by_token_count(64)
    for _ in range(5):
        inferencer.infer_prefill_only([warmup_prompt], max_tokens=1)
    time.sleep(2)

    # 构建实验队列：(token_count, repeat_id, prompt) 的列表，随机打乱
    experiment_queue = []
    for token_count in input_token_counts:
        for repeat_id in range(1, repeats + 1):
            prompt = load_generator.generate_prompt_by_token_count(token_count)
            actual_tokens = load_generator.count_tokens(prompt)
            experiment_queue.append((token_count, repeat_id, actual_tokens, prompt))

    # 打乱顺序，避免系统性偏差
    random.shuffle(experiment_queue)
    total_experiments = len(experiment_queue)

    print(f"\n实验队列已构建：{len(input_token_counts)} 个输入长度 × {repeats} 次重复 = {total_experiments} 次实验")
    print("开始连续推理...")

    # 启动功率监测（连续监测整个实验过程）
    monitor = PowerMonitor(sample_interval=0.02)  # 更高的采样频率：50Hz
    monitor.start()
    experiment_start_time = time.time()

    # 运行所有实验，记录每个实验的时间戳
    results = []
    for idx, (target_tokens, repeat_id, actual_tokens, prompt) in enumerate(tqdm(experiment_queue, desc="连续推理中")):
        # 记录开始时间
        inference_start = time.time()

        # 执行推理
        result = inferencer.infer_prefill_only([prompt], max_tokens=1)[0]

        # 记录结束时间
        inference_end = time.time()

        # 保存结果
        results.append({
            "index": idx,
            "target_tokens": target_tokens,
            "repeat_id": repeat_id,
            "batch_size": 1,
            "actual_tokens": actual_tokens,
            "ttft_ms": result["ttft"],
            "inference_start": inference_start,
            "inference_end": inference_end,
            "inference_duration": inference_end - inference_start,
            "prompt_preview": prompt[:50] if len(prompt) > 50 else prompt
        })

        # 推理间隔：给GPU功率回落和稳定的时间
        time.sleep(0.2)

    # 停止监测
    experiment_end_time = time.time()
    power_data = monitor.stop()
    total_experiment_duration = experiment_end_time - experiment_start_time

    print(f"\n推理完成，开始分析功率时间线...")

    # 分析每个推理期间的功率数据
    final_results = analyze_power_timeline(
        results,
        power_data,
        experiment_start_time,
        time_padding_ms=time_padding_ms,
    )

    # 保存完整功率时间线
    experiment_id = f"prefill_modeling_continuous_{int(time.time())}"
    timeline_file = f"{output_dir}/{experiment_id}_power_timeline.csv"

    with open(timeline_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["timestamp", "relative_time", "power_w", "memory_gb", "temperature_c"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pd in power_data:
            writer.writerow({
                "timestamp": pd["timestamp"],
                "relative_time": pd["timestamp"] - experiment_start_time,
                "power_w": pd["power_w"],
                "memory_gb": pd["memory_gb"],
                "temperature_c": pd["temperature_c"]
            })

    # 保存推理结果
    result_file = f"{output_dir}/{experiment_id}_raw.csv"

    with open(result_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["index", "target_tokens", "repeat_id", "batch_size", "actual_tokens", "ttft_ms",
                     "inference_start", "inference_end", "inference_duration",
                     "avg_power_w", "peak_power_w", "min_power_w", "total_energy_j",
                     "dynamic_power_w", "dynamic_energy_j", "idle_baseline_w",
                     "prompt_preview"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_results)

    # 生成聚合结果
    aggregated = aggregate_results(final_results)
    agg_file = f"{output_dir}/{experiment_id}_aggregated.csv"

    with open(agg_file, 'w', newline='', encoding='utf-8') as f:
        agg_fieldnames = ["target_tokens", "avg_actual_tokens", "count", "batch_size",
                         "avg_power_w", "std_power_w", "peak_power_w",
                         "avg_energy_j", "std_energy_j",
                         "avg_dynamic_power_w", "std_dynamic_power_w",
                         "avg_dynamic_energy_j", "std_dynamic_energy_j",
                         "avg_idle_baseline_w",
                         "avg_ttft_ms", "std_ttft_ms"]
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        for row in aggregated:
            writer.writerow(row)

    print(f"\n实验完成！")
    print(f"总耗时: {total_experiment_duration:.1f}s")
    print(f"功率采样点: {len(power_data)}")
    print(f"原始数据保存至: {result_file}")
    print(f"功率时间线保存至: {timeline_file}")
    print(f"聚合数据保存至: {agg_file}")

    return {
        "raw_file": result_file,
        "timeline_file": timeline_file,
        "aggregated_file": agg_file,
        "record_count": len(final_results)
    }


def analyze_power_timeline(results: List[Dict], power_data: List[Dict],
                          experiment_start_time: float,
                          time_padding_ms: float = 40.0) -> List[Dict]:
    """分析功率时间线，提取每个推理期间的功率数据

    Args:
        results: 推理结果列表
        power_data: 完整功率时间线
        experiment_start_time: 实验开始时间

    Returns:
        更新后的结果列表，包含功率统计
    """
    from statistics import mean, stdev

    # 估计空闲基线功率：使用前10%样本（最少5个点）
    if len(power_data) >= 5:
        baseline_count = max(5, int(len(power_data) * 0.1))
        baseline_count = min(len(power_data), baseline_count)
        idle_baseline_w = mean([x["power_w"] for x in power_data[:baseline_count]])
    else:
        idle_baseline_w = 0.0

    padding_s = max(0.0, time_padding_ms / 1000.0)

    # 为每个推理结果匹配对应的功率数据
    for result in results:
        start_time = result["inference_start"] - padding_s
        end_time = result["inference_end"] + padding_s

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
            # 找到推理前后最近的采样点
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

        # 更新结果（同时记录去基线后的动态功率/能耗）
        dynamic_avg_power = max(0.0, avg_power - idle_baseline_w)
        dynamic_energy = max(0.0, energy - idle_baseline_w * max(0.0, end_time - start_time))

        result["avg_power_w"] = avg_power
        result["peak_power_w"] = peak_power
        result["min_power_w"] = min_power
        result["total_energy_j"] = energy
        result["dynamic_power_w"] = dynamic_avg_power
        result["dynamic_energy_j"] = dynamic_energy
        result["idle_baseline_w"] = idle_baseline_w

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
        dynamic_powers = [r["dynamic_power_w"] for r in group if r.get("dynamic_power_w", 0) >= 0]
        dynamic_energies = [r["dynamic_energy_j"] for r in group if r.get("dynamic_energy_j", 0) >= 0]
        idle_baselines = [r["idle_baseline_w"] for r in group if r.get("idle_baseline_w", 0) > 0]

        agg_row = {
            "target_tokens": target_count,
            "avg_actual_tokens": statistics.mean(actual_tokens_list) if len(actual_tokens_list) > 0 else target_count,
            "count": len(group),
            "batch_size": 1,
            "avg_power_w": statistics.mean(powers) if len(powers) > 0 else 0,
            "std_power_w": statistics.stdev(powers) if len(powers) > 1 else 0,
            "peak_power_w": statistics.mean(peaks) if len(peaks) > 0 else 0,
            "avg_energy_j": statistics.mean(energies) if len(energies) > 0 else 0,
            "std_energy_j": statistics.stdev(energies) if len(energies) > 1 else 0,
            "avg_dynamic_power_w": statistics.mean(dynamic_powers) if len(dynamic_powers) > 0 else 0,
            "std_dynamic_power_w": statistics.stdev(dynamic_powers) if len(dynamic_powers) > 1 else 0,
            "avg_dynamic_energy_j": statistics.mean(dynamic_energies) if len(dynamic_energies) > 0 else 0,
            "std_dynamic_energy_j": statistics.stdev(dynamic_energies) if len(dynamic_energies) > 1 else 0,
            "avg_idle_baseline_w": statistics.mean(idle_baselines) if len(idle_baselines) > 0 else 0,
            "avg_ttft_ms": statistics.mean(ttfts) if len(ttfts) > 0 else 0,
            "std_ttft_ms": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
        }
        aggregated.append(agg_row)

    return aggregated


def generate_dense_input_lengths(min_len: int = 1, max_len: int = 3000,
                                num_points: int = 80, use_log: bool = True) -> List[int]:
    """生成密集的输入长度采样点

    Args:
        min_len: 最小长度
        max_len: 最大长度
        num_points: 采样点数
        use_log: 是否使用对数间距

    Returns:
        输入长度列表
    """
    if use_log:
        # 对数间距（更合理，覆盖多个数量级）
        lengths = np.logspace(np.log10(max(1, min_len)), np.log10(max_len), num_points)
    else:
        # 线性间距
        lengths = np.linspace(min_len, max_len, num_points)

    # 四舍五入并去重
    lengths = np.unique(np.round(lengths).astype(int))
    return sorted(lengths.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预填充阶段离线建模实验（连续推理版本）")
    parser.add_argument("--input-lengths", type=str,
                       default="dense",
                       help="输入token数列表，逗号分隔，或使用 'dense'（密集1-3000，80点）或 'sparse'（稀疏）")
    parser.add_argument("--repeats", type=int, default=20,
                       help="每个输入长度的重复实验次数（默认20，密集模式）")
    parser.add_argument("--power", type=int, default=350,
                       help="功率限制W（默认350W）")
    parser.add_argument("--output-dir", type=str, default="results/prefill_modeling",
                       help="结果保存目录")
    parser.add_argument("--model-path", type=str, default=None,
                       help="模型路径（默认使用Qwen2.5-7B）")
    parser.add_argument("--tokenizer-path", type=str, default="./Qwen2.5-7B-Instruct-AWQ",
                       help="Qwen2.5分词器路径")
    parser.add_argument("--sharegpt-dir", type=str, default="./input/ShareGPT",
                       help="ShareGPT数据集目录")
    parser.add_argument("--time-padding-ms", type=float, default=40.0,
                       help="推理窗口前后补偿时间（毫秒），用于覆盖功率传感器延迟")
    parser.add_argument("--sudo-password", type=str, default=None,
                       help="sudo密码（用于自动设置功率限制）")
    parser.add_argument("--skip-set-power", action="store_true",
                       help="跳过设置功率步骤，使用当前系统功率")

    args = parser.parse_args()

    # 解析输入长度列表
    if args.input_lengths == 'dense':
        # 密集采样：1-3000，约80个点
        input_lengths = generate_dense_input_lengths(1, 3000, num_points=80, use_log=True)
        print(f"使用密集采样模式: {len(input_lengths)} 个点，范围 {min(input_lengths)}-{max(input_lengths)} tokens")
    elif args.input_lengths == 'sparse':
        # 稀疏采样：原始的2的幂次
        input_lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        print(f"使用稀疏采样模式: {len(input_lengths)} 个点")
    else:
        # 自定义列表
        input_lengths = [int(x.strip()) for x in args.input_lengths.split(',')]
        print(f"使用自定义输入长度: {len(input_lengths)} 个点")

    # 计算总数据点数量
    total_points = len(input_lengths) * args.repeats
    print(f"总数据点数量: {total_points}")
    print()

    run_prefill_experiment(
        input_token_counts=input_lengths,
        repeats=args.repeats,
        power_cap=args.power,
        output_dir=args.output_dir,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        sharegpt_dir=args.sharegpt_dir,
        time_padding_ms=args.time_padding_ms,
        sudo_password=args.sudo_password,
        skip_set_power=args.skip_set_power
    )
