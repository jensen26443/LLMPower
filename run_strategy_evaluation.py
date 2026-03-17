#!/usr/bin/env python3
"""
预填充阶段功率分配策略评估实验脚本

测试5种功率策略在真实查询负载下的表现：
1. 线性分配1: P=165W
2. 线性分配2: P=175W
3. 粗粒度分桶: C≤1584→220W, 1584<C≤2176→260W, C>2176→300W
4. Baseline: P=350W
"""
import argparse
import csv
import json
import os
import time
import random
import itertools
from typing import List, Dict, Tuple
from tqdm import tqdm

from power_control import set_power_cap, get_power_cap
from llm_inference import LLMInferencer
from load_generator import LoadGenerator, ShareGPTLoader
from monitor import PowerMonitor


# 测试子集配置（来自idea1.md）
TEST_SUBSETS = [
    {"name": "subset_8", "num_prompts": 8, "target_tokens": 225},
    {"name": "subset_16", "num_prompts": 16, "target_tokens": 504},
    {"name": "subset_32", "num_prompts": 32, "target_tokens": 1581},
    {"name": "subset_64", "num_prompts": 64, "target_tokens": 2175},
    {"name": "subset_103", "num_prompts": 103, "target_tokens": 6053},
    {"name": "subset_112", "num_prompts": 112, "target_tokens": 11106},
    {"name": "subset_119", "num_prompts": 119, "target_tokens": 20295},
]

# 策略定义
STRATEGIES = [
    {"name": "linear_165w", "type": "fixed", "power": 165},
    {"name": "linear_185w", "type": "fixed", "power": 185},
    {"name": "bucket1", "type": "bucket",
     "buckets": [(6054, 165), (11107, 175), (float('inf'), 185)]},
    {"name": "bucket2", "type": "bucket",
     "buckets": [(1584, 180), (2176, 210), (6054, 230), (11107, 240), (float('inf'), 260)]},
    {"name": "baseline_350w", "type": "fixed", "power": 350},
]


def get_power_for_strategy(strategy: Dict, total_tokens: int) -> int:
    """根据策略和总token数计算功率限制"""
    if strategy["type"] == "fixed":
        return strategy["power"]
    elif strategy["type"] == "bucket":
        for threshold, power in strategy["buckets"]:
            if total_tokens <= threshold:
                return power
        return strategy["buckets"][-1][1]
    return 350


def select_subset_prompts(load_generator: LoadGenerator,
                          num_prompts: int,
                          target_tokens: int) -> List[Tuple[str, int]]:
    """为一个子集选择合适的prompts组合

    选择num_prompts个prompts，使它们的总token数接近target_tokens
    """
    # 计算平均每个prompt的token数
    avg_per_prompt = max(1, target_tokens // num_prompts)

    prompts = []
    total_actual_tokens = 0

    # 尝试选择合适长度的prompts
    for i in range(num_prompts):
        # 为每个prompt生成略微不同的长度，使总和更接近目标
        variation = random.randint(-max(1, avg_per_prompt // 10),
                                   max(1, avg_per_prompt // 10))
        target_length = max(1, avg_per_prompt + variation)

        prompt = load_generator.generate_prompt_by_token_count(
            target_length, prefer_sharegpt=True, add_unique_prefix=True)
        actual_tokens = load_generator.count_tokens(prompt)

        prompts.append((prompt, actual_tokens))
        total_actual_tokens += actual_tokens

    return prompts


def run_strategy_evaluation(output_dir: str = "./results1",
                             model_path: str = None,
                             tokenizer_path: str = "./Qwen2.5-7B-Instruct-AWQ",
                             sharegpt_dir: str = "./input/ShareGPT",
                             repeats_per_prompt: int = 100,
                             full_repeats: int = 3,
                             sudo_password: str = None,
                             skip_set_power: bool = False,
                             only_strategy: str = None):
    """运行策略评估实验"""
    data_dir = os.path.join(output_dir, "data")
    img_dir = os.path.join(output_dir, "img")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # 选择要运行的策略
    strategies_to_run = STRATEGIES
    if only_strategy:
        strategies_to_run = [s for s in STRATEGIES if s["name"] == only_strategy]
        if not strategies_to_run:
            print(f"错误: 找不到策略 '{only_strategy}'")
            return None
        print(f"仅运行策略: {only_strategy}")

    # 初始化组件
    if model_path:
        inferencer = LLMInferencer(model_name=model_path)
    else:
        inferencer = LLMInferencer()
    load_generator = LoadGenerator(sharegpt_dir=sharegpt_dir,
                                   tokenizer_name=tokenizer_path)

    # 预热GPU
    print("预热GPU...")
    warmup_prompt = load_generator.generate_prompt_by_token_count(64)
    for _ in range(5):
        inferencer.infer_prefill_only([warmup_prompt], max_tokens=1)
    time.sleep(2)

    # 所有实验结果
    all_raw_results = []
    experiment_id = f"strategy_eval_{int(time.time())}"

    # 主实验循环
    for full_repeat in range(1, full_repeats + 1):
        print(f"\n{'='*60}")
        print(f"完整重复 {full_repeat}/{full_repeats}")
        print(f"{'='*60}")

        # 为每个策略生成独立的子集（避免缓存影响）
        subset_prompts_cache = {}
        for subset in TEST_SUBSETS:
            subset_prompts_cache[subset["name"]] = select_subset_prompts(
                load_generator, subset["num_prompts"], subset["target_tokens"])

        for strategy in strategies_to_run:
            print(f"\n{'-'*60}")
            print(f"策略: {strategy['name']}")
            print(f"{'-'*60}")

            # 对于bucket策略，我们按子集分组运行（每个子集可能需要不同功率）
            if strategy["type"] == "bucket":
                for subset in TEST_SUBSETS:
                    subset_prompts = subset_prompts_cache[subset["name"]]
                    total_tokens = sum(t for _, t in subset_prompts)
                    power = get_power_for_strategy(strategy, total_tokens)

                    print(f"  子集: {subset['name']}, 总token={total_tokens}, 功率={power}W")

                    if not skip_set_power:
                        if not set_power_cap(power, sudo_password=sudo_password):
                            print(f"    设置功率{power}W失败，跳过此子集")
                            continue
                        time.sleep(15)  # 等待功率稳定

                    actual_power = get_power_cap()
                    print(f"    实际功率限制: {actual_power}W")

                    # 运行此子集的实验
                    results = run_single_subset(
                        inferencer, load_generator, subset, subset_prompts,
                        strategy, full_repeat, power, repeats_per_prompt)
                    all_raw_results.extend(results)
            else:
                # 固定功率策略：设置一次功率，运行所有子集
                power = strategy["power"]
                print(f"  固定功率: {power}W")

                if not skip_set_power:
                    if not set_power_cap(power, sudo_password=sudo_password):
                        print(f"    设置功率{power}W失败，跳过此策略")
                        continue
                    time.sleep(15)  # 等待功率稳定

                actual_power = get_power_cap()
                print(f"  实际功率限制: {actual_power}W")

                for subset in TEST_SUBSETS:
                    subset_prompts = subset_prompts_cache[subset["name"]]
                    total_tokens = sum(t for _, t in subset_prompts)
                    print(f"  子集: {subset['name']}, 总token={total_tokens}")

                    results = run_single_subset(
                        inferencer, load_generator, subset, subset_prompts,
                        strategy, full_repeat, power, repeats_per_prompt)
                    all_raw_results.extend(results)

    # 保存原始结果
    raw_file = os.path.join(data_dir, f"{experiment_id}_raw.csv")
    save_raw_results(all_raw_results, raw_file)

    # 生成聚合结果
    aggregated_file = os.path.join(data_dir, f"{experiment_id}_aggregated.csv")
    aggregated = aggregate_results(all_raw_results)
    save_aggregated_results(aggregated, aggregated_file)

    # 保存元数据
    metadata_file = os.path.join(data_dir, f"{experiment_id}_metadata.json")
    save_metadata(metadata_file, full_repeats, repeats_per_prompt)

    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"原始数据: {raw_file}")
    print(f"聚合数据: {aggregated_file}")
    print(f"元数据: {metadata_file}")
    print(f"{'='*60}")

    return {
        "raw_file": raw_file,
        "aggregated_file": aggregated_file,
        "metadata_file": metadata_file,
    }


def run_single_subset(inferencer: LLMInferencer,
                      load_generator: LoadGenerator,
                      subset: Dict,
                      subset_prompts: List[Tuple[str, int]],
                      strategy: Dict,
                      full_repeat: int,
                      power_limit: int,
                      repeats_per_prompt: int) -> List[Dict]:
    """运行单个子集的实验"""
    results = []

    # 构建实验队列：每个prompt重复repeats_per_prompt次
    experiment_queue = []
    for prompt_idx, (prompt, prompt_tokens) in enumerate(subset_prompts):
        for repeat in range(1, repeats_per_prompt + 1):
            experiment_queue.append({
                "prompt_idx": prompt_idx,
                "repeat": repeat,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
            })

    # 打乱顺序
    random.shuffle(experiment_queue)

    # 启动功率监测
    monitor = PowerMonitor(sample_interval=0.02)
    monitor.start()
    time.sleep(1.0)  # 基线采样
    experiment_start = time.time()

    # 运行所有推理
    inference_records = []
    for item in tqdm(experiment_queue, desc=f"    {subset['name']}", leave=False):
        inf_start = time.time()
        result = inferencer.infer_prefill_only([item["prompt"]], max_tokens=1)[0]
        inf_end = time.time()

        inference_records.append({
            "item": item,
            "ttft_ms": result["ttft"],
            "inference_start": inf_start,
            "inference_end": inf_end,
        })

        # 推理间隔
        time.sleep(0.1)

    # 停止监测
    experiment_end = time.time()
    power_data = monitor.stop()

    # 分析功率时间线
    final_results = analyze_power_timeline(
        inference_records, power_data, experiment_start, subset, strategy,
        full_repeat, power_limit)

    return final_results


def analyze_power_timeline(inference_records: List[Dict],
                           power_data: List[Dict],
                           experiment_start_time: float,
                           subset: Dict,
                           strategy: Dict,
                           full_repeat: int,
                           power_limit: int) -> List[Dict]:
    """分析功率时间线并提取每个推理的功率数据"""
    from statistics import mean

    # 计算空闲基线
    baseline_samples = [x["power_w"] for x in power_data
                       if x["timestamp"] < experiment_start_time]
    idle_baseline = mean(baseline_samples) if len(baseline_samples) >= 3 else 0.0

    results = []
    for record in inference_records:
        item = record["item"]
        start_time = record["inference_start"]
        end_time = record["inference_end"]

        # 提取此推理期间的功率数据
        window_points = [(start_time, 0.0)]
        for pd in power_data:
            t = pd["timestamp"]
            if start_time < t < end_time:
                window_points.append((t, pd["power_w"]))
        window_points.append((end_time, 0.0))
        window_points.sort(key=lambda x: x[0])

        # 填充缺失的端点（使用插值）
        def get_power_at(t):
            for i in range(1, len(power_data)):
                if power_data[i-1]["timestamp"] <= t <= power_data[i]["timestamp"]:
                    t1, p1 = power_data[i-1]["timestamp"], power_data[i-1]["power_w"]
                    t2, p2 = power_data[i]["timestamp"], power_data[i]["power_w"]
                    if t2 == t1:
                        return p1
                    ratio = (t - t1) / (t2 - t1)
                    return p1 + ratio * (p2 - p1)
            return power_data[0]["power_w"] if power_data else 0.0

        if len(window_points) == 2:
            window_points[0] = (start_time, get_power_at(start_time))
            window_points[1] = (end_time, get_power_at(end_time))

        # 计算能耗和平均功率
        energy = 0.0
        duration = max(0.0, end_time - start_time)
        for i in range(1, len(window_points)):
            t_prev, p_prev = window_points[i-1]
            t_curr, p_curr = window_points[i]
            dt = t_curr - t_prev
            if dt > 0:
                avg_p = (p_prev + p_curr) / 2
                energy += avg_p * dt

        avg_power = energy / duration if duration > 0 else 0.0
        dynamic_power = max(0.0, avg_power - idle_baseline)
        dynamic_energy = max(0.0, energy - idle_baseline * duration)

        results.append({
            "full_repeat": full_repeat,
            "strategy": strategy["name"],
            "subset": subset["name"],
            "subset_target_tokens": subset["target_tokens"],
            "subset_num_prompts": subset["num_prompts"],
            "power_limit": power_limit,
            "prompt_idx": item["prompt_idx"],
            "prompt_repeat": item["repeat"],
            "prompt_tokens": item["prompt_tokens"],
            "ttft_ms": record["ttft_ms"],
            "inference_start": start_time,
            "inference_end": end_time,
            "inference_duration_s": duration,
            "avg_power_w": avg_power,
            "total_energy_j": energy,
            "dynamic_power_w": dynamic_power,
            "dynamic_energy_j": dynamic_energy,
            "idle_baseline_w": idle_baseline,
        })

    return results


def aggregate_results(raw_results: List[Dict]) -> List[Dict]:
    """按策略和子集聚合结果"""
    from collections import defaultdict
    import statistics

    key_func = lambda r: (r["full_repeat"], r["strategy"], r["subset"])
    grouped = defaultdict(list)
    for r in raw_results:
        grouped[key_func(r)].append(r)

    aggregated = []
    for (full_repeat, strategy, subset), group in sorted(grouped.items()):
        ttfts = [r["ttft_ms"] for r in group]
        energies = [r["total_energy_j"] for r in group]
        powers = [r["avg_power_w"] for r in group]
        dyn_energies = [r["dynamic_energy_j"] for r in group]
        dyn_powers = [r["dynamic_power_w"] for r in group]

        subset_info = next((s for s in TEST_SUBSETS if s["name"] == subset), None)

        aggregated.append({
            "full_repeat": full_repeat,
            "strategy": strategy,
            "subset": subset,
            "subset_target_tokens": subset_info["target_tokens"] if subset_info else 0,
            "subset_num_prompts": subset_info["num_prompts"] if subset_info else 0,
            "power_limit": group[0]["power_limit"],
            "num_samples": len(group),
            "avg_ttft_ms": statistics.mean(ttfts),
            "std_ttft_ms": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
            "avg_energy_j": statistics.mean(energies),
            "std_energy_j": statistics.stdev(energies) if len(energies) > 1 else 0,
            "avg_power_w": statistics.mean(powers),
            "std_power_w": statistics.stdev(powers) if len(powers) > 1 else 0,
            "avg_dynamic_energy_j": statistics.mean(dyn_energies),
            "std_dynamic_energy_j": statistics.stdev(dyn_energies) if len(dyn_energies) > 1 else 0,
            "avg_dynamic_power_w": statistics.mean(dyn_powers),
            "std_dynamic_power_w": statistics.stdev(dyn_powers) if len(dyn_powers) > 1 else 0,
            "total_energy_j_sum": sum(energies),
            "idle_baseline_w": group[0]["idle_baseline_w"],
        })

    return aggregated


def save_raw_results(results: List[Dict], filepath: str):
    """保存原始结果到CSV"""
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_aggregated_results(results: List[Dict], filepath: str):
    """保存聚合结果到CSV"""
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_metadata(filepath: str, full_repeats: int, repeats_per_prompt: int):
    """保存实验元数据"""
    metadata = {
        "timestamp": time.time(),
        "full_repeats": full_repeats,
        "repeats_per_prompt": repeats_per_prompt,
        "subsets": TEST_SUBSETS,
        "strategies": [
            {k: v for k, v in s.items() if k != "buckets"}
            if s["type"] == "fixed" else s
            for s in STRATEGIES
        ],
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预填充阶段功率分配策略评估实验")
    parser.add_argument("--output-dir", type=str, default="./results1",
                       help="结果保存目录")
    parser.add_argument("--model-path", type=str, default=None,
                       help="模型路径")
    parser.add_argument("--tokenizer-path", type=str,
                       default="./Qwen2.5-7B-Instruct-AWQ",
                       help="分词器路径")
    parser.add_argument("--sharegpt-dir", type=str, default="./input/ShareGPT",
                       help="ShareGPT数据集目录")
    parser.add_argument("--repeats-per-prompt", type=int, default=100,
                       help="每个prompt重复次数（默认100）")
    parser.add_argument("--full-repeats", type=int, default=3,
                       help="完整实验重复次数（默认3）")
    parser.add_argument("--sudo-password", type=str, default=None,
                       help="sudo密码")
    parser.add_argument("--skip-set-power", action="store_true",
                       help="跳过设置功率步骤")
    parser.add_argument("--only-strategy", type=str, default=None,
                       help="仅运行指定策略（如 bucket1）")

    args = parser.parse_args()

    run_strategy_evaluation(
        output_dir=args.output_dir,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        sharegpt_dir=args.sharegpt_dir,
        repeats_per_prompt=args.repeats_per_prompt,
        full_repeats=args.full_repeats,
        sudo_password=args.sudo_password,
        skip_set_power=args.skip_set_power,
        only_strategy=args.only_strategy,
    )
