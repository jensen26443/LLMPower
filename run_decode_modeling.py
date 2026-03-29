#!/usr/bin/env python3
"""
解码阶段离线建模实验脚本（vLLM 在线服务模式）

用于拟合：
P_decoding = f(B, KV)
TBT = g(B, KV)

其中：
- B 表示 batch size
- KV 使用近似指标替代，默认记录：
  - context_total_tokens
  - approx_kv_pressure = batch_size * avg_context_len_per_request
  - normalized_kv_blocks = ceil(final_context_len_per_request / 16)
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from llm_inference import LLMInferencer
from load_generator import LoadGenerator
from monitor import PowerMonitor
from power_control import get_power_cap, set_power_cap

DEFAULT_ENABLE_CHUNKED_PREFILL = True
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
DEFAULT_MAX_NUM_SEQS = 64
DEFAULT_QUEUE_SEED = 20260329
DEFAULT_SAMPLING_SEED = 20260329


def percentile(values: List[float], p: float) -> float:
    """使用线性插值计算百分位数。"""
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_metric(values: List[float], prefix: str) -> Dict:
    """为一组指标生成 mean/p50/p95/p99 汇总字段。"""
    if not values:
        return {
            f"avg_{prefix}_ms": 0.0,
            f"mean_{prefix}_ms": 0.0,
            f"p50_{prefix}_ms": 0.0,
            f"p95_{prefix}_ms": 0.0,
            f"p99_{prefix}_ms": 0.0,
        }

    mean_value = statistics.mean(values)
    return {
        f"avg_{prefix}_ms": mean_value,
        f"mean_{prefix}_ms": mean_value,
        f"p50_{prefix}_ms": percentile(values, 50),
        f"p95_{prefix}_ms": percentile(values, 95),
        f"p99_{prefix}_ms": percentile(values, 99),
    }


def build_scheduler_signature(enable_chunked_prefill: bool,
                              max_num_batched_tokens: int,
                              max_num_seqs: int) -> str:
    return (
        f"cp{1 if enable_chunked_prefill else 0}"
        f"_mbt{max_num_batched_tokens}"
        f"_mns{max_num_seqs}"
    )


def build_experiment_metadata(enable_chunked_prefill: bool,
                              max_num_batched_tokens: int,
                              max_num_seqs: int,
                              queue_seed: int,
                              sampling_seed: int) -> Dict:
    return {
        "enable_chunked_prefill": enable_chunked_prefill,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "queue_seed": queue_seed,
        "sampling_seed": sampling_seed,
        "scheduler_signature": build_scheduler_signature(
            enable_chunked_prefill,
            max_num_batched_tokens,
            max_num_seqs,
        ),
    }


def build_experiment_id(prefix: str, metadata: Dict, timestamp: int) -> str:
    return f"{prefix}_{metadata['scheduler_signature']}_{timestamp}"


def build_experiment_queue(batch_sizes: List[int],
                           output_lengths: List[int],
                           repeats: int,
                           queue_seed: int) -> List[Tuple[int, int, int]]:
    queue = []
    for batch_size in batch_sizes:
        for output_length in output_lengths:
            for repeat_id in range(1, repeats + 1):
                queue.append((batch_size, output_length, repeat_id))

    rng = random.Random(queue_seed)
    rng.shuffle(queue)
    return queue


def build_decode_request_extra_body(output_length: int, sampling_seed: int) -> Dict:
    return {
        "min_tokens": output_length,
        "ignore_eos": True,
        "top_p": 1.0,
        "seed": sampling_seed,
    }


def build_power_window_stats(start_time: float, end_time: float, power_data: List[Dict],
                             idle_baseline_w: float, time_padding_ms: float) -> Dict:
    """按给定时间窗口计算平均功率、能耗和峰值。"""
    if not power_data:
        return {
            "avg_power_w": 0.0,
            "peak_power_w": 0.0,
            "min_power_w": 0.0,
            "total_energy_j": 0.0,
            "dynamic_power_w": 0.0,
            "dynamic_energy_j": 0.0,
        }

    def interpolate_power(timestamp: float) -> float:
        if timestamp <= power_data[0]["timestamp"]:
            return power_data[0]["power_w"]
        if timestamp >= power_data[-1]["timestamp"]:
            return power_data[-1]["power_w"]

        for index in range(1, len(power_data)):
            prev_point = power_data[index - 1]
            next_point = power_data[index]
            if prev_point["timestamp"] <= timestamp <= next_point["timestamp"]:
                span = next_point["timestamp"] - prev_point["timestamp"]
                if span <= 0:
                    return next_point["power_w"]
                ratio = (timestamp - prev_point["timestamp"]) / span
                return prev_point["power_w"] + ratio * (next_point["power_w"] - prev_point["power_w"])
        return power_data[-1]["power_w"]

    exact_start = min(start_time, end_time)
    exact_end = max(start_time, end_time)
    padding_s = max(0.0, time_padding_ms / 1000.0)

    window_points = [(exact_start, interpolate_power(exact_start))]
    for point in power_data:
        if exact_start < point["timestamp"] < exact_end:
            window_points.append((point["timestamp"], point["power_w"]))
    window_points.append((exact_end, interpolate_power(exact_end)))
    window_points.sort(key=lambda item: item[0])

    energy = 0.0
    relevant_powers = [power for _, power in window_points]
    for index in range(1, len(window_points)):
        t_prev, p_prev = window_points[index - 1]
        t_curr, p_curr = window_points[index]
        dt = t_curr - t_prev
        if dt > 0:
            energy += (p_prev + p_curr) * 0.5 * dt

    duration = max(0.0, exact_end - exact_start)
    avg_power = energy / duration if duration > 0 else 0.0

    padded_powers = [
        point["power_w"]
        for point in power_data
        if exact_start - padding_s <= point["timestamp"] <= exact_end + padding_s
    ]
    if not padded_powers:
        padded_powers = relevant_powers

    dynamic_energy = max(0.0, energy - idle_baseline_w * duration)
    dynamic_power = dynamic_energy / duration if duration > 0 else 0.0

    return {
        "avg_power_w": avg_power,
        "peak_power_w": max(padded_powers) if padded_powers else 0.0,
        "min_power_w": min(padded_powers) if padded_powers else 0.0,
        "total_energy_j": energy,
        "dynamic_power_w": dynamic_power,
        "dynamic_energy_j": dynamic_energy,
    }


def analyze_batch_power_timeline(batch_results: List[Dict], power_data: List[Dict],
                                 experiment_start_time: float,
                                 time_padding_ms: float = 20.0) -> List[Dict]:
    """分析每个 batch 的 decode 窗口功率。"""
    baseline_samples = [point["power_w"] for point in power_data if point["timestamp"] < experiment_start_time]
    idle_baseline_w = statistics.mean(baseline_samples) if len(baseline_samples) >= 3 else 0.0

    analyzed = []
    for result in batch_results:
        stats = build_power_window_stats(
            start_time=result["decode_start_time"],
            end_time=result["decode_end_time"],
            power_data=power_data,
            idle_baseline_w=idle_baseline_w,
            time_padding_ms=time_padding_ms,
        )
        merged = dict(result)
        merged.update(stats)
        merged["idle_baseline_w"] = idle_baseline_w
        analyzed.append(merged)

    return analyzed


def estimate_request_tbt_ms(request_result: Dict, output_tokens: int) -> float:
    """根据实际输出 token 数重新估计平均 TBT。"""
    if output_tokens > 1 and request_result["ttft"] < request_result["e2e"]:
        return (request_result["e2e"] - request_result["ttft"]) / (output_tokens - 1)
    return request_result.get("tbt", 0.0)


def build_request_diagnostics(request_result: Dict) -> Dict[str, float]:
    ttft_ms = float(request_result.get("ttft", 0.0) or 0.0)
    e2e_ms = float(request_result.get("e2e", 0.0) or 0.0)
    decode_duration_ms = max(0.0, e2e_ms - ttft_ms)
    avg_itl_ms = float(request_result.get("avg_itl", 0.0) or 0.0)
    stream_chunk_count = int(request_result.get("stream_chunk_count", 0) or 0)
    ttft_ratio = (ttft_ms / e2e_ms) if e2e_ms > 0 else 0.0
    return {
        "avg_itl_ms": avg_itl_ms,
        "stream_chunk_count": stream_chunk_count,
        "ttft_ratio": ttft_ratio,
        "decode_duration_ms": decode_duration_ms,
    }


def summarize_batch(index: int, repeat_id: int, batch_size: int, target_output_tokens: int,
                    prompt_token_count: int, request_results: List[Dict],
                    tokenizer: LoadGenerator) -> Tuple[Dict, List[Dict]]:
    """汇总单个 batch 的请求级和 batch 级统计。"""
    request_rows = []
    generated_token_counts = []
    ttfts = []
    tbts = []
    e2es = []
    avg_itls = []
    stream_chunk_counts = []
    ttft_ratios = []
    decode_durations = []
    start_times = []
    end_times = []
    first_token_times = []

    for request_index, request_result in enumerate(request_results):
        output_tokens = request_result.get("token_count", 0)
        if output_tokens <= 0 and request_result["generated_text"]:
            output_tokens = tokenizer.count_tokens(request_result["generated_text"])
        request_tbt = estimate_request_tbt_ms(request_result, output_tokens)
        diagnostics = build_request_diagnostics(request_result)
        request_start_time = request_result.get("start_time_wall", request_result["start_time"])
        request_end_time = request_result.get("end_time_wall", request_result["end_time"])
        decode_start_time = request_result.get("first_token_time_wall") or request_end_time

        generated_token_counts.append(output_tokens)
        ttfts.append(request_result["ttft"])
        tbts.append(request_tbt)
        e2es.append(request_result["e2e"])
        avg_itls.append(diagnostics["avg_itl_ms"])
        stream_chunk_counts.append(diagnostics["stream_chunk_count"])
        ttft_ratios.append(diagnostics["ttft_ratio"])
        decode_durations.append(diagnostics["decode_duration_ms"])
        start_times.append(request_start_time)
        end_times.append(request_end_time)
        first_token_times.append(decode_start_time)

        request_rows.append({
            "batch_index": index,
            "repeat_id": repeat_id,
            "request_index": request_index,
            "batch_size": batch_size,
            "target_output_tokens": target_output_tokens,
            "prompt_token_count": prompt_token_count,
            "actual_output_tokens": output_tokens,
            "ttft_ms": request_result["ttft"],
            "tbt_ms": request_tbt,
            "e2e_ms": request_result["e2e"],
            "avg_itl_ms": diagnostics["avg_itl_ms"],
            "stream_chunk_count": diagnostics["stream_chunk_count"],
            "ttft_ratio": diagnostics["ttft_ratio"],
            "decode_duration_ms": diagnostics["decode_duration_ms"],
            "request_start_time": request_start_time,
            "first_token_time": decode_start_time,
            "request_end_time": request_end_time,
            "generated_preview": request_result["generated_text"][:80],
        })

    avg_generated_tokens = statistics.mean(generated_token_counts) if generated_token_counts else 0.0
    avg_context_len_per_request = prompt_token_count + avg_generated_tokens
    avg_active_context_len_per_request = prompt_token_count + (avg_generated_tokens / 2.0)
    normalized_kv_blocks = math.ceil(avg_context_len_per_request / 16.0) if avg_context_len_per_request > 0 else 0
    total_kv_blocks = normalized_kv_blocks * batch_size

    batch_row = {
        "index": index,
        "repeat_id": repeat_id,
        "batch_size": batch_size,
        "target_output_tokens": target_output_tokens,
        "prompt_token_count": prompt_token_count,
        "avg_generated_tokens": avg_generated_tokens,
        "min_generated_tokens": min(generated_token_counts) if generated_token_counts else 0,
        "max_generated_tokens": max(generated_token_counts) if generated_token_counts else 0,
        "avg_stream_chunk_count": statistics.mean(stream_chunk_counts) if stream_chunk_counts else 0.0,
        "min_stream_chunk_count": min(stream_chunk_counts) if stream_chunk_counts else 0,
        "max_stream_chunk_count": max(stream_chunk_counts) if stream_chunk_counts else 0,
        "avg_request_avg_itl_ms": statistics.mean(avg_itls) if avg_itls else 0.0,
        "avg_ttft_ratio": statistics.mean(ttft_ratios) if ttft_ratios else 0.0,
        "avg_decode_duration_ms": statistics.mean(decode_durations) if decode_durations else 0.0,
        "max_ttft_ms": max(ttfts) if ttfts else 0.0,
        "batch_start_time": min(start_times) if start_times else 0.0,
        "batch_end_time": max(end_times) if end_times else 0.0,
        "decode_start_time": min(first_token_times) if first_token_times else (min(start_times) if start_times else 0.0),
        "decode_end_time": max(end_times) if end_times else 0.0,
        "final_context_len_per_request": avg_context_len_per_request,
        "avg_active_context_len_per_request": avg_active_context_len_per_request,
        "context_total_tokens": batch_size * avg_context_len_per_request,
        "approx_kv_pressure": batch_size * avg_active_context_len_per_request,
        "normalized_kv_blocks": normalized_kv_blocks,
        "total_kv_blocks": total_kv_blocks,
    }
    batch_row.update(summarize_metric(ttfts, "ttft"))
    batch_row.update(summarize_metric(tbts, "tbt"))
    batch_row.update(summarize_metric(e2es, "e2e"))
    return batch_row, request_rows


def aggregate_results(batch_results: List[Dict], request_results: List[Dict], metadata: Dict) -> List[Dict]:
    """按 batch size 和输出长度聚合结果。"""
    batch_grouped: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    request_grouped: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)

    for row in batch_results:
        batch_grouped[(row["batch_size"], row["target_output_tokens"])].append(row)
    for row in request_results:
        request_grouped[(row["batch_size"], row["target_output_tokens"])].append(row)

    aggregated = []
    for batch_size, output_tokens in sorted(batch_grouped.keys()):
        batch_group = batch_grouped[(batch_size, output_tokens)]
        request_group = request_grouped.get((batch_size, output_tokens), [])
        ttfts = [row["ttft_ms"] for row in request_group]
        tbts = [row["tbt_ms"] for row in request_group]
        e2es = [row["e2e_ms"] for row in request_group]
        avg_itls = [row["avg_itl_ms"] for row in request_group]
        stream_chunk_counts = [row["stream_chunk_count"] for row in request_group]
        ttft_ratios = [row["ttft_ratio"] for row in request_group]
        decode_durations = [row["decode_duration_ms"] for row in request_group]

        aggregated_row = {
            "batch_size": batch_size,
            "target_output_tokens": output_tokens,
            "count": len(batch_group),
            "request_count": len(request_group),
            "avg_generated_tokens": statistics.mean(row["avg_generated_tokens"] for row in batch_group),
            "std_generated_tokens": statistics.stdev([row["avg_generated_tokens"] for row in batch_group]) if len(batch_group) > 1 else 0.0,
            "avg_power_w": statistics.mean(row["avg_power_w"] for row in batch_group),
            "std_power_w": statistics.stdev([row["avg_power_w"] for row in batch_group]) if len(batch_group) > 1 else 0.0,
            "peak_power_w": statistics.mean(row["peak_power_w"] for row in batch_group),
            "avg_energy_j": statistics.mean(row["total_energy_j"] for row in batch_group),
            "std_energy_j": statistics.stdev([row["total_energy_j"] for row in batch_group]) if len(batch_group) > 1 else 0.0,
            "avg_dynamic_power_w": statistics.mean(row["dynamic_power_w"] for row in batch_group),
            "std_dynamic_power_w": statistics.stdev([row["dynamic_power_w"] for row in batch_group]) if len(batch_group) > 1 else 0.0,
            "avg_dynamic_energy_j": statistics.mean(row["dynamic_energy_j"] for row in batch_group),
            "std_dynamic_energy_j": statistics.stdev([row["dynamic_energy_j"] for row in batch_group]) if len(batch_group) > 1 else 0.0,
            "std_ttft_ms": statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
            "std_tbt_ms": statistics.stdev(tbts) if len(tbts) > 1 else 0.0,
            "std_e2e_ms": statistics.stdev(e2es) if len(e2es) > 1 else 0.0,
            "avg_stream_chunk_count": statistics.mean(stream_chunk_counts) if stream_chunk_counts else 0.0,
            "std_stream_chunk_count": statistics.stdev(stream_chunk_counts) if len(stream_chunk_counts) > 1 else 0.0,
            "avg_request_avg_itl_ms": statistics.mean(avg_itls) if avg_itls else 0.0,
            "std_request_avg_itl_ms": statistics.stdev(avg_itls) if len(avg_itls) > 1 else 0.0,
            "avg_ttft_ratio": statistics.mean(ttft_ratios) if ttft_ratios else 0.0,
            "std_ttft_ratio": statistics.stdev(ttft_ratios) if len(ttft_ratios) > 1 else 0.0,
            "avg_decode_duration_ms": statistics.mean(decode_durations) if decode_durations else 0.0,
            "std_decode_duration_ms": statistics.stdev(decode_durations) if len(decode_durations) > 1 else 0.0,
            "avg_context_total_tokens": statistics.mean(row["context_total_tokens"] for row in batch_group),
            "avg_approx_kv_pressure": statistics.mean(row["approx_kv_pressure"] for row in batch_group),
            "avg_normalized_kv_blocks": statistics.mean(row["normalized_kv_blocks"] for row in batch_group),
            "avg_idle_baseline_w": statistics.mean(row["idle_baseline_w"] for row in batch_group),
        }
        aggregated_row.update(metadata)
        aggregated_row.update(summarize_metric(ttfts, "ttft"))
        aggregated_row.update(summarize_metric(tbts, "tbt"))
        aggregated_row.update(summarize_metric(e2es, "e2e"))
        aggregated.append(aggregated_row)
    return aggregated


def save_csv(file_path: str, rows: List[Dict], fieldnames: List[str]):
    with open(file_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_concurrent_with_retry(inferencer: LLMInferencer,
                                prompts: List[str],
                                max_tokens: int,
                                temperature: float,
                                extra_body: Optional[Dict],
                                max_retries: int,
                                retry_backoff_sec: float,
                                phase_label: str) -> List[Dict]:
    """带重试的并发推理，避免短暂 503 直接中断整轮实验。"""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return inferencer.infer_concurrent(
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
        except Exception as error:
            last_error = error
            if attempt >= max_retries:
                break
            wait_time = retry_backoff_sec * attempt
            print(
                f"{phase_label} 失败，第 {attempt}/{max_retries} 次尝试异常: {error}，"
                f"{wait_time:.1f}s 后重试..."
            )
            time.sleep(max(0.0, wait_time))

    raise RuntimeError(f"{phase_label} 在 {max_retries} 次尝试后仍失败: {last_error}") from last_error


def run_decode_experiment(batch_sizes: List[int],
                          output_lengths: List[int],
                          repeats: int = 5,
                          power_cap: int = 350,
                          output_dir: str = "results_decode/decode_modeling",
                          model_path: str = "./Qwen2.5-7B-Instruct-AWQ",
                          served_model_name: str = "Qwen2.5-7B-Instruct-AWQ",
                          tokenizer_path: str = "./Qwen2.5-7B-Instruct-AWQ",
                          base_url: str = "http://localhost:8000/v1",
                          api_key: str = "EMPTY",
                          start_server: bool = False,
                          gpu_memory_utilization: float = 0.85,
                          prompt_token_count: int = 1,
                          idle_baseline_sec: float = 2.0,
                          inter_batch_sec: float = 0.8,
                          time_padding_ms: float = 20.0,
                          max_retries: int = 6,
                          retry_backoff_sec: float = 2.0,
                          enable_chunked_prefill: bool = DEFAULT_ENABLE_CHUNKED_PREFILL,
                          max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
                          max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
                          queue_seed: int = DEFAULT_QUEUE_SEED,
                          sampling_seed: int = DEFAULT_SAMPLING_SEED,
                          sudo_password: Optional[str] = None,
                          skip_set_power: bool = False):
    """运行解码阶段离线建模实验。"""
    os.makedirs(output_dir, exist_ok=True)

    if not skip_set_power:
        print(f"设置功率限制为 {power_cap}W")
        if not set_power_cap(power_cap, sudo_password=sudo_password):
            print("设置功率失败，实验终止")
            return None
    else:
        print("跳过功率设置，使用当前系统功率限制")

    actual_power_cap = get_power_cap()
    print(f"实际功率限制: {actual_power_cap}W")
    print("等待功率稳定20秒...")
    time.sleep(20)

    inferencer = LLMInferencer(
        model_name=model_path,
        served_model_name=served_model_name,
        use_service=True,
        base_url=base_url,
        api_key=api_key,
        start_server=start_server,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_prefix_caching=True,
        service_request_mode="completion",
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    load_generator = LoadGenerator(sharegpt_dir="", tokenizer_name=tokenizer_path)
    experiment_metadata = build_experiment_metadata(
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        queue_seed=queue_seed,
        sampling_seed=sampling_seed,
    )

    base_prompt = load_generator.generate_prompt_by_token_count(
        prompt_token_count,
        prefer_sharegpt=False,
        add_unique_prefix=False,
    )
    actual_prompt_tokens = load_generator.count_tokens(base_prompt)
    print(f"实验 prompt token 数: 目标={prompt_token_count}, 实际={actual_prompt_tokens}")
    print(f"实验 prompt 预览: {base_prompt[:50]}")
    print(
        "调度参数: "
        f"chunked_prefill={enable_chunked_prefill}, "
        f"max_num_batched_tokens={max_num_batched_tokens}, "
        f"max_num_seqs={max_num_seqs}, "
        f"queue_seed={queue_seed}, sampling_seed={sampling_seed}"
    )

    print("预热服务...")
    warmup_prompts = [base_prompt] * min(4, max(batch_sizes))
    warmup_extra_body = build_decode_request_extra_body(
        output_length=min(16, max(output_lengths)),
        sampling_seed=sampling_seed,
    )
    for _ in range(2):
        infer_concurrent_with_retry(
            inferencer,
            warmup_prompts,
            max_tokens=min(16, max(output_lengths)),
            temperature=0.0,
            extra_body=warmup_extra_body,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
            phase_label="warmup",
        )
    time.sleep(2)

    experiment_queue = build_experiment_queue(
        batch_sizes=batch_sizes,
        output_lengths=output_lengths,
        repeats=repeats,
        queue_seed=queue_seed,
    )

    print(
        f"\n实验队列已构建：{len(batch_sizes)} 个 batch size × "
        f"{len(output_lengths)} 个输出长度 × {repeats} 次重复 = {len(experiment_queue)} 次实验"
    )

    monitor = PowerMonitor(sample_interval=0.02)
    monitor.start()
    print(f"采样后端: {monitor._backend}")
    if monitor._backend != "pynvml":
        print("警告: 当前使用 nvidia-smi 轮询，短时解码窗口功率会被平滑，建议安装 pynvml 后重跑。")

    print(f"采集空闲基线 {idle_baseline_sec:.1f}s...")
    time.sleep(max(0.0, idle_baseline_sec))
    experiment_start_time = time.time()

    batch_results = []
    request_results = []

    for index, (batch_size, output_length, repeat_id) in enumerate(tqdm(experiment_queue, desc="解码实验中")):
        prompts = [base_prompt] * batch_size
        extra_body = build_decode_request_extra_body(
            output_length=output_length,
            sampling_seed=sampling_seed,
        )
        inference_results = infer_concurrent_with_retry(
            inferencer,
            prompts,
            max_tokens=output_length,
            temperature=0.0,
            extra_body=extra_body,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
            phase_label=f"batch={batch_size}, output={output_length}, repeat={repeat_id}",
        )

        batch_row, request_rows = summarize_batch(
            index=index,
            repeat_id=repeat_id,
            batch_size=batch_size,
            target_output_tokens=output_length,
            prompt_token_count=actual_prompt_tokens,
            request_results=inference_results,
            tokenizer=load_generator,
        )
        batch_row.update(experiment_metadata)
        for request_row in request_rows:
            request_row.update(experiment_metadata)
        batch_results.append(batch_row)
        request_results.extend(request_rows)
        time.sleep(max(0.0, inter_batch_sec))

    experiment_end_time = time.time()
    power_data = monitor.stop()
    total_duration = experiment_end_time - experiment_start_time

    final_batch_results = analyze_batch_power_timeline(
        batch_results,
        power_data,
        experiment_start_time=experiment_start_time,
        time_padding_ms=time_padding_ms,
    )
    aggregated_results = aggregate_results(final_batch_results, request_results, experiment_metadata)

    experiment_id = build_experiment_id(
        prefix="decode_modeling_service",
        metadata=experiment_metadata,
        timestamp=int(time.time()),
    )
    timeline_file = os.path.join(output_dir, f"{experiment_id}_power_timeline.csv")
    raw_batch_file = os.path.join(output_dir, f"{experiment_id}_raw.csv")
    raw_request_file = os.path.join(output_dir, f"{experiment_id}_requests.csv")
    aggregated_file = os.path.join(output_dir, f"{experiment_id}_aggregated.csv")
    metadata_file = os.path.join(output_dir, f"{experiment_id}_metadata.json")

    timeline_rows = [
        {
            "timestamp": point["timestamp"],
            "relative_time": point["timestamp"] - experiment_start_time,
            "power_w": point["power_w"],
            "memory_gb": point["memory_gb"],
            "temperature_c": point["temperature_c"],
        }
        for point in power_data
    ]
    save_csv(
        timeline_file,
        timeline_rows,
        ["timestamp", "relative_time", "power_w", "memory_gb", "temperature_c"],
    )

    save_csv(
        raw_batch_file,
        final_batch_results,
        [
            "index", "repeat_id", "batch_size", "target_output_tokens", "prompt_token_count",
            "enable_chunked_prefill", "max_num_batched_tokens", "max_num_seqs",
            "queue_seed", "sampling_seed", "scheduler_signature",
            "avg_generated_tokens", "min_generated_tokens", "max_generated_tokens",
            "avg_stream_chunk_count", "min_stream_chunk_count", "max_stream_chunk_count",
            "avg_request_avg_itl_ms", "avg_ttft_ratio", "avg_decode_duration_ms",
            "avg_ttft_ms", "mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "max_ttft_ms",
            "avg_tbt_ms", "mean_tbt_ms", "p50_tbt_ms", "p95_tbt_ms", "p99_tbt_ms",
            "avg_e2e_ms", "mean_e2e_ms", "p50_e2e_ms", "p95_e2e_ms", "p99_e2e_ms",
            "batch_start_time", "batch_end_time", "decode_start_time", "decode_end_time",
            "final_context_len_per_request", "avg_active_context_len_per_request",
            "context_total_tokens", "approx_kv_pressure", "normalized_kv_blocks", "total_kv_blocks",
            "avg_power_w", "peak_power_w", "min_power_w", "total_energy_j",
            "dynamic_power_w", "dynamic_energy_j", "idle_baseline_w",
        ],
    )

    save_csv(
        raw_request_file,
        request_results,
        [
            "batch_index", "repeat_id", "request_index", "batch_size", "target_output_tokens",
            "enable_chunked_prefill", "max_num_batched_tokens", "max_num_seqs",
            "queue_seed", "sampling_seed", "scheduler_signature",
            "prompt_token_count", "actual_output_tokens", "ttft_ms", "tbt_ms", "e2e_ms",
            "avg_itl_ms", "stream_chunk_count", "ttft_ratio", "decode_duration_ms",
            "request_start_time", "first_token_time", "request_end_time", "generated_preview",
        ],
    )

    save_csv(
        aggregated_file,
        aggregated_results,
        [
            "batch_size", "target_output_tokens", "count", "request_count",
            "enable_chunked_prefill", "max_num_batched_tokens", "max_num_seqs",
            "queue_seed", "sampling_seed", "scheduler_signature",
            "avg_generated_tokens", "std_generated_tokens",
            "avg_stream_chunk_count", "std_stream_chunk_count",
            "avg_request_avg_itl_ms", "std_request_avg_itl_ms",
            "avg_ttft_ratio", "std_ttft_ratio",
            "avg_decode_duration_ms", "std_decode_duration_ms",
            "avg_power_w", "std_power_w", "peak_power_w",
            "avg_energy_j", "std_energy_j",
            "avg_dynamic_power_w", "std_dynamic_power_w",
            "avg_dynamic_energy_j", "std_dynamic_energy_j",
            "avg_ttft_ms", "mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "std_ttft_ms",
            "avg_tbt_ms", "mean_tbt_ms", "p50_tbt_ms", "p95_tbt_ms", "p99_tbt_ms", "std_tbt_ms",
            "avg_e2e_ms", "mean_e2e_ms", "p50_e2e_ms", "p95_e2e_ms", "p99_e2e_ms", "std_e2e_ms",
            "avg_context_total_tokens", "avg_approx_kv_pressure", "avg_normalized_kv_blocks",
            "avg_idle_baseline_w",
        ],
    )

    with open(metadata_file, "w", encoding="utf-8") as file_obj:
        json.dump(experiment_metadata, file_obj, ensure_ascii=False, indent=2)

    print("\n实验完成！")
    print(f"总耗时: {total_duration:.1f}s")
    print(f"功率采样点: {len(power_data)}")
    print(f"批级原始数据: {raw_batch_file}")
    print(f"请求级原始数据: {raw_request_file}")
    print(f"聚合数据: {aggregated_file}")
    print(f"功率时间线: {timeline_file}")
    print(f"实验元数据: {metadata_file}")
    if aggregated_results:
        sample_row = aggregated_results[0]
        print(
            "示例汇总: "
            f"TTFT mean/p50/p95/p99={sample_row['mean_ttft_ms']:.2f}/"
            f"{sample_row['p50_ttft_ms']:.2f}/{sample_row['p95_ttft_ms']:.2f}/{sample_row['p99_ttft_ms']:.2f} ms, "
            f"TBT mean/p50/p95/p99={sample_row['mean_tbt_ms']:.2f}/"
            f"{sample_row['p50_tbt_ms']:.2f}/{sample_row['p95_tbt_ms']:.2f}/{sample_row['p99_tbt_ms']:.2f} ms, "
            f"E2E mean/p50/p95/p99={sample_row['mean_e2e_ms']:.2f}/"
            f"{sample_row['p50_e2e_ms']:.2f}/{sample_row['p95_e2e_ms']:.2f}/{sample_row['p99_e2e_ms']:.2f} ms"
        )

    return {
        "raw_file": raw_batch_file,
        "request_file": raw_request_file,
        "aggregated_file": aggregated_file,
        "timeline_file": timeline_file,
        "record_count": len(final_batch_results),
    }


def parse_int_list(raw_value: str) -> List[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解码阶段离线建模实验（vLLM 在线服务模式）")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,6,8,12,16,24,32,40,48,50,56,60,64",
                        help="batch size 列表，逗号分隔")
    parser.add_argument("--output-lengths", type=str, default="10,20,40,50,75,100,150,200,300",
                        help="输出长度列表，逗号分隔")
    parser.add_argument("--repeats", type=int, default=5,
                        help="每个配置重复次数")
    parser.add_argument("--power", type=int, default=350,
                        help="功率限制（W）")
    parser.add_argument("--output-dir", type=str, default="results_decode/decode_modeling",
                        help="结果输出目录")
    parser.add_argument("--model-path", type=str, default="./Qwen2.5-7B-Instruct-AWQ",
                        help="模型路径（用于自动启动服务）")
    parser.add_argument("--served-model-name", type=str, default="Qwen2.5-7B-Instruct-AWQ",
                        help="OpenAI API 使用的模型名")
    parser.add_argument("--tokenizer-path", type=str, default="./Qwen2.5-7B-Instruct-AWQ",
                        help="分词器路径")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM OpenAI 服务地址")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="OpenAI 兼容 API key")
    parser.add_argument("--start-server", action="store_true",
                        help="自动启动 vLLM 服务")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="自动启动服务时的显存利用率")
    parser.add_argument("--prompt-token-count", type=int, default=1,
                        help="固定输入 prompt token 数")
    parser.add_argument("--idle-baseline-sec", type=float, default=2.0,
                        help="空闲基线采样时长（秒）")
    parser.add_argument("--inter-batch-sec", type=float, default=0.8,
                        help="batch 间隔（秒）")
    parser.add_argument("--time-padding-ms", type=float, default=20.0,
                        help="峰值统计的时间补偿（毫秒）")
    parser.add_argument("--max-retries", type=int, default=6,
                        help="单次推理失败后的最大重试次数")
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0,
                        help="失败重试的线性退避基准秒数")
    parser.add_argument("--enable-chunked-prefill", action="store_true", default=DEFAULT_ENABLE_CHUNKED_PREFILL,
                        help="显式启用 vLLM chunked prefill")
    parser.add_argument("--max-num-batched-tokens", type=int, default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
                        help="显式固定 vLLM max_num_batched_tokens")
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS,
                        help="显式固定 vLLM max_num_seqs")
    parser.add_argument("--queue-seed", type=int, default=DEFAULT_QUEUE_SEED,
                        help="实验队列随机种子")
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_SAMPLING_SEED,
                        help="请求采样种子")
    parser.add_argument("--sudo-password", type=str, default=None,
                        help="sudo 密码（用于自动设置功率）")
    parser.add_argument("--skip-set-power", action="store_true",
                        help="跳过自动设置功率")

    args = parser.parse_args()

    run_decode_experiment(
        batch_sizes=parse_int_list(args.batch_sizes),
        output_lengths=parse_int_list(args.output_lengths),
        repeats=args.repeats,
        power_cap=args.power,
        output_dir=args.output_dir,
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        tokenizer_path=args.tokenizer_path,
        base_url=args.base_url,
        api_key=args.api_key,
        start_server=args.start_server,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prompt_token_count=args.prompt_token_count,
        idle_baseline_sec=args.idle_baseline_sec,
        inter_batch_sec=args.inter_batch_sec,
        time_padding_ms=args.time_padding_ms,
        max_retries=args.max_retries,
        retry_backoff_sec=args.retry_backoff_sec,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        queue_seed=args.queue_seed,
        sampling_seed=args.sampling_seed,
        sudo_password=args.sudo_password,
        skip_set_power=args.skip_set_power,
    )
