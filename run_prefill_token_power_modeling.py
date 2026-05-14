#!/usr/bin/env python3
"""Run prefill token-power modeling with input length as the only variable."""

import argparse
import csv
import json
import os
import random
import statistics
import time
from typing import Dict, List, Sequence

from tqdm import tqdm


DEFAULT_SHAREGPT_DIR = "./input/ShareGPT"

DEFAULT_INPUT_LENGTHS = [
    # Dense front section: 0-512.
    0,
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    40,
    48,
    64,
    96,
    128,
    160,
    192,
    224,
    256,
    288,
    320,
    352,
    384,
    416,
    448,
    480,
    512,
    # Medium section: 512-3000.
    640,
    768,
    896,
    1024,
    1280,
    1536,
    1792,
    2048,
    2304,
    2560,
    2816,
    3000,
    # Sparse tail: 3000-20000.
    4000,
    5000,
    6000,
    8000,
    10000,
    12000,
    15000,
    18000,
    20000,
]

RAW_FIELDNAMES = [
    "experiment_id",
    "target_input_tokens",
    "actual_input_tokens",
    "repeat_id",
    "prompt_source",
    "start_time",
    "end_time",
    "duration_ms",
    "ttft_ms",
    "measurement_mode",
    "block_request_count",
    "block_success_count",
    "block_error_count",
    "block_warmup_requests",
    "block_min_requests_used",
    "block_max_requests_used",
    "avg_power_w",
    "median_power_w",
    "p95_power_w",
    "peak_power_w",
    "min_power_w",
    "active_avg_power_w",
    "active_median_power_w",
    "active_p95_power_w",
    "active_sample_count",
    "active_sample_fraction",
    "active_power_threshold_w",
    "energy_j",
    "energy_per_request_j",
    "idle_baseline_w",
    "dynamic_power_w",
    "dynamic_energy_j",
    "dynamic_energy_per_request_j",
    "power_sample_count",
    "status",
    "error",
    "first_ttft_ms",
    "min_ttft_ms",
    "max_ttft_ms",
    "prompt_generation_ms",
    "add_unique_prefix",
]

AGG_FIELDNAMES = [
    "experiment_id",
    "target_input_tokens",
    "avg_actual_input_tokens",
    "num_samples",
    "avg_duration_ms",
    "std_duration_ms",
    "avg_ttft_ms",
    "std_ttft_ms",
    "avg_power_w",
    "std_power_w",
    "median_power_w",
    "p95_power_w",
    "peak_power_w",
    "min_power_w",
    "avg_active_power_w",
    "std_active_power_w",
    "active_median_power_w",
    "active_p95_power_w",
    "avg_active_sample_fraction",
    "avg_energy_j",
    "avg_energy_per_request_j",
    "std_energy_j",
    "avg_dynamic_energy_per_request_j",
    "avg_dynamic_power_w",
    "std_dynamic_power_w",
    "avg_dynamic_energy_j",
    "std_dynamic_energy_j",
    "avg_idle_baseline_w",
    "status_ok_count",
    "status_error_count",
]

TIMELINE_FIELDNAMES = [
    "timestamp",
    "relative_time",
    "power_w",
    "memory_gb",
    "temperature_c",
    "graphics_clock_mhz",
    "memory_clock_mhz",
]


def generate_default_input_lengths() -> List[int]:
    """Return the default 0-20000 token schedule with sparse tail after 3000."""
    return list(DEFAULT_INPUT_LENGTHS)


def parse_input_lengths(value: str) -> List[int]:
    """Parse an input length list or the built-in default schedule."""
    normalized = value.strip().lower()
    if normalized in {"default", "0-20000", "0_20000"}:
        return generate_default_input_lengths()
    if normalized == "smoke":
        return [0, 1, 128, 1024]
    lengths = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not lengths:
        raise ValueError("input length list is empty")
    if lengths[0] < 0:
        raise ValueError("input lengths must be non-negative")
    return lengths


def default_repeats_for_length(target_tokens: int) -> int:
    """Return default repeats by input-token zone."""
    if int(target_tokens) <= 512:
        return 5
    if int(target_tokens) <= 3000:
        return 3
    return 2


def default_block_request_bounds(target_tokens: int) -> tuple:
    """Return block min/max request counts by input-token zone."""
    if int(target_tokens) <= 512:
        return 30, 100
    if int(target_tokens) <= 3000:
        return 10, 30
    return 3, 5


def build_experiment_queue(input_lengths: Sequence[int], repeats: int = None) -> List[tuple]:
    """Build `(target_tokens, repeat_id)` requests with optional uniform repeats."""
    queue = []
    for target_tokens in input_lengths:
        repeat_count = int(repeats) if repeats is not None else default_repeats_for_length(int(target_tokens))
        for repeat_id in range(1, repeat_count + 1):
            queue.append((int(target_tokens), repeat_id))
    return queue


def build_prompts_for_length(load_generator,
                             target_tokens: int,
                             count: int,
                             add_unique_prefix: bool) -> tuple:
    """Prepare prompts and token counts outside the measured GPU window."""
    prompts = []
    token_counts = []
    for _ in range(max(0, int(count))):
        if int(target_tokens) == 0:
            prompt = ""
        else:
            prompt = load_generator.generate_prompt_by_token_count(
                int(target_tokens),
                add_unique_prefix=add_unique_prefix,
            )
        prompts.append(prompt)
        token_counts.append(load_generator.count_tokens(prompt))
    return prompts, token_counts


def initialize_csv(file_path: str, fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_rows(file_path: str, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    with open(file_path, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writerows(rows)


def interpolate_power(power_data: Sequence[Dict], timestamp: float) -> float:
    if not power_data:
        return 0.0
    if timestamp <= float(power_data[0]["timestamp"]):
        return float(power_data[0]["power_w"])
    if timestamp >= float(power_data[-1]["timestamp"]):
        return float(power_data[-1]["power_w"])
    for index in range(1, len(power_data)):
        prev_point = power_data[index - 1]
        next_point = power_data[index]
        prev_ts = float(prev_point["timestamp"])
        next_ts = float(next_point["timestamp"])
        if prev_ts <= timestamp <= next_ts:
            span = next_ts - prev_ts
            if span <= 0:
                return float(next_point["power_w"])
            ratio = (timestamp - prev_ts) / span
            return float(prev_point["power_w"]) + ratio * (
                float(next_point["power_w"]) - float(prev_point["power_w"])
            )
    return float(power_data[-1]["power_w"])


def build_power_window_stats(power_data: Sequence[Dict],
                             start_time: float,
                             end_time: float,
                             idle_baseline_w: float) -> Dict[str, float]:
    """Compute power statistics and energy by integrating the sampled timeline."""
    if end_time <= start_time or not power_data:
        return {
            "avg_power_w": 0.0,
            "median_power_w": 0.0,
            "p95_power_w": 0.0,
            "peak_power_w": 0.0,
            "min_power_w": 0.0,
            "active_avg_power_w": 0.0,
            "active_median_power_w": 0.0,
            "active_p95_power_w": 0.0,
            "active_sample_count": 0,
            "active_sample_fraction": 0.0,
            "active_power_threshold_w": 0.0,
            "energy_j": 0.0,
            "energy_per_request_j": 0.0,
            "dynamic_power_w": 0.0,
            "dynamic_energy_j": 0.0,
            "dynamic_energy_per_request_j": 0.0,
            "power_sample_count": 0,
        }

    samples = [
        item for item in power_data
        if start_time <= float(item["timestamp"]) <= end_time
    ]
    points = [(start_time, interpolate_power(power_data, start_time))]
    points.extend((float(item["timestamp"]), float(item["power_w"])) for item in samples)
    points.append((end_time, interpolate_power(power_data, end_time)))
    points = sorted(points, key=lambda item: item[0])

    energy_j = 0.0
    for index in range(1, len(points)):
        dt = max(0.0, points[index][0] - points[index - 1][0])
        avg_power = (points[index][1] + points[index - 1][1]) / 2.0
        energy_j += avg_power * dt

    duration_s = max(1e-9, end_time - start_time)
    window_powers = [point[1] for point in points]
    avg_power_w = energy_j / duration_s
    dynamic_power_w = max(0.0, avg_power_w - idle_baseline_w)
    dynamic_energy_j = max(0.0, energy_j - idle_baseline_w * duration_s)
    active_threshold_w = max(idle_baseline_w + 30.0, idle_baseline_w * 1.25)
    active_powers = [power for power in window_powers if power >= active_threshold_w]
    if not active_powers and window_powers:
        active_powers = window_powers

    return {
        "avg_power_w": avg_power_w,
        "median_power_w": statistics.median(window_powers),
        "p95_power_w": percentile(window_powers, 95.0),
        "peak_power_w": max(window_powers),
        "min_power_w": min(window_powers),
        "active_avg_power_w": statistics.mean(active_powers) if active_powers else 0.0,
        "active_median_power_w": statistics.median(active_powers) if active_powers else 0.0,
        "active_p95_power_w": percentile(active_powers, 95.0),
        "active_sample_count": len(active_powers),
        "active_sample_fraction": len(active_powers) / max(1, len(window_powers)),
        "active_power_threshold_w": active_threshold_w,
        "energy_j": energy_j,
        "energy_per_request_j": 0.0,
        "dynamic_power_w": dynamic_power_w,
        "dynamic_energy_j": dynamic_energy_j,
        "dynamic_energy_per_request_j": 0.0,
        "power_sample_count": len(samples),
    }


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(pct) / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate_rows(experiment_id: str, rows: Sequence[Dict]) -> List[Dict]:
    groups: Dict[int, List[Dict]] = {}
    for row in rows:
        groups.setdefault(int(row["target_input_tokens"]), []).append(row)

    def mean(values: List[float]) -> float:
        return statistics.mean(values) if values else 0.0

    def stdev(values: List[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    aggregated = []
    for target_tokens in sorted(groups):
        group = groups[target_tokens]
        ok_rows = [row for row in group if row.get("status") == "ok"]
        source_rows = ok_rows if ok_rows else group

        def values(key: str) -> List[float]:
            vals = []
            for row in source_rows:
                try:
                    vals.append(float(row.get(key, 0.0)))
                except (TypeError, ValueError):
                    pass
            return vals

        aggregated.append({
            "experiment_id": experiment_id,
            "target_input_tokens": target_tokens,
            "avg_actual_input_tokens": mean(values("actual_input_tokens")),
            "num_samples": len(source_rows),
            "avg_duration_ms": mean(values("duration_ms")),
            "std_duration_ms": stdev(values("duration_ms")),
            "avg_ttft_ms": mean(values("ttft_ms")),
            "std_ttft_ms": stdev(values("ttft_ms")),
            "avg_power_w": mean(values("avg_power_w")),
            "std_power_w": stdev(values("avg_power_w")),
            "median_power_w": mean(values("median_power_w")),
            "p95_power_w": mean(values("p95_power_w")),
            "peak_power_w": max(values("peak_power_w") or [0.0]),
            "min_power_w": min(values("min_power_w") or [0.0]),
            "avg_active_power_w": mean(values("active_avg_power_w")),
            "std_active_power_w": stdev(values("active_avg_power_w")),
            "active_median_power_w": mean(values("active_median_power_w")),
            "active_p95_power_w": mean(values("active_p95_power_w")),
            "avg_active_sample_fraction": mean(values("active_sample_fraction")),
            "avg_energy_j": mean(values("energy_j")),
            "avg_energy_per_request_j": mean(values("energy_per_request_j")),
            "std_energy_j": stdev(values("energy_j")),
            "avg_dynamic_power_w": mean(values("dynamic_power_w")),
            "std_dynamic_power_w": stdev(values("dynamic_power_w")),
            "avg_dynamic_energy_j": mean(values("dynamic_energy_j")),
            "avg_dynamic_energy_per_request_j": mean(values("dynamic_energy_per_request_j")),
            "std_dynamic_energy_j": stdev(values("dynamic_energy_j")),
            "avg_idle_baseline_w": mean(values("idle_baseline_w")),
            "status_ok_count": len(ok_rows),
            "status_error_count": len(group) - len(ok_rows),
        })
    return aggregated


def save_power_timeline(file_path: str, power_data: Sequence[Dict], experiment_start_time: float) -> None:
    initialize_csv(file_path, TIMELINE_FIELDNAMES)
    rows = []
    for item in power_data:
        rows.append({
            "timestamp": item.get("timestamp", 0.0),
            "relative_time": float(item.get("timestamp", 0.0)) - experiment_start_time,
            "power_w": item.get("power_w", 0.0),
            "memory_gb": item.get("memory_gb", 0.0),
            "temperature_c": item.get("temperature_c", 0),
            "graphics_clock_mhz": item.get("graphics_clock_mhz", 0.0),
            "memory_clock_mhz": item.get("memory_clock_mhz", 0.0),
        })
    append_csv_rows(file_path, TIMELINE_FIELDNAMES, rows)


def run_prefill_token_power_modeling(input_lengths: Sequence[int],
                                     repeats: int = None,
                                     power_cap: int = 350,
                                     output_dir: str = "experiment_results/prefill_token_power_modeling/default",
                                     model_path: str = None,
                                     tokenizer_path: str = "./Qwen2.5-7B-Instruct-AWQ",
                                     sharegpt_dir: str = DEFAULT_SHAREGPT_DIR,
                                     device_index: int = 0,
                                     sample_interval: float = 0.02,
                                     idle_baseline_sec: float = 2.0,
                                     inter_request_sleep_sec: float = 0.2,
                                     warmup_repeats: int = 5,
                                     power_settle_sec: float = 20.0,
                                     measurement_mode: str = "block",
                                     block_target_window_sec: float = 2.0,
                                     block_min_requests: int = None,
                                     block_max_requests: int = None,
                                     block_warmup_requests: int = 1,
                                     block_cooldown_sec: float = 1.0,
                                     add_unique_prefix: bool = False,
                                     skip_set_power: bool = False,
                                     sudo_password: str = None,
                                     seed: int = 2026,
                                     shuffle: bool = True) -> Dict[str, str]:
    """Run the token-length-only prefill power modeling experiment."""
    from llm_inference import LLMInferencer
    from load_generator import LoadGenerator
    from monitor import PowerMonitor
    from power_control import SudoKeepAlive, get_power_cap, set_power_cap

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    measurement_mode = measurement_mode.strip().lower()
    if measurement_mode not in {"request", "block"}:
        raise ValueError("measurement_mode must be 'request' or 'block'")

    keepalive = None
    if not skip_set_power:
        keepalive = SudoKeepAlive()
        keepalive.start(sudo_password=sudo_password)
        print(f"设置功率上限为 {power_cap}W")
        if not set_power_cap(power_cap, device_index=device_index, sudo_password=sudo_password):
            raise RuntimeError("failed to set GPU power cap")
    else:
        print("跳过功率设置，使用当前系统功率上限")

    actual_power_cap = get_power_cap(device_index=device_index)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"功率设置/监控 GPU index: {device_index}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices or '<unset>'}")
    if cuda_visible_devices and cuda_visible_devices != str(device_index):
        print("警告: CUDA_VISIBLE_DEVICES 与 --device-index 不一致，请确认 vLLM 和 nvidia-smi 指向同一物理 GPU。")
    print(f"当前功率上限: {actual_power_cap}W")
    if not skip_set_power and power_settle_sec > 0:
        print(f"等待功率稳定 {power_settle_sec:.1f} 秒...")
        time.sleep(power_settle_sec)

    inferencer = LLMInferencer(model_name=model_path) if model_path else LLMInferencer()
    load_generator = LoadGenerator(sharegpt_dir=sharegpt_dir, tokenizer_name=tokenizer_path)

    print("预热 GPU...")
    if warmup_repeats > 0:
        warmup_prompt = load_generator.generate_prompt_by_token_count(64, add_unique_prefix=True)
        for _ in range(max(0, warmup_repeats)):
            inferencer.infer_prefill_only([warmup_prompt], max_tokens=1)
    time.sleep(1.0)

    experiment_id = f"prefill_token_power_modeling_{int(time.time())}"
    raw_path = os.path.join(output_dir, f"{experiment_id}_raw.csv")
    agg_path = os.path.join(output_dir, f"{experiment_id}_aggregated.csv")
    timeline_path = os.path.join(output_dir, f"{experiment_id}_power_timeline.csv")
    metadata_path = os.path.join(output_dir, f"{experiment_id}_metadata.json")
    initialize_csv(raw_path, RAW_FIELDNAMES)

    queue = build_experiment_queue(input_lengths, repeats=repeats)
    if shuffle:
        random.shuffle(queue)

    monitor = PowerMonitor(device_index=device_index, sample_interval=sample_interval)
    monitor.start()
    print(f"功率采样后端: {monitor._backend}")
    if monitor._backend != "pynvml":
        print("警告: 当前使用 nvidia-smi 轮询，短时 prefill 功率可能被平滑。")
    print(f"采集空闲基线 {idle_baseline_sec:.1f}s...")
    time.sleep(max(0.0, idle_baseline_sec))
    experiment_start_time = time.time()

    raw_rows: List[Dict] = []
    try:
        baseline_samples = [
            float(item["power_w"])
            for item in monitor.power_data
            if float(item["timestamp"]) < experiment_start_time
        ]
        idle_baseline_w = statistics.mean(baseline_samples) if baseline_samples else 0.0

        for target_tokens, repeat_id in tqdm(queue, desc="Prefill token-power modeling"):
            default_min_requests, default_max_requests = default_block_request_bounds(target_tokens)
            block_min_requests_used = (
                int(block_min_requests)
                if block_min_requests is not None
                else default_min_requests
            )
            block_max_requests_used = (
                int(block_max_requests)
                if block_max_requests is not None
                else default_max_requests
            )
            block_max_requests_used = max(block_min_requests_used, block_max_requests_used)
            if measurement_mode == "request":
                block_min_requests_used = 1
                block_max_requests_used = 1

            prompt_source = "minimal" if target_tokens == 0 else "generated"
            planned_measured_requests = (
                block_min_requests_used
                if measurement_mode == "request" or float(block_target_window_sec) <= 0
                else block_max_requests_used
            )
            prompt_generation_start = time.time()
            warmup_prompts, _ = build_prompts_for_length(
                load_generator,
                target_tokens,
                int(block_warmup_requests) if measurement_mode == "block" else 0,
                add_unique_prefix=add_unique_prefix,
            )
            measured_prompts, measured_token_counts = build_prompts_for_length(
                load_generator,
                target_tokens,
                planned_measured_requests,
                add_unique_prefix=add_unique_prefix,
            )
            prompt_generation_ms = (time.time() - prompt_generation_start) * 1000.0

            if measurement_mode == "block" and block_cooldown_sec > 0:
                time.sleep(block_cooldown_sec)
            if measurement_mode == "block" and warmup_prompts:
                for warmup_block_prompt in warmup_prompts:
                    try:
                        inferencer.infer_prefill_only([warmup_block_prompt], max_tokens=1)
                    except Exception:
                        pass

            actual_input_token_values = []
            start_time = time.time()
            status = "ok"
            errors = []
            ttft_values = []
            block_request_count = 0
            block_success_count = 0
            block_error_count = 0
            while block_request_count < len(measured_prompts):
                request_prompt = measured_prompts[block_request_count]
                if block_request_count < len(measured_token_counts):
                    actual_input_token_values.append(measured_token_counts[block_request_count])
                block_request_count += 1
                try:
                    result = inferencer.infer_prefill_only([request_prompt], max_tokens=1)[0]
                    ttft_values.append(float(result.get("ttft", 0.0)))
                    block_success_count += 1
                except Exception as exc:
                    errors.append(str(exc))
                    block_error_count += 1
                end_time = time.time()
                if measurement_mode == "request":
                    break
                if block_request_count >= block_max_requests_used:
                    break
                if block_request_count >= block_min_requests_used and (
                    end_time - start_time
                ) >= float(block_target_window_sec):
                    break
            if block_success_count == 0:
                status = "error"
            elif block_error_count > 0:
                status = "partial_error"
            error = " | ".join(errors[:3])
            ttft_ms = statistics.mean(ttft_values) if ttft_values else 0.0
            end_time = time.time()

            power_stats = build_power_window_stats(
                monitor.power_data,
                start_time,
                end_time,
                idle_baseline_w,
            )
            row = {
                "experiment_id": experiment_id,
                "target_input_tokens": target_tokens,
                "actual_input_tokens": int(round(statistics.mean(actual_input_token_values))) if actual_input_token_values else 0,
                "repeat_id": repeat_id,
                "prompt_source": prompt_source,
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": (end_time - start_time) * 1000.0,
                "ttft_ms": ttft_ms if status == "ok" else 0.0,
                "measurement_mode": measurement_mode,
                "block_request_count": block_request_count,
                "block_success_count": block_success_count,
                "block_error_count": block_error_count,
                "block_warmup_requests": int(block_warmup_requests),
                "block_min_requests_used": block_min_requests_used,
                "block_max_requests_used": block_max_requests_used,
                "idle_baseline_w": idle_baseline_w,
                "status": status,
                "error": error,
                **power_stats,
            }
            request_count = max(1, block_success_count)
            row["energy_per_request_j"] = row["energy_j"] / request_count
            row["dynamic_energy_per_request_j"] = row["dynamic_energy_j"] / request_count
            row["first_ttft_ms"] = ttft_values[0] if ttft_values else 0.0
            row["min_ttft_ms"] = min(ttft_values) if ttft_values else 0.0
            row["max_ttft_ms"] = max(ttft_values) if ttft_values else 0.0
            row["prompt_generation_ms"] = prompt_generation_ms
            row["add_unique_prefix"] = bool(add_unique_prefix)
            raw_rows.append(row)
            append_csv_rows(raw_path, RAW_FIELDNAMES, [row])
            time.sleep(max(0.0, inter_request_sleep_sec))
    finally:
        power_data = monitor.stop()
        if keepalive is not None:
            keepalive.stop()

    save_power_timeline(timeline_path, power_data, experiment_start_time)
    aggregated = aggregate_rows(experiment_id, raw_rows)
    initialize_csv(agg_path, AGG_FIELDNAMES)
    append_csv_rows(agg_path, AGG_FIELDNAMES, aggregated)

    metadata = {
        "experiment_id": experiment_id,
        "power_cap_w": power_cap,
        "actual_power_cap_w": actual_power_cap,
        "device_index": int(device_index),
        "cuda_visible_devices": cuda_visible_devices,
        "input_lengths": list(map(int, input_lengths)),
        "repeats": repeats,
        "default_repeats_by_zone": {
            "0_512": 5,
            "513_3000": 3,
            "3001_20000": 2,
        },
        "total_requests": len(queue),
        "max_tokens": 1,
        "sample_interval": sample_interval,
        "idle_baseline_sec": idle_baseline_sec,
        "inter_request_sleep_sec": inter_request_sleep_sec,
        "power_settle_sec": power_settle_sec,
        "measurement_mode": measurement_mode,
        "block_target_window_sec": block_target_window_sec,
        "block_min_requests": block_min_requests,
        "block_max_requests": block_max_requests,
        "block_default_request_bounds_by_zone": {
            "0_512": [30, 100],
            "513_3000": [10, 30],
            "3001_20000": [3, 5],
        },
        "block_warmup_requests": block_warmup_requests,
        "block_cooldown_sec": block_cooldown_sec,
        "add_unique_prefix": bool(add_unique_prefix),
        "seed": seed,
        "shuffle": shuffle,
        "monitor_backend": getattr(monitor, "_backend", "unknown"),
        "raw_path": raw_path,
        "aggregated_path": agg_path,
        "timeline_path": timeline_path,
    }
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=False)

    print(f"原始数据: {raw_path}")
    print(f"聚合数据: {agg_path}")
    print(f"功率时间线: {timeline_path}")
    print(f"元数据: {metadata_path}")
    return {
        "raw_path": raw_path,
        "aggregated_path": agg_path,
        "timeline_path": timeline_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prefill token-power modeling.")
    parser.add_argument("--input-lengths", default="default",
                        help="Comma-separated token lengths, 'default' for dense 0-512 with <=32-token steps, medium 512-3000, sparse 3000-20000, or 'smoke'.")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Uniform repeats per token length. Default uses zones: 0-512=5, 513-3000=3, >3000=2.")
    parser.add_argument("--power", type=int, default=350)
    parser.add_argument("--output-dir", default="experiment_results/prefill_token_power_modeling/default")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tokenizer-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--sharegpt-dir", default=DEFAULT_SHAREGPT_DIR)
    parser.add_argument("--device-index", type=int, default=0,
                        help="Physical GPU index for nvidia-smi power control and monitoring.")
    parser.add_argument("--sample-interval", type=float, default=0.02)
    parser.add_argument("--idle-baseline-sec", type=float, default=2.0)
    parser.add_argument("--inter-request-sleep-sec", type=float, default=0.2)
    parser.add_argument("--warmup-repeats", type=int, default=5)
    parser.add_argument("--power-settle-sec", type=float, default=20.0,
                        help="Seconds to wait after setting power cap before loading the model.")
    parser.add_argument("--measurement-mode", choices=["request", "block"], default="block",
                        help="Use per-request windows or continuous block windows per token length.")
    parser.add_argument("--block-target-window-sec", type=float, default=2.0,
                        help="Minimum measured block duration in block mode.")
    parser.add_argument("--block-min-requests", type=int, default=None,
                        help="Minimum requests per measured block. Default uses zones: 0-512=30, 513-3000=10, >3000=3.")
    parser.add_argument("--block-max-requests", type=int, default=None,
                        help="Maximum requests per measured block. Default uses zones: 0-512=100, 513-3000=30, >3000=5.")
    parser.add_argument("--block-warmup-requests", type=int, default=1,
                        help="Same-token warmup requests before each measured block.")
    parser.add_argument("--block-cooldown-sec", type=float, default=1.0,
                        help="Cooldown before each measured block.")
    parser.add_argument("--add-unique-prefix", action="store_true",
                        help="Add a random prefix to each prompt. Off by default because this script disables prefix caching.")
    parser.add_argument("--skip-set-power", action="store_true")
    parser.add_argument("--sudo-password", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    input_lengths = parse_input_lengths(args.input_lengths)
    queue = build_experiment_queue(input_lengths, repeats=args.repeats)
    print(f"输入 token 长度: {input_lengths}")
    if args.repeats is None:
        print("重复次数: 默认分段 0-512=5, 513-3000=3, >3000=2")
    else:
        print(f"重复次数: 每个 token 点 {args.repeats}")
    print(f"总请求数: {len(queue)}")

    run_prefill_token_power_modeling(
        input_lengths=input_lengths,
        repeats=args.repeats,
        power_cap=args.power,
        output_dir=args.output_dir,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        sharegpt_dir=args.sharegpt_dir,
        device_index=args.device_index,
        sample_interval=args.sample_interval,
        idle_baseline_sec=args.idle_baseline_sec,
        inter_request_sleep_sec=args.inter_request_sleep_sec,
        warmup_repeats=args.warmup_repeats,
        power_settle_sec=args.power_settle_sec,
        measurement_mode=args.measurement_mode,
        block_target_window_sec=args.block_target_window_sec,
        block_min_requests=args.block_min_requests,
        block_max_requests=args.block_max_requests,
        block_warmup_requests=args.block_warmup_requests,
        block_cooldown_sec=args.block_cooldown_sec,
        add_unique_prefix=args.add_unique_prefix,
        skip_set_power=args.skip_set_power,
        sudo_password=args.sudo_password,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )


if __name__ == "__main__":
    main()
