#!/usr/bin/env python3
"""
并发 prefill-only 功率评估脚本。

一次性提交 q 条请求，每条只生成 1 个 token，用于近似测量并发 prefill 的
TTFT / Power / Energy 表现。
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
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from llm_inference import LLMInferencer
from load_generator import LoadGenerator
from monitor import PowerMonitor
from power_control import SudoKeepAlive, get_power_cap, set_power_cap


QUERY_GROUPS = [
    {"query_count": 8, "target_input_tokens": 225},
    {"query_count": 16, "target_input_tokens": 504},
    {"query_count": 32, "target_input_tokens": 1581},
    {"query_count": 64, "target_input_tokens": 2175},
    {"query_count": 103, "target_input_tokens": 6053},
    {"query_count": 112, "target_input_tokens": 11106},
    {"query_count": 119, "target_input_tokens": 20295},
]

PREFILL_TOKEN_POWER_FIT = {
    "breakpoint": 3000.0,
    "front_linear": {
        "slope": 0.03576230118185584,
        "intercept": 184.63676032395085,
    },
    "tail_log": {
        "scale": 9.566932909648392,
        "intercept": 278.3867873741286,
    },
    "min_power_w": 150,
    "max_power_w": 350,
}

MANUAL_PREFILL_BUCKETS = [
    {"min_query_count": 0, "max_query_count": 16, "power": 200},
    {"min_query_count": 17, "max_query_count": 64, "power": 220},
    {"min_query_count": 65, "max_query_count": 10_000, "power": 260},
]

STRATEGIES = [
    {"name": "baseline_350w", "type": "fixed", "power": 350},
    {"name": "prefill_token_fit", "type": "token_fit", "power_offset_w": 0},
    {"name": "prefill_token_fit_plus25w", "type": "token_fit", "power_offset_w": 25},
    {"name": "prefill_manual_buckets", "type": "manual_buckets", "buckets": MANUAL_PREFILL_BUCKETS},
]

RAW_FIELDNAMES = [
    "full_repeat",
    "strategy",
    "query_count",
    "target_input_tokens",
    "actual_input_tokens",
    "batch_repeat",
    "power_limit",
    "actual_power_limit",
    "avg_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "avg_e2e_ms",
    "avg_power_w",
    "total_energy_j",
    "peak_power_w",
    "num_requests",
]

AGG_FIELDNAMES = [
    "full_repeat",
    "strategy",
    "query_count",
    "target_input_tokens",
    "actual_input_tokens",
    "power_limit",
    "actual_power_limit",
    "num_samples",
    "avg_ttft_ms",
    "avg_e2e_ms",
    "avg_energy_j",
    "avg_power_w",
]


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * ratio
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def initialize_csv_file(file_path: str, fieldnames: List[str]):
    with open(file_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_rows(file_path: str, fieldnames: List[str], rows: List[Dict]):
    if not rows:
        return
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    with open(file_path, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def write_json_file(file_path: str, payload: Dict):
    with open(file_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def write_progress_file(file_path: str,
                        experiment_id: str,
                        total_blocks: int,
                        completed_blocks: int,
                        status: str,
                        current_block: Optional[Dict] = None,
                        started_at: Optional[float] = None,
                        last_completed_block: Optional[Dict] = None,
                        error: Optional[str] = None):
    payload = {
        "experiment_id": experiment_id,
        "status": status,
        "total_blocks": total_blocks,
        "completed_blocks": completed_blocks,
        "current_block": current_block,
        "last_completed_block": last_completed_block,
        "started_at": started_at,
        "updated_at": time.time(),
    }
    if error is not None:
        payload["error"] = error
    write_json_file(file_path, payload)


def build_experiment_blocks(strategies: Sequence[Dict],
                            query_groups: Sequence[Dict],
                            full_repeats: int) -> List[Dict]:
    blocks = []
    for full_repeat in range(1, full_repeats + 1):
        for strategy in strategies:
            for query_group in query_groups:
                blocks.append({
                    "full_repeat": full_repeat,
                    "strategy_name": strategy["name"],
                    "query_count": int(query_group["query_count"]),
                    "target_input_tokens": int(query_group["target_input_tokens"]),
                })
    return blocks


def validate_actual_power_limit(expected_power: int, actual_power: float, tolerance_w: float = 5.0):
    if abs(float(actual_power) - float(expected_power)) > tolerance_w:
        raise RuntimeError(
            f"Power limit mismatch: expected {expected_power}W, got {actual_power:.1f}W"
        )


def wait_for_power_limit(expected_power: int,
                         device_index: int = 0,
                         timeout_sec: float = 3.0,
                         poll_interval_sec: float = 0.1,
                         tolerance_w: float = 5.0) -> float:
    deadline = time.time() + timeout_sec
    last_power = get_power_cap(device_index=device_index)
    while time.time() <= deadline:
        last_power = get_power_cap(device_index=device_index)
        if abs(float(last_power) - float(expected_power)) <= tolerance_w:
            return float(last_power)
        time.sleep(poll_interval_sec)
    raise RuntimeError(
        f"Power limit mismatch: expected {expected_power}W, got {last_power:.1f}W"
    )


def clamp_power(power_w: float,
                min_power_w: int = PREFILL_TOKEN_POWER_FIT["min_power_w"],
                max_power_w: int = PREFILL_TOKEN_POWER_FIT["max_power_w"]) -> int:
    return int(round(max(float(min_power_w), min(float(max_power_w), float(power_w)))))


def evaluate_prefill_token_power_fit(total_input_tokens: int) -> float:
    """根据 token-power 建模结果，把输入 token 数映射为 prefill 推荐功率。"""
    tokens = max(1.0, float(total_input_tokens))
    breakpoint = float(PREFILL_TOKEN_POWER_FIT["breakpoint"])
    if tokens <= breakpoint:
        front = PREFILL_TOKEN_POWER_FIT["front_linear"]
        return float(front["slope"]) * tokens + float(front["intercept"])
    tail = PREFILL_TOKEN_POWER_FIT["tail_log"]
    return float(tail["scale"]) * math.log(tokens / breakpoint) + float(tail["intercept"])


def recommend_manual_bucket_power(query_count: int, buckets: Sequence[Dict]) -> int:
    for bucket in buckets:
        if int(bucket["min_query_count"]) <= int(query_count) <= int(bucket["max_query_count"]):
            return int(bucket["power"])
    return int(buckets[-1]["power"])


def recommend_prefill_power(strategy: Dict,
                            query_count: int,
                            total_input_tokens: int) -> int:
    """统一入口：根据策略类型返回本 batch 的 prefill power cap。"""
    strategy_type = strategy.get("type", "fixed")
    if strategy_type == "fixed":
        return int(strategy["power"])
    if strategy_type == "token_fit":
        power = evaluate_prefill_token_power_fit(total_input_tokens)
        power += float(strategy.get("power_offset_w", 0))
        return clamp_power(power)
    if strategy_type == "manual_buckets":
        return recommend_manual_bucket_power(query_count, strategy["buckets"])
    raise ValueError(f"Unknown prefill strategy type: {strategy_type}")


def select_subset_prompts(load_generator: LoadGenerator,
                          num_prompts: int,
                          target_tokens: int,
                          rng: random.Random) -> List[Dict]:
    avg_per_prompt = max(1, target_tokens // num_prompts)
    prompts = []
    for _ in range(num_prompts):
        variation = rng.randint(-max(1, avg_per_prompt // 10), max(1, avg_per_prompt // 10))
        target_length = max(1, avg_per_prompt + variation)
        prompt = load_generator.generate_prompt_by_token_count(
            target_length,
            prefer_sharegpt=True,
            add_unique_prefix=True,
        )
        prompt_tokens = load_generator.count_tokens(prompt)
        prompts.append({
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
        })
    return prompts


def build_query_group_prompt_sets(load_generator: LoadGenerator,
                                  query_groups: Sequence[Dict],
                                  repeats_per_batch: int,
                                  warmup_batches: int,
                                  monitor_warmup_batches: int,
                                  queue_seed: int,
                                  full_repeat: int) -> Dict[int, List[List[Dict]]]:
    """提前构造所有 batch，保证不同策略在同一 full repeat 中面对同一组请求。"""
    total_batches = warmup_batches + monitor_warmup_batches + repeats_per_batch
    rng = random.Random(queue_seed + full_repeat)
    prompt_sets = {}
    for query_group in query_groups:
        prompt_batches = []
        for _ in range(total_batches):
            batch = select_subset_prompts(
                load_generator,
                num_prompts=int(query_group["query_count"]),
                target_tokens=int(query_group["target_input_tokens"]),
                rng=rng,
            )
            rng.shuffle(batch)
            prompt_batches.append(batch)
        prompt_sets[int(query_group["query_count"])] = prompt_batches
    return prompt_sets


def build_service_extra_body(sampling_seed: int) -> Dict:
    return {
        "top_p": 1.0,
        "seed": sampling_seed,
    }


def build_power_window_stats(start_time: float, end_time: float, power_data: List[Dict]) -> Dict[str, float]:
    if not power_data:
        return {"avg_power_w": 0.0, "total_energy_j": 0.0, "peak_power_w": 0.0}

    samples = [point for point in power_data if start_time <= point["timestamp"] <= end_time]
    if not samples:
        return {"avg_power_w": 0.0, "total_energy_j": 0.0, "peak_power_w": 0.0}

    duration = max(0.0, end_time - start_time)
    avg_power = statistics.mean(point["power_w"] for point in samples)
    return {
        "avg_power_w": avg_power,
        "total_energy_j": avg_power * duration,
        "peak_power_w": max(point["power_w"] for point in samples),
    }


def summarize_batch_metrics(results: List[Dict]) -> Dict[str, float]:
    ttfts = [float(item["ttft"]) for item in results]
    e2es = [float(item["e2e"]) for item in results]
    return {
        "num_requests": len(results),
        "avg_ttft_ms": statistics.mean(ttfts) if ttfts else 0.0,
        "p50_ttft_ms": percentile(ttfts, 0.50),
        "p95_ttft_ms": percentile(ttfts, 0.95),
        "p99_ttft_ms": percentile(ttfts, 0.99),
        "avg_e2e_ms": statistics.mean(e2es) if e2es else 0.0,
    }


def aggregate_raw_rows(raw_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (
            row["full_repeat"],
            row["strategy"],
            row["query_count"],
            row["target_input_tokens"],
            row["power_limit"],
            row["actual_power_limit"],
        )
        grouped[key].append(row)

    aggregated = []
    for key, rows in sorted(grouped.items()):
        aggregated.append({
            "full_repeat": key[0],
            "strategy": key[1],
            "query_count": key[2],
            "target_input_tokens": key[3],
            "actual_input_tokens": statistics.mean(float(item["actual_input_tokens"]) for item in rows),
            "power_limit": key[4],
            "actual_power_limit": key[5],
            "num_samples": len(rows),
            "avg_ttft_ms": statistics.mean(float(item["avg_ttft_ms"]) for item in rows),
            "avg_e2e_ms": statistics.mean(float(item["avg_e2e_ms"]) for item in rows),
            "avg_energy_j": statistics.mean(float(item["total_energy_j"]) for item in rows),
            "avg_power_w": statistics.mean(float(item["avg_power_w"]) for item in rows),
        })
    return aggregated


def run_prefill_concurrent_evaluation(output_dir: str,
                                      model_path: str,
                                      served_model_name: str,
                                      tokenizer_path: str,
                                      sharegpt_dir: str,
                                      base_url: str,
                                      repeats_per_batch: int,
                                      full_repeats: int,
                                      warmup_batches: int,
                                      monitor_warmup_batches: int,
                                      inter_batch_sec: float,
                                      queue_seed: int,
                                      sampling_seed: int,
                                      device_index: int,
                                      sudo_password: Optional[str],
                                      skip_set_power: bool,
                                      strategy_names: Optional[Sequence[str]] = None):
    """运行 prefill-only 策略评估。

    该实验将 max_tokens 固定为 1，尽量隔离 prefill 阶段，重点观察 TTFT 和能耗。
    """
    os.makedirs(output_dir, exist_ok=True)
    experiment_id = f"prefill_concurrent_eval_{int(time.time())}"
    raw_path = os.path.join(output_dir, f"{experiment_id}_raw.csv")
    agg_path = os.path.join(output_dir, f"{experiment_id}_aggregated.csv")
    metadata_path = os.path.join(output_dir, f"{experiment_id}_metadata.json")
    progress_path = os.path.join(output_dir, f"{experiment_id}_progress.json")

    initialize_csv_file(raw_path, RAW_FIELDNAMES)
    initialize_csv_file(agg_path, AGG_FIELDNAMES)

    strategies = STRATEGIES
    if strategy_names:
        selected_names = {item.strip() for item in strategy_names if item and item.strip()}
        strategies = [item for item in STRATEGIES if item["name"] in selected_names]
        missing = sorted(selected_names - {item["name"] for item in strategies})
        if missing:
            raise ValueError(f"Unknown strategy names: {', '.join(missing)}")

    inferencer = LLMInferencer(
        model_name=model_path,
        use_service=True,
        base_url=base_url,
        served_model_name=served_model_name,
        service_request_mode="completion",
    )
    load_generator = LoadGenerator(sharegpt_dir=sharegpt_dir, tokenizer_name=tokenizer_path)

    metadata = {
        "experiment_id": experiment_id,
        "strategies": [item["name"] for item in strategies],
        "query_groups": QUERY_GROUPS,
        "repeats_per_batch": repeats_per_batch,
        "full_repeats": full_repeats,
        "warmup_batches": warmup_batches,
        "monitor_warmup_batches": monitor_warmup_batches,
        "queue_seed": queue_seed,
        "sampling_seed": sampling_seed,
        "device_index": device_index,
        "base_url": base_url,
        "model_path": model_path,
        "served_model_name": served_model_name,
        "tokenizer_path": tokenizer_path,
        "sharegpt_dir": sharegpt_dir,
        "skip_set_power": skip_set_power,
        "started_at": time.time(),
    }
    write_json_file(metadata_path, metadata)

    total_blocks = len(strategies) * len(QUERY_GROUPS) * full_repeats
    completed_blocks = 0
    raw_rows: List[Dict] = []

    keep_alive = None
    if not skip_set_power:
        keep_alive = SudoKeepAlive(interval_sec=60.0)
        if not keep_alive.start(sudo_password=sudo_password):
            raise RuntimeError("Failed to initialize sudo keepalive")

    try:
        write_progress_file(
            progress_path,
            experiment_id=experiment_id,
            total_blocks=total_blocks,
            completed_blocks=completed_blocks,
            status="running",
            started_at=metadata["started_at"],
        )

        warmup_prompt = load_generator.generate_prompt_by_token_count(32, prefer_sharegpt=True, add_unique_prefix=True)
        inferencer.infer_concurrent(
            [warmup_prompt] * 4,
            max_tokens=1,
            temperature=0.0,
            extra_body=build_service_extra_body(sampling_seed),
        )

        for full_repeat in range(1, full_repeats + 1):
            prompt_sets = build_query_group_prompt_sets(
                load_generator=load_generator,
                query_groups=QUERY_GROUPS,
                repeats_per_batch=repeats_per_batch,
                warmup_batches=warmup_batches,
                monitor_warmup_batches=monitor_warmup_batches,
                queue_seed=queue_seed,
                full_repeat=full_repeat,
            )
            strategy_order = list(strategies[full_repeat - 1:]) + list(strategies[:full_repeat - 1])

            for strategy in strategy_order:
                for query_group in QUERY_GROUPS:
                    query_count = int(query_group["query_count"])
                    target_input_tokens = int(query_group["target_input_tokens"])
                    power_limit = recommend_prefill_power(
                        strategy,
                        query_count=query_count,
                        total_input_tokens=target_input_tokens,
                    )
                    if not skip_set_power:
                        if not set_power_cap(power_limit, device_index=device_index, sudo_password=sudo_password):
                            raise RuntimeError(f"Failed to set power limit {power_limit}W")
                        actual_power_limit = wait_for_power_limit(power_limit, device_index=device_index)
                    else:
                        actual_power_limit = get_power_cap(device_index=device_index)

                    current_block = {
                        "full_repeat": full_repeat,
                        "strategy_name": strategy["name"],
                        "query_count": query_count,
                        "target_input_tokens": target_input_tokens,
                        "power_limit": power_limit,
                    }
                    write_progress_file(
                        progress_path,
                        experiment_id=experiment_id,
                        total_blocks=total_blocks,
                        completed_blocks=completed_blocks,
                        status="running",
                        current_block=current_block,
                        started_at=metadata["started_at"],
                    )

                    batches = prompt_sets[query_count]
                    warmup_slice = batches[:warmup_batches]
                    monitor_warmup_slice = batches[warmup_batches:warmup_batches + monitor_warmup_batches]
                    measurement_slice = batches[warmup_batches + monitor_warmup_batches:]
                    # warmup 用于预热服务；monitor_warmup 用于让功率采样进入稳定状态，不计入统计。
                    extra_body = build_service_extra_body(sampling_seed)

                    for warmup_batch in warmup_slice:
                        inferencer.infer_concurrent(
                            [item["prompt"] for item in warmup_batch],
                            max_tokens=1,
                            temperature=0.0,
                            extra_body=extra_body,
                        )
                        time.sleep(inter_batch_sec)

                    monitor = PowerMonitor(device_index=device_index, sample_interval=0.02)
                    monitor.start()
                    for warmup_batch in monitor_warmup_slice:
                        inferencer.infer_concurrent(
                            [item["prompt"] for item in warmup_batch],
                            max_tokens=1,
                            temperature=0.0,
                            extra_body=extra_body,
                        )

                    block_rows = []
                    for batch_repeat, batch_prompts in enumerate(
                        tqdm(measurement_slice, desc=f"{strategy['name']} q={query_count}", leave=False),
                        start=1,
                    ):
                        wall_start = time.time()
                        results = inferencer.infer_concurrent(
                            [item["prompt"] for item in batch_prompts],
                            max_tokens=1,
                            temperature=0.0,
                            extra_body=extra_body,
                        )
                        wall_end = time.time()
                        batch_metrics = summarize_batch_metrics(results)
                        block_rows.append({
                            "full_repeat": full_repeat,
                            "strategy": strategy["name"],
                            "query_count": query_count,
                            "target_input_tokens": target_input_tokens,
                            "actual_input_tokens": sum(int(item["prompt_tokens"]) for item in batch_prompts),
                            "batch_repeat": batch_repeat,
                            "power_limit": power_limit,
                            "actual_power_limit": actual_power_limit,
                            "inference_start": wall_start,
                            "inference_end": wall_end,
                            **batch_metrics,
                        })
                        time.sleep(inter_batch_sec)

                    power_data = monitor.stop()
                    for row in block_rows:
                        power_stats = build_power_window_stats(
                            row["inference_start"],
                            row["inference_end"],
                            power_data,
                        )
                        # 对每个 measured batch 截取自己的推理窗口，避免把其他 batch 的功率混入。
                        row.update(power_stats)
                        row.pop("inference_start", None)
                        row.pop("inference_end", None)
                    raw_rows.extend(block_rows)
                    append_csv_rows(raw_path, RAW_FIELDNAMES, block_rows)
                    aggregated_rows = aggregate_raw_rows(raw_rows)
                    initialize_csv_file(agg_path, AGG_FIELDNAMES)
                    append_csv_rows(agg_path, AGG_FIELDNAMES, aggregated_rows)

                    completed_blocks += 1
                    write_progress_file(
                        progress_path,
                        experiment_id=experiment_id,
                        total_blocks=total_blocks,
                        completed_blocks=completed_blocks,
                        status="running",
                        started_at=metadata["started_at"],
                        last_completed_block=current_block,
                    )
    except Exception as exc:
        write_progress_file(
            progress_path,
            experiment_id=experiment_id,
            total_blocks=total_blocks,
            completed_blocks=completed_blocks,
            status="failed",
            started_at=metadata["started_at"],
            error=str(exc),
        )
        raise
    finally:
        if keep_alive is not None:
            keep_alive.stop()

    metadata["finished_at"] = time.time()
    write_json_file(metadata_path, metadata)
    write_progress_file(
        progress_path,
        experiment_id=experiment_id,
        total_blocks=total_blocks,
        completed_blocks=completed_blocks,
        status="completed",
        started_at=metadata["started_at"],
    )
    return {
        "raw_file": raw_path,
        "aggregated_file": agg_path,
        "metadata_file": metadata_path,
        "progress_file": progress_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run concurrent prefill-only evaluation.")
    parser.add_argument("--output-dir", default="results_prefil")
    parser.add_argument("--model-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--served-model-name", default="Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--tokenizer-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--sharegpt-dir", default="./input/ShareGPT")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--repeats-per-batch", type=int, default=10)
    parser.add_argument("--full-repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--monitor-warmup-batches", type=int, default=1)
    parser.add_argument("--inter-batch-sec", type=float, default=0.3)
    parser.add_argument("--queue-seed", type=int, default=20260401)
    parser.add_argument("--sampling-seed", type=int, default=20260401)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--sudo-password", default=None)
    parser.add_argument("--skip-set-power", action="store_true")
    parser.add_argument("--strategy-names", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_prefill_concurrent_evaluation(
        output_dir=args.output_dir,
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        tokenizer_path=args.tokenizer_path,
        sharegpt_dir=args.sharegpt_dir,
        base_url=args.base_url,
        repeats_per_batch=args.repeats_per_batch,
        full_repeats=args.full_repeats,
        warmup_batches=args.warmup_batches,
        monitor_warmup_batches=args.monitor_warmup_batches,
        inter_batch_sec=args.inter_batch_sec,
        queue_seed=args.queue_seed,
        sampling_seed=args.sampling_seed,
        device_index=args.device_index,
        sudo_password=args.sudo_password,
        skip_set_power=args.skip_set_power,
        strategy_names=[item.strip() for item in args.strategy_names.split(",")] if args.strategy_names else None,
    )


if __name__ == "__main__":
    main()
