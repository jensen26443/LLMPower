#!/usr/bin/env python3
"""
解码阶段功率限制策略评估实验脚本

对比不同 decode 功率限制策略在固定 1-token 输入、固定输出长度下的
TBT / TTFT / E2E / 功率 / 能耗表现。
"""
import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from llm_inference import LLMInferencer
from load_generator import LoadGenerator
from monitor import PowerMonitor
from power_control import (
    SudoKeepAlive,
    get_power_cap,
    set_power_cap,
)


DECODE_OUTPUT_LENGTHS = [8, 16, 32, 64, 128]

DECODE_STRATEGIES = [
    {
        "name": "scheme1_fit_bucket",
        "type": "bucket",
        "buckets": [(8, 150), (16, 150), (32, 180), (float("inf"), 220)],
    },
    {
        "name": "scheme2_fit_plus20",
        "type": "bucket",
        "buckets": [(8, 150), (16, 170), (32, 200), (float("inf"), 240)],
    },
    {
        "name": "scheme3_balanced_v2",
        "type": "bucket",
        "buckets": [(8, 170), (16, 170), (32, 200), (64, 220), (float("inf"), 220)],
    },
    {
        "name": "scheme4_latency_v2",
        "type": "bucket",
        "buckets": [(8, 180), (16, 180), (32, 210), (64, 230), (float("inf"), 230)],
    },
    {
        "name": "baseline_350w",
        "type": "fixed",
        "power": 350,
    },
]

RAW_FIELDNAMES = [
    "full_repeat", "strategy", "output_length", "power_limit", "actual_power_limit",
    "concurrency", "batch_repeat", "prompt_token_count", "num_requests", "actual_output_tokens",
    "avg_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "avg_tbt_ms", "p50_tbt_ms", "p95_tbt_ms", "p99_tbt_ms",
    "avg_e2e_ms", "p50_e2e_ms", "p95_e2e_ms", "p99_e2e_ms",
    "avg_power_w", "total_energy_j", "peak_power_w",
    "inference_start", "inference_end",
]

AGG_FIELDNAMES = [
    "full_repeat", "strategy", "output_length", "concurrency", "power_limit", "num_samples",
    "avg_ttft_ms", "avg_tbt_ms", "avg_e2e_ms", "avg_energy_j", "avg_power_w",
]


def build_decode_eval_prompts(load_generator: LoadGenerator,
                              prompt_token_count: int,
                              repeats_per_prompt: int) -> List[str]:
    return [
        load_generator.generate_prompt_by_token_count(
            prompt_token_count,
            prefer_sharegpt=True,
            add_unique_prefix=False,
        )
        for _ in range(repeats_per_prompt)
    ]


def build_prompt_batches(prompts: List[str], batch_size: int) -> List[List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(prompts) % batch_size != 0:
        raise ValueError("prompt count must be divisible by batch_size")
    return [
        prompts[index:index + batch_size]
        for index in range(0, len(prompts), batch_size)
    ]


def split_prompt_batches(prompts: Sequence[Sequence[str]],
                         warmup_count: int,
                         monitor_warmup_count: int):
    warmup_end = warmup_count
    monitor_warmup_end = warmup_end + monitor_warmup_count
    return (
        prompts[:warmup_end],
        prompts[warmup_end:monitor_warmup_end],
        prompts[monitor_warmup_end:],
    )


def build_output_prompt_sets(load_generator: LoadGenerator,
                             output_lengths: List[int],
                             prompt_token_count: int,
                             repeats_per_batch: int,
                             concurrency: int,
                             warmup_batches: int,
                             monitor_warmup_batches: int) -> Dict[int, tuple[List[List[str]], List[List[str]], List[List[str]]]]:
    prompt_sets = {}
    total_batches = warmup_batches + monitor_warmup_batches + repeats_per_batch
    total_prompts = total_batches * concurrency
    for output_length in output_lengths:
        prompts = build_decode_eval_prompts(
            load_generator,
            prompt_token_count=prompt_token_count,
            repeats_per_prompt=total_prompts,
        )
        prompt_batches = build_prompt_batches(prompts, batch_size=concurrency)
        prompt_sets[output_length] = split_prompt_batches(
            prompt_batches,
            warmup_count=warmup_batches,
            monitor_warmup_count=monitor_warmup_batches,
        )
    return prompt_sets


def build_decode_eval_extra_body(output_length: int, sampling_seed: int) -> Dict:
    return {
        "min_tokens": output_length,
        "ignore_eos": True,
        "top_p": 1.0,
        "seed": sampling_seed,
    }


def get_power_for_decode_strategy(strategy: Dict, output_length: int) -> int:
    if strategy["type"] == "fixed":
        return int(strategy["power"])

    for threshold, power in strategy["buckets"]:
        if output_length <= threshold:
            return int(power)
    return int(strategy["buckets"][-1][1])


def validate_actual_power_limit(expected_power: int, actual_power: float, tolerance_w: float = 5.0) -> bool:
    if abs(float(actual_power) - float(expected_power)) > tolerance_w:
        raise RuntimeError(
            f"Power limit mismatch: expected {expected_power}W, got {actual_power:.1f}W"
        )
    return True


def estimate_request_tbt_ms(request_result: Dict, output_tokens: int) -> float:
    if output_tokens > 1 and request_result["ttft"] < request_result["e2e"]:
        return (request_result["e2e"] - request_result["ttft"]) / (output_tokens - 1)
    return float(request_result.get("tbt", 0.0))


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


def save_csv(file_path: str, rows: List[Dict], fieldnames: List[str]):
    with open(file_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    if error:
        payload["error"] = error
    write_json_file(file_path, payload)


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * ratio
    lower = math.floor(rank)
    upper = math.ceil(rank)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_batch_metrics(results: List[Dict]) -> Dict[str, float]:
    ttfts = [float(item["ttft"]) for item in results]
    tbts = [estimate_request_tbt_ms(item, int(item.get("token_count", 0))) for item in results]
    e2es = [float(item["e2e"]) for item in results]
    output_tokens = [int(item.get("token_count", 0)) for item in results]
    return {
        "num_requests": len(results),
        "actual_output_tokens": int(round(statistics.mean(output_tokens))) if output_tokens else 0,
        "avg_ttft_ms": statistics.mean(ttfts) if ttfts else 0.0,
        "p50_ttft_ms": percentile(ttfts, 0.5),
        "p95_ttft_ms": percentile(ttfts, 0.95),
        "p99_ttft_ms": percentile(ttfts, 0.99),
        "avg_tbt_ms": statistics.mean(tbts) if tbts else 0.0,
        "p50_tbt_ms": percentile(tbts, 0.5),
        "p95_tbt_ms": percentile(tbts, 0.95),
        "p99_tbt_ms": percentile(tbts, 0.99),
        "avg_e2e_ms": statistics.mean(e2es) if e2es else 0.0,
        "p50_e2e_ms": percentile(e2es, 0.5),
        "p95_e2e_ms": percentile(e2es, 0.95),
        "p99_e2e_ms": percentile(e2es, 0.99),
    }


def rotate_sequence(items: Sequence, offset: int) -> List:
    if not items:
        return []
    shift = offset % len(items)
    return list(items[shift:]) + list(items[:shift])


def build_experiment_blocks(strategies: Sequence[Dict],
                            output_lengths: Sequence[int],
                            full_repeats: int) -> List[Dict]:
    blocks = []
    for full_repeat in range(1, full_repeats + 1):
        strategy_order = rotate_sequence(strategies, full_repeat - 1)
        output_order = rotate_sequence(output_lengths, full_repeat - 1)
        for strategy in strategy_order:
            for output_length in output_order:
                blocks.append({
                    "full_repeat": full_repeat,
                    "strategy": strategy,
                    "strategy_name": strategy["name"],
                    "output_length": int(output_length),
                })
    return blocks


def aggregate_results(raw_results: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in raw_results:
        grouped[(row["full_repeat"], row["strategy"], row["output_length"], row["concurrency"])].append(row)

    aggregated = []
    for (full_repeat, strategy, output_length, concurrency), group in sorted(grouped.items()):
        ttfts = [row["avg_ttft_ms"] for row in group]
        tbts = [row["avg_tbt_ms"] for row in group]
        e2es = [row["avg_e2e_ms"] for row in group]
        energies = [row["total_energy_j"] for row in group]
        powers = [row["avg_power_w"] for row in group]

        aggregated.append({
            "full_repeat": full_repeat,
            "strategy": strategy,
            "output_length": output_length,
            "concurrency": concurrency,
            "power_limit": group[0]["power_limit"],
            "num_samples": len(group),
            "avg_ttft_ms": statistics.mean(ttfts),
            "avg_tbt_ms": statistics.mean(tbts),
            "avg_e2e_ms": statistics.mean(e2es),
            "avg_energy_j": statistics.mean(energies),
            "avg_power_w": statistics.mean(powers),
        })
    return aggregated


def run_decode_strategy_evaluation(
    output_dir: str = "results_decode/strategy_evaluation",
    model_path: str = "./Qwen2.5-7B-Instruct-AWQ",
    tokenizer_path: str = "./Qwen2.5-7B-Instruct-AWQ",
    served_model_name: str = "Qwen2.5-7B-Instruct-AWQ",
    base_url: str = "http://localhost:8000/v1",
    repeats_per_batch: int = 100,
    full_repeats: int = 3,
    concurrency: int = 8,
    prompt_token_count: int = 1,
    sampling_seed: int = 20260329,
    warmup_batches: int = 3,
    monitor_warmup_batches: int = 1,
    sudo_password: Optional[str] = None,
    skip_set_power: bool = False,
    only_strategy: Optional[str] = None,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    if prompt_token_count != 1:
        raise ValueError("decode strategy evaluation requires prompt_token_count=1")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    strategies = DECODE_STRATEGIES
    if only_strategy:
        strategies = [item for item in strategies if item["name"] == only_strategy]
        if not strategies:
            raise ValueError(f"unknown strategy: {only_strategy}")

    inferencer = LLMInferencer(
        use_service=True,
        model_name=model_path,
        served_model_name=served_model_name,
        base_url=base_url,
        service_request_mode="completion",
    )
    load_generator = LoadGenerator(sharegpt_dir="", tokenizer_name=tokenizer_path)
    base_prompt = load_generator.generate_prompt_by_token_count(
        prompt_token_count,
        prefer_sharegpt=True,
        add_unique_prefix=False,
    )

    warmup_extra_body = build_decode_eval_extra_body(16, sampling_seed)
    for _ in range(2):
        inferencer.infer_concurrent([base_prompt] * 4, max_tokens=16, temperature=0.0, extra_body=warmup_extra_body)
    time.sleep(1.0)

    raw_results = []
    experiment_id = f"decode_strategy_eval_{int(time.time())}"
    raw_file = os.path.join(output_dir, f"{experiment_id}_raw.csv")
    aggregated_file = os.path.join(output_dir, f"{experiment_id}_aggregated.csv")
    metadata_file = os.path.join(output_dir, f"{experiment_id}_metadata.json")
    progress_file = os.path.join(output_dir, f"{experiment_id}_progress.json")
    started_at = time.time()

    metadata = {
        "timestamp": started_at,
        "repeats_per_batch": repeats_per_batch,
        "full_repeats": full_repeats,
        "concurrency": concurrency,
        "output_lengths": DECODE_OUTPUT_LENGTHS,
        "warmup_batches": warmup_batches,
        "monitor_warmup_batches": monitor_warmup_batches,
        "order_mode": "rotate_by_full_repeat",
        "strategies": DECODE_STRATEGIES,
    }
    write_json_file(metadata_file, metadata)
    initialize_csv_file(raw_file, RAW_FIELDNAMES)
    initialize_csv_file(aggregated_file, AGG_FIELDNAMES)

    blocks = build_experiment_blocks(strategies, DECODE_OUTPUT_LENGTHS, full_repeats)
    total_blocks = len(blocks)
    write_progress_file(
        progress_file,
        experiment_id=experiment_id,
        total_blocks=total_blocks,
        completed_blocks=0,
        status="running",
        started_at=started_at,
    )

    sudo_keepalive = None
    completed_blocks = 0
    prompt_sets_by_repeat: Dict[int, Dict[int, tuple[List[List[str]], List[List[str]], List[List[str]]]]] = {}
    if not skip_set_power:
        sudo_keepalive = SudoKeepAlive(interval_sec=60.0)
        if not sudo_keepalive.start(sudo_password=sudo_password):
            raise RuntimeError("failed to initialize sudo keepalive")

    try:
        overall_progress = tqdm(blocks, desc="config blocks", unit="block")
        for block_index, block in enumerate(overall_progress, start=1):
            full_repeat = block["full_repeat"]
            strategy = block["strategy"]
            output_length = block["output_length"]
            overall_progress.set_postfix(
                full_repeat=full_repeat,
                strategy=strategy["name"],
                output=output_length,
            )
            write_progress_file(
                progress_file,
                experiment_id=experiment_id,
                total_blocks=total_blocks,
                completed_blocks=block_index - 1,
                status="running",
                current_block={
                    "full_repeat": full_repeat,
                    "strategy_name": strategy["name"],
                    "output_length": output_length,
                },
                started_at=started_at,
            )

            if full_repeat not in prompt_sets_by_repeat:
                prompt_sets_by_repeat[full_repeat] = build_output_prompt_sets(
                    load_generator,
                    output_lengths=DECODE_OUTPUT_LENGTHS,
                    prompt_token_count=prompt_token_count,
                    repeats_per_batch=repeats_per_batch,
                    concurrency=concurrency,
                    warmup_batches=warmup_batches,
                    monitor_warmup_batches=monitor_warmup_batches,
                )
            prompt_sets = prompt_sets_by_repeat[full_repeat]

            power_limit = get_power_for_decode_strategy(strategy, output_length)
            if not skip_set_power:
                if not set_power_cap(power_limit, sudo_password=sudo_password):
                    raise RuntimeError(
                        f"failed to set power limit {power_limit}W for "
                        f"{strategy['name']} output={output_length}"
                    )
                time.sleep(10)
                actual_power_limit = get_power_cap()
                validate_actual_power_limit(power_limit, actual_power_limit)
            else:
                actual_power_limit = get_power_cap()

            warmup_prompts, monitor_warmup_prompts, measurement_prompts = prompt_sets[output_length]
            extra_body = build_decode_eval_extra_body(output_length, sampling_seed)

            for warmup_prompt_batch in warmup_prompts:
                inferencer.infer_concurrent(
                    warmup_prompt_batch,
                    max_tokens=output_length,
                    temperature=0.0,
                    extra_body=extra_body,
                )

            monitor = PowerMonitor(sample_interval=0.02)
            monitor.start()
            for prompt_batch in monitor_warmup_prompts:
                inferencer.infer_concurrent(
                    prompt_batch,
                    max_tokens=output_length,
                    temperature=0.0,
                    extra_body=extra_body,
                )

            records = []
            for repeat_index, prompt_batch in enumerate(
                tqdm(measurement_prompts, desc=f"{strategy['name']} x={output_length}", leave=False),
                start=1,
            ):
                start_time = time.time()
                results = inferencer.infer_concurrent(
                    prompt_batch,
                    max_tokens=output_length,
                    temperature=0.0,
                    extra_body=extra_body,
                )
                end_time = time.time()
                batch_summary = summarize_batch_metrics(results)
                records.append({
                    "full_repeat": full_repeat,
                    "strategy": strategy["name"],
                    "output_length": output_length,
                    "concurrency": concurrency,
                    "power_limit": power_limit,
                    "batch_repeat": repeat_index,
                    "prompt_token_count": prompt_token_count,
                    **batch_summary,
                    "inference_start": start_time,
                    "inference_end": end_time,
                })

            power_data = monitor.stop()
            for row in records:
                stats = build_power_window_stats(row["inference_start"], row["inference_end"], power_data)
                row.update(stats)
                row["actual_power_limit"] = actual_power_limit
                raw_results.append(row)
            append_csv_rows(raw_file, RAW_FIELDNAMES, records)
            aggregated_results = aggregate_results(raw_results)
            save_csv(aggregated_file, aggregated_results, AGG_FIELDNAMES)
            completed_blocks = block_index
            write_progress_file(
                progress_file,
                experiment_id=experiment_id,
                total_blocks=total_blocks,
                completed_blocks=completed_blocks,
                status="running",
                current_block=None,
                last_completed_block={
                    "full_repeat": full_repeat,
                    "strategy_name": strategy["name"],
                    "output_length": output_length,
                },
                started_at=started_at,
            )
    except KeyboardInterrupt:
        write_progress_file(
            progress_file,
            experiment_id=experiment_id,
            total_blocks=total_blocks,
            completed_blocks=completed_blocks,
            status="interrupted",
            started_at=started_at,
            error="KeyboardInterrupt",
        )
        raise
    except Exception as error:
        write_progress_file(
            progress_file,
            experiment_id=experiment_id,
            total_blocks=total_blocks,
            completed_blocks=completed_blocks,
            status="failed",
            started_at=started_at,
            error=str(error),
        )
        raise
    finally:
        if sudo_keepalive:
            sudo_keepalive.stop()

    write_progress_file(
        progress_file,
        experiment_id=experiment_id,
        total_blocks=total_blocks,
        completed_blocks=total_blocks,
        status="completed",
        started_at=started_at,
    )

    return {
        "raw_file": raw_file,
        "aggregated_file": aggregated_file,
        "metadata_file": metadata_file,
        "progress_file": progress_file,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解码阶段功率限制策略评估实验")
    parser.add_argument("--output-dir", type=str, default="results_decode/strategy_evaluation")
    parser.add_argument("--model-path", type=str, default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--tokenizer-path", type=str, default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--served-model-name", type=str, default="Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--repeats-per-batch", type=int, default=100)
    parser.add_argument("--full-repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-token-count", type=int, default=1)
    parser.add_argument("--sampling-seed", type=int, default=20260329)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--monitor-warmup-batches", type=int, default=1)
    parser.add_argument("--sudo-password", type=str, default=None)
    parser.add_argument("--skip-set-power", action="store_true")
    parser.add_argument("--only-strategy", type=str, default=None)
    args = parser.parse_args()

    outputs = run_decode_strategy_evaluation(
        output_dir=args.output_dir,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        served_model_name=args.served_model_name,
        base_url=args.base_url,
        repeats_per_batch=args.repeats_per_batch,
        full_repeats=args.full_repeats,
        concurrency=args.concurrency,
        prompt_token_count=args.prompt_token_count,
        sampling_seed=args.sampling_seed,
        warmup_batches=args.warmup_batches,
        monitor_warmup_batches=args.monitor_warmup_batches,
        sudo_password=args.sudo_password,
        skip_set_power=args.skip_set_power,
        only_strategy=args.only_strategy,
    )
    print("实验完成，输出文件：")
    for key, value in outputs.items():
        print(f"- {key}: {value}")
