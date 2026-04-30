#!/usr/bin/env python3
"""
固定完整请求的 GPU power-cap 能量扫描实验。

实验固定 query_count、output_length、prompt 采样种子和 batch 顺序，
逐个 power cap 运行完整 batch，记录 J/output_token。
"""
import argparse
import csv
import json
import os
import random
import statistics
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


QUERY_GROUPS = [
    {"query_count": 8, "target_input_tokens": 225},
    {"query_count": 16, "target_input_tokens": 504},
    {"query_count": 32, "target_input_tokens": 1581},
    {"query_count": 64, "target_input_tokens": 2175},
    {"query_count": 96, "target_input_tokens": 11106},
    {"query_count": 128, "target_input_tokens": 22873},
]

DEFAULT_POWER_CAPS = [150, 170, 190, 210, 230, 250, 275, 300, 350]
DEFAULT_OUTPUT_LENGTHS = [100]

RAW_FIELDNAMES = [
    "experiment_id",
    "power_cap_w",
    "actual_power_cap_w",
    "query_count",
    "target_input_tokens",
    "actual_input_tokens",
    "output_length",
    "batch_repeat",
    "num_requests",
    "actual_output_tokens",
    "total_energy_j",
    "energy_per_output_token_j",
    "avg_power_w",
    "peak_power_w",
    "avg_sm_clock_mhz",
    "avg_mem_clock_mhz",
    "avg_ttft_ms",
    "avg_tbt_ms",
    "avg_e2e_ms",
]

AGG_FIELDNAMES = [
    "experiment_id",
    "power_cap_w",
    "query_count",
    "target_input_tokens",
    "actual_input_tokens",
    "output_length",
    "num_samples",
    "num_requests",
    "actual_output_tokens",
    "total_energy_j",
    "energy_per_output_token_j",
    "avg_power_w",
    "peak_power_w",
    "avg_sm_clock_mhz",
    "avg_mem_clock_mhz",
    "avg_ttft_ms",
    "avg_tbt_ms",
    "avg_e2e_ms",
]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def compute_energy_per_output_token(total_energy_j: float, actual_output_tokens: int) -> float:
    if int(actual_output_tokens) <= 0:
        return 0.0
    return float(total_energy_j) / int(actual_output_tokens)


def build_query_groups(query_counts: Sequence[int]) -> List[Dict]:
    known = {int(item["query_count"]): int(item["target_input_tokens"]) for item in QUERY_GROUPS}
    groups = []
    for query_count in query_counts:
        query_count = int(query_count)
        groups.append({
            "query_count": query_count,
            "target_input_tokens": known.get(query_count, max(1, query_count * 128)),
        })
    return groups


def build_experiment_blocks(query_groups: Sequence[Dict],
                            output_lengths: Sequence[int],
                            power_caps: Sequence[int]) -> List[Dict]:
    blocks = []
    for power_cap in power_caps:
        for query_group in query_groups:
            for output_length in output_lengths:
                blocks.append({
                    "power_cap_w": int(power_cap),
                    "query_count": int(query_group["query_count"]),
                    "target_input_tokens": int(query_group.get("target_input_tokens", 0)),
                    "output_length": int(output_length),
                })
    return blocks


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


def build_service_extra_body(output_length: int, sampling_seed: int) -> Dict:
    return {
        "min_tokens": int(output_length),
        "ignore_eos": True,
        "top_p": 1.0,
        "seed": int(sampling_seed),
    }


def summarize_request_metrics(results: List[Dict]) -> Dict[str, float]:
    ttfts = [float(item.get("ttft", 0.0)) for item in results]
    tbts = [float(item.get("tbt", 0.0)) for item in results]
    e2es = [float(item.get("e2e", 0.0)) for item in results]
    output_tokens = [int(item.get("token_count", 0)) for item in results]
    return {
        "num_requests": len(results),
        "actual_output_tokens": sum(output_tokens),
        "avg_ttft_ms": statistics.mean(ttfts) if ttfts else 0.0,
        "avg_tbt_ms": statistics.mean(tbts) if tbts else 0.0,
        "avg_e2e_ms": statistics.mean(e2es) if e2es else 0.0,
    }


def build_power_stats(power_data: List[Dict]) -> Dict[str, float]:
    if not power_data:
        return {
            "avg_power_w": 0.0,
            "total_energy_j": 0.0,
            "peak_power_w": 0.0,
            "avg_sm_clock_mhz": 0.0,
            "avg_mem_clock_mhz": 0.0,
        }
    avg_power = statistics.mean(float(point.get("power_w", 0.0)) for point in power_data)
    return {
        "avg_power_w": avg_power,
        "total_energy_j": calculate_total_energy(power_data),
        "peak_power_w": max(float(point.get("power_w", 0.0)) for point in power_data),
        "avg_sm_clock_mhz": statistics.mean(float(point.get("graphics_clock_mhz", 0.0)) for point in power_data),
        "avg_mem_clock_mhz": statistics.mean(float(point.get("memory_clock_mhz", 0.0)) for point in power_data),
    }


def calculate_total_energy(power_data: List[Dict]) -> float:
    if len(power_data) < 2:
        return 0.0
    total_energy = 0.0
    for index in range(1, len(power_data)):
        previous = power_data[index - 1]
        current = power_data[index]
        dt = float(current["timestamp"]) - float(previous["timestamp"])
        avg_power = (float(previous["power_w"]) + float(current["power_w"])) / 2.0
        total_energy += avg_power * max(0.0, dt)
    return total_energy


def aggregate_raw_rows(raw_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (
            row.get("experiment_id", ""),
            int(row["power_cap_w"]),
            int(row["query_count"]),
            int(row.get("target_input_tokens", 0)),
            int(row["output_length"]),
        )
        grouped[key].append(row)

    aggregated = []
    for key, rows in sorted(grouped.items()):
        total_output_tokens = sum(int(float(item["actual_output_tokens"])) for item in rows)
        total_energy = sum(float(item["total_energy_j"]) for item in rows)
        aggregated.append({
            "experiment_id": key[0],
            "power_cap_w": key[1],
            "query_count": key[2],
            "target_input_tokens": key[3],
            "actual_input_tokens": statistics.mean(float(item["actual_input_tokens"]) for item in rows),
            "output_length": key[4],
            "num_samples": len(rows),
            "num_requests": sum(int(float(item["num_requests"])) for item in rows),
            "actual_output_tokens": total_output_tokens,
            "total_energy_j": total_energy,
            "energy_per_output_token_j": compute_energy_per_output_token(total_energy, total_output_tokens),
            "avg_power_w": statistics.mean(float(item["avg_power_w"]) for item in rows),
            "peak_power_w": max(float(item.get("peak_power_w", 0.0)) for item in rows),
            "avg_sm_clock_mhz": statistics.mean(float(item.get("avg_sm_clock_mhz", 0.0)) for item in rows),
            "avg_mem_clock_mhz": statistics.mean(float(item.get("avg_mem_clock_mhz", 0.0)) for item in rows),
            "avg_ttft_ms": statistics.mean(float(item["avg_ttft_ms"]) for item in rows),
            "avg_tbt_ms": statistics.mean(float(item["avg_tbt_ms"]) for item in rows),
            "avg_e2e_ms": statistics.mean(float(item["avg_e2e_ms"]) for item in rows),
        })
    return aggregated


def select_subset_prompts(load_generator,
                          num_prompts: int,
                          target_tokens: int,
                          rng: random.Random) -> List[Dict]:
    avg_per_prompt = max(1, int(target_tokens) // int(num_prompts))
    prompts = []
    for _ in range(int(num_prompts)):
        jitter = max(1, avg_per_prompt // 10)
        target_length = max(1, avg_per_prompt + rng.randint(-jitter, jitter))
        prompt = load_generator.generate_prompt_by_token_count(
            target_length,
            prefer_sharegpt=True,
            add_unique_prefix=True,
        )
        prompts.append({
            "prompt": prompt,
            "prompt_tokens": load_generator.count_tokens(prompt),
        })
    rng.shuffle(prompts)
    return prompts


def build_prompt_sets(load_generator,
                      query_groups: Sequence[Dict],
                      output_lengths: Sequence[int],
                      repeats_per_power_cap: int,
                      warmup_batches: int,
                      queue_seed: int) -> Dict[Tuple[int, int], List[List[Dict]]]:
    prompt_sets = {}
    total_batches = int(repeats_per_power_cap) + int(warmup_batches)
    for query_group in query_groups:
        query_count = int(query_group["query_count"])
        target_input_tokens = int(query_group["target_input_tokens"])
        for output_length in output_lengths:
            rng = random.Random(int(queue_seed) + query_count * 1000 + int(output_length))
            batches = [
                select_subset_prompts(
                    load_generator=load_generator,
                    num_prompts=query_count,
                    target_tokens=target_input_tokens,
                    rng=rng,
                )
                for _ in range(total_batches)
            ]
            prompt_sets[(query_count, int(output_length))] = batches
    return prompt_sets


def run_power_cap_energy_sweep(output_dir: str,
                               model_path: str,
                               served_model_name: str,
                               tokenizer_path: str,
                               sharegpt_dir: str,
                               base_url: str,
                               query_counts: Sequence[int],
                               output_lengths: Sequence[int],
                               power_caps: Sequence[int],
                               repeats_per_power_cap: int,
                               warmup_batches: int,
                               inter_batch_sec: float,
                               queue_seed: int,
                               sampling_seed: int,
                               sudo_password: Optional[str],
                               skip_set_power: bool):
    from llm_inference import LLMInferencer
    from load_generator import LoadGenerator
    from monitor import PowerMonitor
    from power_control import SudoKeepAlive, get_power_cap, set_power_cap

    os.makedirs(output_dir, exist_ok=True)
    experiment_id = f"power_cap_energy_sweep_{int(time.time())}"
    raw_path = os.path.join(output_dir, f"{experiment_id}_raw.csv")
    agg_path = os.path.join(output_dir, f"{experiment_id}_aggregated.csv")
    metadata_path = os.path.join(output_dir, f"{experiment_id}_metadata.json")
    progress_path = os.path.join(output_dir, f"{experiment_id}_progress.json")

    initialize_csv_file(raw_path, RAW_FIELDNAMES)
    initialize_csv_file(agg_path, AGG_FIELDNAMES)

    query_groups = build_query_groups(query_counts)
    blocks = build_experiment_blocks(query_groups, output_lengths, power_caps)
    metadata = {
        "experiment_id": experiment_id,
        "query_groups": query_groups,
        "output_lengths": [int(value) for value in output_lengths],
        "power_caps": [int(value) for value in power_caps],
        "repeats_per_power_cap": int(repeats_per_power_cap),
        "warmup_batches": int(warmup_batches),
        "inter_batch_sec": float(inter_batch_sec),
        "queue_seed": int(queue_seed),
        "sampling_seed": int(sampling_seed),
        "base_url": base_url,
        "model_path": model_path,
        "served_model_name": served_model_name,
        "tokenizer_path": tokenizer_path,
        "sharegpt_dir": sharegpt_dir,
        "skip_set_power": bool(skip_set_power),
        "started_at": time.time(),
    }
    write_json_file(metadata_path, metadata)

    inferencer = LLMInferencer(
        model_name=model_path,
        use_service=True,
        base_url=base_url,
        served_model_name=served_model_name,
        service_request_mode="completion",
    )
    load_generator = LoadGenerator(sharegpt_dir=sharegpt_dir, tokenizer_name=tokenizer_path)
    prompt_sets = build_prompt_sets(
        load_generator=load_generator,
        query_groups=query_groups,
        output_lengths=output_lengths,
        repeats_per_power_cap=repeats_per_power_cap,
        warmup_batches=warmup_batches,
        queue_seed=queue_seed,
    )

    keep_alive = None
    if not skip_set_power:
        keep_alive = SudoKeepAlive(interval_sec=60.0)
        if not keep_alive.start(sudo_password=sudo_password):
            raise RuntimeError("Failed to initialize sudo keepalive")

    completed_blocks = 0
    raw_rows: List[Dict] = []
    try:
        write_progress_file(
            progress_path,
            experiment_id=experiment_id,
            total_blocks=len(blocks),
            completed_blocks=completed_blocks,
            status="running",
            started_at=metadata["started_at"],
        )

        for block in blocks:
            write_progress_file(
                progress_path,
                experiment_id=experiment_id,
                total_blocks=len(blocks),
                completed_blocks=completed_blocks,
                status="running",
                current_block=block,
                started_at=metadata["started_at"],
            )
            power_cap = int(block["power_cap_w"])
            output_length = int(block["output_length"])
            query_count = int(block["query_count"])
            if not skip_set_power and not set_power_cap(power_cap, sudo_password=sudo_password):
                raise RuntimeError(f"Failed to set GPU power cap to {power_cap}W")
            actual_power_cap = get_power_cap()
            time.sleep(2.0)

            batches = prompt_sets[(query_count, output_length)]
            extra_body = build_service_extra_body(output_length, sampling_seed)

            for warmup_index in range(int(warmup_batches)):
                prompts = [item["prompt"] for item in batches[warmup_index]]
                inferencer.infer_concurrent(
                    prompts,
                    max_tokens=output_length,
                    temperature=0.0,
                    extra_body=extra_body,
                )
                time.sleep(float(inter_batch_sec))

            block_rows = []
            for batch_offset in range(int(repeats_per_power_cap)):
                batch_index = int(warmup_batches) + batch_offset
                batch_prompts = batches[batch_index]
                prompts = [item["prompt"] for item in batch_prompts]
                actual_input_tokens = sum(int(item["prompt_tokens"]) for item in batch_prompts)

                monitor = PowerMonitor()
                monitor.start()
                results = inferencer.infer_concurrent(
                    prompts,
                    max_tokens=output_length,
                    temperature=0.0,
                    extra_body=extra_body,
                )
                power_data = monitor.stop()
                power_stats = build_power_stats(power_data)
                request_metrics = summarize_request_metrics(results)
                energy_per_token = compute_energy_per_output_token(
                    power_stats["total_energy_j"],
                    int(request_metrics["actual_output_tokens"]),
                )
                row = {
                    "experiment_id": experiment_id,
                    "power_cap_w": power_cap,
                    "actual_power_cap_w": actual_power_cap,
                    "query_count": query_count,
                    "target_input_tokens": int(block["target_input_tokens"]),
                    "actual_input_tokens": actual_input_tokens,
                    "output_length": output_length,
                    "batch_repeat": batch_offset + 1,
                    **request_metrics,
                    **power_stats,
                    "energy_per_output_token_j": energy_per_token,
                }
                block_rows.append(row)
                raw_rows.append(row)
                append_csv_rows(raw_path, RAW_FIELDNAMES, [row])
                time.sleep(float(inter_batch_sec))

            completed_blocks += 1
            aggregate_rows = aggregate_raw_rows(raw_rows)
            initialize_csv_file(agg_path, AGG_FIELDNAMES)
            append_csv_rows(agg_path, AGG_FIELDNAMES, aggregate_rows)
            write_progress_file(
                progress_path,
                experiment_id=experiment_id,
                total_blocks=len(blocks),
                completed_blocks=completed_blocks,
                status="running",
                last_completed_block=block,
                started_at=metadata["started_at"],
            )
            print(
                f"完成 power={power_cap}W q={query_count} output={output_length}: "
                f"{len(block_rows)} batches"
            )

        write_progress_file(
            progress_path,
            experiment_id=experiment_id,
            total_blocks=len(blocks),
            completed_blocks=completed_blocks,
            status="completed",
            started_at=metadata["started_at"],
        )
    except Exception as exc:
        write_progress_file(
            progress_path,
            experiment_id=experiment_id,
            total_blocks=len(blocks),
            completed_blocks=completed_blocks,
            status="failed",
            current_block=blocks[completed_blocks] if completed_blocks < len(blocks) else None,
            started_at=metadata["started_at"],
            error=str(exc),
        )
        raise
    finally:
        if keep_alive is not None:
            keep_alive.stop()

    return {
        "experiment_id": experiment_id,
        "raw_path": raw_path,
        "aggregated_path": agg_path,
        "metadata_path": metadata_path,
        "progress_path": progress_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-load power-cap energy sweep.")
    parser.add_argument("--output-dir", default="experiment_results/power_cap_energy_sweep/default")
    parser.add_argument("--model-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--tokenizer-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--served-model-name", default="Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--sharegpt-dir", default="./input/ShareGPT")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--query-counts", default="64")
    parser.add_argument("--output-lengths", default="100")
    parser.add_argument("--power-caps", default="150,170,190,210,230,250,275,300,350")
    parser.add_argument("--repeats-per-power-cap", type=int, default=5)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--inter-batch-sec", type=float, default=0.3)
    parser.add_argument("--queue-seed", type=int, default=20260425)
    parser.add_argument("--sampling-seed", type=int, default=20260425)
    parser.add_argument("--sudo-password", default=None)
    parser.add_argument("--skip-set-power", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = run_power_cap_energy_sweep(
        output_dir=args.output_dir,
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        tokenizer_path=args.tokenizer_path,
        sharegpt_dir=args.sharegpt_dir,
        base_url=args.base_url,
        query_counts=parse_int_list(args.query_counts),
        output_lengths=parse_int_list(args.output_lengths),
        power_caps=parse_int_list(args.power_caps),
        repeats_per_power_cap=args.repeats_per_power_cap,
        warmup_batches=args.warmup_batches,
        inter_batch_sec=args.inter_batch_sec,
        queue_seed=args.queue_seed,
        sampling_seed=args.sampling_seed,
        sudo_password=args.sudo_password,
        skip_set_power=args.skip_set_power,
    )
    print("输出文件:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
