#!/usr/bin/env python3
"""
Prefill 阶段频率扫频实验。

在并发 prefill-only 口径下，固定当前 prefill 功率桶位，扫 `SM/MEM` 频率组合，
为后续多旋钮前馈策略选择 prefill 频率档位。
"""
import argparse
import csv
import json
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
    apply_hardware_profile,
    get_default_power_limit,
    get_power_cap,
    probe_clock_capabilities,
    reset_hardware_profile,
)
from run_feedforward_evaluation import V2_PREFILL_BUCKETS, get_prefill_power_for_total_tokens
from run_prefill_concurrent_evaluation import (
    QUERY_GROUPS,
    append_csv_rows,
    build_power_window_stats,
    build_query_group_prompt_sets,
    build_service_extra_body,
    initialize_csv_file,
    summarize_batch_metrics,
    write_json_file,
    write_progress_file,
)


PREFILL_BUCKETS = V2_PREFILL_BUCKETS
PREFILL_SM_SWEEP_TARGETS = [1800, 2200, 2600, 2800, 3105]

STRATEGIES = [
    {"name": "baseline_prefill_profile", "type": "baseline"},
    {"name": "prefill_freq_sweep", "type": "sweep"},
]

RAW_FIELDNAMES = [
    "full_repeat",
    "strategy",
    "query_count",
    "target_input_tokens",
    "actual_input_tokens",
    "batch_repeat",
    "power_limit",
    "clock_profile_name",
    "sm_clock_mhz",
    "mem_clock_mhz",
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
    "clock_profile_name",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "num_samples",
    "avg_ttft_ms",
    "avg_e2e_ms",
    "avg_energy_j",
    "avg_power_w",
]


def build_clock_profiles(capabilities: Dict) -> List[Dict]:
    supported_pairs = capabilities.get("supported_clock_pairs", [])
    if supported_pairs:
        max_mem = max(int(pair["memory_mhz"]) for pair in supported_pairs)
        supported_sm = sorted(
            {
                int(pair["graphics_mhz"])
                for pair in supported_pairs
                if int(pair["memory_mhz"]) == max_mem
            }
        )
        if supported_sm:
            selected_sms = []
            for target in PREFILL_SM_SWEEP_TARGETS:
                chosen = min(supported_sm, key=lambda sm: (abs(sm - target), -sm))
                if chosen not in selected_sms:
                    selected_sms.append(chosen)
            return [{
                "clock_profile_name": "baseline_default",
                "sm_clock_mhz": None,
                "mem_clock_mhz": None,
            }] + [
                {
                    "clock_profile_name": f"profile_{index}",
                    "sm_clock_mhz": int(sm_mhz),
                    "mem_clock_mhz": int(max_mem),
                }
                for index, sm_mhz in enumerate(selected_sms, start=1)
            ]

    sampled_pairs = capabilities.get("sampled_clock_pairs", [])
    profiles = []
    for index, pair in enumerate(sampled_pairs, start=1):
        profiles.append({
            "clock_profile_name": f"profile_{index}",
            "sm_clock_mhz": int(pair["graphics_mhz"]),
            "mem_clock_mhz": int(pair["memory_mhz"]),
        })
    return profiles


def aggregate_raw_rows(raw_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (
            row["full_repeat"],
            row["strategy"],
            row["query_count"],
            row["target_input_tokens"],
            row["power_limit"],
            row["clock_profile_name"],
            row["sm_clock_mhz"],
            row["mem_clock_mhz"],
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
            "clock_profile_name": key[5],
            "sm_clock_mhz": key[6],
            "mem_clock_mhz": key[7],
            "num_samples": len(rows),
            "avg_ttft_ms": statistics.mean(float(item["avg_ttft_ms"]) for item in rows),
            "avg_e2e_ms": statistics.mean(float(item["avg_e2e_ms"]) for item in rows),
            "avg_energy_j": statistics.mean(float(item["total_energy_j"]) for item in rows),
            "avg_power_w": statistics.mean(float(item["avg_power_w"]) for item in rows),
        })
    return aggregated


def run_prefill_freq_sweep(output_dir: str,
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
                           sudo_password: Optional[str],
                           skip_set_power: bool,
                           sample_count: int,
                           strategy_names: Optional[Sequence[str]] = None):
    os.makedirs(output_dir, exist_ok=True)
    experiment_id = f"prefill_freq_sweep_{int(time.time())}"
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

    capabilities = probe_clock_capabilities(sample_count=sample_count, min_sm_mhz=1000, min_mem_mhz=5000)
    clock_profiles = build_clock_profiles(capabilities)
    default_power = get_default_power_limit() or 350

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
        "prefill_buckets": PREFILL_BUCKETS,
        "repeats_per_batch": repeats_per_batch,
        "full_repeats": full_repeats,
        "warmup_batches": warmup_batches,
        "monitor_warmup_batches": monitor_warmup_batches,
        "queue_seed": queue_seed,
        "sampling_seed": sampling_seed,
        "base_url": base_url,
        "model_path": model_path,
        "served_model_name": served_model_name,
        "tokenizer_path": tokenizer_path,
        "sharegpt_dir": sharegpt_dir,
        "skip_set_power": skip_set_power,
        "clock_capability_json": capabilities,
        "clock_profiles": clock_profiles,
        "started_at": time.time(),
    }
    write_json_file(metadata_path, metadata)

    total_profile_blocks = 0
    for strategy in strategies:
        if strategy["type"] == "baseline":
            total_profile_blocks += len(QUERY_GROUPS)
        else:
            total_profile_blocks += len(QUERY_GROUPS) * max(1, len(clock_profiles))
    total_blocks = total_profile_blocks * full_repeats
    completed_blocks = 0
    raw_rows: List[Dict] = []

    def apply_profile(power_w: int, sm_mhz: Optional[int], mem_mhz: Optional[int]) -> bool:
        if skip_set_power:
            return True
        return apply_hardware_profile(
            power_w=power_w,
            sm_mhz=sm_mhz,
            mem_mhz=mem_mhz,
            sudo_password=sudo_password,
        )

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
            for strategy in strategies:
                for query_group in QUERY_GROUPS:
                    query_count = int(query_group["query_count"])
                    routing_tokens = int(query_group["target_input_tokens"])
                    profiles = [{"clock_profile_name": "baseline_default", "sm_clock_mhz": None, "mem_clock_mhz": None}]
                    if strategy["type"] != "baseline":
                        profiles = clock_profiles or [{"clock_profile_name": "unsupported_default", "sm_clock_mhz": None, "mem_clock_mhz": None}]

                    for profile in profiles:
                        current_block = {
                            "full_repeat": full_repeat,
                            "strategy_name": strategy["name"],
                            "query_count": query_count,
                            "clock_profile_name": profile["clock_profile_name"],
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

                        power_limit = 350 if strategy["type"] == "baseline" else get_prefill_power_for_total_tokens(
                            routing_tokens,
                            PREFILL_BUCKETS,
                        )
                        if not apply_profile(power_limit, profile["sm_clock_mhz"], profile["mem_clock_mhz"]):
                            raise RuntimeError(
                                f"Failed to apply profile power={power_limit}, sm={profile['sm_clock_mhz']}, mem={profile['mem_clock_mhz']}"
                            )

                        batches = prompt_sets[query_count]
                        warmup_slice = batches[:warmup_batches]
                        monitor_warmup_slice = batches[warmup_batches:warmup_batches + monitor_warmup_batches]
                        measurement_slice = batches[warmup_batches + monitor_warmup_batches:]
                        extra_body = build_service_extra_body(sampling_seed)

                        for warmup_batch in warmup_slice:
                            inferencer.infer_concurrent(
                                [item["prompt"] for item in warmup_batch],
                                max_tokens=1,
                                temperature=0.0,
                                extra_body=extra_body,
                            )
                            time.sleep(inter_batch_sec)

                        monitor = PowerMonitor(sample_interval=0.02)
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
                            tqdm(
                                measurement_slice,
                                desc=f"{strategy['name']} q={query_count} {profile['clock_profile_name']}",
                                leave=False,
                            ),
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
                            metric_stats = summarize_batch_metrics(results)
                            block_rows.append({
                                "full_repeat": full_repeat,
                                "strategy": strategy["name"],
                                "query_count": query_count,
                                "target_input_tokens": routing_tokens,
                                "actual_input_tokens": sum(int(item["prompt_tokens"]) for item in batch_prompts),
                                "batch_repeat": batch_repeat,
                                "power_limit": power_limit,
                                "clock_profile_name": profile["clock_profile_name"],
                                "sm_clock_mhz": profile["sm_clock_mhz"],
                                "mem_clock_mhz": profile["mem_clock_mhz"],
                                "inference_start": wall_start,
                                "inference_end": wall_end,
                                **metric_stats,
                            })
                            time.sleep(inter_batch_sec)

                        power_data = monitor.stop()
                        for row in block_rows:
                            power_stats = build_power_window_stats(
                                row["inference_start"],
                                row["inference_end"],
                                power_data,
                            )
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
        if not skip_set_power:
            reset_hardware_profile(default_power_w=default_power, sudo_password=sudo_password)
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


def main():
    parser = argparse.ArgumentParser(description="Run prefill frequency sweep.")
    parser.add_argument("--output-dir", default="results_freq/prefill_freq_sweep")
    parser.add_argument("--model-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--served-model-name", default="Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--tokenizer-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--sharegpt-dir", default="./filtered_prompts")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--repeats-per-batch", type=int, default=10)
    parser.add_argument("--full-repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--monitor-warmup-batches", type=int, default=1)
    parser.add_argument("--inter-batch-sec", type=float, default=0.8)
    parser.add_argument("--queue-seed", type=int, default=20260329)
    parser.add_argument("--sampling-seed", type=int, default=20260329)
    parser.add_argument("--sudo-password", default=os.environ.get("SUDO_PASSWORD"))
    parser.add_argument("--skip-set-power", action="store_true")
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--strategy-names", default=None)
    args = parser.parse_args()

    strategy_names = None
    if args.strategy_names:
        strategy_names = [item.strip() for item in str(args.strategy_names).split(",") if item.strip()]

    run_prefill_freq_sweep(
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
        sudo_password=args.sudo_password,
        skip_set_power=args.skip_set_power,
        sample_count=args.sample_count,
        strategy_names=strategy_names,
    )


if __name__ == "__main__":
    main()
