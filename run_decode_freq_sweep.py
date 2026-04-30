#!/usr/bin/env python3
"""
Decode 阶段频率扫频实验。

使用少量代表性场景覆盖低/中/高 decode R 区间，在固定 decode 功率前馈下，
扫 `SM/MEM` 频率组合，选择适合 decode 阶段的频率档位。
"""
import argparse
import json
import os
import statistics
import threading
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
    probe_clock_capabilities,
    reset_hardware_profile,
)
from run_feedforward_evaluation import (
    QUERY_GROUPS,
    V2_PREFILL_BUCKETS,
    append_csv_rows,
    build_power_window_stats,
    build_service_extra_body,
    compute_kvb,
    get_prefill_power_for_total_tokens,
    initialize_csv_file,
    select_subset_prompts,
    summarize_request_metrics,
    write_json_file,
    write_progress_file,
)


REPRESENTATIVE_SCENARIOS = [
    {"query_count": 8, "output_length": 100, "bucket_name": "low_r"},
    {"query_count": 32, "output_length": 100, "bucket_name": "mid_r"},
    {"query_count": 128, "output_length": 200, "bucket_name": "high_r"},
]

DECODE_POWER_BUCKETS = [
    (1.0, 170),
    (2.0, 200),
    (float("inf"), 215),
]

STRATEGIES = [
    {"name": "baseline_decode_profile", "type": "baseline"},
    {"name": "decode_freq_sweep", "type": "sweep"},
]

RAW_FIELDNAMES = [
    "full_repeat",
    "strategy",
    "bucket_name",
    "query_count",
    "target_input_tokens",
    "output_length",
    "batch_repeat",
    "prefill_power_limit",
    "decode_power_scheme",
    "clock_profile_name",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "power_change_count",
    "clock_change_count",
    "power_event_trace_json",
    "avg_ttft_ms",
    "avg_tbt_ms",
    "avg_e2e_ms",
    "avg_power_w",
    "total_energy_j",
    "peak_power_w",
    "num_requests",
    "actual_output_tokens",
]

AGG_FIELDNAMES = [
    "full_repeat",
    "strategy",
    "bucket_name",
    "query_count",
    "target_input_tokens",
    "output_length",
    "prefill_power_limit",
    "decode_power_scheme",
    "clock_profile_name",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "num_samples",
    "avg_ttft_ms",
    "avg_tbt_ms",
    "avg_e2e_ms",
    "avg_energy_j",
    "avg_power_w",
]


def build_clock_profiles(capabilities: Dict) -> List[Dict]:
    profiles = []
    for index, pair in enumerate(capabilities.get("sampled_clock_pairs", []), start=1):
        profiles.append({
            "clock_profile_name": f"profile_{index}",
            "sm_clock_mhz": int(pair["graphics_mhz"]),
            "mem_clock_mhz": int(pair["memory_mhz"]),
        })
    return profiles


def get_decode_power_for_kvb(kvb: float) -> int:
    for threshold, power in DECODE_POWER_BUCKETS:
        if kvb <= threshold:
            return int(power)
    return int(DECODE_POWER_BUCKETS[-1][1])


def build_scenario_prompt_sets(load_generator: LoadGenerator,
                               repeats_per_batch: int,
                               warmup_batches: int,
                               monitor_warmup_batches: int,
                               queue_seed: int,
                               full_repeat: int) -> Dict[str, List[List[Dict]]]:
    rng = __import__("random").Random(queue_seed + full_repeat)
    target_tokens_by_query = {
        int(item["query_count"]): int(item["target_input_tokens"])
        for item in QUERY_GROUPS
    }
    total_batches = repeats_per_batch + warmup_batches + monitor_warmup_batches
    prompt_sets = {}
    for scenario in REPRESENTATIVE_SCENARIOS:
        batches = []
        for _ in range(total_batches):
            batch = select_subset_prompts(
                load_generator=load_generator,
                num_prompts=int(scenario["query_count"]),
                target_tokens=target_tokens_by_query[int(scenario["query_count"])],
            )
            rng.shuffle(batch)
            batches.append(batch)
        prompt_sets[scenario["bucket_name"]] = batches
    return prompt_sets


class DecodeSweepController:
    def __init__(self,
                 strategy: Dict,
                 prompt_token_counts: Sequence[int],
                 routing_input_tokens: int,
                 set_profile_callback,
                 clock_profile: Dict):
        self.strategy = strategy
        self.prompt_token_counts = [int(item) for item in prompt_token_counts]
        self.routing_input_tokens = int(routing_input_tokens)
        self.set_profile_callback = set_profile_callback
        self.clock_profile = clock_profile
        self.generated_token_counts = [0] * len(self.prompt_token_counts)
        self.finished = [False] * len(self.prompt_token_counts)
        self.prefill_power_limit = None
        self.current_power_limit = None
        self.power_change_count = 0
        self.clock_change_count = 0
        self.power_event_trace: List[Dict] = []
        self._lock = threading.Lock()

    def _apply(self, power_w: int, reason: str, wall_time: Optional[float] = None, kvb: Optional[float] = None):
        sm_mhz = None
        mem_mhz = None
        if self.strategy["type"] != "baseline" and reason.startswith("decode"):
            sm_mhz = self.clock_profile["sm_clock_mhz"]
            mem_mhz = self.clock_profile["mem_clock_mhz"]
        if not self.set_profile_callback(power_w, sm_mhz, mem_mhz):
            raise RuntimeError(f"Failed to apply decode sweep profile power={power_w}")
        if self.current_power_limit is not None and self.current_power_limit != power_w:
            self.power_change_count += 1
        if reason.startswith("decode") and (sm_mhz is not None or mem_mhz is not None):
            self.clock_change_count += 1
        self.current_power_limit = power_w
        self.power_event_trace.append({
            "power_limit": int(power_w),
            "reason": reason,
            "sm_clock_mhz": sm_mhz,
            "mem_clock_mhz": mem_mhz,
            "wall_time": wall_time,
            "kvb": kvb,
        })

    def start(self) -> int:
        with self._lock:
            power = 350 if self.strategy["type"] == "baseline" else get_prefill_power_for_total_tokens(
                self.routing_input_tokens,
                V2_PREFILL_BUCKETS,
            )
            self.prefill_power_limit = power
            self._apply(power, reason="prefill")
            return power

    def handle_stream_event(self, event: Dict):
        with self._lock:
            request_index = int(event["request_index"])
            generated_tokens = int(event.get("generated_tokens", 0))
            event_type = event["event_type"]
            wall_time = event.get("wall_time")
            if 0 <= request_index < len(self.generated_token_counts):
                self.generated_token_counts[request_index] = max(self.generated_token_counts[request_index], generated_tokens)
                if event_type == "finished":
                    self.finished[request_index] = True

            if event_type not in {"first_token", "chunk", "finished"}:
                return
            kvb = compute_kvb(self.prompt_token_counts, self.generated_token_counts, self.finished)
            if kvb <= 0:
                return
            power = 350 if self.strategy["type"] == "baseline" else get_decode_power_for_kvb(kvb)
            if power == self.current_power_limit and (self.strategy["type"] == "baseline" or event_type != "first_token"):
                return
            self._apply(power, reason="decode", wall_time=wall_time, kvb=kvb)


def aggregate_raw_rows(raw_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (
            row["full_repeat"],
            row["strategy"],
            row["bucket_name"],
            row["query_count"],
            row["target_input_tokens"],
            row["output_length"],
            row["prefill_power_limit"],
            row["decode_power_scheme"],
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
            "bucket_name": key[2],
            "query_count": key[3],
            "target_input_tokens": key[4],
            "output_length": key[5],
            "prefill_power_limit": key[6],
            "decode_power_scheme": key[7],
            "clock_profile_name": key[8],
            "sm_clock_mhz": key[9],
            "mem_clock_mhz": key[10],
            "num_samples": len(rows),
            "avg_ttft_ms": statistics.mean(float(item["avg_ttft_ms"]) for item in rows),
            "avg_tbt_ms": statistics.mean(float(item["avg_tbt_ms"]) for item in rows),
            "avg_e2e_ms": statistics.mean(float(item["avg_e2e_ms"]) for item in rows),
            "avg_energy_j": statistics.mean(float(item["total_energy_j"]) for item in rows),
            "avg_power_w": statistics.mean(float(item["avg_power_w"]) for item in rows),
        })
    return aggregated


def run_decode_freq_sweep(output_dir: str,
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
    experiment_id = f"decode_freq_sweep_{int(time.time())}"
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
    query_target_tokens = {
        int(item["query_count"]): int(item["target_input_tokens"])
        for item in QUERY_GROUPS
    }

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
        "representative_scenarios": REPRESENTATIVE_SCENARIOS,
        "decode_power_buckets": DECODE_POWER_BUCKETS,
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
            total_profile_blocks += len(REPRESENTATIVE_SCENARIOS)
        else:
            total_profile_blocks += len(REPRESENTATIVE_SCENARIOS) * max(1, len(clock_profiles))
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
            max_tokens=8,
            temperature=0.0,
            extra_body=build_service_extra_body(output_length=8, sampling_seed=sampling_seed),
        )

        for full_repeat in range(1, full_repeats + 1):
            prompt_sets = build_scenario_prompt_sets(
                load_generator=load_generator,
                repeats_per_batch=repeats_per_batch,
                warmup_batches=warmup_batches,
                monitor_warmup_batches=monitor_warmup_batches,
                queue_seed=queue_seed,
                full_repeat=full_repeat,
            )
            for strategy in strategies:
                for scenario in REPRESENTATIVE_SCENARIOS:
                    profiles = [{"clock_profile_name": "baseline_default", "sm_clock_mhz": None, "mem_clock_mhz": None}]
                    if strategy["type"] != "baseline":
                        profiles = clock_profiles or [{"clock_profile_name": "unsupported_default", "sm_clock_mhz": None, "mem_clock_mhz": None}]

                    for profile in profiles:
                        current_block = {
                            "full_repeat": full_repeat,
                            "strategy_name": strategy["name"],
                            "bucket_name": scenario["bucket_name"],
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
                        batches = prompt_sets[scenario["bucket_name"]]
                        warmup_slice = batches[:warmup_batches]
                        monitor_warmup_slice = batches[warmup_batches:warmup_batches + monitor_warmup_batches]
                        measurement_slice = batches[warmup_batches + monitor_warmup_batches:]
                        extra_body = build_service_extra_body(
                            output_length=int(scenario["output_length"]),
                            sampling_seed=sampling_seed,
                        )

                        for warmup_batch in warmup_slice:
                            inferencer.infer_concurrent(
                                [item["prompt"] for item in warmup_batch],
                                max_tokens=int(scenario["output_length"]),
                                temperature=0.0,
                                extra_body=extra_body,
                            )
                            time.sleep(inter_batch_sec)

                        block_rows = []
                        for batch_repeat, batch_prompts in enumerate(
                            tqdm(
                                measurement_slice,
                                desc=f"{strategy['name']} {scenario['bucket_name']} {profile['clock_profile_name']}",
                                leave=False,
                            ),
                            start=1,
                        ):
                            controller = DecodeSweepController(
                                strategy=strategy,
                                prompt_token_counts=[int(item["prompt_tokens"]) for item in batch_prompts],
                                routing_input_tokens=query_target_tokens[int(scenario["query_count"])],
                                set_profile_callback=apply_profile,
                                clock_profile=profile,
                            )
                            prefill_power_limit = controller.start()

                            for warmup_monitor_batch in monitor_warmup_slice if batch_repeat == 1 else []:
                                warmup_controller = DecodeSweepController(
                                    strategy=strategy,
                                    prompt_token_counts=[int(item["prompt_tokens"]) for item in warmup_monitor_batch],
                                    routing_input_tokens=query_target_tokens[int(scenario["query_count"])],
                                    set_profile_callback=apply_profile,
                                    clock_profile=profile,
                                )
                                warmup_controller.start()
                                inferencer.infer_concurrent(
                                    [item["prompt"] for item in warmup_monitor_batch],
                                    max_tokens=int(scenario["output_length"]),
                                    temperature=0.0,
                                    extra_body=extra_body,
                                )
                                time.sleep(inter_batch_sec)

                            power_monitor = PowerMonitor(sample_interval=0.02)
                            power_monitor.start()
                            time.sleep(0.2)
                            wall_start = time.time()
                            results = inferencer.infer_concurrent(
                                [item["prompt"] for item in batch_prompts],
                                max_tokens=int(scenario["output_length"]),
                                temperature=0.0,
                                extra_body=extra_body,
                                stream_hook=controller.handle_stream_event,
                            )
                            wall_end = time.time()
                            power_data = power_monitor.stop()
                            power_stats = build_power_window_stats(wall_start, wall_end, power_data)
                            metric_stats = summarize_request_metrics(results)
                            block_rows.append({
                                "full_repeat": full_repeat,
                                "strategy": strategy["name"],
                                "bucket_name": scenario["bucket_name"],
                                "query_count": int(scenario["query_count"]),
                                "target_input_tokens": query_target_tokens[int(scenario["query_count"])],
                                "output_length": int(scenario["output_length"]),
                                "batch_repeat": batch_repeat,
                                "prefill_power_limit": prefill_power_limit,
                                "decode_power_scheme": "170/200/215",
                                "clock_profile_name": profile["clock_profile_name"],
                                "sm_clock_mhz": profile["sm_clock_mhz"],
                                "mem_clock_mhz": profile["mem_clock_mhz"],
                                "power_change_count": controller.power_change_count,
                                "clock_change_count": controller.clock_change_count,
                                "power_event_trace_json": json.dumps(controller.power_event_trace, ensure_ascii=False),
                                **metric_stats,
                                **power_stats,
                            })
                            time.sleep(inter_batch_sec)

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
    parser = argparse.ArgumentParser(description="Run decode frequency sweep.")
    parser.add_argument("--output-dir", default="results_freq/decode_freq_sweep")
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

    run_decode_freq_sweep(
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
