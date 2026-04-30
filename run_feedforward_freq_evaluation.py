#!/usr/bin/env python3
"""
多旋钮前馈控制评估脚本。

在现有前馈功率控制基础上，引入 prefill / decode 两阶段的 `SM/MEM` 频率前馈，
对比 baseline_350w、ff_v2_recommended 和 ff_v3_freq_recommended。
"""
import argparse
import json
import os
import threading
import time
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
    AGG_FIELDNAMES as BASE_AGG_FIELDNAMES,
    OUTPUT_LENGTHS,
    QUERY_GROUPS,
    V2_PREFILL_BUCKETS,
    append_csv_rows,
    build_power_window_stats,
    build_query_group_prompt_sets,
    build_service_extra_body,
    compute_kvb,
    get_prefill_power_for_total_tokens,
    initialize_csv_file,
    percentile,
    summarize_request_metrics,
    write_json_file,
    write_progress_file,
)


DECODE_BUCKETS = [(1.0, 170), (2.0, 200), (float("inf"), 215)]

STRATEGIES = [
    {
        "name": "baseline_350w",
        "type": "baseline",
        "prefill_power": 350,
        "decode_scheme": "350",
    },
    {
        "name": "ff_v2_recommended",
        "type": "power_only",
        "prefill_buckets": V2_PREFILL_BUCKETS,
        "decode_scheme": "170/200/215",
    },
    {
        "name": "ff_v3_freq_recommended",
        "type": "power_and_clock",
        "prefill_buckets": V2_PREFILL_BUCKETS,
        "decode_scheme": "170/200/215+freq",
    },
]

RAW_FIELDNAMES = [
    field for field in [
        "full_repeat",
        "strategy",
        "query_count",
        "target_input_tokens",
        "actual_input_tokens",
        "output_length",
        "batch_repeat",
        "prefill_power_limit",
        "prefill_sm_clock_mhz",
        "prefill_mem_clock_mhz",
        "decode_scheme",
        "decode_sm_scheme",
        "decode_mem_scheme",
        "power_change_count",
        "clock_change_count",
        "power_event_trace_json",
        "clock_event_trace_json",
        "avg_ttft_ms",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "avg_tbt_ms",
        "p50_tbt_ms",
        "p95_tbt_ms",
        "p99_tbt_ms",
        "avg_e2e_ms",
        "p50_e2e_ms",
        "p95_e2e_ms",
        "p99_e2e_ms",
        "avg_power_w",
        "total_energy_j",
        "peak_power_w",
        "num_requests",
        "actual_output_tokens",
    ]
]

AGG_FIELDNAMES = list(BASE_AGG_FIELDNAMES) + [
    "avg_clock_change_count",
]


def get_decode_power_for_kvb(kvb: float) -> int:
    for threshold, power in DECODE_BUCKETS:
        if kvb <= threshold:
            return int(power)
    return int(DECODE_BUCKETS[-1][1])


def get_decode_bucket_name(kvb: float) -> str:
    if kvb <= 1.0:
        return "low_r"
    if kvb <= 2.0:
        return "mid_r"
    return "high_r"


def load_recommendation(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_prefill_lookup(recommendation: Dict) -> Dict[int, Dict]:
    lookup = {}
    for bucket_name, payload in recommendation.get("buckets", {}).items():
        if payload.get("status") != "ok":
            raise ValueError(f"Unsatisfied prefill recommendation: {bucket_name}")
        lookup[int(payload["power_w"])] = payload
    return lookup


def build_decode_lookup(recommendation: Dict) -> Dict[str, Dict]:
    lookup = {}
    for bucket_name, payload in recommendation.get("buckets", {}).items():
        if payload.get("status") != "ok":
            raise ValueError(f"Unsatisfied decode recommendation: {bucket_name}")
        lookup[str(bucket_name)] = payload
    return lookup


class FeedforwardFreqController:
    def __init__(self,
                 strategy: Dict,
                 prompt_token_counts: Sequence[int],
                 routing_input_tokens: int,
                 set_profile_callback,
                 prefill_lookup: Dict[int, Dict],
                 decode_lookup: Dict[str, Dict]):
        self.strategy = strategy
        self.prompt_token_counts = [int(item) for item in prompt_token_counts]
        self.routing_input_tokens = int(routing_input_tokens)
        self.set_profile_callback = set_profile_callback
        self.prefill_lookup = prefill_lookup
        self.decode_lookup = decode_lookup
        self.generated_token_counts = [0] * len(self.prompt_token_counts)
        self.finished = [False] * len(self.prompt_token_counts)
        self.prefill_power_limit = None
        self.prefill_sm_clock_mhz = None
        self.prefill_mem_clock_mhz = None
        self.current_power_limit = None
        self.current_sm_clock_mhz = None
        self.current_mem_clock_mhz = None
        self.power_change_count = 0
        self.clock_change_count = 0
        self.power_event_trace: List[Dict] = []
        self.clock_event_trace: List[Dict] = []
        self._lock = threading.Lock()

    def _apply(self,
               power_w: int,
               sm_mhz: Optional[int],
               mem_mhz: Optional[int],
               reason: str,
               wall_time: Optional[float] = None,
               kvb: Optional[float] = None):
        if not self.set_profile_callback(power_w, sm_mhz, mem_mhz):
            raise RuntimeError(f"Failed to apply hardware profile power={power_w}, sm={sm_mhz}, mem={mem_mhz}")
        if self.current_power_limit is not None and self.current_power_limit != power_w:
            self.power_change_count += 1
        if (
            self.current_power_limit is not None and
            (self.current_sm_clock_mhz != sm_mhz or self.current_mem_clock_mhz != mem_mhz)
        ):
            self.clock_change_count += 1
        self.current_power_limit = power_w
        self.current_sm_clock_mhz = sm_mhz
        self.current_mem_clock_mhz = mem_mhz
        self.power_event_trace.append({
            "reason": reason,
            "power_limit": int(power_w),
            "sm_clock_mhz": sm_mhz,
            "mem_clock_mhz": mem_mhz,
            "wall_time": wall_time,
            "kvb": kvb,
        })
        self.clock_event_trace.append({
            "reason": reason,
            "sm_clock_mhz": sm_mhz,
            "mem_clock_mhz": mem_mhz,
            "wall_time": wall_time,
            "kvb": kvb,
        })

    def start(self) -> int:
        with self._lock:
            if self.strategy["type"] == "baseline":
                power = 350
                sm_mhz = None
                mem_mhz = None
            else:
                power = get_prefill_power_for_total_tokens(self.routing_input_tokens, V2_PREFILL_BUCKETS)
                sm_mhz = None
                mem_mhz = None
                if self.strategy["type"] == "power_and_clock":
                    recommendation = self.prefill_lookup[power]
                    sm_mhz = recommendation["sm_clock_mhz"]
                    mem_mhz = recommendation["mem_clock_mhz"]
            self.prefill_power_limit = power
            self.prefill_sm_clock_mhz = sm_mhz
            self.prefill_mem_clock_mhz = mem_mhz
            self._apply(power, sm_mhz, mem_mhz, reason="prefill")
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

            if self.strategy["type"] == "baseline":
                power = 350
                sm_mhz = None
                mem_mhz = None
            else:
                power = get_decode_power_for_kvb(kvb)
                sm_mhz = None
                mem_mhz = None
                if self.strategy["type"] == "power_and_clock":
                    recommendation = self.decode_lookup[get_decode_bucket_name(kvb)]
                    sm_mhz = recommendation["sm_clock_mhz"]
                    mem_mhz = recommendation["mem_clock_mhz"]

            if (
                self.current_power_limit == power and
                self.current_sm_clock_mhz == sm_mhz and
                self.current_mem_clock_mhz == mem_mhz
            ):
                return
            self._apply(power, sm_mhz, mem_mhz, reason="decode", wall_time=wall_time, kvb=kvb)


def aggregate_raw_rows(raw_rows: List[Dict]) -> List[Dict]:
    from run_feedforward_evaluation import aggregate_raw_rows as aggregate_power_only

    base_rows = []
    for row in raw_rows:
        base_row = dict(row)
        base_row.pop("prefill_sm_clock_mhz", None)
        base_row.pop("prefill_mem_clock_mhz", None)
        base_row.pop("decode_sm_scheme", None)
        base_row.pop("decode_mem_scheme", None)
        base_row.pop("clock_change_count", None)
        base_row.pop("clock_event_trace_json", None)
        base_rows.append(base_row)

    aggregated = aggregate_power_only(base_rows)
    grouped_clock = {}
    for row in raw_rows:
        key = (
            row["full_repeat"],
            row["strategy"],
            row["query_count"],
            row["target_input_tokens"],
            row["output_length"],
            row["prefill_power_limit"],
            row["decode_scheme"],
        )
        grouped_clock.setdefault(key, []).append(row)

    for row in aggregated:
        key = (
            row["full_repeat"],
            row["strategy"],
            row["query_count"],
            row["target_input_tokens"],
            row["output_length"],
            row["prefill_power_limit"],
            row["decode_scheme"],
        )
        clock_rows = grouped_clock[key]
        row["avg_clock_change_count"] = statistics.mean(float(item["clock_change_count"]) for item in clock_rows)
    return aggregated


def run_feedforward_freq_evaluation(output_dir: str,
                                    model_path: str,
                                    served_model_name: str,
                                    tokenizer_path: str,
                                    sharegpt_dir: str,
                                    base_url: str,
                                    output_lengths: Sequence[int],
                                    repeats_per_batch: int,
                                    full_repeats: int,
                                    warmup_batches: int,
                                    monitor_warmup_batches: int,
                                    inter_batch_sec: float,
                                    queue_seed: int,
                                    sampling_seed: int,
                                    sudo_password: Optional[str],
                                    skip_set_power: bool,
                                    prefill_recommendation_path: str,
                                    decode_recommendation_path: str,
                                    strategy_names: Optional[Sequence[str]] = None):
    os.makedirs(output_dir, exist_ok=True)
    experiment_id = f"feedforward_freq_eval_{int(time.time())}"
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

    prefill_lookup = build_prefill_lookup(load_recommendation(prefill_recommendation_path))
    decode_lookup = build_decode_lookup(load_recommendation(decode_recommendation_path))
    clock_capabilities = probe_clock_capabilities(sample_count=6, min_sm_mhz=1000, min_mem_mhz=5000)
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
        "output_lengths": list(output_lengths),
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
        "prefill_recommendation_path": prefill_recommendation_path,
        "decode_recommendation_path": decode_recommendation_path,
        "clock_capability_json": clock_capabilities,
        "started_at": time.time(),
    }
    write_json_file(metadata_path, metadata)

    total_blocks = len(strategies) * len(QUERY_GROUPS) * len(output_lengths) * full_repeats
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
            output_order = list(output_lengths[full_repeat - 1:]) + list(output_lengths[:full_repeat - 1])

            for strategy in strategy_order:
                for query_group in QUERY_GROUPS:
                    query_count = int(query_group["query_count"])
                    batches = prompt_sets[query_count]
                    for output_length in output_order:
                        current_block = {
                            "full_repeat": full_repeat,
                            "strategy_name": strategy["name"],
                            "query_count": query_count,
                            "output_length": int(output_length),
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

                        warmup_slice = batches[:warmup_batches]
                        monitor_warmup_slice = batches[warmup_batches:warmup_batches + monitor_warmup_batches]
                        measurement_slice = batches[warmup_batches + monitor_warmup_batches:]

                        extra_body = build_service_extra_body(
                            output_length=int(output_length),
                            sampling_seed=sampling_seed,
                        )

                        for warmup_batch in warmup_slice:
                            inferencer.infer_concurrent(
                                [item["prompt"] for item in warmup_batch],
                                max_tokens=int(output_length),
                                temperature=0.0,
                                extra_body=extra_body,
                            )
                            time.sleep(inter_batch_sec)

                        block_rows = []
                        for batch_repeat, batch_prompts in enumerate(
                            tqdm(
                                measurement_slice,
                                desc=f"{strategy['name']} q={query_count} l={output_length}",
                                leave=False,
                            ),
                            start=1,
                        ):
                            controller = FeedforwardFreqController(
                                strategy=strategy,
                                prompt_token_counts=[int(item["prompt_tokens"]) for item in batch_prompts],
                                routing_input_tokens=int(query_group["target_input_tokens"]),
                                set_profile_callback=apply_profile,
                                prefill_lookup=prefill_lookup,
                                decode_lookup=decode_lookup,
                            )
                            prefill_power_limit = controller.start()

                            for warmup_monitor_batch in monitor_warmup_slice if batch_repeat == 1 else []:
                                warmup_controller = FeedforwardFreqController(
                                    strategy=strategy,
                                    prompt_token_counts=[int(item["prompt_tokens"]) for item in warmup_monitor_batch],
                                    routing_input_tokens=int(query_group["target_input_tokens"]),
                                    set_profile_callback=apply_profile,
                                    prefill_lookup=prefill_lookup,
                                    decode_lookup=decode_lookup,
                                )
                                warmup_controller.start()
                                inferencer.infer_concurrent(
                                    [item["prompt"] for item in warmup_monitor_batch],
                                    max_tokens=int(output_length),
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
                                max_tokens=int(output_length),
                                temperature=0.0,
                                extra_body=extra_body,
                                stream_hook=controller.handle_stream_event,
                            )
                            wall_end = time.time()
                            power_data = power_monitor.stop()
                            metric_stats = summarize_request_metrics(results)
                            power_stats = build_power_window_stats(wall_start, wall_end, power_data)
                            block_rows.append({
                                "full_repeat": full_repeat,
                                "strategy": strategy["name"],
                                "query_count": query_count,
                                "target_input_tokens": int(query_group["target_input_tokens"]),
                                "actual_input_tokens": sum(int(item["prompt_tokens"]) for item in batch_prompts),
                                "output_length": int(output_length),
                                "batch_repeat": batch_repeat,
                                "prefill_power_limit": prefill_power_limit,
                                "prefill_sm_clock_mhz": controller.prefill_sm_clock_mhz,
                                "prefill_mem_clock_mhz": controller.prefill_mem_clock_mhz,
                                "decode_scheme": strategy["decode_scheme"],
                                "decode_sm_scheme": json.dumps(decode_lookup, ensure_ascii=False) if strategy["type"] == "power_and_clock" else "",
                                "decode_mem_scheme": json.dumps(decode_lookup, ensure_ascii=False) if strategy["type"] == "power_and_clock" else "",
                                "power_change_count": controller.power_change_count,
                                "clock_change_count": controller.clock_change_count,
                                "power_event_trace_json": json.dumps(controller.power_event_trace, ensure_ascii=False),
                                "clock_event_trace_json": json.dumps(controller.clock_event_trace, ensure_ascii=False),
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
    parser = argparse.ArgumentParser(description="Run feedforward frequency evaluation.")
    parser.add_argument("--output-dir", default="results_freq/feedforward_freq_evaluation")
    parser.add_argument("--model-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--served-model-name", default="Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--tokenizer-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--sharegpt-dir", default="./filtered_prompts")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--output-lengths", default="100,200")
    parser.add_argument("--repeats-per-batch", type=int, default=10)
    parser.add_argument("--full-repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--monitor-warmup-batches", type=int, default=1)
    parser.add_argument("--inter-batch-sec", type=float, default=0.8)
    parser.add_argument("--queue-seed", type=int, default=20260329)
    parser.add_argument("--sampling-seed", type=int, default=20260329)
    parser.add_argument("--sudo-password", default=os.environ.get("SUDO_PASSWORD"))
    parser.add_argument("--skip-set-power", action="store_true")
    parser.add_argument("--prefill-recommendation-path", default="results_freq/prefill_freq_sweep_full/images/prefill_freq_recommendation.json")
    parser.add_argument("--decode-recommendation-path", default="results_freq/decode_freq_sweep_full/images/decode_freq_recommendation.json")
    parser.add_argument("--strategy-names", default=None)
    args = parser.parse_args()

    output_lengths = [int(item.strip()) for item in str(args.output_lengths).split(",") if item.strip()]
    strategy_names = None
    if args.strategy_names:
        strategy_names = [item.strip() for item in str(args.strategy_names).split(",") if item.strip()]

    run_feedforward_freq_evaluation(
        output_dir=args.output_dir,
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        tokenizer_path=args.tokenizer_path,
        sharegpt_dir=args.sharegpt_dir,
        base_url=args.base_url,
        output_lengths=output_lengths,
        repeats_per_batch=args.repeats_per_batch,
        full_repeats=args.full_repeats,
        warmup_batches=args.warmup_batches,
        monitor_warmup_batches=args.monitor_warmup_batches,
        inter_batch_sec=args.inter_batch_sec,
        queue_seed=args.queue_seed,
        sampling_seed=args.sampling_seed,
        sudo_password=args.sudo_password,
        skip_set_power=args.skip_set_power,
        prefill_recommendation_path=args.prefill_recommendation_path,
        decode_recommendation_path=args.decode_recommendation_path,
        strategy_names=strategy_names,
    )


if __name__ == "__main__":
    main()
