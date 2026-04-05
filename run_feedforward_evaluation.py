#!/usr/bin/env python3
"""
前馈控制与基线方案对比评估脚本。

实验按 query_count / output_length 组织，使用 ShareGPT prompt 构造真实请求，
prefill 阶段按总输入 token 数前馈设功率，decode 阶段按外部近似 KVB 前馈设功率。
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

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
    {"query_count": 96, "target_input_tokens": 11106},
    {"query_count": 128, "target_input_tokens": 22873},
]

OUTPUT_LENGTHS = [100, 200]

DEFAULT_PREFILL_BUCKETS = [
    (6054, 165),
    (11107, 175),
    (float("inf"), 185),
]

V2_PREFILL_BUCKETS = [
    (1043, 200),
    (4114, 220),
    (float("inf"), 260),
]

STRATEGIES = [
    {
        "name": "baseline_350w",
        "type": "baseline",
        "prefill_power": 350,
        "decode_buckets": [(float("inf"), 350)],
        "decode_scheme": "350",
    },
    {
        "name": "ff_idea5",
        "type": "feedforward",
        "prefill_buckets": DEFAULT_PREFILL_BUCKETS,
        "decode_buckets": [(1.0, 170), (2.0, 200), (3.0, 220), (float("inf"), 220)],
        "decode_scheme": "170/200/220/220",
    },
    {
        "name": "ff_optimized",
        "type": "feedforward",
        "prefill_buckets": DEFAULT_PREFILL_BUCKETS,
        "decode_buckets": [(1.0, 150), (2.0, 180), (3.0, 220), (float("inf"), 220)],
        "decode_scheme": "150/180/220/220",
    },
    {
        "name": "ff_v2_recommended",
        "type": "feedforward",
        "prefill_buckets": V2_PREFILL_BUCKETS,
        "decode_buckets": [(1.0, 170), (2.0, 200), (float("inf"), 215)],
        "decode_scheme": "170/200/215",
    },
]

RAW_FIELDNAMES = [
    "full_repeat",
    "strategy",
    "query_count",
    "target_input_tokens",
    "actual_input_tokens",
    "output_length",
    "batch_repeat",
    "prefill_power_limit",
    "decode_scheme",
    "power_change_count",
    "power_event_trace_json",
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

AGG_FIELDNAMES = [
    "full_repeat",
    "strategy",
    "query_count",
    "target_input_tokens",
    "actual_input_tokens",
    "output_length",
    "prefill_power_limit",
    "decode_scheme",
    "num_samples",
    "avg_ttft_ms",
    "avg_tbt_ms",
    "avg_e2e_ms",
    "avg_energy_j",
    "avg_power_w",
]


def get_prefill_power_for_total_tokens(total_tokens: int,
                                       prefill_buckets: Optional[Sequence[Sequence[float]]] = None) -> int:
    buckets = prefill_buckets or DEFAULT_PREFILL_BUCKETS
    for threshold, power in buckets:
        if total_tokens <= threshold:
            return int(power)
    return int(buckets[-1][1])


def get_decode_power_for_kvb(strategy: Dict, kvb: float) -> int:
    for threshold, power in strategy["decode_buckets"]:
        if kvb <= threshold:
            return int(power)
    return int(strategy["decode_buckets"][-1][1])


def compute_kvb(prompt_token_counts: Sequence[int],
                generated_token_counts: Sequence[int],
                finished: Sequence[bool]) -> float:
    active_blocks = []
    for prompt_tokens, generated_tokens, is_finished in zip(
        prompt_token_counts,
        generated_token_counts,
        finished,
    ):
        if is_finished:
            continue
        context_len = int(prompt_tokens) + max(0, int(generated_tokens))
        active_blocks.append(math.ceil(context_len / 16))
    if not active_blocks:
        return 0.0
    return sum(active_blocks) / len(active_blocks)


def validate_actual_power_limit(expected_power: int, actual_power: float, tolerance_w: float = 5.0) -> bool:
    if abs(float(actual_power) - float(expected_power)) > tolerance_w:
        raise RuntimeError(
            f"Power limit mismatch: expected {expected_power}W, got {actual_power:.1f}W"
        )
    return True


def wait_for_power_limit(expected_power: int,
                         timeout_sec: float = 3.0,
                         poll_interval_sec: float = 0.1,
                         tolerance_w: float = 5.0) -> float:
    deadline = time.time() + timeout_sec
    last_power = get_power_cap()
    while time.time() <= deadline:
        last_power = get_power_cap()
        if abs(float(last_power) - float(expected_power)) <= tolerance_w:
            return float(last_power)
        time.sleep(poll_interval_sec)
    raise RuntimeError(
        f"Power limit mismatch: expected {expected_power}W, got {last_power:.1f}W"
    )


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * ratio
    lower = math.floor(rank)
    upper = math.ceil(rank)
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
                            output_lengths: Sequence[int],
                            full_repeats: int) -> List[Dict]:
    blocks = []
    for full_repeat in range(1, full_repeats + 1):
        for strategy in strategies:
            for query_group in query_groups:
                for output_length in output_lengths:
                    blocks.append({
                        "full_repeat": full_repeat,
                        "strategy_name": strategy["name"],
                        "query_count": int(query_group["query_count"]),
                        "target_input_tokens": int(query_group.get("target_input_tokens", 0)),
                        "output_length": int(output_length),
                    })
    return blocks


def estimate_request_tbt_ms(request_result: Dict, output_tokens: int) -> float:
    if output_tokens > 1 and request_result["ttft"] < request_result["e2e"]:
        return (request_result["e2e"] - request_result["ttft"]) / (output_tokens - 1)
    return float(request_result.get("tbt", 0.0))


def summarize_request_metrics(results: List[Dict]) -> Dict[str, float]:
    ttfts = [float(item["ttft"]) for item in results]
    tbts = [estimate_request_tbt_ms(item, int(item.get("token_count", 0))) for item in results]
    e2es = [float(item["e2e"]) for item in results]
    output_tokens = [int(item.get("token_count", 0)) for item in results]
    return {
        "num_requests": len(results),
        "actual_output_tokens": int(round(statistics.mean(output_tokens))) if output_tokens else 0,
        "avg_ttft_ms": statistics.mean(ttfts) if ttfts else 0.0,
        "p50_ttft_ms": percentile(ttfts, 0.50),
        "p95_ttft_ms": percentile(ttfts, 0.95),
        "p99_ttft_ms": percentile(ttfts, 0.99),
        "avg_tbt_ms": statistics.mean(tbts) if tbts else 0.0,
        "p50_tbt_ms": percentile(tbts, 0.50),
        "p95_tbt_ms": percentile(tbts, 0.95),
        "p99_tbt_ms": percentile(tbts, 0.99),
        "avg_e2e_ms": statistics.mean(e2es) if e2es else 0.0,
        "p50_e2e_ms": percentile(e2es, 0.50),
        "p95_e2e_ms": percentile(e2es, 0.95),
        "p99_e2e_ms": percentile(e2es, 0.99),
    }


def aggregate_raw_rows(raw_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
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
        grouped[key].append(row)

    aggregated = []
    for key, rows in sorted(grouped.items()):
        avg_input_tokens = statistics.mean(float(item["actual_input_tokens"]) for item in rows)
        aggregated.append({
            "full_repeat": key[0],
            "strategy": key[1],
            "query_count": key[2],
            "target_input_tokens": key[3],
            "actual_input_tokens": avg_input_tokens,
            "output_length": key[4],
            "prefill_power_limit": key[5],
            "decode_scheme": key[6],
            "num_samples": len(rows),
            "avg_ttft_ms": statistics.mean(float(item["avg_ttft_ms"]) for item in rows),
            "avg_tbt_ms": statistics.mean(float(item["avg_tbt_ms"]) for item in rows),
            "avg_e2e_ms": statistics.mean(float(item["avg_e2e_ms"]) for item in rows),
            "avg_energy_j": statistics.mean(float(item["total_energy_j"]) for item in rows),
            "avg_power_w": statistics.mean(float(item["avg_power_w"]) for item in rows),
        })
    return aggregated


def select_subset_prompts(load_generator: LoadGenerator,
                          num_prompts: int,
                          target_tokens: int) -> List[Dict]:
    avg_per_prompt = max(1, target_tokens // num_prompts)
    prompts = []
    for _ in range(num_prompts):
        variation = random.randint(-max(1, avg_per_prompt // 10), max(1, avg_per_prompt // 10))
        target_length = max(1, avg_per_prompt + variation)
        prompt = load_generator.generate_prompt_by_token_count(
            target_length,
            prefer_sharegpt=True,
            add_unique_prefix=True,
        )
        actual_tokens = load_generator.count_tokens(prompt)
        prompts.append({
            "prompt": prompt,
            "prompt_tokens": actual_tokens,
        })
    return prompts


def build_query_group_prompt_sets(load_generator: LoadGenerator,
                                  query_groups: Sequence[Dict],
                                  repeats_per_batch: int,
                                  warmup_batches: int,
                                  monitor_warmup_batches: int,
                                  queue_seed: int,
                                  full_repeat: int) -> Dict[int, List[List[Dict]]]:
    prompt_sets = {}
    total_batches = repeats_per_batch + warmup_batches + monitor_warmup_batches
    rng = random.Random(queue_seed + full_repeat)

    for query_group in query_groups:
        prompts = []
        for batch_repeat in range(total_batches):
            batch_prompts = select_subset_prompts(
                load_generator=load_generator,
                num_prompts=int(query_group["query_count"]),
                target_tokens=int(query_group["target_input_tokens"]),
            )
            rng.shuffle(batch_prompts)
            prompts.append(batch_prompts)
        prompt_sets[int(query_group["query_count"])] = prompts
    return prompt_sets


def build_service_extra_body(output_length: int, sampling_seed: int) -> Dict:
    return {
        "min_tokens": output_length,
        "ignore_eos": True,
        "top_p": 1.0,
        "seed": sampling_seed,
    }


def rotate_items(items: Sequence, offset: int) -> List:
    if not items:
        return []
    offset = offset % len(items)
    return list(items[offset:]) + list(items[:offset])


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


class FeedforwardController:
    def __init__(self,
                 strategy: Dict,
                 prompt_token_counts: Sequence[int],
                 total_input_tokens: int,
                 routing_input_tokens: Optional[int],
                 set_power_callback: Callable[[int], bool]):
        self.strategy = strategy
        self.prompt_token_counts = [int(value) for value in prompt_token_counts]
        self.total_input_tokens = int(total_input_tokens)
        self.routing_input_tokens = int(routing_input_tokens if routing_input_tokens is not None else total_input_tokens)
        self.set_power_callback = set_power_callback
        self.generated_token_counts = [0] * len(self.prompt_token_counts)
        self.finished = [False] * len(self.prompt_token_counts)
        self._lock = threading.Lock()
        self.prefill_power_limit: Optional[int] = None
        self.current_power_limit: Optional[int] = None
        self.power_change_count = 0
        self.power_event_trace: List[Dict] = []

    def _apply_power(self, power: int, reason: str, wall_time: Optional[float] = None, kvb: Optional[float] = None):
        if self.current_power_limit == power:
            return False
        verify = reason == "prefill"
        try:
            applied = self.set_power_callback(power, verify=verify)
        except TypeError:
            applied = self.set_power_callback(power)
        if not applied:
            raise RuntimeError(f"Failed to set GPU power cap to {power}W")
        if self.current_power_limit is not None:
            self.power_change_count += 1
        self.current_power_limit = power
        self.power_event_trace.append({
            "power_limit": int(power),
            "reason": reason,
            "wall_time": wall_time,
            "kvb": kvb,
        })
        return True

    def start(self) -> int:
        with self._lock:
            if self.strategy["type"] == "baseline":
                power = int(self.strategy["prefill_power"])
            else:
                power = get_prefill_power_for_total_tokens(
                    self.routing_input_tokens,
                    prefill_buckets=self.strategy.get("prefill_buckets"),
                )
            self.prefill_power_limit = power
            self._apply_power(power, reason="prefill")
            return power

    def handle_stream_event(self, event: Dict):
        with self._lock:
            request_index = int(event["request_index"])
            event_type = event["event_type"]
            generated_tokens = int(event.get("generated_tokens", 0))
            wall_time = event.get("wall_time")

            if 0 <= request_index < len(self.generated_token_counts):
                self.generated_token_counts[request_index] = max(
                    self.generated_token_counts[request_index],
                    generated_tokens,
                )
                if event_type == "finished":
                    self.finished[request_index] = True

            if self.strategy["type"] == "baseline":
                return
            if event_type not in {"first_token", "chunk", "finished"}:
                return

            kvb = compute_kvb(
                prompt_token_counts=self.prompt_token_counts,
                generated_token_counts=self.generated_token_counts,
                finished=self.finished,
            )
            if kvb <= 0:
                return
            next_power = get_decode_power_for_kvb(self.strategy, kvb)
            self._apply_power(next_power, reason="decode", wall_time=wall_time, kvb=kvb)


def run_feedforward_evaluation(output_dir: str,
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
                               strategy_names: Optional[Sequence[str]] = None,
                               only_strategy: Optional[str] = None):
    os.makedirs(output_dir, exist_ok=True)
    experiment_id = f"feedforward_eval_{int(time.time())}"
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
    if only_strategy:
        strategies = [item for item in STRATEGIES if item["name"] == only_strategy]
        if not strategies:
            raise ValueError(f"Unknown strategy: {only_strategy}")

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
        "started_at": time.time(),
    }
    write_json_file(metadata_path, metadata)

    total_blocks = len(strategies) * len(QUERY_GROUPS) * len(output_lengths) * full_repeats
    completed_blocks = 0
    raw_rows: List[Dict] = []

    def apply_power_cap(power: int, verify: bool = False) -> bool:
        if skip_set_power:
            return True
        if not set_power_cap(power, sudo_password=sudo_password):
            return False
        if verify:
            wait_for_power_limit(power)
        return True

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
            strategy_order = rotate_items(strategies, full_repeat - 1)
            output_order = rotate_items(list(output_lengths), full_repeat - 1)

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

                        block_rows = []
                        for warmup_batch in warmup_slice:
                            inferencer.infer_concurrent(
                                [item["prompt"] for item in warmup_batch],
                                max_tokens=int(output_length),
                                temperature=0.0,
                                extra_body=extra_body,
                            )
                            time.sleep(inter_batch_sec)

                        for batch_repeat, batch_prompts in enumerate(
                            tqdm(measurement_slice, desc=f"{strategy['name']} q={query_count} l={output_length}", leave=False),
                            start=1,
                        ):
                            prompt_token_counts = [int(item["prompt_tokens"]) for item in batch_prompts]
                            total_input_tokens = sum(prompt_token_counts)
                            controller = FeedforwardController(
                                strategy=strategy,
                                prompt_token_counts=prompt_token_counts,
                                total_input_tokens=total_input_tokens,
                                routing_input_tokens=int(query_group["target_input_tokens"]),
                                set_power_callback=apply_power_cap,
                            )
                            prefill_power_limit = controller.start()

                            for warmup_monitor_batch in monitor_warmup_slice if batch_repeat == 1 else []:
                                warmup_controller = FeedforwardController(
                                    strategy=strategy,
                                    prompt_token_counts=[int(item["prompt_tokens"]) for item in warmup_monitor_batch],
                                    total_input_tokens=sum(int(item["prompt_tokens"]) for item in warmup_monitor_batch),
                                    routing_input_tokens=int(query_group["target_input_tokens"]),
                                    set_power_callback=apply_power_cap,
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
                            power_stats = build_power_window_stats(wall_start, wall_end, power_data)
                            metric_stats = summarize_request_metrics(results)

                            row = {
                                "full_repeat": full_repeat,
                                "strategy": strategy["name"],
                                "query_count": query_count,
                                "target_input_tokens": int(query_group["target_input_tokens"]),
                                "actual_input_tokens": total_input_tokens,
                                "output_length": int(output_length),
                                "batch_repeat": batch_repeat,
                                "prefill_power_limit": prefill_power_limit,
                                "decode_scheme": strategy["decode_scheme"],
                                "power_change_count": controller.power_change_count,
                                "power_event_trace_json": json.dumps(controller.power_event_trace, ensure_ascii=False),
                                **metric_stats,
                                **power_stats,
                            }
                            block_rows.append(row)
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
    parser = argparse.ArgumentParser(description="Run feedforward power-control evaluation.")
    parser.add_argument("--output-dir", default="results_decode/feedforward_evaluation")
    parser.add_argument("--model-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--served-model-name", default="Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--tokenizer-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--sharegpt-dir", default="./input/ShareGPT")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--output-lengths", default="100,200")
    parser.add_argument("--repeats-per-batch", type=int, default=10)
    parser.add_argument("--full-repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--monitor-warmup-batches", type=int, default=1)
    parser.add_argument("--inter-batch-sec", type=float, default=0.3)
    parser.add_argument("--queue-seed", type=int, default=20260401)
    parser.add_argument("--sampling-seed", type=int, default=20260401)
    parser.add_argument("--sudo-password", default=None)
    parser.add_argument("--skip-set-power", action="store_true")
    parser.add_argument("--strategy-names", default=None)
    parser.add_argument("--only-strategy", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_lengths = [int(item.strip()) for item in args.output_lengths.split(",") if item.strip()]
    run_feedforward_evaluation(
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
        strategy_names=[item.strip() for item in args.strategy_names.split(",")] if args.strategy_names else None,
        only_strategy=args.only_strategy,
    )


if __name__ == "__main__":
    main()
