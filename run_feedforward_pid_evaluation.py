#!/usr/bin/env python3
"""
前馈 + PID 控制评估脚本。

对比 baseline_350w、ff_v2_recommended 和 ff_v2_pid。
PID 首版采用：
- prefill: 批次间 TTFT 反馈
- decode: 批次内 TBT 反馈
"""
import argparse
import json
import math
import os
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from tqdm import tqdm

from llm_inference import LLMInferencer
from load_generator import LoadGenerator
from monitor import PowerMonitor
from power_control import SudoKeepAlive, set_power_cap
from run_feedforward_evaluation import (
    OUTPUT_LENGTHS,
    QUERY_GROUPS,
    V2_PREFILL_BUCKETS,
    append_csv_rows,
    build_power_window_stats,
    build_query_group_prompt_sets,
    build_service_extra_body,
    compute_kvb,
    get_decode_power_for_kvb,
    get_prefill_power_for_total_tokens,
    initialize_csv_file,
    summarize_request_metrics,
    wait_for_power_limit,
    write_json_file,
    write_progress_file,
)


DEFAULT_TARGETS_PATH = os.path.join(os.path.dirname(__file__), "feedforward_pid_targets.json")

LEGACY_PID_CONFIG = {
    "kp_prefill": 0.2,
    "kp_decode": 0.2,
    "ki": 0.002,
    "kd": 0.005,
    "interval_sec": 2.0,
    "delta_limit_w": 10.0,
    "max_step_w": 10.0,
    "ttft_budget_ratio": 1.05,
    "tbt_budget_ratio": 1.03,
    "power_min_w": 150,
    "power_max_w": 350,
}

PID_CONFIG = {
    "kp_prefill": 0.1,
    "kp_decode": 0.1,
    "ki": 0.0,
    "kd": 0.0,
    "interval_sec": 2.0,
    "delta_limit_w": 5.0,
    "max_step_w": 10.0,
    "ttft_budget_ratio": 1.05,
    "tbt_budget_ratio": 1.03,
    "power_min_w": 150,
    "power_max_w": 350,
    "deadband_ms": 1.0,
    "min_pid_samples": 4,
    "min_power_change_w": 5.0,
}

STRATEGIES = [
    {
        "name": "baseline_350w",
        "type": "baseline",
        "prefill_power": 350,
        "decode_buckets": [(float("inf"), 350)],
        "decode_scheme": "350",
        "pid_enabled": False,
    },
    {
        "name": "ff_v2_recommended",
        "type": "feedforward",
        "prefill_buckets": V2_PREFILL_BUCKETS,
        "decode_buckets": [(1.0, 170), (2.0, 200), (float("inf"), 215)],
        "decode_scheme": "170/200/215",
        "pid_enabled": False,
    },
    {
        "name": "ff_v2_pid",
        "type": "feedforward_pid",
        "prefill_buckets": V2_PREFILL_BUCKETS,
        "decode_buckets": [(1.0, 170), (2.0, 200), (float("inf"), 215)],
        "decode_scheme": "170/200/215+pid",
        "pid_enabled": True,
        "pid_prefill_enabled": True,
        "pid_config": LEGACY_PID_CONFIG,
    },
    {
        "name": "ff_v2_pid_stable",
        "type": "feedforward_pid",
        "prefill_buckets": V2_PREFILL_BUCKETS,
        "decode_buckets": [(1.0, 170), (2.0, 200), (float("inf"), 215)],
        "decode_scheme": "170/200/215+pid_stable",
        "pid_enabled": True,
        "pid_prefill_enabled": False,
        "pid_config": PID_CONFIG,
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
    "pid_enabled",
    "pid_update_count",
    "pid_prefill_delta_w",
    "pid_decode_delta_w",
    "pid_prefill_error",
    "pid_decode_error",
    "power_change_count",
    "power_event_trace_json",
    "pid_event_trace_json",
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
    "avg_sm_clock_mhz",
    "avg_mem_clock_mhz",
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
    "pid_enabled",
    "num_samples",
    "avg_ttft_ms",
    "avg_tbt_ms",
    "avg_e2e_ms",
    "avg_energy_j",
    "avg_power_w",
    "avg_sm_clock_mhz",
    "avg_mem_clock_mhz",
    "avg_power_change_count",
    "avg_pid_update_count",
    "avg_pid_prefill_delta_w",
    "avg_pid_decode_delta_w",
    "avg_pid_prefill_error",
    "avg_pid_decode_error",
]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def clamp_power_step(current_power: int, desired_power: int, max_step_w: int) -> int:
    delta = int(round(desired_power - current_power))
    if delta > max_step_w:
        return current_power + max_step_w
    if delta < -max_step_w:
        return current_power - max_step_w
    return int(round(desired_power))


def load_pid_targets(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Invalid PID target file: {path}")
    return payload


def get_pid_targets_for_query(targets: Dict[str, Dict[str, float]],
                              query_count: int,
                              output_length: int,
                              ttft_budget_ratio: float = PID_CONFIG["ttft_budget_ratio"],
                              tbt_budget_ratio: float = PID_CONFIG["tbt_budget_ratio"]) -> Dict[str, float]:
    key = f"{int(query_count)}/{int(output_length)}"
    if key not in targets:
        raise KeyError(f"Missing PID target for {key}")
    item = targets[key]
    ttft_baseline = float(item["ttft_baseline_ms"])
    tbt_baseline = float(item["tbt_baseline_ms"])
    ttft_source = "baseline"
    tbt_source = "baseline"
    for candidate in ("ttft_ff_ms", "ff_ttft_ms", "ttft_feedforward_ms", "feedforward_ttft_ms"):
        if candidate in item:
            ttft_baseline = float(item[candidate])
            ttft_source = "feedforward"
            break
    for candidate in ("tbt_ff_ms", "ff_tbt_ms", "tbt_feedforward_ms", "feedforward_tbt_ms"):
        if candidate in item:
            tbt_baseline = float(item[candidate])
            tbt_source = "feedforward"
            break
    target_source = "feedforward" if ttft_source == "feedforward" and tbt_source == "feedforward" else "baseline"
    return {
        "ttft_baseline_ms": ttft_baseline,
        "tbt_baseline_ms": tbt_baseline,
        "ttft_target_ms": ttft_baseline * ttft_budget_ratio,
        "tbt_target_ms": tbt_baseline * tbt_budget_ratio,
        "target_source": target_source,
    }


@dataclass
class PIDState:
    kp: float
    ki: float
    kd: float
    output_limit_w: float
    integral_limit: float = 5000.0
    integral: float = 0.0
    prev_error: Optional[float] = None
    delta_w: float = 0.0

    def update(self, actual_value: float, target_value: float) -> Dict[str, float]:
        error = float(actual_value) - float(target_value)
        self.integral = clamp(self.integral + error, -self.integral_limit, self.integral_limit)
        derivative = 0.0 if self.prev_error is None else error - self.prev_error
        self.prev_error = error
        raw_delta = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.delta_w = clamp(raw_delta, -self.output_limit_w, self.output_limit_w)
        return {
            "error": error,
            "delta_w": self.delta_w,
            "integral": self.integral,
            "derivative": derivative,
        }


class FeedforwardPIDController:
    def __init__(self,
                 strategy: Dict,
                 prompt_token_counts: Sequence[int],
                 total_input_tokens: int,
                 routing_input_tokens: int,
                 set_power_callback: Callable[[int, bool], bool],
                 target_metrics: Dict[str, float],
                 prefill_pid_state: Optional[PIDState],
                 pid_config: Dict[str, float],
                 pid_enabled: bool = False):
        self.strategy = strategy
        self.prompt_token_counts = [int(item) for item in prompt_token_counts]
        self.total_input_tokens = int(total_input_tokens)
        self.routing_input_tokens = int(routing_input_tokens)
        self.set_power_callback = set_power_callback
        self.target_metrics = target_metrics
        self.prefill_pid_state = prefill_pid_state
        self.pid_enabled = bool(pid_enabled)
        self.pid_config = pid_config

        self.generated_token_counts = [0] * len(self.prompt_token_counts)
        self.finished = [False] * len(self.prompt_token_counts)
        self.first_token_wall_times: List[Optional[float]] = [None] * len(self.prompt_token_counts)
        self.last_token_wall_times: List[Optional[float]] = [None] * len(self.prompt_token_counts)

        self.prefill_base_power_limit: Optional[int] = None
        self.prefill_power_limit: Optional[int] = None
        self.current_power_limit: Optional[int] = None
        self.current_decode_base_power: Optional[int] = None

        self.power_change_count = 0
        self.power_event_trace: List[Dict] = []
        self.pid_event_trace: List[Dict] = []
        self.pid_update_count = 0

        self.pid_prefill_delta_w = float(prefill_pid_state.delta_w) if (self.pid_enabled and prefill_pid_state) else 0.0
        self.pid_decode_delta_w = 0.0
        self.pid_prefill_error: Optional[float] = None
        self.pid_decode_error: Optional[float] = None

        self._lock = threading.Lock()
        self._last_decode_pid_wall_time: Optional[float] = None
        self.decode_pid_state = PIDState(
            kp=pid_config["kp_decode"],
            ki=pid_config["ki"],
            kd=pid_config["kd"],
            output_limit_w=pid_config["delta_limit_w"],
        )

    def _clip_power(self, desired_power: float) -> int:
        return int(round(clamp(
            desired_power,
            self.pid_config["power_min_w"],
            self.pid_config["power_max_w"],
        )))

    def _apply_power(self,
                     desired_power: float,
                     reason: str,
                     wall_time: Optional[float] = None,
                     kvb: Optional[float] = None,
                     base_power: Optional[float] = None,
                     pid_delta: Optional[float] = None,
                     pid_error: Optional[float] = None,
                     verify: bool = False) -> bool:
        desired = self._clip_power(desired_power)
        if self.current_power_limit is not None:
            min_change_w = float(self.pid_config.get("min_power_change_w", 0.0))
            if abs(float(desired) - float(self.current_power_limit)) < min_change_w:
                return False
        if self.current_power_limit is None:
            next_power = desired
        else:
            next_power = clamp_power_step(
                current_power=int(self.current_power_limit),
                desired_power=desired,
                max_step_w=int(self.pid_config["max_step_w"]),
            )
        if self.current_power_limit == next_power:
            return False
        if not self.set_power_callback(next_power, verify):
            raise RuntimeError(f"Failed to set GPU power cap to {next_power}W")
        if self.current_power_limit is not None:
            self.power_change_count += 1
        self.current_power_limit = next_power
        self.power_event_trace.append({
            "power_limit": int(next_power),
            "reason": reason,
            "wall_time": wall_time,
            "kvb": kvb,
            "base_power": base_power,
            "pid_delta_w": pid_delta,
            "pid_error": pid_error,
        })
        return True

    def start(self) -> int:
        with self._lock:
            if self.strategy["type"] == "baseline":
                base_power = int(self.strategy["prefill_power"])
            else:
                base_power = get_prefill_power_for_total_tokens(
                    self.routing_input_tokens,
                    prefill_buckets=self.strategy.get("prefill_buckets"),
                )
            self.prefill_base_power_limit = int(base_power)
            desired = float(base_power)
            if self.pid_enabled:
                desired += float(self.pid_prefill_delta_w)
            self._apply_power(
                desired,
                reason="prefill",
                base_power=base_power,
                pid_delta=self.pid_prefill_delta_w if self.pid_enabled else 0.0,
                verify=True,
            )
            self.prefill_power_limit = int(self.current_power_limit)
            return int(self.prefill_power_limit)

    def _estimate_current_avg_tbt_ms(self) -> Optional[float]:
        estimates = []
        for first_wall, last_wall, generated_tokens in zip(
            self.first_token_wall_times,
            self.last_token_wall_times,
            self.generated_token_counts,
        ):
            if first_wall is None or last_wall is None or int(generated_tokens) <= 1:
                continue
            estimates.append((float(last_wall) - float(first_wall)) * 1000.0 / (int(generated_tokens) - 1))
        if not estimates:
            return None
        return float(statistics.mean(estimates))

    def _count_decode_pid_samples(self) -> int:
        return sum(
            1
            for first_wall, last_wall, generated_tokens in zip(
                self.first_token_wall_times,
                self.last_token_wall_times,
                self.generated_token_counts,
            )
            if first_wall is not None
            and last_wall is not None
            and int(generated_tokens) > 1
        )

    def handle_stream_event(self, event: Dict):
        with self._lock:
            request_index = int(event["request_index"])
            event_type = event["event_type"]
            generated_tokens = int(event.get("generated_tokens", 0))
            wall_time = float(event.get("wall_time", time.time()))

            if 0 <= request_index < len(self.generated_token_counts):
                self.generated_token_counts[request_index] = max(self.generated_token_counts[request_index], generated_tokens)
                if event_type == "first_token" and self.first_token_wall_times[request_index] is None:
                    self.first_token_wall_times[request_index] = wall_time
                if event_type in {"first_token", "chunk", "finished"}:
                    self.last_token_wall_times[request_index] = wall_time
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

            base_decode_power = get_decode_power_for_kvb(self.strategy, kvb)
            self.current_decode_base_power = base_decode_power
            desired_power = float(base_decode_power) + (self.pid_decode_delta_w if self.pid_enabled else 0.0)
            self._apply_power(
                desired_power,
                reason="decode_feedforward",
                wall_time=wall_time,
                kvb=kvb,
                base_power=base_decode_power,
                pid_delta=self.pid_decode_delta_w if self.pid_enabled else 0.0,
            )

            if not self.pid_enabled:
                return

            if self._last_decode_pid_wall_time is None:
                self._last_decode_pid_wall_time = wall_time
                return
            if wall_time - self._last_decode_pid_wall_time < self.pid_config["interval_sec"]:
                return

            estimate_tbt = self._estimate_current_avg_tbt_ms()
            if estimate_tbt is None:
                self._last_decode_pid_wall_time = wall_time
                return
            if self._count_decode_pid_samples() < int(self.pid_config.get("min_pid_samples", 1)):
                self._last_decode_pid_wall_time = wall_time
                return
            error = float(estimate_tbt) - float(self.target_metrics["tbt_target_ms"])
            if abs(error) < float(self.pid_config.get("deadband_ms", 0.0)):
                self.pid_decode_error = error
                self._last_decode_pid_wall_time = wall_time
                return

            update = self.decode_pid_state.update(
                actual_value=estimate_tbt,
                target_value=float(self.target_metrics["tbt_target_ms"]),
            )
            self.pid_update_count += 1
            self.pid_decode_delta_w = float(update["delta_w"])
            self.pid_decode_error = float(update["error"])
            self.pid_event_trace.append({
                "stage": "decode",
                "wall_time": wall_time,
                "kvb": kvb,
                "base_power": base_decode_power,
                "actual_tbt_ms": estimate_tbt,
                "target_tbt_ms": float(self.target_metrics["tbt_target_ms"]),
                "pid_error": float(update["error"]),
                "pid_delta_w": float(update["delta_w"]),
            })
            desired_power = float(base_decode_power) + self.pid_decode_delta_w
            self._apply_power(
                desired_power,
                reason="pid_decode",
                wall_time=wall_time,
                kvb=kvb,
                base_power=base_decode_power,
                pid_delta=self.pid_decode_delta_w,
                pid_error=self.pid_decode_error,
            )
            self._last_decode_pid_wall_time = wall_time

    def finalize_batch(self, metric_stats: Dict[str, float]):
        if not self.pid_enabled or self.prefill_pid_state is None:
            return
        update = self.prefill_pid_state.update(
            actual_value=float(metric_stats["avg_ttft_ms"]),
            target_value=float(self.target_metrics["ttft_target_ms"]),
        )
        self.pid_update_count += 1
        self.pid_prefill_error = float(update["error"])
        self.pid_event_trace.append({
            "stage": "prefill",
            "base_power": self.prefill_base_power_limit,
            "applied_power": self.prefill_power_limit,
            "actual_ttft_ms": float(metric_stats["avg_ttft_ms"]),
            "target_ttft_ms": float(self.target_metrics["ttft_target_ms"]),
            "pid_error": float(update["error"]),
            "next_pid_delta_w": float(update["delta_w"]),
        })


def aggregate_raw_rows(raw_rows: List[Dict]) -> List[Dict]:
    def mean_or_zero(values: List[float]) -> float:
        return statistics.mean(values) if values else 0.0

    grouped = defaultdict(list)
    for row in raw_rows:
        key = (
            row["full_repeat"],
            row["strategy"],
            row["query_count"],
            row["target_input_tokens"],
            row["output_length"],
            row["decode_scheme"],
            row["pid_enabled"],
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
            "output_length": key[4],
            "prefill_power_limit": statistics.mean(float(item["prefill_power_limit"]) for item in rows),
            "decode_scheme": key[5],
            "pid_enabled": key[6],
            "num_samples": len(rows),
            "avg_ttft_ms": statistics.mean(float(item["avg_ttft_ms"]) for item in rows),
            "avg_tbt_ms": statistics.mean(float(item["avg_tbt_ms"]) for item in rows),
            "avg_e2e_ms": statistics.mean(float(item["avg_e2e_ms"]) for item in rows),
            "avg_energy_j": statistics.mean(float(item["total_energy_j"]) for item in rows),
            "avg_power_w": statistics.mean(float(item["avg_power_w"]) for item in rows),
            "avg_sm_clock_mhz": statistics.mean(float(item.get("avg_sm_clock_mhz", 0.0)) for item in rows),
            "avg_mem_clock_mhz": statistics.mean(float(item.get("avg_mem_clock_mhz", 0.0)) for item in rows),
            "avg_power_change_count": statistics.mean(float(item["power_change_count"]) for item in rows),
            "avg_pid_update_count": statistics.mean(float(item["pid_update_count"]) for item in rows),
            "avg_pid_prefill_delta_w": statistics.mean(float(item["pid_prefill_delta_w"]) for item in rows),
            "avg_pid_decode_delta_w": statistics.mean(float(item["pid_decode_delta_w"]) for item in rows),
            "avg_pid_prefill_error": mean_or_zero([float(item["pid_prefill_error"]) for item in rows if item["pid_prefill_error"] not in ("", None)]),
            "avg_pid_decode_error": mean_or_zero([float(item["pid_decode_error"]) for item in rows if item["pid_decode_error"] not in ("", None)]),
        })
    return aggregated


def run_feedforward_pid_evaluation(output_dir: str,
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
                                   pid_targets_path: str,
                                   strategy_names: Optional[Sequence[str]] = None):
    os.makedirs(output_dir, exist_ok=True)
    experiment_id = f"feedforward_pid_eval_{int(time.time())}"
    raw_path = os.path.join(output_dir, f"{experiment_id}_raw.csv")
    agg_path = os.path.join(output_dir, f"{experiment_id}_aggregated.csv")
    metadata_path = os.path.join(output_dir, f"{experiment_id}_metadata.json")
    progress_path = os.path.join(output_dir, f"{experiment_id}_progress.json")

    initialize_csv_file(raw_path, RAW_FIELDNAMES)
    initialize_csv_file(agg_path, AGG_FIELDNAMES)

    strategies = STRATEGIES
    if strategy_names:
        selected = {item.strip() for item in strategy_names if item and item.strip()}
        strategies = [item for item in STRATEGIES if item["name"] in selected]
        missing = sorted(selected - {item["name"] for item in strategies})
        if missing:
            raise ValueError(f"Unknown strategy names: {', '.join(missing)}")

    pid_targets = load_pid_targets(pid_targets_path)
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
        "pid_targets_path": pid_targets_path,
        "pid_config": {
            item["name"]: item.get("pid_config", PID_CONFIG)
            for item in strategies
            if item.get("pid_enabled")
        },
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
            for strategy in strategies:
                for query_group in QUERY_GROUPS:
                    query_count = int(query_group["query_count"])
                    batches = prompt_sets[query_count]
                    for output_length in output_lengths:
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

                        pid_config = strategy.get("pid_config") or PID_CONFIG
                        target_metrics = get_pid_targets_for_query(
                            pid_targets,
                            query_count=query_count,
                            output_length=int(output_length),
                            ttft_budget_ratio=pid_config["ttft_budget_ratio"],
                            tbt_budget_ratio=pid_config["tbt_budget_ratio"],
                        )
                        prefill_pid_state = None
                        if strategy.get("pid_enabled") and strategy.get("pid_prefill_enabled", True):
                            prefill_pid_state = PIDState(
                                kp=pid_config["kp_prefill"],
                                ki=pid_config["ki"],
                                kd=pid_config["kd"],
                                output_limit_w=pid_config["delta_limit_w"],
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
                            tqdm(measurement_slice, desc=f"{strategy['name']} q={query_count} l={output_length}", leave=False),
                            start=1,
                        ):
                            prompt_token_counts = [int(item["prompt_tokens"]) for item in batch_prompts]
                            total_input_tokens = sum(prompt_token_counts)
                            pid_enabled = bool(strategy.get("pid_enabled"))
                            controller = FeedforwardPIDController(
                                strategy=strategy,
                                prompt_token_counts=prompt_token_counts,
                                total_input_tokens=total_input_tokens,
                                routing_input_tokens=int(query_group["target_input_tokens"]),
                                set_power_callback=apply_power_cap,
                                target_metrics=target_metrics,
                                prefill_pid_state=prefill_pid_state,
                                pid_config=pid_config,
                                pid_enabled=pid_enabled,
                            )
                            controller.start()

                            for warmup_monitor_batch in monitor_warmup_slice if batch_repeat == 1 else []:
                                warmup_controller = FeedforwardPIDController(
                                    strategy=strategy,
                                    prompt_token_counts=[int(item["prompt_tokens"]) for item in warmup_monitor_batch],
                                    total_input_tokens=sum(int(item["prompt_tokens"]) for item in warmup_monitor_batch),
                                    routing_input_tokens=int(query_group["target_input_tokens"]),
                                    set_power_callback=apply_power_cap,
                                    target_metrics=target_metrics,
                                    prefill_pid_state=None,
                                    pid_config=pid_config,
                                    pid_enabled=False,
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
                            controller.finalize_batch(metric_stats)

                            row = {
                                "full_repeat": full_repeat,
                                "strategy": strategy["name"],
                                "query_count": query_count,
                                "target_input_tokens": int(query_group["target_input_tokens"]),
                                "actual_input_tokens": total_input_tokens,
                                "output_length": int(output_length),
                                "batch_repeat": batch_repeat,
                                "prefill_power_limit": controller.prefill_power_limit,
                                "decode_scheme": strategy["decode_scheme"],
                                "pid_enabled": int(pid_enabled),
                                "pid_update_count": controller.pid_update_count,
                                "pid_prefill_delta_w": controller.pid_prefill_delta_w,
                                "pid_decode_delta_w": controller.pid_decode_delta_w,
                                "pid_prefill_error": controller.pid_prefill_error if controller.pid_prefill_error is not None else "",
                                "pid_decode_error": controller.pid_decode_error if controller.pid_decode_error is not None else "",
                                "power_change_count": controller.power_change_count,
                                "power_event_trace_json": json.dumps(controller.power_event_trace, ensure_ascii=False),
                                "pid_event_trace_json": json.dumps(controller.pid_event_trace, ensure_ascii=False),
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
    parser = argparse.ArgumentParser(description="Run feedforward PID evaluation.")
    parser.add_argument("--output-dir", default="results_decode/feedforward_pid_evaluation")
    parser.add_argument("--model-path", default="./Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    parser.add_argument("--served-model-name", default="Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    parser.add_argument("--tokenizer-path", default="./Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
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
    parser.add_argument("--pid-targets-path", default=DEFAULT_TARGETS_PATH)
    parser.add_argument("--strategy-names", default="baseline_350w,ff_v2_recommended,ff_v2_pid_stable")
    return parser.parse_args()


def main():
    args = parse_args()
    output_lengths = [int(item.strip()) for item in args.output_lengths.split(",") if item.strip()]
    strategy_names = [item.strip() for item in args.strategy_names.split(",") if item.strip()]
    run_feedforward_pid_evaluation(
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
        pid_targets_path=args.pid_targets_path,
        strategy_names=strategy_names,
    )


if __name__ == "__main__":
    main()
