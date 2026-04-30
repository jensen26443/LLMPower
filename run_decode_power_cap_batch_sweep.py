#!/usr/bin/env python3
"""
Decode 阶段固定 power-cap batch sweep。

固定 prefill 前馈策略，只扫描 first token 之后的 decode power cap，
以完整 batch 的 J/output_token 作为主指标。
"""
import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from llm_inference import LLMInferencer
from load_generator import LoadGenerator
from monitor import PowerMonitor
from power_control import SudoKeepAlive, set_power_cap
from run_feedforward_evaluation import (
    QUERY_GROUPS,
    V2_PREFILL_BUCKETS,
    build_power_window_stats,
    build_service_extra_body,
    compute_kvb,
    get_prefill_power_for_total_tokens,
    percentile,
    select_subset_prompts,
)


DEFAULT_DECODE_POWER_CAPS = [150, 170, 190, 210, 230, 250, 300]
DEFAULT_OUTPUT_LENGTHS = [100]

RAW_FIELDNAMES = [
    "experiment_id",
    "query_count",
    "target_input_tokens",
    "actual_input_tokens",
    "output_length",
    "decode_power_cap_w",
    "batch_repeat",
    "prefill_power_limit",
    "power_change_count",
    "power_event_trace_json",
    "first_kvb",
    "num_requests",
    "actual_output_tokens",
    "total_energy_j",
    "energy_per_output_token_j",
    "avg_power_w",
    "peak_power_w",
    "avg_sm_clock_mhz",
    "avg_mem_clock_mhz",
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
]

AGG_FIELDNAMES = [
    "experiment_id",
    "query_count",
    "target_input_tokens",
    "actual_input_tokens",
    "output_length",
    "decode_power_cap_w",
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
    "first_kvb",
]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def compute_energy_per_output_token(total_energy_j: float, actual_output_tokens: int) -> float:
    if int(actual_output_tokens) <= 0:
        return 0.0
    return float(total_energy_j) / int(actual_output_tokens)


def estimate_request_tbt_ms(request_result: Dict, output_tokens: int) -> float:
    if output_tokens > 1 and request_result["ttft"] < request_result["e2e"]:
        return (request_result["e2e"] - request_result["ttft"]) / (output_tokens - 1)
    return float(request_result.get("tbt", 0.0))


def summarize_decode_batch_metrics(results: List[Dict]) -> Dict[str, float]:
    ttfts = [float(item["ttft"]) for item in results]
    output_tokens = [int(item.get("token_count", 0)) for item in results]
    tbts = [estimate_request_tbt_ms(item, output_tokens[index]) for index, item in enumerate(results)]
    e2es = [float(item["e2e"]) for item in results]
    return {
        "num_requests": len(results),
        "actual_output_tokens": sum(output_tokens),
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


def build_query_groups(query_counts: Sequence[int]) -> List[Dict]:
    known = {int(item["query_count"]): int(item["target_input_tokens"]) for item in QUERY_GROUPS}
    return [
        {
            "query_count": int(query_count),
            "target_input_tokens": known.get(int(query_count), max(1, int(query_count) * 128)),
        }
        for query_count in query_counts
    ]


def build_experiment_blocks(query_groups: Sequence[Dict],
                            output_lengths: Sequence[int],
                            decode_power_caps: Sequence[int]) -> List[Dict]:
    blocks = []
    for query_group in query_groups:
        for output_length in output_lengths:
            for decode_power in decode_power_caps:
                blocks.append({
                    "query_count": int(query_group["query_count"]),
                    "target_input_tokens": int(query_group.get("target_input_tokens", 0)),
                    "output_length": int(output_length),
                    "decode_power_cap_w": int(decode_power),
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


class FixedDecodePowerController:
    def __init__(self,
                 prefill_power: int,
                 decode_power: int,
                 set_power_callback: Callable,
                 prompt_token_counts: Optional[Sequence[int]] = None):
        self.prefill_power = int(prefill_power)
        self.decode_power = int(decode_power)
        self.set_power_callback = set_power_callback
        self.prompt_token_counts = [int(value) for value in prompt_token_counts] if prompt_token_counts else []
        self.generated_token_counts = [0] * len(self.prompt_token_counts)
        self.finished = [False] * len(self.prompt_token_counts)
        self.current_power_limit: Optional[int] = None
        self.prefill_power_limit: Optional[int] = None
        self.power_change_count = 0
        self.power_event_trace: List[Dict] = []
        self.first_kvb: Optional[float] = None

    def _apply_power(self, power: int, reason: str, wall_time: Optional[float] = None, kvb: Optional[float] = None):
        if self.current_power_limit == int(power):
            return False
        verify = reason == "prefill"
        try:
            applied = self.set_power_callback(int(power), verify=verify)
        except TypeError:
            applied = self.set_power_callback(int(power))
        if not applied:
            raise RuntimeError(f"Failed to set GPU power cap to {power}W")
        if self.current_power_limit is not None:
            self.power_change_count += 1
        self.current_power_limit = int(power)
        self.power_event_trace.append({
            "power_limit": int(power),
            "reason": reason,
            "wall_time": wall_time,
            "kvb": kvb,
        })
        return True

    def start(self) -> int:
        self.prefill_power_limit = self.prefill_power
        self._apply_power(self.prefill_power, reason="prefill")
        return self.prefill_power

    def _compute_event_kvb(self, event: Dict) -> Optional[float]:
        if event.get("kvb") is not None:
            return float(event["kvb"])
        if not self.prompt_token_counts:
            return None
        request_index = int(event.get("request_index", -1))
        if 0 <= request_index < len(self.generated_token_counts):
            self.generated_token_counts[request_index] = max(
                self.generated_token_counts[request_index],
                int(event.get("generated_tokens", 0)),
            )
            if event.get("event_type") == "finished":
                self.finished[request_index] = True
        kvb = compute_kvb(self.prompt_token_counts, self.generated_token_counts, self.finished)
        return kvb if kvb > 0 else None

    def handle_stream_event(self, event: Dict):
        event_type = event.get("event_type")
        if event_type not in {"first_token", "chunk", "finished"}:
            return
        kvb = self._compute_event_kvb(event)
        if self.first_kvb is None and kvb is not None:
            self.first_kvb = kvb
        self._apply_power(
            self.decode_power,
            reason="decode",
            wall_time=event.get("wall_time"),
            kvb=kvb,
        )


def aggregate_raw_rows(raw_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (
            row.get("experiment_id", ""),
            int(row["query_count"]),
            int(row.get("target_input_tokens", 0)),
            int(row["output_length"]),
            int(row["decode_power_cap_w"]),
        )
        grouped[key].append(row)

    aggregated = []
    for key, rows in sorted(grouped.items()):
        total_output_tokens = sum(int(float(item["actual_output_tokens"])) for item in rows)
        total_energy = sum(float(item["total_energy_j"]) for item in rows)
        aggregated.append({
            "experiment_id": key[0],
            "query_count": key[1],
            "target_input_tokens": key[2],
            "actual_input_tokens": statistics.mean(float(item["actual_input_tokens"]) for item in rows),
            "output_length": key[3],
            "decode_power_cap_w": key[4],
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
            "p50_ttft_ms": statistics.mean(float(item.get("p50_ttft_ms", 0.0)) for item in rows),
            "p95_ttft_ms": statistics.mean(float(item.get("p95_ttft_ms", 0.0)) for item in rows),
            "p99_ttft_ms": statistics.mean(float(item.get("p99_ttft_ms", 0.0)) for item in rows),
            "avg_tbt_ms": statistics.mean(float(item["avg_tbt_ms"]) for item in rows),
            "p50_tbt_ms": statistics.mean(float(item.get("p50_tbt_ms", 0.0)) for item in rows),
            "p95_tbt_ms": statistics.mean(float(item.get("p95_tbt_ms", 0.0)) for item in rows),
            "p99_tbt_ms": statistics.mean(float(item.get("p99_tbt_ms", 0.0)) for item in rows),
            "avg_e2e_ms": statistics.mean(float(item["avg_e2e_ms"]) for item in rows),
            "p50_e2e_ms": statistics.mean(float(item.get("p50_e2e_ms", 0.0)) for item in rows),
            "p95_e2e_ms": statistics.mean(float(item.get("p95_e2e_ms", 0.0)) for item in rows),
            "p99_e2e_ms": statistics.mean(float(item.get("p99_e2e_ms", 0.0)) for item in rows),
            "first_kvb": statistics.mean(float(item.get("first_kvb") or 0.0) for item in rows),
        })
    return aggregated


def build_prompt_sets(load_generator: LoadGenerator,
                      query_groups: Sequence[Dict],
                      output_lengths: Sequence[int],
                      repeats_per_cap: int,
                      warmup_batches: int,
                      monitor_warmup_batches: int,
                      queue_seed: int) -> Dict[Tuple[int, int], List[List[Dict]]]:
    import random

    prompt_sets = {}
    total_batches = int(repeats_per_cap) + int(warmup_batches) + int(monitor_warmup_batches)
    for query_group in query_groups:
        query_count = int(query_group["query_count"])
        target_tokens = int(query_group["target_input_tokens"])
        for output_length in output_lengths:
            rng = random.Random(int(queue_seed) + query_count * 1000 + int(output_length))
            batches = []
            for _ in range(total_batches):
                batch = select_subset_prompts(
                    load_generator=load_generator,
                    num_prompts=query_count,
                    target_tokens=target_tokens,
                )
                rng.shuffle(batch)
                batches.append(batch)
            prompt_sets[(query_count, int(output_length))] = batches
    return prompt_sets


def run_decode_power_cap_batch_sweep(output_dir: str,
                                     model_path: str,
                                     served_model_name: str,
                                     tokenizer_path: str,
                                     sharegpt_dir: str,
                                     base_url: str,
                                     query_counts: Sequence[int],
                                     output_lengths: Sequence[int],
                                     decode_power_caps: Sequence[int],
                                     repeats_per_cap: int,
                                     warmup_batches: int,
                                     monitor_warmup_batches: int,
                                     inter_batch_sec: float,
                                     queue_seed: int,
                                     sampling_seed: int,
                                     sudo_password: Optional[str],
                                     skip_set_power: bool):
    os.makedirs(output_dir, exist_ok=True)
    experiment_id = f"decode_power_cap_batch_sweep_{int(time.time())}"
    raw_path = os.path.join(output_dir, f"{experiment_id}_raw.csv")
    agg_path = os.path.join(output_dir, f"{experiment_id}_aggregated.csv")
    metadata_path = os.path.join(output_dir, f"{experiment_id}_metadata.json")
    progress_path = os.path.join(output_dir, f"{experiment_id}_progress.json")

    initialize_csv_file(raw_path, RAW_FIELDNAMES)
    initialize_csv_file(agg_path, AGG_FIELDNAMES)

    query_groups = build_query_groups(query_counts)
    blocks = build_experiment_blocks(query_groups, output_lengths, decode_power_caps)
    metadata = {
        "experiment_id": experiment_id,
        "query_groups": query_groups,
        "output_lengths": [int(value) for value in output_lengths],
        "decode_power_caps": [int(value) for value in decode_power_caps],
        "repeats_per_cap": int(repeats_per_cap),
        "warmup_batches": int(warmup_batches),
        "monitor_warmup_batches": int(monitor_warmup_batches),
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
        repeats_per_cap=repeats_per_cap,
        warmup_batches=warmup_batches,
        monitor_warmup_batches=monitor_warmup_batches,
        queue_seed=queue_seed,
    )

    def apply_power_cap(power: int, verify: bool = False) -> bool:
        if skip_set_power:
            return True
        return set_power_cap(power, sudo_password=sudo_password)

    keep_alive = None
    if not skip_set_power:
        keep_alive = SudoKeepAlive(interval_sec=60.0)
        if not keep_alive.start(sudo_password=sudo_password):
            raise RuntimeError("Failed to initialize sudo keepalive")

    completed_blocks = 0
    raw_rows: List[Dict] = []
    try:
        write_progress_file(progress_path, experiment_id, len(blocks), completed_blocks, "running", started_at=metadata["started_at"])
        for block in blocks:
            write_progress_file(
                progress_path,
                experiment_id,
                len(blocks),
                completed_blocks,
                "running",
                current_block=block,
                started_at=metadata["started_at"],
            )
            query_count = int(block["query_count"])
            output_length = int(block["output_length"])
            decode_power = int(block["decode_power_cap_w"])
            prefill_power = get_prefill_power_for_total_tokens(
                int(block["target_input_tokens"]),
                prefill_buckets=V2_PREFILL_BUCKETS,
            )
            batches = prompt_sets[(query_count, output_length)]
            warmup_slice = batches[:warmup_batches]
            monitor_warmup_slice = batches[warmup_batches:warmup_batches + monitor_warmup_batches]
            measurement_slice = batches[warmup_batches + monitor_warmup_batches:]
            extra_body = build_service_extra_body(output_length, sampling_seed)

            for warmup_batch in warmup_slice:
                controller = FixedDecodePowerController(prefill_power, decode_power, apply_power_cap)
                controller.start()
                inferencer.infer_concurrent(
                    [item["prompt"] for item in warmup_batch],
                    max_tokens=output_length,
                    temperature=0.0,
                    extra_body=extra_body,
                    stream_hook=controller.handle_stream_event,
                )
                time.sleep(float(inter_batch_sec))

            block_rows = []
            for batch_repeat, batch_prompts in enumerate(
                tqdm(measurement_slice, desc=f"decode={decode_power} q={query_count} l={output_length}", leave=False),
                start=1,
            ):
                for monitor_warmup_batch in monitor_warmup_slice if batch_repeat == 1 else []:
                    controller = FixedDecodePowerController(prefill_power, decode_power, apply_power_cap)
                    controller.start()
                    inferencer.infer_concurrent(
                        [item["prompt"] for item in monitor_warmup_batch],
                        max_tokens=output_length,
                        temperature=0.0,
                        extra_body=extra_body,
                        stream_hook=controller.handle_stream_event,
                    )
                    time.sleep(float(inter_batch_sec))

                prompt_token_counts = [int(item["prompt_tokens"]) for item in batch_prompts]
                controller = FixedDecodePowerController(
                    prefill_power=prefill_power,
                    decode_power=decode_power,
                    set_power_callback=apply_power_cap,
                    prompt_token_counts=prompt_token_counts,
                )
                controller.start()
                power_monitor = PowerMonitor(sample_interval=0.02)
                power_monitor.start()
                time.sleep(0.2)
                wall_start = time.time()
                results = inferencer.infer_concurrent(
                    [item["prompt"] for item in batch_prompts],
                    max_tokens=output_length,
                    temperature=0.0,
                    extra_body=extra_body,
                    stream_hook=controller.handle_stream_event,
                )
                wall_end = time.time()
                power_data = power_monitor.stop()
                power_stats = build_power_window_stats(wall_start, wall_end, power_data)
                metric_stats = summarize_decode_batch_metrics(results)
                energy_per_token = compute_energy_per_output_token(
                    power_stats["total_energy_j"],
                    int(metric_stats["actual_output_tokens"]),
                )
                row = {
                    "experiment_id": experiment_id,
                    "query_count": query_count,
                    "target_input_tokens": int(block["target_input_tokens"]),
                    "actual_input_tokens": sum(prompt_token_counts),
                    "output_length": output_length,
                    "decode_power_cap_w": decode_power,
                    "batch_repeat": batch_repeat,
                    "prefill_power_limit": prefill_power,
                    "power_change_count": controller.power_change_count,
                    "power_event_trace_json": json.dumps(controller.power_event_trace, ensure_ascii=False),
                    "first_kvb": controller.first_kvb or 0.0,
                    **metric_stats,
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
                experiment_id,
                len(blocks),
                completed_blocks,
                "running",
                last_completed_block=block,
                started_at=metadata["started_at"],
            )
            print(f"完成 q={query_count} output={output_length} decode={decode_power}W: {len(block_rows)} batches")
    except Exception as exc:
        write_progress_file(
            progress_path,
            experiment_id,
            len(blocks),
            completed_blocks,
            "failed",
            current_block=blocks[completed_blocks] if completed_blocks < len(blocks) else None,
            started_at=metadata["started_at"],
            error=str(exc),
        )
        raise
    finally:
        if keep_alive is not None:
            keep_alive.stop()

    metadata["finished_at"] = time.time()
    write_json_file(metadata_path, metadata)
    write_progress_file(progress_path, experiment_id, len(blocks), completed_blocks, "completed", started_at=metadata["started_at"])
    return {
        "raw_file": raw_path,
        "aggregated_file": agg_path,
        "metadata_file": metadata_path,
        "progress_file": progress_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run decode fixed power-cap batch sweep.")
    parser.add_argument("--output-dir", default="experiment_results/decode_power_cap_batch_sweep/smoke_q64_out100")
    parser.add_argument("--model-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--served-model-name", default="Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--tokenizer-path", default="./Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--sharegpt-dir", default="./input/ShareGPT")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--query-counts", default="64")
    parser.add_argument("--output-lengths", default="100")
    parser.add_argument("--decode-power-caps", default="150,170,190,210,230,250,300")
    parser.add_argument("--repeats-per-cap", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--monitor-warmup-batches", type=int, default=1)
    parser.add_argument("--inter-batch-sec", type=float, default=0.3)
    parser.add_argument("--queue-seed", type=int, default=20260426)
    parser.add_argument("--sampling-seed", type=int, default=20260426)
    parser.add_argument("--sudo-password", default=None)
    parser.add_argument("--skip-set-power", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = run_decode_power_cap_batch_sweep(
        output_dir=args.output_dir,
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        tokenizer_path=args.tokenizer_path,
        sharegpt_dir=args.sharegpt_dir,
        base_url=args.base_url,
        query_counts=parse_int_list(args.query_counts),
        output_lengths=parse_int_list(args.output_lengths),
        decode_power_caps=parse_int_list(args.decode_power_caps),
        repeats_per_cap=args.repeats_per_cap,
        warmup_batches=args.warmup_batches,
        monitor_warmup_batches=args.monitor_warmup_batches,
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
