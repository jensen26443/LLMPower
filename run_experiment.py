import argparse
import csv
import os
import time
from tqdm import tqdm
from power_control import set_power_cap, get_power_cap
from llm_inference import LLMInferencer
from load_generator import LoadGenerator
from monitor import PowerMonitor

def run_single_experiment(power_cap: int, load_type: str = "mixed", request_count: int = 20,
                         concurrency: int = 1, output_dir: str = "results", skip_set_power: bool = False,
                         model_path: str = None, max_tokens: int = 100, sudo_password: str = None):
    """运行单次实验"""
    os.makedirs(output_dir, exist_ok=True)

    # 设置功率限制
    if not skip_set_power:
        print(f"设置功率限制为 {power_cap}W")
        if not set_power_cap(power_cap, sudo_password=sudo_password):
            print("设置功率失败，跳过本次实验")
            return None
    else:
        print(f"跳过功率设置，使用当前系统功率限制")

    actual_power_cap = get_power_cap()
    print(f"实际功率限制: {actual_power_cap}W")
    print("等待功率稳定20秒...")
    time.sleep(20)

    # 初始化组件
    if model_path:
        inferencer = LLMInferencer(model_name=model_path)
    else:
        inferencer = LLMInferencer()
    load_generator = LoadGenerator()
    monitor = PowerMonitor()

    # 生成负载
    prompts = load_generator.generate_load(load_type, request_count)

    # 预热GPU（5次请求）
    print("预热GPU...")
    for _ in range(5):
        inferencer.infer(["你好"], max_tokens=max_tokens)
    time.sleep(2)

    # 开始监测
    monitor.start()
    start_time = time.time()

    print("开始推理...")
    # 执行推理
    all_results = []
    if concurrency == 1:
        # 串行推理
        for prompt in tqdm(prompts, desc=f"功率{power_cap}W实验中"):
            result = inferencer.infer([prompt], max_tokens=max_tokens)[0]
            all_results.append(result)
    else:
        # 批量并行推理
        results = inferencer.infer(prompts, max_tokens=max_tokens)
        all_results.extend(results)

    # 停止监测
    end_time = time.time()
    power_data = monitor.stop()
    total_energy = monitor.calculate_total_energy()
    total_time = end_time - start_time

    print("保存数据...")
    # 保存结果
    if concurrency == 1:
        experiment_id = f"{power_cap}W_{load_type}_{request_count}q_{int(time.time())}"
    else:
        experiment_id = f"{power_cap}W_{load_type}_{concurrency}c_{int(time.time())}"

    # 保存推理结果
    with open(f"{output_dir}/{experiment_id}_inference.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "generated_text", "token_count", "ttft", "tbt", "e2e"])
        writer.writeheader()
        writer.writerows(all_results)

    # 保存功率数据
    monitor.save_to_csv(f"{output_dir}/{experiment_id}_power.csv")

    # 保存实验元数据
    avg_ttft = sum(r["ttft"] for r in all_results) / len(all_results) if all_results else 0
    avg_tbt = sum(r["tbt"] for r in all_results) / len(all_results) if all_results else 0
    avg_e2e = sum(r["e2e"] for r in all_results) / len(all_results) if all_results else 0
    total_tokens = sum(r["token_count"] for r in all_results) if all_results else 0
    throughput = total_tokens / total_time if total_time > 0 else 0  # tokens/s

    metadata = {
        "experiment_id": experiment_id,
        "power_cap_w": power_cap,
        "actual_power_cap_w": actual_power_cap,
        "load_type": load_type,
        "request_count": request_count,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "total_time_s": total_time,
        "total_energy_j": total_energy,
        "total_tokens": total_tokens,
        "throughput_tps": throughput,
        "avg_ttft_ms": avg_ttft,
        "avg_tbt_ms": avg_tbt,
        "avg_e2e_ms": avg_e2e,
        "edp": avg_e2e * total_energy  # 能耗延迟乘积
    }

    with open(f"{output_dir}/{experiment_id}_metadata.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=metadata.keys())
        writer.writeheader()
        writer.writerow(metadata)

    print(f"\n实验完成！")
    print(f"总耗时: {total_time:.2f}s")
    print(f"总能耗: {total_energy:.2f}J")
    print(f"吞吐率: {throughput:.2f} tokens/s")
    print(f"平均E2E延迟: {avg_e2e:.2f}ms")
    print(f"EDP: {metadata['edp']:.2f}")
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM推理功率控制实验")
    parser.add_argument("--power", type=int, required=True, help="功率限制W (RTX4080建议150-350W)")
    parser.add_argument("--load-type", type=str, default="mixed", choices=["short", "long", "mixed"], help="负载类型")
    parser.add_argument("--count", type=int, default=20, help="请求数量")
    parser.add_argument("--concurrency", type=int, default=1, help="并发度")
    parser.add_argument("--output-dir", type=str, default="results", help="结果保存目录")
    parser.add_argument("--skip-set-power", action="store_true", help="跳过设置功率步骤，使用当前系统功率")
    parser.add_argument("--model-path", type=str, default=None, help="模型路径（默认使用Qwen2.5-7B）")
    parser.add_argument("--max-tokens", type=int, default=100, help="最大生成token数量（默认100）")
    parser.add_argument("--sudo-password", type=str, default=None, help="sudo密码（用于自动设置功率限制）")
    parser.add_argument("--show-power-info", action="store_true", help="显示GPU功率信息并退出")
    args = parser.parse_args()

    if args.show_power_info:
        from power_control import get_gpu_name, get_power_cap, get_default_power_limit, get_max_power_limit, suggest_power_range
        print(f"GPU型号: {get_gpu_name() or '未知'}")
        print(f"当前功率限制: {get_power_cap()}W")
        default_power = get_default_power_limit()
        if default_power:
            print(f"默认功率限制: {default_power}W")
        max_power = get_max_power_limit()
        if max_power:
            print(f"最大功率限制: {max_power}W")
        power_range = suggest_power_range()
        print(f"\n建议功率范围: {power_range['min']}W - {power_range['max']}W")
        print(f"建议功率档位: {power_range['steps']}")
        exit(0)

    run_single_experiment(args.power, args.load_type, args.count, args.concurrency, args.output_dir, args.skip_set_power, args.model_path, args.max_tokens, args.sudo_password)
