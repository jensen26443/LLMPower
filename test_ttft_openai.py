"""
测试使用 OpenAI 兼容 API 的 TTFT/TPOT/E2E 测量方法。

需要先启动 vLLM 服务：
bash start_vllm_server.sh
"""

import argparse
import statistics
from typing import Iterable, List

from llm_inference import LLMInferencer
from load_generator import LoadGenerator


def percentile(values: Iterable[float], p: float) -> float:
    """使用线性插值计算百分位数，尽量贴近 bench serve 的输出方式。"""
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def print_metric_summary(name: str, values: List[float]):
    """打印 mean/p50/p95/p99 汇总。"""
    print(f"{name}:")
    print(f"  mean: {statistics.mean(values):.2f} ms" if values else "  mean: 0.00 ms")
    print(f"  p50:  {percentile(values, 50):.2f} ms")
    print(f"  p95:  {percentile(values, 95):.2f} ms")
    print(f"  p99:  {percentile(values, 99):.2f} ms")


def build_prompts(load_generator: LoadGenerator, num_prompts: int, input_len: int) -> List[str]:
    """生成一批接近指定输入长度的唯一 prompt。"""
    prompts = []
    for _ in range(num_prompts):
        prompts.append(
            load_generator.generate_prompt_by_token_count(
                input_len,
                prefer_sharegpt=True,
                add_unique_prefix=True,
            )
        )
    return prompts


def main():
    parser = argparse.ArgumentParser(description="测试 OpenAI API 模式下的 TTFT/TPOT/E2E 测量")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="vLLM 服务地址")
    parser.add_argument("--model-name", type=str, default="Qwen2.5-7B-Instruct-AWQ", help="模型名称或路径")
    parser.add_argument("--warmup", type=int, default=5, help="预热请求数")
    parser.add_argument("--num-prompts", type=int, default=20, help="正式测试请求数")
    parser.add_argument("--input-len", type=int, default=32, help="目标输入 token 数")
    parser.add_argument("--max-tokens", type=int, default=50, help="最大输出 token 数")
    args = parser.parse_args()

    print("测试 OpenAI 兼容 API 的 TTFT/TPOT/E2E 测量")
    print("=" * 60)

    print("\n正在连接 vLLM 服务...")
    inferencer = LLMInferencer(
        model_name=args.model_name,
        served_model_name=args.model_name,
        use_service=True,
        base_url=args.base_url,
        start_server=False,
    )
    load_generator = LoadGenerator(tokenizer_name=args.model_name)

    print("\n生成测试 prompts...")
    warmup_prompts = build_prompts(load_generator, args.warmup, args.input_len)
    test_prompts = build_prompts(load_generator, args.num_prompts, args.input_len)

    if args.warmup > 0:
        print(f"\n开始预热 {args.warmup} 次...")
        inferencer.infer(warmup_prompts, max_tokens=args.max_tokens)

    print(f"\n开始正式测试，共 {args.num_prompts} 个请求...")
    print("-" * 60)
    results = inferencer.infer(test_prompts, max_tokens=args.max_tokens)

    ttfts = [res["ttft"] for res in results]
    tpots = [res["tpot"] for res in results]
    e2es = [res["e2e"] for res in results]
    token_counts = [res["token_count"] for res in results]

    print(f"请求数: {len(results)}")
    print(f"平均输出 token 数: {statistics.mean(token_counts):.2f}" if token_counts else "平均输出 token 数: 0.00")
    print_metric_summary("TTFT", ttfts)
    print_metric_summary("TPOT", tpots)
    print_metric_summary("E2E", e2es)

    print("\n前 3 个样本:")
    for i, res in enumerate(results[:3]):
        print(f"  Sample {i + 1}: tokens={res['token_count']}, ttft={res['ttft']:.2f} ms, "
              f"tpot={res['tpot']:.2f} ms, e2e={res['e2e']:.2f} ms")


if __name__ == "__main__":
    main()
