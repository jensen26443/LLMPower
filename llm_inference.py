from vllm import LLM, SamplingParams
import time
import os
from typing import List, Dict


class LLMInferencer:
    def __init__(self, model_name: str = "./Qwen2.5-7B-Instruct-AWQ", disable_prefix_caching: bool = True):
        # 展开~为绝对路径
        model_name = os.path.expanduser(model_name)

        # 禁用前缀缓存以防止KV cache命中导致TTFT测量失真
        self.llm = LLM(
            model=model_name,
            quantization="awq",
            gpu_memory_utilization=0.85,
            enable_prefix_caching=not disable_prefix_caching
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=100
        )

    def infer(self, prompts: List[str], max_tokens: int = None) -> List[Dict]:
        """执行推理，返回包含延迟指标的结果

        测量方法（vLLM 0.17.0 兼容）：
        - 优先尝试使用 output.metrics.first_token_time 获取准确 TTFT
        - 如不可用，回退到两次调用方法：
          1. max_tokens=1 测量 TTFT（预填充 + 1个token解码）
          2. 完整生成测量 E2E
        - TBT = (E2E - TTFT) / (token_count - 1)
        """
        if max_tokens:
            self.sampling_params.max_tokens = max_tokens

        results = []
        for prompt in prompts:
            # 第一步：完整生成获取 E2E 和输出
            full_start = time.time()
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            full_end = time.time()
            e2e = (full_end - full_start) * 1000

            # 提取生成结果
            generated_text = ""
            token_count = 0
            final_output = outputs[0] if outputs else None

            if final_output and len(final_output.outputs) > 0:
                generated_text = final_output.outputs[0].text
                token_count = len(final_output.outputs[0].token_ids)

            # 第二步：获取 TTFT
            ttft = 0.0
            use_metrics = False

            # 优先尝试使用 vLLM 的 metrics（如果可用）
            if final_output and hasattr(final_output, 'metrics'):
                metrics = final_output.metrics
                if hasattr(metrics, 'first_token_time'):
                    # first_token_time 是相对于请求开始的时间（秒）
                    ttft = metrics.first_token_time * 1000
                    use_metrics = True

            # 如果 metrics 不可用，使用两次调用方法
            if not use_metrics or ttft <= 0:
                # 用 max_tokens=1 测量 TTFT
                temp_params = SamplingParams(
                    temperature=self.sampling_params.temperature,
                    top_p=self.sampling_params.top_p,
                    max_tokens=1,
                )
                ttft_start = time.time()
                self.llm.generate([prompt], temp_params, use_tqdm=False)
                ttft_end = time.time()
                ttft = (ttft_end - ttft_start) * 1000

            # 第三步：计算 TBT
            avg_tbt = 0.0
            if token_count > 1:
                if ttft < e2e:
                    avg_tbt = (e2e - ttft) / (token_count - 1)
                else:
                    # 降级：如果 TTFT 测量有问题，用平均时间
                    avg_tbt = e2e / token_count

            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "token_count": token_count,
                "ttft": ttft,
                "tbt": avg_tbt,
                "e2e": e2e
            })

        return results


    def infer_prefill_only(self, prompts: List[str], max_tokens: int = 1) -> List[Dict]:
        """仅执行一次生成用于预填充建模。

        说明：
        - 为避免旧版 `infer` 中"TTFT测量 + 完整生成"导致的重复请求，本方法只发起一次 vLLM
          生成调用。
        - 在 `max_tokens=1` 时，端到端耗时可近似视作 TTFT（解码开销仅 1 token，且很小）。
        """
        temp_params = SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            max_tokens=max_tokens,
        )

        results = []
        for prompt in prompts:
            start = time.time()
            outputs = self.llm.generate([prompt], temp_params, use_tqdm=False)
            end = time.time()

            latency_ms = (end - start) * 1000
            generated_text = ""
            token_count = 0
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text
                token_count = len(outputs[0].outputs[0].token_ids)

            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "token_count": token_count,
                "ttft": latency_ms,
                "tbt": 0.0,
                "e2e": latency_ms,
            })

        return results


if __name__ == "__main__":
    inferencer = LLMInferencer()
    results = inferencer.infer(["你好"])
    print(f"推理结果: {results[0]['generated_text'][:100]}...")
