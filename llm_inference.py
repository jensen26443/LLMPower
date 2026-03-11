from vllm import LLM, SamplingParams
import time
import os
from typing import List, Dict


class LLMInferencer:
    def __init__(self, model_name: str = "./Qwen2.5-7B-Instruct-AWQ"):
        # 展开~为绝对路径
        model_name = os.path.expanduser(model_name)

        self.llm = LLM(
            model=model_name,
            quantization="awq",
            gpu_memory_utilization=0.85
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=100
        )

    def infer(self, prompts: List[str], max_tokens: int = None) -> List[Dict]:
        """执行推理，返回包含延迟指标的结果"""
        if max_tokens:
            self.sampling_params.max_tokens = max_tokens

        results = []
        for prompt in prompts:
            # 方法：分别测量TTFT和完整生成
            # TTFT: 使用max_tokens=1测量第一个token的时间
            # 完整生成: 测量总时间，然后TBT=(总时间-TTFT)/(token数-1)

            # 第一步：测量TTFT
            ttft = 0.0
            try:
                temp_params = SamplingParams(
                    temperature=self.sampling_params.temperature,
                    top_p=self.sampling_params.top_p,
                    max_tokens=1
                )
                ttft_start = time.time()
                self.llm.generate([prompt], temp_params, use_tqdm=False)
                ttft_end = time.time()
                ttft = (ttft_end - ttft_start) * 1000  # 转换为ms
            except Exception as e:
                print(f"警告: TTFT测量失败，使用估算值: {e}")
                ttft = 100.0  # 回退估算值

            # 第二步：完整生成并测量E2E
            full_start = time.time()
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            full_end = time.time()

            generated_text = ""
            token_count = 0
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text
                token_count = len(outputs[0].outputs[0].token_ids)

            e2e = (full_end - full_start) * 1000  # 转换为ms

            # 第三步：计算平均TBT
            avg_tbt = 0.0
            if token_count > 1:
                avg_tbt = (e2e - ttft) / (token_count - 1)
            elif token_count == 1:
                avg_tbt = 0.0  # 只有一个token时没有TBT

            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "token_count": token_count,
                "ttft": ttft,
                "tbt": avg_tbt,
                "e2e": e2e
            })

        return results


if __name__ == "__main__":
    inferencer = LLMInferencer()
    results = inferencer.infer(["你好"])
    print(f"推理结果: {results[0]['generated_text'][:100]}...")
