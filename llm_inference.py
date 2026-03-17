
import time
import os
import subprocess
import atexit
from typing import List, Dict, Optional

# 尝试导入 vLLM 离线模式
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# 尝试导入 OpenAI 客户端
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMInferencer:
    def __init__(
        self,
        model_name: str = "./Qwen2.5-7B-Instruct-AWQ",
        use_service: bool = False,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        start_server: bool = False,
        gpu_memory_utilization: float = 0.85,
        disable_prefix_caching: bool = True
    ):
        """
        初始化推理器，支持两种模式：

        1. 离线模式 (use_service=False, 默认): 使用 vLLM.LLM.generate()
        2. 服务模式 (use_service=True): 使用 OpenAI 兼容 API 连接 vLLM 服务

        Args:
            model_name: 模型路径或名称
            use_service: 是否使用服务模式
            base_url: vLLM 服务地址（服务模式）
            api_key: API 密钥（vLLM 默认 "EMPTY"）
            start_server: 是否自动启动 vLLM 服务（服务模式）
            gpu_memory_utilization: GPU 显存利用率
            disable_prefix_caching: 是否禁用前缀缓存
        """
        self.model_name = model_name
        self.use_service = use_service
        self.client = None
        self.llm = None
        self.sampling_params = None
        self.server_process: Optional[subprocess.Popen] = None

        if use_service:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI client not available. Install with: pip install openai")
            self._init_service_mode(
                base_url=base_url,
                api_key=api_key,
                start_server=start_server,
                gpu_memory_utilization=gpu_memory_utilization,
                disable_prefix_caching=disable_prefix_caching
            )
        else:
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM not available. Install with: pip install vllm==0.17.0")
            self._init_offline_mode(
                gpu_memory_utilization=gpu_memory_utilization,
                disable_prefix_caching=disable_prefix_caching
            )

    def _init_offline_mode(self, gpu_memory_utilization: float, disable_prefix_caching: bool):
        """初始化离线模式"""
        model_name = os.path.expanduser(self.model_name)
        self.llm = LLM(
            model=model_name,
            quantization="awq",
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=not disable_prefix_caching
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=100
        )

    def _init_service_mode(
        self,
        base_url: str,
        api_key: str,
        start_server: bool,
        gpu_memory_utilization: float,
        disable_prefix_caching: bool
    ):
        """初始化服务模式"""
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        if start_server:
            self._start_vllm_server(
                self.model_name,
                gpu_memory_utilization,
                disable_prefix_caching
            )

    def _start_vllm_server(
        self,
        model_path: str,
        gpu_memory_utilization: float,
        disable_prefix_caching: bool
    ):
        """自动启动 vLLM API 服务"""
        model_path = os.path.expanduser(model_path)

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--quantization", "awq",
            "--gpu-memory-utilization", str(gpu_memory_utilization),
        ]

        if disable_prefix_caching:
            cmd.append("--disable-prefix-caching")

        print(f"正在启动 vLLM 服务: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(cmd)

        # 等待服务启动
        print("等待 vLLM 服务启动...")
        time.sleep(30)

        # 注册退出时关闭服务
        atexit.register(self._stop_server)

    def _stop_server(self):
        """停止 vLLM 服务"""
        if self.server_process:
            print("正在停止 vLLM 服务...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            print("vLLM 服务已停止")

    def infer(self, prompts: List[str], max_tokens: int = None, temperature: float = 0.7) -&gt; List[Dict]:
        """
        执行推理，返回包含延迟指标的结果

        离线模式：使用两次调用方法测量 TTFT/TBT
        服务模式：使用 OpenAI 兼容 API 的流式输出准确测量 TTFT/TBT
        """
        if self.use_service:
            return self._infer_service(prompts, max_tokens, temperature)
        else:
            return self._infer_offline(prompts, max_tokens, temperature)

    def _infer_offline(self, prompts: List[str], max_tokens: int = None, temperature: float = 0.7) -&gt; List[Dict]:
        """离线模式推理：两次调用方法"""
        if max_tokens:
            self.sampling_params.max_tokens = max_tokens
        self.sampling_params.temperature = temperature

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

            if final_output and len(final_output.outputs) &gt; 0:
                generated_text = final_output.outputs[0].text
                token_count = len(final_output.outputs[0].token_ids)

            # 第二步：获取 TTFT - 使用 max_tokens=1 测量
            ttft = 0.0
            use_metrics = False

            # 优先尝试使用 vLLM 的 metrics（如果可用）
            if final_output and hasattr(final_output, 'metrics'):
                metrics = final_output.metrics
                if hasattr(metrics, 'first_token_time'):
                    ttft = metrics.first_token_time * 1000
                    use_metrics = True

            # 如果 metrics 不可用，使用两次调用方法
            if not use_metrics or ttft &lt;= 0:
                temp_params = SamplingParams(
                    temperature=temperature,
                    top_p=0.95,
                    max_tokens=1,
                )
                ttft_start = time.time()
                self.llm.generate([prompt], temp_params, use_tqdm=False)
                ttft_end = time.time()
                ttft = (ttft_end - ttft_start) * 1000

            # 第三步：计算 TBT
            avg_tbt = 0.0
            if token_count &gt; 1:
                if ttft &lt; e2e:
                    avg_tbt = (e2e - ttft) / (token_count - 1)
                else:
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

    def _infer_service(self, prompts: List[str], max_tokens: int = 100, temperature: float = 0.7) -&gt; List[Dict]:
        """服务模式推理：流式 API 准确测量"""
        results = []

        for prompt in prompts:
            first_token_time = None
            full_start = time.time()
            token_times = []
            generated_text = ""

            try:
                # 使用流式 API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )

                for chunk in response:
                    current_time = time.time()
                    if first_token_time is None:
                        first_token_time = current_time
                    token_times.append(current_time)

                    if chunk.choices[0].delta.content:
                        generated_text += chunk.choices[0].delta.content

            except Exception as e:
                print(f"警告: 推理失败: {e}")
                # 回退到非流式方法
                full_start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False
                )
                full_end = time.time()
                first_token_time = full_end
                token_times = [full_end]
                generated_text = response.choices[0].message.content or ""

            full_end = time.time()
            token_count = len(token_times)

            # 计算延迟指标
            e2e = (full_end - full_start) * 1000

            if first_token_time is not None:
                ttft = (first_token_time - full_start) * 1000
            else:
                ttft = e2e

            # 计算平均 TBT
            avg_tbt = 0.0
            if len(token_times) &gt; 1:
                tbts = []
                for i in range(1, len(token_times)):
                    tbt = (token_times[i] - token_times[i-1]) * 1000
                    tbts.append(tbt)
                avg_tbt = sum(tbts) / len(tbts)
            elif token_count &gt; 1:
                # 降级方案
                avg_tbt = (e2e - ttft) / (token_count - 1) if ttft &lt; e2e else e2e / token_count

            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "token_count": token_count,
                "ttft": ttft,
                "tbt": avg_tbt,
                "e2e": e2e
            })

        return results

    def infer_prefill_only(self, prompts: List[str], max_tokens: int = 1) -&gt; List[Dict]:
        """
        仅执行一次生成用于预填充建模。

        使用 max_tokens=1，端到端耗时可近似视作 TTFT。
        """
        if self.use_service:
            return self._infer_prefill_only_service(prompts, max_tokens)
        else:
            return self._infer_prefill_only_offline(prompts, max_tokens)

    def _infer_prefill_only_offline(self, prompts: List[str], max_tokens: int = 1) -&gt; List[Dict]:
        """离线模式预填充测量"""
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
            if outputs and len(outputs) &gt; 0:
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

    def _infer_prefill_only_service(self, prompts: List[str], max_tokens: int = 1) -&gt; List[Dict]:
        """服务模式预填充测量"""
        results = []

        for prompt in prompts:
            start = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                stream=False
            )
            end = time.time()

            latency_ms = (end - start) * 1000
            generated_text = response.choices[0].message.content or ""
            token_count = 1

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
    # 默认使用离线模式（向后兼容）
    inferencer = LLMInferencer(use_service=False)
    results = inferencer.infer(["你好"])
    print(f"推理结果: {results[0]['generated_text'][:100]}...")
    print(f"TTFT: {results[0]['ttft']:.2f}ms, TBT: {results[0]['tbt']:.2f}ms")

