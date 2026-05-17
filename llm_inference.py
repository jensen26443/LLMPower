import json
import time
import os
import subprocess
import atexit
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Dict, Optional

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

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# 服务模式下用于回退统计输出 token 数
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS_TOKENIZER = True
except ImportError:
    HAS_TRANSFORMERS_TOKENIZER = False


class LLMInferencer:
    """统一封装离线 vLLM 和 OpenAI 兼容服务两种推理模式。

    结题实验主要使用服务模式：通过流式返回记录首 token、后续 token 和完成时间，
    从同一请求链路中计算 TTFT、TBT、E2E，保证不同功率策略的对比口径一致。
    """

    def __init__(
        self,
        model_name: str = "./Qwen2.5-7B-Instruct-AWQ",
        served_model_name: Optional[str] = None,
        use_service: bool = False,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        start_server: bool = False,
        gpu_memory_utilization: float = 0.85,
        disable_prefix_caching: bool = True,
        service_request_mode: str = "chat",
        enable_chunked_prefill: bool = True,
        max_num_batched_tokens: int = 2048,
        max_num_seqs: int = 64,
    ):
        """
        初始化推理器，支持两种模式：

        1. 离线模式 (use_service=False, 默认): 使用 vLLM.LLM.generate()
        2. 服务模式 (use_service=True): 使用 OpenAI 兼容 API 连接 vLLM 服务

        Args:
            model_name: 模型路径或名称
            served_model_name: 服务模式下 OpenAI API 使用的模型名
            use_service: 是否使用服务模式
            base_url: vLLM 服务地址（服务模式）
            api_key: API 密钥（vLLM 默认 "EMPTY"）
            start_server: 是否自动启动 vLLM 服务（服务模式）
            gpu_memory_utilization: GPU 显存利用率
            disable_prefix_caching: 是否禁用前缀缓存
            service_request_mode: 服务模式请求类型，支持 "chat" 或 "completion"
            enable_chunked_prefill: 是否显式启用 chunked prefill
            max_num_batched_tokens: 显式固定 max_num_batched_tokens
            max_num_seqs: 显式固定 max_num_seqs
        """
        self.model_name = model_name
        self.served_model_name = served_model_name or model_name
        self.use_service = use_service
        self.client = None
        self.llm = None
        self.sampling_params = None
        self.tokenizer = None
        self.server_process: Optional[subprocess.Popen] = None
        self.base_url = base_url
        self.api_key = api_key
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        if service_request_mode not in {"chat", "completion"}:
            raise ValueError(f"Unsupported service_request_mode: {service_request_mode}")
        self.service_request_mode = service_request_mode
        if self.service_request_mode == "completion" and not HAS_HTTPX:
            raise ImportError("httpx is required for completion service mode")

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
            enable_prefix_caching=not disable_prefix_caching,
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
        self.base_url = base_url
        self.api_key = api_key
        self.client = self._create_service_client()

        if start_server:
            self._start_vllm_server(
                self.model_name,
                gpu_memory_utilization,
                disable_prefix_caching
            )

    def _create_service_client(self):
        """创建 OpenAI 兼容客户端。"""
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def _build_service_request_kwargs(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        extra_body: Optional[Dict],
        include_stream_options: bool = True,
    ) -> Dict:
        """按服务请求模式构造 API 参数。"""
        if self.service_request_mode == "completion":
            # completion 模式用于固定 prompt / output token 的实验，便于和 vLLM bench 口径对齐。
            request_kwargs = {
                "model": self.served_model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": include_stream_options,
            }
            if extra_body:
                request_kwargs.update(extra_body)
        else:
            # chat 模式保留给普通对话式调用；主实验默认走 completion 模式。
            request_kwargs = {
                "model": self.served_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

        if include_stream_options:
            request_kwargs["stream_options"] = {"include_usage": True}
        if extra_body and self.service_request_mode != "completion":
            request_kwargs["extra_body"] = extra_body
        return request_kwargs

    def _create_service_completion(self, client, request_kwargs: Dict, stream: bool):
        """根据请求模式发起 completions/chat.completions 请求。"""
        return client.chat.completions.create(**request_kwargs, stream=stream)

    def _build_completion_http_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _iter_completion_stream_http(self, request_kwargs: Dict):
        url = f"{self.base_url.rstrip('/')}/completions"
        with httpx.stream(
            "POST",
            url,
            headers=self._build_completion_http_headers(),
            json=request_kwargs,
            timeout=120.0,
            trust_env=False,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                yield json.loads(payload)

    def _create_completion_http(self, request_kwargs: Dict) -> Dict:
        url = f"{self.base_url.rstrip('/')}/completions"
        response = httpx.post(
            url,
            headers=self._build_completion_http_headers(),
            json=request_kwargs,
            timeout=120.0,
            trust_env=False,
        )
        response.raise_for_status()
        return response.json()

    def _extract_stream_text(self, chunk) -> Optional[str]:
        """从流式返回中提取当前 chunk 的文本。"""
        if isinstance(chunk, dict):
            # httpx 直连 /completions 时返回 dict；OpenAI SDK 返回对象，两种格式都兼容。
            choices = chunk.get("choices") or []
            if not choices:
                return None
            return choices[0].get("text")

        if not getattr(chunk, "choices", None):
            return None

        choice = chunk.choices[0]
        if self.service_request_mode == "completion":
            return getattr(choice, "text", None)

        delta = getattr(choice, "delta", None)
        return getattr(delta, "content", None) if delta is not None else None

    def _extract_response_text(self, response) -> str:
        """从非流式返回中提取文本。"""
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if not choices:
                return ""
            return choices[0].get("text", "") or ""

        if not getattr(response, "choices", None):
            return ""

        choice = response.choices[0]
        if self.service_request_mode == "completion":
            return getattr(choice, "text", "") or ""

        message = getattr(choice, "message", None)
        return getattr(message, "content", "") if message is not None else ""

    def _get_tokenizer(self):
        """延迟加载 tokenizer，用于服务模式下回退统计输出 token 数。"""
        if self.tokenizer is not None or not HAS_TRANSFORMERS_TOKENIZER:
            return self.tokenizer

        tokenizer_name = os.path.expanduser(self.model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"警告: 加载 tokenizer 失败: {e}")
            self.tokenizer = None

        return self.tokenizer

    def _count_generated_tokens(self, generated_text: str) -> int:
        """统计生成文本对应的 token 数，尽量与 vllm bench serve 口径保持一致。"""
        if not generated_text:
            return 0

        tokenizer = self._get_tokenizer()
        if tokenizer:
            try:
                return len(tokenizer.encode(generated_text, add_special_tokens=False))
            except Exception:
                pass

        # 与 vllm bench serve 的保守回退一致：未知时至少按 1 个输出 token 处理。
        return 1

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
            "--served-model-name", self.served_model_name,
            "--quantization", "awq",
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
            "--max-num-seqs", str(self.max_num_seqs),
        ]
        if self.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        if disable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")

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

    def infer(self, prompts: List[str], max_tokens: int = None, temperature: float = 0.7,
              extra_body: Optional[Dict] = None) -> List[Dict]:
        """
        执行推理，返回包含延迟指标的结果

        离线模式：使用两次调用方法测量 TTFT/TBT
        服务模式：使用 OpenAI 兼容 API 的流式输出准确测量 TTFT/TBT
        """
        if self.use_service:
            return self._infer_service(prompts, max_tokens, temperature, extra_body=extra_body)
        else:
            return self._infer_offline(prompts, max_tokens, temperature)

    def _infer_offline(self, prompts: List[str], max_tokens: int = None, temperature: float = 0.7) -> List[Dict]:
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

            if final_output and len(final_output.outputs) > 0:
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
            if not use_metrics or ttft <= 0:
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
            if token_count > 1:
                if ttft < e2e:
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

    def infer_concurrent(self, prompts: List[str], max_tokens: int = 100, temperature: float = 0.7,
                         extra_body: Optional[Dict] = None,
                         stream_hook: Optional[Callable[[Dict], None]] = None) -> List[Dict]:
        """
        服务模式下并发发起多个请求，用于触发 vLLM 在线 batching。

        返回结果顺序与输入 prompts 保持一致。
        """
        if not self.use_service:
            return self.infer(prompts, max_tokens=max_tokens, temperature=temperature)

        results: List[Optional[Dict]] = [None] * len(prompts)
        start_event = threading.Event()

        def run_single(index: int, prompt: str):
            client = self._create_service_client()
            start_event.wait()
            result = self._infer_service_single(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                client=client,
                extra_body=extra_body,
                request_index=index,
                stream_hook=stream_hook,
            )
            result["request_index"] = index
            return index, result

        with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as executor:
            futures = [
                executor.submit(run_single, index, prompt)
                for index, prompt in enumerate(prompts)
            ]
            time.sleep(0.05)
            start_event.set()

            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result

        return [result for result in results if result is not None]

    def _infer_service(self, prompts: List[str], max_tokens: int = 100, temperature: float = 0.7,
                       extra_body: Optional[Dict] = None,
                       stream_hook: Optional[Callable[[Dict], None]] = None) -> List[Dict]:
        """服务模式推理：流式 API 准确测量"""
        results = []

        for prompt in prompts:
            results.append(
                self._infer_service_single(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    client=self.client,
                    extra_body=extra_body,
                    request_index=len(results),
                    stream_hook=stream_hook,
                )
            )

        return results

    def _infer_service_single(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7,
                              client=None, extra_body: Optional[Dict] = None,
                              request_index: Optional[int] = None,
                              stream_hook: Optional[Callable[[Dict], None]] = None) -> Dict:
        """服务模式下单请求推理，返回完整时间戳信息。"""
        client = client or self.client
        first_token_time = None
        full_start = time.perf_counter()
        wall_start = time.time()
        token_times = []
        generated_text = ""
        output_token_count = 0
        first_token_wall_time = None

        request_kwargs = self._build_service_request_kwargs(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
            include_stream_options=True,
        )

        try:
            if self.service_request_mode == "completion":
                response = self._iter_completion_stream_http(request_kwargs)
            else:
                response = self._create_service_completion(client, request_kwargs, stream=True)

            for chunk in response:
                current_time = time.perf_counter()
                current_wall_time = time.time()
                if getattr(chunk, "usage", None) and chunk.usage.completion_tokens is not None:
                    output_token_count = chunk.usage.completion_tokens
                elif isinstance(chunk, dict):
                    usage = chunk.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        output_token_count = usage["completion_tokens"]
                delta = self._extract_stream_text(chunk)
                if delta:
                    if first_token_time is None:
                        first_token_time = current_time
                        first_token_wall_time = current_wall_time
                        event_type = "first_token"
                    else:
                        event_type = "chunk"
                    token_times.append(current_time)
                    generated_text += delta
                    generated_tokens = output_token_count if output_token_count > 0 else self._count_generated_tokens(generated_text)
                    if stream_hook is not None:
                        stream_hook({
                            "request_index": request_index,
                            "event_type": event_type,
                            "wall_time": current_wall_time,
                            "generated_tokens": generated_tokens,
                        })

        except Exception as e:
            print(f"警告: 推理失败: {e}")
            fallback_request_kwargs = self._build_service_request_kwargs(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
                include_stream_options=False,
            )
            if self.service_request_mode == "completion":
                response = self._create_completion_http(fallback_request_kwargs)
            else:
                response = self._create_service_completion(client, fallback_request_kwargs, stream=False)
            full_end = time.perf_counter()
            wall_end = time.time()
            first_token_time = full_end
            first_token_wall_time = wall_end
            response_text = self._extract_response_text(response)
            token_times = [full_end] if response_text else []
            generated_text = response_text
            if getattr(response, "usage", None) and response.usage.completion_tokens is not None:
                output_token_count = response.usage.completion_tokens
            elif isinstance(response, dict):
                usage = response.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    output_token_count = usage["completion_tokens"]
            if stream_hook is not None:
                generated_tokens = output_token_count if output_token_count > 0 else self._count_generated_tokens(generated_text)
                stream_hook({
                    "request_index": request_index,
                    "event_type": "finished",
                    "wall_time": wall_end,
                    "generated_tokens": generated_tokens,
                })
            return self._build_service_result(
                prompt=prompt,
                generated_text=generated_text,
                full_start=full_start,
                full_end=full_end,
                first_token_time=first_token_time,
                token_times=token_times,
                output_token_count=output_token_count,
                wall_start=wall_start,
                wall_end=wall_end,
                first_token_wall_time=first_token_wall_time,
            )

        full_end = time.perf_counter()
        wall_end = time.time()
        if stream_hook is not None:
            generated_tokens = output_token_count if output_token_count > 0 else self._count_generated_tokens(generated_text)
            stream_hook({
                "request_index": request_index,
                "event_type": "finished",
                "wall_time": wall_end,
                "generated_tokens": generated_tokens,
            })
        return self._build_service_result(
            prompt=prompt,
            generated_text=generated_text,
            full_start=full_start,
            full_end=full_end,
            first_token_time=first_token_time,
            token_times=token_times,
            output_token_count=output_token_count,
            wall_start=wall_start,
            wall_end=wall_end,
            first_token_wall_time=first_token_wall_time,
        )

    def _build_service_result(self, prompt: str, generated_text: str, full_start: float, full_end: float,
                              first_token_time: Optional[float], token_times: List[float],
                              output_token_count: int = 0,
                              wall_start: Optional[float] = None,
                              wall_end: Optional[float] = None,
                              first_token_wall_time: Optional[float] = None) -> Dict:
        """整理服务模式的延迟结果。"""
        token_count = output_token_count if output_token_count > 0 else self._count_generated_tokens(generated_text)
        chunk_count = len(token_times)
        e2e = (full_end - full_start) * 1000

        if first_token_time is not None:
            ttft = (first_token_time - full_start) * 1000
        else:
            ttft = e2e

        itls = []
        if chunk_count > 1:
            for i in range(1, chunk_count):
                itls.append((token_times[i] - token_times[i - 1]) * 1000)

        avg_itl = sum(itls) / len(itls) if itls else 0.0
        tpot = 0.0
        if token_count > 1:
            tpot = (e2e - ttft) / (token_count - 1) if ttft < e2e else e2e / token_count

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "token_count": token_count,
            "ttft": ttft,
            "tbt": tpot,
            "tpot": tpot,
            "avg_itl": avg_itl,
            "itls": itls,
            "e2e": e2e,
            "start_time": full_start,
            "end_time": full_end,
            "first_token_time": first_token_time,
            "start_time_wall": wall_start,
            "end_time_wall": wall_end,
            "first_token_time_wall": first_token_wall_time,
            "token_times": token_times,
            "stream_chunk_count": chunk_count,
        }

    def infer_prefill_only(self, prompts: List[str], max_tokens: int = 1) -> List[Dict]:
        """
        仅执行一次生成用于预填充建模。

        使用 max_tokens=1，端到端耗时可近似视作 TTFT。
        """
        if self.use_service:
            return self._infer_prefill_only_service(prompts, max_tokens)
        else:
            return self._infer_prefill_only_offline(prompts, max_tokens)

    def _infer_prefill_only_offline(self, prompts: List[str], max_tokens: int = 1) -> List[Dict]:
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

    def _infer_prefill_only_service(self, prompts: List[str], max_tokens: int = 1) -> List[Dict]:
        """服务模式预填充测量"""
        results = []

        for prompt in prompts:
            start = time.time()
            response = self.client.chat.completions.create(
                model=self.served_model_name,
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
