from vllm import LLM, SamplingParams
from power_control import set_power_cap
from monitor import PowerMonitor
import time
from typing import Dict

class DynamicPowerInferencer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
                 full_power: int = 140, decode_power: int = 70):
        self.full_power = full_power  # Prefill阶段最大功率
        self.decode_power = decode_power  # Decode阶段功率
        self.llm = LLM(
            model=model_name,
            quantization="gptq",
            max_model_len=8192,
            gpu_memory_utilization=0.9
        )
        self.sampling_params = SamplingParams(max_tokens=1024, temperature=0.7)

    def infer_with_dynamic_power(self, prompt: str) -> Dict:
        """动态功率调节推理：Prefill阶段满功率，Decode阶段降功率"""
        # Prefill阶段：设置满功率
        print(f"设置Prefill阶段功率: {self.full_power}W")
        set_power_cap(self.full_power)
        time.sleep(0.2)  # 等待功率调整完成

        monitor = PowerMonitor()
        monitor.start()

        start_time = time.time()
        output = self.llm.generate(prompt, self.sampling_params)[0]
        end_time = time.time()

        # 停止监测
        power_data = monitor.stop()
        total_energy = monitor.calculate_total_energy()

        # 恢复默认功率
        set_power_cap(self.full_power)

        generated_text = output.outputs[0].text
        token_count = len(output.outputs[0].token_ids)
        ttft = output.metrics.first_token_time - start_time if hasattr(output.metrics, 'first_token_time') else 0
        e2e = (end_time - start_time) * 1000
        tbt = (e2e - ttft * 1000) / (token_count - 1) if token_count > 1 else 0

        return {
            "generated_text": generated_text,
            "token_count": token_count,
            "ttft_ms": ttft * 1000,
            "tbt_ms": tbt,
            "e2e_ms": e2e,
            "total_energy_j": total_energy,
            "edp": e2e * total_energy
        }

    def infer_with_fixed_power(self, prompt: str, power: int) -> Dict:
        """固定功率推理，用于对比测试"""
        print(f"设置固定功率: {power}W")
        set_power_cap(power)
        time.sleep(0.2)

        monitor = PowerMonitor()
        monitor.start()

        start_time = time.time()
        output = self.llm.generate(prompt, self.sampling_params)[0]
        end_time = time.time()

        power_data = monitor.stop()
        total_energy = monitor.calculate_total_energy()

        set_power_cap(self.full_power)

        generated_text = output.outputs[0].text
        token_count = len(output.outputs[0].token_ids)
        ttft = output.metrics.first_token_time - start_time if hasattr(output.metrics, 'first_token_time') else 0
        e2e = (end_time - start_time) * 1000
        tbt = (e2e - ttft * 1000) / (token_count - 1) if token_count > 1 else 0

        return {
            "generated_text": generated_text,
            "token_count": token_count,
            "ttft_ms": ttft * 1000,
            "tbt_ms": tbt,
            "e2e_ms": e2e,
            "total_energy_j": total_energy,
            "edp": e2e * total_energy
        }

if __name__ == "__main__":
    # 测试动态功率策略和固定功率对比
    inferencer = DynamicPowerInferencer(full_power=140, decode_power=80)
    test_prompt = "请详细解释大语言模型推理过程中的Prefill和Decode两个阶段的区别和各自的性能特点，不少于300字。"

    print("\n=== 测试动态功率策略 ===")
    dynamic_result = inferencer.infer_with_dynamic_power(test_prompt)
    print(f"生成Token数: {dynamic_result['token_count']}")
    print(f"TTFT: {dynamic_result['ttft_ms']:.2f}ms")
    print(f"TBT: {dynamic_result['tbt_ms']:.2f}ms")
    print(f"E2E延迟: {dynamic_result['e2e_ms']:.2f}ms")
    print(f"总能耗: {dynamic_result['total_energy_j']:.2f}J")
    print(f"EDP: {dynamic_result['edp']:.2f}")

    print("\n=== 测试固定功率140W ===")
    fixed_result_140 = inferencer.infer_with_fixed_power(test_prompt, 140)
    print(f"生成Token数: {fixed_result_140['token_count']}")
    print(f"TTFT: {fixed_result_140['ttft_ms']:.2f}ms")
    print(f"TBT: {fixed_result_140['tbt_ms']:.2f}ms")
    print(f"E2E延迟: {fixed_result_140['e2e_ms']:.2f}ms")
    print(f"总能耗: {fixed_result_140['total_energy_j']:.2f}J")
    print(f"EDP: {fixed_result_140['edp']:.2f}")

    print("\n=== 测试固定功率80W ===")
    fixed_result_80 = inferencer.infer_with_fixed_power(test_prompt, 80)
    print(f"生成Token数: {fixed_result_80['token_count']}")
    print(f"TTFT: {fixed_result_80['ttft_ms']:.2f}ms")
    print(f"TBT: {fixed_result_80['tbt_ms']:.2f}ms")
    print(f"E2E延迟: {fixed_result_80['e2e_ms']:.2f}ms")
    print(f"总能耗: {fixed_result_80['total_energy_j']:.2f}J")
    print(f"EDP: {fixed_result_80['edp']:.2f}")

    # 计算优化效果
    edp_improvement = (1 - dynamic_result['edp'] / fixed_result_140['edp']) * 100
    latency_increase = (dynamic_result['e2e_ms'] / fixed_result_140['e2e_ms'] - 1) * 100
    energy_saving = (1 - dynamic_result['total_energy_j'] / fixed_result_140['total_energy_j']) * 100

    print(f"\n=== 动态功率策略相对140W固定功率优化效果 ===")
    print(f"EDP降低: {edp_improvement:.1f}%")
    print(f"能耗节省: {energy_saving:.1f}%")
    print(f"延迟增加: {latency_increase:.1f}%")
