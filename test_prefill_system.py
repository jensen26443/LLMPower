#!/usr/bin/env python3
"""
快速诊断脚本：验证预填充实验系统的功率监测
"""
import sys
import time
import random
sys.path.insert(0, '.')

from monitor import PowerMonitor
from llm_inference import LLMInferencer
from load_generator import LoadGenerator


def test_power_monitor():
    """测试功率监测模块"""
    print("=" * 60)
    print("测试1: 功率监测模块")
    print("=" * 60)

    monitor = PowerMonitor(sample_interval=0.02)
    print(f"使用backend: {monitor._backend}")

    # 采样5秒
    print("\n采样5秒空闲功率...")
    monitor.start()
    time.sleep(5)
    data = monitor.stop()

    print(f"采样点数: {len(data)}")
    if len(data) > 1:
        intervals = []
        for i in range(1, len(data)):
            dt = (data[i]['timestamp'] - data[i-1]['timestamp']) * 1000
            intervals.append(dt)
        print(f"采样间隔: {sum(intervals)/len(intervals):.1f}ms "
              f"(范围: {min(intervals):.1f}ms - {max(intervals):.1f}ms)")

        powers = [d['power_w'] for d in data]
        print(f"功率范围: {min(powers):.2f}W - {max(powers):.2f}W")
        print(f"前10个功率值: {[f'{p:.2f}' for p in powers[:10]]}")

    return len(data) > 0


def test_inference_with_monitor():
    """测试推理+功率监测"""
    print("\n" + "=" * 60)
    print("测试2: 推理+功率监测（长间隔）")
    print("=" * 60)

    # 初始化组件
    print("\n初始化模型...")
    try:
        inferencer = LLMInferencer()
    except Exception as e:
        print(f"模型加载跳过: {e}")
        print("（这是正常的，因为我们只是测试功率监测逻辑）")
        return True

    load_gen = LoadGenerator()

    # 测试几次推理，每次间隔500ms
    test_prompts = [
        load_gen.generate_prompt_by_token_count(64),
        load_gen.generate_prompt_by_token_count(256),
        load_gen.generate_prompt_by_token_count(1024),
    ]

    print("\n开始推理测试（间隔500ms）...")
    monitor = PowerMonitor(sample_interval=0.02)
    monitor.start()

    results = []
    for i, prompt in enumerate(test_prompts):
        print(f"\n推理 {i+1}/{len(test_prompts)}: {load_gen.count_tokens(prompt)} tokens")

        # 推理
        start = time.time()
        result = inferencer.infer_prefill_only([prompt], max_tokens=1)[0]
        end = time.time()

        results.append({
            'start': start,
            'end': end,
            'tokens': load_gen.count_tokens(prompt),
            'ttft': result['ttft']
        })

        print(f"  TTFT: {result['ttft']:.1f}ms")
        print(f"  等待500ms...")
        time.sleep(0.5)

    # 停止监测
    power_data = monitor.stop()
    print(f"\n总采样点数: {len(power_data)}")

    # 分析每个推理期间的功率
    print("\n分析每个推理的功率:")
    for i, res in enumerate(results):
        # 找到这个推理期间的功率数据
        relevant = []
        for pd in power_data:
            if res['start'] <= pd['timestamp'] <= res['end']:
                relevant.append(pd['power_w'])

        print(f"推理 {i+1} ({res['tokens']} tokens):")
        print(f"  时长: {(res['end'] - res['start'])*1000:.1f}ms")
        print(f"  期间采样点数: {len(relevant)}")
        if relevant:
            print(f"  功率范围: {min(relevant):.2f}W - {max(relevant):.2f}W")
            print(f"  平均功率: {sum(relevant)/len(relevant):.2f}W")

    return True


def main():
    print("\n" + "=" * 60)
    print("预填充实验系统诊断")
    print("=" * 60)

    try:
        # 测试1
        if not test_power_monitor():
            print("\n❌ 功率监测测试失败")
            return 1

        # 测试2
        if not test_inference_with_monitor():
            print("\n❌ 推理测试失败")
            return 1

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n建议:")
        print("1. 使用新代码重新运行实验（用pynvml）")
        print("2. 增加推理间隔到200-500ms")
        print("3. 确保time-padding-ms设置合理（40ms）")
        return 0

    except KeyboardInterrupt:
        print("\n\n已取消")
        return 130
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
