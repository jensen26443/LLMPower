
"""
测试修复后的 TTFT/TBT 测量方法
"""
from llm_inference import LLMInferencer


def main():
    print("测试修复后的 TTFT/TBT 测量方法")
    print("=" * 60)

    # 初始化推理器
    print("\n正在加载模型...")
    inferencer = LLMInferencer()

    # 测试 prompts
    test_prompts = [
        "你好，请介绍一下你自己。",
        "什么是深度学习？请用简单的语言解释。",
    ]

    print("\n开始推理测试...")
    print("-" * 60)

    results = inferencer.infer(test_prompts, max_tokens=50)

    for i, res in enumerate(results):
        print(f"\nPrompt {i+1}: {test_prompts[i][:50]}...")
        print(f"  Token 数量: {res['token_count']}")
        print(f"  TTFT: {res['ttft']:.2f} ms")
        print(f"  TBT:  {res['tbt']:.2f} ms")
        print(f"  E2E:  {res['e2e']:.2f} ms")
        if res['tbt'] &gt; 0:
            print(f"  TTFT/TBT 比率: {res['ttft']/res['tbt']:.2f}x")

    print("\n" + "=" * 60)
    print("预期结果：")
    print("  - TTFT 应该远大于 TBT（通常 2-10 倍）")
    print("  - TBT 不应该是 0")
    print("  - E2E ≈ TTFT + TBT × (token_count - 1)")


if __name__ == "__main__":
    main()

