import random
from typing import List

class LoadGenerator:
    def __init__(self):
        # 预置不同长度的prompt模板
        self.short_prompts = [
            "介绍一下人工智能的应用场景。",
            "什么是大语言模型？",
            "解释一下什么是机器学习。",
            "Python和Java的区别是什么？",
            "如何提高编程效率？",
            "什么是深度学习？",
            "解释一下神经网络的工作原理。",
            "什么是自然语言处理？",
            "常见的排序算法有哪些？",
            "什么是云计算？"
        ]

        self.long_prompts = [
            "请详细分析大语言模型推理过程中的性能瓶颈，包括显存带宽、计算能力、内存访问等各个方面的影响因素，并给出具体的优化建议。要求分点说明，每个点不少于100字。",
            "对比分析Transformer、RNN、CNN三种深度学习架构在自然语言处理任务中的优缺点，分别从并行性、长程依赖处理、计算复杂度、训练难度等多个维度进行对比，每个维度不少于80字说明。",
            "详细解释vLLM的PagedAttention技术的工作原理，包括它是如何解决传统Transformer推理中KV缓存内存浪费问题的，以及它的实现机制和性能优势，要求不少于300字。",
            "请描述大语言模型的训练过程，从数据预处理、模型架构设计、分布式训练策略、到微调对齐的完整流程，每个步骤说明关键技术和挑战，不少于400字。",
            "分析当前大语言模型推理部署面临的主要挑战，包括低延迟需求、高吞吐需求、显存限制、成本控制等方面，以及对应的解决方案，分点说明不少于5点。"
        ]

    def generate_load(self, load_type: str = "mixed", count: int = 10) -> List[str]:
        """生成指定类型的负载
        load_type: short/long/mixed
        """
        prompts = []
        if load_type == "short":
            for _ in range(count):
                prompts.append(random.choice(self.short_prompts))
        elif load_type == "long":
            for _ in range(count):
                prompts.append(random.choice(self.long_prompts))
        elif load_type == "mixed":
            # 70%短请求，30%长请求
            for _ in range(count):
                if random.random() < 0.7:
                    prompts.append(random.choice(self.short_prompts))
                else:
                    prompts.append(random.choice(self.long_prompts))
        return prompts

if __name__ == "__main__":
    generator = LoadGenerator()
    mixed_load = generator.generate_load("mixed", 5)
    print("生成的混合负载:")
    for i, prompt in enumerate(mixed_load):
        print(f"{i+1}. {prompt[:50]}...")
