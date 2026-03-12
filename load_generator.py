import random
import os
import json
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False


class ShareGPTLoader:
    """ShareGPT数据集加载器（支持JSON和JSONL格式）"""

    def __init__(self, data_dir: str = "./input/ShareGPT",
                 tokenizer_name: str = "./Qwen2.5-7B-Instruct-AWQ",
                 prefer_zh: bool = True,
                 max_prompts: int = 100000):
        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        self.prefer_zh = prefer_zh  # 优先使用中文数据
        self.max_prompts = max_prompts  # 最大加载prompt数量
        self.conversations = []
        self.tokenized_prompts = defaultdict(list)  # token_count -> List[prompt]
        self.tokenizer = None

        if HAS_TOKENIZER:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            except Exception as e:
                print(f"加载tokenizer失败: {e}")

        self._load_data()
        self._build_index()

    def _load_data(self):
        """加载ShareGPT数据集（支持JSON和JSONL格式）"""
        if not os.path.exists(self.data_dir):
            print(f"ShareGPT数据目录不存在: {self.data_dir}")
            return

        # 按优先级排序数据文件
        data_files = self._get_sorted_data_files()

        if not data_files:
            print(f"在 {self.data_dir} 中没有找到数据文件")
            return

        for data_file in data_files:
            if len(self.conversations) >= self.max_prompts:
                break

            try:
                if data_file.endswith('.jsonl'):
                    self._load_jsonl_file(data_file)
                elif data_file.endswith('.json'):
                    self._load_json_file(data_file)
            except Exception as e:
                print(f"加载文件 {os.path.basename(data_file)} 失败: {e}")

        print(f"从ShareGPT加载了 {len(self.conversations)} 条prompt")

    def _get_sorted_data_files(self) -> List[str]:
        """获取排序后的数据文件列表（中文优先）"""
        all_files = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json') or filename.endswith('.jsonl'):
                all_files.append(os.path.join(self.data_dir, filename))

        # 排序：中文优先，然后英文
        def priority_score(filepath):
            filename = os.path.basename(filepath)
            if self.prefer_zh:
                if 'zh' in filename or 'cn' in filename:
                    return 0  # 最高优先级
                elif 'en' in filename:
                    return 2
                else:
                    return 1
            else:
                if 'en' in filename:
                    return 0
                elif 'zh' in filename or 'cn' in filename:
                    return 2
                else:
                    return 1

        return sorted(all_files, key=priority_score)

    def _load_jsonl_file(self, filepath: str):
        """加载JSONL格式文件"""
        filename = os.path.basename(filepath)
        count_before = len(self.conversations)

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if len(self.conversations) >= self.max_prompts:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    prompt = self._extract_human_prompt(item)
                    if prompt:
                        self.conversations.append(prompt)
                except json.JSONDecodeError:
                    continue

        loaded = len(self.conversations) - count_before
        if loaded > 0:
            print(f"  {filename}: 加载了 {loaded} 条prompt")

    def _load_json_file(self, filepath: str):
        """加载JSON格式文件"""
        filename = os.path.basename(filepath)
        count_before = len(self.conversations)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                if len(self.conversations) >= self.max_prompts:
                    break
                prompt = self._extract_human_prompt(item)
                if prompt:
                    self.conversations.append(prompt)

        loaded = len(self.conversations) - count_before
        if loaded > 0:
            print(f"  {filename}: 加载了 {loaded} 条prompt")

    def _extract_human_prompt(self, item: Dict) -> Optional[str]:
        """从item中提取human用户的prompt"""
        # 尝试多种格式
        if 'conversation' in item:
            # ModelScope格式: {"conversation": [{"human": "...", "assistant": "..."}]}
            conv_list = item['conversation']
            if isinstance(conv_list, list) and len(conv_list) > 0:
                first_conv = conv_list[0]
                if isinstance(first_conv, dict):
                    if 'human' in first_conv:
                        prompt = first_conv['human']
                        if prompt and len(prompt.strip()) > 0:
                            return prompt.strip()
                    elif 'user' in first_conv:
                        prompt = first_conv['user']
                        if prompt and len(prompt.strip()) > 0:
                            return prompt.strip()

        elif 'conversations' in item:
            # HuggingFace格式
            conv_list = item['conversations']
            if isinstance(conv_list, list) and len(conv_list) > 0:
                for conv in conv_list:
                    role = conv.get('from') or conv.get('role')
                    if role in ('human', 'user'):
                        prompt = conv.get('value') or conv.get('content')
                        if prompt and len(prompt.strip()) > 0:
                            return prompt.strip()

        return None

    def _build_index(self):
        """构建token数索引"""
        if not self.tokenizer or len(self.conversations) == 0:
            return

        print("正在构建token数索引...")
        for prompt in self.conversations:
            try:
                tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                token_count = len(tokens)
                self.tokenized_prompts[token_count].append(prompt)
            except:
                pass

        print(f"索引构建完成，覆盖token数范围: {min(self.tokenized_prompts.keys()) if self.tokenized_prompts else 0} "
              f"~ {max(self.tokenized_prompts.keys()) if self.tokenized_prompts else 0}")

    def get_prompt_by_token_count(self, target_count: int, tolerance: int = 5) -> Optional[str]:
        """获取接近目标token数的prompt

        Args:
            target_count: 目标token数
            tolerance: 容差范围

        Returns:
            符合要求的prompt，如果没有则返回None
        """
        if not self.tokenized_prompts:
            return None

        # 在容差范围内查找
        for diff in range(0, tolerance + 1):
            # 先查找精确匹配
            if target_count in self.tokenized_prompts:
                return random.choice(self.tokenized_prompts[target_count])
            # 查找略小或略大的
            if (target_count - diff) in self.tokenized_prompts:
                return random.choice(self.tokenized_prompts[target_count - diff])
            if (target_count + diff) in self.tokenized_prompts:
                return random.choice(self.tokenized_prompts[target_count + diff])

        # 如果没有找到，返回最接近的
        closest_count = min(self.tokenized_prompts.keys(),
                           key=lambda x: abs(x - target_count))
        return random.choice(self.tokenized_prompts[closest_count])


class LoadGenerator:
    def __init__(self, sharegpt_dir: str = "./input/ShareGPT",
                 tokenizer_name: str = "./Qwen2.5-7B-Instruct-AWQ"):
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

        # ShareGPT加载器
        self.sharegpt_loader = None
        if os.path.exists(sharegpt_dir):
            try:
                self.sharegpt_loader = ShareGPTLoader(sharegpt_dir, tokenizer_name)
            except Exception as e:
                print(f"ShareGPT加载器初始化失败: {e}")

        # Tokenizer
        self.tokenizer = None
        self.tokenizer_name = tokenizer_name
        if HAS_TOKENIZER:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            except Exception as e:
                print(f"加载tokenizer失败: {e}")

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

    def count_tokens(self, text: str) -> int:
        """计算文本的token数（使用Qwen2.5分词器）"""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except:
                pass
        # 回退估算
        return len(text)

    def generate_prompt_by_token_count(self, token_count: int,
                                       prefer_sharegpt: bool = True) -> str:
        """生成指定token数的prompt

        Args:
            token_count: 目标token数
            prefer_sharegpt: 是否优先使用ShareGPT数据

        Returns:
            接近指定token数的prompt文本
        """
        # 优先尝试从ShareGPT获取真实数据
        if prefer_sharegpt and self.sharegpt_loader:
            prompt = self.sharegpt_loader.get_prompt_by_token_count(token_count, tolerance=max(2, token_count // 10))
            if prompt:
                # 验证token数
                actual_count = self.count_tokens(prompt)
                if abs(actual_count - token_count) <= max(5, token_count // 5):
                    return prompt

        # 如果没有ShareGPT或找不到合适的，使用人工生成的方法
        return self._generate_synthetic_prompt(token_count)

    def _generate_synthetic_prompt(self, token_count: int) -> str:
        """生成合成的prompt（备用方案）"""
        # 基础文本，用于构建
        base_word = "你好"
        base_text = base_word * 1000  # 足够长的基础文本

        if self.tokenizer:
            try:
                # 使用二分法找到接近目标token数的文本
                left = 1
                right = len(base_text)
                best_text = base_text[:min(token_count * 2, len(base_text))]
                best_diff = float('inf')

                for _ in range(20):  # 最多迭代20次
                    mid = (left + right) // 2
                    current_text = base_text[:mid]
                    current_count = self.count_tokens(current_text)

                    diff = abs(current_count - token_count)
                    if diff < best_diff:
                        best_diff = diff
                        best_text = current_text

                    if current_count < token_count:
                        left = mid + 1
                    elif current_count > token_count:
                        right = mid - 1
                    else:
                        break  # 完美匹配

                # 验证最终token数
                final_count = self.count_tokens(best_text)
                if final_count != token_count:
                    # 如果不完全匹配，尝试微调
                    if final_count < token_count:
                        # 添加更多词
                        while self.count_tokens(best_text) < token_count:
                            best_text += base_word
                    else:
                        # 移除一些词
                        while self.count_tokens(best_text) > token_count and len(best_text) > len(base_word):
                            best_text = best_text[:-len(base_word)]

                return best_text

            except Exception as e:
                print(f"使用tokenizer生成prompt失败: {e}，使用估算方法")

        # 回退方案：基于字符数估算
        estimated_chars = token_count
        repeat_text = "你好世界"
        repeat_count = (estimated_chars // len(repeat_text)) + 1
        prompt = (repeat_text * repeat_count)[:estimated_chars]
        return prompt


if __name__ == "__main__":
    generator = LoadGenerator()

    # 测试生成不同token数的prompt
    test_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    print("测试生成不同token数的prompt:")
    print("-" * 60)

    for count in test_counts:
        prompt = generator.generate_prompt_by_token_count(count)
        actual_count = generator.count_tokens(prompt)
        print(f"目标: {count} tokens, 实际: {actual_count} tokens")
        print(f"  文本: {prompt[:80]}..." if len(prompt) > 80 else f"  文本: {prompt}")
        print()
