#!/usr/bin/env python3
"""
下载ShareGPT数据集脚本
"""
import os
import json
import argparse
from typing import List, Dict


def download_from_huggingface(output_dir: str = "./input/ShareGPT"):
    """从HuggingFace下载ShareGPT数据集"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        from datasets import load_dataset
        print("正在从HuggingFace下载ShareGPT数据集...")

        # 加载ShareGPT数据集
        dataset = load_dataset("lmsys/sharegpt", split="train")

        # 保存为JSON格式
        output_file = os.path.join(output_dir, "sharegpt.json")

        # 转换格式并保存
        data = []
        for item in dataset:
            conversations = item.get("conversations", [])
            if len(conversations) > 0:
                data.append({
                    "id": item.get("id", ""),
                    "conversations": conversations
                })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"ShareGPT数据集已保存到: {output_file}")
        print(f"共 {len(data)} 条对话")
        return output_file

    except ImportError:
        print("需要安装datasets库: pip install datasets")
        return None
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载ShareGPT数据集到 ./input/ShareGPT/ 目录")
        return None


def download_small_sample(output_dir: str = "./input/ShareGPT"):
    """创建一个小型示例数据集（用于测试）"""
    os.makedirs(output_dir, exist_ok=True)

    sample_data = [
        {
            "id": "sample_1",
            "conversations": [
                {"from": "human", "value": "你好，请介绍一下自己。"},
                {"from": "gpt", "value": "你好！我是一个AI助手，可以帮你回答各种问题。"}
            ]
        },
        {
            "id": "sample_2",
            "conversations": [
                {"from": "human", "value": "什么是机器学习？请详细解释。"},
                {"from": "gpt", "value": "机器学习是人工智能的一个分支，它使计算机系统能够通过经验自动改进，而无需明确编程。"}
            ]
        },
        {
            "id": "sample_3",
            "conversations": [
                {"from": "human", "value": "请写一篇关于气候变化的短文，包括原因、影响和解决方案。"},
                {"from": "gpt", "value": "气候变化是当今世界面临的最大挑战之一..."}
            ]
        }
    ]

    output_file = os.path.join(output_dir, "sharegpt_sample.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"示例数据集已创建: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载ShareGPT数据集")
    parser.add_argument("--sample", action="store_true",
                       help="仅创建示例数据集（不下载完整数据）")
    parser.add_argument("--output-dir", type=str, default="./input/ShareGPT",
                       help="输出目录")

    args = parser.parse_args()

    if args.sample:
        download_small_sample(args.output_dir)
    else:
        result = download_from_huggingface(args.output_dir)
        if result is None:
            print("\n回退到创建示例数据集...")
            download_small_sample(args.output_dir)
