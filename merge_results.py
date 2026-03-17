#!/usr/bin/env python3
"""
合并实验结果：将新的bucket1结果替换旧结果中的bucket1
"""
import csv
import os
import shutil


def load_csv(filepath):
    """加载CSV文件"""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    return fieldnames, rows


def save_csv(filepath, fieldnames, rows):
    """保存CSV文件"""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_results(old_data_dir, new_data_dir, output_dir):
    """合并结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 查找文件
    old_raw = sorted(glob.glob(os.path.join(old_data_dir, "*_raw.csv")))[-1]
    old_agg = sorted(glob.glob(os.path.join(old_data_dir, "*_aggregated.csv")))[-1]
    old_meta = sorted(glob.glob(os.path.join(old_data_dir, "*_metadata.json")))[-1]

    new_raw = sorted(glob.glob(os.path.join(new_data_dir, "*_raw.csv")))[-1]
    new_agg = sorted(glob.glob(os.path.join(new_data_dir, "*_aggregated.csv")))[-1]

    print(f"旧数据: {old_raw}")
    print(f"新bucket1数据: {new_raw}")

    # 加载数据
    fieldnames_raw, old_raw_rows = load_csv(old_raw)
    _, new_raw_rows = load_csv(new_raw)

    fieldnames_agg, old_agg_rows = load_csv(old_agg)
    _, new_agg_rows = load_csv(new_agg)

    # 过滤旧数据：移除bucket1
    old_raw_filtered = [r for r in old_raw_rows if r["strategy"] != "bucket1"]
    old_agg_filtered = [r for r in old_agg_rows if r["strategy"] != "bucket1"]

    # 添加新的bucket1
    merged_raw = old_raw_filtered + new_raw_rows
    merged_agg = old_agg_filtered + new_agg_rows

    # 保存合并后的数据
    timestamp = int(os.path.getmtime(old_raw))
    merged_raw_file = os.path.join(output_dir, f"strategy_eval_{timestamp}_raw.csv")
    merged_agg_file = os.path.join(output_dir, f"strategy_eval_{timestamp}_aggregated.csv")

    save_csv(merged_raw_file, fieldnames_raw, merged_raw)
    save_csv(merged_agg_file, fieldnames_agg, merged_agg)

    # 复制metadata
    merged_meta_file = os.path.join(output_dir, f"strategy_eval_{timestamp}_metadata.json")
    shutil.copy2(old_meta, merged_meta_file)

    print(f"\n合并完成！")
    print(f"原始数据: {merged_raw_file}")
    print(f"聚合数据: {merged_agg_file}")
    print(f"元数据: {merged_meta_file}")

    return {
        "raw": merged_raw_file,
        "agg": merged_agg_file,
        "meta": merged_meta_file,
    }


if __name__ == "__main__":
    import glob
    import argparse

    parser = argparse.ArgumentParser(description="合并实验结果")
    parser.add_argument("--old-dir", default="./results1/old_results",
                       help="旧数据目录")
    parser.add_argument("--new-dir", default="./results1/bucket1_temp/data",
                       help="新bucket1数据目录")
    parser.add_argument("--output-dir", default="./results1/data",
                       help="输出目录")

    args = parser.parse_args()

    merge_results(args.old_dir, args.new_dir, args.output_dir)
