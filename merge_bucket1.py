#!/usr/bin/env python3
"""
合并新的bucket1数据到旧结果中
"""
import csv
import os
import glob
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


def merge_results():
    # 找到旧数据和新数据
    old_data_dir = "./results1/old_results"
    new_data_dir = "./results1/bucket1_temp/data"
    output_dir = "./results1/data"

    os.makedirs(output_dir, exist_ok=True)

    # 找到文件
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

    # 从旧文件名获取timestamp
    timestamp = os.path.basename(old_raw).split('_')[2]

    # 保存合并后的数据
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
    merge_results()
