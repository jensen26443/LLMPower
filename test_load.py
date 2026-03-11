import os
import pandas as pd

def test_load():
    result_dir = "results"
    all_metadata = []

    if not os.path.exists(result_dir):
        print(f"结果目录 {result_dir} 不存在")
        return

    print(f"扫描目录: {result_dir}")
    for filename in os.listdir(result_dir):
        if filename.endswith("_metadata.csv"):
            print(f"  发现文件: {filename}")
            df = pd.read_csv(f"{result_dir}/{filename}")
            all_metadata.append(df)

    if not all_metadata:
        print("没有找到实验结果")
        return

    df = pd.concat(all_metadata, ignore_index=True)
    print(f"\n总数据行数: {len(df)}")
    print(f"\n数据列: {list(df.columns)}")
    print(f"\n功率值: {sorted(df['power_cap_w'].unique())}")
    print(f"\n并发度: {sorted(df['concurrency'].unique())}")
    print(f"\n前5行数据:")
    print(df[['power_cap_w', 'concurrency', 'total_energy_j', 'avg_e2e_ms', 'avg_ttft_ms', 'avg_tbt_ms']])

    return df

if __name__ == "__main__":
    test_load()
