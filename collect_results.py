import os
import re
import csv
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# ---------------------------------------------------------
# 模型总层数配置表
# ---------------------------------------------------------
MODEL_TOTAL_LAYERS = {
    "Qwen2.5-0.5B": 24, "Qwen2.5-1.5B": 28, "Qwen2.5-3B": 36,
    "Qwen2.5-7B": 28, "Qwen2.5-14B": 48, "Qwen2.5-32B": 64, "Qwen2.5-72B": 80,
    "Gemma-3-4B": 26, "Gemma-3-12B": 40, "Gemma-3-27B": 46,
    "Llama-3-8B": 32, "Llama-3-70B": 80, "Llama-3.1-8B": 32, "Llama-3.1-70B": 80
}

def extract_runs_from_file(log_path):
    """
    核心逻辑更新：不使用 split，而是使用正则表达式查找每一个完整的实验块。
    匹配从 '--output_file' 开始到 'Final Accuracy' 结束的所有内容。
    """
    runs = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 使用正则表达式匹配每一个实验段落
        # 这个正则会捕捉 --output_file 到 Final Accuracy 之间的所有文本
        pattern = re.compile(r"--output_file\s+(\S+).*?=== Zero-shot Steering PoC ===(.*?Done! Final Accuracy:\s*([\d\.]+)%)", re.DOTALL)
        
        matches = pattern.findall(content)
        
        for out_path, block_content, acc_val in matches:
            run_data = {
                "filename": os.path.basename(log_path),
                "model": "Unknown",
                "dataset": "Unknown",
                "layer": None,
                "alpha": None,
                "accuracy": float(acc_val),
                "run_type": "Steer"
            }

            # 1. 判定 run_type (基于 output_file 路径)
            out_path_lower = out_path.lower()
            if "reverse_baseline" in out_path_lower:
                run_data["run_type"] = "Reverse_Baseline"
            elif "repeat_baseline" in out_path_lower:
                run_data["run_type"] = "Repeat_Baseline"
            elif "baseline" in out_path_lower:
                run_data["run_type"] = "Baseline"
            else:
                run_data["run_type"] = "Steer"

            # 2. 提取层数和 Alpha
            ly_m = re.search(r"Layer:\s*(\d+)", block_content)
            al_m = re.search(r"Alpha:\s*([\d\.]+)", block_content)
            if ly_m: run_data["layer"] = int(ly_m.group(1))
            if al_m: run_data["alpha"] = float(al_m.group(1))

            # 3. 提取模型和数据集 (从 block 或 context 中查找)
            model_m = re.search(r"Model:\s*(?:.*/)?([\w\.\-]+)", block_content)
            if model_m: run_data["model"] = model_m.group(1)

            # 数据集名称：从 output_file 的路径中提取更稳健
            # 路径通常是 .../Qwen2.5-7B-Instruct/FOLIO/results...
            path_parts = out_path.split('/')
            if len(path_parts) > 2:
                # 假设数据集名称在文件名之前的那个文件夹
                run_data["dataset"] = path_parts[-2]
            
            runs.append(run_data)
                
    except Exception as e:
        print(f"[Error] 解析 {log_path} 失败: {e}")
    return runs

def process_and_save(df, base_out_dir):
    """保存并绘图"""
    if df.empty: return

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        target_dir = os.path.join(base_out_dir, model, dataset)
        os.makedirs(target_dir, exist_ok=True)
        
        # 排序
        order = {'Baseline': 0, 'Reverse_Baseline': 1, 'Repeat_Baseline': 2, 'Steer': 3}
        group['type_sort'] = group['run_type'].map(order)
        group = group.sort_values(by=["type_sort", "layer", "alpha"]).drop(columns=['type_sort'])
        
        group.to_csv(os.path.join(target_dir, "summary.csv"), index=False)
        
        # 绘图
        steer_data = group[group['run_type'] == 'Steer']
        if not steer_data.empty:
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(12, 7))
            sns.lineplot(data=steer_data, x="layer", y="accuracy", hue="alpha", marker="o", linewidth=2)
            
            # Baseline 线
            colors = {'Baseline': 'red', 'Reverse_Baseline': 'orange', 'Repeat_Baseline': 'blue'}
            for b_type, color in colors.items():
                b_rows = group[group['run_type'] == b_type]
                if not b_rows.empty:
                    val = b_rows['accuracy'].mean()
                    plt.axhline(val, color=color, linestyle='--', label=f"{b_type} ({val}%)")
            
            plt.title(f"Performance Analysis: {model} on {dataset}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig(os.path.join(target_dir, "comparison_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f" [OK] 已完成: {model} / {dataset}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./steering_report")
    args = parser.parse_args()

    all_runs = []
    log_files = glob(os.path.join(args.log_dir, "**/*.log"), recursive=True)
    for f in log_files:
        all_runs.extend(extract_runs_from_file(f))
    
    df = pd.DataFrame(all_runs)
    if not df.empty:
        os.makedirs(args.out_dir, exist_ok=True)
        df.to_csv(os.path.join(args.out_dir, "global_results.csv"), index=False)
        process_and_save(df, args.out_dir)
    else:
        print("未找到有效结果。")

if __name__ == "__main__":
    main()