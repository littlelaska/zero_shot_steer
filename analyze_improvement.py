# 用于生成性能提升热力图的分析脚本
# 需要在collect_results.py运行完成之后运行
# 运行脚本以直接加载run_steer.sh末尾

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def analyze_boost(csv_path, out_dir):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # 1. 提取 Baseline 数据
    baseline_df = df[df['run_type'] == 'Baseline'][['model', 'dataset', 'accuracy']]
    baseline_df = baseline_df.rename(columns={'accuracy': 'baseline_acc'})
    # 如果有多个重复的 Baseline，取平均（通常只有一个）
    baseline_map = baseline_df.groupby(['model', 'dataset'])['baseline_acc'].mean().to_dict()

    # 2. 提取 Steer 数据并计算增量 (ΔAcc)
    steer_df = df[df['run_type'] == 'Steer'].copy()
    
    def calculate_diff(row):
        base = baseline_map.get((row['model'], row['dataset']), None)
        return row['accuracy'] - base if base is not None else None

    steer_df['improvement'] = steer_df.apply(calculate_diff, axis=1)
    steer_df = steer_df.dropna(subset=['improvement'])

    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 绘图 A: 参数敏感度热力图 (针对每个模型+数据集)
    # ---------------------------------------------------------
    for (model, dataset), group in steer_df.groupby(['model', 'dataset']):
        # 透视表：行=Layer, 列=Alpha, 值=Improvement
        pivot_table = group.pivot_table(index='layer', columns='alpha', values='improvement')
        
        plt.figure(figsize=(10, 8))
        # 使用 RdBu_r 颜色：蓝色代表提升，红色代表下降
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
        
        plt.title(f"Accuracy Boost (ΔAcc): {model} on {dataset}\n(Steer - Baseline)", fontsize=14)
        plt.xlabel("Intervention Alpha")
        plt.ylabel("Intervention Layer Index")
        
        img_name = f"boost_heatmap_{model}_{dataset}.png"
        plt.savefig(os.path.join(out_dir, img_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f" [Success] 参数热力图已保存: {img_name}")

    # ---------------------------------------------------------
    # 绘图 B: 任务增量汇总热力图 (模型 vs 数据集)
    # ---------------------------------------------------------
    # 找出每个组合下的最大提升
    summary_pivot = steer_df.pivot_table(index='dataset', columns='model', values='improvement', aggfunc='max')
    
    plt.figure(figsize=(12, 7))
    sns.heatmap(summary_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Max Logic Boost (ΔAcc) across Models & Tasks", fontsize=15, fontweight='bold')
    plt.xlabel("Model Scale")
    plt.ylabel("Evaluation Dataset")
    
    plt.savefig(os.path.join(out_dir, "global_boost_summary.png"), dpi=300, bbox_inches='tight')
    print(f" [Success] 全局增量汇总图已保存: global_boost_summary.png")

    # 保存计算后的数据
    steer_df.to_csv(os.path.join(out_dir, "detailed_improvements.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="global_results.csv 路径")
    parser.add_argument("--out", type=str, default="./improvement_analysis")
    args = parser.parse_args()
    analyze_boost(args.csv, args.out)