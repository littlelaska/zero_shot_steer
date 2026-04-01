# 针对抽取出的模型、数据集、层数以及准确率，进行一个全局的分析，验证是否存在一个普适的“演进规律”，即随着模型规模的增加，最佳表现层数是否也在增加，以及相对深度是否趋于稳定。
# 运行脚本在 run_full_analysis.sh 中，最终会输出一个综合分析报告，包括图表和统计摘要。

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np

# ---------------------------------------------------------
# 模型总层数硬编码表 (用于计算相对深度)
# ---------------------------------------------------------

MODEL_TOTAL_LAYERS = {
    # --- Qwen 系列 (Qwen 2 / 2.5) ---
    "Qwen2.5-0.5B": 24,
    "Qwen2.5-1.5B": 28,
    "Qwen2.5-3B": 36,
    "Qwen2.5-7B": 28,   # Qwen 7B 架构较为特殊，层数少但宽度大
    "Qwen2.5-14B": 48,
    "Qwen2.5-32B": 64,
    "Qwen2.5-72B": 80,
    "Qwen2-57B-A14B": 64, # MoE 版本

    # --- Llama 系列 (Llama 3 / 3.1 / 3.2 / 3.3) ---
    "Llama-3.2-1B": 16,
    "Llama-3.2-3B": 28,
    "Llama-3.1-8B": 32,
    "Llama-3-8B": 32,
    "Llama-3.1-70B": 80,
    "Llama-3.3-70B": 80,
    "Llama-3.1-405B": 126,

    # --- Gemma 系列 (Gemma 2 / 3) ---
    "Gemma-2-2B": 26,
    "Gemma-3-4B": 26,
    "Gemma-2-9B": 42,
    "Gemma-3-12B": 40,   # 预测/常用规格
    "Gemma-2-27B": 46,
    "Gemma-3-27B": 46,

    # --- Mistral / Mixtral 系列 ---
    "Mistral-7B": 32,
    "Mixtral-8x7B": 32,
    "Mixtral-8x22B": 56,
    "Mistral-Small": 40,
    "Mistral-Large": 80,

    # --- 其他 ---
    "DeepSeek-V3": 61,   # 除去 Embedding 和 Output 层
    "Phi-3-mini": 32,
    "Phi-3-medium": 40
}

def parse_model_size(model_name):
    """从模型名提取数值规模 (B)"""
    match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_name)
    return float(match.group(1)) if match else None

def get_total_layers(model_name):
    """
    智能匹配总层数。
    例如将 '/path/to/Qwen2.5-7B-Instruct' 提取出 'Qwen2.5-7B'
    """
    # 1. 提取核心名称 (去掉路径和后缀)
    # 比如 /data/Qwen2.5-7B-Instruct -> Qwen2.5-7B
    clean_name = os.path.basename(model_name).split('-Instruct')[0].split('-it')[0]
    
    # 2. 模糊匹配配置表
    for key, val in MODEL_TOTAL_LAYERS.items():
        if key.lower() in model_name.lower():
            return val
            
    # 3. 如果找不到，尝试从 config.json 实时读取 (需要安装 transformers)
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        return config.num_hidden_layers
    except:
        return None

def analyze(csv_path, out_dir):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. 加载并清洗数据
    df = pd.read_csv(csv_path)
    df['model_size'] = df['model'].apply(parse_model_size)
    df['total_layers'] = df['model'].apply(get_total_layers)
    
    # 过滤掉无法识别规模或总层数的数据
    df = df.dropna(subset=['model_size', 'total_layers'])
    
    # 2. 计算相对深度 (0.0 - 1.0)
    df['relative_depth'] = df['layer'] / df['total_layers']

    # 3. 提取核心结论：每个 (模型, 数据集, Alpha) 下的最佳表现层
    # 先按准确率降序，然后去重保留最高的那一行
    best_df = df.sort_values('accuracy', ascending=False).drop_duplicates(['model', 'dataset', 'alpha'])

    # 4. 绘图：双面板学术图表
    sns.set_theme(style="whitegrid", font="serif")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: 绝对规模 vs 最佳层数 (证明演进规律)
    sns.regplot(data=best_df, x="model_size", y="layer", ax=ax1, 
                scatter_kws={'s':120, 'alpha':0.6, 'edgecolor':'w'}, 
                line_kws={'color':'#e74c3c', 'linestyle':'--'}, ci=None)
    ax1.set_title("A: Absolute Optimal Layer index", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Model Scale (Billion Parameters)", fontsize=12)
    ax1.set_ylabel("Layer Index (h_i)", fontsize=12)

    # Panel B: 绝对规模 vs 相对深度 (证明一致性)
    sns.regplot(data=best_df, x="model_size", y="relative_depth", ax=ax2,
                scatter_kws={'s':120, 'alpha':0.6, 'color':'#2ecc71', 'edgecolor':'w'}, 
                line_kws={'color':'#27ae60', 'linestyle':'--'}, ci=None)
    ax2.set_title("B: Relative Optimal Depth (%)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Model Scale (Billion Parameters)", fontsize=12)
    ax2.set_ylabel("Relative Depth (Layer / Total)", fontsize=12)
    ax2.set_ylim(0, 1.0)
    
    # 在 B 图中画一条均值虚线
    mean_depth = best_df['relative_depth'].mean()
    ax2.axhline(mean_depth, color='gray', alpha=0.5, linestyle=':')
    ax2.text(best_df['model_size'].min(), mean_depth + 0.02, f'Mean: {mean_depth:.1%}', color='gray')

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存结果
    plot_path = os.path.join(out_dir, "scaling_law_comprehensive.png")
    best_df.to_csv(os.path.join(out_dir, "scaling_best_results.csv"), index=False)
    plt.savefig(plot_path, dpi=300)
    
    # 5. 打印统计摘要
    print("\n" + "="*40)
    print("      Scaling Law Analysis Report")
    print("="*40)
    print(f"样本模型总数: {best_df['model'].nunique()}")
    print(f"覆盖数据集: {best_df['dataset'].unique()}")
    print(f"平均最佳相对深度: {mean_depth:.2%}")
    print(f"深度稳定性 (Std): {best_df['relative_depth'].std():.4f}")
    print("-" * 40)
    print(f"结果文件已保存至: {out_dir}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="./scaling_report")
    args = parser.parse_args()
    analyze(args.csv, args.out)