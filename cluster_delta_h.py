import torch
import json
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# ==========================================
# 1. 基础组件 (同步自你的 delta_h_cos.py)
# ==========================================

ASSISTANT_PROMPT = (
    "You are a logical task solver. Read the context, question and options carefully. "
    "First, provide a step-by-step reasoning chain to solve the problem. "
    "Finally, conclude your answer by strictly outputting the single option letter "
    "enclosed in LaTeX box format, for example: \\boxed{A}."
)

def _format_options_from_ex(ex):
    opt_obj = ex.get("options", [])
    if isinstance(opt_obj, list):
        return "Options:\n" + "\n".join(opt_obj)
    if isinstance(opt_obj, dict):
        return "Options:\n" + "\n".join([f"{k}) {v}" for k, v in opt_obj.items()])
    return ""

def build_prompts(ex, tokenizer=None, repeat=False):
    ctx = ex.get("context", "")
    # 针对 commonsense 数据集去掉开头的 "Question: "
    q = ex.get("question", "").lstrip("Question: ")   
    opts = _format_options_from_ex(ex)
    
    tail_prompt = "Please provide the reasoning and the answer."
    base_query = f"Context:\n{ctx}\n\nQuestion:\n{q}\n\n{opts}\n\n"
    
    if repeat:
        user_content = base_query + base_query + tail_prompt
    else:
        user_content = base_query + tail_prompt

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template([
                {"role": "system", "content": ASSISTANT_PROMPT},
                {"role": "user", "content": user_content}
            ], tokenize=False, add_generation_prompt=True)
        except:
            return f"{ASSISTANT_PROMPT}\n\n{user_content}"
    return user_content

# ==========================================
# 2. 聚类分析核心类
# ==========================================

class DeltaHClusterer:
    def __init__(self, model_path, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

    def get_delta_h(self, ex, layer_idx):
        """计算单个样本的 Delta h"""
        text_s = build_prompts(ex, self.tokenizer, repeat=False)
        text_r = build_prompts(ex, self.tokenizer, repeat=True)
        
        def extract_last_hidden(text):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
                # 提取指定层最后一个 token 的状态
                return out.hidden_states[layer_idx][0, -1, :].detach().cpu().float()
        
        h_s = extract_last_hidden(text_s)
        h_r = extract_last_hidden(text_r)
        return (h_r - h_s).numpy()

    def run(self, dataset_configs, samples_per_data, layer_idx, out_path):
        all_vecs = []
        all_labels = []

        for label, file_path in dataset_configs.items():
            print(f"\nProcessing Dataset: {label} ({file_path})")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f] if file_path.endswith('.jsonl') else json.load(f)
            
            # 抽样提取
            samples = data[:samples_per_data]
            for ex in tqdm(samples, desc=f"Extracting {label}"):
                vec = self.get_delta_h(ex, layer_idx)
                all_vecs.append(vec)
                all_labels.append(label)

        # 3. 执行 t-SNE 降维
        print("\nPerforming t-SNE reduction...")
        X = np.array(all_vecs)
        # 对原始向量做一次 L2 归一化，能让聚类更关注方向而非模长
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42, init='pca')
        X_embedded = tsne.fit_transform(X_norm)

        # 4. 绘图
        df = pd.DataFrame({
            'x': X_embedded[:, 0],
            'y': X_embedded[:, 1],
            'Dataset': all_labels
        })

        plt.figure(figsize=(12, 10), dpi=300)
        sns.set_style("whitegrid")
        scatter = sns.scatterplot(
            data=df, x='x', y='y', hue='Dataset', 
            style='Dataset', s=70, alpha=0.8, palette='viridis'
        )
        
        plt.title(f"t-SNE of $\Delta h$ Vectors (Layer {layer_idx})\nModel: {os.path.basename(args.model)}", fontsize=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"\n[Success] Cluster plot saved to: {out_path}")

# ==========================================
# 3. Main 函数完善
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delta h Clustering Analysis")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--datasets", type=str, nargs='+', required=True, 
                        help="Format: Label:Path (e.g. FOLIO:./folio.json)")
    parser.add_argument("--samples", type=int, default=100, help="Samples per dataset")
    parser.add_argument("--layer", type=int, default=16, help="Layer index to analyze")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--out", type=str, default="cluster_plot.png", help="Output filename")
    
    args = parser.parse_args()

    # 解析 Label:Path 映射
    dataset_map = {}
    for item in args.datasets:
        if ':' in item:
            label, path = item.split(':', 1)
            dataset_map[label] = path
        else:
            print(f"Warning: Invalid dataset format '{item}', expected Label:Path")

    if not dataset_map:
        print("Error: No valid datasets provided.")
        exit(1)

    clusterer = DeltaHClusterer(args.model, args.gpu)
    clusterer.run(dataset_map, args.samples, args.layer, args.out)