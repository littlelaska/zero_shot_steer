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
    def __init__(self, model_path, gpu_id, batch_size=8):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        print(f"Loading model: {model_path}...")
        # 必须设置 padding_side="left" 才能直接通过 [:, -1, :] 拿到最后的向量
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()
    
    @torch.no_grad()
    def get_batch_hiddens(self, prompts, layer_idx):
        """一次性获取整个 Batch 的 Last Token 隐状态"""
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(self.model.device)
        
        outputs = self.model(**inputs, output_hidden_states=True)
        # 由于是 left padding，无论长度如何，最后一个有效 token 都在索引 -1 处
        # 维度: [Batch, Hidden_Dim]
        hidden = outputs.hidden_states[layer_idx][:, -1, :].detach().cpu().float()
        return hidden

    # def get_delta_h(self, ex, layer_idx):
    #     """计算单个样本的 Delta h"""
    #     text_s = build_prompts(ex, self.tokenizer, repeat=False)
    #     text_r = build_prompts(ex, self.tokenizer, repeat=True)
        
    #     def extract_last_hidden(text):
    #         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
    #         with torch.no_grad():
    #             out = self.model(**inputs, output_hidden_states=True)
    #             # 提取指定层最后一个 token 的状态
    #             return out.hidden_states[layer_idx][0, -1, :].detach().cpu().float()
        
    #     h_s = extract_last_hidden(text_s)
    #     h_r = extract_last_hidden(text_r)
    #     return (h_r - h_s).numpy()

    def run(self, dataset_configs, samples_per_data, layer_idx, out_path):
        all_vecs = []
        all_labels = []

        for label, file_path in dataset_configs.items():
            print(f"\nProcessing Dataset: {label} ({file_path})")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f] if file_path.endswith('.jsonl') else json.load(f)
            
            # 分 Batch 处理数据
            for i in tqdm(range(0, len(samples), self.batch_size), desc=f"Extracting {label}"):
                batch_ex = samples[i : i + self.batch_size]
                
                # 构造 Prompt 列表
                prompts_s = [build_prompts(ex, self.tokenizer, repeat=False) for ex in batch_ex]
                prompts_r = [build_prompts(ex, self.tokenizer, repeat=True) for ex in batch_ex]
                
                # 获取隐状态
                h_s = self.get_batch_hiddens(prompts_s, layer_idx)
                h_r = self.get_batch_hiddens(prompts_r, layer_idx)
                
                # 计算 Delta h: $h_r - h_s$
                deltas = (h_r - h_s).numpy()
                all_vecs.extend(deltas)
                all_labels.extend([label] * len(batch_ex))

        # 3. 执行 t-SNE 降维
        print("\nPerforming t-SNE reduction...")
        X = np.array(all_vecs)
        # 对原始向量做一次 L2 归一化，能让聚类更关注方向而非模长
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        
        tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42, init='pca')
        X_embedded = tsne.fit_transform(X_norm)

        # 4. 绘图
        df = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'Dataset': all_labels})
        plt.figure(figsize=(12, 10), dpi=300)
        sns.set_style("whitegrid")
        sns.scatterplot(data=df, x='x', y='y', hue='Dataset', style='Dataset', s=70, alpha=0.8, palette='viridis')
        
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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
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

    clusterer = DeltaHClusterer(args.model, args.gpu, batch_size=args.batch_size)
    clusterer.run(dataset_map, args.samples, args.layer, args.out)