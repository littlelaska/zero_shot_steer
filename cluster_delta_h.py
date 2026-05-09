# 对多个数据集同时进行delta h的分析，是否逻辑推理数据集将集中在某个区域

import torch
import json
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入你之前的 build_prompts 等逻辑（此处省略，保持一致）
# ... [插入你 delta_h_cos.py 中的 build_prompts 和 ASSISTANT_PROMPT] ...

class DeltaHClusterer:
    def __init__(self, model_path, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()

    def get_delta_h(self, ex, layer_idx):
        text_s = build_prompts(ex, self.tokenizer, repeat=False)
        text_r = build_prompts(ex, self.tokenizer, repeat=True)
        
        def get_state(text):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
                return out.hidden_states[layer_idx][0, -1, :].detach().cpu().float()
        
        return (get_state(text_r) - get_state(text_s)).numpy()

    def run(self, dataset_configs, samples_per_data, layer_idx, out_path):
        all_vectors = []
        all_labels = []
        
        for label, path in dataset_configs.items():
            print(f"Processing {label}...")
            with open(path, 'r') as f:
                data = [json.loads(line) for line in f] if path.endswith('.jsonl') else json.load(f)
            
            for ex in tqdm(data[:samples_per_data]):
                vec = self.get_delta_h(ex, layer_idx)
                all_vectors.append(vec)
                all_labels.append(label)

        # 降维分析
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        proj = tsne.fit_transform(pd.DataFrame(all_vectors))
        
        df = pd.DataFrame({'x': proj[:, 0], 'y': proj[:, 1], 'Dataset': all_labels})
        
        plt.figure(figsize=(10, 8), dpi=300)
        sns.scatterplot(data=df, x='x', y='y', hue='Dataset', style='Dataset', s=60, alpha=0.7)
        plt.title(f"t-SNE Clustering of $\Delta h$ at Layer {layer_idx}")
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.savefig(out_path)
        print(f"Cluster plot saved to {out_path}")

if __name__ == "__main__":
    # 此处编写 argparse，接收一组 label:path 格式的参数
    # 例如: --data "Logic:./logic.json" "Math:./math.json"
    pass