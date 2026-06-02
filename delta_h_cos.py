# 计算cos_d 的相似度，运行脚本是run_delha_h.sh
import torch
import json
import argparse
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# ==========================================
# 1. 基础组件 (同步自 zero_shot_steering_test.py)
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
    """同步论文中的 Query + Query 构造逻辑"""
    ctx = ex.get("context", "")
    q = ex.get("question", "").lstrip("Question: ")   # 针对commonsense数据集需要去掉前面的question
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
# 2. 多层表征分析器
# ==========================================

class AlignedLayerAnalyzer:
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
        self.num_layers = self.model.config.num_hidden_layers
        self.model.eval()

    def get_all_layer_last_token_states(self, text, max_length=2048):
        """同步提取逻辑：提取 Last Token 的所有隐藏层状态"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # outputs.hidden_states 是一个元组 (embedding, layer1, ..., layerL)
            # 取每一层的最后一个 token: [batch=1, -1, dim] -> [dim]
            all_layers = torch.stack([layer[0, -1, :].detach().cpu().float() for layer in outputs.hidden_states])
        return all_layers # [L+1, Hidden_Dim]

    def extract_dataset_deltas(self, file_path, num_samples):
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        
        samples = data[:num_samples]
        batch_deltas = []
        
        for ex in tqdm(samples, desc=f"Extracting {os.path.basename(file_path)}"):
            # 1. 构造 Single 和 Repeat 的文本
            text_s = build_prompts(ex, self.tokenizer, repeat=False)
            text_r = build_prompts(ex, self.tokenizer, repeat=True)
            # print("The repeat prompt is:\n", text_r)  # 打印重复构造的提示，便于调试

            # 2. 获取所有层的隐向量
            h_s = self.get_all_layer_last_token_states(text_s)
            h_r = self.get_all_layer_last_token_states(text_r)
            
            # 3. 计算 Δh 并存储
            batch_deltas.append(h_r - h_s) # [L+1, Dim]
            
        return torch.stack(batch_deltas) # [N, L+1, Dim]

    def run_analysis(self, data_a, data_b, num_samples, output_dir):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        # 提取两个数据集的偏移向量矩阵
        deltas_a = self.extract_dataset_deltas(data_a, num_samples)
        deltas_b = self.extract_dataset_deltas(data_b, num_samples)
        
        similarities = []
        for l in range(self.num_layers + 1):
            # 老版本代码，直接计算了每个样本的余弦相似度后取平均，但这样会受到样本内噪声的影响。
            # # 去中心化：计算该层的平均任务向量
            # mean_a = deltas_a[:, l, :].mean(dim=0, keepdim=True)
            # mean_b = deltas_b[:, l, :].mean(dim=0, keepdim=True)
            # 提取该层所有样本的 Delta h
            layer_deltas_a = deltas_a[:, l, :]  # [N, Dim]
            layer_deltas_b = deltas_b[:, l, :]  # [N, Dim]
            
            # 【核心改进】先对每个样本进行 L2 归一化，消除模长（能量）带来的偏置 
            norm_a = F.normalize(layer_deltas_a, p=2, dim=-1)
            norm_b = F.normalize(layer_deltas_b, p=2, dim=-1)
            
            # 计算去中心化/无视模长干扰的质心方向 
            mean_a = norm_a.mean(dim=0, keepdim=True)
            mean_b = norm_b.mean(dim=0, keepdim=True)
            
            # 计算余弦相似度
            sim = F.cosine_similarity(mean_a, mean_b).item()
            similarities.append(sim)
        
        # 绘图逻辑 (NeurIPS 风格)
        self.plot_results(similarities, output_dir)
        
        # 保存原始数据
        with open(os.path.join(output_dir, "layer_sim.json"), "w") as f:
            json.dump({"similarities": similarities, "layers": list(range(self.num_layers + 1))}, f)

    def plot_results(self, sims, output_dir):
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(range(len(sims)), sims, color='#02927D', marker='o', markersize=4, linewidth=2, label='Similarity')
        plt.axhline(y=0.8, color='#E6614F', linestyle='--', alpha=0.5)
        plt.title("Task Vector Consistency across Layers (Last Token)", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Cosine Similarity", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "similarity_curve.png"))
        print(f"Plot saved to {output_dir}/similarity_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_a", type=str, required=True)
    parser.add_argument("--data_b", type=str, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out", type=str, default="./layer_analysis")
    args = parser.parse_args()
    
    analyzer = AlignedLayerAnalyzer(args.model, args.gpu)
    analyzer.run_analysis(args.data_a, args.data_b, args.samples, args.out)