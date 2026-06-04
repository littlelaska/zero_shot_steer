# -*- coding: utf-8 -*-
import json
import argparse
import os
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from typing import List

# ==========================================
# 1. 基础组件 (完全同步自你的 zero_shot_steering.py)
# ==========================================

ASSISTANT_PROMPT = (
    "You are a logical task solver. Read the context, question and options carefully. "
    "First, provide a step-by-step reasoning chain to solve the problem. "
    "Finally, conclude your answer by strictly outputting the single option letter "
    "enclosed in LaTeX box format, for example: \\boxed{A}."
)

def load_data_file(path: str, max_n: int = None):
    data = []
    if not os.path.exists(path):
        print(f"[ERROR] 文件不存在: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content: return []
            try:
                full_data = json.loads(content)
                if isinstance(full_data, list):
                    return full_data[:max_n] if max_n else full_data
                elif isinstance(full_data, dict):
                    return [full_data]
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    if not line.strip(): continue
                    try:
                        data.append(json.loads(line))
                    except: continue
                    if max_n is not None and len(data) >= max_n:
                        break
    except Exception as e:
        print(f"[ERROR] 读取文件失败 {path}: {e}")
    return data

def _format_options_from_ex(ex):
    opt_obj = ex.get("options", [])
    if isinstance(opt_obj, list):
        return "Options:\n" + "\n".join(opt_obj)
    if isinstance(opt_obj, dict):
        return "Options:\n" + "\n".join([f"{k}) {v}" for k, v in opt_obj.items()])
    return ""

def build_prompts(ex, tokenizer=None, repeat=False, repeat_times=1):
    ctx = ex.get("context", "")
    q = ex.get("question", "").lstrip("Question: ")  # 针对 commonsense 数据集剔除前缀
    opts = _format_options_from_ex(ex)
    
    tail_prompt = "Please provide the reasoning and the answer."
    base_query = f"Context:\n{ctx}\n\nQuestion:\n{q}\n\n{opts}\n\n"
    
    if repeat:
        if repeat_times > 1:
            user_content = (base_query * repeat_times) + tail_prompt
        else:
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
# 2. 动态向量提取与正交分解核心引擎
# ==========================================

class SubspaceDecomposer:
    def __init__(self, model_path, batch_size=4, max_length=2048):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"Loading tokenizer & model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, prompts: List[str], layer_idx: int):
        """批量提取目标层最后一个 Token 的隐状态"""
        all_hiddens = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.model.device)
            
            outputs = self.model(**inputs, output_hidden_states=True)
            # 严格对齐你代码中的层索引映射机制
            target_idx = layer_idx + 1 if layer_idx >= 0 else layer_idx
            hidden = outputs.hidden_states[target_idx]
            
            last_hidden = hidden[:, -1, :].detach().cpu().float()
            all_hiddens.append(last_hidden)
            
        return torch.cat(all_hiddens, dim=0)

    def run_analysis(self, dataset_configs, samples_per_data, layer_idx, k_variance_ratio=0.85, m_core_components=3):
        """
        根据数据集路径动态提取 Delta h，并进行子空间正交分解
        """
        normalized_deltas = {}
        
        logic_names = ['LogicalDeduction', 'FOLIO', 'ProofWriter', 'AR-LSAT', 'ProntoQA']
        other_names = ['gsm8k', 'commonsenseQA']
        
        # 1. 动态循环读取路径并提取隐状态差分 Delta h
        for label, file_path in dataset_configs.items():
            print(f"\nProcessing Dataset: {label} from {file_path}")
            raw_data = load_data_file(file_path, max_n=samples_per_data)
            if not raw_data:
                print(f"[Warning] 数据集 {label} 为空或不存在，跳过。")
                continue
                
            prompts_s = [build_prompts(x, self.tokenizer, repeat=False) for x in raw_data]
            prompts_r = [build_prompts(x, self.tokenizer, repeat=True) for x in raw_data]
            
            print(f" -> 正在提取基准状态 h_single...")
            h_s = self.extract_features(prompts_s, layer_idx)
            print(f" -> 正在提取重复状态 h_repeat...")
            h_r = self.extract_features(prompts_r, layer_idx)
            
            # 计算独立 Delta h
            deltas = (h_r - h_s).numpy()
            
            # 预先进行 L2 归一化，使得子空间计算专注于“方向”而非单个样本的文本长度能量
            norm = np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-8
            normalized_deltas[label] = deltas / norm

        # 2. 拼建立空间矩阵
        X_logic_list = [normalized_deltas[name] for name in logic_names if name in normalized_deltas]
        X_other_list = [normalized_deltas[name] for name in other_names if name in normalized_deltas]
        
        if not X_logic_list or not X_other_list:
            print("[Error] 逻辑组或对比组的数据未成功提取，无法进行正交分解。")
            return

        X_logic = np.concatenate(X_logic_list, axis=0)
        X_other = np.concatenate(X_other_list, axis=0)
        
        print(f"\n[Matrix Info] 结合后的逻辑组矩阵形状: {X_logic.shape}, 对比组（背景噪声）矩阵形状: {X_other.shape}")

        # 3. 对对比组进行奇异值分解（SVD），构建非纯逻辑的背景子空间基底 U_other
        print("\n[Step 1] 对后2个非逻辑数据集（gsm8k, commonsenseQA）进行 SVD 降维...")
        _, singular_values, Ut = np.linalg.svd(X_other, full_matrices=False)
        U_SVD = Ut.T # 每一列为一个高维特征向量 [d, d]
        
        # 按照方差解释率动态选取需要剔除的干扰维度 k
        variance_explained = (singular_values ** 2) / np.sum(singular_values ** 2)
        cumulative_variance = np.cumsum(variance_explained)
        k = np.where(cumulative_variance >= k_variance_ratio)[0][0] + 1
        print(f" -> 选择剔除前 k={k} 个噪声主成分 (覆盖了对比组 {cumulative_variance[k-1]:.2%} 的方差)")
        
        U_other = U_SVD[:, :k] # 获取背景子空间基底 [d, k]

        # 4. 构造正交投影矩阵 P_perp = I - U_other * U_other^T
        print("\n[Step 2] 构造正交补空间投影矩阵算子...")
        d = X_logic.shape[1]
        I = np.eye(d)
        P_perp = I - np.dot(U_other, U_other.T) # [d, d]

        # 5. 洗净前5个数据集，彻底消去可由对比组解释的任何成分
        print("\n[Step 3] 对前5个逻辑推理数据集进行正交投影清洗...")
        X_logic_pure = np.dot(X_logic, P_perp) # [5N, d]

        # 6. 从清洗后的纯净矩阵中，再次通过 SVD 提取前 m 个最强的纯逻辑共性核心轴
        print(f"\n[Step 4] 从清洗后的数据中提取前 m={m_core_components} 个纯逻辑共有轴...")
        _, _, Ut_pure = np.linalg.svd(X_logic_pure, full_matrices=False)
        U_core_logic = Ut_pure.T[:, :m_core_components] # [d, m]
        
        # 7. 定量验证验证与交叉重构
        print("\n" + "="*50 + "\n[Step 5] 正交补子空间保留能量交叉验证报告...")
        
        verification_results = {}
        for name in logic_names + other_names:
            if name not in normalized_deltas: continue
            X_orig = normalized_deltas[name]
            # 将原始向量投影到这 m 个纯逻辑共有轴上
            projection = np.dot(X_orig, U_core_logic)
            # 计算重构能量（投影模长平方的均值）
            energy = np.mean(np.sum(projection ** 2, axis=1))
            verification_results[name] = energy
            
        for name, eng in verification_results.items():
            tag = "[逻辑推理组]" if name in logic_names else "[控制对比组]"
            print(f" 数据集 {name:<18} {tag} -> 纯逻辑共有子空间保留能量: {eng:.4f}")
            
        print("="*50 + "\n[Success] 子空间正交分解全流程分析完毕。")
        return U_core_logic, P_perp

# ==========================================
# 3. 命令行输入与入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subspace Orthogonal Decomposition for Delta h")
    parser.add_argument("--model", type=str, required=True, help="推理模型的本地绝对路径")
    parser.add_argument("--datasets", type=str, nargs='+', required=True, 
                        help="格式: Label:路径 (例如 FOLIO:./data/folio/train.json)")
    parser.add_argument("--samples", type=int, default=100, help="每个数据集抽取多少条样本参与分析")
    parser.add_argument("--batch_size", type=int, default=4, help="特征提取时的 Batch Size")
    parser.add_argument("--layer", type=int, default=16, help="需要分析的模型隐藏层索引")
    parser.add_argument("--max_length", type=int, default=2048, help="输入的最大截断长度")
    parser.add_argument("--variance_ratio", type=float, default=0.85, help="对比组干扰特征清洗保留率")
    parser.add_argument("--core_components", type=int, default=3, help="提取的共有逻辑轴数量")
    
    args = parser.parse_args()

    # 解析输入的外部数据集映射关系
    dataset_map = {}
    for item in args.datasets:
        if ':' in item:
            label, path = item.split(':', 1)
            dataset_map[label] = path
        else:
            print(f"[Warning] 跳过非法的输入格式: '{item}'，应符合 Label:Path")

    if not dataset_map:
        print("[Error] 未捕获到任何有效的数据集路径参数。")
        exit(1)

    decomposer = SubspaceDecomposer(args.model, batch_size=args.batch_size, max_length=args.max_length)
    decomposer.run_analysis(
        dataset_map, 
        samples_per_data=args.samples, 
        layer_idx=args.layer,
        k_variance_ratio=args.variance_ratio,
        m_core_components=args.core_components
    )