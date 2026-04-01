# -*- coding: utf-8 -*-
import json
import argparse
import os
import re
import torch
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import torch.nn as nn
from typing import Dict, List

# ==========================================
# 1. 基础组件与数据处理
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
            
            # 1. First, try parsing the entire file as a single JSON object/array
            try:
                full_data = json.loads(content)
                if isinstance(full_data, list):
                    return full_data[:max_n] if max_n else full_data
                elif isinstance(full_data, dict):
                    # If the root is a dict, you might need to extract the actual list
                    # e.g., return full_data['data'][:max_n] 
                    return [full_data]
            
            # 2. If it fails, fallback to parsing line-by-line (JSONL)
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    if not line.strip(): continue
                    try:
                        data.append(json.loads(line))
                    except:
                        continue
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

def build_prompts(ex, tokenizer=None, repeat=False, reverse_context=False, pad_repeat=False):
    """
    构建 Prompt。如果 repeat=True，则应用论文中的 Query + Query 策略。
    - repeat=True: 语义重复 Query + Query（论文里的 Prompt Repetition）。
    - pad_repeat=True: 用 pad 字符把 token 长度扩到约 2 倍（语义不变），用于和 repeat 对比。
    """
    ctx = ex.get("context", "")
    q = ex.get("question", "")
    opts = _format_options_from_ex(ex)
    
    tail_prompt = "Please provide the reasoning and the answer."
    base_query = f"Context:\n{ctx}\n\nQuestion:\n{q}\n\n{opts}\n\n"
    if reverse_context:
        base_query = f"Question:\n{q}\n\n{opts}\n\nContext:\n{ctx}\n\n"
    
    # 情况 1：重复语义的 Query + Query
    # 核心：复现论文的 Prompt Repetition
    if repeat and not pad_repeat:
        # 你也可以在这里尝试论文里的变体：base_query + "\n\nLet me repeat that:\n\n" + base_query
        user_content = base_query + base_query + tail_prompt
    else:
        user_content = base_query + tail_prompt
    
    # 注意：
    # pad_repeat 产生“真正的 pad token（attention_mask=0）”应在 tokenizer 编码阶段用 padding='max_length' 实现，
    # 而不是在文本层面追加任何字符。这里保留参数仅用于上层逻辑分支。

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template([
                {"role": "system", "content": ASSISTANT_PROMPT},
                {"role": "user", "content": user_content}
            ], tokenize=False, add_generation_prompt=True)
        except:
            return f"{ASSISTANT_PROMPT}\n\n{user_content}"
    return user_content

def check_is_correct(prediction, ground_truth):
    if not prediction or not ground_truth: return False
    ground_truth = ground_truth.strip().upper()
    
    matches = re.findall(r"\\boxed\{([A-G])\}", prediction)
    if matches: return matches[-1] == ground_truth

    patterns = [r"Final Answer:.*?([A-G])", r"The correct answer is.*?([A-G])"]
    for p in patterns:
        match = re.search(p, prediction, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).upper() == ground_truth

    clean_text = re.sub(r"[^A-G]", "", prediction.split("Answer")[-1])
    if clean_text: return clean_text[-1] == ground_truth
    return False

# ==========================================
# 2. 核心算法：Zero-shot Activation Steering
# ==========================================

class ActivationSteerer:
    def __init__(self, model, tokenizer, layer_idx: int, max_length: int, batch_size: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = model.device
        self.steering_vector = None # 用于存储计算出的 Δh
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.batch_size = batch_size
        if max_length is not None:   # 非空的时候
            self.max_length = max_length
            self.padding = "max_length"
        else:
            self.max_length = 8192
            self.padding = True # 控制输入的最大长度，对所有的batch padding到这个长度，避免由于不同padding带来的性能差异

    def _get_layer_module(self):
        """自动寻找各模型的 transformer layers 容器"""
        if hasattr(self.model, "language_model"): # Gemma 3
            if hasattr(self.model.language_model, "layers"):
                return self.model.language_model.layers[self.layer_idx]
            if hasattr(self.model.language_model, "model"):
                return self.model.language_model.model.layers[self.layer_idx]
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"): # Llama/Qwen
            return self.model.model.layers[self.layer_idx]
        if hasattr(self.model, "layers"): # Gemma 2
            return self.model.layers[self.layer_idx]
        raise AttributeError(f"Could not find layers in {type(self.model)}")

    @torch.no_grad()
    def extract_features(self, prompts: List[str], batch_size: int, max_length: int = None):
        """提取指定层最后一个 Token 的隐状态"""
        # laska 修改，新增 max_length 参数，控制输入的最大长度，对所有的batch padding到这个长度，避免由于不同padding带来的性能差异
        if max_length is not None:
            padding = "max_length"
        else:
            max_length = self.max_length
            padding = True
        all_hiddens = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            # inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(self.device)
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=padding, truncation=True, max_length=max_length).to(self.device)
            
            outputs = self.model(**inputs, output_hidden_states=True)
            target_idx = self.layer_idx + 1 if self.layer_idx >= 0 else self.layer_idx
            hidden = outputs.hidden_states[target_idx] 
            
            last_hidden = hidden[:, -1, :].detach().float()
            all_hiddens.append(last_hidden)
            
        return torch.cat(all_hiddens, dim=0)

    def compute_steering_vector(self, data_samples):
        """
        Step 1 & 2 & 3: 计算 h_single 和 h_repeat，求差分并平均
        """
        batch_size = self.batch_size
        # print(f"\n[Steering] Computing difference vector over {len(data_samples)} calibration samples...")
        print(f"\n[Steering] Computing normalized difference vector...")
        prompts_single = [build_prompts(x, self.tokenizer, repeat=False) for x in data_samples]
        prompts_repeat = [build_prompts(x, self.tokenizer, repeat=True) for x in data_samples]
        
        # 1. 提取两种 Prompt 的隐状态，特征
        h1 = self.extract_features(prompts_single, batch_size)
        h2 = self.extract_features(prompts_repeat, batch_size)

        # # 2. 计算平均差异向量
        # diffs = h2 - h1 
        # mean_diff = diffs.mean(dim=0) # [D]

        # # 3. L2 归一化核心逻辑
        # # 使用 FP32 计算模长以确保稳定性
        # norm = torch.norm(mean_diff, p=2)   # p=2代表用L2范数

        # if norm > 0:
        #     self.steering_vector = mean_diff / norm
        #     print(f" -> Original norm: {norm:.4f}, Vector has been normalized to unit length.")
        # else:
        #     self.steering_vector = mean_diff
        #     print(" [Warning] Difference vector norm is 0, skipping normalization.")

        # 差分: Δh = h2 - h1
        diffs = h2 - h1 
        # 归因均值: 计算整个校准集的平均方向
        self.steering_vector = diffs.mean(dim=0) 
        print(f"[Steering] Vector computed. L2 Norm: {torch.norm(self.steering_vector):.4f}")
        return self.steering_vector

    def _tokenize_pad_repeat(self, prompts: List[str], pad_factor: int, truncation_max_length: int):
        """
        先获取每条样本在不 padding 下的长度，再用 padding='max_length' 补齐到 pad_factor 倍长度。
        这样补出来的 token 会满足 attention_mask==0，可用于统计“额外 pad token”比例。
        """
        if self.tokenizer.pad_token_id is None:
            # 与 __init__ 保持一致：若无 pad_token，用 eos 兜底
            self.tokenizer.pad_token = self.tokenizer.eos_token

        pad_id = self.tokenizer.pad_token_id
        assert pad_id is not None

        input_ids_list = []
        attn_list = []
        max_L = 0

        for p in prompts:
            # 不 padding 的真实长度（仍按 truncation_max_length 截断）
            ids = self.tokenizer(
                p,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=truncation_max_length,
            )["input_ids"]
            real_len = len(ids)
            target_len = max(1, pad_factor * real_len)
            if truncation_max_length is not None:
                target_len = min(target_len, truncation_max_length)

            enc = self.tokenizer(
                p,
                return_tensors="pt",
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=target_len,
            )
            input_ids = enc["input_ids"][0]
            attn = enc["attention_mask"][0]

            input_ids_list.append(input_ids)
            attn_list.append(attn)
            max_L = max(max_L, int(input_ids.shape[0]))

        # batch 内再 pad 到同一个 max_L（也是 pad token, attention_mask=0）
        batch_input_ids = torch.full((len(prompts), max_L), pad_id, dtype=torch.long)
        batch_attn = torch.zeros((len(prompts), max_L), dtype=torch.long)
        for i, (ids, attn) in enumerate(zip(input_ids_list, attn_list)):
            L = int(ids.shape[0])
            batch_input_ids[i, -L:] = ids  # 保持 left padding 习惯
            batch_attn[i, -L:] = attn

        return {"input_ids": batch_input_ids.to(self.device), "attention_mask": batch_attn.to(self.device)}

    def generate_with_steering(
        self,
        prompts: List[str],
        alpha: float = 1.0,
        intervention_mode: str = "static",
        max_length: int = None,
        pad_repeat: bool = False,
        pad_factor: int = 2,
    ):
        """
        Step 4: 将向量注入到残差流进行干预
        """
        if self.steering_vector is None and alpha != 0.0:
            raise ValueError("Steering vector not computed! Run compute_steering_vector first.")
        if max_length is None:
            max_length = self.max_length
            padding = True
        else:
            padding = "max_length"
        if alpha != 0.0:   # 对中间向量进行干预
            print(f"\n[Steering] Applying steering with alpha={alpha} in {intervention_mode} mode...")
            if pad_repeat:
                inputs = self._tokenize_pad_repeat(prompts, pad_factor=pad_factor, truncation_max_length=max_length)
            else:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=padding, truncation=True, max_length=max_length).to(self.device)
            print(inputs.input_ids.shape)
            # 移除这里的 .to(self.device)，我们将在 hook 中动态匹配设备
            vec_base = (self.steering_vector * alpha).to(self.model.dtype)

            def adapter_hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                seq_len = h.shape[1]
                # 新增监控norm的逻辑
                # --- 新增监控逻辑 ---
                with torch.no_grad():
                    # 计算当前 Batch 最后一个 token 的原始范数 [B]
                    orig_norm = torch.norm(h[:, -1, :], p=2, dim=-1).mean().item()
                    orig_std = torch.norm(h[:, -1, :], p=2, dim=-1).std().item()
                    # 计算干预项的范数
                    steer_norm = torch.norm(vec_base, p=2, dim=-1).item()
                    ratio = steer_norm / orig_norm if orig_norm != 0 else 0
                    
                    # 仅在 Prefill 阶段或第一个 Token 时打印，避免日志刷屏
                    if seq_len > 1:
                        print(f" > [Layer {self.layer_idx}] Norm Ratio: {ratio:.2%} (Orig: {orig_norm:.2f}, Steer: {steer_norm:.2f}), Orig Std: {orig_std:.2f}")
                # ------------------ 判断是否干预 ------------------
                
                should_intervene = False
                if intervention_mode == "static" and seq_len > 1:
                    should_intervene = True # 仅在 Prefill 阶段干预
                elif intervention_mode == "dynamic":
                    should_intervene = True # 持续干预

                if should_intervene:
                    # 【修改点】动态匹配当前层所在设备 (Dynamic device matching)
                    vec_inject = vec_base.to(h.device)
                    
                    # 直接加上我们计算好的差分向量
                    h[:, -1, :] = h[:, -1, :] + vec_inject
                    # h[:, -1, :] = vec_inject
                    
                return (h,) + output[1:] if isinstance(output, tuple) else h

            layer_module = self._get_layer_module()
            handle = layer_module.register_forward_hook(adapter_hook)
        else:
            print(f"\n[Steering] Alpha is 0.0, no intervention applied.")
            if pad_repeat:
                inputs = self._tokenize_pad_repeat(prompts, pad_factor=pad_factor, truncation_max_length=max_length)
            else:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=padding, truncation=True, max_length=max_length).to(self.device)
            # ========= 统计本 batch 的 pad 比例（仅统计“batch padding”产生的 pad token）=========
            # 说明：
            # - attention_mask==0 的位置才是 tokenizer 自动补的 pad（batch 对齐 / max_length 对齐）
            # - 若启用 pad_repeat，这里的 pad_len 会包含“补齐到 pad_factor 倍长度”产生的 pad token
            with torch.no_grad():
                input_ids = inputs["input_ids"]
                attn = inputs["attention_mask"]
                B, L = input_ids.shape

                # 1) 基于 attention_mask 统计真正的 tokenizer padding 数量
                real_len = attn.sum(dim=-1)                 # [B]
                pad_len = (attn == 0).sum(dim=-1)           # [B]

                # 2) 额外诊断：不做 padding 时每条样本的长度（已 truncation 到 max_length）
                #    用于判断 pad_len==0 是因为“长度本来就一样”，还是因为“全部被截断到同一长度”
                no_pad = self.tokenizer(
                    prompts,
                    return_tensors=None,
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                )
                no_pad_lens = torch.tensor([len(x) for x in no_pad["input_ids"]], device=attn.device)  # [B]

                pad_real_ratio = pad_len.float() / (no_pad_lens.float() + 1e-8)
                pad_total_ratio = pad_len.float() / float(L)

                any_truncated = (no_pad_lens == max_length).any().item() if max_length is not None else False
                frac_truncated = (no_pad_lens == max_length).float().mean().item() if max_length is not None else 0.0

                print(f"[Padding Stats] Batch size={B}, padded_seq_len(L)={L}, max_length={max_length}")
                print(f"  no_pad_len(min/mean/max) = {no_pad_lens.min().item():.0f} / {no_pad_lens.float().mean().item():.2f} / {no_pad_lens.max().item():.0f}")
                print(f"  pad_len  (min/mean/max)  = {pad_len.min().item():.0f} / {pad_len.float().mean().item():.2f} / {pad_len.max().item():.0f}")
                print(f"  mean_pad/real            = {pad_real_ratio.mean().item():.4f}")
                print(f"  mean_pad/total           = {pad_total_ratio.mean().item():.4f}")
                print(f"  truncated_any={bool(any_truncated)} truncated_frac={frac_truncated:.2%}")
            # ====================================================================
            # print("=*20the inputs size is ")
            # print(inputs.input_ids.shape)
            # exit()
            handle = None
        
        self.model.eval()
        try:
            with torch.no_grad():
                gen_out = self.model.generate(
                    **inputs,
                    max_new_tokens=4096, # 根据推理任务调整
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        finally:
            if alpha != 0.0 and handle is not None:
                handle.remove() # 确保 Hook 被移除
            
        return self.tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    
    # ... 原有的 __init__ 和 extract_features 保持不变 ...

    def generate_with_instance_steering(self, prompts: List[str], alpha: float = 1.0, intervention_mode: str = "static", max_length: int = None):
        """
        [新增功能] 针对每个 Query 实时计算 Δh 并干预
        """
        # laska 新增，初始化maxlength，避免不同padding长度带来的影响
        if max_length is None:
            padding = True
            max_length = self.max_length
        else:
            padding = "max_length"
        # 1. 准备 Single 和 Repeat 两种 Prompt
        prompts_single = [p for p in prompts] # 这里的 p 已经是 build_prompts(repeat=False) 后的结果
        # 注意：这里需要重新 build 带有 repeat=True 的版本用于计算向量
        # 为了方便，我们假设传入的是原始 data 列表，或者在外部处理好。
        # 这里演示在内部重新构建：
        
        # 2. 实时计算当前 Batch 的专属 Δh
        # 注意：这里需要调用你之前定义的 build_prompts 逻辑，或者传入已处理好的 prompts
        # 为了逻辑清晰，我们假设此函数接收的是 list of dict (raw_data)
        raw_samples = prompts # 假设此时传入的是原始数据列表
        p_s = [build_prompts(x, self.tokenizer, repeat=False) for x in raw_samples]
        p_r = [build_prompts(x, self.tokenizer, repeat=True) for x in raw_samples]

        print(f" -> Calculating instance-specific Δh for batch (size={len(raw_samples)})...")
        h1 = self.extract_features(p_s, batch_size=len(p_s)) # [B, D]
        h2 = self.extract_features(p_r, batch_size=len(p_r)) # [B, D]
        
        # 计算每一条数据自己的差分向量，未进行归一化的版本
        # batch_diffs = (h2 - h1) * alpha # [B, D]
        # batch_diffs = batch_diffs.to(self.model.dtype)
        
        # 计算差异
        batch_diffs = h2 - h1 # [B, D]
        # 新增的归一化操作
        # 对每一行（每个样本）独立计算 L2 Norm
        # keepdim=True 是为了后续的广播计算 [B, 1]
        norms = torch.norm(batch_diffs, p=2, dim=-1, keepdim=True)
        print("original norms:", norms.squeeze().tolist())  # 打印原始模长以供调试

        # 避免除以 0
        normalized_diffs = batch_diffs / (norms + 1e-8)
        
        # 最后应用 alpha 强度
        batch_diffs = (normalized_diffs * alpha).to(self.model.dtype)
        # 3. 定义适配 Batch 的 Hook
        def instance_adapter_hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            seq_len = h.shape[1]
            
            # 判断是否干预（逻辑与原代码一致）
            should_intervene = False
            if intervention_mode == "static" and seq_len > 1:
                should_intervene = True 
            elif intervention_mode == "dynamic":
                should_intervene = True

            if should_intervene:
                # 动态匹配设备并将 Δh 注入对应的样本
                vec_inject = batch_diffs.to(h.device)
                
                # h 的形状是 [B, L, D]，我们要把 batch_diffs [B, D] 加到最后一个 token [B, -1, D]
                # 这一行是核心：利用广播机制或直接索引加法
                h[:, -1, :] = h[:, -1, :] + vec_inject
                
            return (h,) + output[1:] if isinstance(output, tuple) else h

        # 4. 执行推理
        # inputs = self.tokenizer(p_s, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(self.device)
        inputs = self.tokenizer(p_s, return_tensors="pt", padding=padding, truncation=True, max_length=max_length).to(self.device)
        layer_module = self._get_layer_module()
        handle = layer_module.register_forward_hook(instance_adapter_hook)
        
        self.model.eval()
        try:
            with torch.no_grad():
                gen_out = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        finally:
            handle.remove()
            
        return self.tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    
    # 新增功能：Logit Lens 分析干预向量的语义信息
    @torch.no_grad()
    def analyze_steering_vector(self, top_k: int = 10):
        """
        使用 Logit Lens 技术将干预向量投影到词表空间，查看其包含的语义信息。
        """
        if self.steering_vector is None:
            print("Error: No steering vector found. Please compute it first.")
            return

        # 1. 获取模型最后的归一化层和输出头
        # 不同模型的命名不一致，这里做通用适配
        if hasattr(self.model, "lm_head"):
            lm_head = self.model.lm_head
        elif hasattr(self.model.language_model, "lm_head"):
            lm_head = self.model.language_model.lm_head
        else:
            raise AttributeError("Could not find lm_head in model.")

        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"): # Llama/Qwen
            final_norm = self.model.model.norm
        elif hasattr(self.model.language_model, "model") and hasattr(self.model.language_model.model, "norm"): # Gemma 3
            final_norm = self.model.language_model.model.norm
        else:
            # 如果找不到，尝试直接搜索具有 LayerNorm/RMSNorm 类型的属性
            final_norm = next((m for m in self.model.modules() if "Norm" in type(m).__name__), None)

        # 2. 准备向量
        # 将向量转为模型精度，并添加 Batch 维度 [1, D]
        vec = self.steering_vector.to(device=self.model.device, dtype=self.model.dtype).unsqueeze(0)

        # 3. 核心步骤：投影
        # 重要：必须先经过模型最后的 LayerNorm/RMSNorm，否则分布会极度扭曲
        if final_norm:
            vec = final_norm(vec)
        
        # 投影到词表大小的 Logits 空间 [1, Vocab_Size]
        logits = lm_head(vec)
        
        # 4. 获取 Top-K Token
        probs = torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(probs, top_k)
        
        top_values = top_values.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()

        print(f"\n=== Logit Lens Analysis (Top {top_k} Tokens) ===")
        print(f"{'Token':<15} | {'Probability':<12}")
        print("-" * 30)
        for val, idx in zip(top_values, top_indices):
            token_str = self.tokenizer.decode([idx]).strip()
            # 转换一些不可见字符
            token_str = token_str.replace("\n", "\\n")
            print(f"{token_str:<15} | {val:.4%}")
    
    # 生成tsne降维图，展示不同题目类型的 Δh 分布情况
    @torch.no_grad()
    def analyze_delta_h_tsne(self, data_samples: List[dict], save_path: str = "tsne_distribution.png", label_key: str = None):
        """
        计算不同题目产生的 Δh，进行 t-SNE 降维并在本地保存空间分布图。
        
        Args:
            data_samples: 题目样本列表。
            save_path: 图片保存路径。
            label_key: (可选) data_samples 中用于区分题目类型的 key (例如 'task_type' 或 'source')。
                    如果提供，图表中的点将按类型着色。
        """
        N = len(data_samples)
        if N < 5:
            print("[Error] 样本数量太少（少于5个），无法进行有效的 t-SNE 分析。")
            return

        print(f"正在提取 {N} 个样本的独立 Δh 并进行 FP32 转换...")
        
        # 1. 提取隐状态并立即转为 FP32 (在 CPU 上计算降维，FP32 更稳健)
        prompts_s = [build_prompts(x, self.tokenizer, repeat=False) for x in data_samples]
        prompts_r = [build_prompts(x, self.tokenizer, repeat=True) for x in data_samples]
        
        # 获取隐状态 [N, D]
        h_s = self.extract_features(prompts_s, batch_size=self.batch_size).cpu().float()
        h_r = self.extract_features(prompts_r, batch_size=self.batch_size).cpu().float()
        
        # 2. 计算每个样本的独立 Δh 并进行 L2 归一化
        # 归一化很重要，因为 t-SNE 基于距离，我们关心的是方向差异
        deltas = h_r - h_s 
        deltas_norm = F.normalize(deltas, p=2, dim=-1).numpy() # 转换为 NumPy 用于 sklearn
        
        # 3. 准备标签 (用于着色)
        labels = []
        if label_key and N > 0 and label_key in data_samples[0]:
            labels = [x[label_key] for x in data_samples]
            print(f" -> 已根据 '{label_key}' 提取标签用于着色。")
        else:
            labels = ["All Samples"] * N # 如果没有标签，使用统一颜色
            print(" -> 未提供有效 label_key，所有点将使用统一颜色。")

        # 4. 执行 t-SNE 降维
        print(f"正在执行 t-SNE 降维 (维度: {deltas_norm.shape[1]} -> 2)...")
        
        # 参数调整建议：
        # perplexity: 困惑度，考虑局部邻居的数量。样本少设小点(5-30)，样本多设大点(30-50)。
        # random_state: 锁定随机种子，保证每次运行图形一致。
        tsne_model = TSNE(
            n_components=2, 
            perplexity=min(30, N - 1), # 自动调整 perplexity
            random_state=42, 
            max_iter=1000, 
            init='pca', # 使用 PCA 初始化能捕捉更好的全局结构
            n_jobs=-1 # 使用所有 CPU 核心
        )
        
        tsne_results = tsne_model.fit_transform(deltas_norm) # 结果形状 [N, 2]
        
        # 5. 使用 Pandas 和 Seaborn 绘图
        print("正在生成分布图...")
        df = pd.DataFrame({
            'tsne_1': tsne_results[:, 0],
            'tsne_2': tsne_results[:, 1],
            'Type': labels
        })
        
        plt.figure(figsize=(10, 8))
        
        # 根据是否有多种标签选择不同的绘图方式
        if len(set(labels)) > 1:
            scatter = sns.scatterplot(
                data=df, 
                x='tsne_1', y='tsne_2', 
                hue='Type', # 按类型着色
                style='Type', # 不同类型使用不同形状
                palette='viridis', # 颜色盘
                s=100, # 点的大小
                alpha=0.8 # 透明度
            )
        else:
            scatter = sns.scatterplot(
                data=df, 
                x='tsne_1', y='tsne_2', 
                s=100, color='royalblue', alpha=0.7
            )
        
        plt.title(f"t-SNE Visualization of Δh Distribution\n(Model: {self.model.config._name_or_path} | Layer: {self.layer_idx})", fontsize=14)
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 防止图例遮挡图像
        if len(set(labels)) > 1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # 6. 保存到本地
        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f" [Success] t-SNE 分布图已保存至: {os.path.abspath(save_path)}")
        
        plt.close()
        
        return tsne_results

# ==========================================
# 3. 主流程
# ==========================================


def main():
    parser = argparse.ArgumentParser()
    # 用少量的源域数据作为校准集（求差分），无需标签！
    parser.add_argument("--calib_file", type=str, required=True, help="用于计算干预向量的无标签校准集")
    parser.add_argument("--test_file", type=str, required=True, help="用于最终测试的文件")
    parser.add_argument("--output_file", type=str, default="steering_results.jsonl")
    
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it") 
    parser.add_argument("--layer", type=int, default=15, help="干预层 (通常中后期层 15-30 效果好)")
    
    parser.add_argument("--calib_samples", type=int, default=100, help="使用多少条数据计算均值向量")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--intervention_mode", type=str, default="static", choices=["static", "dynamic"])
    parser.add_argument("--alpha", type=float, default=1.0, help="干预强度")
    # laska 20260317 新的测试逻辑
    parser.add_argument("--reverse_context", default=False, action="store_true", help="是否对context进行后置操作")
    parser.add_argument("--instance_steering", default=False, action="store_true", help="是否从单个样例的角度对激活进行干预")
    parser.add_argument("--repeat", default=False, action="store_true", help="是否对prompt进行重复，作为一个baseline")
    parser.add_argument("--pad_repeat", default=False, action="store_true", help="是否使用pad字符把长度扩展到约2倍，作为对照baseline")
    parser.add_argument("--pad_factor", type=int, default=2, help="pad_repeat 时补齐倍率（默认 2 倍）")
    parser.add_argument("--max_length", type=int, help="控制输入的最大长度，对所有的batch padding到这个长度，避免由于不同padding带来的性能差异")
    parser.add_argument("--dataset", type=str, default="LogicalDeduction", help="当前测试的数据集名称，用于分析和命名输出文件")
    # 新增一个，选取部分数据用于测试
    parser.add_argument("--max_test_samples", type=int, default = 1000, help="如果指定，则仅使用前 N 条测试数据进行推理")

    args = parser.parse_args()

    print(f"=== Zero-shot Steering PoC ===")
    print(f"Model: {args.model}")
    print(f"Dataset:{args.test_file}")
    print(f"Layer: {args.layer} | Alpha: {args.alpha} | Mode: {args.intervention_mode}")
    print(f"==============================")
    
    # 1. Load Data
    if not args.instance_steering and args.alpha != 0.0:
        print(f"Loading calibration data from {args.calib_file} (max {args.calib_samples} samples)...")
        calib_data = load_data_file(args.calib_file, max_n=args.calib_samples)
    test_data = load_data_file(args.test_file, max_n=args.max_test_samples)
    
    if not args.instance_steering and args.alpha != 0.0:
        if not calib_data:
            print("[Error] Calibration data empty.")
            return
    if not test_data:
        print("[Error] Test Data empty.")
        return

    # 2. Load Model
    print(f"Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    # model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="auto")
    steerer = ActivationSteerer(model, tokenizer, layer_idx=args.layer, batch_size=args.eval_batch_size, max_length=args.max_length)

    # 3. Compute Steering Vector (Zero-shot, No Labels Needed)
    # laska修改，某些情况下不需要进入这个函数 1. instance steering 模式下每个样例单独计算向量 2. alpha=0 的情况下不需要计算向量（虽然不计算向量也不会报错，但为了效率我们直接跳过）
    if (not args.instance_steering) and (args.alpha != 0.0):
        steerer.compute_steering_vector(calib_data)
       
        # 新增一个对干预向量进行分析的步骤
        steerer.analyze_steering_vector(top_k=20)
        # 动态生成文件名
        tsne_file_name = f"analysis/{args.dataset}_tsne_layer{args.layer}_alpha{args.alpha}.png"
        
        # 调用分析，指定数据中用于着色的 key 为 'task_type'
        steerer.analyze_delta_h_tsne(
            calib_data, 
            save_path=tsne_file_name
        )
            
    # 4. Inference on Test Set
    print(f"\n=== Starting Inference ===")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    open(args.output_file, "w").close() 

    correct_count = 0
    total_count = 0
    pbar = tqdm(total=len(test_data), desc="Evaluating")
    
    
    # 在 main() 中修改推理循环部分
    for i in range(0, len(test_data), args.eval_batch_size):
        batch_ex = test_data[i : i + args.eval_batch_size]
        
        # --- 修改点：根据需求选择干预模式 ---
        if args.instance_steering:   # 针对单个样例进行干预
            # 模式 A: 每个数据算自己的向量 (传入原始 batch 数据)
            batch_outputs = steerer.generate_with_instance_steering(
                batch_ex, 
                alpha=args.alpha, 
                intervention_mode=args.intervention_mode
            )
        else:
            # 模式 B: 使用之前计算好的全局平均向量 (原始逻辑)
            # baseline 的单个 prompt、reverse、repeat、pad_repeat 都在这里处理
            if args.repeat and not args.pad_repeat:   # 对prompt进行重复计算，作为一个baseline
                # 语义重复 baseline
                batch_prompts = [
                    build_prompts(x, tokenizer, repeat=True, reverse_context=False, pad_repeat=False)
                    for x in batch_ex
                ]
                # print(f"Batch Prompts Example (Repeat):\n{batch_prompts[0]}...")  # 打印一个示例 Prompt 以供调试
                # exit()
            elif args.pad_repeat:
                # 仅用 pad 字符扩长的 baseline（语义不变）
                batch_prompts = [
                    build_prompts(x, tokenizer, repeat=False, reverse_context=args.reverse_context, pad_repeat=True)
                    for x in batch_ex
                ]
            else:     # 其他情况（包括reverse）仍然使用原来的构建方式
                # 普通 / reverse baseline
                batch_prompts = [
                    build_prompts(x, tokenizer, repeat=False, reverse_context=args.reverse_context, pad_repeat=False)
                    for x in batch_ex
                ]
                # print(f"Batch Prompts Example:\n{batch_prompts[0]}...")  # 打印一个示例 Prompt 以供调试
                # exit()
            batch_outputs = steerer.generate_with_steering(
                batch_prompts, 
                alpha=args.alpha, 
                intervention_mode=args.intervention_mode,
                pad_repeat=args.pad_repeat,
                pad_factor=args.pad_factor,
            )
        # --- 剩下的保存逻辑不变 ---
        # batch_prompts = [build_prompts(x, tokenizer, repeat=False, reverse_context=args.reverse_context) for x in batch_ex] # 注意测试时是单次 Prompt!

        with open(args.output_file, "a", encoding="utf-8") as f:
            for j, output_text in enumerate(batch_outputs):
                ex = batch_ex[j]
                ground_truth = ex.get("answer", "").strip()
                is_correct = check_is_correct(output_text, ground_truth)
                
                if is_correct: correct_count += 1
                total_count += 1
                
                f.write(json.dumps({
                    "id": ex.get("id", str(total_count)),
                    "prediction": output_text,
                    "ground_truth": ground_truth,
                    "is_correct": is_correct
                }, ensure_ascii=False) + "\n")
        
        pbar.update(len(batch_ex))
        pbar.set_postfix({"Acc": f"{correct_count/total_count:.2%}"})

    pbar.close()
    print(f"\nDone! Final Accuracy: {correct_count/total_count:.2%}")

if __name__ == "__main__":
    main()