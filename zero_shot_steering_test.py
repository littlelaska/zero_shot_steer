# -*- coding: utf-8 -*-
import json
import argparse
import os
import re
import torch
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
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

def build_prompts(ex, tokenizer=None, repeat=False, reverse_context=False, pad_repeat=False, repeat_times=1):
    """
    构建 Prompt。如果 repeat=True，则应用论文中的 Query + Query 策略。
    - repeat=True: 语义重复 Query + Query（论文里的 Prompt Repetition）。
    - pad_repeat=True: 用 pad 字符把 token 长度扩到约 2 倍（语义不变），用于和 repeat 对比。
    """
    ctx = ex.get("context", "")
    q = ex.get("question", "").lstrip("Question: ")     # 针对commonsense数据集需要去掉前面的question
    opts = _format_options_from_ex(ex)
    
    tail_prompt = "Please provide the reasoning and the answer."
    base_query = f"Context:\n{ctx}\n\nQuestion:\n{q}\n\n{opts}\n\n"
    if reverse_context:
        base_query = f"Question:\n{q}\n\n{opts}\n\nContext:\n{ctx}\n\n"
    
    # 情况 1：重复语义的 Query + Query
    # 核心：复现论文的 Prompt Repetition
    if repeat and not pad_repeat:
        if repeat_times > 1:
            user_content = (base_query * repeat_times) + tail_prompt
        # 你也可以在这里尝试论文里的变体：base_query + "\n\nLet me repeat that:\n\n" + base_query
        else:
            user_content = base_query + base_query + tail_prompt
    else:
        user_content = base_query + tail_prompt
    
    # 情况2，pad_repeat：用 pad token 把输入扩展到约 2 倍长度，保持语义不变
    # pad_repeat 需要考虑template的长度，
    if pad_repeat and tokenizer:
        token_count = len(tokenizer(user_content, add_special_tokens=False)["input_ids"])
        pad_str = (tokenizer.pad_token or tokenizer.eos_token+" ") * token_count
        user_content = pad_str + user_content

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
        
        for p in formatted:
            ids = self.tokenizer(p, add_special_tokens=False, padding=False, truncation=True, max_length=self.max_length)["input_ids"]
            real_len = len(ids)
            # 计算需要补充的 pad 数量
            max_len = min(real_len * pad_factor, truncation_max_length) # 确保至少是原长度
            pad_len = max(0, max_len - real_len)
            
            # 直接通过列表拼接避免循环调用 tokenizer
            new_ids = [pad_id] * pad_len + ids # 左填充
            new_attn = [1] * len(new_ids) # 强制设置为 1，模拟计算开销
            
            input_ids_list.append(torch.tensor(new_ids))
            attn_list.append(torch.tensor(new_attn))
            
        # Batch 对齐 
        batch_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        batch_attn = pad_sequence(attn_list, batch_first=True, padding_value=0)
        
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
                    # 1.确保设备匹配
                    vec_inject = vec_base.to(h.device)
                    
                    # 2. 提取原始激活值及其L2范数，[Batch, Dim] -> [Batch, 1]
                    # 使用 keepdim= True 保持维度以便后续广播
                    orig_last_token = h[:, -1, :]
                    orig_norm = torch.norm(orig_last_token, p=2, dim=-1, keepdim=True)  # [B, 1]

                    # 3. 执行干预加法
                    # 直接加上我们计算好的差分向量
                    steered_last_token = orig_last_token + vec_inject
                    
                    # 4. 计算干预之后的范数
                    steered_norm = torch.norm(steered_last_token, p=2, dim=-1, keepdim=True)  # [B, 1]

                    # 5. 执行模长恢复：Scale = Original_Norm / Steered_Norm
                    # 加入 1e-8 防止除以 0
                    h[:, -1, :] = steered_last_token * (orig_norm / (steered_norm + 1e-8))
                    # h[:, -1, :] = h[:, -1, :] + vec_inject
                    # h[:, -1, :] = vec_inject
                    # (可选) 调试打印：检查比例是否恒定为 100%
                    # print(f"Restored Norm Ratio: {torch.norm(h[:, -1, :], p=2, dim=-1).mean() / orig_norm.mean():.4f}")
                    # exit()
                    
                return (h,) + output[1:] if isinstance(output, tuple) else h

            layer_module = self._get_layer_module()
            handle = layer_module.register_forward_hook(adapter_hook)
        else:   # alpha=0时，主要是跑baseline
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
    
    def generate_with_gte_steering(self, prompts: List[str], alpha: float = 1.0, gte_model=None, gte_tokenizer=None, max_length: int = None):
        if max_length is None:
            padding = True
            max_length = self.max_length
        else:
            padding = "max_length"
        # 1. 构建prompt
        single_prompts = [build_prompts(x, self.tokenizer, repeat=False) for x in prompts]

        gte_instruction = "Identify and represent the core logical premises, relational constraints, and deductive dependencies within the text to support step-by-step reasoning."
        
        # 2. 按照gte模板构建数据，并抽取干预向量
        # 使用gte模型抽取干预向量
        input_template = "Instruct: {gte_instruction}\nQuery: {query}"
        input_prompt = [input_template.format(gte_instruction=gte_instruction, query=p) for p in single_prompts]
        gte_inputs = gte_tokenizer(input_prompt, return_tensors="pt", padding=padding, truncation=True, max_length=max_length).to(gte_model.device)
        with torch.no_grad():
            gte_outputs = gte_model(**gte_inputs, output_hidden_states=True)
        # gte_hidden = gte_outputs.hidden_states[-1][:, -1, :]
        # print("gte_hidden dims are:", gte_hidden.shape)
        
        # 修改为按照抽取同层的隐向量
        # 使用 self.layer_idx 获取 GTE 模型中对应层的 hidden_states
        # 注意：hidden_states[0] 是 embedding 层，所以 layer_idx + 1 对应第 layer_idx 层 transformer 的输出
        try:
            # 获取与 LLM 干预层索引一致的 GTE 隐藏层状态
            # 如果 GTE 层数少于 LLM，这里需要做越界检查
            target_layer_idx = self.layer_idx + 1 
            gte_hidden = gte_outputs.hidden_states[target_layer_idx][:, -1, :]
            print(f"Extracted GTE features from layer {self.layer_idx} (index {target_layer_idx})")
        except IndexError:
            # 兜底方案：如果 GTE 模型层数不够，则取其最后一层
            gte_hidden = gte_outputs.hidden_states[-1][:, -1, :]
            print(f"Warning: Layer {self.layer_idx} out of range for GTE model. Using last layer instead.")
        # ------------------
        # 维度对齐检查（针对可能存在的 Hidden Size 不一致）
        print("the hidden states are extract from layer {}, gte_hidden dims are {}".format(target_layer_idx, gte_hidden.shape))

        main_model_dim = next(self.model.parameters()).shape[-1]
        if gte_hidden.shape[-1] != main_model_dim:
            # 如果维度不一致，必须进行对齐，这里建议抛出错误或添加一个投影矩阵
            print(f"Dimension Mismatch: GTE({gte_hidden.shape[-1]}) vs LLM({main_model_dim})")
            # 简单补齐示例（仅用于跑通代码，实际建议用线性投影）
            if gte_hidden.shape[-1] < main_model_dim:
                padding_vec = torch.zeros(gte_hidden.shape[0], main_model_dim - gte_hidden.shape[-1]).to(gte_hidden)
                gte_hidden = torch.cat([gte_hidden, padding_vec], dim=-1)
        
        normalized_gte = F.normalize(gte_hidden,p=2, dim=-1)
        # print("original norms: ", normalized_gte.squeeze().tolist())

        vec_base = normalized_gte.to(device=self.device, dtype=self. model.dtype)

        def gte_hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            # 只在 Prefill 阶段干预最后一个 token
            
            if h.shape[1] > 1:
                # --- 核心：模长对齐策略 ---
                # 提取当前层原始激活值的平均模长
                orig_token = h[:, -1, :]
                orig_norm = torch.norm(orig_token, p=2, dim=-1, keepdim=True)
                # 计算原始norm。用于后续计算干预比例
                orig_norm_value = orig_norm.mean().item()
                orig_std = orig_norm.std().item()
                
                # 缩放 GTE 向量：使干预信号的强度与原始激活值匹配
                # 注入值 = 方向(vec_base) * 强度(alpha) * 基础能量(orig_norm)
                scaled_vec = vec_base * alpha * orig_norm
                
                # 注入并保持总模长不变（防止数值崩溃）
                steered_token = orig_token + scaled_vec
                steered_norm = torch.norm(steered_token, p=2, dim=-1, keepdim=True)
                steered_norm_value = steered_norm.mean().item()
                steered_std = steered_norm.std().item()
                # 计算干预强度
                ratio = steered_norm_value / orig_norm_value if orig_norm_value != 0 else 0
                # 仅在 Prefill 阶段或第一个 Token 时打印，避免日志刷屏
            
                print(f" > [Layer {self.layer_idx}] Norm Ratio: {ratio:.2%} (Orig: {orig_norm_value:.2f}, Steer: {steered_norm_value:.2f}), Orig Std: {orig_std:.2f}")

                
                h[:, -1, :] = steered_token * (orig_norm / (steered_norm + 1e-8))
                
            return (h,) + output[1:] if isinstance(output, tuple) else h

        # 3. 注册并执行
        layer_module = self._get_layer_module()
        handle = layer_module.register_forward_hook(gte_hook)
        # 构建模型输入
        inputs = self.tokenizer(single_prompts, return_tensors="pt", padding=padding, truncation=True, max_length=max_length).to(self.device) 
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        finally:
            handle.remove()
            
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)


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

    @torch.no_grad()
    def analyze_all_states_tsne(self, data_samples: List[dict], save_path: str = "tsne_trajectory.png", label_key: str = None):
        """
        绘制 h_s, h_r 和 Delta_h 的 t-SNE 分布，并用箭头连接每个样本的演进轨迹。
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from sklearn.manifold import TSNE
        import torch.nn.functional as F

        N = len(data_samples)
        if N < 5:
            print("[Error] 样本数量太少，无法进行 t-SNE 分析。")
            return

        # 1. 提取表征 [cite: 279-301]
        prompts_s = [build_prompts(x, repeat=False) for x in data_samples] # [cite: 85-114]
        prompts_r = [build_prompts(x, repeat=True) for x in data_samples]  # [cite: 85-114]
        #  20260422 新增三次prompt，画图看是否会让模型趋势更强
        prompts_r3 = [build_prompts(x, self.tokenizer, repeat=True, repeat_times=3) for x in data_samples]

        h_s = self.extract_features(prompts_s, batch_size=self.batch_size).cpu().float()
        h_r = self.extract_features(prompts_r, batch_size=self.batch_size).cpu().float()
        h_r3 = self.extract_features(prompts_r3, batch_size=self.batch_size).cpu().float()

        delta = h_r - h_s
        
        # 2. 拼接数据进行全局降维
        # 顺序: [0:N] 是 h_s, [N:2N] 是 h_r, [2N:3N] 是 delta
        # combined_vectors = torch.cat([h_s, h_r, delta], dim=0)
        # 顺序: [0:N] h_s, [N:2N] h_r2, [2N:3N] h_r3, [3N:4N] delta
        combined_vectors = torch.cat([h_s, h_r, h_r3, delta], dim=0)
        combined_norm = F.normalize(combined_vectors, p=2, dim=-1).numpy()
        
        print(f"正在执行全局 t-SNE 降维 (总点数: {3*N})...")
        tsne_model = TSNE(
            n_components=2, 
            # perplexity=min(30, (3*N) - 1), 
            perplexity=min(30, (4*N) - 1), 
            random_state=42, 
            init='pca', 
            n_jobs=-1
        )
        tsne_results = tsne_model.fit_transform(combined_norm)
        
        # 3. 准备绘图数据
        # vector_types = ["Single_h_s"] * N + ["Repeat_h_r"] * N + ["Delta_h"] * N
        vector_types = (["Single_h_s"] * N + 
                        ["Repeat2_h_r2"] * N + 
                        ["Repeat3_h_r3"] * N + 
                        ["Delta_h"] * N)
        task_labels = []
        if label_key and label_key in data_samples[0]:
            # task_labels = [str(x[label_key]) for x in data_samples] * 3
            task_labels = [str(x[label_key]) for x in data_samples] * 4
        else:
            # task_labels = ["Default"] * (3 * N)
            task_labels = ["Default"] * (4 * N)

        df = pd.DataFrame({
            'x': tsne_results[:, 0],
            'y': tsne_results[:, 1],
            'State': vector_types,
            'Task': task_labels
        })
        
        # 4. 绘图
        # plt.figure(figsize=(14, 11))
        plt.figure(figsize=(15, 12))
        
        # 定义颜色调色板
        palette = {
            'Single_h_s': '#3498db',    # 蓝色
            'Repeat2_h_r2': '#e67e22',  # 橙色
            'Repeat3_h_r3': '#e74c3c',  # 红色
            'Delta_h': '#2ecc71'        # 绿色
        }
        # 绘制背景点
        scatter = sns.scatterplot(
            data=df, x='x', y='y', hue='State', 
            style='Task' if len(set(task_labels)) > 1 else None,
            # palette={'Single_h_s': '#3498db', 'Repeat_h_r': '#e74c3c', 'Delta_h': '#2ecc71'},
            palette=palette,
            # s=120, alpha=0.8, edgecolor='w', zorder=3
            s=130, alpha=0.8, edgecolor='w', zorder=3
        )

        # --- 核心改进：绘制演进连线 ---
        print("正在绘制样本轨迹连线...")
        for i in range(N):
            # 从 h_s (索引 i) 指向 h_r (索引 N+i)
            plt.annotate(
                '', 
                xy=(tsne_results[N + i, 0], tsne_results[N + i, 1]), 
                xytext=(tsne_results[i, 0], tsne_results[i, 1]),
                arrowprops=dict(
                    arrowstyle='->', 
                    color='gray', 
                    lw=0.8, 
                    alpha=0.3,
                    shrinkA=5, shrinkB=5 # 稍微缩进，避免盖住点
                ),
                zorder=1 # 放在点下面
            )
            # 第二段：h_r -> h_r3
            plt.annotate('', 
                xy=(tsne_results[2*N + i, 0], tsne_results[2*N + i, 1]), 
                xytext=(tsne_results[N + i, 0], tsne_results[N + i, 1]),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8, alpha=0.3, shrinkA=4, shrinkB=4),
                zorder=2
            )
        
        # 5. 美化图表
        plt.title(f"Latent Space Trajectory: $h_s \\rightarrow h_r$ (Layer: {self.layer_idx})\nArrows indicate the logical reasoning shift per sample", 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f" [Success] 轨迹图已保存: {save_path}")
        
        plt.close()
        return tsne_results
# ==========================================
# 3. 主流程
# ==========================================

def vllm_generate_batch(llm, prompts: List[str], max_new_tokens: int = 4096) -> List[str]:
    """Greedy 解码，与 HF `do_sample=False` 对齐；仅返回新生成片段（与 HF 全序列 decode 在内容上通常等价于答案部分）。"""
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text for out in outputs]

# 测试用embedding模型
class GTEInjectedSteerer:
    def __init__(self, reasoning_model, gte_model, tokenizer, layer_idx):
        self.model = reasoning_model
        self.gte_model = gte_model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = reasoning_model.device
    
    # 可以更换不同的instruction
    def format_gte_query(self, instruction, text_context):
        # GTE 官方要求：Instruct:{任务}\nQuery:{内容}
        return f"Instruct: {instruction}\nQuery: {text_context}"

    @torch.no_grad()
    def extract_gte_guidance_vector(self, instruction_text):
        """
        从 GTE 模型中提取指令引导向量
        """
        # GTE 官方推荐的 Prompt 格式
        input_text = f"Instruct: {instruction_text}\nQuery: "
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.gte_model.device)
        
        # 提取 GTE 最后一层的隐藏状态
        outputs = self.gte_model(**inputs, output_hidden_states=True)
        # 获取最后一个 token 的表征 [1, Dim]
        # 注意：GTE 默认会进行 L2 归一化，我们先拿到原始方向
        gte_hidden = outputs.hidden_states[-1][:, -1, :]
        return F.normalize(gte_hidden, p=2, dim=-1)

    def inject_and_generate(self, prompts, guidance_vector, alpha=1.0):
        """
        将 GTE 向量缩放并注入推理模型
        """
        # 1. 编码推理任务的输入
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        
        # 2. 定义带模长恢复的 Hook [cite: 310-353]
        # 将 GTE 向量转移到推理模型精度和设备
        vec_base = guidance_vector.to(device=self.device, dtype=self.model.dtype)

        def gte_hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            # 只在 Prefill 阶段干预最后一个 token
            if h.shape[1] > 1:
                # --- 核心：模长对齐策略 ---
                # 提取当前层原始激活值的平均模长
                orig_token = h[:, -1, :]
                orig_norm = torch.norm(orig_token, p=2, dim=-1, keepdim=True)
                
                # 缩放 GTE 向量：使干预信号的强度与原始激活值匹配
                # 注入值 = 方向(vec_base) * 强度(alpha) * 基础能量(orig_norm)
                scaled_vec = vec_base * alpha * orig_norm
                
                # 注入并保持总模长不变（防止数值崩溃）
                steered_token = orig_token + scaled_vec
                steered_norm = torch.norm(steered_token, p=2, dim=-1, keepdim=True)
                h[:, -1, :] = steered_token * (orig_norm / (steered_norm + 1e-8))
                
            return (h,) + output[1:] if isinstance(output, tuple) else h

        # 3. 注册并执行
        layer_module = self.model.model.layers[self.layer_idx] # Qwen 架构适配
        handle = layer_module.register_forward_hook(gte_hook)
        
        try:
            out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        finally:
            handle.remove()
            
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)


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
    # vLLM：仅用于无 steer 的 baseline（alpha=0），与 HF 路径共用同一套 build_prompts
    parser.add_argument("--use_vllm", action="store_true", help="使用 vLLM 推理（仅支持 alpha=0、非 instance_steering；不支持 pad_repeat）")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9, help="vLLM gpu_memory_utilization")
    parser.add_argument("--vllm_max_model_len", type=int, default=None, help="可选，传给 vLLM 的 max_model_len")
    
    # 20260427 新增gte模型
    parser.add_argument("--gte_model_path",type=str, default="/data_a100/models/gte-Qwen2-7B-instruct", help="gte model load path")
    parser.add_argument("--steering_mode", type=str, default="llm_steer",help="可选值llm_steer/gte_steer，分别代表用原始llm和gte模型抽取干预向量")

    args = parser.parse_args()

    if args.use_vllm:
        if args.alpha != 0.0:
            raise SystemExit("--use_vllm 仅支持无干预 baseline，请设 --alpha 0.0")
        if args.instance_steering:
            raise SystemExit("--use_vllm 与 --instance_steering 不兼容（实例级 steer 需 HF forward hook）")

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
    llm = None
    steerer = None
    if args.use_vllm:
        from vllm import LLM

        llm_kw = dict(
            model=args.model,
            trust_remote_code=True,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        if args.vllm_max_model_len is not None:
            llm_kw["max_model_len"] = args.vllm_max_model_len
        llm = LLM(**llm_kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        # model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="auto")
        steerer = ActivationSteerer(model, tokenizer, layer_idx=args.layer, batch_size=args.eval_batch_size, max_length=args.max_length)
    
    # 如果是gte干预，则需要加载gte模型
    if args.steering_mode == "gte_steer":
        gte_tokenizer = AutoTokenizer.from_pretrained(args.gte_model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.gte_model_path, trust_remote_code=True)
        # 2. 手动补上缺失的属性 (Qwen2 默认通常是 1000000.0)
        if not hasattr(config, 'rope_theta'):
            config.rope_theta = 1000000.0
        gte_model = AutoModel.from_pretrained(
            args.gte_model_path,
            config=config, 
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
            )
        
        gte_model.config.use_cache = False
        
    # 3. Compute Steering Vector (Zero-shot, No Labels Needed)
    # laska修改，某些情况下不需要进入这个函数 1. instance steering 模式下每个样例单独计算向量 2. alpha=0 的情况下不需要计算向量（虽然不计算向量也不会报错，但为了效率我们直接跳过）
    if (not args.use_vllm) and (not args.instance_steering) and (args.alpha != 0.0) and (args.steering_mode != "gte_steer"):
        steerer.compute_steering_vector(calib_data)
       
        # 新增一个对干预向量进行分析的步骤
        steerer.analyze_steering_vector(top_k=20)
        # 动态生成文件名
        model_name = args.model.split("/")[-1]  # 取模型名称的最后一部分
        tsne_file_name = f"analysis/{args.dataset}_tsne_layer{args.layer}_{model_name}.png"
        
        # 调用分析，指定数据中用于着色的 key 为 'task_type'
        # steerer.analyze_delta_h_tsne(
        #     calib_data, 
        #     save_path=tsne_file_name
        # )
        steerer.analyze_all_states_tsne(
            calib_data, 
            save_path=tsne_file_name
        )
            
    # 4. Inference on Test Set
    # exit()
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
        # laska 20260427 新增使用gte进行steer
        elif args.steering_mode == "gte_steer" and args.alpha != 0.0:
            batch_outputs = steerer.generate_with_gte_steering(
                batch_ex,
                alpha=args.alpha,
                gte_model=gte_model, 
                gte_tokenizer=gte_tokenizer
                )
        else:
            # 模式 B: 使用之前计算好的全局平均向量 (原始逻辑)
            # baseline 的单个 prompt、reverse、repeat、pad_repeat 都在这里处理
            batch_prompts = [build_prompts(x,tokenizer,repeat=args.repeat,reverse_context=args.reverse_context,pad_repeat=args.pad_repeat) for x in batch_ex]
            # print(f"Batch Prompts Example:\n{batch_prompts[0]}...")  # 打印一个示例 Prompt 以供调试
            # exit()
            if args.use_vllm:
                batch_outputs = vllm_generate_batch(llm, batch_prompts)
            else:
                batch_outputs = steerer.generate_with_steering(
                    batch_prompts, 
                    alpha=args.alpha, 
                    intervention_mode=args.intervention_mode,
                    pad_repeat=args.pad_repeat,
                    pad_factor=args.pad_factor,
                )

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