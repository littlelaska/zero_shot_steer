import json
import numpy as np
import torch
import re
import argparse
import os
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer, util
ASSISTANT_PROMPT = "You are a logical task solver. Read the context, question and options carefully. Then, provide a step-by-step reasoning chain to solve the problem. Finally, output the answer."

# ================= 0. 参数配置 =================

def parse_args():
    parser = argparse.ArgumentParser(description="Full Contextualized Projection Alignment")
    parser.add_argument("--model_name", type=str, required=True, help="e.g., Gemma-2-27b")
    parser.add_argument("--model_path", type=str, required=True, help="Path to local model")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size for vLLM")
    parser.add_argument("--source_path", type=str, required=True, help="Source domain pool (with CoT)")
    parser.add_argument("--target_dev_path", type=str, required=True, help="Target domain dev set")
    parser.add_argument("--target_pseudo_path", type=str, required=True, help="Target domain pseudo-CoT")
    parser.add_argument("--k_shot", type=int, default=3, help="Number of shots to select for final inference")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for embedding calculation")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--source_domain", type=str, default="ProofWriter")
    parser.add_argument("--target_domain", type=str, default="LogicalDeduction")
    return parser.parse_args()

# ================= 1. 文本格式化工具 =================

def load_data(path):
    print(f"Loading data: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_basic_content(item, include_cot=False, include_answer=False, cot_key='reasoning_cot'):
    """基础内容格式化：Context + Question [+ CoT] [+ Answer]"""
    text = ""
    if item.get('context'):
        text += f"Context: {item['context']}\n"
    text += f"Question: {item['question']}\n"
    
    # --- 修改开始：处理 Options 列表格式 ---
    opts = item.get('options', [])
    if opts:
        if isinstance(opts, list):
            # 如果是列表，用空格拼接： "A) ... B) ... C) ..."
            opts_str = ' '.join(opts)
        else:
            # 如果已经是字符串，直接使用
            opts_str = str(opts)
        text += f"Options: {opts_str}\n"
    # --- 修改结束 ---
        
    if include_cot:
        cot_text = item.get(cot_key, '')
        text += f"Reasoning: {cot_text}\n"
    else:
        # Zero-shot 结尾，引导模型开始生成
        text += "Reasoning:" 
        
    if include_answer and include_cot:
        text += f"Answer: {item.get('answer', '')}"
        
    return text

def format_zero_shot_prompt(item, tokenizer):
    """Zero-Shot: [Target Query]"""
    user_content = format_basic_content(item, include_cot=False)
    messages = [{"role": "system", "content": ASSISTANT_PROMPT}, {"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_ideal_prompt(target_item, pseudo_demo_item, tokenizer):
    """
    Ideal State (In-Domain Proxy): [Target Pseudo Demo] + [Target Query]
    注意：pseudo_demo_item 必须不同于 target_item
    """
    # Demo 部分: 使用 pseudo_cot
    demo_text = format_basic_content(pseudo_demo_item, include_cot=True, include_answer=True, cot_key='reasoning_cot')
    # Query 部分
    target_text = format_basic_content(target_item, include_cot=False)
    
    full_content = f"{demo_text}\n\n{target_text}"
    messages = [{"role": "system", "content": ASSISTANT_PROMPT}, {"role": "user", "content": full_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_contextualized_candidate_prompt(target_item, source_item, tokenizer):
    """Candidate State: [Source Demo] + [Target Query]"""
    # Demo 部分: 使用 reasoning_cot
    demo_text = format_basic_content(source_item, include_cot=True, include_answer=True, cot_key='reasoning_cot')
    # Query 部分
    target_text = format_basic_content(target_item, include_cot=False)
    
    full_content = f"{demo_text}\n\n{target_text}"
    messages = [{"role": "system", "content": ASSISTANT_PROMPT}, {"role": "user", "content": full_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_answer(generated_text):
    if not generated_text: return "Unknown"
    boxed_match = re.search(r'\\boxed\{([A-G])\}', generated_text)
    if boxed_match: return boxed_match.group(1).upper()
    
    patterns = [
        r'[Aa]nswer:\s*([A-G])(?!\w)', 
        r'[Aa]nswer\s+is\s+:?\s*([A-G])(?!\w)',
        r'[Aa]nswer:\s*\(([A-G])\)',
        r'[Aa]nswer:\s*([A-G])\)'
    ]
    for pattern in patterns:
        match = re.search(pattern, generated_text)
        if match: return match.group(1).upper()

    bold_match = re.search(r'\*\*([A-G])\*\*', generated_text)
    if bold_match: return bold_match.group(1).upper()

    lines = generated_text.split('\n')
    for line in reversed(lines):
        line_match = re.search(r'^\s*([A-G])\)', line)
        if line_match: return line_match.group(1).upper()
            
    all_matches = re.findall(r'\b([A-G])\b', generated_text)
    if all_matches: return all_matches[-1].upper()

    return "Unknown"

# ================= 2. 向量提取引擎 =================

# def get_embeddings(model, tokenizer, texts, batch_size):
#     vectors = []
#     model.eval()
#     with torch.no_grad():
#         for i in range(0, len(texts), batch_size):
#             batch_texts = texts[i : i + batch_size]
#             inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
#             outputs = model(**inputs)
#             hidden = outputs.hidden_states[-1]
#             mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
#             mean_embeddings = torch.sum(hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
#             vectors.append(mean_embeddings.cpu().numpy())
#     if not vectors: return np.array([])
#     return np.concatenate(vectors, axis=0)

def get_embeddings(model, tokenizer, texts, batch_size):
    vectors = []
    # 确保 tokenizer 设置为右侧 padding (默认通常是右侧，但在生成任务中常设为左侧，这里需注意)
    # 如果你的 tokenizer.padding_side 是 'left'，需要临时改一下或者直接取 [-1]
    # 这里我们假设是标准的 Right Padding (tokenizer(texts) 的默认行为)
    tokenizer.padding_side = "right" 
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # 1. Tokenize
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=8192
            ).to(model.device)
            
            # 2. Forward Pass
            outputs = model(**inputs)
            # 获取最后一层的隐状态: (Batch_Size, Seq_Len, Hidden_Size)
            hidden_states = outputs.hidden_states[-1] 
            
            # 3. 提取 Last Token 索引
            # attention_mask 形如 [1, 1, 1, 0, 0]，sum 后为 3，减 1 得到索引 2
            # 这代表最后一个有效 token 的位置
            last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
            
            # 4. Gather 提取向量
            # 构建 batch 索引: [0, 1, ..., B-1]
            batch_indices = torch.arange(last_token_indices.shape[0], device=model.device)
            
            # 从 (Batch, Seq, Hidden) 中取出 (Batch, Hidden)
            # 相当于: hidden_states[0, len0], hidden_states[1, len1]...
            last_token_embeddings = hidden_states[batch_indices, last_token_indices]
            
            # 转为 numpy 并存入列表
            vectors.append(last_token_embeddings.cpu().float().numpy())
            
    if not vectors: return np.array([])
    return np.concatenate(vectors, axis=0)

# ================= 3. 主流程 =================

# ================= 3. 主流程 (修改版) =================

def main():
    args = parse_args()
    print(f"=== Starting Adaptive Projection Alignment Analysis (Dynamic Shots) ===")
    
    source_pool = load_data(args.source_path)
    target_dev = load_data(args.target_dev_path)
    target_pseudo = load_data(args.target_pseudo_path)
    
    # 这里的 k_shot 代表“最大搜索深度”，即最多尝试到几-shot
    max_k_shot = args.k_shot 
    
    print("\n>>> Initializing HF Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.model_path, output_hidden_states=True, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    
    # --- Step A: 计算 Target 基准向量 (保持不变) ---
    print("\n>>> [Step A] Computing Target Baselines (h_zs & h_in)...")
    zs_prompts = [format_zero_shot_prompt(t, tokenizer) for t in target_dev]
    h_zs = get_embeddings(model, tokenizer, zs_prompts, args.batch_size)
    
    # 构造 Ideal State (Pseudo)
    n_samples = len(target_dev)
    rotated_pseudo = [target_pseudo[i-1] for i in range(n_samples)]
    in_prompts = [format_ideal_prompt(t, p, tokenizer) for t, p in zip(target_dev, rotated_pseudo)]
    h_in = get_embeddings(model, tokenizer, in_prompts, args.batch_size)
    
    # 计算理想方向 Delta_in
    delta_in = h_in - h_zs 
    
    # --- Step B: 动态 Shot 数选择 (核心修改) ---
    print("\n>>> [Step B] Scanning & Selecting Optimal Shot Count...")
    
    # 准备粗排检索器
    retriever = SentenceTransformer('/userhome/huggingface/text2vec-large-chinese', device='cuda')
    source_texts = [s['question'] for s in source_pool]
    source_embs = retriever.encode(source_texts, convert_to_tensor=True, show_progress_bar=False)
    
    final_best_prompts = []     # 存储每个 Query 最终选定的最佳 Prompt 文本
    final_stats = []            # 存储选择的 shot 数量，用于分析
    
    # 逐个 Target 处理
    for t_idx in tqdm(range(len(target_dev)), desc="Optimizing Shots"):
        target_item = target_dev[t_idx]
        curr_h_zs = h_zs[t_idx]      # Shape: (H,)
        curr_delta_in = delta_in[t_idx] # Shape: (H,)
        
        # 1. 粗筛：获取语义最接近的 Top-K 个 Demo 的索引（按相关性排序）
        target_q_emb = retriever.encode(target_item['question'], convert_to_tensor=True)
        hits = util.semantic_search(target_q_emb, source_embs, top_k=max_k_shot)[0]
        # 得到基础队列：[d1, d2, d3, ... dk]
        ordered_indices = [hit['corpus_id'] for hit in hits]
        ordered_demos = [source_pool[idx] for idx in ordered_indices]
        
        # 2. 构建累积 Prompt 列表
        # list: [Prompt(1-shot), Prompt(2-shot), ..., Prompt(k-shot)]
        cumulative_prompts = []
        
        # 预先处理 Target 部分
        target_text_part = format_basic_content(target_item, include_cot=False)
        
        current_demo_context = ""
        for i, demo in enumerate(ordered_demos):
            # 累加 Demo 文本
            demo_text = format_basic_content(demo, include_cot=True, include_answer=True)
            current_demo_context += f"{demo_text}\n\n"
            
            # 拼接完整 Prompt
            full_content = f"{current_demo_context}{target_text_part}"
            messages = [{"role": "system", "content": ASSISTANT_PROMPT}, {"role": "user", "content": full_content}]
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            cumulative_prompts.append(prompt_str)
            
        # 3. 计算所有累积 Prompt 的 Embedding
        # h_cumulative shape: (max_k_shot, Hidden_Size)
        h_cumulative = get_embeddings(model, tokenizer, cumulative_prompts, batch_size=max_k_shot)
        
        if len(h_cumulative) == 0:
            # 异常兜底，默认选 k-shot
            final_best_prompts.append(cumulative_prompts[-1])
            final_stats.append(max_k_shot)
            continue

        # 4. 计算投影并选择
        # Delta_out = h(k-shot) - h(0-shot)
        # 注意广播机制: (K, H) - (H,) -> (K, H)
        delta_out_all = h_cumulative - curr_h_zs
        
        # Projection Score = Delta_out · Delta_in
        # (K, H) dot (H,) -> (K,)
        dot_products = np.dot(delta_out_all, curr_delta_in)
        
        # 2. 计算模长 (Norms)
        # norm_out shape: (K,)
        norm_out = np.linalg.norm(delta_out_all, axis=1) + 1e-9 # 加个 epsilon 防止除零
        
        # norm_in shape: scalar (对于当前 query 是固定的，其实不除也可以，但为了严谨还是除一下)
        norm_in = np.linalg.norm(curr_delta_in) + 1e-9
        
        # 3. 计算余弦相似度 (Cosine Similarity)
        # Cosine = (A . B) / (|A| * |B|)
        scores = dot_products / (norm_out * norm_in)
        
        # 找到分数最大的索引
        best_idx = np.argmax(scores) # 0 代表 1-shot, 1 代表 2-shot...
        optimal_shot_count = best_idx + 1
        
        # 存下那个得分最高的 Prompt
        final_best_prompts.append(cumulative_prompts[best_idx])
        final_stats.append({
            "id": t_idx,
            "optimal_shots": int(optimal_shot_count),
            "max_score": float(scores[best_idx]),
            "candidates_indices": [int(x) for x in ordered_indices[:optimal_shot_count]]
        })

    # --- 释放显存 ---
    del model
    del retriever
    if 'retriever' in locals(): del retriever
    if 'source_embs' in locals(): del source_embs
    
    gc.collect()
    torch.cuda.empty_cache()
        
    import time
    time.sleep(5)
    
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    
    # --- Step C: 推理 (直接使用选好的 Prompt) ---
    print("\n>>> [Step C] Running Inference with vLLM...")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp_size, dtype="float16", gpu_memory_utilization=0.90, enforce_eager=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    
    # 这里的 final_best_prompts 已经是上面选出来的“2-shot”或“3-shot”的完整字符串了
    outputs = llm.generate(final_best_prompts, sampling_params)
    
    correct = 0
    save_data = []
    
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        pred = extract_answer(gen_text)
        gt = str(target_dev[i].get('answer', '')) 
        
        is_correct = (pred.strip().upper() == gt.strip().upper())
        if is_correct: correct += 1
        
        save_data.append({
            "id": i,
            "optimal_shots": final_stats[i]["optimal_shots"], # 记录用了几个shot
            "prompt": final_best_prompts[i],
            "generated_reasoning": gen_text,
            "prediction": pred,
            "ground_truth": gt,
            "is_correct": is_correct,
            "selected_indices": final_stats[i]["candidates_indices"]
        })
        
    # --- 保存结果 ---
    output_dir = f"results/{args.model_name}/adaptive_dynamic_shots"
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{output_dir}/dynamic_shot_selection.json"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
        
    print(f"\nFinal Accuracy: {correct / len(target_dev):.2%}")
    # 打印平均用了多少个shot
    avg_shots = sum([x['optimal_shots'] for x in final_stats]) / len(final_stats)
    print(f"Average Shots Used: {avg_shots:.2f}")
    print(f"Results saved to: {output_filename}")

if __name__ == "__main__":
    main()