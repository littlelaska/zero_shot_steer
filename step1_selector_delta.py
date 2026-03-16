import json
import numpy as np
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

ASSISTANT_PROMPT = "You are a logical task solver. Read the context, question and options carefully. Then, provide a step-by-step reasoning chain to solve the problem. Finally, output the answer."

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True, help="Source domain pool for demonstrations")
    # 拆分目标域数据：Train 用于计算流形平移，Test 用于最终评估
    parser.add_argument("--target_train_path", type=str, required=True, help="Target domain training data for calculating the shift vector")
    parser.add_argument("--target_test_path", type=str, required=True, help="Target domain test data for evaluation")
    parser.add_argument("--k_shot", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--output_json", type=str, default="delta_rep_selected_prompts.json")
    return parser.parse_args()

def format_query_only(item):
    text = ""
    if item.get('context'): text += f"Context: {item['context']}\n"
    text += f"Question: {item['question']}\n"
    opts = item.get('options', [])
    if opts:
        opts_str = ' '.join(opts) if isinstance(opts, list) else str(opts)
        text += f"Options: {opts_str}\n"
    return text

def format_full_demo(item, cot_key='reasoning_cot'):
    text = format_query_only(item)
    text += f"Reasoning: {item.get(cot_key, '')}\n"
    text += f"Answer: {item.get('answer', '')}"
    return text

def apply_chat_template(tokenizer, content):
    messages = [{"role": "system", "content": ASSISTANT_PROMPT}, {"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def get_embeddings(model, tokenizer, texts, batch_size):
    vectors = []
    tokenizer.padding_side = "right" 
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Embeddings", leave=False):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[-1] 
            last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
            batch_indices = torch.arange(last_token_indices.shape[0], device=model.device)
            last_token_embeddings = hidden_states[batch_indices, last_token_indices]
            vectors.append(last_token_embeddings.cpu().float().numpy())
    return np.concatenate(vectors, axis=0) if vectors else np.array([])

def main():
    args = parse_args()
    source_pool = json.load(open(args.source_path))
    target_train = json.load(open(args.target_train_path))
    target_test = json.load(open(args.target_test_path))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.model_path, output_hidden_states=True, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

    # ==========================================
    # Phase 1: 提取源域-目标域差分向量 (严格使用 Target Train)
    # ==========================================
    print("Step 1/3: Calculating Source-Target Shift Vector (using Training distributions)...")
    source_queries = [apply_chat_template(tokenizer, format_query_only(s)) for s in source_pool]
    target_train_queries = [apply_chat_template(tokenizer, format_query_only(t)) for t in target_train]
    
    E_q_s = get_embeddings(model, tokenizer, source_queries, args.batch_size)
    E_q_t_train = get_embeddings(model, tokenizer, target_train_queries, args.batch_size)
    
    # 计算均值期望 [cite: 14]
    mean_E_q_s = np.mean(E_q_s, axis=0)
    mean_E_q_t_train = np.mean(E_q_t_train, axis=0)
    
    # 提取无污染的平移向量 
    delta_S_T = mean_E_q_t_train - mean_E_q_s  

    # ==========================================
    # Phase 2: 源域 Demo 的表征对齐 
    # ==========================================
    print("Step 2/3: Aligning Source Demonstrations...")
    source_demos = [apply_chat_template(tokenizer, format_full_demo(s)) for s in source_pool]
    E_d_s = get_embeddings(model, tokenizer, source_demos, args.batch_size)
    
    # 将源域 Demo 平移到目标域流形 [cite: 16, 18]
    V_tilde_d_s = E_d_s + delta_S_T  
    V_tilde_d_s_norm = V_tilde_d_s / (np.linalg.norm(V_tilde_d_s, axis=1, keepdims=True) + 1e-9)

    # ==========================================
    # Phase 3: 提取 Target Test 表征并进行跨域检索
    # ==========================================
    print("Step 3/3: Retrieving and Building Prompts for Test Set...")
    # 提前计算目标域测试集的表征，提升检索阶段速度
    target_test_queries = [apply_chat_template(tokenizer, format_query_only(t)) for t in target_test]
    E_q_test = get_embeddings(model, tokenizer, target_test_queries, args.batch_size)
    
    final_output = []
    
    for t_idx, target_item in enumerate(tqdm(target_test, desc="Selecting Shots")):
        # 使用当前 Test Query 的表征进行匹配 [cite: 20]
        v_q_test = E_q_test[t_idx] 
        v_q_test_norm = v_q_test / (np.linalg.norm(v_q_test) + 1e-9)
        
        # 计算对齐后的 Demo 表征与当前测试 Query 之间的相似度 [cite: 20, 21]
        scores = np.dot(V_tilde_d_s_norm, v_q_test_norm)
        
        top_k_indices = np.argsort(scores)[::-1][:args.k_shot]
        ordered_demos = [source_pool[idx] for idx in top_k_indices]
        
        current_demo_context = ""
        for demo in ordered_demos:
            current_demo_context += f"{format_full_demo(demo)}\n\n"
            
        target_text_part = format_query_only(target_item)
        target_text_part += "Reasoning:" 
        full_content = f"{current_demo_context}{target_text_part}"
        
        final_prompt = apply_chat_template(tokenizer, full_content)
        
        final_output.append({
            "target_id": t_idx,
            "best_prompt": final_prompt,
            "selected_demo_indices": top_k_indices.tolist(),
            "ground_truth": str(target_item.get('answer', ''))
        })

    with open(args.output_json, 'w') as f:
        json.dump(final_output, f, indent=4)
    print(f"Selection complete. Saved to {args.output_json}")

if __name__ == "__main__":
    main()