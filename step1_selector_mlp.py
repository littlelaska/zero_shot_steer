import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ==========================================
# 新增: 导入 T-SNE 和可视化库
# ==========================================
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

ASSISTANT_PROMPT = "You are a logical task solver. Read the context, question and options carefully. Then, provide a step-by-step reasoning chain to solve the problem. Finally, output the answer."

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--target_train_path", type=str, required=True)
    parser.add_argument("--target_test_path", type=str, required=True)
    parser.add_argument("--k_shot", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--output_json", type=str, default="delta_rep_selected_prompts.json")
    # MLP 训练超参数
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for MLP")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda_sem", type=float, default=10000, help="Weight for Semantic Regularization Loss")
    
    # 新增: T-SNE 可视化输出路径
    parser.add_argument("--tsne_output", type=str, default="./tsne_comparison.png", help="Path to save T-SNE plot")
    parser.add_argument("--baseline_double_query", action="store_true", help="Skip retrieval and duplicate the query in the prompt.")
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

# ==========================================
# 新增: T-SNE 可视化绘图函数
# ==========================================
def plot_tsne_comparison(source_before, source_after, target, save_path):
    print("Running T-SNE for visualization (This might take a moment)...")
    tsne = TSNE(n_components=2, random_state=42)
    
    # 构建训练前(Before)的特征集与标签
    features_before = np.concatenate([source_before, target], axis=0)
    labels_before = np.array([0]*len(source_before) + [1]*len(target))
    reduced_before = tsne.fit_transform(features_before)
    
    # 构建训练后(After)的特征集与标签
    features_after = np.concatenate([source_after, target], axis=0)
    labels_after = np.array([0]*len(source_after) + [1]*len(target))
    reduced_after = tsne.fit_transform(features_after)
    
    # 开始绘图
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制 Before Alignment
    axs[0].scatter(reduced_before[labels_before==0, 0], reduced_before[labels_before==0, 1], label='Source Domain (Original)', alpha=0.6, color='blue')
    axs[0].scatter(reduced_before[labels_before==1, 0], reduced_before[labels_before==1, 1], label='Target Domain', alpha=0.6, color='red')
    axs[0].set_title('Before MLP Alignment')
    axs[0].legend()
    
    # 绘制 After Alignment
    axs[1].scatter(reduced_after[labels_after==0, 0], reduced_after[labels_after==0, 1], label='Source Domain (Aligned)', alpha=0.6, color='blue')
    axs[1].scatter(reduced_after[labels_after==1, 0], reduced_after[labels_after==1, 1], label='Target Domain', alpha=0.6, color='red')
    axs[1].set_title('After MLP Alignment')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"T-SNE visualization saved to {save_path}")

# ==========================================
# 1. 残差网络结构与损失函数
# ==========================================
class ResidualMapper(nn.Module):
    """采用残差结构以保留源域 Demo 原本的基础语义特征"""
    def __init__(self, hidden_dim, mlp_hidden_dim=None):
        super().__init__()
        if mlp_hidden_dim is None:
            mlp_hidden_dim = hidden_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        return x + self.mlp(x)

def rbf_kernel(x, y, gamma=None):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.exp(-gamma * dist)

def compute_mmd_loss(x, y):
    xx = rbf_kernel(x, x)
    yy = rbf_kernel(y, y)
    xy = rbf_kernel(x, y)
    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_semantic_loss(orig_x, mapped_x):
    orig_norm = F.normalize(orig_x, p=2, dim=1)
    mapped_norm = F.normalize(mapped_x, p=2, dim=1)
    
    sim_orig = torch.mm(orig_norm, orig_norm.t())
    sim_mapped = torch.mm(mapped_norm, mapped_norm.t())
    
    return F.mse_loss(sim_mapped, sim_orig)

def main():
    args = parse_args()
    source_pool = json.load(open(args.source_path))
    target_train = json.load(open(args.target_train_path))
    target_test = json.load(open(args.target_test_path))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        
    if args.baseline_double_query:
        print("Running Baseline: Double Query (No Retrieval)...")
        final_output = []
        
        for t_idx, target_item in enumerate(tqdm(target_test, desc="Formatting Double Query Prompts")):
            target_text_part = format_query_only(target_item)
            
            # 将 Query 重复两次，并在末尾加上 Reasoning:
            full_content = f"{target_text_part}\n\n{target_text_part}Reasoning:"
            
            final_prompt = apply_chat_template(tokenizer, full_content)
            
            final_output.append({
                "target_id": t_idx,
                "best_prompt": final_prompt,
                "selected_demo_indices": [], # 没有使用任何 demo
                "ground_truth": str(target_item.get('answer', ''))
            })

        with open(args.output_json, 'w') as f:
            json.dump(final_output, f, indent=4)
        print(f"Baseline selection complete. Saved to {args.output_json}")
        return # 直接退出，不执行后续的模型加载和 MLP 对齐    
    
    model = AutoModel.from_pretrained(args.model_path, output_hidden_states=True, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    device = model.device

    print("Step 1/3: Extracting Base Embeddings...")
    source_queries = [apply_chat_template(tokenizer, format_query_only(s)) for s in source_pool]
    target_train_queries = [apply_chat_template(tokenizer, format_query_only(t)) for t in target_train]
    source_demos = [apply_chat_template(tokenizer, format_full_demo(s)) for s in source_pool]
    
    E_q_s_np = get_embeddings(model, tokenizer, source_queries, args.batch_size)
    E_q_t_np = get_embeddings(model, tokenizer, target_train_queries, args.batch_size)
    E_d_s_np = get_embeddings(model, tokenizer, source_demos, args.batch_size)

    # 转换为 Tensor 以进行梯度训练
    E_q_s = torch.tensor(E_q_s_np, dtype=torch.float32, device=device)
    E_q_t = torch.tensor(E_q_t_np, dtype=torch.float32, device=device)
    E_d_s = torch.tensor(E_d_s_np, dtype=torch.float32, device=device)

    # ==========================================
    # Phase 2: 训练 MLP 实现流形对齐
    # ==========================================
    print("Step 2/3: Training Non-linear Mapping for Alignment...")
    hidden_dim = E_q_s.shape[1]
    mapper = ResidualMapper(hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(mapper.parameters(), lr=args.lr)

    mapper.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        mapped_E_q_s = mapper(E_q_s)
        
        loss_mmd = compute_mmd_loss(mapped_E_q_s, E_q_t)
        loss_sem = compute_semantic_loss(E_q_s, mapped_E_q_s)
        total_loss = loss_mmd + args.lambda_sem * loss_sem
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Total Loss: {total_loss.item():.4f} (MMD: {loss_mmd.item():.4f}, Sem: {loss_sem.item():.9f})")

    # ==========================================
    # 新增: 绘制对齐前后的特征分布对比图
    # ==========================================
    mapper.eval()
    with torch.no_grad():
        # 获取源域 Query 经过对齐后的特征
        V_tilde_q_s = mapper(E_q_s).cpu().numpy()
        # 获取源域 Demo 经过对齐后的特征 (供后续检索使用)
        V_tilde_d_s = mapper(E_d_s).cpu().numpy()
        
    # 调用绘制 T-SNE 图像
    plot_tsne_comparison(source_before=E_q_s_np, source_after=V_tilde_q_s, target=E_q_t_np, save_path=args.tsne_output)

    # 归一化用于后续余弦相似度计算
    V_tilde_d_s_norm = V_tilde_d_s / (np.linalg.norm(V_tilde_d_s, axis=1, keepdims=True) + 1e-9)

    # ==========================================
    # Phase 3: 提取 Target Test 表征并进行跨域检索
    # ==========================================
    print("Step 3/3: Retrieving and Building Prompts for Test Set...")
    target_test_queries = [apply_chat_template(tokenizer, format_query_only(t)) for t in target_test]
    E_q_test_np = get_embeddings(model, tokenizer, target_test_queries, args.batch_size)
    
    final_output = []
    
    for t_idx, target_item in enumerate(tqdm(target_test, desc="Selecting Shots")):
        v_q_test = E_q_test_np[t_idx] 
        v_q_test_norm = v_q_test / (np.linalg.norm(v_q_test) + 1e-9)
        
        # 直接计算映射后 Demo 表征与目标 Query 表征间的打分
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