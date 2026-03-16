import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区域 =================
# 请修改为你的实际模型路径
MODEL_PATH = "/userhome/huggingface/Qwen2.5-32B-Instruct/" 
BATCH_SIZE = 1
MAX_LENGTH = 2048
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 数据加载 =================

def load_paired_data(cot_file, traj_file, limit=200):
    """
    加载并配对数据。
    """
    data_pairs = []
    
    if not os.path.exists(cot_file) or not os.path.exists(traj_file):
        print(f"Error: Files not found. \n{cot_file}\n{traj_file}")
        return []

    with open(cot_file, 'r', encoding='utf-8') as f_cot, \
         open(traj_file, 'r', encoding='utf-8') as f_traj:
        
        cot_lines = f_cot.readlines()
        traj_lines = f_traj.readlines()
        
        num_samples = min(len(cot_lines), len(traj_lines), limit)
        
        for i in range(num_samples):
            try:
                cot_item = json.loads(cot_lines[i])
                traj_item = json.loads(traj_lines[i])
                
                # 1. 提取 Query (以 Trajectory 文件为准)
                messages = traj_item.get("messages", [])
                query = next((m["content"] for m in messages if m["role"] == "user"), "")
                
                if not query:
                    continue

                # 2. 构造 Raw CoT 输入
                # 结构: Query + <think> content
                raw_cot_content = cot_item.get("structured_cot") or cot_item.get("cot") or ""
                if not raw_cot_content:
                    raw_cot_content = cot_item.get("output", "")
                
                raw_cot_text = f"User: {query}\n\nAssistant: <think>\n{raw_cot_content}\n</think>"

                # 3. 构造 Agent CoT 输入
                # 结构: 扁平化整个交互历史 (Thought -> Code -> Observation)
                agent_cot_text = ""
                for msg in messages:
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    agent_cot_text += f"{role}: {content}\n"
                
                data_pairs.append({
                    "raw_cot": raw_cot_text,
                    "agent_cot": agent_cot_text
                })
                
            except json.JSONDecodeError:
                continue
                
    print(f"Loaded {len(data_pairs)} valid sample pairs.")
    return data_pairs

# ================= 核心模型函数 (已修改) =================

def get_hidden_state_by_layer(model, tokenizer, texts, layer_idx=-1):
    """
    提取指定层 (layer_idx) 最后一个 token 的隐状态。
    Args:
        layer_idx (int): 层索引。0=Embedding, 1=Layer1, -1=Last Layer。
    """
    model.eval()
    embeddings = []
    
    desc = f"Extracting Layer {layer_idx}" if layer_idx != -1 else "Extracting Last Layer"
    
    # 分批处理
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch_texts = texts[i : i + BATCH_SIZE]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_LENGTH
        ).to(device)
        
        with torch.no_grad():
            # [重要] output_hidden_states=True 必须开启
            outputs = model(**inputs, output_hidden_states=True)
            
            # outputs.hidden_states 是一个 tuple，包含 (embeddings, layer_1, ..., layer_N)
            # 长度通常是 num_layers + 1
            try:
                target_layer_state = outputs.hidden_states[layer_idx]
            except IndexError:
                print(f"Error: Layer index {layer_idx} out of bounds.")
                return None

            # 提取最后一个实义 Token (非 padding)
            last_token_indices = inputs.attention_mask.sum(dim=1) - 1
            
            # 维度: (batch_size, hidden_dim)
            batch_embeddings = target_layer_state[torch.arange(target_layer_state.shape[0]), last_token_indices]
            
            embeddings.append(batch_embeddings.cpu().numpy())
            
    return np.vstack(embeddings)

# ================= 可视化函数 (已修改) =================

def plot_tsne(raw_embeddings, agent_embeddings, title="t-SNE Visualization", filename="tsne.png"):
    """
    执行 t-SNE 并保存图片。
    """
    # 合并数据
    combined_data = np.vstack([raw_embeddings, agent_embeddings])
    
    # t-SNE 降维
    # perplexity 设为 min(30, N/4) 以适应小样本
    perp = min(30, len(combined_data) // 4)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(combined_data)
    
    # 拆分回两组
    num_raw = len(raw_embeddings)
    raw_tsne = tsne_results[:num_raw]
    agent_tsne = tsne_results[num_raw:]
    
    # 绘图
    plt.figure(figsize=(12, 8))
    plt.scatter(raw_tsne[:, 0], raw_tsne[:, 1], c='blue', alpha=0.6, label='Raw CoT (Internal Thought)', s=40)
    plt.scatter(agent_tsne[:, 0], agent_tsne[:, 1], c='red', alpha=0.6, label='Agent CoT (Tool Trajectory)', s=40)
    
    # 连线 (仅针对前 50 个样本，避免太乱)
    for i in range(min(50, num_raw)):
        plt.plot([raw_tsne[i, 0], agent_tsne[i, 0]], 
                 [raw_tsne[i, 1], agent_tsne[i, 1]], 
                 color='gray', alpha=0.15, linewidth=0.8)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=300)
    plt.close() # 关闭图表释放内存
    print(f"Saved plot to {filename}")

# ================= 主流程 =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot_file", type=str, default="mathoai_cot.jsonl")
    parser.add_argument("--traj_file", type=str, default="mathoai_trajectories.jsonl")
    parser.add_argument("--limit", type=int, default=200, help="Number of samples to visualize")
    args = parser.parse_args()

    # 1. 加载数据
    data_pairs = load_paired_data(args.cot_file, args.traj_file, args.limit)
    if not data_pairs:
        return

    raw_texts = [d['raw_cot'] for d in data_pairs]
    agent_texts = [d['agent_cot'] for d in data_pairs]

    # 2. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. 定义要探测的层
    # 建议根据你的模型层数调整。Qwen2.5-32B 约有 64 层。
    # 这里选取：底层、浅中层、中层、中高层、高层、输出层
    target_layers = [0, 8, 16, 24, 32, 48, -1] 
    
    print(f"Starting Layer-wise Analysis on layers: {target_layers}")

    # 4. 逐层提取并绘图
    for layer in target_layers:
        print(f"\nProcessing Layer {layer}...")
        
        raw_emb = get_hidden_state_by_layer(model, tokenizer, raw_texts, layer_idx=layer)
        agent_emb = get_hidden_state_by_layer(model, tokenizer, agent_texts, layer_idx=layer)
        
        if raw_emb is not None and agent_emb is not None:
            layer_name = f"Layer_{layer}" if layer != -1 else "Last_Layer"
            plot_tsne(
                raw_emb, 
                agent_emb, 
                title=f"Visualization of Hidden States - {layer_name}", 
                filename=f"tsne_{layer_name}.png"
            )

    print("\nAll visualizations complete.")

if __name__ == "__main__":
    main()