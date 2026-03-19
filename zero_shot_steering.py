import json
import argparse
import os
import re
import torch
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def build_prompts(ex, tokenizer=None, repeat=False, reverse_context=False):
    """
    构建 Prompt。如果 repeat=True，则应用论文中的 Query + Query 策略。
    """
    ctx = ex.get("context", "")
    q = ex.get("question", "")
    opts = _format_options_from_ex(ex)
    
    base_query = f"Context:\n{ctx}\n\nQuestion:\n{q}\n\n{opts}\n\nPlease provide the reasoning and the answer."
    if reverse_context:
        base_query = f"Question:\n{q}\n\n{opts}\n\nContext:\n{ctx}\n\nPlease provide the reasoning and the answer."
        base_query = f"Question:\n{q}\n\nOptions:{opts}\n\nContext:\n{ctx}\n\nPlease provide the reasoning and the answer."
    
    # 核心：复现论文的 Prompt Repetition
    if repeat:
        # 你也可以在这里尝试论文里的变体：base_query + "\n\nLet me repeat that:\n\n" + base_query
        user_content = base_query + "\n\n" + base_query 
    else:
        user_content = base_query
    
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
    def __init__(self, model, tokenizer, layer_idx: int):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = model.device
        self.steering_vector = None # 用于存储计算出的 Δh
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
    def extract_features(self, prompts: List[str], batch_size=4):
        """提取指定层最后一个 Token 的隐状态"""
        all_hiddens = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(self.device)
            
            outputs = self.model(**inputs, output_hidden_states=True)
            target_idx = self.layer_idx + 1 if self.layer_idx >= 0 else self.layer_idx
            hidden = outputs.hidden_states[target_idx] 
            
            last_hidden = hidden[:, -1, :].detach().float()
            all_hiddens.append(last_hidden)
            
        return torch.cat(all_hiddens, dim=0)

    def compute_steering_vector(self, data_samples, batch_size=4):
        """
        Step 1 & 2 & 3: 计算 h_single 和 h_repeat，求差分并平均
        """
        print(f"\n[Steering] Computing difference vector over {len(data_samples)} calibration samples...")
        prompts_single = [build_prompts(x, self.tokenizer, repeat=False) for x in data_samples]
        prompts_repeat = [build_prompts(x, self.tokenizer, repeat=True) for x in data_samples]

        h1 = self.extract_features(prompts_single, batch_size)
        h2 = self.extract_features(prompts_repeat, batch_size)

        # 差分: Δh = h2 - h1
        diffs = h2 - h1 
        # 归因均值: 计算整个校准集的平均方向
        self.steering_vector = diffs.mean(dim=0) 
        
        return self.steering_vector

    def generate_with_steering(self, prompts: List[str], alpha: float = 1.0, intervention_mode: str = "static"):
        """
        Step 4: 将向量注入到残差流进行干预
        """
        if self.steering_vector is None and alpha != 0.0:
            raise ValueError("Steering vector not computed! Run compute_steering_vector first.")
        if alpha !=0.0:   # 对中间向量进行干预
            print(f"\n[Steering] Applying steering with alpha={alpha} in {intervention_mode} mode...")
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(self.device)
            # 移除这里的 .to(self.device)，我们将在 hook 中动态匹配设备
            vec_base = (self.steering_vector * alpha).to(self.model.dtype)

            def adapter_hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                seq_len = h.shape[1]
                
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
                    
                return (h,) + output[1:] if isinstance(output, tuple) else h

            layer_module = self._get_layer_module()
            handle = layer_module.register_forward_hook(adapter_hook)
        else:
            print(f"\n[Steering] Alpha is 0.0, no intervention applied.")
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(self.device)
            handle = None
        
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

    def generate_with_instance_steering(self, prompts: List[str], alpha: float = 1.0, intervention_mode: str = "static"):
        """
        [新增功能] 针对每个 Query 实时计算 Δh 并干预
        """
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
        
        # 计算每一条数据自己的差分向量
        batch_diffs = (h2 - h1) * alpha # [B, D]
        batch_diffs = batch_diffs.to(self.model.dtype)

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
        inputs = self.tokenizer(p_s, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(self.device)
        layer_module = self._get_layer_module()
        handle = layer_module.register_forward_hook(instance_adapter_hook)
        
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
    test_data = load_data_file(args.test_file, max_n=None)
    
    if not args.instance_steering and args.alpha != 0.0:
        if not calib_data:
            print("[Error] Calibration data empty.")
            return
    if not test_data:
        print("[Error] Test Data empty.")
        return

    # 2. Load Model
    print(f"Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    steerer = ActivationSteerer(model, tokenizer, layer_idx=args.layer)

    # 3. Compute Steering Vector (Zero-shot, No Labels Needed)
    # laska修改，某些情况下不需要进入这个函数 1. instance steering 模式下每个样例单独计算向量 2. alpha=0 的情况下不需要计算向量（虽然不计算向量也不会报错，但为了效率我们直接跳过）
    if (not args.instance_steering) and (args.alpha != 0.0):
        steerer.compute_steering_vector(calib_data, batch_size=args.eval_batch_size)

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
            # baseline的单个prompt、reverse以及重复prompt将在这个分支中进行计算
            if args.repeat:   # 对prompt进行重复计算，作为一个baseline
                batch_prompts = [build_prompts(x, tokenizer, repeat=True) for x in batch_ex]
            else:     # 其他情况（包括reverse）仍然使用原来的构建方式
                batch_prompts = [build_prompts(x, tokenizer, repeat=False, reverse_context=args.reverse_context) for x in batch_ex]
            batch_outputs = steerer.generate_with_steering(
                batch_prompts, 
                alpha=args.alpha, 
                intervention_mode=args.intervention_mode
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