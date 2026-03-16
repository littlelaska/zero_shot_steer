import argparse
import json
import io
import contextlib
import sys
import concurrent.futures
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from vllm import LLM, SamplingParams
from func_timeout import func_timeout, FunctionTimedOut

# 尝试导入 smolagents
try:
    from smolagents import LocalPythonExecutor
except ImportError:
    print("Error: smolagents not found. Please run `pip install smolagents`")
    sys.exit(1)

# ================= 配置区域 =================
# 请根据实际路径修改
DEFAULT_MODEL = "/userhome/huggingface/Qwen2.5-32B-Instruct/"

# System Prompt: 强制引导 "Thought -> Code" 模式
SYSTEM_PROMPT = """You are an expert AI assistant capable of solving complex problems using Python code.
You must follow a strict "Thought -> Code -> Observation" cycle.

**Guidelines:**
1. **Plan First**: Before writing any code, specifically describe your plan in a 'Thought' block.
2. **Code**: Write executable Python code to solve the current step. Enclose code in markdown code blocks, e.g., ```python ... ```.
3. **Tools**: You have access to a python interpreter with libraries: math, sympy, numpy, itertools, etc.
4. **Conclusion**: Once you have the result, output the final answer in the format: FINAL ANSWER: \\boxed{your_answer}.

Start by analyzing the request."""

# ================= 工具函数 =================

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    [Critical Fix] 使用堆栈逻辑提取 \\boxed{} 中的内容，支持嵌套括号 (如 \\frac{})
    解决了正则无法匹配嵌套括号的问题。
    """
    # 从后往前找最后一个 boxed，通常最后的才是最终结论
    start_marker = "\\boxed{"
    start_idx = text.rfind(start_marker)
    
    if start_idx == -1:
        return None

    # 指针移动到 "{" 之后
    idx = start_idx + len(start_marker)
    balance = 1 # 括号平衡计数器
    content_start = idx
    
    while idx < len(text):
        char = text[idx]
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1
            
        if balance == 0:
            # 找到匹配的结束括号，返回中间的内容
            return text[content_start:idx]
        
        idx += 1
        
    return None

def normalize_answer(ans: Any) -> str:
    """标准化答案格式以进行比对 (去空格、转小写、去逗号)"""
    if ans is None: return ""
    return str(ans).strip().replace(",", "").replace(" ", "").lower()

def run_smol_task(executor: LocalPythonExecutor, code: str) -> str:
    """执行 Python 代码，带 5秒 超时控制"""
    output_capture = io.StringIO()
    
    def _unsafe_exec():
        with contextlib.redirect_stdout(output_capture):
            try:
                result = executor(code)
                if result is not None:
                    print(result)
            except Exception as e:
                # 捕获错误并打印，作为 Observation 返回给模型
                print(f"Error: {e}")

    try:
        func_timeout(5, _unsafe_exec)
        output = output_capture.getvalue().strip()
        if not output:
            output = "Code executed successfully (no output)."
        return output
    except FunctionTimedOut:
        return "Execution Error: Code execution timed out (limit: 5s). Likely an infinite loop."
    except Exception as e:
        return f"Execution Error: {str(e)}"

# ================= 核心类 =================

class BatchAgentRunner:
    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.9):
        print(f"正在加载 vLLM 模型: {model_path} ...")
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=4, # 根据显卡数量调整
            max_model_len=8192
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            # 增加停止词，防止模型替用户生成 Observation
            stop=["Observation:", "<end_code>", "Observation"] 
        )

    def process_batch(self, batch_data: List[Dict], max_steps: int = 6) -> List[Dict]:
        active_indices = list(range(len(batch_data)))
        histories = []
        
        # 初始化
        for item in batch_data:
            # 兼容不同数据集的字段名
            question = item.get('problem') or item.get('question') or item.get('instruction') or item.get('input')
            if not question:
                question = "Error: No question found"
            
            # 构造初始对话
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
            
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            histories.append({
                "original_data": item,
                "messages": messages, 
                "current_prompt": prompt_text,
                "finished": False,
                "final_answer": None,
                "is_correct": False,
                "executor_obj": LocalPythonExecutor(
                    additional_authorized_imports=["math", "sympy", "itertools", "collections", "re", "random", "fractions", "numpy"]
                )
            })

        # 多步推理循环
        for step in range(max_steps):
            if not active_indices:
                break
            
            current_prompts = [histories[i]["current_prompt"] for i in active_indices]
            outputs = self.llm.generate(current_prompts, self.sampling_params)
            
            next_active_indices = []
            execution_tasks = [] 
            
            # --- 解析与调度 ---
            for i, output in zip(active_indices, outputs):
                generated_text = output.outputs[0].text
                
                # 更新历史
                histories[i]["messages"].append({"role": "assistant", "content": generated_text})
                histories[i]["current_prompt"] += generated_text
                
                # 1. 检查代码 (优先级最高)
                code_pattern = r"```python\s*(.*?)\s*```"
                match = re.search(code_pattern, generated_text, re.DOTALL)
                
                # 2. [关键修改] 检查 Boxed 答案 (即使没有 "FINAL ANSWER" 字样)
                extracted_answer = extract_boxed_answer(generated_text)
                
                # 3. 检查文字结束标记 (作为保底)
                has_final_text = "FINAL ANSWER" in generated_text or "final answer" in generated_text.lower()
                
                if match:
                    # 发现代码 -> 准备执行
                    code_block = match.group(1).strip()
                    execution_tasks.append({
                        "index": i,
                        "code": code_block,
                        "executor": histories[i]["executor_obj"]
                    })
                    next_active_indices.append(i)
                    
                elif extracted_answer is not None:
                    # [关键修改] 只要提取到 boxed 内容，就认为结束
                    histories[i]["final_answer"] = extracted_answer
                    histories[i]["finished"] = True
                    
                    # 立即校验正确性
                    ground_truth = histories[i]["original_data"].get('answer') or histories[i]["original_data"].get('solution')
                    if ground_truth:
                        norm_gt = normalize_answer(ground_truth)
                        norm_pred = normalize_answer(extracted_answer)
                        # 宽松匹配：只要预测答案包含在 GT 中，或者 GT 包含在预测中 (处理单位或格式差异)
                        if norm_pred in norm_gt or norm_gt in norm_pred:
                             histories[i]["is_correct"] = True
                             
                elif has_final_text:
                    # 模型说了结束，但没提取到 boxed (可能是格式错)，强制结束避免复读
                    histories[i]["finished"] = True
                    
                else:
                    # 既没代码也没答案 -> 发送提醒
                    reminder = "\nObservation: You did not output code or a final answer. Please write python code or output FINAL ANSWER."
                    histories[i]["messages"].append({"role": "user", "content": reminder})
                    histories[i]["current_prompt"] += reminder
                    next_active_indices.append(i)

            # --- 并行执行代码 ---
            if execution_tasks:
                results = {}
                # 使用线程池并发执行，避免 IO 阻塞
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor_pool:
                    future_to_idx = {
                        executor_pool.submit(run_smol_task, t["executor"], t["code"]): t["index"]
                        for t in execution_tasks
                    }
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        results[idx] = future.result()
                
                # 将结果反馈给模型
                for idx, obs in results.items():
                    obs_text = f"\nObservation: {obs}\n"
                    histories[idx]["messages"].append({"role": "user", "content": obs_text})
                    histories[idx]["current_prompt"] += obs_text
            
            active_indices = next_active_indices

        return histories

# ================= 主流程 =================

def main():
    parser = argparse.ArgumentParser(description="Collect Agent Trajectories for Distillation")
    parser.add_argument("--input_file", type=str, default="mathoai.jsonl")
    parser.add_argument("--output_file", type=str, default="distillation_data.jsonl")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    data = []
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Input file {args.input_file} not found.")
        sys.exit(1)
    
    if args.limit > 0:
        data = data[:args.limit]
    
    print(f"Loaded {len(data)} samples.")
    runner = BatchAgentRunner(args.model)

    print(f"Start processing... Output will be streamed to {args.output_file}")
    
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(data), args.batch_size), desc="Distilling"):
            batch = data[i : i + args.batch_size]
            try:
                results = runner.process_batch(batch)
                
                for res in results:
                    record = {
                        "messages": res["messages"],     # SFT 训练核心数据
                        "is_correct": res["is_correct"], # 过滤标记
                        "generated_answer": res["final_answer"],
                        "ground_truth": res["original_data"].get('answer', ''),
                        "original_id": res["original_data"].get('id', '')
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
            except Exception as e:
                print(f"Batch Error: {e}")
                # 打印 traceback 方便调试
                import traceback
                traceback.print_exc()
                continue

    print(f"\nDone! Saved to {args.output_file}")
    print("To filter correct trajectories for training, run:")
    print(f"grep '\"is_correct\": true' {args.output_file} > train_clean.jsonl")

if __name__ == "__main__":
    main()