import argparse
import json
import os
import sys
from typing import List, Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ================= 配置区域 =================
# 建议使用支持推理能力的模型，或者通过 Prompt 强行诱导普通模型进行思考
DEFAULT_MODEL = "/userhome/huggingface/Qwen2.5-32B-Instruct/"

# System Prompt: 强制要求模型使用 <think> 标签进行思考
# 这是收集高质量 CoT 的关键
SYSTEM_PROMPT_COT = """You are an expert assistant who can answer the given question accurately and provide
clear reasoning.
When answering questions, follow these guidelines:
1. Provide a clear and structured reasoning first
2. Follow up with a final answer, must in the <answer> </answer> tag. For example,
<answer> xxx </answer>.
3. The answer must be succinct and final. For math problems, return the answer using
LaTeX in the \boxed format.
4. If the question requires multiple steps or facts, break down your reasoning accordingly
5. Be precise and factual in your responses
6. If you’re unsure about something, acknowledge the uncertainty
Now, please answer the following question:
"""

# ================= 核心类 =================

class CoTCollector:
    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.9):
        print(f"Loading vLLM model for CoT generation: {model_path} ...")
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=4, # 根据你的显卡数量调整
            max_model_len=8192
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        # 采样参数：通常 CoT 需要一定的多样性来探索不同路径，但为了质量通常设为低温
        self.sampling_params = SamplingParams(
            temperature=0.6, # 稍微给一点温度，让思维更发散；如果要确定性推理，设为 0
            top_p=0.95,
            max_tokens=8192, # CoT 通常很长，给足空间
            stop=[] # 不设特殊停止词，让模型自然讲完
        )

    def generate_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """
        批量生成带有思维链的回复
        """
        prompts = []
        original_items = []
        
        for item in batch_data:
            # 兼容常见数据集格式
            question = item.get('problem') or item.get('question') or item.get('instruction') or item.get('input')
            if not question:
                continue
                
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_COT},
                {"role": "user", "content": question}
            ]
            
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            prompts.append(prompt_token_ids)
            original_items.append(item)

        if not prompts:
            return []

        # vLLM 批量生成
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        results = []
        for item, output in zip(original_items, outputs):
            generated_text = output.outputs[0].text
            
            # 解析：尝试分离 <think> 和 正文
            cot_content = ""
            final_answer = ""
            
            # 简单的解析逻辑
            if "<think>" in generated_text and "</think>" in generated_text:
                parts = generated_text.split("</think>")
                cot_part = parts[0].replace("<think>", "").strip()
                answer_part = parts[1].strip()
                
                cot_content = cot_part
                final_answer = answer_part
            else:
                # 如果模型没遵循格式，就全部存下来
                cot_content = generated_text
                final_answer = "Parse Error: Structure not found"

            results.append({
                "original_data": item,
                "full_response": generated_text,
                "cot": cot_content,
                "answer": final_answer
            })
            
        return results

# ================= 主流程 =================

def main():
    parser = argparse.ArgumentParser(description="Collect Raw Chain of Thought (CoT)")
    parser.add_argument("--input_file", type=str, default="mathoai.jsonl")
    parser.add_argument("--output_file", type=str, default="cot_data.jsonl")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=50) # CoT 不需要 Python 执行，Batch 可以大一点
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
    
    collector = CoTCollector(args.model)

    print(f"Start generating CoT... Output: {args.output_file}")
    
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(data), args.batch_size), desc="Generating CoT"):
            batch = data[i : i + args.batch_size]
            try:
                results = collector.generate_batch(batch)
                
                for res in results:
                    # 保存为 SFT 格式 (DeepSeek-R1 风格)
                    record = {
                        "instruction": res["original_data"].get('question', ''),
                        "output": res["full_response"], # 包含 <think> 的完整回答
                        "structured_cot": res["cot"],   # 仅思维链
                        "structured_response": res["answer"], # 仅答案
                        "ground_truth": res["original_data"].get('answer', '')
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
            except Exception as e:
                print(f"Batch Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"Done! CoT data saved to {args.output_file}")

if __name__ == "__main__":
    main()