import json
import re
import argparse
import os
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, default="selected_prompts.json")
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="final_results.json")
    return parser.parse_args()

def extract_answer(generated_text):
    if not generated_text: return "Unknown"
    
    # 1. 匹配 LaTeX 格式 \boxed{A}
    boxed_match = re.search(r'\\boxed\{([A-G])\}', generated_text)
    if boxed_match: return boxed_match.group(1).upper()
    
    # 2. 核心正则匹配：覆盖常见的回答模板
    patterns = [
        # 匹配 "The correct answer is: A" 或 "The correct option is: A"
        r'[Cc]orrect\s+(?:answer|option)\s+is\s*:?\s*([A-G])(?!\w)', 
        # 匹配 "Answer is: A" (将 \s+ 改为 \s* 修复了原代码冒号前没有空格就匹配失败的 Bug)
        r'[Aa]nswer\s+is\s*:?\s*([A-G])(?!\w)',                      
        r'[Oo]ption\s+is\s*:?\s*([A-G])(?!\w)',                      
        r'(?:Final\s+)?[Aa]nswer:\s*([A-G])(?!\w)',                  
        r'[Aa]nswer:\s*\(([A-G])\)',                                 
        r'[Aa]nswer:\s*([A-G])\)'                                    
    ]
    for pattern in patterns:
        match = re.search(pattern, generated_text)
        if match: return match.group(1).upper()

    # 3. 匹配 Markdown 加粗的选项 **A**
    bold_match = re.search(r'\*\*([A-G])\*\*', generated_text)
    if bold_match: return bold_match.group(1).upper()

    # 4. 倒序查找行首的 A) 或 A.
    lines = generated_text.split('\n')
    for line in reversed(lines):
        line_match = re.search(r'^\s*([A-G])[)\.]', line)
        if line_match: return line_match.group(1).upper()
            
    # 5. 极端兜底匹配：提取文本中出现的最后一个独立的 A-G 字母
    all_matches = re.findall(r'\b([A-G])\b', generated_text)
    if all_matches: return all_matches[-1].upper()

    return "Unknown"

def main():
    args = parse_args()
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    prompts = [item['best_prompt'] for item in data]
    
    # 启动 vLLM
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        dtype="float16",
        gpu_memory_utilization=0.9, # 关键：留出20%空间给系统和残留
        max_model_len=16384,
        enforce_eager=True
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=8192)
    
    outputs = llm.generate(prompts, sampling_params)
    
    correct = 0
    results = []
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        pred = extract_answer(gen_text)
        gt = data[i]['ground_truth']
        
        is_correct = (pred == gt.strip().upper())
        if is_correct: correct += 1
        
        results.append({
            "id": data[i]['target_id'],
            "prediction": pred,
            "ground_truth": gt,
            "is_correct": is_correct,
            "text": gen_text
        })
    
    print(f"Final Accuracy: {correct / len(results):.2%}")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()