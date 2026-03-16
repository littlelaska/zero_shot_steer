import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import io
import contextlib

# =================配置区域=================
# 这里使用 Qwen2.5-0.5B-Instruct 是因为论文中使用了 Qwen 系列作为主要模型 [cite: 248]
# 并且 0.5B 可以在大多数本地机器（甚至无GPU）上运行以进行测试。
MODEL_PATH = "/userhome/huggingface/Qwen2.5-32B-Instruct/" 
device = "cuda" if torch.cuda.is_available() else "cpu"

# =================核心组件：工具（环境）=================
def python_interpreter(code_str):
    """
    模拟论文中的 Python 代码执行环境 [cite: 233, 252]。
    它可以捕获 print() 的输出作为 Observation。
    """
    output_capture = io.StringIO()
    try:
        # 简单的沙箱模拟，实际生产中请使用 docker 或 E2B [cite: 747]
        with contextlib.redirect_stdout(output_capture):
            exec(code_str, {'__name__': '__main__', 'print': print})
        result = output_capture.getvalue()
        if not result:
            result = "Code executed successfully but produced no output. Did you forget to print?"
        return result.strip()
    except Exception as e:
        return f"Execution Error: {str(e)}"

# =================核心组件：提示词（Prompt）=================
# 来源于论文附录 Prompt D.2 [cite: 814-832]
SYSTEM_PROMPT = """You are an expert assistant who can solve any task using code blobs.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of Thought:, Code:, and Observation: sequences.

Rules:
1. Always provide a Thought: sequence, and a Code: sequence ending with <end_code>, else you will fail.
2. Use print() to output important information for the next step.
3. The output of print() will appear in the Observation: field.
4. In the end you have to return a final answer using print("FINAL ANSWER: " + answer).
"""

# =================核心逻辑：Agent 循环=================
def run_agent_loop(question, max_steps=5):
    print(f"正在加载本地模型: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    # 初始化对话历史
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    # 将对话转换为模型的输入格式
    text_history = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"\n=== 任务开始: {question} ===\n")

    for step in range(max_steps):
        # 1. 本地模型推理 (Thought + Action Generation)
        inputs = tokenizer([text_history], return_tensors="pt").to(device)
        
        # 停止词设置：遇到 <end_code> 或 Observation 就停止，防止模型自问自答
        stop_words = ["<end_code>", "Observation:"]
        
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.0, # 论文中提到的 Greedy Decoding 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 打印模型的思考和生成的代码
        print(f"\n[Step {step+1}] Model Output:\n{response}")
        
        # 更新历史
        text_history += response
        
        # 2. 解析代码 (Parse Action)
        # 寻找 ```py ... ``` 或者 Code: ... <end_code>
        code_block = ""
        if "```python" in response:
            code_block = response.split("```python")[1].split("```")[0]
        elif "Code:" in response and "<end_code>" in response:
             # 提取 Code: 和 <end_code> 之间的内容
            code_block = response.split("Code:")[1].split("<end_code>")[0]
        
        # 如果模型输出了最终答案或没有代码，则终止
        if "FINAL ANSWER:" in response or not code_block.strip():
            print("\n=== 任务完成或无代码生成 ===")
            break

        # 3. 执行代码并获取观察结果 (Observation)
        print(f"\n[Step {step+1}] Executing Code...")
        observation = python_interpreter(code_block.strip())
        
        print(f"[Step {step+1}] Observation: {observation}")
        
        # 4. 将观察结果拼接回历史 (Append Observation)
        # 格式必须严格遵循论文：Observation: ... [cite: 817]
        obs_text = f"\nObservation: {observation}\n"
        text_history += obs_text

    return text_history

if __name__ == "__main__":
    # 示例问题：需要计算和逻辑推理的任务
    question = "Calculate the sum of the first 15 Fibonacci numbers."
    run_agent_loop(question)