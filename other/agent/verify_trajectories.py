import json
import re
import os
import sys
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接出 Qwen_Math/evaluation 的完整路径
qwen_eval_path = os.path.join(current_dir, 'Qwen_Math', 'evaluation')

# 将其加入到系统路径的最前面
sys.path.insert(0, qwen_eval_path)
# ================= 假设你把 MATHEvaluator 代码保存在 math_evaluator.py =================
# 如果不是，请修改这里导入你的类

from evaluator import MATHEvaluator


def extract_final_answer(trajectory_log):
    """
    从 Agent 的完整日志中提取 FINAL ANSWER 后面的内容。
    """
    # 匹配模式：寻找 "FINAL ANSWER: " 及其后的内容
    # 考虑到 Agent 可能会打印引号等，我们尽量宽容匹配
    pattern = r"FINAL ANSWER:\s*(.*)"
    matches = re.findall(pattern, trajectory_log)
    
    if matches:
        # 取最后一个匹配到的答案（防止中间步骤也有类似输出）
        raw_answer = matches[-1].strip()
        
        # 清理可能存在的额外引号 (因为代码里是 print("FINAL ANSWER: " + answer))
        # 有时候模型会输出 FINAL ANSWER: "42"，我们需要去掉引号
        if raw_answer.startswith('"') and raw_answer.endswith('"'):
            raw_answer = raw_answer[1:-1]
        if raw_answer.startswith("'") and raw_answer.endswith("'"):
            raw_answer = raw_answer[1:-1]
            
        return raw_answer
    return None

def main():
    # 1. 配置路径
    input_file = "mathoai_trajectories.jsonl"  # 你的轨迹文件
    output_file = "mathoai_verified.jsonl"     # 验证后的结果
    
    # 2. 初始化评估器
    # data_name 参数决定了 parse_ground_truth 如何解析 dataset
    # 假设 math_oai 是 Qwen-Math 支持的标准格式之一
    evaluator = MATHEvaluator(data_name='math') 
    
    print(f"正在加载数据: {input_file} ...")
    
    results = []
    correct_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 3. 遍历评估
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines, desc="Evaluating"):
            if not line.strip():
                continue
                
            item = json.loads(line)
            total_count += 1
            
            # --- 步骤 A: 提取 Agent 答案 ---
            # item['full_prompt_log'] 包含了完整的交互历史
            agent_answer_text = extract_final_answer(item.get('full_prompt_log', ''))
            
            is_correct = False
            
            if agent_answer_text:
                # --- 步骤 B: 构造适配 Evaluator 的“伪预测” ---
                # MATHEvaluator.score 内部会调用 get_pred -> run_execute
                # 为了防止 run_execute 重新去跑 Agent 的复杂代码（可能报错或很慢），
                # 我们直接构造一个最简单的 boxed 格式，强行告诉 Evaluator 这是答案。
                # 例如：如果 Agent 说是 42，我们就构造 "\boxed{42}"
                fake_prediction = f"\\boxed{{{agent_answer_text}}}"
                
                # --- 步骤 C: 调用 Evaluator ---
                # prediction: 我们构造的 boxed 字符串
                # sample: 原始数据 item['original_data']，里面包含 ground truth
                try:
                    # score 返回 True/False
                    is_correct = evaluator.score(fake_prediction, item['original_data'])
                except Exception as e:
                    print(f"Error evaluating sample {total_count}: {e}")
                    is_correct = False
            else:
                # 没找到答案算错
                is_correct = False

            if is_correct:
                correct_count += 1

            # --- 步骤 D: 保存结果 ---
            # 我们把评估结果写回记录，方便后续筛选
            item['evaluation'] = {
                'extracted_answer': agent_answer_text,
                'is_correct': is_correct
            }
            
            # 实时写入
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            results.append(item)

    # 4. 输出统计
    accuracy = correct_count / total_count if total_count > 0 else 0
    print("\n" + "="*30)
    print(f"评估完成！")
    print(f"总样本数: {total_count}")
    print(f"正确样本数: {correct_count}")
    print(f"准确率 (Accuracy): {accuracy:.2%}")
    print(f"结果已保存至: {output_file}")
    print("="*30)

if __name__ == "__main__":
    main()