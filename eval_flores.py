import os
import argparse
import sacrebleu
from datasets import load_dataset

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Evaluation on FLORES Dataset using vLLM")
    # 模型与硬件配置
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="Hugging Face model ID or local path")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    
    # 数据类型配置
    parser.add_argument("--data_type", type=str, default="parquet", choices=["parquet", "text"], help="Type of FLORES data files to load (parquet or text)")

    # 任务配置
    parser.add_argument("--src_lang", type=str, default="eng_Latn", help="Source language code (e.g., eng_Latn)")
    parser.add_argument("--tgt_lang", type=str, default="zho_Simpl", help="Target language code (e.g., zho_Simpl)")
    parser.add_argument("--is_chat_model", action="store_true", default=True, help="Whether the model is an Instruct/Chat model")
    
    # Prompt 实验策略配置
    parser.add_argument("--prompt_strategy", type=str, default="single", choices=["single", "repeated"], 
                        help="Prompt configuration strategy: 'single' or 'repeated' for reinforcing context/steering.")
    
    # Few-shot 配置
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples (0 for zero-shot)")
    
    return parser.parse_args()


def load_flores_data(src_lang, tgt_lang, shots=0, data_type="text", local_data_dir="data/flores"):
    """从本地绝对/相对路径加载 FLORES 数据"""
    # 1. 拼接本地文件的路径
    if data_type == "parquet":
        src_test_path = os.path.join(local_data_dir, f"{src_lang}/devtest-00000-of-00001.parquet")
        tgt_test_path = os.path.join(local_data_dir, f"{tgt_lang}/devtest-00000-of-00001.parquet")
        if shots > 0:
            src_dev_path = os.path.join(local_data_dir, f"{src_lang}/dev-00000-of-00001.parquet")
            tgt_dev_path = os.path.join(local_data_dir, f"{tgt_lang}/dev-00000-of-00001.parquet")   
    elif data_type == "text":
        local_data_dir = "data/flores101_dataset"
        src_test_path = os.path.join(local_data_dir, f"devtest/{src_lang}.devtest")
        tgt_test_path = os.path.join(local_data_dir, f"devtest/{tgt_lang}.devtest")
        if shots > 0:
            src_dev_path = os.path.join(local_data_dir, f"dev/{src_lang}.dev")
            tgt_dev_path = os.path.join(local_data_dir, f"dev/{tgt_lang}.dev")
    else:
        raise ValueError(f"Unsupported data_type: {data_type}. Choose 'parquet' or 'text'.")

    print(f"[*] Loading local FLORES devtest from:\n  - {src_test_path}\n  - {tgt_test_path}")
    
    dev_data = None
    # 2. 加载数据集 (本地单个文件加载后，split 需固定填写 "train")
    if src_test_path.endswith(".parquet") and tgt_test_path.endswith(".parquet"):
        print("[*] Detected local Parquet files. Loading using datasets.load_dataset...")
        src_test = load_dataset("parquet", data_files=src_test_path, split="train")
        tgt_test = load_dataset("parquet", data_files=tgt_test_path, split="train")
        if shots > 0:
            print(f"[*] Loading local FLORES dev (Few-shot) from:\n  - {src_dev_path}\n  - {tgt_dev_path}")
            src_dev = load_dataset("parquet", data_files=src_dev_path, split="train")
            tgt_dev = load_dataset("parquet", data_files=tgt_dev_path, split="train")
            dev_data = {"src": src_dev, "tgt": tgt_dev}
    else:
        src_test = load_dataset("text", data_files=src_test_path, split="train")
        tgt_test = load_dataset("text", data_files=tgt_test_path, split="train")
        if shots > 0:
            print(f"[*] Loading local FLORES dev (Few-shot) from:\n  - {src_dev_path}\n  - {tgt_dev_path}")
            src_dev = load_dataset("text", data_files=src_dev_path, split="train")
            tgt_dev = load_dataset("text", data_files=tgt_dev_path, split="train")
            dev_data = {"src": src_dev, "tgt": tgt_dev}
        
    return src_test, tgt_test, dev_data


def build_prompts(src_test, src_lang, tgt_lang, is_chat_model, tokenizer, data_type="text", dev_data=None, shots=0, prompt_strategy="single"):
    """根据模型类型、Few-shot 设置以及 Prompt 策略（单次/重复）构建输入 Prompts"""
    prompts = []
    
    # 统一动态列名 Key，防止 Few-shot 部分在使用 Parquet 时引发 KeyError
    key_name = "sentence" if data_type == "parquet" else "text"
    
    lang_mapping = {
        "eng_Latn": "English", "zho_Hans": "Chinese (Simplified)", "zho_Hant": "Chinese (Traditional)",
        "deu_Latn": "German", "fra_Latn": "French", "jpn_Jpan": "Japanese",
        "eng": "English", "zho_simpl": "Chinese (Simplified)", "zho_trad": "Chinese (Traditional)",
        "deu": "German", "fra": "French", "jpn": "Japanese",
        "kor": "Korean", "spa_Latn": "Spanish", "rus_Cyrl": "Russian",
    }
    src_name = lang_mapping.get(src_lang, src_lang)
    tgt_name = lang_mapping.get(tgt_lang, tgt_lang)

    print(f"[*] Constructing prompts ({'Chat/Instruct' if is_chat_model else 'Base'} mode, {shots}-shot, strategy: {prompt_strategy})...")
    
    for idx, item in enumerate(src_test):
        src_text = item[key_name]
        
        if is_chat_model:
            # ==================== 1. 指令/对话模型逻辑 ====================
            messages = []
            system_prompt = f"You are a professional translator. Translate the following text from {src_name} to {tgt_name}. Provide only the final translation without any explanations or extra commentary."
            messages.append({"role": "system", "content": system_prompt})
            
            # 插入 Few-shot 示例
            if shots > 0 and dev_data:
                for i in range(shots):
                    dev_src = dev_data['src'][i][key_name]
                    dev_tgt = dev_data['tgt'][i][key_name]
                    
                    if prompt_strategy == "repeated":
                        # 对 Few-shot 样本也进行同样的重复增强，确保上下文表征模式一致
                        dev_user = f"Text to translate:\n{dev_src}\n\nRemember: Translate the above text from {src_name} to {tgt_name}. Provide only the final translation without any commentary:\n{dev_src}"
                    else:
                        dev_user = f"Text to translate:\n{dev_src}"
                        
                    messages.append({"role": "user", "content": dev_user})
                    messages.append({"role": "assistant", "content": dev_tgt})
            
            # 插入当前测试样本
            if prompt_strategy == "repeated":
                # 【重复 Prompt 策略】：通过在尾部引入二次强调与文本复述，强化注意力机制并改变表征激活
                current_user = f"Text to translate:\n{src_text}\n\nRemember: Translate the above text from {src_name} to {tgt_name}. Provide only the final translation without any commentary:\n{src_text}"
            else:
                current_user = f"Text to translate:\n{src_text}"
                
            messages.append({"role": "user", "content": current_user})
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        else:
            # ==================== 2. 基座模型续写逻辑 ====================
            prompt_str = ""
            if shots > 0 and dev_data:
                for i in range(shots):
                    dev_src = dev_data['src'][i][key_name]
                    dev_tgt = dev_data['tgt'][i][key_name]
                    if prompt_strategy == "repeated":
                        prompt_str += f"{src_name}: {dev_src}\nRemember: Translate to {tgt_name}.\n{src_name} (repeated): {dev_src}\n{tgt_name}: {dev_tgt}\n\n"
                    else:
                        prompt_str += f"{src_name}: {dev_src}\n{tgt_name}: {dev_tgt}\n\n"
            
            if prompt_strategy == "repeated":
                prompt_str += f"{src_name}: {src_text}\nRemember: Translate to {tgt_name}.\n{src_name} (repeated): {src_text}\n{tgt_name}:"
            else:
                prompt_str += f"{src_name}: {src_text}\n{tgt_name}:"
                
        prompts.append(prompt_str)
        
    return prompts


def calculate_metrics(predictions, references, tgt_lang):
    """使用 sacrebleu 计算标准的 BLEU 和 ChrF++ 指标"""
    print("[*] Calculating evaluation metrics...")
    
    if tgt_lang.startswith("zho"):
        tokenize_strategy = "zh"
    elif tgt_lang.startswith("jpn"):
        tokenize_strategy = "ja-mecab"
    elif tgt_lang.startswith("kor"):
        tokenize_strategy = "ko-mecab"
    else:
        tokenize_strategy = "13a"
        
    bleu_result = sacrebleu.corpus_bleu(predictions, [references], tokenize=tokenize_strategy)
    chrf_result = sacrebleu.corpus_chrf(predictions, [references])
    
    print("\n" + "="*40)
    print(f"📊 Evaluation Results ({tgt_lang})")
    print("="*40)
    print(f"BLEU Score  : {bleu_result.score:.2f}")
    print(f"ChrF++ Score: {chrf_result.score:.2f}")
    print(f"SacreBLEU Signature: {getattr(bleu_result, 'signature', 'N/A')}")
    print(f"Detailed BLEU Report: {bleu_result.format()}")
    print("="*40 + "\n")


def main():
    args = parse_args()
    
    if args.data_type == "parquet":
        LOCAL_DIR = "data/flores"
    else:
        LOCAL_DIR = "data/flores101_dataset"
        
    # 1. 加载数据
    src_test, tgt_test, dev_data = load_flores_data(
        args.src_lang, 
        args.tgt_lang, 
        args.shots, 
        args.data_type,
        local_data_dir=LOCAL_DIR
    )
    
    if args.data_type == "parquet":
        references = [item['sentence'] for item in tgt_test]
    elif args.data_type == "text":
        references = [item['text'] for item in tgt_test]
        
    # 2. 初始化 Tokenizer 并构建 Prompts
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    prompts = build_prompts(
        src_test=src_test, 
        src_lang=args.src_lang, 
        tgt_lang=args.tgt_lang, 
        is_chat_model=args.is_chat_model, 
        tokenizer=tokenizer, 
        data_type=args.data_type,
        dev_data=dev_data, 
        shots=args.shots,
        prompt_strategy=args.prompt_strategy  # 👈 传入 Prompt 策略参数
    )
    
    # 3. 初始化 vLLM 引擎并执行批量推理
    print(f"[*] Initializing vLLM with {args.model_name} (TP Size: {args.tensor_parallel_size})...")
    llm = LLM(
        model=args.model_name, 
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )
    
    stop_tokens = ["\n", "English:", "Chinese:"] if not args.is_chat_model else None
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=512,
        stop=stop_tokens
    )
    
    print(f"[*] Running batch inference over {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)
    
    predictions = [output.outputs[0].text.strip() for output in outputs]
    
    # 4. 指标评估
    calculate_metrics(predictions, references, args.tgt_lang)
    
    # 将预测结果保存至本地以便后续进行指标消融对比
    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)
    filename = f"preds_{args.src_lang}_to_{args.tgt_lang}_{args.prompt_strategy}.txt"
    with open(f"{out_dir}/{filename}", "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred.replace("\n", " ") + "\n")
    print(f"[+] Predictions saved to {out_dir}/{filename}")


if __name__ == "__main__":
    main()