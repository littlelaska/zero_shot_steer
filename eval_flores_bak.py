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
    
    # Few-shot 配置
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples (0 for zero-shot)")
    
    return parser.parse_args()


def load_flores_data(src_lang, tgt_lang, shots=0, data_type="text", local_data_dir="data/flores"):
    """从本地绝对/相对路径的 Parquet 文件加载 FLORES 数据"""
    import os
    # 1. 拼接本地文件的路径 (请根据你实际的本地文件名修改此处规则)
    if data_type == "parquet":
        local_data_dir = "data/flores"  # 请根据实际情况修改
        src_test_path = os.path.join(local_data_dir, f"{src_lang}/devtest-00000-of-00001.parquet")
        tgt_test_path = os.path.join(local_data_dir, f"{tgt_lang}/devtest-00000-of-00001.parquet")
        if shots > 0:
            src_dev_path = os.path.join(local_data_dir, f"{src_lang}/dev-00000-of-00001.parquet")
            tgt_dev_path = os.path.join(local_data_dir, f"{tgt_lang}/dev-00000-of-00001.parquet")   
    # 非parquet文件的加载逻辑
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
    # 2. 加载测试集 (本地单个文件加载后，split 需固定填写 "train")
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
    
    # if shots > 0:
    #     src_dev_path = os.path.join(local_data_dir, f"{src_lang}/dev-00000-of-00001.parquet")
    #     tgt_dev_path = os.path.join(local_data_dir, f"{tgt_lang}/dev-00000-of-00001.parquet")
        
    #     print(f"[*] Loading local FLORES dev (Few-shot) from:\n  - {src_dev_path}\n  - {tgt_dev_path}")
        
    #     # 加载开发集
    #     src_dev = load_dataset("parquet", data_files=src_dev_path, split="train")
    #     tgt_dev = load_dataset("parquet", data_files=tgt_dev_path, split="train")
    #     dev_data = {"src": src_dev, "tgt": tgt_dev}
        
    return src_test, tgt_test, dev_data


    # ---- 开始执行加载逻辑 ----
    print(f"[*] Automatically detecting and loading local files from '{local_data_dir}'...")
    
    # 加载测试集
    src_test = load_single_file(src_lang, "devtest")
    tgt_test = load_single_file(tgt_lang, "devtest")
    
    dev_data = None
    if shots > 0:
        # 如果需要 few-shot，加载开发集
        src_dev = load_single_file(src_lang, "dev")
        tgt_dev = load_single_file(tgt_lang, "dev")
        dev_data = {"src": src_dev, "tgt": tgt_dev}
        
    return src_test, tgt_test, dev_data

def build_prompts(src_test, src_lang, tgt_lang, is_chat_model, tokenizer, data_type="text",dev_data=None, shots=0):
    """根据模型类型与 Few-shot 设置构建输入 Prompts"""
    prompts = []
    
    # 简单的语言代码映射（用于 Prompt 提示语，可根据需要自行扩展）
    lang_mapping = {
        "eng_Latn": "English",
        "zho_Hans": "Chinese (Simplified)",
        "zho_Hant": "Chinese (Traditional)",
        "deu_Latn": "German",
        "fra_Latn": "French",
        "jpn_Jpan": "Japanese",
        "eng":"English",
        "zho_simpl":"Chinese (Simplified)",
        "zho_trad":"Chinese (Traditional)",
        "deu":"German",
        "fra":"French",
        "jpn":"Japanese"
    }
    src_name = lang_mapping.get(src_lang, src_lang)
    tgt_name = lang_mapping.get(tgt_lang, tgt_lang)

    print(f"[*] Constructing prompts ({'Chat/Instruct' if is_chat_model else 'Base'} mode, {shots}-shot)...")
    
    for idx, item in enumerate(src_test):
        if data_type == "parquet":
            src_text = item['sentence']
        elif data_type == "text":
            src_text = item['text']
        else:
            raise ValueError(f"Unsupported data_type: {data_type}. Choose 'parquet' or 'text'.")
        
        if is_chat_model:
            # 1. 指令/对话模型逻辑
            messages = []
            system_prompt = f"You are a professional translator. Translate the following text from {src_name} to {tgt_name}. Provide only the final translation without any explanations or extra commentary."
            messages.append({"role": "system", "content": system_prompt})
            
            # 插入 Few-shot 示例
            if shots > 0 and dev_data:
                for i in range(shots):
                    messages.append({"role": "user", "content": f"Text to translate:\n{dev_data['src'][i]['text']}"})
                    messages.append({"role": "assistant", "content": dev_data['tgt'][i]['text']})
            
            # 插入当前测试样本
            messages.append({"role": "user", "content": f"Text to translate:\n{src_text}"})
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        else:
            # 2. 基座模型续写逻辑 (Few-shot 效果更好)
            prompt_str = ""
            if shots > 0 and dev_data:
                for i in range(shots):
                    prompt_str += f"{src_name}: {dev_data['src'][i]['text']}\n{tgt_name}: {dev_data['tgt'][i]['text']}\n\n"
            prompt_str += f"{src_name}: {src_text}\n{tgt_name}:"
            
        prompts.append(prompt_str)
        
    return prompts


def calculate_metrics(predictions, references, tgt_lang):
    """使用 sacrebleu 计算标准的 BLEU 和 ChrF++ 指标"""
    print("[*] Calculating evaluation metrics...")
    
    # 动态选择分词策略
    if tgt_lang.startswith("zho"):
        tokenize_strategy = "zh"
    elif tgt_lang.startswith("jpn"):
        tokenize_strategy = "ja-mecab"
    elif tgt_lang.startswith("kor"):
        tokenize_strategy = "ko-mecab"
    else:
        tokenize_strategy = "13a"  # 绝大多数西方语言的通用分词
        
    # 计算 BLEU
    bleu_result = sacrebleu.corpus_bleu(
        predictions, 
        [references], 
        tokenize=tokenize_strategy
    )
    
    # 计算 ChrF++
    chrf_result = sacrebleu.corpus_chrf(predictions, [references])
    
    print("\n" + "="*40)
    print(f"📊 Evaluation Results ({tgt_lang})")
    print("="*40)
    print(f"BLEU Score  : {bleu_result.score:.2f}")
    print(f"ChrF++ Score: {chrf_result.score:.2f}")
    # print(f"SacreBLEU Signature: {bleu_result.signature}")
    # 选项 A：使用 getattr 保护，如果找不到属性就输出 N/A，100% 不会崩溃（推荐）
    print(f"SacreBLEU Signature: {getattr(bleu_result, 'signature', 'N/A')}")
    # 如果你还想看到诸如 1-4 gram 的精确度、BP惩罚项等详细信息，可以加上下面这行：
    print(f"Detailed BLEU Report: {bleu_result.format()}")
    print("="*40 + "\n")


def main():
    args = parse_args()
    # 设定你存放 parquet 文件的本地目录
    if args.data_type == "parquet":
        LOCAL_DIR = "data/flores"
    else:
        LOCAL_DIR = "data/flores101_dataset"
    # 1. 加载数据
    # 传入 LOCAL_DIR
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
    # vLLM 内部集成了 Tokenizer，但为了在不启动 LLM 引擎前构建 Chat Template，可以先用 transformers 载入
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    prompts = build_prompts(
        src_test=src_test, 
        src_lang=args.src_lang, 
        tgt_lang=args.tgt_lang, 
        is_chat_model=args.is_chat_model, 
        tokenizer=tokenizer, 
        data_type=args.data_type,
        dev_data=dev_data, 
        shots=args.shots
    )
    
    # 3. 初始化 vLLM 引擎并执行批量推理
    print(f"[*] Initializing vLLM with {args.model_name} (TP Size: {args.tensor_parallel_size})...")
    llm = LLM(
        model=args.model_name, 
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )
    
    # 机器翻译通常使用 Greedy Decoding (temperature=0.0) 以确保稳定性和可复现性
    stop_tokens = ["\n", "English:", "Chinese:"] if not args.is_chat_model else None
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=512,
        stop=stop_tokens
    )
    
    print(f"[*] Running batch inference over {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 提取并清洗模型输出
    predictions = [output.outputs[0].text.strip() for output in outputs]
    
    # 4. 指标评估
    calculate_metrics(predictions, references, args.tgt_lang)
    
    # 可选：将预测结果保存至本地以便后续使用 COMET 评估
    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/preds_{args.src_lang}_to_{args.tgt_lang}.txt", "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred.replace("\n", " ") + "\n")
    print(f"[+] Predictions saved to {out_dir}/preds_{args.src_lang}_to_{args.tgt_lang}.txt")


if __name__ == "__main__":
    main()