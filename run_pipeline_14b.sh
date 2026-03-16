#!/bin/bash
set -e  # 遇到错误立即停止

# ================= 基础配置 =================
BASE_PATH="/userhome/huggingface"

# 模型配置 (切换模型时只需修改这两行)
MODEL_DIR_NAME="Qwen2.5-14B-Instruct" 
MODEL_SHORT="qwen2.5-14b" # 用于文件命名的简写
MODEL_PATH="$BASE_PATH/$MODEL_DIR_NAME"

# 定义所有的域和 k-shot 数量
DOMAINS=("FOLIO" "ProofWriter" "LogicalDeduction")
K_SHOTS=(0 1 2 3 4)

# 确保输出目录存在
mkdir -p ./tmp

echo ">>> Starting Cross-Domain Pipeline for $MODEL_DIR_NAME..."

# ================= 嵌套循环：源域 -> 目标域 -> K-shot =================
for SOURCE in "${DOMAINS[@]}"; do
    for TARGET in "${DOMAINS[@]}"; do
        
        # 排除同域情况（作为纯粹的跨域评测。如果你也需要跑同域评测，可以注释掉这三行）
        if [ "$SOURCE" == "$TARGET" ]; then
            continue
        fi
        
        # 配置当前方向的数据路径
        SOURCE_DATA="/code/data/$SOURCE/${SOURCE}_train_cot.json"
        TARGET_DEV="/code/data/$TARGET/${TARGET}_dev_cot.json"
        TARGET_PSEUDO="/code/data/$TARGET/${TARGET}_train_cot.json"
        
        for K in "${K_SHOTS[@]}"; do
            
            # 动态生成文件名，包含模型、源域、目标域和k-shot信息
            TMP_PROMPTS="./tmp/prompts_${MODEL_SHORT}_${SOURCE}_to_${TARGET}_${K}shot.json"
            OUTPUT_FILE="./tmp/results_${MODEL_SHORT}_${SOURCE}_to_${TARGET}_${K}shot.json"
            
            # ================= 新增：跳过已完成实验的判断 =================
            if [ -f "$OUTPUT_FILE" ]; then
                echo "================================================================="
                echo "[Skip] Result already exists for $SOURCE -> $TARGET ($K-shot)."
                echo "       File: $OUTPUT_FILE"
                echo "================================================================="
                continue # 文件已存在，直接跳过，进行下一个 k-shot 或下一个方向
            fi
            # ==============================================================
            
            echo "================================================================="
            echo "[Task] Model: $MODEL_SHORT | Transfer: $SOURCE -> $TARGET | K-shot: $K"
            echo "================================================================="
            
            echo "Step 1: Running Selector"
            CUDA_VISIBLE_DEVICES=0,1,2,3 python step1_selector_mlp.py \
                --model_path "$MODEL_PATH" \
                --source_path "$SOURCE_DATA" \
                --target_train_path "$TARGET_PSEUDO" \
                --target_test_path "$TARGET_DEV" \
                --k_shot "$K" \
                --batch_size 4 \
                --output_json "$TMP_PROMPTS"

            echo "Step 1 Finished. Cleaning memory for 5 seconds..."
            sleep 5

            echo "Step 2: Running vLLM Inference"
            CUDA_VISIBLE_DEVICES=0,1,2,3 python step2_inference.py \
                --model_path "$MODEL_PATH" \
                --input_json "$TMP_PROMPTS" \
                --tp_size 4 \
                --output_file "$OUTPUT_FILE"
                
            echo ">>> Finished $SOURCE -> $TARGET with ${K}-shot"
            echo ""
        done
    done
done

echo "All tasks completed successfully!"