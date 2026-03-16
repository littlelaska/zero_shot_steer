#!/bin/bash
set -e  # 遇到错误立即停止

# ================= 基础配置 =================
BASE_PATH="/userhome/huggingface"

# 模型配置
MODEL_DIR_NAME="Qwen2.5-32B-Instruct" 
MODEL_SHORT="qwen2.5-32b"
MODEL_PATH="$BASE_PATH/$MODEL_DIR_NAME"

# 定义所有的域和 k-shot 数量
DOMAINS=("FOLIO" "ProofWriter" "LogicalDeduction")
K_SHOTS=(5 6 7 8 9)

# 任务开关 (方便你自由控制跑哪一部分)
RUN_BASELINE=true
RUN_CROSS_DOMAIN=false

# 确保输出目录存在
mkdir -p ./tmp

echo ">>> Starting Pipeline for $MODEL_DIR_NAME..."

# =================================================================
# 新增模块：Baseline (Double Query) - 仅依赖 Target 域
# =================================================================
if [ "$RUN_BASELINE" = true ]; then
    echo ">>> Phase 1: Running Baseline (Double Query)..."
    for TARGET in "${DOMAINS[@]}"; do
        TARGET_DEV="/code/data/$TARGET/${TARGET}_dev_cot.json"
        
        # 你的 Python 代码中 source_path 和 target_train_path 是必填项 (required=True)
        # 所以我们需要传一个有效的文件路径给它，即便 baseline 逻辑里根本不会用到它们
        DUMMY_PATH="/code/data/$TARGET/${TARGET}_train_cot.json"
        
        TMP_PROMPTS="./tmp/prompts_${MODEL_SHORT}_${TARGET}_repeat.json"
        OUTPUT_FILE="./tmp/results_${MODEL_SHORT}_${TARGET}_repeat.json"

        if [ -f "$OUTPUT_FILE" ]; then
            echo "-----------------------------------------------------------------"
            echo "[Skip] Baseline result already exists for $TARGET."
            echo "       File: $OUTPUT_FILE"
            echo "-----------------------------------------------------------------"
            continue
        fi

        echo "-----------------------------------------------------------------"
        echo "[Task] Model: $MODEL_SHORT | Baseline | Target: $TARGET"
        echo "-----------------------------------------------------------------"
        
        echo "Step 1: Running Selector (Baseline mode)"
        # 加入了 --baseline_double_query 标志
        CUDA_VISIBLE_DEVICES=0,1,2,3 python step1_selector_mlp.py \
            --model_path "$MODEL_PATH" \
            --source_path "$DUMMY_PATH" \
            --target_train_path "$DUMMY_PATH" \
            --target_test_path "$TARGET_DEV" \
            --output_json "$TMP_PROMPTS" \
            --baseline_double_query

        echo "Step 1 Finished. Cleaning memory for 5 seconds..."
        sleep 5

        echo "Step 2: Running vLLM Inference (Baseline)"
        CUDA_VISIBLE_DEVICES=0,1,2,3 python step2_inference.py \
            --model_path "$MODEL_PATH" \
            --input_json "$TMP_PROMPTS" \
            --tp_size 4 \
            --output_file "$OUTPUT_FILE"
            
        echo ">>> Finished Baseline for $TARGET"
        echo ""
    done
fi

# =================================================================
# 原有模块：跨域检索 (Cross-Domain Retrieval)
# =================================================================
if [ "$RUN_CROSS_DOMAIN" = true ]; then
    echo ">>> Phase 2: Running Cross-Domain Retrieval..."
    for SOURCE in "${DOMAINS[@]}"; do
        for TARGET in "${DOMAINS[@]}"; do
            
            # 排除同域情况
            if [ "$SOURCE" == "$TARGET" ]; then
                continue
            fi
            
            SOURCE_DATA="/code/data/$SOURCE/${SOURCE}_train_cot.json"
            TARGET_DEV="/code/data/$TARGET/${TARGET}_dev_cot.json"
            TARGET_PSEUDO="/code/data/$TARGET/${TARGET}_train_cot.json"
            
            for K in "${K_SHOTS[@]}"; do
                
                TMP_PROMPTS="./tmp/prompts_${MODEL_SHORT}_${SOURCE}_to_${TARGET}_${K}shot.json"
                OUTPUT_FILE="./tmp/results_${MODEL_SHORT}_${SOURCE}_to_${TARGET}_${K}shot.json"
                
                if [ -f "$OUTPUT_FILE" ]; then
                    echo "================================================================="
                    echo "[Skip] Result already exists for $SOURCE -> $TARGET ($K-shot)."
                    echo "       File: $OUTPUT_FILE"
                    echo "================================================================="
                    continue 
                fi
                
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
fi

echo "All tasks completed successfully!"