#!/bin/bash

# --- 实验参数配置 ---
MODEL_PATH="/data_a100/models/Qwen2.5-3B-Instruct"  # 修改为你的模型路径
MODEL_NAME=$(basename "$MODEL_PATH")
DATA_A="FOLIO"
DATA_B="AR-LSAT"
DATA_A_PATH="./data/${DATA_A}/train.json"
DATA_B_PATH="./data/${DATA_B}/train.json"
SAMPLES=1000
GPU_ID="7"  # 指定使用的显卡序号
# TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./delta_h_results/${DATA_A}_${DATA_B}_sim_analysis_${MODEL_NAME}"

# --- 执行分析 ---
echo ">>> Starting Aligned Analysis on GPU $GPU_ID..."
echo ">>> Model: $MODEL_PATH"

python delta_h_cos.py \
    --model "$MODEL_PATH" \
    --data_a "$DATA_A_PATH" \
    --data_b "$DATA_B_PATH" \
    --samples $SAMPLES \
    --gpu "$GPU_ID" \
    --out "$OUTPUT_DIR"

echo ">>> Analysis Complete. Results saved in $OUTPUT_DIR"
