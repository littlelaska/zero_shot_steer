#!/bin/bash

GPU=0,1,2,3
export CUDA_VISIBLE_DEVICES="${GPU}"

# ================= 配置区域 =================
# 1. 模型绝对路径
MODEL_PATH="llms/Qwen2.5-3B-Instruct"
MODEL_NAME=$(basename "$MODEL_PATH")

# 2. 实验参数 (Zero-shot Steering)
# 因为是零样本干预，我们不再需要区分 SOURCE，直接在特定数据集上验证
DATASET="ProofWriter"  # 也可以换成 "FOLIO" 或 "ProofWriter"
LAYERS="12 16 20 24"        # 建议扫几个不同的层位，寻找“全局信息整合”最集中的层
ALPHAS="0.5 1.0 1.5"        # 干预强度网格搜索
MODE="static"
CALIB_SAMPLES=1000           # 用于提取 Δh 的无标签样本数量
CONTEXT_REVERSE=true         # 用于将context放在question和option之后

# ================= 路径准备 =================
# 构造新的输出文件夹路径
OUT_DIR="./zero_shot_steering/${MODE}/${MODEL_NAME}/${DATASET}"
mkdir -p "$OUT_DIR"

# 数据集路径设置 (复用目标域的 train 作为校准，dev 作为测试)
BASE_DATA_DIR="data/${DATASET}"
CALIB_FILE="${BASE_DATA_DIR}/train.json"
TEST_FILE="${BASE_DATA_DIR}/dev.json"

# laska修改，新增
if [ "$CONTEXT_REVERSE" = true ]; then
  python zero_shot_steering.py \
      --calib_file "$CALIB_FILE" \
      --test_file "$TEST_FILE" \
      --model "$MODEL_PATH" \
      --layer 24 \
      --calib_samples 10 \
      --alpha 0.0 \
      --intervention_mode "$MODE" \
      --reverse_context \
      --output_file "${OUT_DIR}/results_reverse_alpha_0.0.jsonl"
fi


# ================= 循环执行 =================
# 嵌套循环跑网格搜索 (层数 x 干预强度)
for layer in $LAYERS
do
    for alpha in $ALPHAS
    do
        echo "--------------------------------------------------"
        echo "Model: $MODEL_NAME | Dataset: $DATASET"
        echo "Steering Layer: $layer | Alpha: $alpha | Mode: $MODE"
        echo "Output Path: $OUT_DIR"
        echo "--------------------------------------------------"
        
        python zero_shot_steering.py \
            --calib_file "$CALIB_FILE" \
            --test_file "$TEST_FILE" \
            --model "$MODEL_PATH" \
            --layer $layer \
            --calib_samples $CALIB_SAMPLES \
            --alpha $alpha \
            --intervention_mode "$MODE" \
            --output_file "${OUT_DIR}/results_layer_${layer}_alpha_${alpha}.jsonl"

        echo "Done: Layer $layer, Alpha $alpha"
    done
done

# 跑一个 Baseline (alpha=0.0，不加干预) 用于对比
echo "--------------------------------------------------"
echo "Running Baseline (No Intervention, Alpha=0.0)"
echo "--------------------------------------------------"
python zero_shot_steering.py \
    --calib_file "$CALIB_FILE" \
    --test_file "$TEST_FILE" \
    --model "$MODEL_PATH" \
    --layer 24 \
    --calib_samples 10 \
    --alpha 0.0 \
    --intervention_mode "$MODE" \
    --output_file "${OUT_DIR}/results_baseline_alpha_0.0.jsonl"