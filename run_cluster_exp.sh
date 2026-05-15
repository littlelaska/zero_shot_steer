#!/bin/bash

# 配置环境
MODEL_PATH="llms/Qwen2.5-3B-Instruct"
MODEL_NAME=$(basename "$MODEL_PATH")
LAYER_TO_CHECK=4
LAYER_END=28
SAMPLE_COUNT=200 # 建议每个数据集 100-300 个样本以保证散点图清晰
GPU_ID="0,1,2,3"
BATCH_SIZE=16
# 定义输出路径
OUT_DIR="./cluster_results/${MODEL_NAME}"
mkdir -p $OUT_DIR

# 运行聚类分析
# 传入的数据集建议涵盖：多个逻辑数据集(应聚在一起) + 常识数据集(应分开) + 负对照(应分散)
for LAYER in $(seq 1 "$LAYER_END"); do
    python cluster_delta_h.py \
        --model "$MODEL_PATH" \
        --layer $LAYER \
        --batch_size $BATCH_SIZE \
        --samples $SAMPLE_COUNT \
        --gpu "$GPU_ID" \
        --out "$OUT_DIR/tsne_comparison_layer_${LAYER}.png" \
        --datasets \
            "LogicalDeduction:./data/LogicalDeduction/train.json" \
            "FOLIO:./data/FOLIO/train.json" \
            "AR-LSAT:./data/AR-LSAT/train.json" \
            "ProntoQA:./data/ProntoQA/dev.json" \
            "ProofWriter:./data/ProofWriter/train.json" \
            "CommonsenseQA:./data/commonsense/train.json" \
            "gsm8k:./data/gsm8k/train.json"
done
# # 运行聚类分析
# # 传入的数据集建议涵盖：多个逻辑数据集(应聚在一起) + 常识数据集(应分开) + 负对照(应分散)
# python cluster_delta_h.py \
#     --model "$MODEL_PATH" \
#     --layer $LAYER_TO_CHECK \
#     --batch_size $BATCH_SIZE \
#     --samples $SAMPLE_COUNT \
#     --gpu "$GPU_ID" \
#     --out "$OUT_DIR/tsne_comparison_layer_${LAYER_TO_CHECK}.png" \
#     --datasets \
#         "LogicalDeduction:./data/LogicalDeduction/train.json" \
#         "FOLIO:./data/FOLIO/train.json" \
#         "AR-LSAT:./data/AR-LSAT/train.json" \
#         "ProntoQA:./data/ProntoQA/dev.json" \
#         "ProofWriter:./data/ProofWriter/train.json" \
#         "CommonsenseQA:./data/commonsense/train.json" \
#         "gsm8k:./data/gsm8k/train.json"