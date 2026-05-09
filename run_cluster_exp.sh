#!/bin/bash

# 配置环境
MODEL_PATH="/data_a100/models/Qwen2.5-7B-Instruct"
LAYER_TO_CHECK=16
SAMPLE_COUNT=200 # 建议每个数据集 100-300 个样本以保证散点图清晰
GPU_ID="7"

# 定义输出路径
OUT_DIR="./cluster_results"
mkdir -p $OUT_DIR

# 运行聚类分析
# 传入的数据集建议涵盖：多个逻辑数据集(应聚在一起) + 常识数据集(应分开) + 负对照(应分散)
python cluster_delta_h.py \
    --model "$MODEL_PATH" \
    --layer $LAYER_TO_CHECK \
    --samples $SAMPLE_COUNT \
    --gpu "$GPU_ID" \
    --out "$OUT_DIR/tsne_comparison_layer_${LAYER_TO_CHECK}.png" \
    --datasets \
        "FOLIO:./data/FOLIO/train.json" \
        "AR-LSAT:./data/AR-LSAT/train.json" \
        "ProntoQA:./data/ProntoQA/test.json" \
        "CommonsenseQA:./data/CSQA/train.json" \
        "SocialIQA:./data/SocialIQA/train.json"