#!/bin/bash

MODEL_PATH="/data_a100/models/Qwen2.5-7B-Instruct"
LAYER=16 # 选择你之前观察到交叉现象的关键中层
SAMPLES=150 # 每个数据集抽样 150 个点足以看清聚类
OUT_DIR="./clustering_results"
mkdir -p $OUT_DIR

# 运行聚类分析
# 建议对比：逻辑类(FOLIO/AR-LSAT) vs 常识类(CommonsenseQA) vs 负对照(Nonsense)
python cluster_delta_h.py \
    --model "$MODEL_PATH" \
    --layer $LAYER \
    --samples $SAMPLES \
    --datasets \
        "FOLIO:./data/FOLIO/train.json" \
        "AR-LSAT:./data/AR-LSAT/train.json" \
        "ProntoQA:./data/ProntoQA/test.json" \
        "CommonsenseQA:./data/CSQA/train.json" \
        "SocialIQA:./data/SocialIQA/train.json" \
        "Nonsense:./data/nonsense_baseline.json" \
    --gpu "7" \
    --out "$OUT_DIR/tsne_layer_${LAYER}.png"