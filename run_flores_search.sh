#!/bin/bash

# 1. 显卡与通用配置
GPU=1,2,5,6
export CUDA_VISIBLE_DEVICES="${GPU}"
MODEL_NAME="Qwen2.5-3B-Instruct"

# 2. 吞吐量配置：在此处统一修改推理的 Batch Size
BATCH_SIZE=16  # 可以根据显存大小调整，如 8, 16, 32

# 3. 搜索空间定义
LAYERS=(8 10 12 14 16 18)
ALPHAS=(0.0 0.2 0.5 1.0 1.2 1.5)

# 创建结果目录
mkdir -p eval_results

# 4. 定义全局总日志文件路径
TIMESTAMP=$(date +"%m%d_%H%M")
TOTAL_LOG="logs/${MODEL_NAME}/flores/sweep_total_${TIMESTAMP}.log"

# 🔥【核心修复】动态创建日志文件的所有上级父目录
mkdir -p "$(dirname "$TOTAL_LOG")"

echo "=== 🚀 开始网格参数搜索 (Grid Sweep) ==="
echo "[*] 待测试 Layer 序列: ${LAYERS[*]}"
echo "[*] 待测试 Alpha 序列: ${ALPHAS[*]}"
echo "[*] 推理 Batch Size   : ${BATCH_SIZE}"
echo "[*] 🔔 运行日志将实时输出并保存至: ${TOTAL_LOG}"
echo "================================================="

# 全局重定向接管（由于上步创建了目录，这里将畅通无阻）
exec > >(tee -a "$TOTAL_LOG") 2>&1

for layer in "${LAYERS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        
        echo "-------------------------------------------------"
        echo "[Sweep] 正在运行 -> Layer: ${layer} | Alpha: ${alpha}"
        echo "-------------------------------------------------"
        
        # 动态生成每个参数组合的 jsonl 结果文件
        OUTPUT_FILE="eval_results/flores_L${layer}_A${alpha}.jsonl"
        
        # 执行 Python 评测程序
        python zero_shot_steering_test.py \
            --model "/home/hit/models/${MODEL_NAME}" \
            --dataset "flores" \
            --data_type "text" \
            --flores_dir "data/flores101_dataset" \
            --src_lang "jpn" \
            --tgt_lang "kor" \
            --layer "${layer}" \
            --alpha "${alpha}" \
            --calib_samples 500 \
            --max_test_samples 1012 \
            --eval_batch_size "${BATCH_SIZE}" \
            --output_file "${OUTPUT_FILE}"
            
        echo "[Sweep] Layer: ${layer} | Alpha: ${alpha} 运行结束。"
        echo ""
    done
done

echo "================================================="
echo "🎉 所有参数网格搜索已全部完成！结果已保存在 eval_results/ 目录下。"