#!/bin/bash

# ================= 配置区 =================
LOG_ROOT="./logs"   # 你的实验日志根目录
ANALYSIS_DIR="./steering_report"     # 分析结果存放地
# ==========================================

mkdir -p "$ANALYSIS_DIR"

echo "Step 1: 正在从日志中提取所有 Accuracy 数据..."
# 调用你之前的 collect_results.py
python collect_results.py \
    --log_dir "$LOG_ROOT" \
    --out_dir "$ANALYSIS_DIR"

GLOBAL_CSV="${ANALYSIS_DIR}/global_results.csv"

if [ -f "$GLOBAL_CSV" ]; then
    echo "Step 2: 汇总表已生成，开始执行 Scaling Law 深度分析..."
    # 调用最新的整合分析脚本
    python comprehensive_scaling_analysis.py \
        --csv "$GLOBAL_CSV" \
        --out "${ANALYSIS_DIR}/scaling_analysis"
    
    echo "=================================================="
    echo "分析全部完成！"
    echo "1. 各模型/数据集详情见: ${ANALYSIS_DIR}/"
    echo "2. Scaling Law 趋势图见: ${ANALYSIS_DIR}/scaling_analysis/scaling_law_comprehensive.png"
    echo "3. 最佳层数汇总表见: ${ANALYSIS_DIR}/scaling_analysis/scaling_best_results.csv"
    echo "=================================================="
else
    echo "[Error] 未能找到 global_all_results.csv，请检查 collect_results.py 是否正常运行。"
fi