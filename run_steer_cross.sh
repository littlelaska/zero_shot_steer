#!/bin/bash

GPU=3,6
export CUDA_VISIBLE_DEVICES="${GPU}"

# ================= 配置区域 =================
# 1. 模型绝对路径
MODEL_PATH="/data_a100/models/Qwen2.5-7B-Instruct"
MODEL_NAME=$(basename "$MODEL_PATH")

# 2. 实验参数 (Zero-shot Steering)
# 因为是零样本干预，我们不再需要区分 SOURCE，直接在特定数据集上验证
TEST_DATASET="LogicalDeduction"  # 当前要进行测试的目标域数据集
TRAIN_DATASET="FOLIO"  # 用于抽取delta_h的校准数据集，通常和测试数据集相同（零样本干预），也可以换成其他数据集（比如 LogicalDeduction FOLIO ProntoQA AR-LSAT ProofWriter）
LAYERS="12 16 20 24"        # 建议扫几个不同的层位，寻找“全局信息整合”最集中的层
LAYERS="6 10 12 16 20 24 26 30 34"        # 建议扫几个不同的层位，寻找“全局信息整合”最集中的层
LAYERS="6 10 12 16 20 24 26"        # 建议扫几个不同的层位，寻找“全局信息整合”最集中的层
ALPHAS="0.5 1 1.5"        # 干预强度网格搜索
MODE="static"
CALIB_SAMPLES=1000           # 用于提取 Δh 的无标签样本数量
CONTEXT_REVERSE=true         # 用于将context放在question和option之后
EVAL_BATCH_SIZE=16           # 控制测试时的batch_size大小
INSTANCE_STEERING=false       # 控制干预向量是单个还是一致的
# MAX_LENGTH=1024               # 控制输入的最大长度，对所有的batch padding到这个长度，避免由于不同padding带来的性能差异
# MAX_TEST_SAMPLES=10           # 控制测试时的样本数量，避免测试时间过长（你可以根据需要调整这个值，或者设置为 None 来使用全部样本）

# ================= 路径准备 =================
# 构造新的输出文件夹路径
OUT_DIR="./zero_shot_steering/${MODE}/${MODEL_NAME}/${TEST_DATASET}_cross_from_${TRAIN_DATASET}"
mkdir -p "$OUT_DIR"
# ================= 日志存放准备 =================
LOG_DIR="./logs/${MODEL_NAME}/${TEST_DATASET}"
mkdir -p "$LOG_DIR"

# 新增：定义日志文件路径（带日期时间，防止覆盖之前的实验记录）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/experiment_${TIMESTAMP}.log"

# 【核心修改】重定向所有输出到日志文件，同时在终端显示
# >(tee -a "$LOG_FILE") 表示将标准输出同步写入文件
# 2>&1 表示将标准错误也指向标准输出
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "开始实验任务: ${TIMESTAMP}"
echo "日志保存路径: ${LOG_FILE}"
echo "=================================================="

# ==============================
# 按目标域返回对应 SPLIT
# ==============================
get_split_by_dataset() {
  local dataset="$1"
  case "${dataset}" in
    "ProntoQA")         echo "dev" ;;
    "AR-LSAT")          echo "test" ;;
    "ProofWriter")      echo "test" ;;
    "FOLIO")            echo "dev" ;;
    "LogicalDeduction") echo "dev" ;;
    *)                  echo "test" ;;
  esac
}

# 数据集路径设置 (复用目标域的 train 作为校准，dev 作为测试)
TEST_DATA_DIR="data/${TEST_DATASET}"
TRAIN_DATA_DIR="data/${TRAIN_DATASET}"
test_split=$(get_split_by_dataset "$TEST_DATASET")
CALIB_FILE="${TRAIN_DATA_DIR}/train.json"
TEST_FILE="${TEST_DATA_DIR}/${test_split}.json"

RUN_CMD="python zero_shot_steering_test.py \
            --dataset ${TEST_DATASET} \
            --calib_file ${CALIB_FILE} \
            --test_file ${TEST_FILE} \
            --model ${MODEL_PATH} \
            --calib_samples ${CALIB_SAMPLES} \
            --eval_batch_size ${EVAL_BATCH_SIZE} \
            --intervention_mode ${MODE}"

if [ "$MAX_TEST_SAMPLES" != "None" ] && [ -n "$MAX_TEST_SAMPLES" ]; then
    RUN_CMD="$RUN_CMD --max_test_samples $MAX_TEST_SAMPLES"
    REPEAT_CMD="$RUN_CMD --max_test_samples $MAX_TEST_SAMPLES"
else
    RUN_CMD="$RUN_CMD" # 不传这个参数，让 Python 使用默认值
    REPEAT_CMD="$RUN_CMD"
fi

# 在拼接 RUN_CMD 时，只有当 MAX_LENGTH 不是 None 时才添加该参数
if [ "$MAX_LENGTH" != "None" ] && [ -n "$MAX_LENGTH" ]; then
    RUN_CMD="$RUN_CMD --max_length $MAX_LENGTH"
    REPEAT_CMD="$RUN_CMD --max_length $(( 2 * MAX_LENGTH ))"
else
    RUN_CMD="$RUN_CMD" # 不传这个参数，让 Python 使用默认值
    REPEAT_CMD="$RUN_CMD"
fi

# 先跑baseline的结果
# 跑一个 Baseline (alpha=0.0，不加干预) 用于对比
echo "--------------------------------------------------"
echo "Running Baseline (No Intervention, Alpha=0.0)"
echo "--------------------------------------------------"
BASELINE_CMD="$RUN_CMD --alpha 0.0 --output_file ${OUT_DIR}/results_${EVAL_BATCH_SIZE}_baseline_alpha_0.0.jsonl"
echo "RUN_CMD: ${BASELINE_CMD}"
echo "--------------------------------------------------"
${BASELINE_CMD}

# # laska修改，新增一个reverse的baseline
# echo "--------------------------------------------------"
# echo "Running Reverse Baseline (No Intervention, Alpha=0.0)"
# echo "--------------------------------------------------"
# if [ "$CONTEXT_REVERSE" = true ]; then
#   REVERSE_BASELINE_CMD="$RUN_CMD --alpha 0.0 --reverse_context --output_file ${OUT_DIR}/results_reverse_baseline_alpha_0.0.jsonl"
#   echo "RUN_CMD: ${REVERSE_BASELINE_CMD}"
#   echo "--------------------------------------------------"
#   ${REVERSE_BASELINE_CMD}
# fi

# laska修改，新增一个prompt repeat的baseline
echo "--------------------------------------------------"
echo "Running Prompt Repeat Baseline (No Intervention, Alpha=0.0)"
echo "--------------------------------------------------"
REPEAT_CMD="$REPEAT_CMD --alpha 0.0 --repeat --output_file ${OUT_DIR}/results_repeat_baseline_alpha_0.0.jsonl"
echo "RUN_CMD: ${REPEAT_CMD}"
echo "--------------------------------------------------"
${REPEAT_CMD}

# 0331新增，新增一个用pad进行prompt重复的baseline
echo "--------------------------------------------------"
echo "Running Padding Token Repeat Baseline (No Intervention, Alpha=0.0)"
echo "--------------------------------------------------"
PAD_REPEAT_CMD="$REPEAT_CMD --alpha 0.0 --pad_repeat --output_file ${OUT_DIR}/results_pad_repeat_baseline_alpha_0.0.jsonl"
echo "RUN_CMD: ${PAD_REPEAT_CMD}"
echo "--------------------------------------------------"
${PAD_REPEAT_CMD}

# 是否对单个样例实施定制化的干预
if [ "$INSTANCE_STEERING" = true ]; then
  RUN_CMD="$RUN_CMD --instance_steering"
fi
# 是否将context放在question和option之后
# if [ "$CONTEXT_REVERSE" = true ]; then
#   RUN_CMD="$RUN_CMD --reverse_context"
# fi

# 按照layers和alphas的组合进行网格搜索
# ================= 循环执行 =================
# 嵌套循环跑网格搜索 (层数 x 干预强度)
for layer in $LAYERS
do
    for alpha in $ALPHAS
    do  
        OUT_FILE="${OUT_DIR}/results_layer_${layer}_alpha_${alpha}.jsonl"
        if [ "$INSTANCE_STEERING" = true ]; then
          OUT_FILE="${OUT_DIR}/instance_results_layer_${layer}_alpha_${alpha}.jsonl"
        fi
        # if [ "$CONTEXT_REVERSE" = true ]; then
        #   OUT_FILE="${OUT_DIR}/results_reverse_layer_${layer}_alpha_${alpha}.jsonl"
        # fi
        SUB_RUN_CMD="$RUN_CMD --layer ${layer} --alpha ${alpha} --output_file ${OUT_FILE}"
        
        echo "--------------------------------------------------"
        echo "时间: $(date)"
        echo "Model: $MODEL_NAME | Dataset: $TEST_DATASET"
        echo "Steering Layer: $layer | Alpha: $alpha | Mode: $MODE"
        echo "Output Path: $OUT_DIR"
        echo "Output File: $OUT_FILE"
        echo "RUN_CMD: $SUB_RUN_CMD"
        echo "--------------------------------------------------"
        
        # 执行命令
        ${SUB_RUN_CMD}

        echo "完成: Layer $layer, Alpha $alpha"
    done
done

echo "=================================================="
echo "实验全部结束: $(date)"
echo "所有日志已保存至: ${LOG_FILE}"
echo "=================================================="

# # 实验结束后，自动生成 CSV 汇总和 PNG 趋势图
# echo "--------------------------------------------------"
# python collect_results.py --log_dir "./logs" --out_dir "./steering_report"
# echo "--------------------------------------------------"

# # ... 在生成汇总表之后 ...
# echo "Step 3: 正在生成性能提升热力图..."
# python analyze_improvement.py \
#     --csv "./steering_report/global_results.csv" \
#     --out "./steering_report/improvement_visuals"