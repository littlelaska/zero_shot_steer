#!/bin/bash

GPU=0
export CUDA_VISIBLE_DEVICES="${GPU}"

# ================= 配置区域 =================
# 1. 模型绝对路径
MODEL_PATH="/home/hit/models/Qwen2.5-7B-Instruct"
MODEL_NAME=$(basename "$MODEL_PATH")
GTE_MODEL_PATH='/pcl_data/users/laska/models/gte-Qwen2-7B-instruct'

# 2. 实验参数 (Zero-shot Steering)
# 因为是零样本干预，我们不再需要区分 SOURCE，直接在特定数据集上验证
DATASET="FOLIO"  # 也可以换成 "FOLIO" 或 "ProofWriter"(LogicalDeduction FOLIO ProntoQA AR-LSAT ProofWriter)
LAYERS="6 10 12 16 20 24 26 30 34"        # 建议扫几个不同的层位，寻找“全局信息整合”最集中的层
LAYERS="6 10 14 18 22 26 30 34 38 42 44"        # 建议扫几个不同的层位，寻找“全局信息整合”最集中的层
# LAYERS="6 10 12 16 20 24 26"        # 建议扫几个不同的层位，寻找“全局信息整合”最集中的层
ALPHAS="0.5 1 1.5"        # 干预强度网格搜索
MODE="static"
CALIB_SAMPLES=1000           # 用于提取 Δh 的无标签样本数量
CONTEXT_REVERSE=true         # 用于将context放在question和option之后
EVAL_BATCH_SIZE=16          # 控制测试时的batch_size大小
INSTANCE_STEERING=false       # 控制干预向量是单个还是一致的
REPEAT_TIMES=2              # 控制用于抽取Δh的prompt重复次数，None表示不重复，使用原始prompt；整数表示重复多少次后进行抽取
# vLLM 无 steer baseline：仅对第一条 Baseline 命令生效；repeat/pad 仍走 HF（另起进程）
USE_VLLM=true
# VLLM_MAX_MODEL_LEN=8192      # 可选，传给 --vllm_max_model_len
# MAX_LENGTH=1024               # 控制输入的最大长度，对所有的batch padding到这个长度，避免由于不同padding带来的性能差异
GTE_STEER=false    # 是否使用gte进行steering
GTE_SAME_LAYER=true  # 是否抽取GTE模型和LLM干预的同一层，true表示抽取GTE模型的LAYERS中指定的层，false表示抽取GTE模型的最后一层

# MEAN_STEERING=true  # 是否使用平均steering（该值为true的时候delta h是直接取平均值而非差分向量）

# MAX_TEST_SAMPLES=10           # 控制测试时的样本数量，避免测试时间过长（你可以根据需要调整这个值，或者设置为 None 来使用全部样本）

# ================= 路径准备 =================
# 构造新的输出文件夹路径
OUT_DIR="./zero_shot_steering/${MODE}/${MODEL_NAME}/${DATASET}"
mkdir -p "$OUT_DIR"
# ================= 日志存放准备 =================
LOG_DIR="./logs/${MODEL_NAME}/${DATASET}"
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
    "commonsense")      echo "dev" ;;
    *)                  echo "test" ;;
  esac
}

# 数据集路径设置 (复用目标域的 train 作为校准，dev 作为测试)
BASE_DATA_DIR="data/${DATASET}"
test_split=$(get_split_by_dataset "$DATASET")
CALIB_FILE="${BASE_DATA_DIR}/train.json"
TEST_FILE="${BASE_DATA_DIR}/${test_split}.json"

RUN_CMD="python zero_shot_steering.py \
            --dataset ${DATASET} \
            --calib_file ${CALIB_FILE} \
            --test_file ${TEST_FILE} \
            --model ${MODEL_PATH} \
            --calib_samples ${CALIB_SAMPLES} \
            --eval_batch_size ${EVAL_BATCH_SIZE} \
            --repeat_times ${REPEAT_TIMES} \
            --intervention_mode ${MODE}"

# 在拼接 RUN_CMD 时，只有当 MAX_TEST_SAMPLE 不是 None 时才添加该参数
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

# 20260403 提前进行baseline的vllm参数控制
if [ "$USE_VLLM" = true ]; then
    BASELINE_CMD_ORI="$RUN_CMD --alpha 0.0 --use_vllm"
    REPEAT_CMD="$RUN_CMD --alpha 0.0 --use_vllm"
    if [ -n "$VLLM_MAX_MODEL_LEN" ]; then
      BASELINE_CMD_ORI="$BASELINE_CMD --vllm_max_model_len $VLLM_MAX_MODEL_LEN"
      REPEAT_CMD="$REPEAT_CMD --vllm_max_model_len $VLLM_MAX_MODEL_LEN"
    fi
else
    BASELINE_CMD_ORI="$RUN_CMD --alpha 0.0"
    REPEAT_CMD="$RUN_CMD --alpha 0.0"
fi

# 先跑baseline的结果
# 跑一个 Baseline (alpha=0.0，不加干预) 用于对比
echo "--------------------------------------------------"
echo "Running Baseline (No Intervention, Alpha=0.0)"
echo "--------------------------------------------------"
BASELINE_CMD="$BASELINE_CMD_ORI --output_file ${OUT_DIR}/results_${EVAL_BATCH_SIZE}_baseline_alpha_0.0.jsonl"
echo "RUN_CMD: ${BASELINE_CMD}"
echo "--------------------------------------------------"
${BASELINE_CMD}

<<<<<<< Updated upstream
# laska修改，新增一个reverse的baseline
echo "--------------------------------------------------"
echo "Running Reverse Baseline (No Intervention, Alpha=0.0)"
echo "--------------------------------------------------"
if [ "$CONTEXT_REVERSE" = true ]; then
  REVERSE_BASELINE_CMD="$BASELINE_CMD_ORI --reverse_context --output_file ${OUT_DIR}/results_reverse_baseline_alpha_0.0.jsonl"
  echo "RUN_CMD: ${REVERSE_BASELINE_CMD}"
  echo "--------------------------------------------------"
  ${REVERSE_BASELINE_CMD}
fi
=======
# # laska修改，新增一个reverse的baseline
# echo "--------------------------------------------------"
# echo "Running Reverse Baseline (No Intervention, Alpha=0.0)"
# echo "--------------------------------------------------"
# if [ "$CONTEXT_REVERSE" = true ]; then
#   REVERSE_BASELINE_CMD="$BASELINE_CMD_ORI --reverse_context --output_file ${OUT_DIR}/results_reverse_baseline_alpha_0.0.jsonl"
#   echo "RUN_CMD: ${REVERSE_BASELINE_CMD}"
#   echo "--------------------------------------------------"
#   ${REVERSE_BASELINE_CMD}
# fi

# # laska修改，新增一个prompt repeat的baseline
# echo "--------------------------------------------------"
# echo "Running Prompt Repeat Baseline (No Intervention, Alpha=0.0)"
# echo "--------------------------------------------------"
# REPEAT_CMD="$REPEAT_CMD --repeat --output_file ${OUT_DIR}/results_${EVAL_BATCH_SIZE}_repeat_baseline_alpha_0.0.jsonl"
# echo "RUN_CMD: ${REPEAT_CMD}"
# echo "--------------------------------------------------"
# ${REPEAT_CMD}
>>>>>>> Stashed changes

# laska修改，新增一个prompt repeat的baseline
echo "--------------------------------------------------"
echo "Running Prompt Repeat Baseline (No Intervention, Alpha=0.0)"
echo "--------------------------------------------------"
REPEAT_CMD="$REPEAT_CMD --repeat --output_file ${OUT_DIR}/results_${EVAL_BATCH_SIZE}_repeat_baseline_alpha_0.0.jsonl"
echo "RUN_CMD: ${REPEAT_CMD}"
echo "--------------------------------------------------"
${REPEAT_CMD}

# 0331新增，新增一个用pad进行prompt重复的baseline
echo "--------------------------------------------------"
echo "Running Padding Token Repeat Baseline (No Intervention, Alpha=0.0)"
echo "--------------------------------------------------"
PAD_REPEAT_CMD="$REPEAT_CMD --pad_repeat --output_file ${OUT_DIR}/results_${EVAL_BATCH_SIZE}_pad_repeat_baseline_alpha_0.0.jsonl"
echo "RUN_CMD: ${PAD_REPEAT_CMD}"
echo "--------------------------------------------------"
${PAD_REPEAT_CMD}

# # 是否对单个样例实施定制化的干预
# if [ "$INSTANCE_STEERING" = true ]; then
#   RUN_CMD="$RUN_CMD --instance_steering"
# fi
# # 是否将context放在question和option之后
# # if [ "$CONTEXT_REVERSE" = true ]; then
# #   RUN_CMD="$RUN_CMD --reverse_context"
# # fi

# 20260427 进行gte模型steer干预
if [ "$GTE_STEER" = true ]; then
    RUN_CMD="$RUN_CMD --steering_mode gte_steer --gte_model_path ${GTE_MODEL_PATH}"
fi
if [ "$GTE_STEER" = true ] && [ "$GTE_SAME_LAYER" = true ]; then
    RUN_CMD="$RUN_CMD --gte_same_layer"
fi

# 20260612 计算平均向量进行干预
if [ "$MEAN_STEERING" = true ]; then
    RUN_CMD="$RUN_CMD --mean_steering"
fi

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
        # gte steer的结果文件命名区分开
        if [ "$GTE_STEER" = true ]; then
          OUT_FILE="${OUT_DIR}/gte_results_layer_${layer}_alpha_${alpha}.jsonl"
        fi
        # if [ "$CONTEXT_REVERSE" = true ]; then
        #   OUT_FILE="${OUT_DIR}/results_reverse_layer_${layer}_alpha_${alpha}.jsonl"
        # fi
        SUB_RUN_CMD="$RUN_CMD --layer ${layer} --alpha ${alpha} --output_file ${OUT_FILE}"
        
        echo "--------------------------------------------------"
        echo "时间: $(date)"
        echo "Model: $MODEL_NAME | Dataset: $DATASET"
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