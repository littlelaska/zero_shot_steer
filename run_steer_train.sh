#!/bin/bash

GPU=0,1,2,7
export CUDA_VISIBLE_DEVICES="${GPU}"

# ================= 配置区域 =================
# 1. 模型绝对路径
MODEL_PATH="/data_a100/models/Qwen2.5-3B-Instruct"
MODEL_NAME=$(basename "$MODEL_PATH")

# 2. 实验参数 (Zero-shot Steering)
# 因为是零样本干预，我们不再需要区分 SOURCE，直接在特定数据集上验证
DATASET="AR-LSAT"  # 也可以换成 "FOLIO" 或 "ProofWriter"(LogicalDeduction FOLIO ProntoQA AR-LSAT ProofWriter)
LAYERS="12 16 20 24"        # 建议扫几个不同的层位，寻找“全局信息整合”最集中的层
LAYERS="6 10 12 16 20 24 26 30 34"        # 建议扫几个不同的层位，寻找“全局信息整合”最集中的层
ALPHAS="0.5 1.0 1.5"        # 干预强度网格搜索
MODE="static"
CALIB_SAMPLES=1000           # 用于提取 Δh 的无标签样本数量
CONTEXT_REVERSE=true         # 用于将context放在question和option之后
EVAL_BATCH_SIZE=1            # 控制测试时的batch_size大小
INSTANCE_STEERING=false       # 控制干预向量是单个还是一致的
MAX_LENGTH=1024              # 控制输入的最大长度，对所有的batch padding到这个长度，避免由于不同padding带来的性能差异

# ================= 路径准备 =================
# 构造新的输出文件夹路径
OUT_DIR="./zero_shot_steering/${MODE}/${MODEL_NAME}/${DATASET}"
mkdir -p "$OUT_DIR"

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
    "LogicalDeduction") echo "dev" ;;  # 你没说明，这里默认 test
    *)                  echo "test" ;;  # 默认值
  esac
}

# 数据集路径设置 (复用目标域的 train 作为校准，dev 作为测试)
BASE_DATA_DIR="data/${DATASET}"
test_split=$(get_split_by_dataset "$DATASET")
CALIB_FILE="${BASE_DATA_DIR}/train.json"
TEST_FILE="${BASE_DATA_DIR}/${test_split}.json"

RUN_CMD="python zero_shot_steering.py \
            --calib_file ${CALIB_FILE} \
            --test_file ${TEST_FILE} \
            --model ${MODEL_PATH} \
            --calib_samples ${CALIB_SAMPLES} \
            --eval_batch_size ${EVAL_BATCH_SIZE} \
            --intervention_mode ${MODE}"

# 先跑baseline的结果
# 跑一个 Baseline (alpha=0.0，不加干预) 用于对比
echo "--------------------------------------------------"
echo "Running Baseline (No Intervention, Alpha=0.0)"
echo "--------------------------------------------------"
BASELINE_CMD="$RUN_CMD --alpha 0.0 --output_file ${OUT_DIR}/results_${EVAL_BATCH_SIZE}_baseline_alpha_0.0.jsonl --max_length ${MAX_LENGTH}"
echo "RUN_CMD: ${BASELINE_CMD}"
echo "--------------------------------------------------"
${BASELINE_CMD}

# laska修改，新增一个reverse的baseline
echo "--------------------------------------------------"
echo "Running Reverse Baseline (No Intervention, Alpha=0.0)"
echo "--------------------------------------------------"
if [ "$CONTEXT_REVERSE" = true ]; then
  REVERSE_BASELINE_CMD="$RUN_CMD --alpha 0.0 --reverse_context --output_file ${OUT_DIR}/results_reverse_baseline_alpha_0.0.jsonl --max_length ${MAX_LENGTH}"
  echo "RUN_CMD: ${REVERSE_BASELINE_CMD}"
  echo "--------------------------------------------------"
  ${REVERSE_BASELINE_CMD}
fi

# laska修改，新增一个prompt repeat的baseline
echo "--------------------------------------------------"
echo "Running Prompt Repeat Baseline (No Intervention, Alpha=0.0)"
echo "--------------------------------------------------"
REPEAT_CMD="$RUN_CMD --alpha 0.0 --repeat --output_file ${OUT_DIR}/results_repeat_baseline_alpha_0.0.jsonl --max_length $(( 2 * MAX_LENGTH ))"
echo "RUN_CMD: ${REPEAT_CMD}"
echo "--------------------------------------------------"
${REPEAT_CMD}

# # 是否对单个样例实施定制化的干预
# if [ "$INSTANCE_STEERING" = true ]; then
#   RUN_CMD="$RUN_CMD --instance_steering"
# fi
# # 是否将context放在question和option之后
# # if [ "$CONTEXT_REVERSE" = true ]; then
# #   RUN_CMD="$RUN_CMD --reverse_context"
# # fi

# # 按照layers和alphas的组合进行网格搜索
# # ================= 循环执行 =================
# # 嵌套循环跑网格搜索 (层数 x 干预强度)
# for layer in $LAYERS
# do
#     for alpha in $ALPHAS
#     do  
#         OUT_FILE="${OUT_DIR}/results_layer_${layer}_alpha_${alpha}.jsonl"
#         if [ "$INSTANCE_STEERING" = true ]; then
#           OUT_FILE="${OUT_DIR}/instance_results_layer_${layer}_alpha_${alpha}.jsonl"
#         fi
#         # if [ "$CONTEXT_REVERSE" = true ]; then
#         #   OUT_FILE="${OUT_DIR}/results_reverse_layer_${layer}_alpha_${alpha}.jsonl"
#         # fi
#         SUB_RUN_CMD="$RUN_CMD --layer ${layer} --alpha ${alpha} --output_file ${OUT_FILE}"
#         echo "--------------------------------------------------"
#         echo "Model: $MODEL_NAME | Dataset: $DATASET"
#         echo "Steering Layer: $layer | Alpha: $alpha | Mode: $MODE"
#         echo "Output Path: $OUT_DIR"
#         echo "Output File: $OUT_FILE"
#         echo "RUN_CMD: $SUB_RUN_CMD"
#         echo "--------------------------------------------------"
        
#         ${SUB_RUN_CMD}

#         echo "Done: Layer $layer, Alpha $alpha"
#     done
# done
