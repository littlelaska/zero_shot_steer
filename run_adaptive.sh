# 设置基础路径
BASE_PATH="/userhome/huggingface"    
# echo "Running Qwen2.5-3B..."
# CUDA_VISIBLE_DEVICES=0,1,2,3 python adaptive_projection_selector.py \
#     --model_name "3B" \
#     --model_path "$BASE_PATH/Qwen2.5-3B-Instruct" \
#     --tp_size 4 \
#     --source_path "/code/data/ProofWriter/ProofWriter_train_cot.json" \
#     --target_dev_path "/code/data/LogicalDeduction/LogicalDeduction_dev_cot.json" \
#     --target_pseudo_path "/code/data/LogicalDeduction/LogicalDeduction_train_cot.json" \
#     --k_shot 6 \
#     --batch_size 8
    
# echo "Running Qwen2.5-7B..."
# CUDA_VISIBLE_DEVICES=0,1,2,3 python adaptive_projection_selector.py \
#     --model_name "7B" \
#     --model_path "$BASE_PATH/Qwen2.5-7B-Instruct" \
#     --tp_size 4 \
#     --source_path "/code/data/ProofWriter/ProofWriter_train_cot.json" \
#     --target_dev_path "/code/data/LogicalDeduction/LogicalDeduction_dev_cot.json" \
#     --target_pseudo_path "/code/data/LogicalDeduction/LogicalDeduction_train_cot.json" \
#     --k_shot 6 \
#     --batch_size 8
    
echo "Running Qwen2.5-14B..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python adaptive_projection_selector.py \
    --model_name "14B" \
    --model_path "$BASE_PATH/Qwen2.5-14B-Instruct" \
    --tp_size 4 \
    --source_path "/code/data/ProofWriter/ProofWriter_train_cot.json" \
    --target_dev_path "/code/data/LogicalDeduction/LogicalDeduction_dev_cot.json" \
    --target_pseudo_path "/code/data/LogicalDeduction/LogicalDeduction_train_cot.json" \
    --k_shot 4 \
    --batch_size 8
    
echo "Running Qwen2.5-32B..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python adaptive_projection_selector.py \
    --model_name "32B" \
    --model_path "$BASE_PATH/Qwen2.5-32B-Instruct" \
    --tp_size 4 \
    --source_path "/code/data/ProofWriter/ProofWriter_train_cot.json" \
    --target_dev_path "/code/data/LogicalDeduction/LogicalDeduction_dev_cot.json" \
    --target_pseudo_path "/code/data/LogicalDeduction/LogicalDeduction_train_cot.json" \
    --k_shot 4 \
    --batch_size 8