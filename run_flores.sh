GPU=7
export CUDA_VISIBLE_DEVICES="${GPU}"
MODEL_PATH="/home/hit/models/Qwen2.5-3B-Instruct"
DATA_TYPE="text"   # 可选 parquet 或 text，影响加载文件的路径
# 需要注意parquet形式和text形式的tgt_lang的命名不一样，parquet形式是zho_Hans，text形式是zho_Simpl
python eval_flores.py \
    --model_name ${MODEL_PATH} \
    --data_type ${DATA_TYPE} \
    --src_lang eng \
    --tgt_lang zho_simpl \
    --shots 0 \
    --prompt_strategy "repeated"