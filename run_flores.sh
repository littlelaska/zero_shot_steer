GPU=7
export CUDA_VISIBLE_DEVICES="${GPU}"
MODEL_PATH="/data_a100/models/Qwen2.5-3B-Instruct"

python eval_flores.py \
    --model_name ${MODEL_PATH} \
    --src_lang eng_Latn \
    --tgt_lang zho_Hans