CUDA_VISIBLE_DEVICES=0,1,2,3

python collect_cot_reasoning.py \
    --input_file ../mathoai.jsonl \
    --model "/userhome/huggingface/Qwen2.5-32B-Instruct/" \
    --output_file ../mathoai_cot.jsonl \
    --batch_size 512