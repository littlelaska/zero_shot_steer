CUDA_VISIBLE_DEVICES=0,1,2,3

python collect_trajectories_vllm.py \
    --input_file mathoai.jsonl \
    --model "/userhome/huggingface/Qwen2.5-32B-Instruct/" \
    --output_file mathoai_trajectories.jsonl \
    --batch_size 512