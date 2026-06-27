
GPU=1,2,3,4
export CUDA_VISIBLE_DEVICES="${GPU}"


python zero_shot_steering_test.py \
    --model "/home/hit/models/Qwen2.5-3B-Instruct" \
    --dataset "flores" \
    --data_type "text" \
    --flores_dir "data/flores101_dataset" \
    --src_lang "jpn" \
    --tgt_lang "kor" \
    --layer 10 \
    --alpha 0 \
    --eval_batch_size 16 \
    --repeat \
    --calib_samples 500 \
    --max_test_samples 1012 \
    --output_file "eval_results/flores_steer_output.jsonl"