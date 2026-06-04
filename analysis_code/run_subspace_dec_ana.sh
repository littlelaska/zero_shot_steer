python subspace_decomposition_analysis.py \
    --model "/pcl_data/users/laska/models/Qwen2.5-7B-Instruct" \
    --layer 16 \
    --samples 150 \
    --batch_size 8 \
    --core_components 5 \
    --datasets \
    LogicalDeduction:../data/LogicalDeduction/train.json \
    FOLIO:../data/FOLIO/train.json \
    ProofWriter:../data/ProofWriter/train.json \
    AR-LSAT:../data/AR-LSAT/train.json \
    ProntoQA:../data/ProntoQA/dev.json \
    gsm8k:../data/gsm8k/train.json \
    commonsenseQA:../data/commonsense/train.json