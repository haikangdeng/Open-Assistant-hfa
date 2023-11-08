#!/bin/bash

datasets=("synthetic_oasst_export" "synthetic_anthropic_rlhf" "synthetic_hf_summary_pairs" "synthetic_shp" "synthetic_hellaswag" "synthetic_webgpt")

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python eval_by_gold_rm.py \
        --out_dir ../../lhf/reports/dpo \
        --dataset "$dataset" \
        --model ../../lhf/saved_models/dpo/llama2-7b \
        --model_dtype bf16 \
        --gold_rm ../../lhf/saved_models/gold-rm/llama2-7b \
        --gold_rm_dtype bf16 \
        --batch_size 4
done