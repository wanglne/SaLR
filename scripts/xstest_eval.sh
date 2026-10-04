#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"
: "${CKPT_DIR:?Set CKPT_DIR to the trained SaLR checkpoint directory}"

python eval_xstest_to_completions.py \
    --model_name_or_path "$MODEL_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --output_dir "outputs/xstest" \
    --input_path "${INPUT_PATH:-data/xstest.json}" \
    --output_path "${OUTPUT_PATH:-outputs/xstest/completions.json}" \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --batch_size "${BATCH_SIZE:-1}" \
    --num_latent 6 \
    --use_prj True --prj_dim 2048 --prj_no_ln False --prj_dropout 0.0 \
    --inf_latent_iterations 6 --inf_num_iterations 1 \
    --remove_eos True --use_lora True \
    --greedy False \
    "$@"
