#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SAVE_DIR="${SAVE_DIR:-outputs/train/llama1b}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
DATA_PATH="${DATA_PATH:-data/SaLR.json}"

mkdir -p "$SAVE_DIR"

python train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name SaLR_llama1b \
    --logging_dir "$SAVE_DIR/logs"\
    --logging_steps 10 \
    --model_name_or_path "$MODEL_PATH" \
    --data_name SaLR \
    --data_path "$DATA_PATH" \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --bf16 \
    --num_train_epochs 10 \
    --learning_rate 8e-4 \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --save_safetensors False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --do_train \
    --report_to none \
    --num_latent 6 \
    --safety_prefix_length 4 \
    --logging_strategy "steps" \
    --use_prj True \
    --prj_dim 2048 \
    --prj_dropout 0.0 \
    --distill_loss_div_std True \
    --exp_mode False \
    --exp_data_num 200 \
    --remove_eos True \
    --distill_loss_factor 20 \
    --print_ref_model_stats True \
    --max_token_num 2000 \
    "$@"
