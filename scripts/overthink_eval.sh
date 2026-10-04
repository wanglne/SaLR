#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CKPT_DIR:?Set CKPT_DIR to the trained SaLR checkpoint directory}"
python -m evaluation.overthink \
  --model_name_or_path "${MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}" \
  --ckpt_dir "$CKPT_DIR" --output_dir outputs/overthink \
  --mode both --dataset freshqa --data-dir data/overthink \
  --num-attacks 7 --max-new-tokens 32768 \
  --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init \
  --num_latent 6 --use_prj True --prj_dim 2048 \
  --inf_latent_iterations 6 --inf_num_iterations 1 \
  --remove_eos True --use_lora True --greedy False \
  "$@"
