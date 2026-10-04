#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CKPT_DIR:?Set CKPT_DIR to the trained SaLR checkpoint directory}"
: "${DATA_PATH:?Set DATA_PATH to the Malicious-Educator parquet file}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to the judge model served by your endpoint}"
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  set -- --openai_base_url "$OPENAI_BASE_URL" "$@"
fi
python -m evaluation.malicious_educator \
  --model_name_or_path "${MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}" \
  --ckpt_dir "$CKPT_DIR" --output_dir outputs/hcot \
  --data_path "$DATA_PATH" --setting hcot --num_attempts 5 \
  --judge_model "$JUDGE_MODEL" --asr_threshold 3 \
  --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init \
  --num_latent 6 --use_prj True --prj_dim 2048 \
  --inf_latent_iterations 6 --inf_num_iterations 1 \
  --remove_eos True --use_lora True --greedy False \
  "$@"
