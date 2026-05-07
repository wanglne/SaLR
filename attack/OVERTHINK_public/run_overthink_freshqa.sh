# Project paths.
SCRIPT_PATH="overthink_local_eval.py"
DATA_DIR="dataset"
MODEL_PATH="model/Llama-3.2-3B-Instruct"

# Generation / evaluation settings.
NUM_ATTACKS=7
MAX_NEW_TOKENS=32768
DTYPE="bf16"

echo "[INFO] Running OverThink evaluation on FreshQA..."

python "${SCRIPT_PATH}" \
  --mode both \
  --dataset freshqa \
  --data-dir "${DATA_DIR}" \
  --model_type "latent" \
  --model_name_or_path "${MODEL_PATH}" \
  --num-attacks "${NUM_ATTACKS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --dtype "${DTYPE}" \
  --output-jsonl outputs/overthink_freshqa_Llama-3.2-3B-Instruct.jsonl \
  --ckpt_dir salr/SaLR_llama3b/SaLR_llama3b/Llama-3.2-3B-Instruct/ep_10/lr_0.0004/ \
  --batch_size 128 \
  --model_max_length 512 \
  --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init \
  --num_latent 6 \
  --use_prj True \
  --prj_dim 2048 \
  --prj_no_ln False \
  --prj_dropout 0.0 \
  --inf_latent_iterations 6 \
  --inf_num_iterations 1 \
  --remove_eos True \
  --use_lora True 

echo "[INFO] FreshQA finished."
