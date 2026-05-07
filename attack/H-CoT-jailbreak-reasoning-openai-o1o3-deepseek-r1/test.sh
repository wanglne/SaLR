export OPENAI_API_KEY="YOURAPIKEY"
export OPENAI_API_URL="YOUEAPIURL"

python eval_malicious_educator.py \
  --data_path data/train-00000-of-00001.parquet \
  --model_type latent \
  --model_name_or_path model/Llama-3.2-3B-Instruct \
  --output_dir outputs \
  --output_name malicious_educator_Llama-3.2-3B-Instruct_request_5attempts_gpt-5.1.jsonl\
  --setting request \
  --num_attempts 5 \
  --judge_model gpt-5.1 \
  --openai_api_key "$OPENAI_API_KEY" \
  --openai_base_url "$OPENAI_API_URL" \
  --openai_timeout 120 \
  --openai_max_retries 3 \
  --asr_threshold 3 \
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
  2>&1 | tee logs/eval_request.log