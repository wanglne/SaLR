import logging
import math
import os
import json
from typing import Dict, List

import torch
import transformers
from torch.nn import functional as F

from peft import LoraConfig, TaskType
from safetensors.torch import load_file

from src.model import (
    SALR,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def build_lora_config(model_args):
    if not model_args.lora_init:
        raise NotImplementedError("Only lora_init=True is supported in this eval script.")

    task_type = TaskType.CAUSAL_LM
    model_name = model_args.model_name_or_path.lower()

    if any(name in model_name for name in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif "phi" in model_name:
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif "gpt2" in model_name:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

    return LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )


def load_model_and_tokenizer(model_args, training_args):
    lora_config = build_lora_config(model_args)
    model = SALR(model_args, training_args, lora_config)

    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"), map_location="cpu")

    model.load_state_dict(state_dict, strict=False)
    model.salr.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    model = model.to("cuda")
    model.to(torch.bfloat16)
    model.eval()

    return model, tokenizer


def load_safety_data(input_path: str) -> Dict[str, List[str]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input safety json must be a dict like {'advbench_1': [...], ...}")

    for k, v in data.items():
        if not isinstance(v, list):
            raise ValueError(f"Each value in safety json must be a list, but got {type(v)} for key {k}")

    return data


def generate_batch(model, tokenizer, questions: List[str], training_args, max_new_tokens=256):
    batch = tokenizer(
        questions,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=training_args.model_max_length,
    )

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
    else:
        bot_tensor = torch.tensor(
            [tokenizer.eos_token_id, model.bot_id],
            dtype=torch.long
        ).expand(batch["input_ids"].size(0), 2)

    batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
    batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model.salr(
            input_ids=batch["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=batch["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        for _ in range(training_args.inf_latent_iterations):
            outputs = model.salr(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

        if training_args.remove_eos:
            eot_emb = model.get_embd(model.salr, model.model_name)(
                torch.tensor([model.eot_id], dtype=torch.long, device="cuda")
            ).unsqueeze(0).to(device)
        else:
            eot_emb = model.get_embd(model.salr, model.model_name)(
                torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device="cuda")
            ).unsqueeze(0).to(device)

        eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)
        output = eot_emb

        batch_size = batch["input_ids"].size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
        pred_tokens = [[] for _ in range(batch_size)]

        for _ in range(max_new_tokens):
            out = model.salr(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                output_attentions=False,
                past_key_values=past_key_values
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.salr.config.vocab_size - 1]

            if training_args.greedy:
                next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
            else:
                temperature = 0.1
                top_k = 40
                top_p = 0.95

                logits = logits / temperature

                if top_k > 1:
                    top_k_values, _ = torch.topk(logits, top_k, dim=-1)
                    min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                    logits[logits < min_top_k_value] = -float("inf")

                if top_p < 1.0:
                    sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    if sorted_indices_to_remove.any():
                        sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                        sorted_indices_to_remove[:, 0] = False

                    for b in range(logits.size(0)):
                        logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

            for b in range(batch_size):
                if not finished[b]:
                    pred_tokens[b].append(next_token_ids[b].item())
                    if next_token_ids[b] == tokenizer.eos_token_id:
                        finished[b] = True

            if finished.all():
                break

            output = model.get_embd(model.salr, model.model_name)(next_token_ids).unsqueeze(1).to(device)

    decoded = [tokenizer.decode(x, skip_special_tokens=True) for x in pred_tokens]
    return decoded


def evaluation_safety(model_args, data_args, training_args):
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    logging.warning("Loading safety test data")
    safety_data = load_safety_data(data_args.input_path)

    results = {}
    total_cases = sum(len(v) for v in safety_data.values())
    print(f"Total safety test cases: {total_cases}")

    for bench_name, test_cases in safety_data.items():
        print(f"\nProcessing {bench_name} | num_cases={len(test_cases)}")
        results[bench_name] = []

        num_batches = math.ceil(len(test_cases) / data_args.batch_size)

        for i in range(num_batches):
            batch_questions = test_cases[i * data_args.batch_size: (i + 1) * data_args.batch_size]            
            decoded_outputs = generate_batch(
                model=model,
                tokenizer=tokenizer,
                questions=batch_questions,
                training_args=training_args,
                max_new_tokens=256,
            )

            for q, gen in zip(batch_questions, decoded_outputs):
                print("=" * 80)
                print(f"[{bench_name}]")
                print("TEST CASE:")
                print(q)
                print("GENERATION:")
                print(gen)
                print("=" * 80)

                results[bench_name].append({
                    "test_case": q,
                    "generation": gen,
                })

        if data_args.output_path:
            with open(data_args.output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Intermediate results saved to {data_args.output_path}")

    return results


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not hasattr(data_args, "input_path"):
        raise ValueError("Please add --input_path to DataArguments / command line.")
    if not hasattr(data_args, "output_path"):
        raise ValueError("Please add --output_path to DataArguments / command line.")

    results = evaluation_safety(model_args, data_args, training_args)

    with open(data_args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Final results saved to {data_args.output_path}")