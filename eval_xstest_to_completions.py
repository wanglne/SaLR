import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

import torch
import transformers
from torch.nn import functional as F

from peft import LoraConfig, TaskType
from safetensors.torch import load_file

from src.model import SALR, ModelArguments, TrainingArguments


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EvalArguments:
    input_path: str = field(
        default="HarmBench/test_cases/xstest.json",
        metadata={"help": "Input json path. Format: {'1': ['question'], '2': ['question'], ...}"},
    )
    output_path: str = field(
        default="HarmBench/completions/llama3.1_xstest_none.json",
        metadata={"help": "Output json path."},
    )
    batch_size: int = field(default=1, metadata={"help": "Batch size for generation."})
    max_new_tokens: int = field(default=256, metadata={"help": "Maximum new tokens to generate."})


def build_lora_config(model_args: ModelArguments) -> LoraConfig:
    if not model_args.lora_init:
        raise NotImplementedError("Only lora_init=True is supported in this eval script.")

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
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )


def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    device = get_device()
    lora_config = build_lora_config(model_args)
    model = SALR(model_args, training_args, lora_config)

    safetensors_path = os.path.join(model_args.ckpt_dir, "model.safetensors")
    bin_path = os.path.join(model_args.ckpt_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Cannot find model.safetensors or pytorch_model.bin in {model_args.ckpt_dir}")

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
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    model = model.to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    model.eval()

    return model, tokenizer


def load_xstest(input_path: str) -> Dict[str, List[str]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input json must be a dict, for example: {'1': ['question'], '2': ['question']}")

    normalized: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, str):
            normalized[key] = [value]
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            normalized[key] = value
        else:
            raise ValueError(f"Value of key {key!r} must be a string or a list of strings, got: {type(value)}")

    return normalized


def save_json(obj: Any, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    questions: List[str],
    training_args: TrainingArguments,
    max_new_tokens: int = 256,
) -> List[str]:
    device = get_device()

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
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(
            batch["input_ids"].size(0), 2
        )

    batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
    batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model.salr(
        input_ids=batch["input_ids"],
        use_cache=True,
        output_hidden_states=True,
        past_key_values=None,
        attention_mask=batch["attention_mask"],
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
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

    embed_layer = model.get_embd(model.salr, model.model_name)
    if training_args.remove_eos:
        eot_ids = torch.tensor([model.eot_id], dtype=torch.long, device=device)
    else:
        eot_ids = torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device=device)

    output_embeds = embed_layer(eot_ids).unsqueeze(0).expand(batch["input_ids"].size(0), -1, -1)

    batch_size = batch["input_ids"].size(0)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pred_tokens: List[List[int]] = [[] for _ in range(batch_size)]

    for _ in range(max_new_tokens):
        out = model.salr(
            inputs_embeds=output_embeds,
            output_hidden_states=False,
            attention_mask=None,
            use_cache=True,
            output_attentions=False,
            past_key_values=past_key_values,
        )
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, : model.salr.config.vocab_size - 1]

        if training_args.greedy:
            next_token_ids = torch.argmax(logits, dim=-1)
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
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                sorted_indices_to_remove[:, 0] = False

                for b in range(logits.size(0)):
                    logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

        for b in range(batch_size):
            token_id = int(next_token_ids[b].item())
            if not finished[b]:
                pred_tokens[b].append(token_id)
                if token_id == tokenizer.eos_token_id:
                    finished[b] = True

        if bool(finished.all().item()):
            break

        output_embeds = embed_layer(next_token_ids).unsqueeze(1).to(device)

    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_tokens]


def evaluate_xstest(model_args: ModelArguments, training_args: TrainingArguments, eval_args: EvalArguments):
    logging.warning("Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    logging.warning(f"Loading test cases from {eval_args.input_path}")
    input_data = load_xstest(eval_args.input_path)

    total_cases = sum(len(v) for v in input_data.values())
    print(f"Total test cases: {total_cases}")

    results: Dict[str, List[Dict[str, str]]] = {}

    for key, test_cases in input_data.items():
        print(f"\nProcessing key={key} | num_cases={len(test_cases)}")
        results[key] = []
        num_batches = math.ceil(len(test_cases) / eval_args.batch_size)

        for batch_idx in range(num_batches):
            start = batch_idx * eval_args.batch_size
            end = start + eval_args.batch_size
            batch_questions = test_cases[start:end]

            generations = generate_batch(
                model=model,
                tokenizer=tokenizer,
                questions=batch_questions,
                training_args=training_args,
                max_new_tokens=eval_args.max_new_tokens,
            )

            for question, generation in zip(batch_questions, generations):
                print("=" * 80)
                print(f"[{key}]")
                print("TEST CASE:")
                print(question)
                print("GENERATION:")
                print(generation)
                print("=" * 80)

                results[key].append(
                    {
                        "test_case": question,
                        "generation": generation,
                    }
                )

            # Save after every batch to avoid losing progress if the job stops midway.
            save_json(results, eval_args.output_path)
            print(f"Intermediate results saved to {eval_args.output_path}")

    save_json(results, eval_args.output_path)
    print(f"Final results saved to {eval_args.output_path}")
    return results


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, EvalArguments))
    model_args, training_args, eval_args = parser.parse_args_into_dataclasses()
    evaluate_xstest(model_args, training_args, eval_args)
