#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Evaluate SaLR on numerical reasoning and CommonsenseQA benchmarks."""

import os
import re
import transformers
from datasets import load_dataset, concatenate_datasets
from src.model import ModelArguments, DataArguments, TrainingArguments
from evaluation.generation import load_model_and_tokenizer, generate_batch, save_json


def extract_answer(sentence, data_name):
    if data_name == "commonsense":
        tail = sentence.split("The answer is:")[-1].strip()
        match = re.match(r"\(?([A-E])\)?(?:\b|$)", tail)
        return match.group(1) if match else None
    numbers = re.findall(r"-?\d+\.?\d*", sentence.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def load_reasoning_data(data_name):
    if data_name == "gsm-hard":
        return load_dataset("juyoung-trl/gsm-hard")["train"], "instruction", "response"
    if data_name == "multi-arith":
        return load_dataset("ChilleD/MultiArith")["test"], "question", "final_ans"
    if data_name == "svamp":
        data = load_dataset("ChilleD/SVAMP")
        return (
            concatenate_datasets([data["train"], data["test"]]),
            "question_concat",
            "Answer",
        )
    if data_name == "gsm8k":
        return load_dataset("gsm8k", "main")["test"], "question", "answer"
    if data_name == "commonsense":
        return (
            load_dataset("zen-E/CommonsenseQA-GPT4omini")["validation"],
            "question",
            "answer",
        )
    raise ValueError(f"Unsupported benchmark: {data_name}")


def evaluation(model_args, data_args, training_args):
    if data_args.batch_size < 1 or training_args.inf_num_iterations < 1:
        raise ValueError("batch_size and inf_num_iterations must be positive.")
    dataset, question_key, answer_key = load_reasoning_data(data_args.data_name)
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)
    runs = []
    for run in range(training_args.inf_num_iterations):
        records = []
        for start in range(0, len(dataset), data_args.batch_size):
            examples = [
                dataset[i]
                for i in range(start, min(start + data_args.batch_size, len(dataset)))
            ]
            questions = [
                str(e[question_key]).strip().replace("  ", " ") for e in examples
            ]
            generations = generate_batch(model, tokenizer, questions, training_args)
            for example, question, generation in zip(examples, questions, generations):
                raw_gold = str(example[answer_key]).split("####")[-1].strip()
                gold = (
                    raw_gold
                    if data_args.data_name == "commonsense"
                    else float(raw_gold.replace(",", ""))
                )
                pred = extract_answer(generation, data_args.data_name)
                records.append(
                    dict(
                        question=question,
                        generation=generation,
                        prediction=pred,
                        answer=gold,
                        correct=pred == gold,
                    )
                )
        accuracy = sum(row["correct"] for row in records) / len(records)
        print(f"{data_args.data_name} run {run + 1}: accuracy={100 * accuracy:.2f}%")
        runs.append(dict(accuracy=accuracy, records=records))
    result = dict(
        benchmark=data_args.data_name,
        mean_accuracy=sum(x["accuracy"] for x in runs) / len(runs),
        runs=runs,
    )
    path = data_args.output_path or os.path.join(
        training_args.output_dir, data_args.data_name + ".json"
    )
    save_json(result, path)
    print(f"Saved results to {path}")
    return result


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    evaluation(*parser.parse_args_into_dataclasses())
