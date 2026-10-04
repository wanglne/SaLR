import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from transformers import HfArgumentParser
from src.model import ModelArguments, TrainingArguments
from evaluation.generation import load_model_and_tokenizer, generate_batch_with_lengths

DATASET_FILE_MAP = {
    "freshqa": "freshQA_attack.csv",
    "freshQA": "freshQA_attack.csv",
    "freshQA_attack": "freshQA_attack.csv",
    "freshqa_attack": "freshQA_attack.csv",
    "musr_murder_mystery": "MuSR/murder_mystery.csv",
    "MuSR_murder_mystery": "MuSR/murder_mystery.csv",
    "murder_mystery": "MuSR/murder_mystery.csv",
    "murder_mystery_dataset": "MuSR/murder_mystery.csv",
    "musr_object_placement": "MuSR/object_placement.csv",
    "MuSR_object_placement": "MuSR/object_placement.csv",
    "object_placement": "MuSR/object_placement.csv",
    "object_placement_dataset": "MuSR/object_placement.csv",
}


def sort_attack_columns(columns: Iterable[str]) -> List[str]:
    """Sort Attack_Prompt_1 / Attack_Source_1 style columns by numeric suffix."""
    attack_cols = [
        col
        for col in columns
        if col.startswith("Attack_Prompt_") or col.startswith("Attack_Source_")
    ]

    def sort_key(col: str):
        suffix = col.split("_")[-1]
        try:
            return int(suffix)
        except ValueError:
            return suffix

    return sorted(attack_cols, key=sort_key)


def find_base_prompt_column(df: pd.DataFrame) -> str:
    """Find the base prompt column used by OverThink local CSV / HF splits."""
    candidates = ["Base_Prompt", "Base_Source", "Source", "source", "prompt", "Prompt"]
    for col in candidates:
        if col in df.columns:
            return col
    if "question" in df.columns and "source" in df.columns:
        return "__build_from_question_source__"
    if "Question" in df.columns and "Source" in df.columns:
        return "__build_from_question_source__"
    raise ValueError(
        "Cannot find base prompt column. Expected one of Base_Prompt/Base_Source/Source/source/prompt, or question+source."
    )


def build_original_prompt(question: str, source: str) -> str:
    """Same style as the original OverThink script."""
    return f'You are an helpful agent who will answer the following user question "{question}" Use the following retrieved context: "{source}".'


def infer_question(row: pd.Series, prompt: str) -> str:
    """Best-effort extraction of the user question for easier jsonl inspection."""
    for col in ["Question", "question", "query", "Query", "input", "Input"]:
        val = row.get(col, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    match = re.search(
        'following user question\\s+"(.*?)"\\s+Use the following retrieved context',
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return " ".join(match.group(1).split())
    match = re.search("Question\\s*:\\s*(.*?)(?:\\n|$)", prompt, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def get_reference_answer(row: pd.Series) -> str:
    for col in ["Answer", "answer", "answers", "gold", "label"]:
        val = row.get(col, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if val is not None and (not pd.isna(val)):
            return str(val)
    return ""


def safe_json_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(safe_json_dumps(record) + "\n")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skip broken jsonl line {line_no}: {exc}")


def extract_reasoning_text(text: str) -> str:
    """
    Extract visible reasoning text.

    For local Llama-3.2-1B-Instruct, there is usually no hidden reasoning metadata.
    If the model emits <think>...</think>, we count that span as reasoning tokens.
    Otherwise reasoning tokens are recorded as 0.
    """
    if not isinstance(text, str) or not text:
        return ""
    matches = re.findall("<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return "\n".join((m.strip() for m in matches if m.strip()))
    open_match = re.search("<think>(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if open_match:
        return open_match.group(1).strip()
    return ""


def count_text_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def format_avg(total: float, count: int) -> str:
    if count <= 0:
        return "n/a"
    return f"{total / count:.2f}"


def load_overthink_dataset(
    dataset_name: str,
    data_dir: str = "datasets",
    source: str = "local",
    hf_dataset_name: str = "akumar0927/OverThink",
) -> Tuple[pd.DataFrame, str, List[str], str]:
    """
    Load an OverThink dataset.

    Args:
        dataset_name:
            One of: freshqa, squad, murder_mystery, object_placement, team_allocation,
            or a direct CSV path.
        data_dir:
            Local directory containing OverThink generated CSV files.
            The official repo uses "dataset/", but this function also supports "datasets/".
        source:
            "local" or "hf".
        hf_dataset_name:
            HF dataset name if source == "hf".

    Returns:
        df, base_prompt_col, attack_cols, resolved_dataset_label
    """
    source = source.lower()
    if source == "hf":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Please install datasets first: pip install datasets"
            ) from exc
        split_aliases = {
            "freshqa": "freshQA_attack",
            "freshqa_attack": "freshQA_attack",
            "freshQA_attack": "freshQA_attack",
            "squad": "squad_attack",
            "squad_attack": "squad_attack",
            "murder_mystery": "MuSR_murder_mystery",
            "murder_mystery_dataset": "MuSR_murder_mystery",
            "object_placement": "MuSR_object_placement",
            "object_placement_dataset": "MuSR_object_placement",
            "team_allocation": "MuSR_team_allocation",
            "team_allocation_dataset": "MuSR_team_allocation",
        }
        split = split_aliases.get(dataset_name, dataset_name)
        ds = load_dataset(hf_dataset_name, split=split)
        df = ds.to_pandas()
        dataset_label = split
    elif source == "local":
        raw_path = Path(dataset_name)
        if raw_path.exists() and raw_path.suffix.lower() == ".csv":
            csv_path = raw_path
            dataset_label = raw_path.stem
        else:
            if dataset_name not in DATASET_FILE_MAP:
                available = ", ".join(sorted(DATASET_FILE_MAP.keys()))
                raise ValueError(
                    f"Unknown dataset_name={dataset_name}. Available aliases: {available}. You may also pass a direct CSV path."
                )
            rel_path = Path(DATASET_FILE_MAP[dataset_name])
            candidates = [
                Path(data_dir) / rel_path,
                Path("dataset") / rel_path,
                Path("datasets") / rel_path,
            ]
            csv_path = next((p for p in candidates if p.exists()), None)
            if csv_path is None:
                tried = "\n  - ".join((str(p) for p in candidates))
                raise FileNotFoundError(
                    f"Cannot find CSV for dataset={dataset_name}. Tried:\n  - {tried}\nPlease set --data-dir to the folder containing OverThink CSV files."
                )
            dataset_label = dataset_name
        df = pd.read_csv(csv_path)
    else:
        raise ValueError("--source must be either 'local' or 'hf'.")
    base_prompt_col = find_base_prompt_column(df)
    attack_cols = sort_attack_columns(df.columns)
    if not attack_cols:
        raise ValueError(
            "No attack columns found. Expected Attack_Prompt_1... or Attack_Source_1..."
        )
    return (df, base_prompt_col, attack_cols, dataset_label)


def run_local_model_test(
    model_args,
    training_args,
    df: pd.DataFrame,
    base_prompt_col: str,
    attack_cols: List[str],
    dataset_label: str,
    model_path: str,
    output_jsonl: str,
    start_index: int = 0,
    limit: Optional[int] = None,
    num_attacks: Optional[int] = None,
    max_new_tokens: int = 512,
    overwrite: bool = False,
) -> Path:
    """
    Load a local Transformers model, evaluate Base_Prompt and Attack_Prompt_i,
    and append one json object per prompt into outputs/*.jsonl.

    The jsonl contains:
        question, prompt, output, reasoning_tokens, output_tokens, latency, etc.
    """
    attack_cols = attack_cols[:num_attacks] if num_attacks is not None else attack_cols
    output_path = Path(output_jsonl)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Select a new --output-jsonl or pass --overwrite to replace it."
            )
        output_path.unlink()
    (model, tokenizer) = load_model_and_tokenizer(model_args, training_args)
    rows = list(df.iterrows())
    processed_rows = 0
    token_sums: Dict[str, float] = {"base_prompt": 0.0}
    token_counts: Dict[str, int] = {"base_prompt": 0}
    output_token_sums: Dict[str, float] = {"base_prompt": 0.0}
    output_token_counts: Dict[str, int] = {"base_prompt": 0}
    for i in range(len(attack_cols)):
        token_sums[f"attack_prompt_{i + 1}"] = 0.0
        token_counts[f"attack_prompt_{i + 1}"] = 0
        output_token_sums[f"attack_prompt_{i + 1}"] = 0.0
        output_token_counts[f"attack_prompt_{i + 1}"] = 0
    for row_index, row in tqdm(rows, desc="Evaluating OverThink prompts"):
        if row_index < start_index:
            continue
        if limit is not None and processed_rows >= limit:
            break
        if base_prompt_col == "__build_from_question_source__":
            q = row.get("question", row.get("Question", ""))
            src = row.get("source", row.get("Source", ""))
            base_prompt = build_original_prompt(str(q), str(src))
        else:
            base_prompt = row.get(base_prompt_col, "")
        if not isinstance(base_prompt, str) or not base_prompt.strip():
            print(f"[WARN] Empty base prompt at row {row_index}; skipped.")
            continue
        prompts: List[Tuple[str, str, str]] = [("base", "base_prompt", base_prompt)]
        for attack_i, col in enumerate(attack_cols, start=1):
            attack_prompt = row.get(col, "")
            if isinstance(attack_prompt, str) and attack_prompt.strip():
                prompts.append(
                    (f"attack_{attack_i}", f"attack_prompt_{attack_i}", attack_prompt)
                )
        for prompt_type, metric_key, prompt in prompts:
            question = infer_question(row, prompt)
            answer = get_reference_answer(row)
            start_time = time.time()
            (input_lens, output_lens, decoded_outputs) = generate_batch_with_lengths(
                model=model,
                tokenizer=tokenizer,
                questions=[prompt],
                training_args=training_args,
                max_new_tokens=max_new_tokens,
            )
            latency = time.time() - start_time
            input_len = input_lens[0]
            output_tokens = output_lens[0]
            output_text = decoded_outputs[0]
            reasoning_text = extract_reasoning_text(output_text)
            reasoning_tokens = count_text_tokens(tokenizer, reasoning_text)
            token_sums[metric_key] += reasoning_tokens
            token_counts[metric_key] += 1
            output_token_sums[metric_key] += output_tokens
            output_token_counts[metric_key] += 1
            record = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "dataset": dataset_label,
                "row_index": int(row_index),
                "prompt_type": prompt_type,
                "metric_key": metric_key,
                "model_path": model_path,
                "question": question,
                "reference_answer": answer,
                "prompt": prompt,
                "output": output_text,
                "reasoning_text": reasoning_text,
                "reasoning_tokens": reasoning_tokens,
                "output_tokens": output_tokens,
                "input_tokens": input_len,
                "latency_sec": latency,
                "max_new_tokens": max_new_tokens,
                "greedy": training_args.greedy,
                "latent_steps": training_args.inf_latent_iterations,
            }
            append_jsonl(output_path, record)
        processed_rows += 1
        avg_reasoning = ", ".join(
            (
                f"{k}={format_avg(token_sums[k], token_counts[k])}"
                for k in sorted(token_sums.keys())
            )
        )
        avg_output = ", ".join(
            (
                f"{k}={format_avg(output_token_sums[k], output_token_counts[k])}"
                for k in sorted(output_token_sums.keys())
            )
        )
        print(f"[RUNNING] Average reasoning tokens: {avg_reasoning}", flush=True)
        print(f"[RUNNING] Average output tokens: {avg_output}", flush=True)
    print(f"[INFO] Saved jsonl to: {output_path}")
    return output_path


def summarize_jsonl(jsonl_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load a jsonl file and print statistics.

    This function prints average reasoning tokens, average output tokens,
    average latency, and attack/base output-token ratios.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find jsonl file: {path}")
    groups: Dict[str, Dict[str, float]] = {}
    for rec in iter_jsonl(path):
        key = rec.get("metric_key") or rec.get("prompt_type") or "unknown"
        if key not in groups:
            groups[key] = {
                "count": 0,
                "reasoning_tokens": 0.0,
                "output_tokens": 0.0,
                "input_tokens": 0.0,
                "latency_sec": 0.0,
            }
        groups[key]["count"] += 1
        groups[key]["reasoning_tokens"] += float(rec.get("reasoning_tokens") or 0)
        groups[key]["output_tokens"] += float(rec.get("output_tokens") or 0)
        groups[key]["input_tokens"] += float(rec.get("input_tokens") or 0)
        groups[key]["latency_sec"] += float(rec.get("latency_sec") or 0)
    if not groups:
        print("[WARN] Empty jsonl; no statistics to print.")
        return groups
    print("\n========== Summary ==========")
    print(f"File: {path}")
    avg_reasoning_line = ", ".join(
        (
            f"{key}={format_avg(groups[key]['reasoning_tokens'], int(groups[key]['count']))}"
            for key in sorted(groups.keys())
        )
    )
    print(f"Average reasoning tokens: {avg_reasoning_line}")
    print("\nDetailed statistics:")
    print(
        f"{'prompt':<18} {'n':>6} {'avg_reason':>12} {'avg_output':>12} {'avg_input':>12} {'avg_latency':>12} {'output/base':>12}"
    )
    base_avg_output = None
    if "base_prompt" in groups and groups["base_prompt"]["count"] > 0:
        base_avg_output = (
            groups["base_prompt"]["output_tokens"] / groups["base_prompt"]["count"]
        )
    for key in sorted(groups.keys()):
        n = int(groups[key]["count"])
        avg_reason = groups[key]["reasoning_tokens"] / n if n else 0.0
        avg_output = groups[key]["output_tokens"] / n if n else 0.0
        avg_input = groups[key]["input_tokens"] / n if n else 0.0
        avg_latency = groups[key]["latency_sec"] / n if n else 0.0
        if base_avg_output and key != "base_prompt":
            ratio = avg_output / base_avg_output
            ratio_str = f"{ratio:.2f}x"
        elif key == "base_prompt":
            ratio_str = "1.00x"
        else:
            ratio_str = "n/a"
        print(
            f"{key:<18} {n:>6d} {avg_reason:>12.2f} {avg_output:>12.2f} {avg_input:>12.2f} {avg_latency:>12.2f} {ratio_str:>12}"
        )
    return groups


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate local Llama-3.2-1B-Instruct on OverThink attack datasets."
    )
    parser.add_argument(
        "--mode",
        choices=["run", "summarize", "both"],
        default="both",
        help="run: test model; summarize: summarize existing jsonl; both: run then summarize",
    )
    parser.add_argument(
        "--source",
        choices=["local", "hf"],
        default="local",
        help="Load OverThink dataset from local CSV files or Hugging Face.",
    )
    parser.add_argument(
        "--dataset",
        default="freshqa",
        help="Dataset alias or direct CSV path. Examples: squad, freshqa, murder_mystery, object_placement, team_allocation.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/overthink",
        help="Local data dir containing OverThink CSVs. Official repo may use dataset/.",
    )
    parser.add_argument(
        "--hf-dataset-name",
        default="akumar0927/OverThink",
        help="HF dataset name when --source hf.",
    )
    parser.add_argument(
        "--num-attacks",
        type=int,
        default=None,
        help="How many Attack_Prompt_i / Attack_Source_i columns to evaluate.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Output jsonl path. Default: outputs/overthink_<dataset>_<model_name>.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing JSONL. Otherwise an existing output is rejected to avoid mixing runs.",
    )
    return parser


def default_output_path(dataset_name: str, model_path: str) -> str:
    model_name = Path(model_path.rstrip("/")).name or "model"
    safe_dataset = re.sub("[^A-Za-z0-9_.-]+", "_", dataset_name)
    safe_model = re.sub("[^A-Za-z0-9_.-]+", "_", model_name)
    return str(Path("outputs") / f"overthink_{safe_dataset}_{safe_model}.jsonl")


def parse_args():
    argv = sys.argv[1:]
    normal_parser = build_arg_parser()
    (normal_args, normal_unknown) = normal_parser.parse_known_args(argv)
    hf_parser = HfArgumentParser((ModelArguments, TrainingArguments))
    parsed = hf_parser.parse_args_into_dataclasses(
        args=normal_unknown, return_remaining_strings=True, look_for_args_file=False
    )
    (model_args, training_args, hf_unknown) = parsed
    if hf_unknown:
        normal_parser.error("Unrecognized arguments: " + " ".join(hf_unknown))
    normal_args.model_name_or_path = model_args.model_name_or_path
    return (normal_args, model_args, training_args)


def main() -> None:
    (args, model_args, training_args) = parse_args()
    output_jsonl = args.output_jsonl or str(
        Path(training_args.output_dir)
        / Path(default_output_path(args.dataset, args.model_name_or_path)).name
    )
    if args.mode in ["run", "both"]:
        (df, base_col, attack_cols, dataset_label) = load_overthink_dataset(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            source=args.source,
            hf_dataset_name=args.hf_dataset_name,
        )
        print(f"[INFO] Loaded dataset: {dataset_label}")
        print(f"[INFO] Number of rows: {len(df)}")
        print(f"[INFO] Base prompt column: {base_col}")
        print(f"[INFO] Attack columns: {attack_cols}")
        run_local_model_test(
            model_args=model_args,
            training_args=training_args,
            df=df,
            base_prompt_col=base_col,
            attack_cols=attack_cols,
            dataset_label=dataset_label,
            model_path=args.model_name_or_path,
            output_jsonl=output_jsonl,
            start_index=args.start_index,
            limit=args.limit,
            num_attacks=args.num_attacks,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
        )
    if args.mode in ["summarize", "both"]:
        summarize_jsonl(output_jsonl)


if __name__ == "__main__":
    main()
