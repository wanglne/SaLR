"""Prepare local training data or convert an upstream XSTest CSV into prompt JSON."""

import argparse
import csv
import json
from pathlib import Path


def read_records(path):
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = (
            [json.loads(line) for line in f if line.strip()]
            if path.suffix == ".jsonl"
            else json.load(f)
        )
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array or JSONL records: {path}")
    return data


def prepare_training(gsm8k, safechain):
    result = []
    for source, path in [("gsm8k", gsm8k), ("safechain", safechain)]:
        for idx, row in enumerate(read_records(path)):
            if source == "gsm8k" and "cot" not in row:
                parts = str(row["answer"]).rsplit("####", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"GSM8K row {idx}: expected '####' or an explicit cot field"
                    )
                cot, answer = parts
            else:
                cot, answer = row.get("cot", ""), row.get("answer", "")
            fields = dict(
                question=str(row.get("question", "")).strip(),
                cot=str(cot).strip(),
                answer=str(answer).strip(),
            )
            if not fields["question"] or not fields["answer"]:
                raise ValueError(
                    f"{source} row {idx}: question and answer must be nonempty"
                )
            if source == "safechain" and (row.get("error") or not fields["cot"]):
                raise ValueError(
                    f"safechain row {idx}: missing generated chain or an API error. "
                    "Retry failed construction records before merging; no records were discarded."
                )
            result.append(dict(source=source, **fields))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser(
        "train",
        help="Merge exported GSM8K and compressed SafeChain records without resampling",
    )
    train.add_argument("--gsm8k", required=True)
    train.add_argument("--safechain", required=True)
    train.add_argument("--output", default="data/SaLR.json")
    download = sub.add_parser(
        "download", help="Export official training sources into local JSON"
    )
    download.add_argument(
        "--dataset", choices=["safechain", "gsm8k-aug"], required=True
    )
    download.add_argument(
        "--revision",
        default="main",
        help="HF dataset revision; select an exact commit for reproducibility",
    )
    download.add_argument("--output", required=True)
    xstest = sub.add_parser("xstest")
    xstest.add_argument("--input", required=True, help="Upstream xstest_prompts.csv")
    xstest.add_argument("--output", default="data/xstest.json")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(
            f"Output exists; select a new --output path: {args.output}"
        )
    if args.command == "download":
        from datasets import load_dataset

        repo = "UWNSL/SafeChain" if args.dataset == "safechain" else "zen-E/GSM8k-Aug"
        data = load_dataset(repo, split="train", revision=args.revision)
        result = []
        for index, row in enumerate(data):
            if args.dataset == "gsm8k-aug":
                result.append({key: row[key] for key in ["question", "cot", "answer"]})
            else:
                parts = row["response"].split("</think>", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"SafeChain row {index}: expected a </think> separator in response"
                    )
                reasoning, answer = parts
                result.append(
                    dict(
                        question=row["instruction"],
                        label=row["label"],
                        reasoning=reasoning.removeprefix("<think>").strip(),
                        answer=answer.strip(),
                    )
                )
    elif args.command == "train":
        result = prepare_training(args.gsm8k, args.safechain)
    else:
        with open(args.input, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        result = {row["id"]: [row["prompt"]] for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate XSTest IDs")
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(f"Output exists; select a new --output path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(result)} records to {path}")


if __name__ == "__main__":
    main()
