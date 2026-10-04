# SaLR

This is the official repository for **Safety-Aware Latent Space Reasoning in Large Language Models**, accepted at **NeurIPS 2026** 🎉.

![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-68488B)
[![arXiv: coming soon](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b)](#paper)
[![Proceedings: coming soon](https://img.shields.io/badge/Proceedings-coming%20soon-00629B)](#paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Latent reasoning compresses textual chain-of-thought into continuous hidden states, but can weaken safety alignment. **SaLR** introduces compact safety supervision into latent reasoning through teacher-student self-distillation. It combines four-block safe-chains with safety response prefix distillation and mathematical answer-onset distillation to improve jailbreak robustness while preserving reasoning utility and efficiency.

<p align="center">
  <img src="figs/intro.png" width="100%" alt="SaLR motivation: jailbreak vulnerability, safety separation, compact safe-chains, and safety versus utility">
</p>

## News

- 🎉 **Safety-Aware Latent Space Reasoning in Large Language Models** has been accepted at **NeurIPS 2026**.

## Method

Safety reasoning from SafeChain is compressed by Qwen3-32B into four blocks:

```text
<<request>> <<risk>> <<action>> <<followup>>
```

The action is `refuse`, `partial`, or `allow`; each free-text block is prompted to contain at most five words. During training, the teacher observes the explicit chain and response, while the student reasons with continuous latent states and predicts the response. Safety examples use response-prefix hidden state distillation with **K=4**; mathematical examples use single-position distillation at the answer onset. Both branches share the language model, following the CODI self-distillation framework.

<p align="center">
  <img src="figs/method.png" width="100%" alt="SaLR training architecture with safety prefix and mathematical single-position distillation">
</p>

## Installation

Use Python 3.10, preferably the original experiment version **3.10.14**, and a CUDA GPU for training and model evaluation.

```bash
git clone https://github.com/wanglne/SaLR.git
cd SaLR
conda create -n salr python=3.10.14 -y
conda activate salr
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA build of PyTorch appropriate for your GPU environment. The original experiment versions are PyTorch 2.7.1, Transformers 4.52.4, PEFT 0.15.2, and Datasets 3.6.0. Access to the Meta LLaMA model repositories must be granted to your Hugging Face account; authenticate with `huggingface-cli login` before downloading gated resources.

## Training data

The main paper setting mixes **385,620 GSM8K-Aug** mathematical examples with **40,000 processed SafeChain** examples. Training data and model checkpoints are prepared locally rather than bundled in this repository.

### 1. Export source data

The data preparation utility exports [GSM8K-Aug](https://huggingface.co/datasets/zen-E/GSM8k-Aug) and [SafeChain](https://huggingface.co/datasets/UWNSL/SafeChain). SafeChain's `instruction`, `label`, and `response` fields are normalized into `question`, `label`, `reasoning`, and `answer`, splitting reasoning and the answer at `</think>`. It fails if that delimiter is missing.

```bash
python scripts/prepare_data.py download \
  --dataset gsm8k-aug --output data/gsm8k_aug.json
python scripts/prepare_data.py download \
  --dataset safechain --output data/safechain_raw.json
```

Use `--revision <dataset-commit>` to select a particular upstream revision. If you already have the normalized source records used in your experiment, use those files directly.

### 2. Construct compact safe-chains

Serve **Qwen3-32B** through an OpenAI-compatible endpoint. This repository includes the compression client; the model server is configured separately. The defaults retain the original generation settings: temperature 0.2, top-p 0.8, top-k 20, maximum 256 new tokens, and `enable_thinking=False`.

```bash
python safe-chain-generate/generate.py \
  --input data/safechain_raw.json \
  --output data/safechain_compressed.json \
  --base-url http://localhost:8001/v1 \
  --model Qwen3-32B
```

Set `OPENAI_API_KEY` if the endpoint requires authentication. The model name must match the name served by your endpoint. The script saves intermediate results and resumes from an existing output after checking the input records. It records validation flags and errors for inspection. Add `--retry-failed` to retry saved API errors or empty chains. The merge step stops on these failures rather than silently training on empty safety chains; no data is automatically discarded.

### 3. Merge training records

```bash
python scripts/prepare_data.py train \
  --gsm8k data/gsm8k_aug.json \
  --safechain data/safechain_compressed.json \
  --output data/SaLR.json
```

Each JSON or JSONL training record has this schema:

```json
{
  "source": "safechain",
  "question": "How can I secure my email account?",
  "cot": "<<account security request>> <<low-risk benign intent>> <<allow>> <<offer defensive security help>>",
  "answer": "Use a strong password and enable multi-factor authentication."
}
```

Use `source: "gsm8k"` for mathematical examples, with symbolic `<<step>>` chains in `cot` and a numerical `answer`. Training retains the original token-length filtering controlled by `--max_token_num`.

## Training

| Backbone | Default model source | Learning rate | Epochs | Batch / accumulation |
| --- | --- | --- | --- | --- |
| LLaMA 3.2-1B-Instruct | `meta-llama/Llama-3.2-1B-Instruct` | 8e-4 | 10 | 16 / 4 |
| LLaMA 3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | 4e-4 | 10 | 8 / 8 |
| LLaMA 3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 1e-4 | 6 | 1 / 8 |

The scripts use a safety prefix length of 4 (`--safety_prefix_length`), 6 latent states, LoRA rank 128 / alpha 32, a 2048-dimensional projection module, and distillation weight 20. Teacher and student response loss weights are both 1.

```bash
bash scripts/train_llama1b_SaLR.sh
bash scripts/train_llama3b_SaLR.sh
bash scripts/train_llama8b_SaLR.sh
```

Select local model/data paths or adjust batch size through environment variables and extra arguments:

```bash
MODEL_PATH=/path/to/Llama-3.2-3B-Instruct \
DATA_PATH=/path/to/SaLR.json \
SAVE_DIR=outputs/train/llama3b \
bash scripts/train_llama3b_SaLR.sh --per_device_train_batch_size 4
```

Final checkpoints follow the original directory layout. For the default 3B script:

```text
outputs/train/llama3b/SaLR_llama3b/Llama-3.2-3B-Instruct/ep_10/lr_0.0004/
```

Evaluation expects a SaLR checkpoint containing `pytorch_model.bin` or `model.safetensors`. Select the same backbone, LoRA, projection, and latent settings used for training.

## Evaluation

### Mathematical reasoning

The reasoning evaluator supports `gsm8k`, `gsm-hard`, `multi-arith`, `svamp`, and `commonsense`. `gsm8k` uses the original GSM8K test set; GSM-Hard uses its `train` split, and SVAMP uses the combined train and test splits, retaining the old evaluator's choices.

```bash
export CKPT_DIR=outputs/train/llama3b/SaLR_llama3b/Llama-3.2-3B-Instruct/ep_10/lr_0.0004
DATA_NAME=gsm8k bash scripts/test_llama.sh
DATA_NAME=gsm-hard bash scripts/test_llama.sh
DATA_NAME=multi-arith bash scripts/test_llama.sh
DATA_NAME=svamp bash scripts/test_llama.sh
```

Results include predictions, reference answers, per-example correctness, and accuracy in `outputs/reasoning/<dataset>.json`. Extra arguments such as `--inf_num_iterations 5` can be appended to the shell command. Defaults use greedy decoding, matching the original reasoning script.

### Jailbreak robustness

Use prepared attack prompts for HarmBench, AdvBench, JailbreakBench, or MaliciousInstruct. GCG/PAIR attack generation is performed with the upstream attack implementations. This repository generates SaLR responses to those prompts and scores text-only behaviors with the HarmBench classifier.

Input JSON is keyed by behavior ID and stores one or more attack prompts per behavior:

```json
{"behavior_id": ["prepared prompt 1", "prepared prompt 2"]}
```

```bash
INPUT_PATH=data/attacks/advbench_pair.json \
OUTPUT_PATH=outputs/safety/advbench_pair.json \
bash scripts/safe_llama_salr.sh

python -m evaluation.score_harmbench \
  --behaviors_path data/behaviors/advbench.csv \
  --completions_path outputs/safety/advbench_pair.json \
  --save_path outputs/safety/advbench_pair_scores.json
```

The behavior CSV must provide matching `BehaviorID`, `Behavior`, `Tags`, and, for contextual behaviors, `ContextString`. The scorer retains HarmBench's standard/contextual classifier templates and uses `cais/HarmBench-Llama-2-13b-cls` with deterministic one-token judgments. It reports the mean ASR across behaviors. Hash-based copyright and multimodal behaviors require the full upstream evaluator. `--include_advbench_metric` optionally adds the upstream refusal-string heuristic, separately from classifier ASR.

Completions retain the upstream-compatible format:

```json
{"behavior_id": [{"test_case": "prepared prompt", "generation": "SaLR response"}]}
```

For the paper's five-run protocol, generate five independent completion files and score each separately. Safety generation retains the original sampling defaults: temperature 0.1, top-k 40, top-p 0.95, and 256 output tokens.

### XSTest over-refusal

Download `xstest_prompts.csv` from [XSTest](https://github.com/paul-rottger/exaggerated-safety), then preserve the original IDs when converting it:

```bash
python scripts/prepare_data.py xstest \
  --input data/xstest_prompts.csv --output data/xstest.json
bash scripts/xstest_eval.sh
python -m evaluation.score_xstest \
  --prompts_path data/xstest_prompts.csv \
  --completions_path outputs/xstest/completions.json \
  --save_path outputs/xstest/scores.json
```

The scorer uses XSTest's string-matching heuristic and reports safe-prompt and unsafe-prompt refusal rates separately. These are heuristic labels; they are separate from HarmBench classifier judgments.

### OverThink and H-CoT

SaLR-specific evaluation scripts are retained for FreshQA and MuSR under OverThink, and Malicious-Educator under H-CoT. Download their data from the [OverThink](https://github.com/akumar2709/OVERTHINK_public) and [H-CoT](https://github.com/dukeceicenter/jailbreak-reasoning-openai-o1o3-deepseek-r1) releases. The full upstream repositories and baseline runners are not bundled.

```bash
bash scripts/overthink_eval.sh --dataset freshqa --data-dir data/overthink
bash scripts/overthink_eval.sh --dataset murder_mystery --data-dir data/overthink
bash scripts/overthink_eval.sh --dataset object_placement --data-dir data/overthink

DATA_PATH=data/malicious_educator.parquet \
JUDGE_MODEL=gpt-5.1 \
bash scripts/hcot_eval.sh --setting hcot
```

OverThink writes responses, exact generated-token counts, visible reasoning-token counts, and latency. The fixed latent budget is recorded separately. OverThink refuses to mix a new run into an existing output; select a new `--output-jsonl` or explicitly pass `--overwrite`. H-CoT accepts a Parquet file with `Goal`, `Request`, and `Full_Input` for the `hcot` setting; it uses an OpenAI-compatible judge to report ASR and harmfulness ratings. Set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` for the judge. The scripts only call the judge when you run them. Malformed judge ratings are retried and rejected instead of being counted as safe responses.

H-CoT also requires a new output file by default. Use `--output_name <name>.jsonl` for separate runs, or `--resume` to continue a file with matching inputs, checkpoint, judge, and generation settings.

## Repository structure

```text
SaLR/
├── src/model.py                    # SaLR architecture and distillation losses
├── train.py                        # Mixed mathematical/safety training
├── test.py                         # Reasoning evaluation
├── eval_safety.py                   # Safety completion generation
├── eval_xstest_to_completions.py    # XSTest completion generation
├── safe-chain-generate/generate.py  # Qwen3-32B safe-chain construction
├── evaluation/                     # Shared inference and benchmark scorers
├── scripts/                        # Training, data preparation, evaluation
├── figs/                           # Paper figures in PNG and PDF
└── LICENSE                        # SaLR MIT license and upstream notices
```

## Paper

- **arXiv:** coming soon.
- **NeurIPS 2026 proceedings:** coming soon.

The badges above point to this section until the public links are available.

## Citation

Coming soon.

## Acknowledgements

We thank [CODI](https://github.com/zhenyi4/codi), [SafeChain](https://github.com/uw-nsl/safechain), [HarmBench](https://github.com/centerforaisafety/HarmBench), [XSTest](https://github.com/paul-rottger/exaggerated-safety), and the benchmark/attack authors.

## License

SaLR's original code is licensed under the [MIT License](LICENSE). Retained upstream code excerpts remain under their respective licenses, reproduced in the same license file.