# SaLR

This repository contains the implementation of **SaLR: Safety-Aware Latent Space Reasoning in Large Language Models**.

## Environment

We recommend using Python 3.10.14.

```bash
conda create -n salr python=3.10.14
conda activate salr
pip install -r requirements.txt
```

## Training

Training scripts are provided in the `scripts/` directory.

### Train SaLR on LLaMA 3.2-1B

```bash
bash scripts/train_llama1b_SaLR.sh
```

### Train SaLR on LLaMA 3.2-3B

```bash
bash scripts/train_llama3b_SaLR.sh
```

### Train SaLR on LLaMA 3.1-8B

```bash
bash scripts/train_llama8b_SaLR.sh
```

## Evaluation

### Mathematical reasoning evaluation

To evaluate mathematical reasoning performance, run:

```bash
bash scripts/test_llama.sh
```

### Safety evaluation

To evaluate safety robustness, run:

```bash
bash scripts/safe_llama_salr.sh
```
