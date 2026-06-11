# Nemotron Model Reasoning Challenge

LoRA fine-tuning scaffold for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/).

## Setup

```bash
uv sync --extra dev          # core + dev tooling
uv run pre-commit install
```

GPU training also needs `uv sync --extra gpu` and a manual Unsloth install (CUDA-specific).

## Data

```bash
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/
unzip data/'*.zip' -d data/
```

## Run (local proxy smoke)

```bash
uv run main train          # model=proxy train=smoke -> writes adapter
uv run main eval           # boxed accuracy on held-out puzzles
uv run main package        # -> submission.zip
```

## Run (real model)

```bash
uv run main train model=nemotron_nano train=qlora_t4   # Kaggle T4
uv run main train model=nemotron_nano train=lora_a100  # A100 bf16
```
