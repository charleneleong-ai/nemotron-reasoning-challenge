# Nemotron Model Reasoning Challenge

LoRA fine-tuning scaffold for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/).

## Setup

```bash
uv sync --extra dev          # core + dev tooling
uv run pre-commit install
```

GPU training also needs `uv sync --extra gpu` and a manual Unsloth install (CUDA-specific).

## Data

Put your Kaggle API token in `.env` as `KAGGLE_API_TOKEN=...` (Kaggle → Settings → API),
then:

```bash
uv run main download    # -> data/train.csv (id,prompt,answer), data/test.csv
```

`train.csv` has 9,500 rows. The loader reads CSV or JSONL and preserves string answers
(leading-zero binary strings like `01000011` are not coerced to ints).

## Run (local proxy smoke)

Uses a tiny `SmolLM2-135M` proxy so the whole pipeline runs on CPU/MPS:

```bash
uv run main train data.max_samples=40    # model=proxy train=smoke -> writes adapter
uv run main eval  data.max_samples=40 eval.max_new_tokens=64   # boxed accuracy on dev
uv run main package                       # -> submission.zip
```

## Run (real model — needs a CUDA GPU)

`nvidia/Nemotron-3-Nano-30B-A3B-BF16` (30B) requires a CUDA GPU (4-bit QLoRA via
bitsandbytes / bf16) — it does **not** run on Mac CPU/MPS. Run on an A100 box or a
Kaggle GPU notebook after `uv sync --extra gpu`:

```bash
uv run main train model=nemotron_nano train=qlora_t4   # Kaggle T4 (4-bit)
uv run main train model=nemotron_nano train=lora_a100  # A100 bf16
uv run main eval  model=nemotron_nano
uv run main package                                    # rank<=32 enforced -> submission.zip
```
