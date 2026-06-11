# Nemotron Model Reasoning Challenge — Submission Scaffold

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation

## Goal

Scaffold an experimentation repo for the [NVIDIA Nemotron Model Reasoning
Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/).
The deliverable is a config-switchable LoRA fine-tuning pipeline that runs a tiny
proxy model locally (CPU/MPS) for fast iteration and the real Nemotron base model
on GPU (A100) or Kaggle for the final submission — all through one code path.

## Competition facts (the contract we build to)

- **Base model:** `nvidia/Nemotron-3-Nano-30B-A3B-BF16` — 30B total / 3B active,
  Mamba-Transformer MoE.
- **Submission artifact:** `submission.zip` containing a LoRA adapter with **rank ≤ 32**
  and an `adapter_config.json`.
- **Task:** logical-reasoning puzzles — infer and apply a hidden transformation rule.
  Domains include bit manipulation and algebraic equations.
- **Evaluation:** accuracy. The model must emit its final answer inside `\boxed{...}`.
  Scored by exact string match OR numeric tolerance ±1e-2.
- **Deadline:** 2026-06-15. License CC BY 4.0.

## Key design decision: the compute switch

A 30B model cannot load on a Mac (CPU/MPS), so the local "smoke" path trains a tiny
proxy model. The same modules, CLI, and config schema drive both:

| Selector | Model | Train preset | Runs on | Purpose |
|----------|-------|--------------|---------|---------|
| `model=proxy` | `HuggingFaceTB/SmolLM2-135M` | `train=smoke` (~8 ex) | Mac CPU/MPS | smoke baseline + test target |
| `model=nemotron_nano` | `nvidia/Nemotron-3-Nano-30B-A3B-BF16` | `train=qlora_t4` | Kaggle T4 (4-bit) | final submission |
| `model=nemotron_nano` | same | `train=lora_a100` | A100 (bf16) | larger experiments |

Real training is driven by the user after the scaffold lands. This pass delivers the
proxy path running green end-to-end and packaging a valid `submission.zip`.

## Repository structure

```
nemotron-reasoning-challenge/
├── pyproject.toml            # uv; core deps + optional [gpu] extra (unsloth, bitsandbytes)
├── mise.toml                 # test / lint / typecheck / train / eval / package tasks
├── .pre-commit-config.yaml   # ruff + mypy
├── README.md                 # setup + CLI usage
├── REPORT.md                 # writeup template (hypothesis / method / results / next)
├── config/                   # Hydra config tree
│   ├── experiment.yaml       # top-level compose (defaults list)
│   ├── model/{proxy,nemotron_nano}.yaml
│   ├── data/default.yaml
│   ├── train/{smoke,qlora_t4,lora_a100}.yaml   # rank ≤ 32
│   └── eval/boxed_accuracy.yaml
├── src/
│   ├── main.py               # typer CLI: prepare | train | eval | package
│   ├── config/               # pydantic schemas + loader; rank ≤ 32 validator
│   ├── data/puzzles.py       # problems.jsonl -> chat/SFT records with \boxed{} target
│   ├── train/lora.py         # Unsloth-if-importable else PEFT; 4bit (T4) / bf16 (A100)
│   ├── eval/boxed.py         # parse \boxed{}, exact + ±1e-2 tolerance scorer
│   └── submission/package.py # adapter + adapter_config.json -> submission.zip
├── data/                     # gitignored; kaggle download target
├── tests/                    # pytest, one file per area
└── docs/superpowers/specs/   # this doc
```

## Data flow

`problems.jsonl` → `data/puzzles.py` (format each puzzle into a chat prompt whose
target answer is wrapped in `\boxed{}`) → `train/lora.py` (SFT a rank-≤32 LoRA) →
`eval/boxed.py` (boxed-accuracy on held-out puzzles) → `submission/package.py`
(zip the adapter directory).

## Components

- **`src/config`** — pydantic schemas mirroring the Hydra tree; a loader that composes
  config and validates invariants (LoRA `rank ≤ 32`, known model selector). Single source
  of truth for run parameters.
- **`src/data/puzzles.py`** — pure functions: load JSONL, render a puzzle into a
  prompt/answer pair, wrap the gold answer in `\boxed{}`. No model dependency, so it is
  fully unit-testable.
- **`src/train/lora.py`** — builds the LoRA config, loads the selected model (4-bit via
  bitsandbytes on T4, bf16 on A100, fp32 tiny proxy locally), runs SFT, writes the adapter.
  Prefers Unsloth when importable, falls back to PEFT.
- **`src/eval/boxed.py`** — `extract_boxed(text)` and `score(pred, gold)` (exact match,
  then numeric ±1e-2). Independent of the training stack.
- **`src/submission/package.py`** — assembles the adapter dir (weights + `adapter_config.json`)
  into `submission.zip`; asserts the rank-≤32 invariant at package time.

## Error handling

- Config loader raises on invalid selectors or `rank > 32` before any model load.
- `extract_boxed` returns `None` on malformed/missing `\boxed{}` rather than throwing;
  the scorer treats `None` as incorrect.
- GPU-only deps live behind the `[gpu]` extra and a lazy import in `train/lora.py`, so the
  proxy/CPU path and the test suite never require them.

## Testing

Behavior-focused pytest, one file per area, parametrized to collapse near-duplicates:

- `\boxed{}` extraction including nested/malformed/missing input.
- Scorer: exact match vs numeric ±1e-2 boundary cases (e.g. 1.005 vs 1.00).
- Config schema: `rank=64` rejected, `rank=32` accepted; unknown model selector rejected.
- Puzzle → SFT record shape and `\boxed{}` wrapping.
- Packager: produces a zip containing `adapter_config.json`.

The proxy smoke train (full pipeline) is marked `@pytest.mark.slow`.

## Out of scope (this pass)

- Real Nemotron training runs, sweeps, synthetic-data augmentation, RL — user-driven after
  the scaffold lands; `train/lora.py` is the entry point.
- Kaggle dataset download automation beyond a documented `kaggle competitions download` step.
