# Nemotron Reasoning Challenge Scaffold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a config-switchable LoRA fine-tuning pipeline for the NVIDIA Nemotron Model Reasoning Challenge that runs a tiny proxy model locally end-to-end and the real Nemotron base on GPU/Kaggle through the same code path.

**Architecture:** Hydra-composed config validated by pydantic (LoRA `rank ≤ 32` enforced) drives four pure-ish stages — `data` (puzzles → `\boxed{}` SFT records), `train` (PEFT/Unsloth LoRA SFT), `eval` (boxed-accuracy scorer), `submission` (zip the adapter). A `model` selector swaps a 135M proxy (CPU/MPS) for `nvidia/Nemotron-3-Nano-30B-A3B-BF16` (4-bit T4 / bf16 A100). A typer CLI exposes `prepare | train | eval | package`.

**Tech Stack:** Python 3.11+, uv, Hydra + OmegaConf, pydantic v2, transformers, peft, trl, datasets, (optional) unsloth + bitsandbytes behind a `[gpu]` extra, pytest, ruff, mypy, mise, rich, typer.

---

## File Structure

```
src/
  __init__.py
  main.py                  # typer CLI: prepare | train | eval | package
  logger.py                # rich logging setup
  config/
    __init__.py
    schemas.py             # pydantic: ModelConfig, DataConfig, TrainConfig(rank≤32), EvalConfig, ExperimentConfig
    loader.py              # Hydra compose -> validated ExperimentConfig
    settings.py            # env-driven Settings (HF_TOKEN, output dirs)
  data/
    __init__.py
    puzzles.py             # load problems.jsonl -> Puzzle list -> SFT chat records w/ \boxed{}
  train/
    __init__.py
    lora.py                # build LoraConfig; SFT via Unsloth-if-importable else PEFT+TRL
  eval/
    __init__.py
    boxed.py               # extract_boxed(), score(), evaluate()
  submission/
    __init__.py
    package.py             # adapter dir + adapter_config.json -> submission.zip; asserts rank≤32
config/
  experiment.yaml
  model/{proxy,nemotron_nano}.yaml
  data/default.yaml
  train/{smoke,qlora_t4,lora_a100}.yaml
  eval/boxed_accuracy.yaml
tests/
  conftest.py
  test_config.py
  test_puzzles.py
  test_boxed.py
  test_package.py
  test_pipeline.py         # slow: proxy smoke train -> eval -> package
data/.gitkeep              # kaggle download target (gitignored otherwise)
```

---

### Task 0: Project skeleton

**Files:**
- Create: `pyproject.toml`, `mise.toml`, `.pre-commit-config.yaml`, `.gitignore`, `README.md`, `REPORT.md`
- Create: `src/__init__.py`, `src/logger.py`, and empty `__init__.py` in each `src/` subpackage
- Create: `data/.gitkeep`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "nemotron-reasoning-challenge"
version = "0.1.0"
description = "LoRA fine-tuning scaffold for the NVIDIA Nemotron Model Reasoning Challenge"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "typer>=0.12.0",
  "rich>=13.7.0",
  "pydantic>=2.11.0",
  "pydantic-settings>=2.11.0",
  "hydra-core>=1.3.2",
  "omegaconf>=2.3.0",
  "torch>=2.3.0",
  "transformers>=4.45,<5",
  "datasets>=2.20.0",
  "peft>=0.13.0",
  "trl>=0.11.0",
  "tqdm>=4.67.0",
]

[project.optional-dependencies]
gpu = [
  "bitsandbytes>=0.43.0",
  "accelerate>=0.34.0",
]
# Unsloth is installed manually per-env (CUDA-specific wheels); see README.
dev = [
  "pytest>=8.4.0",
  "pytest-xdist>=3.6.0",
  "ruff>=0.11.10",
  "mypy>=1.15.0",
  "pre-commit>=3.7.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[project.scripts]
main = "src.main:app"

[tool.ruff]
fix = true
unsafe-fixes = false

[tool.ruff.lint]
select = ["F", "E", "W", "I", "UP", "B", "Q", "N"]
ignore = ["E501", "B008", "N812"]
fixable = ["ALL"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["N802", "N803"]

[tool.mypy]
disallow_untyped_defs = true
ignore_missing_imports = true
show_error_codes = true
plugins = ["pydantic.mypy"]
packages = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --strict-markers"
markers = [
  "slow: full pipeline / model download (deselect with '-m \"not slow\"')",
]
filterwarnings = ["ignore::DeprecationWarning"]
```

- [ ] **Step 2: Create `mise.toml`**

```toml
min_version = "2024.1.0"

[tasks.test]
description = "Fast tests only (excludes slow), parallel"
run = "uv run pytest -n auto -m 'not slow'"

[tasks."test:all"]
description = "Full suite including slow pipeline test"
run = "uv run pytest -n auto"

[tasks.lint]
description = "Ruff check + format check"
run = "uv run ruff check src tests && uv run ruff format --check src tests"

[tasks.typecheck]
description = "mypy"
run = "uv run mypy"

[tasks.train]
description = "Train a LoRA adapter (default: proxy/smoke)"
run = "uv run main train {{arg(name='overrides', default='')}}"

[tasks.eval]
description = "Evaluate an adapter (boxed accuracy)"
run = "uv run main eval {{arg(name='overrides', default='')}}"

[tasks.package]
description = "Package adapter -> submission.zip"
run = "uv run main package {{arg(name='overrides', default='')}}"
```

- [ ] **Step 3: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.10
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.15.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]
        pass_filenames: false
        args: [--config-file=pyproject.toml]
```

- [ ] **Step 4: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.DS_Store
.env
# data + run artifacts
data/*
!data/.gitkeep
outputs/
*.zip
adapters/
```

- [ ] **Step 5: Create package `__init__.py` files and `data/.gitkeep`**

Create empty files: `src/__init__.py`, `src/config/__init__.py`, `src/data/__init__.py`, `src/train/__init__.py`, `src/eval/__init__.py`, `src/submission/__init__.py`, `data/.gitkeep`.

- [ ] **Step 6: Create `src/logger.py`**

```python
"""Rich-backed logging setup."""

import logging

from rich.logging import RichHandler


def get_logger(name: str = "nemotron") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
        logger.propagate = False
    return logger
```

- [ ] **Step 7: Create minimal `README.md` and `REPORT.md`**

`README.md`:

```markdown
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
```

`REPORT.md`:

```markdown
# Nemotron Reasoning Challenge — Report

## Hypothesis

## Method
- Base model:
- Data / prompt format:
- LoRA config (rank ≤ 32):
- Training:

## Results
| run | boxed accuracy | notes |
|-----|----------------|-------|

## What worked / didn't

## Next steps
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: project skeleton (pyproject, mise, tooling, logger, docs)"
```

---

### Task 1: Config schemas + loader (rank ≤ 32 enforced)

**Files:**
- Create: `src/config/schemas.py`, `src/config/loader.py`, `src/config/settings.py`
- Test: `tests/test_config.py`, `tests/conftest.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 2: Write failing test `tests/test_config.py`**

```python
import pytest
from pydantic import ValidationError

from src.config.loader import load_experiment_config
from src.config.schemas import TrainConfig


class TestTrainConfigRankGuard:
    def test_rank_32_accepted(self):
        cfg = TrainConfig(preset="smoke", lora_rank=32, lora_alpha=32)
        assert cfg.lora_rank == 32

    @pytest.mark.parametrize("bad_rank", [33, 64, 128])
    def test_rank_above_32_rejected(self, bad_rank):
        with pytest.raises(ValidationError):
            TrainConfig(preset="smoke", lora_rank=bad_rank, lora_alpha=bad_rank)


class TestLoader:
    def test_default_compose_is_proxy_smoke(self):
        cfg = load_experiment_config([])
        assert cfg.model.kind == "proxy"
        assert cfg.train.preset == "smoke"
        assert cfg.train.lora_rank <= 32

    def test_override_to_nemotron(self):
        cfg = load_experiment_config(["model=nemotron_nano", "train=lora_a100"])
        assert cfg.model.hf_id == "nvidia/Nemotron-3-Nano-30B-A3B-BF16"
        assert cfg.train.lora_rank <= 32
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: src.config.schemas` / `loader`.

- [ ] **Step 4: Write `src/config/schemas.py`**

```python
"""Pydantic schemas for the Hydra-composed experiment config."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["proxy", "nemotron"]
    hf_id: str
    # 4-bit on T4, bf16 on A100, full precision for the tiny proxy.
    load_in_4bit: bool = False
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    max_seq_length: int = Field(2048, gt=0)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    prompt_field: str = "problem"
    answer_field: str = "answer"
    eval_fraction: float = Field(0.1, gt=0.0, lt=1.0)
    seed: int = 42
    max_samples: int | None = Field(None, gt=0)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: Literal["smoke", "qlora_t4", "lora_a100"]
    lora_rank: int = Field(16, gt=0, le=32)  # competition hard cap
    lora_alpha: int = Field(32, gt=0)
    lora_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    learning_rate: float = Field(2e-4, gt=0.0)
    num_epochs: int = Field(1, gt=0)
    max_steps: int = Field(-1)  # -1 = full epoch(s); >0 caps steps (smoke)
    batch_size: int = Field(1, gt=0)
    grad_accum: int = Field(1, gt=0)
    output_dir: str = "adapters/run"


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tolerance: float = Field(1e-2, ge=0.0)
    max_new_tokens: int = Field(512, gt=0)
    output_dir: str = "outputs/eval"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
```

- [ ] **Step 5: Write `src/config/loader.py`**

```python
"""Hydra config composer returning a validated ExperimentConfig."""

from typing import Any

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.config.schemas import ExperimentConfig


def load_experiment_config(
    overrides: list[str] | None = None,
    config_name: str = "experiment",
    config_path: str = "../../config",
) -> ExperimentConfig:
    """Compose Hydra config + validate. Raises ValidationError on bad fields (e.g. rank>32)."""
    with initialize(version_base=hydra.__version__, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides or [])
        cfg_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
        return ExperimentConfig(**cfg_dict)
```

- [ ] **Step 6: Write `src/config/settings.py`**

```python
"""Env-driven runtime settings."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    HF_TOKEN: SecretStr | None = None


settings = Settings()
```

- [ ] **Step 7: Create the Hydra config tree**

`config/experiment.yaml`:

```yaml
defaults:
  - model: proxy
  - data: default
  - train: smoke
  - eval: boxed_accuracy
  - _self_

experiment_name: nemotron-scaffold
```

`config/model/proxy.yaml`:

```yaml
kind: proxy
hf_id: HuggingFaceTB/SmolLM2-135M
load_in_4bit: false
dtype: float32
max_seq_length: 1024
```

`config/model/nemotron_nano.yaml`:

```yaml
kind: nemotron
hf_id: nvidia/Nemotron-3-Nano-30B-A3B-BF16
load_in_4bit: true
dtype: bfloat16
max_seq_length: 2048
```

`config/data/default.yaml`:

```yaml
path: data/problems.jsonl
prompt_field: problem
answer_field: answer
eval_fraction: 0.1
seed: 42
max_samples: null
```

`config/train/smoke.yaml`:

```yaml
preset: smoke
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0
learning_rate: 0.0002
num_epochs: 1
max_steps: 2
batch_size: 1
grad_accum: 1
output_dir: adapters/smoke
```

`config/train/qlora_t4.yaml`:

```yaml
preset: qlora_t4
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 0.0002
num_epochs: 1
max_steps: -1
batch_size: 1
grad_accum: 8
output_dir: adapters/qlora_t4
```

`config/train/lora_a100.yaml`:

```yaml
preset: lora_a100
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
learning_rate: 0.0002
num_epochs: 2
max_steps: -1
batch_size: 4
grad_accum: 4
output_dir: adapters/lora_a100
```

`config/eval/boxed_accuracy.yaml`:

```yaml
tolerance: 0.01
max_new_tokens: 512
output_dir: outputs/eval
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (4 tests).

- [ ] **Step 9: Commit**

```bash
git add src/config config tests/test_config.py tests/conftest.py
git commit -m "feat: config schemas + Hydra loader with rank<=32 guard"
```

---

### Task 2: Puzzle data loading + SFT formatting

**Files:**
- Create: `src/data/puzzles.py`
- Test: `tests/test_puzzles.py`, `tests/fixtures/mini_problems.jsonl`

- [ ] **Step 1: Create fixture `tests/fixtures/mini_problems.jsonl`**

```jsonl
{"id": "p1", "problem": "Apply the rule: double the input. Input: 6.", "answer": "12"}
{"id": "p2", "problem": "Apply the rule: XOR with 1. Input bit: 0.", "answer": "1"}
{"id": "p3", "problem": "Solve for x: 2x + 4 = 10.", "answer": "3"}
{"id": "p4", "problem": "Apply the rule: add 1.5. Input: 2.5.", "answer": "4.0"}
```

- [ ] **Step 2: Write failing test `tests/test_puzzles.py`**

```python
from pathlib import Path

import pytest

from src.config.schemas import DataConfig
from src.data.puzzles import Puzzle, load_puzzles, split_puzzles, to_sft_record

FIXTURE = Path(__file__).parent / "fixtures" / "mini_problems.jsonl"


@pytest.fixture
def cfg() -> DataConfig:
    return DataConfig(path=str(FIXTURE), eval_fraction=0.25, seed=0)


class TestLoad:
    def test_loads_all_rows(self, cfg):
        puzzles = load_puzzles(cfg)
        assert len(puzzles) == 4
        assert all(isinstance(p, Puzzle) for p in puzzles)
        assert puzzles[0].answer == "12"

    def test_max_samples_caps(self, cfg):
        cfg.max_samples = 2
        assert len(load_puzzles(cfg)) == 2

    def test_custom_field_names(self, tmp_path):
        f = tmp_path / "alt.jsonl"
        f.write_text('{"id": "a", "q": "Q?", "sol": "9"}\n')
        c = DataConfig(path=str(f), prompt_field="q", answer_field="sol")
        p = load_puzzles(c)[0]
        assert p.prompt == "Q?" and p.answer == "9"


class TestSplit:
    def test_split_is_disjoint_and_complete(self, cfg):
        train, dev = split_puzzles(load_puzzles(cfg), cfg)
        assert len(dev) == 1  # 25% of 4
        assert len(train) == 3
        ids = {p.id for p in train} | {p.id for p in dev}
        assert len(ids) == 4

    def test_split_is_seed_stable(self, cfg):
        a = split_puzzles(load_puzzles(cfg), cfg)[1]
        b = split_puzzles(load_puzzles(cfg), cfg)[1]
        assert [p.id for p in a] == [p.id for p in b]


class TestSftRecord:
    def test_answer_wrapped_in_boxed(self):
        rec = to_sft_record(Puzzle(id="x", prompt="Q?", answer="42"))
        assert r"\boxed{42}" in rec["text"]
        assert "Q?" in rec["text"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_puzzles.py -v`
Expected: FAIL — `ModuleNotFoundError: src.data.puzzles`.

- [ ] **Step 4: Write `src/data/puzzles.py`**

```python
"""Load reasoning puzzles and format them as SFT records with \\boxed{} targets."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from src.config.schemas import DataConfig

SYSTEM_PROMPT = (
    "You are a careful reasoning solver. Identify the underlying transformation "
    "rule, apply it, and give the final answer inside \\boxed{}."
)


@dataclass(frozen=True)
class Puzzle:
    id: str
    prompt: str
    answer: str


def load_puzzles(cfg: DataConfig) -> list[Puzzle]:
    path = Path(cfg.path)
    puzzles: list[Puzzle] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            puzzles.append(
                Puzzle(
                    id=str(row.get("id", i)),
                    prompt=str(row[cfg.prompt_field]),
                    answer=str(row[cfg.answer_field]),
                )
            )
    if cfg.max_samples is not None:
        puzzles = puzzles[: cfg.max_samples]
    return puzzles


def split_puzzles(puzzles: list[Puzzle], cfg: DataConfig) -> tuple[list[Puzzle], list[Puzzle]]:
    order = list(puzzles)
    random.Random(cfg.seed).shuffle(order)
    n_dev = max(1, round(len(order) * cfg.eval_fraction))
    return order[n_dev:], order[:n_dev]


def to_sft_record(puzzle: Puzzle) -> dict[str, str]:
    text = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{puzzle.prompt}\n"
        f"<|assistant|>\nThe answer is \\boxed{{{puzzle.answer}}}."
    )
    return {"id": puzzle.id, "text": text}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_puzzles.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add src/data/puzzles.py tests/test_puzzles.py tests/fixtures/mini_problems.jsonl
git commit -m "feat: puzzle loader, seed-stable split, boxed SFT formatting"
```

---

### Task 3: Boxed-answer evaluation

**Files:**
- Create: `src/eval/boxed.py`
- Test: `tests/test_boxed.py`

- [ ] **Step 1: Write failing test `tests/test_boxed.py`**

```python
import pytest

from src.eval.boxed import extract_boxed, score


class TestExtractBoxed:
    @pytest.mark.parametrize(
        "text,expected",
        [
            (r"so \boxed{42}.", "42"),
            (r"\boxed{3.14} is pi", "3.14"),
            (r"nested \boxed{x = \frac{1}{2}}", r"x = \frac{1}{2}"),
            ("no box here", None),
            (r"\boxed{}", ""),
        ],
    )
    def test_extract(self, text, expected):
        assert extract_boxed(text) == expected


class TestScore:
    @pytest.mark.parametrize(
        "pred,gold,ok",
        [
            ("42", "42", True),       # exact string
            (" 42 ", "42", True),     # whitespace-insensitive
            ("3.140", "3.14", True),  # numeric equal
            ("3.149", "3.14", False), # outside 1e-2
            ("3.145", "3.14", True),  # within 1e-2
            ("cat", "dog", False),    # non-numeric mismatch
            (None, "42", False),      # missing box
        ],
    )
    def test_score(self, pred, gold, ok):
        assert score(pred, gold, tolerance=1e-2) is ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_boxed.py -v`
Expected: FAIL — `ModuleNotFoundError: src.eval.boxed`.

- [ ] **Step 3: Write `src/eval/boxed.py`**

```python
"""Parse \\boxed{} answers and score against gold (exact or numeric tolerance)."""


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...}, handling nested braces. None if absent."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    out: list[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out)


def _as_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def score(pred: str | None, gold: str, tolerance: float = 1e-2) -> bool:
    """Exact string match (whitespace-insensitive), else numeric within tolerance."""
    if pred is None:
        return False
    p, g = pred.strip(), gold.strip()
    if p == g:
        return True
    pf, gf = _as_float(p), _as_float(g)
    if pf is not None and gf is not None:
        return abs(pf - gf) <= tolerance
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_boxed.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/boxed.py tests/test_boxed.py
git commit -m "feat: boxed-answer extraction + exact/tolerance scorer"
```

---

### Task 4: Submission packaging

**Files:**
- Create: `src/submission/package.py`
- Test: `tests/test_package.py`

- [ ] **Step 1: Write failing test `tests/test_package.py`**

```python
import json
import zipfile
from pathlib import Path

import pytest

from src.submission.package import package_submission


def _make_adapter(dir_: Path, rank: int) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "adapter_config.json").write_text(json.dumps({"r": rank, "peft_type": "LORA"}))
    (dir_ / "adapter_model.safetensors").write_bytes(b"\x00\x01")
    return dir_


class TestPackage:
    def test_zip_contains_adapter_files(self, tmp_path):
        adapter = _make_adapter(tmp_path / "ad", rank=16)
        out = package_submission(adapter, tmp_path / "submission.zip")
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
        assert "adapter_config.json" in names
        assert "adapter_model.safetensors" in names

    def test_rank_above_32_rejected(self, tmp_path):
        adapter = _make_adapter(tmp_path / "ad", rank=64)
        with pytest.raises(ValueError, match="rank"):
            package_submission(adapter, tmp_path / "submission.zip")

    def test_missing_adapter_config_rejected(self, tmp_path):
        bare = tmp_path / "bare"
        bare.mkdir()
        (bare / "adapter_model.safetensors").write_bytes(b"\x00")
        with pytest.raises(FileNotFoundError):
            package_submission(bare, tmp_path / "submission.zip")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_package.py -v`
Expected: FAIL — `ModuleNotFoundError: src.submission.package`.

- [ ] **Step 3: Write `src/submission/package.py`**

```python
"""Package a trained LoRA adapter directory into submission.zip (rank <= 32 enforced)."""

import json
import zipfile
from pathlib import Path

MAX_RANK = 32


def package_submission(adapter_dir: Path, out_zip: Path) -> Path:
    """Zip the adapter dir contents (flat). Validates adapter_config.json exists and r <= 32."""
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    rank = json.loads(config_path.read_text()).get("r")
    if rank is not None and rank > MAX_RANK:
        raise ValueError(f"LoRA rank {rank} exceeds competition cap of {MAX_RANK}")

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(adapter_dir.iterdir()):
            if f.is_file():
                z.write(f, arcname=f.name)
    return out_zip
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_package.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/submission/package.py tests/test_package.py
git commit -m "feat: submission.zip packager with rank<=32 enforcement"
```

---

### Task 5: LoRA training (Unsloth-if-available else PEFT/TRL)

**Files:**
- Create: `src/train/lora.py`
- Test: covered by the slow pipeline test in Task 7 (no isolated unit test — training is I/O + model heavy)

- [ ] **Step 1: Write `src/train/lora.py`**

```python
"""Train a LoRA adapter via Unsloth when importable, else PEFT + TRL SFTTrainer."""

from pathlib import Path

import torch
from datasets import Dataset

from src.config.schemas import ExperimentConfig
from src.data.puzzles import load_puzzles, split_puzzles, to_sft_record
from src.logger import get_logger

logger = get_logger(__name__)

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _target_modules() -> list[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def train_adapter(cfg: ExperimentConfig) -> Path:
    """Run SFT and write a LoRA adapter to cfg.train.output_dir. Returns that path."""
    puzzles = load_puzzles(cfg.data)
    train_puzzles, _ = split_puzzles(puzzles, cfg.data)
    dataset = Dataset.from_list([to_sft_record(p) for p in train_puzzles])
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from unsloth import FastLanguageModel  # noqa: F401

        logger.info("Unsloth available — using FastLanguageModel path")
        return _train_unsloth(cfg, dataset, out_dir)
    except ImportError:
        logger.info("Unsloth not found — using PEFT + TRL path")
        return _train_peft(cfg, dataset, out_dir)


def _train_peft(cfg: ExperimentConfig, dataset: Dataset, out_dir: Path) -> Path:
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_kwargs: dict = {"torch_dtype": _DTYPES[cfg.model.dtype]}
    if cfg.model.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_DTYPES[cfg.model.dtype],
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model.hf_id, **model_kwargs)

    peft_config = LoraConfig(
        r=cfg.train.lora_rank,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=_target_modules(),
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_epochs,
        max_steps=cfg.train.max_steps,
        max_seq_length=cfg.model.max_seq_length,
        dataset_text_field="text",
        logging_steps=1,
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def _train_unsloth(cfg: ExperimentConfig, dataset: Dataset, out_dir: Path) -> Path:
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.hf_id,
        max_seq_length=cfg.model.max_seq_length,
        dtype=_DTYPES[cfg.model.dtype],
        load_in_4bit=cfg.model.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.train.lora_rank,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=_target_modules(),
    )
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_epochs,
        max_steps=cfg.train.max_steps,
        max_seq_length=cfg.model.max_seq_length,
        dataset_text_field="text",
        logging_steps=1,
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
```

- [ ] **Step 2: Sanity import check**

Run: `uv run python -c "from src.train.lora import train_adapter; print('ok')"`
Expected: prints `ok` (no execution, just import).

- [ ] **Step 3: Commit**

```bash
git add src/train/lora.py
git commit -m "feat: LoRA SFT trainer (Unsloth-if-available else PEFT/TRL)"
```

---

### Task 6: Evaluation runner (generate + score)

**Files:**
- Modify: `src/eval/boxed.py` (add `evaluate`)
- Test: extend `tests/test_boxed.py` with a generation-free `evaluate` test using a stub

- [ ] **Step 1: Add failing test to `tests/test_boxed.py`**

```python
class TestEvaluate:
    def test_evaluate_scores_predictions(self):
        from src.data.puzzles import Puzzle
        from src.eval.boxed import evaluate

        puzzles = [Puzzle(id="a", prompt="Q1", answer="2"), Puzzle(id="b", prompt="Q2", answer="5")]
        # generate_fn returns model text; first correct, second wrong
        texts = iter([r"answer \boxed{2}", r"answer \boxed{9}"])
        result = evaluate(puzzles, generate_fn=lambda _p: next(texts), tolerance=1e-2)
        assert result["accuracy"] == 0.5
        assert result["n"] == 2
        assert result["correct"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_boxed.py::TestEvaluate -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate'`.

- [ ] **Step 3: Add `evaluate` to `src/eval/boxed.py`**

```python
from collections.abc import Callable

from src.data.puzzles import Puzzle


def evaluate(
    puzzles: list[Puzzle],
    generate_fn: Callable[[str], str],
    tolerance: float = 1e-2,
) -> dict[str, float | int]:
    """Run generate_fn over each puzzle prompt, score the boxed answer, return aggregate."""
    correct = 0
    for p in puzzles:
        pred = extract_boxed(generate_fn(p.prompt))
        if score(pred, p.answer, tolerance):
            correct += 1
    n = len(puzzles)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0}
```

Add `from collections.abc import Callable` and `from src.data.puzzles import Puzzle` to the top of the file (hoisted imports).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_boxed.py -v`
Expected: PASS (all classes).

- [ ] **Step 5: Commit**

```bash
git add src/eval/boxed.py tests/test_boxed.py
git commit -m "feat: evaluate() runner over a generate_fn"
```

---

### Task 7: CLI wiring + slow end-to-end smoke

**Files:**
- Create: `src/main.py`
- Create: `src/eval/runner.py` (builds a HF generate_fn from a trained adapter, calls `evaluate`)
- Test: `tests/test_pipeline.py` (slow)

- [ ] **Step 1: Write `src/eval/runner.py`**

```python
"""Load a base model + trained adapter and evaluate boxed accuracy."""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.schemas import ExperimentConfig
from src.data.puzzles import SYSTEM_PROMPT, load_puzzles, split_puzzles
from src.eval.boxed import evaluate
from src.logger import get_logger

logger = get_logger(__name__)
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def run_eval(cfg: ExperimentConfig, adapter_dir: Path) -> dict[str, float | int]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.model.hf_id, torch_dtype=_DTYPES[cfg.model.dtype])
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    def generate_fn(prompt: str) -> str:
        text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cfg.eval.max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    _, dev = split_puzzles(load_puzzles(cfg.data), cfg.data)
    result = evaluate(dev, generate_fn, tolerance=cfg.eval.tolerance)

    out_dir = Path(cfg.eval.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    logger.info("boxed accuracy %.3f (%d/%d)", result["accuracy"], result["correct"], result["n"])
    return result
```

- [ ] **Step 2: Write `src/main.py`**

```python
"""Typer CLI: prepare | train | eval | package."""

from pathlib import Path

import typer
from rich import print as rich_print

from src.config.loader import load_experiment_config
from src.data.puzzles import load_puzzles, split_puzzles

app = typer.Typer(
    name="main",
    help="Nemotron reasoning challenge scaffold",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.command()
def prepare(overrides: list[str] = typer.Argument(None)) -> None:
    """Load + split puzzles, print dataset stats."""
    cfg = load_experiment_config(overrides)
    train, dev = split_puzzles(load_puzzles(cfg.data), cfg.data)
    rich_print(f"[green]loaded[/green] {len(train)} train / {len(dev)} dev from {cfg.data.path}")


@app.command()
def train(overrides: list[str] = typer.Argument(None)) -> None:
    """Train a LoRA adapter."""
    from src.train.lora import train_adapter

    cfg = load_experiment_config(overrides)
    out = train_adapter(cfg)
    rich_print(f"[green]adapter saved[/green] -> {out}")


@app.command()
def eval(overrides: list[str] = typer.Argument(None)) -> None:  # noqa: A001
    """Evaluate a trained adapter (boxed accuracy)."""
    from src.eval.runner import run_eval

    cfg = load_experiment_config(overrides)
    result = run_eval(cfg, Path(cfg.train.output_dir))
    rich_print(f"[cyan]accuracy[/cyan] {result['accuracy']:.3f}")


@app.command()
def package(overrides: list[str] = typer.Argument(None)) -> None:
    """Package the trained adapter into submission.zip."""
    from src.submission.package import package_submission

    cfg = load_experiment_config(overrides)
    out = package_submission(Path(cfg.train.output_dir), Path("submission.zip"))
    rich_print(f"[green]packaged[/green] -> {out}")


if __name__ == "__main__":
    app()
```

Note: `train`/`eval`/`package` use function-local imports of the heavy modules so `prepare` and `--help` stay fast and import-light. This is an allowed local-import exception (heavy/optional deps).

- [ ] **Step 3: Write slow pipeline test `tests/test_pipeline.py`**

```python
import shutil
import zipfile
from pathlib import Path

import pytest

from src.config.loader import load_experiment_config
from src.eval.runner import run_eval
from src.submission.package import package_submission
from src.train.lora import train_adapter

FIXTURE = Path(__file__).parent / "fixtures" / "mini_problems.jsonl"


@pytest.mark.slow
def test_proxy_pipeline_end_to_end(tmp_path):
    adapter_dir = tmp_path / "adapter"
    cfg = load_experiment_config(
        [
            "model=proxy",
            "train=smoke",
            f"data.path={FIXTURE}",
            f"train.output_dir={adapter_dir}",
            f"eval.output_dir={tmp_path / 'eval'}",
        ]
    )
    out = train_adapter(cfg)
    assert (Path(out) / "adapter_config.json").exists()

    result = run_eval(cfg, Path(out))
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["n"] >= 1

    zip_path = package_submission(Path(out), tmp_path / "submission.zip")
    with zipfile.ZipFile(zip_path) as z:
        assert "adapter_config.json" in z.namelist()

    shutil.rmtree(adapter_dir, ignore_errors=True)
```

- [ ] **Step 4: Run fast tests (exclude slow) to confirm nothing regressed**

Run: `uv run pytest -m "not slow" -v`
Expected: all PASS.

- [ ] **Step 5: Run the slow pipeline test (downloads SmolLM2-135M once)**

Run: `uv run pytest tests/test_pipeline.py -v -m slow`
Expected: PASS. First run downloads the 135M proxy (~250MB) and trains 2 steps on CPU.

- [ ] **Step 6: Verify the CLI end-to-end manually**

Run:
```bash
uv run main train data.path=tests/fixtures/mini_problems.jsonl
uv run main eval data.path=tests/fixtures/mini_problems.jsonl
uv run main package
```
Expected: adapter saved, an accuracy printed, `submission.zip` created in repo root.

- [ ] **Step 7: Commit**

```bash
git add src/main.py src/eval/runner.py tests/test_pipeline.py
git commit -m "feat: typer CLI (prepare/train/eval/package) + slow e2e proxy smoke"
```

---

### Task 8: Lint, typecheck, simplify pass, finalize

- [ ] **Step 1: Lint + typecheck**

Run: `mise run lint && mise run typecheck`
Expected: clean. Fix any ruff/mypy findings inline and re-run.

- [ ] **Step 2: Run the full fast suite under pre-commit**

Run: `uv run pre-commit run --all-files && uv run pytest -m "not slow"`
Expected: hooks pass, tests green. Stage any hook-modified files.

- [ ] **Step 3: Simplification review (per CLAUDE.md, before final commit)**

Run `/simplify` (or the `code-review` skill) on the working-tree diff. Fold findings into the relevant commits via `git commit --amend` / interactive fixups, not a new "cleanup" commit.

- [ ] **Step 4: Final commit if anything changed**

```bash
git add -A
git commit -m "chore: lint, typecheck, simplification pass"
```

---

## Self-Review

- **Spec coverage:** model switch (Task 1 configs) ✓; puzzles→`\boxed{}` SFT (Task 2) ✓; LoRA SFT Unsloth-else-PEFT, 4bit/bf16 (Task 5) ✓; boxed scorer exact+tolerance (Tasks 3, 6) ✓; rank≤32 enforced at config (Task 1) and package (Task 4) ✓; `submission.zip` with `adapter_config.json` (Task 4) ✓; tests one-file-per-area, slow smoke (Tasks 1–7) ✓; README/REPORT (Task 0) ✓; mise tasks (Task 0) ✓.
- **Type consistency:** `Puzzle(id, prompt, answer)`, `to_sft_record -> {"id","text"}`, `extract_boxed -> str|None`, `score(pred,gold,tolerance) -> bool`, `evaluate -> {"n","correct","accuracy"}`, `train_adapter(cfg) -> Path`, `run_eval(cfg, adapter_dir) -> dict`, `package_submission(adapter_dir, out_zip) -> Path` — consistent across tasks.
- **Placeholders:** none — all code blocks are concrete.
- **Known caveat (carried from spec):** the real `problems.jsonl` field names are assumed `problem`/`answer`; `DataConfig.prompt_field`/`answer_field` make this overridable once the Kaggle data is downloaded and inspected. Adjust `config/data/default.yaml` then.
