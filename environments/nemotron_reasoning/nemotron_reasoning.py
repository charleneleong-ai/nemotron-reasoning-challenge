"""Nemotron reasoning-puzzle RLVR environment for Prime hosted training.

Single-turn: a puzzle prompt -> the model thinks then emits ``\\boxed{answer}``.
Reward = boxed-answer correctness (exact / ±1e-2) plus a small shaping bonus for
``<think>…</think>`` then ``\\boxed{}``. The answers are machine-checkable, so the
grader IS the verifier — no reward model.

    prime env push  --path environments/nemotron_reasoning
    prime train     configs/rl/nemotron.toml

The grader (extract_boxed / score) is inlined rather than imported from src/ so the
pushed package is self-contained inside Prime's training container.
"""

import csv
import re
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

_DATA = Path(__file__).parent / "data" / "train.csv"
_THINK_THEN_BOXED = re.compile(
    r"<think>.*?</think>.*?\\boxed\{.*?\}", re.IGNORECASE | re.DOTALL
)


def extract_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}``, handling nested braces. None if absent."""
    start = text.rfind(r"\boxed{")
    if start == -1:
        return None
    i, depth, out = start + len(r"\boxed{"), 1, []
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


def score(pred: str | None, gold: str, tolerance: float = 1e-2) -> bool:
    """Exact match (whitespace-insensitive), else numeric within tolerance."""
    if pred is None:
        return False
    p, g = pred.strip(), gold.strip()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) <= tolerance
    except (ValueError, TypeError):
        return False


def _text(completion: Any) -> str:
    """Normalise a completion (plain string or [{role, content}, ...])."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return str(completion[-1].get("content", ""))
    return str(completion)


def boxed_reward(completion: Any, answer: str, **_: Any) -> float:
    """1.0 if the completion's ``\\boxed{}`` matches the gold answer, else 0.0."""
    return 1.0 if score(extract_boxed(_text(completion)), str(answer)) else 0.0


def format_reward(completion: Any, **_: Any) -> float:
    """1.0 if the completion has ``<think>…</think>`` then a ``\\boxed{}``."""
    return 1.0 if _THINK_THEN_BOXED.search(_text(completion)) else 0.0


def _rows(path: Path, n_tasks: int | None, start: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if i < start:
                continue
            if n_tasks is not None and len(out) >= n_tasks:
                break
            out.append(
                {
                    "question": r["prompt"],
                    "answer": str(r["answer"]),
                    "info": {"id": r.get("id", str(i))},
                }
            )
    return out


def load_environment(
    n_tasks: int | None = 4000,
    start: int = 0,
    format_weight: float = 0.1,
    data_path: str = "",
    **kwargs: Any,
) -> vf.Environment:
    """Puzzle prompt -> boxed answer. n_tasks: train subset (None = all rows)."""
    path = Path(data_path) if data_path else _DATA
    dataset = Dataset.from_list(_rows(path, n_tasks, start))
    rubric = vf.Rubric(
        funcs=[boxed_reward, format_reward], weights=[1.0, format_weight]
    )
    return vf.SingleTurnEnv(
        dataset=dataset, rubric=rubric, message_type="chat", **kwargs
    )
