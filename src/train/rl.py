"""Verifiable rewards for the GSPO/GRPO RLVR stage.

The competition answer is machine-checkable (`\\boxed{}` exact / ±1e-2), so RL needs no reward
model — these reward functions ARE the grader. They follow TRL's reward-function signature
(`fn(completions, **cols) -> list[float]`), where dataset columns (e.g. `answer`) arrive as
kwargs. GSPO vs GRPO is a trainer setting (`importance_sampling_level="sequence"`), not a
reward change — see notebooks/kaggle_gspo.ipynb.
"""

import re
from typing import Any

from src.eval.boxed import extract_boxed, score

# Nemotron's chat template prefills "<think>\n" into the prompt, so the completion contains
# only the closing "</think>" — match on that + a boxed answer, not an opening tag.
_THINK_THEN_BOXED = re.compile(r"</think>.*?\\boxed\{.*?\}", re.IGNORECASE | re.DOTALL)


def _text(completion: Any) -> str:
    """Normalise a TRL completion (plain string or conversational [{role, content}, ...])."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return str(completion[-1].get("content", ""))
    return str(completion)


def boxed_reward(completions: list[Any], answer: list[str], **_: Any) -> list[float]:
    """1.0 if the completion's \\boxed{} matches the gold answer (exact / ±1e-2), else 0.0."""
    return [
        1.0 if score(extract_boxed(_text(c)), str(a)) else 0.0
        for c, a in zip(completions, answer, strict=False)
    ]


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    """Small shaping reward: 1.0 if the completion closes </think> then has a \\boxed{}."""
    return [1.0 if _THINK_THEN_BOXED.search(_text(c)) else 0.0 for c in completions]
