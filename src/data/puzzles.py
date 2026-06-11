"""Load reasoning puzzles and format them via the model's chat template."""

import csv
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from src.config.schemas import DataConfig


def format_target(answer: str) -> str:
    """Assistant turn content in thinking mode: empty think block + boxed answer."""
    return f"<think>\n\n</think>\n\n\\boxed{{{answer}}}"


@dataclass(frozen=True)
class Puzzle:
    id: str
    prompt: str
    answer: str


def to_sft_text(puzzle: Puzzle, tokenizer: Any) -> str:
    """Full SFT training text: user turn + assistant turn, via the chat template."""
    messages = [
        {"role": "user", "content": puzzle.prompt},
        {"role": "assistant", "content": format_target(puzzle.answer)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_inference_prompt(prompt: str, tokenizer: Any) -> str:
    """Inference prompt: user turn only, with the generation prompt appended."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _read_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield string-valued rows from a .csv or .jsonl file (leading zeros preserved)."""
    if path.suffix == ".csv":
        with path.open(newline="") as fh:
            yield from csv.DictReader(fh)
    else:
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def load_puzzles(cfg: DataConfig) -> list[Puzzle]:
    rows: Iterator[dict[str, str]] = _read_rows(Path(cfg.path))
    if cfg.max_samples is not None:
        rows = islice(rows, cfg.max_samples)
    return [
        Puzzle(
            id=str(row.get("id", i)),
            prompt=str(row[cfg.prompt_field]),
            answer=str(row[cfg.answer_field]),
        )
        for i, row in enumerate(rows)
    ]


def split_puzzles(
    puzzles: list[Puzzle], cfg: DataConfig
) -> tuple[list[Puzzle], list[Puzzle]]:
    order = list(puzzles)
    random.Random(cfg.seed).shuffle(order)
    n_dev = max(1, round(len(order) * cfg.eval_fraction))
    return order[n_dev:], order[:n_dev]
