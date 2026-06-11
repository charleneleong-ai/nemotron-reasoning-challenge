"""Load reasoning puzzles and format them as SFT records with \\boxed{} targets."""

import csv
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from src.config.schemas import DataConfig

SYSTEM_PROMPT = (
    "You are a careful reasoning solver. Identify the underlying transformation "
    "rule, apply it, and give the final answer inside \\boxed{}."
)


def build_prompt(prompt: str) -> str:
    """The system+user+assistant-open prefix shared by training and inference."""
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"


@dataclass(frozen=True)
class Puzzle:
    id: str
    prompt: str
    answer: str


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


def to_sft_record(puzzle: Puzzle) -> dict[str, str]:
    text = f"{build_prompt(puzzle.prompt)}The answer is \\boxed{{{puzzle.answer}}}."
    return {"id": puzzle.id, "text": text}
