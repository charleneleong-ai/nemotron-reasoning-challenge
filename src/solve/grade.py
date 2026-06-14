"""Per-category solver scoreboard.

Runs each registered category solver over its slice of the labeled dataset and
reports accuracy (solver_answer == gold). This grades the *solvers* before any
GPU is spent — a category at high accuracy is ready to generate SFT traces.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from src.solve.registry import SOLVERS


def grade(rows: list[dict[str, str]]) -> dict[str, tuple[int, int]]:
    """Return {category: (correct, total)} over rows whose category has a solver."""
    score: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        solver = SOLVERS.get(row["category"])
        if solver is None:
            continue
        score[row["category"]][1] += 1
        if (solver.solve(row["prompt"]) or "").strip() == row["answer"].strip():
            score[row["category"]][0] += 1
    return {cat: (c, t) for cat, (c, t) in score.items()}


def main(data: Path = Path("data/train_labeled.csv")) -> None:
    rows = list(csv.DictReader(data.open()))
    results = grade(rows)
    print(f"{'category':20s} {'acc':>8s}  correct/total")
    for cat in sorted(results):
        correct, total = results[cat]
        print(f"{cat:20s} {correct / total:>7.1%}  {correct}/{total}")


if __name__ == "__main__":
    main()
