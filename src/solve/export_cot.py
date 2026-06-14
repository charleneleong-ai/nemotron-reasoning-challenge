"""Produce a hybrid cot.jsonl: verified solver reasoning where we can, else the
existing trace.

The SFT pipeline consumes {id, prompt, answer, think} (think is reasoning only;
the trainer appends the boxed answer). We keep every existing row and overwrite
`think` with a deterministic solver's trace whenever that solver reproduces the
gold answer — replacing possibly-wrong generic traces with guaranteed-correct
ones for the categories we cover.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.categories import classify
from src.solve.registry import SOLVERS


def _think(trace: str) -> str:
    """Reasoning without the trailing \\boxed{} (the trainer adds it from `answer`)."""
    return trace.split("\\boxed{")[0].rstrip()


def upgrade(rows: list[dict[str, str]]) -> int:
    """Overwrite `think` in place with a correct solver trace where available; count upgrades."""
    upgraded = 0
    for row in rows:
        solver = SOLVERS.get(classify(row["prompt"]) or "")
        if solver is None:
            continue
        if (solver.solve(row["prompt"]) or "").strip() != str(row["answer"]).strip():
            continue
        trace = solver.reason(row["prompt"])
        if trace is not None:
            row["think"] = _think(trace)
            upgraded += 1
    return upgraded


def main(
    cot: Path = Path("data/cot.jsonl"),
    out: Path = Path("data/cot_hybrid.jsonl"),
) -> None:
    rows = [json.loads(line) for line in cot.open()]
    upgraded = upgrade(rows)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(
        f"hybrid cot: {len(rows)} rows, {upgraded} upgraded to solver traces -> {out}"
    )


if __name__ == "__main__":
    main()
