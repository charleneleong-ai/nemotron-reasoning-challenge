"""Produce a hybrid cot.jsonl: verified solver reasoning where we can, else the
existing trace.

The SFT pipeline consumes {id, prompt, answer, think} (think is reasoning only;
the trainer appends the boxed answer). We keep every existing row and overwrite
`think` with:
  1. a deterministic solver's trace when that solver reproduces the gold answer, else
  2. a gold-conditioned oracle trace (e.g. cryptarithm/equation) keyed by id, else
  3. the existing generic trace.
Optionally hold out a val id list so the training corpus stays leak-free.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.categories import classify
from src.solve.registry import SOLVERS, matches

# Oracle preamble lines that are meta-commentary, not reasoning to imitate.
_ORACLE_DROP = ("CRYPTARITHM_ORACLE", "This trace is gold-conditioned", "Final answer:")


def _think(trace: str) -> str:
    """Reasoning without the trailing \\boxed{} (the trainer adds it from `answer`)."""
    return trace.split("\\boxed{")[0].rstrip()


def _clean_oracle(completion: str) -> str:
    """Strip the oracle meta preamble and boxed/final-answer tail, keeping the reasoning."""
    kept = [
        ln
        for ln in completion.splitlines()
        if not any(ln.strip().startswith(d) for d in _ORACLE_DROP)
    ]
    return _think("\n".join(kept)).strip()


def load_oracle(path: Path) -> dict[str, str]:
    """id -> cleaned reasoning, from a {id, completion} oracle-traces jsonl (empty if absent)."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.open():
        t = json.loads(line)
        out[str(t["id"])] = _clean_oracle(t["completion"])
    return out


def upgrade(
    rows: list[dict[str, str]], oracle: dict[str, str] | None = None
) -> tuple[int, int]:
    """Overwrite `think` with a solver trace, else a gold oracle trace. Returns (solver, oracle)."""
    oracle = oracle or {}
    solver_n = oracle_n = 0
    for row in rows:
        solver = SOLVERS.get(classify(row["prompt"]) or "")
        if solver and matches(solver.solve(row["prompt"]), str(row["answer"])):
            trace = solver.reason(row["prompt"])
            if trace is not None:
                row["think"] = _think(trace)
                solver_n += 1
                continue
        if oracle_think := oracle.get(str(row.get("id"))):
            row["think"] = oracle_think
            oracle_n += 1
    return solver_n, oracle_n


def main(
    cot: Path = Path("data/cot.jsonl"),
    oracle_path: Path = Path("data/oracle_reasoning_traces.jsonl"),
    val_ids_path: Path = Path("data/val_ids.txt"),
    out: Path = Path("data/cot_hybrid.jsonl"),
    exclude_val: bool = False,
) -> None:
    rows = [json.loads(line) for line in cot.open()]
    oracle = load_oracle(oracle_path)
    solver_n, oracle_n = upgrade(rows, oracle)

    if exclude_val and val_ids_path.exists():
        val = set(val_ids_path.read_text().split())
        kept = [r for r in rows if str(r["id"]) not in val]
    else:
        kept = rows

    with out.open("w") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")
    held = len(rows) - len(kept)
    print(
        f"cot: {len(kept)} rows ({solver_n} solver + {oracle_n} oracle upgraded), "
        f"{held} val held out -> {out}"
    )


if __name__ == "__main__":
    main()
