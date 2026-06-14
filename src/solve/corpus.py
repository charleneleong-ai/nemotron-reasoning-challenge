"""Assemble the SFT corpus by fusing two sources of *correct* reasoning traces.

For each problem:
  - base-correct (correctness == 'true'): keep the base model's own completion.
  - base-wrong but a category solver gets the gold answer: emit a solver trace.
  - otherwise: skip (no trustworthy trace yet — these are the gaps to close).

This is the winner's recipe: imitate correct reasoning, and manufacture it with
deterministic solvers exactly where the base model fails. Output: corpus.jsonl
with {prompt, completion, source, category}.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from src.solve.registry import SOLVERS

csv.field_size_limit(10**8)


def build_corpus(traj_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in traj_rows:
        cat, prompt, gold = r["problem type"], r["prompt"], r["correct answer"].strip()
        if r["correctness"] == "true":
            out.append(_entry(r["id"], prompt, r["generated"], "base", cat))
            continue
        solver = SOLVERS.get(cat)
        if solver and (solver.solve(prompt) or "").strip() == gold:
            trace = solver.reason(prompt)
            if trace is not None:
                out.append(_entry(r["id"], prompt, trace, "solver", cat))
    return out


def _entry(
    pid: str, prompt: str, completion: str, source: str, category: str
) -> dict[str, str]:
    return {
        "id": pid,
        "prompt": prompt,
        "completion": completion,
        "source": source,
        "category": category,
    }


def main(
    traj: Path = Path("data/traj/nemotron_traj.csv"),
    out: Path = Path("data/corpus.jsonl"),
) -> None:
    rows = list(csv.DictReader(traj.open()))
    corpus = build_corpus(rows)
    with out.open("w") as fh:
        for entry in corpus:
            fh.write(json.dumps(entry) + "\n")

    by_source = Counter(e["source"] for e in corpus)
    by_cat = Counter((e["category"], e["source"]) for e in corpus)
    print(f"corpus: {len(corpus)}/{len(rows)} problems covered  -> {out}")
    print(f"  by source: base={by_source['base']}  solver={by_source['solver']}")
    print(f"\n{'category':20s} {'base':>6s} {'solver':>7s} {'total':>6s} {'gap':>5s}")
    totals = Counter(r["problem type"] for r in rows)
    for cat in sorted(totals):
        b, s = by_cat[(cat, "base")], by_cat[(cat, "solver")]
        print(f"{cat:20s} {b:>6d} {s:>7d} {b + s:>6d} {totals[cat] - b - s:>5d}")


if __name__ == "__main__":
    main()
