"""Classify Wonderland puzzle prompts into their algorithmic category.

The competition's training puzzles fall into 6 deterministic categories, each
announced by a fixed opening phrase. The raw csv has no label column, so a
substring match recovers it — which lets us route each puzzle to a category
solver that generates a correct chain-of-thought trace for SFT.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

# Category -> a phrase that appears verbatim in every prompt of that category.
_SIGNATURES: dict[str, str] = {
    "bit_manipulation": "secret bit manipulation rule",
    "gravity": "gravitational constant has been secretly changed",
    "unit_conversion": "secret unit conversion is applied to measurements",
    "numeral": "converted into a different numeral system",
    "cipher": "secret encryption rules are used on text",
    "equation_numeric": "transformation rules is applied to equations",
}

CATEGORIES: tuple[str, ...] = tuple(_SIGNATURES)


def classify(prompt: str) -> str | None:
    """Return the puzzle category for `prompt`, or None if no signature matches."""
    for category, phrase in _SIGNATURES.items():
        if phrase in prompt:
            return category
    return None


def label_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add a `category` field to each row (id, prompt, answer) in place."""
    for row in rows:
        row["category"] = classify(row["prompt"]) or "unknown"
    return rows


def label_dataset(
    src: Path = Path("data/train.csv"),
    dst: Path = Path("data/train_labeled.csv"),
) -> Counter[str]:
    """Write `src` plus a `category` column to `dst`; return the category counts."""
    rows = label_rows(list(csv.DictReader(src.open())))
    fieldnames = list(rows[0]) if rows else ["id", "prompt", "answer", "category"]
    with dst.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return Counter(row["category"] for row in rows)


if __name__ == "__main__":
    counts = label_dataset()
    total = sum(counts.values())
    for category, n in counts.most_common():
        print(f"{n:5d}  {category}")
    print(f"{total:5d}  TOTAL  ({counts.get('unknown', 0)} unknown)")
