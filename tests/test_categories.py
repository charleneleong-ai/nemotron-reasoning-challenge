"""Tests for puzzle category classification."""

import csv
from pathlib import Path

import pytest

from src.data.categories import CATEGORIES, classify, label_rows

_EXAMPLES = {
    "bit_manipulation": "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
    "gravity": "In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some examples",
    "unit_conversion": "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example: 1.0 m becomes",
    "numeral": "In Alice's Wonderland, numbers are secretly converted into a different numeral system.",
    "cipher": "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:",
    "equation_numeric": "In Alice's Wonderland, a secret set of transformation rules is applied to equations.",
}


@pytest.mark.parametrize("category,prompt", _EXAMPLES.items())
def test_classify_each_category(category: str, prompt: str) -> None:
    assert classify(prompt) == category


def test_unknown_prompt_returns_none() -> None:
    assert classify("Something entirely unrelated to the puzzles.") is None


def test_label_rows_adds_category() -> None:
    rows = [{"id": "1", "prompt": _EXAMPLES["cipher"], "answer": "x"}]
    assert label_rows(rows)[0]["category"] == "cipher"


@pytest.mark.skipif(
    not Path("data/train.csv").exists(), reason="competition data not present"
)
def test_full_dataset_fully_classified() -> None:
    """Every real prompt must classify into a known category — no 'unknown'."""
    rows = list(csv.DictReader(Path("data/train.csv").open()))
    labels = [classify(r["prompt"]) for r in rows]
    assert None not in labels
    assert set(labels) == set(CATEGORIES)
