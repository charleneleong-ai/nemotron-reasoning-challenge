"""Tests for hybrid cot.jsonl assembly."""

from src.solve.export_cot import _think, upgrade

_NUMERAL_PROMPT = (
    "In Alice's Wonderland, numbers are secretly converted into a different "
    "numeral system. Some examples are given below:\n11 -> XI\n"
    "Now, write the number 38 in the Wonderland numeral system."
)


def test_think_strips_boxed_tail() -> None:
    assert _think("reasoning here\n\\boxed{XXXVIII}") == "reasoning here"


def test_upgrade_replaces_solvable_keeps_rest() -> None:
    rows = [
        {"prompt": _NUMERAL_PROMPT, "answer": "XXXVIII", "think": "old generic"},
        {"prompt": "unrelated non-puzzle text", "answer": "?", "think": "keep me"},
    ]
    assert upgrade(rows) == 1
    assert "Roman" in rows[0]["think"] and "\\boxed" not in rows[0]["think"]
    assert rows[1]["think"] == "keep me"


def test_upgrade_skips_when_solver_answer_mismatches_gold() -> None:
    rows = [{"prompt": _NUMERAL_PROMPT, "answer": "WRONG", "think": "untouched"}]
    assert upgrade(rows) == 0
    assert rows[0]["think"] == "untouched"
