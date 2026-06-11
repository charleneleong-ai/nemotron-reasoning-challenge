import pytest

from src.eval.boxed import extract_boxed, score


class TestExtractBoxed:
    @pytest.mark.parametrize(
        "text,expected",
        [
            (r"so \boxed{42}.", "42"),
            (r"\boxed{3.14} is pi", "3.14"),
            (r"nested \boxed{x = \frac{1}{2}}", r"x = \frac{1}{2}"),
            ("no box here", None),
            (r"\boxed{}", ""),
        ],
    )
    def test_extract(self, text, expected):
        assert extract_boxed(text) == expected


class TestScore:
    @pytest.mark.parametrize(
        "pred,gold,ok",
        [
            ("42", "42", True),  # exact string
            (" 42 ", "42", True),  # whitespace-insensitive
            ("3.140", "3.14", True),  # numeric equal
            ("3.145", "3.14", True),  # diff 0.005 < tol
            ("3.15", "3.14", True),  # diff 0.010 == tol (inclusive)
            ("3.16", "3.14", False),  # diff 0.020 > tol
            ("cat", "dog", False),  # non-numeric mismatch
            (None, "42", False),  # missing box
        ],
    )
    def test_score(self, pred, gold, ok):
        assert score(pred, gold, tolerance=1e-2) is ok
