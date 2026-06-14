"""Tests for category solvers, verbalizers, the scoreboard, and corpus assembly."""

import pytest

from src.solve.cipher import solve_cipher
from src.solve.corpus import build_corpus
from src.solve.grade import grade
from src.solve.gravity import reason_gravity, solve_gravity
from src.solve.numeral import reason_numeral, solve_numeral, to_roman
from src.solve.unit_conversion import reason_unit_conversion, solve_unit_conversion

_NUMERAL_PROMPT = (
    "In Alice's Wonderland, numbers are secretly converted into a different "
    "numeral system. Some examples are given below:\n11 -> XI\n"
    "Now, write the number 38 in the Wonderland numeral system."
)


@pytest.mark.parametrize(
    "n,roman",
    [(38, "XXXVIII"), (67, "LXVII"), (100, "C"), (4, "IV"), (1944, "MCMXLIV")],
)
def test_to_roman(n: int, roman: str) -> None:
    assert to_roman(n) == roman


def test_solve_numeral_extracts_target() -> None:
    assert solve_numeral(_NUMERAL_PROMPT) == "XXXVIII"


def test_solve_numeral_no_target_returns_none() -> None:
    assert solve_numeral("no target number here") is None


def test_grade_scores_only_known_categories() -> None:
    rows = [
        {"category": "numeral", "prompt": _NUMERAL_PROMPT, "answer": "XXXVIII"},
        {"category": "numeral", "prompt": _NUMERAL_PROMPT, "answer": "WRONG"},
        {
            "category": "bit_manipulation",
            "prompt": "...",
            "answer": "x",
        },  # no solver yet
    ]
    results = grade(rows)
    assert results["numeral"] == (1, 2)
    assert "bit_manipulation" not in results


_GRAVITY_PROMPT = (
    "In Alice's Wonderland, the gravitational constant has been secretly changed.\n"
    "For t = 1.37s, distance = 14.92 m\nFor t = 4.27s, distance = 144.96 m\n"
    "Now, determine the falling distance for t = 4.41s given d = 0.5*g*t^2."
)
_UNIT_PROMPT = (
    "In Alice's Wonderland, a secret unit conversion is applied to measurements.\n"
    "10.08 m becomes 6.69\n17.83 m becomes 11.83\n"
    "Now, convert the following measurement: 25.09 m"
)


@pytest.mark.parametrize(
    "reason,solve,prompt",
    [
        (reason_numeral, solve_numeral, _NUMERAL_PROMPT),
        (reason_gravity, solve_gravity, _GRAVITY_PROMPT),
        (reason_unit_conversion, solve_unit_conversion, _UNIT_PROMPT),
    ],
)
def test_verbalizer_boxes_the_solver_answer(reason, solve, prompt) -> None:
    """Every trace must end in \\boxed{} carrying the solver's own answer."""
    trace = reason(prompt)
    assert trace.rstrip().endswith(f"\\boxed{{{solve(prompt)}}}")


_CIPHER_PROMPT = (
    "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
    "ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley\n"
    "pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle\n"
    "gbcpovb tqorbog bxo zrswtrj pffq -> student creates the magical door\n"
    "bxo sfjpov pqrsfv dfjjfig -> the golden dragon follows\n"
    "nqwvtogg qorpg bxo zegboqwfcg gotqob -> princess reads the mysterious secret\n"
    "Now, decrypt the following text: trb wzrswvog hffk"
)


def test_solve_cipher_infers_unseen_letters_via_vocab() -> None:
    """'book' needs letters absent from the examples — vocab matching recovers them."""
    assert solve_cipher(_CIPHER_PROMPT) == "cat imagines book"


def test_solve_cipher_no_target_returns_none() -> None:
    assert solve_cipher("no decrypt instruction here -> mapping") is None


def test_build_corpus_source_selection() -> None:
    rows = [
        # base correct -> keep base completion
        {
            "problem type": "numeral",
            "prompt": _NUMERAL_PROMPT,
            "correct answer": "XXXVIII",
            "generated": "base trace \\boxed{XXXVIII}",
            "correctness": "true",
        },
        # base wrong, solver correct -> solver trace
        {
            "problem type": "numeral",
            "prompt": _NUMERAL_PROMPT,
            "correct answer": "XXXVIII",
            "generated": "wrong",
            "correctness": "false",
        },
        # base wrong, no solver -> skipped
        {
            "problem type": "cipher",
            "prompt": "...",
            "correct answer": "x",
            "generated": "wrong",
            "correctness": "false",
        },
    ]
    corpus = build_corpus(rows)
    assert [e["source"] for e in corpus] == ["base", "solver"]
    assert corpus[0]["completion"] == "base trace \\boxed{XXXVIII}"
    assert corpus[1]["completion"].endswith("\\boxed{XXXVIII}")
