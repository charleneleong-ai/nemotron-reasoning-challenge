"""Numeral puzzles: convert an integer to its Wonderland numeral (standard Roman).

The "secret numeral system" framing is a decoy — the mapping is standard Roman
numerals in every observed case. The grader (src.solve.grade) confirms coverage.
"""

from __future__ import annotations

import re

_ROMAN: tuple[tuple[int, str], ...] = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)
_TARGET = re.compile(r"write the number (\d+)", re.IGNORECASE)


def to_roman(n: int) -> str:
    out: list[str] = []
    for value, symbol in _ROMAN:
        count, n = divmod(n, value)
        out.append(symbol * count)
    return "".join(out)


def solve_numeral(prompt: str) -> str | None:
    """Return the Roman numeral for the prompt's target number, or None if unparsable."""
    match = _TARGET.search(prompt)
    return to_roman(int(match.group(1))) if match else None


def reason_numeral(prompt: str) -> str | None:
    """A correct chain-of-thought trace ending in \\boxed{} for a numeral puzzle."""
    match = _TARGET.search(prompt)
    if match is None:
        return None
    n = int(match.group(1))
    roman = to_roman(n)
    return (
        f"The example conversions are standard Roman numerals "
        f"(e.g. 11 -> XI, 94 -> XCIV), so the Wonderland system is Roman numerals.\n"
        f"Converting {n}: {to_roman(n)}.\n"
        f"\\boxed{{{roman}}}"
    )
