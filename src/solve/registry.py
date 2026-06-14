"""Single source of truth mapping each puzzle category to its solver.

Adding a new category solver is a one-line edit here — `grade` and `corpus`
both read this registry, so neither needs touching.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.solve import bit_manipulation, cipher, gravity, numeral, unit_conversion


@dataclass(frozen=True)
class Solver:
    """A category's deterministic answer solver and matching trace verbalizer."""

    category: str
    solve: Callable[[str], str | None]
    reason: Callable[[str], str | None]


SOLVERS: dict[str, Solver] = {
    s.category: s
    for s in (
        Solver("numeral", numeral.solve_numeral, numeral.reason_numeral),
        Solver("cipher", cipher.solve_cipher, cipher.reason_cipher),
        Solver(
            "bit_manipulation",
            bit_manipulation.solve_bit_manipulation,
            bit_manipulation.reason_bit_manipulation,
        ),
        Solver("gravity", gravity.solve_gravity, gravity.reason_gravity),
        Solver(
            "unit_conversion",
            unit_conversion.solve_unit_conversion,
            unit_conversion.reason_unit_conversion,
        ),
    )
}
