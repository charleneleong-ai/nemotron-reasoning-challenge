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


def matches(pred: str | None, gold: str, rel_tol: float = 1e-2) -> bool:
    """Grade a solver answer like the competition: exact string, or decimal within 1% rel.

    The tolerance path is gated on a decimal point in the answer so it applies only to
    real-valued families (gravity, unit_conversion) — never to binary strings like
    '11111110', which would otherwise float-match '11111111' within 1%.
    """
    if pred is None:
        return False
    if pred.strip() == gold.strip():
        return True
    if "." not in gold:
        return False
    try:
        p, g = float(pred), float(gold)
    except ValueError:
        return False
    return abs(p - g) <= rel_tol * max(abs(g), 1e-9) + 1e-9


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
