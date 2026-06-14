"""Unit-conversion puzzles: a secret linear scaling y = k*x.

Infer k by least-squares through the origin (k = sum(xy)/sum(x^2)) from the
rounded examples, then convert the target and format to 2 dp (gold keeps the
trailing zero, e.g. 19.00).
"""

from __future__ import annotations

import re

_EX = re.compile(r"([\d.]+)\s*m\s+becomes\s+([\d.]+)", re.IGNORECASE)
_TARGET = re.compile(
    r"convert the following measurement:\s*([\d.]+)\s*m", re.IGNORECASE
)


def _fit(prompt: str) -> tuple[float, float] | None:
    """Return (k, target_x) for the linear scaling y = k*x, or None."""
    pairs = [(float(x), float(y)) for x, y in _EX.findall(prompt)]
    target = _TARGET.search(prompt)
    if not pairs or target is None:
        return None
    den = sum(x**2 for x, _ in pairs)
    if den == 0:
        return None
    k = sum(x * y for x, y in pairs) / den
    return k, float(target.group(1))


def solve_unit_conversion(prompt: str) -> str | None:
    fit = _fit(prompt)
    if fit is None:
        return None
    k, x = fit
    return f"{k * x:.2f}"


def reason_unit_conversion(prompt: str) -> str | None:
    """A correct chain-of-thought trace ending in \\boxed{} for a unit-conversion puzzle."""
    fit = _fit(prompt)
    if fit is None:
        return None
    k, x = fit
    return (
        f"The conversion is linear, y = k*x. Fitting the examples gives k ≈ {k:.4f}.\n"
        f"For {x} m: {k:.4f}*{x} = {k * x:.2f}.\n"
        f"\\boxed{{{k * x:.2f}}}"
    )
