"""Gravity puzzles: d = 0.5*g*t^2 with a secret g inferred from observations.

Each example (t, d) is rounded to 2 dp, so we recover g by least-squares through
the origin (d = a*t^2, a = 0.5*g) — averaging out the rounding — then evaluate at
the target t and round to 2 dp.
"""

from __future__ import annotations

import re

_OBS = re.compile(r"t\s*=\s*([\d.]+)\s*s,\s*distance\s*=\s*([\d.]+)\s*m", re.IGNORECASE)
_TARGET = re.compile(r"for\s+t\s*=\s*([\d.]+)\s*s\s+given", re.IGNORECASE)


def _fit(prompt: str) -> tuple[float, float] | None:
    """Return (a, target_t) where a = 0.5*g from least-squares, or None."""
    obs = [(float(t), float(d)) for t, d in _OBS.findall(prompt)]
    target = _TARGET.search(prompt)
    if not obs or target is None:
        return None
    den = sum(t**4 for t, _ in obs)
    if den == 0:
        return None
    a = sum(d * t**2 for t, d in obs) / den
    return a, float(target.group(1))


def solve_gravity(prompt: str) -> str | None:
    fit = _fit(prompt)
    if fit is None:
        return None
    a, t = fit
    return str(round(a * t**2, 2))


def reason_gravity(prompt: str) -> str | None:
    """A correct chain-of-thought trace ending in \\boxed{} for a gravity puzzle."""
    fit = _fit(prompt)
    if fit is None:
        return None
    a, t = fit
    return (
        f"The relation is d = 0.5*g*t^2. Fitting g to the observations gives "
        f"g ≈ {2 * a:.4f}.\n"
        f"For t = {t}: d = 0.5*{2 * a:.4f}*{t}^2 = {round(a * t**2, 2)}.\n"
        f"\\boxed{{{round(a * t**2, 2)}}}"
    )
