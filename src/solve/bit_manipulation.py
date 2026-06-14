"""Bit-manipulation puzzles: recover a fixed 8-bit transform from input->output examples.

Eight examples under-determine an arbitrary per-bit function (spurious fits), so we
search a *structured* DSL of bit operations instead — single ops (rotate/shift/NOT/
reverse/identity) and pairwise combinations (XOR/AND/OR of two such ops). Structured
rules almost never coincidentally fit eight examples. We predict only when every
DSL program consistent with the examples agrees on the query output (else abstain).
"""

from __future__ import annotations

import re
from collections.abc import Callable

_EX = re.compile(r"([01]{8})\s*->\s*([01]{8})")
_TARGET = re.compile(r"output for:\s*([01]{8})", re.IGNORECASE)
_MASK = 0xFF


Op = Callable[[int], int]
Bin = Callable[[int, int], int]


def _reverse(x: int) -> int:
    return int(f"{x:08b}"[::-1], 2)


def _rotl(k: int) -> Op:
    return lambda x: ((x << k) | (x >> (8 - k))) & _MASK


def _rotr(k: int) -> Op:
    return lambda x: ((x >> k) | (x << (8 - k))) & _MASK


def _shl(k: int) -> Op:
    return lambda x: (x << k) & _MASK


def _shr(k: int) -> Op:
    return lambda x: x >> k


def _combine(u1: Op, u2: Op, bo: Bin) -> Op:
    return lambda x: bo(u1(x), u2(x))


def _unary() -> list[tuple[str, Op]]:
    ops: list[tuple[str, Op]] = [
        ("identity", lambda x: x),
        ("NOT", lambda x: ~x & _MASK),
        ("reverse", _reverse),
    ]
    for k in range(1, 8):
        ops += [
            (f"rotate-left-{k}", _rotl(k)),
            (f"rotate-right-{k}", _rotr(k)),
            (f"shift-left-{k}", _shl(k)),
            (f"shift-right-{k}", _shr(k)),
        ]
    return ops


_UNARY = _unary()
_BIN: list[tuple[str, Bin]] = [
    ("XOR", lambda a, b: a ^ b),
    ("AND", lambda a, b: a & b),
    ("OR", lambda a, b: a | b),
]
# (name, program) pairs — single op, or a pairwise combination of two ops.
_PROGRAMS: list[tuple[str, Op]] = list(_UNARY)
for _n1, _u1 in _UNARY:
    for _n2, _u2 in _UNARY:
        for _nb, _bo in _BIN:
            _PROGRAMS.append((f"({_n1}) {_nb} ({_n2})", _combine(_u1, _u2, _bo)))


def _infer(prompt: str) -> tuple[int, str] | None:
    """Return (query output byte, simplest rule name) if uniquely determined, else None."""
    pairs = _EX.findall(prompt)
    target = _TARGET.search(prompt)
    if not pairs or target is None:
        return None
    ins = [int(i, 2) for i, _ in pairs]
    outs = [int(o, 2) for _, o in pairs]
    qi = int(target.group(1), 2)
    preds: set[int] = set()
    rule = None
    for name, f in _PROGRAMS:
        if all(f(a) == b for a, b in zip(ins, outs, strict=True)):
            preds.add(f(qi))
            if rule is None:
                rule = name
    if len(preds) != 1:
        return None
    return preds.pop(), rule or "a fixed bit operation"


def solve_bit_manipulation(prompt: str) -> str | None:
    got = _infer(prompt)
    return f"{got[0]:08b}" if got else None


def reason_bit_manipulation(prompt: str) -> str | None:
    """A correct chain-of-thought trace ending in \\boxed{} for a bit-manipulation puzzle."""
    got = _infer(prompt)
    if got is None:
        return None
    out, rule = got
    return (
        "The transform is a fixed bit operation. Searching shifts, rotations, NOT, "
        f"reverse and their XOR/AND/OR combinations, the rule consistent with every "
        f"example is: {rule}.\n"
        f"Applying it to the query gives {out:08b}.\n"
        f"\\boxed{{{out:08b}}}"
    )
