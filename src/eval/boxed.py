"""Parse \\boxed{} answers and score against gold (exact or numeric tolerance)."""

from collections.abc import Callable

from src.data.puzzles import Puzzle


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...}, handling nested braces. None if absent."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    out: list[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out)


def _as_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def score(pred: str | None, gold: str, tolerance: float = 1e-2) -> bool:
    """Exact string match (whitespace-insensitive), else numeric within tolerance."""
    if pred is None:
        return False
    p, g = pred.strip(), gold.strip()
    if p == g:
        return True
    pf, gf = _as_float(p), _as_float(g)
    if pf is not None and gf is not None:
        return abs(pf - gf) <= tolerance
    return False


def evaluate(
    puzzles: list[Puzzle],
    generate_fn: Callable[[str], str],
    tolerance: float = 1e-2,
) -> dict[str, float | int]:
    """Run generate_fn over each puzzle prompt, score the boxed answer, return aggregate."""
    correct = 0
    for p in puzzles:
        pred = extract_boxed(generate_fn(p.prompt))
        if score(pred, p.answer, tolerance):
            correct += 1
    n = len(puzzles)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0}
