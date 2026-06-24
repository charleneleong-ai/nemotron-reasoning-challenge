"""Cipher puzzles: a per-letter substitution recovered from example pairs.

Each example "<cipher> -> <plain>" aligns word-by-word then letter-by-letter to
build a partial cipher->plain map. The plaintext vocabulary is a tiny closed set
(~77 Wonderland words gathered from the examples), so when the target contains
letters the examples never showed, we resolve the remaining letters by matching
each cipher word against that vocabulary by length + letter-repetition pattern +
consistency with the known map, propagating newly-learned letters until fixpoint.
We still abstain (never guess) if anything stays under-determined.
"""

from __future__ import annotations

import re

_TARGET = re.compile(r"decrypt the following text:\s*(.+)", re.IGNORECASE)

# Closed Wonderland plaintext vocabulary (every cipher answer is one of these).
_VOCAB: frozenset[str] = frozenset(
    "above alice ancient around beyond bird book bright castle cat cave chases "
    "clever colorful creates crystal curious dark discovers door dragon draws "
    "dreams explores follows forest found garden golden hatter hidden imagines in "
    "inside island key king knight library magical map message mirror mountain "
    "mouse mysterious near ocean palace potion princess puzzle queen rabbit reads "
    "school secret sees silver story strange student studies teacher the through "
    "tower treasure turtle under valley village watches wise wizard wonderland writes".split()
)


def _pattern(word: str) -> tuple[int, ...]:
    seen: dict[str, int] = {}
    return tuple(seen.setdefault(ch, len(seen)) for ch in word)


def _parse(prompt: str) -> tuple[dict[str, str], set[str]]:
    """Return (partial cipher->plain map, plaintext vocabulary) from the examples."""
    mapping: dict[str, str] = {}
    vocab: set[str] = set(_VOCAB)
    for line in prompt.splitlines():
        if " -> " not in line or line.lower().lstrip().startswith("now"):
            continue
        cipher, plain = line.split(" -> ", 1)
        cwords, pwords = cipher.split(), plain.split()
        vocab.update(pwords)
        if len(cwords) != len(pwords):
            continue
        for cw, pw in zip(cwords, pwords, strict=True):
            if len(cw) == len(pw):
                mapping.update(zip(cw, pw, strict=True))
    return mapping, vocab


def _candidates(
    cword: str, mapping: dict[str, str], reverse: dict[str, str], vocab: set[str]
) -> list[str]:
    """Vocab words matching cword by pattern and consistent with the bijective map."""
    pat = _pattern(cword)
    out = []
    for word in vocab:
        if len(word) != len(cword) or _pattern(word) != pat:
            continue
        # forward: known cipher letters must decode as in `word`;
        # reverse: a plain letter can't be claimed by two cipher letters.
        if all(
            mapping.get(c, p) == p and reverse.get(p, c) == c
            for c, p in zip(cword, word, strict=True)
        ):
            out.append(word)
    return out


def _build_map(prompt: str) -> dict[str, str]:
    """Partial map plus letters inferred from vocab word-matching (to fixpoint)."""
    mapping, vocab = _parse(prompt)
    reverse = {p: c for c, p in mapping.items()}
    target = _TARGET.search(prompt)
    words = target.group(1).split() if target else []
    resolved: set[str] = set()
    changed = True
    while changed:
        changed = False
        for cword in words:
            if cword in resolved:
                continue
            cands = _candidates(cword, mapping, reverse, vocab)
            if len(cands) == 1:
                for c, p in zip(cword, cands[0], strict=True):
                    mapping[c], reverse[p] = p, c
                resolved.add(cword)
                changed = True
    return mapping


def _decrypt(text: str, mapping: dict[str, str]) -> str | None:
    out: list[str] = []
    for ch in text:
        if not ch.isalpha():
            out.append(ch)
        elif ch in mapping:
            out.append(mapping[ch])
        else:
            return None  # under-determined — abstain
    return "".join(out)


def solve_cipher(prompt: str) -> str | None:
    target = _TARGET.search(prompt)
    if target is None:
        return None
    return _decrypt(target.group(1).strip(), _build_map(prompt))


def reason_cipher(prompt: str) -> str | None:
    """A correct chain-of-thought trace ending in \\boxed{} for a cipher puzzle."""
    target = _TARGET.search(prompt)
    if target is None:
        return None
    cipher_text = target.group(1).strip()
    plain = _decrypt(cipher_text, _build_map(prompt))
    if plain is None:
        return None
    return (
        "Each example maps ciphertext to plaintext by a fixed letter substitution. "
        "Aligning the example words letter by letter recovers the cipher alphabet.\n"
        f"Applying it to '{cipher_text}' gives '{plain}'.\n"
        f"\\boxed{{{plain}}}"
    )
