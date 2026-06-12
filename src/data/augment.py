"""Generate synthetic <think> CoT reasoning traces for puzzles (hybrid CoT pipeline).

The external (Gemini) half: given a puzzle + its gold answer, produce the step-by-step
reasoning that derives the answer. The trace fills the `<think>...</think>` block that the
trainer wraps as `<think>{trace}</think>\\boxed{{answer}}`. The base-model rejection-sampling
half runs on Kaggle; `merge_cot` prefers those on-distribution traces over Gemini's.
"""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from src.data.puzzles import Puzzle
from src.logger import get_logger

logger = get_logger(__name__)

# An async "prompt -> text" function; injectable so tests don't need the API.
CoTGenerator = Callable[[str], Awaitable[str]]

PROMPT = (
    "You are shown a logical-reasoning puzzle and its correct final answer. Explain, step "
    "by step, the hidden transformation rule and how applying it yields that answer. Be "
    "concise and concrete. Do NOT output the final answer or any \\boxed{{...}} — only the "
    "reasoning.\n\nPuzzle:\n{prompt}\n\nCorrect answer: {answer}\n\nStep-by-step reasoning:"
)

_THINK_TAGS = re.compile(r"</?think>", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{[^}]*\}")


def build_cot_prompt(puzzle: Puzzle) -> str:
    return PROMPT.format(prompt=puzzle.prompt, answer=puzzle.answer)


def clean_think(text: str) -> str:
    """Strip any think tags / boxed answers the model emitted; keep only the reasoning."""
    text = _THINK_TAGS.sub("", text)
    text = _BOXED.sub("", text)
    return text.strip()


async def generate_cot(
    puzzles: list[Puzzle],
    generate_fn: CoTGenerator,
    concurrency: int = 8,
    on_result: Callable[[Puzzle, str], None] | None = None,
) -> dict[str, str]:
    """Map puzzle id -> cleaned reasoning trace. Failed/empty generations are omitted.

    `on_result(puzzle, think)` fires as each trace lands — pass a jsonl writer for
    crash-resilient, resumable runs over large sets.
    """
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, str] = {}

    async def _one(p: Puzzle) -> None:
        async with sem:
            try:
                raw = await generate_fn(build_cot_prompt(p))
            except Exception as e:  # noqa: BLE001 — one bad call shouldn't kill the batch
                logger.warning("cot gen failed for %s: %s", p.id, e)
                return
        think = clean_think(raw)
        if think:
            out[p.id] = think
            if on_result is not None:
                on_result(p, think)

    await asyncio.gather(*(_one(p) for p in puzzles))
    return out


def load_done_ids(path: Path) -> set[str]:
    """Ids already present in a CoT jsonl — skip these to resume a partial run."""
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["id"])
        for line in path.read_text().splitlines()
        if line.strip()
    }


def gemini_generator(model: str, api_key: str) -> CoTGenerator:
    """Build an async generate_fn backed by the google-genai SDK."""
    from google import genai

    client = genai.Client(api_key=api_key)

    async def _gen(prompt: str) -> str:
        resp = await client.aio.models.generate_content(model=model, contents=prompt)
        return resp.text or ""

    return _gen


def write_cot(puzzles: list[Puzzle], cot: dict[str, str], path: Path) -> int:
    """Write {id, prompt, answer, think} jsonl for puzzles that got a trace. Returns count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for p in puzzles:
            if p.id in cot:
                fh.write(
                    json.dumps(
                        {
                            "id": p.id,
                            "prompt": p.prompt,
                            "answer": p.answer,
                            "think": cot[p.id],
                        }
                    )
                    + "\n"
                )
                n += 1
    return n


def merge_cot(*sources: dict[str, str]) -> dict[str, str]:
    """Merge id -> trace maps; earlier sources win (pass self-distilled traces first)."""
    merged: dict[str, str] = {}
    for src in reversed(sources):
        merged.update(src)
    return merged
