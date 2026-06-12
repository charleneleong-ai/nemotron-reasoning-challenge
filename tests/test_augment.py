import asyncio
import json

import pytest

from src.data.augment import (
    build_cot_prompt,
    clean_think,
    generate_cot,
    merge_cot,
    write_cot,
)
from src.data.puzzles import Puzzle

PUZZLES = [Puzzle("a", "Double 6.", "12"), Puzzle("b", "XOR 0 with 1.", "1")]


class TestPrompt:
    def test_includes_puzzle_and_answer(self):
        p = build_cot_prompt(PUZZLES[0])
        assert "Double 6." in p and "12" in p


class TestCleanThink:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("  reason here  ", "reason here"),
            ("<think>step a</think>", "step a"),
            (r"derive it \boxed{12} done", "derive it  done"),
            (r"<think>x</think> then \boxed{1}", "x then"),
        ],
    )
    def test_strips_tags_and_boxed(self, raw, expected):
        assert clean_think(raw) == expected


class TestGenerateCot:
    def test_maps_id_to_cleaned_trace(self):
        async def fake(prompt):
            return r"<think>because</think> \boxed{x}"

        out = asyncio.run(generate_cot(PUZZLES, fake, concurrency=2))
        assert out == {"a": "because", "b": "because"}

    def test_failed_generation_is_omitted(self):
        async def flaky(prompt):
            if "XOR" in prompt:
                raise RuntimeError("boom")
            return "ok trace"

        out = asyncio.run(generate_cot(PUZZLES, flaky))
        assert set(out) == {"a"} and out["a"] == "ok trace"

    def test_empty_trace_is_omitted(self):
        async def empty(prompt):
            return r"<think></think>\boxed{12}"

        assert asyncio.run(generate_cot(PUZZLES[:1], empty)) == {}


class TestWriteAndMerge:
    def test_write_cot_only_traced_puzzles(self, tmp_path):
        path = tmp_path / "cot.jsonl"
        n = write_cot(PUZZLES, {"a": "trace a"}, path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert n == 1
        assert rows == [
            {"id": "a", "prompt": "Double 6.", "answer": "12", "think": "trace a"}
        ]

    def test_merge_prefers_earlier_source(self):
        self_distilled = {"a": "from base"}
        gemini = {"a": "from gemini", "b": "from gemini"}
        assert merge_cot(self_distilled, gemini) == {
            "a": "from base",
            "b": "from gemini",
        }
