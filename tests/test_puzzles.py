from pathlib import Path

import pytest

from src.config.schemas import DataConfig
from src.data.puzzles import (
    Puzzle,
    build_inference_prompt,
    format_target,
    load_puzzles,
    split_puzzles,
    to_sft_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_problems.jsonl"


@pytest.fixture(scope="module")
def proxy_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")


@pytest.fixture
def cfg() -> DataConfig:
    return DataConfig(path=str(FIXTURE), eval_fraction=0.25, seed=0)


class TestLoad:
    def test_loads_all_rows(self, cfg):
        puzzles = load_puzzles(cfg)
        assert len(puzzles) == 4
        assert all(isinstance(p, Puzzle) for p in puzzles)
        assert puzzles[0].answer == "12"

    def test_max_samples_caps(self, cfg):
        cfg.max_samples = 2
        assert len(load_puzzles(cfg)) == 2

    def test_custom_field_names(self, tmp_path):
        f = tmp_path / "alt.jsonl"
        f.write_text('{"id": "a", "q": "Q?", "sol": "9"}\n')
        c = DataConfig(path=str(f), prompt_field="q", answer_field="sol")
        p = load_puzzles(c)[0]
        assert p.prompt == "Q?" and p.answer == "9"

    def test_loads_csv_preserving_leading_zeros(self, tmp_path):
        f = tmp_path / "mini.csv"
        f.write_text(
            "id,prompt,answer\n"
            "00066667,Solve X,01000011\n"
            "000b53cf,Solve Y,cat imagines book\n"
        )
        c = DataConfig(path=str(f), prompt_field="prompt", answer_field="answer")
        puzzles = load_puzzles(c)
        assert [p.id for p in puzzles] == ["00066667", "000b53cf"]
        assert puzzles[0].answer == "01000011"  # not coerced to int 1000011
        assert puzzles[1].answer == "cat imagines book"


class TestSplit:
    def test_split_is_disjoint_and_complete(self, cfg):
        train, dev = split_puzzles(load_puzzles(cfg), cfg)
        assert len(dev) == 1  # 25% of 4
        assert len(train) == 3
        ids = {p.id for p in train} | {p.id for p in dev}
        assert len(ids) == 4

    def test_split_is_seed_stable(self, cfg):
        a = split_puzzles(load_puzzles(cfg), cfg)[1]
        b = split_puzzles(load_puzzles(cfg), cfg)[1]
        assert [p.id for p in a] == [p.id for p in b]


class TestFormatTarget:
    def test_thinking_mode_and_boxed(self):
        out = format_target("42")
        assert "<think>" in out and "</think>" in out
        assert r"\boxed{42}" in out


@pytest.mark.slow
class TestChatFormat:
    def test_sft_text_has_prompt_and_boxed(self, proxy_tokenizer):
        text = to_sft_text(Puzzle("x", "What is 2+2?", "4"), proxy_tokenizer)
        assert "What is 2+2?" in text
        assert r"\boxed{4}" in text

    def test_inference_prompt_omits_target(self, proxy_tokenizer):
        text = build_inference_prompt("What is 2+2?", proxy_tokenizer)
        assert "What is 2+2?" in text
        assert r"\boxed{" not in text
