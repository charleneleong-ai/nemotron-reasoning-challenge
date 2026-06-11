from pathlib import Path

import pytest

from src.config.schemas import DataConfig
from src.data.puzzles import Puzzle, load_puzzles, split_puzzles, to_sft_record

FIXTURE = Path(__file__).parent / "fixtures" / "mini_problems.jsonl"


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


class TestSftRecord:
    def test_answer_wrapped_in_boxed(self):
        rec = to_sft_record(Puzzle(id="x", prompt="Q?", answer="42"))
        assert r"\boxed{42}" in rec["text"]
        assert "Q?" in rec["text"]
