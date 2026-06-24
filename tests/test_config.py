import pytest
from pydantic import ValidationError

from src.config.loader import load_experiment_config
from src.config.schemas import TrainConfig


class TestTrainConfigRankGuard:
    def test_rank_32_accepted(self):
        cfg = TrainConfig(preset="smoke", lora_rank=32, lora_alpha=32)
        assert cfg.lora_rank == 32

    @pytest.mark.parametrize("bad_rank", [33, 64, 128])
    def test_rank_above_32_rejected(self, bad_rank):
        with pytest.raises(ValidationError):
            TrainConfig(preset="smoke", lora_rank=bad_rank, lora_alpha=bad_rank)


class TestLoader:
    def test_default_compose_is_proxy_smoke(self):
        cfg = load_experiment_config([])
        assert cfg.model.kind == "proxy"
        assert cfg.train.preset == "smoke"
        assert cfg.train.lora_rank <= 32

    def test_override_to_nemotron(self):
        cfg = load_experiment_config(["model=nemotron_nano", "train=lora_a100"])
        assert cfg.model.kind == "nemotron"
        assert cfg.model.hf_id.startswith("nvidia/Nemotron-3-Nano")
        assert cfg.train.lora_rank <= 32
