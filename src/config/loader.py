"""Hydra config composer returning a validated ExperimentConfig."""

from typing import Any

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.config.schemas import ExperimentConfig


def load_experiment_config(
    overrides: list[str] | None = None,
    config_name: str = "experiment",
    config_path: str = "../../config",
) -> ExperimentConfig:
    """Compose Hydra config + validate. Raises ValidationError on bad fields (e.g. rank>32)."""
    with initialize(version_base=hydra.__version__, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides or [])
        cfg_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
        return ExperimentConfig(**cfg_dict)
