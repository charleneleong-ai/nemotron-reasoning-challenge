"""Pydantic schemas for the Hydra-composed experiment config."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["proxy", "nemotron"]
    hf_id: str
    load_in_4bit: bool = False
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    max_seq_length: int = Field(2048, gt=0)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    prompt_field: str = "prompt"
    answer_field: str = "answer"
    # Optional CoT jsonl ({id, think}); when set, traces fill the <think> block in SFT.
    cot_path: str | None = None
    eval_fraction: float = Field(0.1, gt=0.0, lt=1.0)
    seed: int = 42
    max_samples: int | None = Field(None, gt=0)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: Literal["smoke", "qlora_t4", "lora_a100"]
    lora_rank: int = Field(16, gt=0, le=32)
    lora_alpha: int = Field(32, gt=0)
    lora_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    learning_rate: float = Field(2e-4, gt=0.0)
    num_epochs: int = Field(1, gt=0)
    max_steps: int = Field(-1)
    batch_size: int = Field(1, gt=0)
    grad_accum: int = Field(1, gt=0)
    output_dir: str = "adapters/run"
    lora_target_modules: list[str] | str = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tolerance: float = Field(1e-2, ge=0.0)
    max_new_tokens: int = Field(512, gt=0)
    output_dir: str = "outputs/eval"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
