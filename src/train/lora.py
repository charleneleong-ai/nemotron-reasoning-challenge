"""Train a LoRA adapter via Unsloth when importable, else PEFT + TRL SFTTrainer."""

from pathlib import Path

import torch
from datasets import Dataset

from src.config.schemas import ExperimentConfig
from src.data.puzzles import load_puzzles, split_puzzles, to_sft_record
from src.logger import get_logger

logger = get_logger(__name__)

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _target_modules() -> list[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def train_adapter(cfg: ExperimentConfig) -> Path:
    """Run SFT and write a LoRA adapter to cfg.train.output_dir. Returns that path."""
    puzzles = load_puzzles(cfg.data)
    train_puzzles, _ = split_puzzles(puzzles, cfg.data)
    dataset = Dataset.from_list([to_sft_record(p) for p in train_puzzles])
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from unsloth import FastLanguageModel  # noqa: F401

        logger.info("Unsloth available — using FastLanguageModel path")
        return _train_unsloth(cfg, dataset, out_dir)
    except ImportError:
        logger.info("Unsloth not found — using PEFT + TRL path")
        return _train_peft(cfg, dataset, out_dir)


def _train_peft(cfg: ExperimentConfig, dataset: Dataset, out_dir: Path) -> Path:
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_kwargs: dict = {"torch_dtype": _DTYPES[cfg.model.dtype]}
    if cfg.model.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_DTYPES[cfg.model.dtype],
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model.hf_id, **model_kwargs)

    peft_config = LoraConfig(
        r=cfg.train.lora_rank,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=_target_modules(),
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_epochs,
        max_steps=cfg.train.max_steps,
        max_length=cfg.model.max_seq_length,
        dataset_text_field="text",
        logging_steps=1,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def _train_unsloth(cfg: ExperimentConfig, dataset: Dataset, out_dir: Path) -> Path:
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.hf_id,
        max_seq_length=cfg.model.max_seq_length,
        dtype=_DTYPES[cfg.model.dtype],
        load_in_4bit=cfg.model.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.train.lora_rank,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=_target_modules(),
    )
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_epochs,
        max_steps=cfg.train.max_steps,
        max_length=cfg.model.max_seq_length,
        dataset_text_field="text",
        logging_steps=1,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
