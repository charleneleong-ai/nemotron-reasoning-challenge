"""Train a LoRA adapter via Unsloth when importable, else PEFT + TRL SFTTrainer."""

from pathlib import Path

import torch
from datasets import Dataset

from src.config.schemas import ExperimentConfig
from src.data.puzzles import Puzzle, load_puzzles, split_puzzles, to_sft_text
from src.logger import get_logger

logger = get_logger(__name__)

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _build_dataset(puzzles: list[Puzzle], tokenizer: object) -> Dataset:
    return Dataset.from_list(
        [{"id": p.id, "text": to_sft_text(p, tokenizer)} for p in puzzles]
    )


def train_adapter(cfg: ExperimentConfig) -> Path:
    """Run SFT and write a LoRA adapter to cfg.train.output_dir. Returns that path."""
    train_puzzles, _ = split_puzzles(load_puzzles(cfg.data), cfg.data)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from unsloth import FastLanguageModel  # noqa: F401

        logger.info("Unsloth available — using FastLanguageModel path")
        return _train_unsloth(cfg, train_puzzles, out_dir)
    except ImportError:
        logger.info("Unsloth not found — using PEFT + TRL path")
        return _train_peft(cfg, train_puzzles, out_dir)


def _train_peft(
    cfg: ExperimentConfig, train_puzzles: list[Puzzle], out_dir: Path
) -> Path:
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
    dataset = _build_dataset(train_puzzles, tokenizer)

    peft_config = LoraConfig(
        r=cfg.train.lora_rank,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=cfg.train.lora_target_modules,
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


def _resolve_target_modules(
    target_modules: list[str] | str, model: torch.nn.Module
) -> list[str]:
    """Unsloth's get_peft_model needs a concrete list; expand "all-linear" from the model."""
    if target_modules != "all-linear":
        return list(target_modules)
    names = {
        name.split(".")[-1]
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
    }
    names.discard("lm_head")
    return sorted(names)


def _train_unsloth(
    cfg: ExperimentConfig, train_puzzles: list[Puzzle], out_dir: Path
) -> Path:
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.hf_id,
        max_seq_length=cfg.model.max_seq_length,
        dtype=_DTYPES[cfg.model.dtype],
        load_in_4bit=cfg.model.load_in_4bit,
    )
    dataset = _build_dataset(train_puzzles, tokenizer)
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.train.lora_rank,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=_resolve_target_modules(cfg.train.lora_target_modules, model),
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
