"""Load a base model + trained adapter and evaluate boxed accuracy."""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.schemas import ExperimentConfig
from src.data.puzzles import build_inference_prompt, load_puzzles, split_puzzles
from src.eval.boxed import evaluate
from src.logger import get_logger

logger = get_logger(__name__)
_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def run_eval(cfg: ExperimentConfig, adapter_dir: Path) -> dict[str, float | int]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_id, torch_dtype=_DTYPES[cfg.model.dtype]
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    def generate_fn(prompt: str) -> str:
        text = build_inference_prompt(prompt, tokenizer)
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=cfg.eval.max_new_tokens, do_sample=False
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    _, dev = split_puzzles(load_puzzles(cfg.data), cfg.data)
    result = evaluate(dev, generate_fn, tolerance=cfg.eval.tolerance)

    out_dir = Path(cfg.eval.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    logger.info(
        "boxed accuracy %.3f (%d/%d)",
        result["accuracy"],
        result["correct"],
        result["n"],
    )
    return result
