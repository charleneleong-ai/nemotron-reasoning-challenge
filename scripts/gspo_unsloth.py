"""Round-two GSPO RLVR on a single A100 80GB via Unsloth (4-bit + colocated vLLM).

Unlike the bf16 TRL path (which OOMs or runs uncached/slow) and Prime hosted (GRPO-only,
opaque rank), Unsloth fits the 30B NemotronH MoE on one card in 4-bit with vLLM sharing the
same quantized weights — so we get fast rollouts AND true GSPO (sequence-level importance
sampling) AND a controllable rank<=32 adapter we can submit to Kaggle directly.

    .venv-unsloth/bin/python scripts/gspo_unsloth.py --max-steps 5 --max-prompts 64   # smoke
    .venv-unsloth/bin/python scripts/gspo_unsloth.py                                   # full

Runs in .venv-unsloth (py3.12, unsloth+vllm) — kept separate from the project .venv (py3.14).
Self-contained: rewards + data loading inlined so it needs no project install.
"""

import csv
import re
import zipfile
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

KAGGLE_MODEL = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
_CLOSED_THEN_BOXED = re.compile(r"</think>.*?\\boxed\{.*?\}", re.IGNORECASE | re.DOTALL)
# Mamba in/out projections + MLP up/down — matches scripts/gspo_a100.py's proven target set.
TARGET_MODULES = ["in_proj", "out_proj", "up_proj", "down_proj"]


def extract_boxed(text: str) -> str | None:
    start = text.rfind(r"\boxed{")
    if start == -1:
        return None
    i, depth, out = start + len(r"\boxed{"), 1, []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out)


def score(pred: str | None, gold: str, tolerance: float = 1e-2) -> bool:
    if pred is None:
        return False
    p, g = pred.strip(), gold.strip()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) <= tolerance
    except (ValueError, TypeError):
        return False


def _text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return str(completion[-1].get("content", ""))
    return str(completion)


def boxed_reward(completions: list[Any], answer: list[str], **_: Any) -> list[float]:
    """1.0 if the completion's \\boxed{} matches gold (exact / ±1e-2), else 0.0."""
    return [
        1.0 if score(extract_boxed(_text(c)), str(a)) else 0.0
        for c, a in zip(completions, answer, strict=False)
    ]


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    """1.0 if the completion closes </think> then emits a \\boxed{} (template prefills <think>)."""
    return [1.0 if _CLOSED_THEN_BOXED.search(_text(c)) else 0.0 for c in completions]


def _load_rows(path: Path, max_prompts: int | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    with path.open(newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if max_prompts is not None and i >= max_prompts:
                break
            out.append({"prompt": r["prompt"], "answer": str(r["answer"])})
    return out


@app.command()
def main(
    model_path: str = typer.Option(
        "", help="Local model path; blank => kagglehub download."
    ),
    data_path: str = typer.Option(
        "data/train.csv", help="Puzzles csv (id,prompt,answer)."
    ),
    out: str = typer.Option("adapters/gspo_unsloth", help="Output adapter dir."),
    lora_rank: int = typer.Option(32, help="LoRA rank (<=32 for Kaggle)."),
    num_generations: int = typer.Option(8, help="GSPO group size G."),
    max_prompts: int = typer.Option(4000, help="Train prompts subset."),
    max_steps: int = typer.Option(500, help="GSPO optimizer steps."),
    max_prompt_len: int = typer.Option(1024),
    max_completion_len: int = typer.Option(
        4096, help="Reasoning needs room for <think>+boxed."
    ),
    max_seq_len: int = typer.Option(6144, help=">= prompt+completion."),
    lr: float = typer.Option(1e-6),
    beta: float = typer.Option(0.04, help="KL to reference (anti-collapse)."),
    gpu_mem_util: float = typer.Option(
        0.6, help="Fraction reserved for vLLM colocate."
    ),
    fast_inference: bool = typer.Option(
        True,
        help="Colocated vLLM rollouts. Off => HF generation (no unsloth<->vllm version lock).",
    ),
) -> None:
    """GSPO (sequence-level IS) on a single A100 via Unsloth 4-bit. Rank<=32 adapter."""
    from unsloth import FastLanguageModel  # noqa: I001 — must import before trl/transformers
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    base = model_path
    if not base:
        import kagglehub

        base = kagglehub.model_download(KAGGLE_MODEL)
    print(f"base model: {base}", flush=True)

    load_kwargs: dict[str, Any] = dict(
        model_name=base,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        trust_remote_code=True,  # NemotronH ships custom modeling code
        fast_inference=fast_inference,
    )
    if fast_inference:  # vLLM-only knobs
        load_kwargs.update(max_lora_rank=lora_rank, gpu_memory_utilization=gpu_mem_util)
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    rows = _load_rows(Path(data_path), max_prompts)
    ds = Dataset.from_list(
        [
            {
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "answer": r["answer"],
            }
            for r in rows
        ]
    )
    print(f"RL prompts: {len(ds)}", flush=True)

    cfg = GRPOConfig(
        output_dir=out,
        importance_sampling_level="sequence",  # GSPO (vs "token" = GRPO)
        use_vllm=fast_inference,
        num_generations=num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        learning_rate=lr,
        beta=beta,
        max_steps=max_steps,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        processing_class=tokenizer,
        reward_funcs=[boxed_reward, format_reward],
        train_dataset=ds,
    )
    trainer.train()

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    zip_path = _package(out_dir)
    print(f"done. adapter -> {out} | submission -> {zip_path}", flush=True)


def _package(adapter_dir: Path, out_zip: Path = Path("submission_unsloth.zip")) -> Path:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            f = adapter_dir / name
            if f.exists():
                z.write(f, arcname=name)
    return out_zip


if __name__ == "__main__":
    app()
