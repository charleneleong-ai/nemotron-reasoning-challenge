"""Run GSPO (RLVR) on your own CUDA box (A100/H100), warm-started from the SFT adapter.

Unlike Kaggle's offline BYOD kernel, here you control the env — so vLLM rollouts work and
there's no 12h cap. Produces `submission.zip` (the rank-<=32 adapter) to submit to Kaggle.

    pip install -e ".[gpu]" trl vllm kagglehub mamba-ssm causal-conv1d
    export KAGGLE_API_TOKEN=...            # for kagglehub model + (optional) submit
    python scripts/gspo_a100.py --adapter path/to/sft_adapter_dir            # train only
    python scripts/gspo_a100.py --adapter path/to/sft_adapter_dir --submit   # train + submit

Or submit separately:  uv run main submit   /   kaggle competitions submit -f submission.zip

GSPO = sequence-level importance sampling (MoE-stable) — set via importance_sampling_level.
"""

import zipfile
from pathlib import Path

import torch
import typer
from rich import print as rich_print

from src.config.schemas import DataConfig
from src.data.puzzles import build_inference_prompt, load_puzzles
from src.train.rl import boxed_reward, format_reward

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

KAGGLE_MODEL = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"


@app.command()
def main(
    adapter: str = typer.Option(
        None, help="SFT adapter dir to warm-start from (0.61). None = from base."
    ),
    data_path: str = typer.Option(
        "data/train.csv", help="Puzzles csv (id,prompt,answer)."
    ),
    cot_path: str = typer.Option(
        "data/cot.jsonl",
        help="CoT jsonl (only used to skip; RL needs prompts+answers).",
    ),
    out: str = typer.Option("adapters/gspo", help="Output adapter dir."),
    model_path: str = typer.Option(
        "", help="Local base-model path; blank => kagglehub download."
    ),
    lora_rank: int = typer.Option(32, help="LoRA rank (<=32) if training from base."),
    num_generations: int = typer.Option(8, help="GSPO group size G."),
    max_prompts: int = typer.Option(2000, help="Train prompts subset."),
    max_steps: int = typer.Option(500, help="GSPO optimizer steps."),
    max_completion_len: int = typer.Option(1024),
    lr: float = typer.Option(1e-6),
    beta: float = typer.Option(0.04, help="KL to the reference (anti-collapse)."),
    use_vllm: bool = typer.Option(
        True, help="vLLM rollouts (fast). Set False if vLLM lacks Nemotron support."
    ),
    submit: bool = typer.Option(
        False,
        help="After training, submit submission.zip to Kaggle (needs KAGGLE_API_TOKEN).",
    ),
    message: str = typer.Option("GSPO RLVR (A100)", help="Kaggle submission message."),
) -> None:
    """GSPO RL on a CUDA box; writes a rank<=32 adapter + submission.zip."""
    import kagglehub
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    base = model_path or kagglehub.model_download(KAGGLE_MODEL)
    rich_print(f"[cyan]base model:[/cyan] {base}")
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
    )

    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        rich_print(f"[green]warm-started[/green] from {adapter}")
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    puzzles = load_puzzles(
        DataConfig(path=data_path, cot_path=cot_path, max_samples=max_prompts)
    )
    ds = _build_dataset(puzzles, tokenizer)
    rich_print(f"[cyan]RL prompts:[/cyan] {len(ds)}")

    cfg = GRPOConfig(
        output_dir=out,
        importance_sampling_level="sequence",  # GSPO (vs "token" = GRPO)
        num_generations=num_generations,
        max_completion_length=max_completion_len,
        learning_rate=lr,
        beta=beta,
        max_steps=max_steps,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_vllm=use_vllm,
        bf16=True,
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        reward_funcs=[boxed_reward, format_reward],
        train_dataset=ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out)

    zip_path = _package(Path(out))
    rich_print(f"[green]done.[/green] adapter -> {out} | submission -> {zip_path}")

    if submit:
        from src.submission.submit import submit as submit_zip

        submit_zip(zip_path, message)
        rich_print("[green]submitted[/green] to Kaggle")


def _build_dataset(puzzles: list, tokenizer: object) -> object:
    from datasets import Dataset

    return Dataset.from_list(
        [
            {"prompt": build_inference_prompt(p.prompt, tokenizer), "answer": p.answer}
            for p in puzzles
        ]
    )


def _package(adapter_dir: Path, out_zip: Path = Path("submission.zip")) -> Path:
    """Zip adapter_config.json + adapter_model.safetensors into submission.zip."""
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            f = adapter_dir / name
            if f.exists():
                z.write(f, arcname=name)
    return out_zip


if __name__ == "__main__":
    app()
