"""Local SFT of a rank-32 LoRA on the A100, mirroring the Kaggle recipe that scored 0.62.

Same model/rank/format as the fast SFT kernel; the change is the data — train on
cot_hybrid.jsonl, where 5999/9500 reasoning traces are deterministic-solver output
(verified correct) instead of generic CoT. SFT has no generation loop, so the
NemotronH KV-cache slowness that blocks local RL doesn't apply here.

COMPLETION-ONLY LOSS: the puzzle prompts are high-entropy random data (cipher
gibberish, random bits/floats) — training to predict them floods the loss (~9) and
wastes capacity. We mask everything up to the template's prefilled `<think>\\n` (the
inference boundary) and train only on the reasoning + boxed answer, like the 0.62 run.

    .venv-unsloth/bin/python scripts/sft_local.py --max-steps 5 --max-rows 64   # smoke
    .venv-unsloth/bin/python scripts/sft_local.py                                # full

Self-contained (runs in .venv-unsloth, no project install).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

KAGGLE_MODEL = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
TARGET_MODULES = ["in_proj", "out_proj", "up_proj", "down_proj"]


def format_target(answer: str, think: str) -> str:
    """Full assistant turn: <think> reasoning </think> then the boxed answer."""
    return f"<think>\n{think.strip()}\n</think>\n\n\\boxed{{{answer}}}"


def _tokenize(rows: list[dict[str, str]], tokenizer: Any, max_len: int) -> list[dict]:
    """Tokenize to {input_ids, attention_mask, labels} with the prompt masked (-100).

    Labels cover only the tokens after the inference boundary (the prefilled `<think>\\n`),
    i.e. exactly what the model must generate. Examples longer than max_len are dropped
    (not truncated) so the boxed answer is never cut from the label.
    """
    out: list[dict] = []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"]}]
        prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        msgs.append(
            {"role": "assistant", "content": format_target(r["answer"], r["think"])}
        )
        full_ids = tokenizer.apply_chat_template(msgs)
        if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) > max_len:
            continue
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        out.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    return out


def _patch_nemotron_moe(model: Any) -> None:
    """Fix NemotronH's MoE index_add_ dtype mismatch under mixed precision (forward + train)."""
    import inspect
    import textwrap

    bad = "final_hidden_states.index_add_(0, token_indices, weighted_output)"
    fix = "final_hidden_states.index_add_(0, token_indices, weighted_output.to(final_hidden_states.dtype))"
    for module in model.modules():
        cls = type(module)
        if getattr(cls, "_moe_dtype_fixed", False) or not hasattr(cls, "moe"):
            continue
        try:
            src = textwrap.dedent(inspect.getsource(cls.moe))
        except (OSError, TypeError):
            continue
        if bad not in src:
            continue
        ns: dict[str, Any] = {}
        exec(
            compile(src.replace(bad, fix), "<moe_patch>", "exec"),
            cls.moe.__globals__,
            ns,
        )
        cls.moe = ns["moe"]
        cls._moe_dtype_fixed = True
        print(f"patched {cls.__name__}.moe dtype cast", flush=True)


@app.command()
def main(
    model_path: str = typer.Option("", help="Local model path; blank => kagglehub."),
    data_path: str = typer.Option(
        "data/cot_hybrid.jsonl", help="{id,prompt,answer,think} jsonl."
    ),
    adapter: str = typer.Option(
        "", help="Warm-start from this LoRA adapter dir (blank => fresh)."
    ),
    out: str = typer.Option("adapters/sft_local", help="Output adapter dir."),
    lora_rank: int = typer.Option(32),
    lora_alpha: int = typer.Option(16),
    epochs: float = typer.Option(2.0),
    max_steps: int = typer.Option(
        -1, help="Override epochs for a smoke (-1 = use epochs)."
    ),
    max_rows: int = typer.Option(0, help="Train subset for a smoke (0 = all)."),
    batch: int = typer.Option(2),
    grad_accum: int = typer.Option(4),
    lr: float = typer.Option(2e-5),
    max_seq_len: int = typer.Option(1536),
) -> None:
    """SFT a rank-32 LoRA on cot_hybrid.jsonl (bf16, completion-only loss) + package submission.zip."""
    from unsloth import FastLanguageModel  # noqa: I001 — import before trl/transformers
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    base = model_path
    if not base:
        import kagglehub

        base = kagglehub.model_download(KAGGLE_MODEL)
    print(f"base model: {base} | bf16 SFT (completion-only)", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=max_seq_len,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
    )
    if adapter:
        model.load_adapter(adapter, adapter_name="default")
        print(f"warm-started from {adapter}", flush=True)
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=TARGET_MODULES,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    _patch_nemotron_moe(model)

    rows = [json.loads(line) for line in Path(data_path).open()]
    if max_rows:
        rows = rows[:max_rows]
    examples = _tokenize(rows, tokenizer, max_seq_len)
    ds = Dataset.from_list(examples)
    trained = sum(sum(1 for x in e["labels"] if x != -100) for e in examples)
    print(
        f"SFT examples: {len(ds)}/{len(rows)} (avg {trained / max(len(ds), 1):.0f} trained tokens)",
        flush=True,
    )

    args = SFTConfig(
        output_dir=out,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_steps=max_steps,
        max_length=max_seq_len,
        logging_steps=1,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=1,
        report_to=[],
        bf16=True,
        # 8-bit Adam: the MoE-expert LoRA (128 experts) has large optimizer state; fp32
        # Adam pushed the card to 99% pre-training and OOM'd. 8-bit cuts it ~4x.
        optim="adamw_8bit",
        # Feed our pre-tokenized, prompt-masked rows straight through (SFTTrainer tolerates
        # Unsloth's device_map'd model where plain Trainer refuses).
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tokenizer,  # pass our trust_remote_code tokenizer; else SFTTrainer
        # auto-loads a processor without trust_remote_code and EOFs on the [y/N] prompt
    )
    trainer.train()

    Path(out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    zip_path = _package(Path(out))
    print(f"done. adapter -> {out} | submission -> {zip_path}", flush=True)


def _package(
    adapter_dir: Path, out_zip: Path = Path("submission_sft_local.zip")
) -> Path:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            f = adapter_dir / name
            if f.exists():
                z.write(f, arcname=name)
    return out_zip


if __name__ == "__main__":
    app()
