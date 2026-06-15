"""Score a trained LoRA adapter on the held-out 502 val problems (local estimate).

The val ids (data/val_ids.txt) were excluded from data/cot_train.jsonl, so this is a
leak-free proxy for the hidden Kaggle test. Loads base + adapter, generates a completion
per val problem, extracts the boxed answer, and grades with the competition tolerance.

    .venv-unsloth/bin/python scripts/eval_val.py --adapter adapters/sft_valexcl
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import typer

# Runs in .venv-unsloth (no project install) — make `src` importable from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.categories import classify  # noqa: E402
from src.solve.registry import matches  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
KAGGLE_MODEL = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
_BOXED = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)


def _extract(text: str) -> str | None:
    m = _BOXED.findall(text)
    return m[-1].strip() if m else None


@app.command()
def main(
    adapter: str = typer.Option(
        "adapters/sft_valexcl", help="Trained LoRA adapter dir."
    ),
    model_path: str = typer.Option(""),
    val_ids: str = typer.Option("data/val_ids.txt"),
    data: str = typer.Option("data/train.csv"),
    max_new_tokens: int = typer.Option(1024),
) -> None:
    from unsloth import FastLanguageModel

    base = model_path
    if not base:
        import kagglehub

        base = kagglehub.model_download(KAGGLE_MODEL)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=6144,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
    )
    model.load_adapter(adapter)
    FastLanguageModel.for_inference(model)

    ids = set(Path(val_ids).read_text().split())
    rows = [r for r in csv.DictReader(open(data)) if r["id"] in ids]
    score: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i, r in enumerate(rows):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(
            out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        cat = classify(r["prompt"]) or "unknown"
        score[cat][1] += 1
        if matches(_extract(text), r["answer"]):
            score[cat][0] += 1
        if (i + 1) % 25 == 0:
            done = sum(c for c, _ in score.values())
            tot = sum(t for _, t in score.values())
            print(f"[{i + 1}/{len(rows)}] running acc {done}/{tot}", flush=True)

    print("\n=== val accuracy by category ===")
    tot = [0, 0]
    for cat in sorted(score):
        c, t = score[cat]
        tot[0] += c
        tot[1] += t
        print(f"{cat:18s} {c:>3d}/{t:<3d} ({c / t * 100:.0f}%)")
    print(f"{'OVERALL':18s} {tot[0]:>3d}/{tot[1]:<3d} ({tot[0] / tot[1] * 100:.1f}%)")


if __name__ == "__main__":
    app()
