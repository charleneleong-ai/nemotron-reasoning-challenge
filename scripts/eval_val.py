"""Score a trained LoRA adapter on the held-out 502 val problems (local estimate).

The val ids (data/val_ids.txt) were excluded from data/cot_train.jsonl, so this is a
leak-free proxy for the hidden Kaggle test. Loads base + adapter, generates a completion
per val problem, extracts the boxed answer, and grades with the competition tolerance.

NemotronH ships a broken generation cache: `prepare_inputs_for_generation` returns the
cache under `past_key_values`, but `forward()` reads `cache_params` — so the cache is
dropped every step and decode re-processes the whole prefix (O(n^2), ~2 min/problem).
`_patch_generation_cache` rebinds a corrected `prepare_inputs_for_generation` that threads
the cache through `cache_params` (the key transformers' generate loop already persists),
sets the `conv_kernel_size` attribute the shipped cache class forgets (the mixer reads it
during prefill), and fixes `update_{conv,ssm}_state`, which deref `.device` on a list instead
of the indexed layer tensor. Together they restore true KV/SSM-cached decode.

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


def _patch_generation_cache(model: object) -> bool:
    """Rebind NemotronH's `prepare_inputs_for_generation` to thread the cache via `cache_params`.

    The shipped version returns the cache under `past_key_values`, which `forward()` ignores
    (it reads `cache_params`), so every decode step rebuilds an empty cache. We swap the key to
    `cache_params` — the cache name transformers' generate loop already carries forward — so the
    Mamba SSM/conv and attention KV states persist across steps. Returns True if patched.
    """
    cls = next(
        (
            type(m)
            for m in model.modules()  # type: ignore[attr-defined]
            if type(m).__name__ == "NemotronHForCausalLM"
        ),
        None,
    )
    if cls is None:
        return False
    cache_cls = sys.modules[cls.__module__].HybridMambaAttentionDynamicCache

    # The shipped cache's update_{conv,ssm}_state read `self.<states>.device`, but those
    # attrs are lists — index the layer first. Rebind both with the device fix.
    def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
        dev = self.conv_states[layer_idx].device
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(dev)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(
                shifts=-1, dims=-1
            )
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(dev)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx, new_ssm_state):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]

    cache_cls.update_conv_state = update_conv_state
    cache_cls.update_ssm_state = update_ssm_state

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        past = kwargs.get("cache_params")
        empty = past is None
        if not empty:
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]
        else:
            past = cache_cls(
                self.config, input_ids.shape[0], self.dtype, device=self.device
            )
            # The shipped cache __init__ never sets conv_kernel_size, but both the
            # cuda and torch mixer paths read it during prefill — add it here.
            past.conv_kernel_size = self.config.conv_kernel

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = (
            {"inputs_embeds": inputs_embeds}
            if inputs_embeds is not None and empty
            else {"input_ids": input_ids.contiguous()}
        )
        model_inputs.update(
            cache_params=past,
            position_ids=position_ids,
            use_cache=use_cache,
            attention_mask=attention_mask,
            cache_position=cache_position,
        )
        return model_inputs

    cls.prepare_inputs_for_generation = prepare_inputs_for_generation
    return True


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
    print(
        "KV/SSM cache patch: "
        + ("applied" if _patch_generation_cache(model) else "NOT APPLIED"),
        flush=True,
    )

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
