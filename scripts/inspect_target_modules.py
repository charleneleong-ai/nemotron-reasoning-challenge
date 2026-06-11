"""Inspect a base model's nn.Linear module-name suffixes WITHOUT downloading weights.

Loads the architecture on the meta device via accelerate.init_empty_weights() so the
~60GB checkpoint is never fetched, then prints the unique torch.nn.Linear suffixes so you
can pick concrete `lora_target_modules` instead of "all-linear".

Usage (run on a CUDA box with the custom Nemotron kernels available):
    uv run python scripts/inspect_target_modules.py \
        --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
"""

import typer

from src.logger import get_logger

logger = get_logger("inspect_target_modules")

DEFAULT_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _linear_suffixes(model: object) -> list[str]:
    import torch

    suffixes: set[str] = set()
    for name, module in model.named_modules():  # type: ignore[attr-defined]
        if isinstance(module, torch.nn.Linear):
            suffixes.add(name.rsplit(".", 1)[-1])
    return sorted(suffixes)


@app.command()
def main(
    model: str = typer.Option(DEFAULT_MODEL, help="HF model id to inspect."),
) -> None:
    """Print the sorted unique nn.Linear suffixes + a ready-to-paste YAML line."""
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        logger.error(
            "Missing transformers/accelerate. Install the [gpu] extra: "
            "`uv sync --extra gpu`. (%s)",
            exc,
        )
        raise typer.Exit(code=1) from exc

    logger.info("Loading config for %s (trust_remote_code=True)...", model)
    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except (OSError, ValueError, ImportError) as exc:
        logger.error(
            "Failed to load AutoConfig for %s. Check HF auth (gated model) and "
            "network access. (%s)",
            model,
            exc,
        )
        raise typer.Exit(code=1) from exc

    logger.info("Materializing architecture on the meta device (no weights)...")
    try:
        with init_empty_weights():
            empty_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True
            )
    except ImportError as exc:
        logger.error(
            "Custom Nemotron kernels failed to import. This script must run on a "
            "CUDA box with trust_remote_code dependencies installed (mamba-ssm, "
            "causal-conv1d, etc.). (%s)",
            exc,
        )
        raise typer.Exit(code=1) from exc

    suffixes = _linear_suffixes(empty_model)
    if not suffixes:
        logger.warning("No torch.nn.Linear modules found — nothing to target.")
        raise typer.Exit(code=1)

    logger.info("Found %d unique nn.Linear suffixes:", len(suffixes))
    for suffix in suffixes:
        typer.echo(f"  - {suffix}")

    yaml_list = ", ".join(suffixes)
    typer.echo("")
    typer.echo("Ready-to-paste YAML (pick a subset for LoRA):")
    typer.echo(f"lora_target_modules: [{yaml_list}]")


if __name__ == "__main__":
    app()
