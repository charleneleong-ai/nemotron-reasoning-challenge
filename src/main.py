"""Typer CLI: download | prepare | train | eval | package."""

from pathlib import Path

import typer
from rich import print as rich_print

from src.config.loader import load_experiment_config
from src.data.puzzles import load_puzzles, split_puzzles

app = typer.Typer(
    name="main",
    help="Nemotron reasoning challenge scaffold",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.command()
def download() -> None:
    """Download + extract the Kaggle competition data into data/ (reads KAGGLE_API_TOKEN from .env)."""
    from src.data.download import download_dataset

    dest = download_dataset()
    rich_print(f"[green]data ready[/green] -> {dest}")


@app.command()
def prepare(overrides: list[str] = typer.Argument(None)) -> None:
    """Load + split puzzles, print dataset stats."""
    cfg = load_experiment_config(overrides)
    train, dev = split_puzzles(load_puzzles(cfg.data), cfg.data)
    rich_print(
        f"[green]loaded[/green] {len(train)} train / {len(dev)} dev from {cfg.data.path}"
    )


@app.command()
def train(overrides: list[str] = typer.Argument(None)) -> None:
    """Train a LoRA adapter."""
    from src.train.lora import train_adapter

    cfg = load_experiment_config(overrides)
    out = train_adapter(cfg)
    rich_print(f"[green]adapter saved[/green] -> {out}")


@app.command()
def eval(overrides: list[str] = typer.Argument(None)) -> None:  # noqa: A001
    """Evaluate a trained adapter (boxed accuracy)."""
    from src.eval.runner import run_eval

    cfg = load_experiment_config(overrides)
    result = run_eval(cfg, Path(cfg.train.output_dir))
    rich_print(f"[cyan]accuracy[/cyan] {result['accuracy']:.3f}")


@app.command()
def package(overrides: list[str] = typer.Argument(None)) -> None:
    """Package the trained adapter into submission.zip."""
    from src.submission.package import package_submission

    cfg = load_experiment_config(overrides)
    out = package_submission(Path(cfg.train.output_dir), Path("submission.zip"))
    rich_print(f"[green]packaged[/green] -> {out}")


@app.command()
def submit(
    message: str = typer.Option("submission", help="Submission description."),
) -> None:
    """Upload submission.zip to the Kaggle competition (reads KAGGLE_API_TOKEN from .env)."""
    from src.submission.submit import submit as submit_zip

    submit_zip(Path("submission.zip"), message)
    rich_print("[green]submitted[/green] submission.zip")


if __name__ == "__main__":
    app()
