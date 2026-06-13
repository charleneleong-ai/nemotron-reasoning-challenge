"""Pull a finished Kaggle training kernel's adapter, validate it, and submit.

The Kaggle training notebook (e.g. `nemotron-train-2ep-cot`) writes
`submission.zip` (a flat rank<=32 LoRA adapter) to `/kaggle/working` but does
NOT submit. This script closes that gap: it pulls the kernel output, validates
the zip, stages it at the repo-root `submission.zip`, and (optionally) submits
to the competition.

    # kernel is already COMPLETE -> pull + validate + stage (no submit)
    uv run python scripts/submit_from_kernel.py --no-submit

    # poll until the kernel finishes, then pull + validate + stage + submit
    uv run python scripts/submit_from_kernel.py --watch --submit \
        -m "CoT-SFT 2 epochs (r32 LoRA)"

    # default kernel is charyeezy/nemotron-train-2ep-cot
    uv run python scripts/submit_from_kernel.py --kernel <owner>/<slug> --watch --submit

Reads KAGGLE_API_TOKEN from .env (same auth path as `main download`/`main submit`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich import print as rich_print

from src.data.download import COMPETITION

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

DEFAULT_KERNEL = "charyeezy/nemotron-train-2ep-cot"
MAX_RANK = 32
REQUIRED_MEMBERS = ("adapter_config.json", "adapter_model.safetensors")
# Terminal Kaggle kernel states (the bare enum tail, e.g. "COMPLETE").
DONE_OK = {"COMPLETE"}
DONE_BAD = {"ERROR", "CANCELACKNOWLEDGED", "CANCELREQUESTED"}


def _ensure_token() -> None:
    """Load KAGGLE_API_TOKEN from .env into the env for child CLI calls."""
    load_dotenv()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError(
            "KAGGLE_API_TOKEN not set - add it to .env (Kaggle settings -> API token)."
        )


def _kernel_status(kernel: str) -> str:
    """Return the bare kernel status tail, e.g. 'RUNNING' or 'COMPLETE'."""
    out = subprocess.run(
        ["kaggle", "kernels", "status", kernel],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # e.g.: <kernel> has status "KernelWorkerStatus.COMPLETE"
    tail = out.rsplit(".", 1)[-1].strip().strip('"')
    return tail.upper()


def _wait_until_done(kernel: str, poll_s: float, log_path: Path | None) -> str:
    """Poll the kernel until it reaches a terminal state; return that state."""
    while True:
        status = _kernel_status(kernel)
        stamp = time.strftime("%H:%M:%S", time.gmtime())
        line = f"{stamp} {kernel} -> {status}"
        rich_print(f"[cyan]{line}[/cyan]")
        if log_path is not None:
            with log_path.open("a") as fh:
                fh.write(line + "\n")
                fh.flush()
        if status in DONE_OK or status in DONE_BAD:
            return status
        time.sleep(poll_s)


def _pull_output(kernel: str, dest: Path) -> Path:
    """Download the kernel output into `dest`; return the submission.zip path."""
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", kernel, "-p", str(dest)],
        check=True,
    )
    zip_path = dest / "submission.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not in kernel output - did the notebook package a submission?"
        )
    return zip_path


def _validate(zip_path: Path) -> dict[str, int]:
    """Validate the submission zip: integrity, members, and LoRA rank <= 32.

    Returns a small summary dict (rank, total uncompressed bytes). Raises on any
    fatal problem so a bad adapter is never submitted.
    """
    if zip_path.stat().st_size == 0:
        raise ValueError(f"{zip_path} is empty (0 bytes) - pull likely interrupted.")

    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f"corrupt entry in {zip_path}: {bad}")

        names = set(z.namelist())
        missing = [m for m in REQUIRED_MEMBERS if m not in names]
        if missing:
            raise ValueError(f"{zip_path} missing required files: {missing}")

        cfg = json.loads(z.read("adapter_config.json"))
        total = sum(i.file_size for i in z.infolist())

    rank = cfg.get("r")
    if rank is None:
        raise ValueError("adapter_config.json has no 'r' (LoRA rank) field.")
    if rank > MAX_RANK:
        raise ValueError(f"LoRA rank {rank} exceeds competition cap of {MAX_RANK}.")

    return {"rank": rank, "uncompressed_bytes": total}


def _submit(zip_path: Path, message: str) -> None:
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            COMPETITION,
            "-f",
            str(zip_path),
            "-m",
            message,
        ],
        check=True,
    )


@app.command()
def main(
    kernel: str = typer.Option(
        DEFAULT_KERNEL, help="owner/slug of the training kernel."
    ),
    watch: bool = typer.Option(
        False, help="Poll the kernel until it finishes before pulling."
    ),
    poll_s: float = typer.Option(120.0, help="Seconds between status polls (--watch)."),
    submit: bool = typer.Option(
        False,
        "--submit/--no-submit",
        help="Submit after validation. --no-submit only pulls + validates + stages.",
    ),
    message: str = typer.Option(
        "CoT-SFT (Kaggle kernel adapter)", "-m", "--message", help="Submission message."
    ),
    workdir: str = typer.Option(
        "outputs/kernel_pull", help="Where to download the kernel output."
    ),
    out_zip: str = typer.Option(
        "submission.zip", help="Repo-root path to stage the validated zip."
    ),
    watch_log: str = typer.Option(
        "logs/train_2ep_watch.log", help="Append status polls here when watching."
    ),
) -> None:
    """Pull -> validate -> stage -> (optionally) submit a Kaggle kernel adapter."""
    _ensure_token()

    if watch:
        log_path = Path(watch_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        status = _wait_until_done(kernel, poll_s, log_path)
    else:
        status = _kernel_status(kernel)
        rich_print(f"[cyan]{kernel} status:[/cyan] {status}")

    if status not in DONE_OK:
        raise typer.Exit(code=1 if status in DONE_BAD else 2)  # bad=1, still-running=2

    dest = Path(workdir)
    zip_path = _pull_output(kernel, dest)
    summary = _validate(zip_path)
    rich_print(
        f"[green]valid[/green] rank={summary['rank']} "
        f"uncompressed={summary['uncompressed_bytes'] / 1e9:.2f} GB -> {zip_path}"
    )

    staged = Path(out_zip)
    if zip_path.resolve() != staged.resolve():
        shutil.copy2(zip_path, staged)
    rich_print(f"[green]staged[/green] -> {staged}")

    if not submit:
        rich_print(
            "[yellow]--no-submit[/yellow]: staged only. To submit:\n"
            f'  uv run main submit -m "{message}"'
        )
        return

    _submit(staged, message)
    rich_print(f"[green]submitted[/green] {staged} to {COMPETITION}")


if __name__ == "__main__":
    app()
