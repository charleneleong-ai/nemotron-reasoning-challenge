"""Submit submission.zip to the Kaggle competition (reads KAGGLE_API_TOKEN from .env)."""

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from src.data.download import COMPETITION
from src.logger import get_logger

logger = get_logger(__name__)


def submit(
    zip_path: Path = Path("submission.zip"), message: str = "submission"
) -> None:
    """Upload `zip_path` to Kaggle via the CLI. Requires KAGGLE_API_TOKEN in .env."""
    load_dotenv()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError(
            "KAGGLE_API_TOKEN not set — add it to .env (Kaggle settings -> API token)."
        )
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found — run `main package` first.")

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
    logger.info("submitted %s", zip_path)
