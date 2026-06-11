"""Download + extract the Kaggle competition dataset using credentials from .env."""

import os
import subprocess
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from src.logger import get_logger

logger = get_logger(__name__)

COMPETITION = "nvidia-nemotron-model-reasoning-challenge"


def download_dataset(dest: Path = Path("data")) -> Path:
    """Pull the competition zip via the Kaggle CLI and extract it into `dest`.

    Reads `KAGGLE_API_TOKEN` from `.env` (Kaggle CLI >=2.2 auth). Raises if it is unset.
    """
    load_dotenv()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError(
            "KAGGLE_API_TOKEN not set — add it to .env (Kaggle settings -> API token)."
        )

    dest.mkdir(parents=True, exist_ok=True)
    # load_dotenv() above put KAGGLE_API_TOKEN into os.environ; the child inherits it.
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(dest)],
        check=True,
    )
    zip_path = dest / f"{COMPETITION}.zip"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    logger.info("extracted %s -> %s", zip_path.name, dest)
    return dest
