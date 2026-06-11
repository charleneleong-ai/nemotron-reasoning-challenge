"""Rich-backed logging setup."""

import logging

from rich.logging import RichHandler


def get_logger(name: str = "nemotron") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
        logger.propagate = False
    return logger
