"""
Central logging configuration for BESTA.
"""

import logging
import sys
from typing import Optional


DEFAULT_FORMAT = (
    "[%(name)s | %(asctime)s] "
    "(%(levelname)-8s) "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    overwrite: bool = False,
):
    """
    Configure the root BESTA logger.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str, optional
        If provided, logs are written to this file.
    overwrite : bool
        If True, overwrite log_file instead of appending.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("besta")
    logger.setLevel(numeric_level)
    logger.propagate = False  # prevent duplicate logs

    # Remove previous handlers
    logger.handlers.clear()

    formatter = logging.Formatter(DEFAULT_FORMAT, DATE_FORMAT)

    # Console handler
    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # File handler (optional)
    if log_file:
        mode = "w" if overwrite else "a"
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logging configured successfully.")


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger under the ``besta`` namespace."""
    base = logging.getLogger("besta")
    if module_name.startswith("besta."):
        return base.getChild(module_name[len("besta."):])
    return base.getChild(module_name)
