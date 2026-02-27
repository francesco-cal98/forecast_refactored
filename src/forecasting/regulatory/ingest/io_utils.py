"""I/O utilities and minimal logging for ingestion workflows."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd


def get_logger(name: str, level: str | int = "INFO") -> logging.Logger:
    """Creates and returns a configured logger.

    Args:
        name: Logger name.
        level: Log level as string or int.
    """
    logger = logging.getLogger(name)
    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), logging.INFO)
    else:
        lvl = int(level)
    logger.setLevel(lvl)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(lvl)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def ensure_dirs(*paths: Iterable[str | Path]) -> None:
    """Ensures that directories for given paths exist.

    If a path points to a file, ensures the parent directory; if a directory,
    ensures that directory. Accepts strings or Path objects.
    """
    for p in paths:
        if p is None:
            continue
        path = Path(p)
        # If it has a suffix, treat as file path
        dir_path = path if path.suffix == "" else path.parent
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)


def safe_to_parquet(df: pd.DataFrame, path: str | Path, engine: str = "pyarrow") -> None:
    """Safely writes a DataFrame to Parquet, creating parent dirs.

    Writes even empty DataFrames (schema-only files) without raising.
    """
    ensure_dirs(path)
    df.to_parquet(Path(path), engine=engine, index=False)

