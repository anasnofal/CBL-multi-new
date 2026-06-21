from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(log_dir: Path) -> Path:
    """Configure console and file logging for an ETL run."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        log_dir / f"etl_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_file
