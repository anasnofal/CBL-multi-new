from __future__ import annotations
from pathlib import Path


def iter_csv_files(root: Path) -> list[Path]:
    """Find CSV files recursively in a stable order."""

    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def ensure_parent(path: Path) -> None:
    """Create a file path's parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
