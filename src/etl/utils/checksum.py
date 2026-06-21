"""Checksum helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.etl.config import CHECKSUM_CHUNK_SIZE


def md5sum(path: Path, chunk_size: int = CHECKSUM_CHUNK_SIZE) -> str:
    """Calculate the MD5 digest for a file without loading it into memory."""

    digest = hashlib.md5()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: Path, expected_md5: str) -> bool:
    """Return True when path exists and its MD5 matches expected_md5."""

    if not path.exists():
        return False
    return md5sum(path) == expected_md5.lower()

