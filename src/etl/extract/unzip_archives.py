"""Extract downloaded archive ZIP files."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

from ..config import (
    ARCHIVE_DIR,
    ARCHIVES,
    EXTRACTED_DIR,
    LOG_DIR,
    ArchiveConfig,
    ensure_directories,
)
from ..utils.logging_utils import setup_logging


logger = logging.getLogger(__name__)


def extract_archive(
    archive: ArchiveConfig,
    archive_dir: Path = ARCHIVE_DIR,
    extracted_dir: Path = EXTRACTED_DIR,
) -> bool:
    """Extract one archive. Return True if extracted, False if skipped."""

    zip_path = archive_dir / archive.filename
    output_dir = extracted_dir / archive.name
    marker_path = output_dir / ".extracted"

    if marker_path.exists():
        logger.info("Skipping %s, already extracted.", archive.name)
        return False

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing ZIP file: {zip_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    logger.info("Extracting %s", zip_path)
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(output_dir)

    marker_path.write_text("ok\n", encoding="utf-8")
    return True


def extract_archives(archives: list[ArchiveConfig] | None = None) -> tuple[int, int]:
    """Extract all archives. Return (extracted, skipped)."""

    extracted = 0
    skipped = 0

    for archive in ARCHIVES if archives is None else archives:
        was_extracted = extract_archive(archive)
        if was_extracted:
            extracted += 1
        else:
            skipped += 1

    return extracted, skipped


def main() -> None:
    ensure_directories()
    setup_logging(LOG_DIR)
    extracted, skipped = extract_archives()
    print(f"Extraction complete. Extracted {extracted}, skipped {skipped}.")


if __name__ == "__main__":
    main()

