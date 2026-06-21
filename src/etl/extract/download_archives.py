"""Download UK Police archive ZIP files."""

from __future__ import annotations

import logging
from pathlib import Path

import requests
from tqdm import tqdm

from ..config import (
    ARCHIVE_DIR,
    ARCHIVES,
    DOWNLOAD_CHUNK_SIZE,
    LOG_DIR,
    ArchiveConfig,
    ensure_directories,
)
from ..utils.checksum import md5sum
from ..utils.logging_utils import setup_logging


logger = logging.getLogger(__name__)


def download_archive(archive: ArchiveConfig, archive_dir: Path = ARCHIVE_DIR) -> bool:
    """Download one archive. Return True if downloaded, False if skipped."""

    archive_dir.mkdir(parents=True, exist_ok=True)
    zip_path = archive_dir / archive.filename

    if zip_path.exists() and md5sum(zip_path) == archive.md5:
        logger.info("Skipping %s, file already exists and checksum is correct.", archive.name)
        return False

    if zip_path.exists():
        logger.warning("Checksum failed for %s. Downloading it again.", archive.name)

    temp_path = zip_path.with_suffix(".zip.part")
    if temp_path.exists():
        temp_path.unlink()

    logger.info("Downloading %s", archive.url)
    with requests.get(archive.url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with temp_path.open("wb") as file_obj:
            with tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=archive.name,
                leave=False,
            ) as progress:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        file_obj.write(chunk)
                        progress.update(len(chunk))

    temp_path.replace(zip_path)

    actual_md5 = md5sum(zip_path)
    if actual_md5 != archive.md5:
        raise ValueError(f"Bad checksum for {archive.name}: expected {archive.md5}, got {actual_md5}")

    logger.info("Downloaded %s", archive.name)
    return True


def download_archives(archives: list[ArchiveConfig] | None = None) -> tuple[int, int]:
    """Download all archives. Return (downloaded, skipped)."""

    downloaded = 0
    skipped = 0

    for archive in ARCHIVES if archives is None else archives:
        was_downloaded = download_archive(archive)
        if was_downloaded:
            downloaded += 1
        else:
            skipped += 1

    return downloaded, skipped


def main() -> None:
    ensure_directories()
    setup_logging(LOG_DIR)
    downloaded, skipped = download_archives()
    print(f"Download complete. Downloaded {downloaded}, skipped {skipped}.")


if __name__ == "__main__":
    main()

