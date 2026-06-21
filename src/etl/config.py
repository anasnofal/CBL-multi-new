"""Configuration for the UK Police archive ETL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path("/Volumes/SSD/police_data")

RAW_DIR = BASE_DIR / "raw"
ARCHIVE_DIR = RAW_DIR / "archives"
EXTRACTED_DIR = RAW_DIR / "extracted"

PROCESSED_DIR = BASE_DIR / "processed"
PARQUET_DIR = PROCESSED_DIR / "parquet"

LOG_DIR = BASE_DIR / "logs"
METADATA_DIR = BASE_DIR / "metadata"

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
CHECKSUM_CHUNK_SIZE = 1024 * 1024
CSV_CHUNK_SIZE = 100_000

DATASET_TYPES = ("street", "outcomes", "stop_and_search")


@dataclass(frozen=True)
class ArchiveConfig:
    """A single archive plus the month window that should be kept."""

    name: str
    url: str
    md5: str
    start_month: str
    end_month: str

    @property
    def filename(self) -> str:
        return f"{self.name}.zip"


ARCHIVES = [
    ArchiveConfig(
        name="2017-03",
        url="https://data.police.uk/data/archive/2017-03.zip",
        md5="54cc262535dd69d426c9d2133a8b1fd1",
        start_month="2010-12",
        end_month="2017-03",
    ),
    ArchiveConfig(
        name="2020-03",
        url="https://data.police.uk/data/archive/2020-03.zip",
        md5="0c6e71e4b68b952506ebc1a092e36102",
        start_month="2017-04",
        end_month="2020-03",
    ),
    ArchiveConfig(
        name="2023-03",
        url="https://data.police.uk/data/archive/2023-03.zip",
        md5="91ee27086b790a542d7ccfa7dc9b391d",
        start_month="2020-04",
        end_month="2023-03",
    ),
    ArchiveConfig(
        name="2026-03",
        url="https://data.police.uk/data/archive/2026-03.zip",
        md5="6dde462489389445877f3988ef3f4f4b",
        start_month="2023-04",
        end_month="2026-03",
    ),
]


def ensure_directories() -> None:
    """Create all expected output directories."""

    directories = [
        ARCHIVE_DIR,
        EXTRACTED_DIR,
        PARQUET_DIR,
        LOG_DIR,
        METADATA_DIR,
        *(PARQUET_DIR / dataset_type for dataset_type in DATASET_TYPES),
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

