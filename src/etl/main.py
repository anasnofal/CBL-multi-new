"""Run the UK Police ETL."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .config import LOG_DIR, METADATA_DIR, PARQUET_DIR, ensure_directories
from .extract.download_archives import download_archives
from .extract.unzip_archives import extract_archives
from .transform.process_csvs import transform_and_load_archives
from .utils.logging_utils import setup_logging


logger = logging.getLogger(__name__)


def write_run_summary(summary: dict) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = METADATA_DIR / "etl_run_summary.json"
    output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_pipeline() -> None:
    ensure_directories()
    setup_logging(LOG_DIR)

    started_at = datetime.now(timezone.utc)
    logger.info("Starting ETL")

    downloaded, download_skipped = download_archives()
    extracted, extract_skipped = extract_archives()
    row_counts, months = transform_and_load_archives()

    finished_at = datetime.now(timezone.utc)
    summary = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "archives_downloaded": downloaded,
        "archives_download_skipped": download_skipped,
        "archives_extracted": extracted,
        "archives_extract_skipped": extract_skipped,
        "rows_written": row_counts,
        "months_loaded": sorted(months),
        "output_path": str(PARQUET_DIR),
    }
    write_run_summary(summary)

    logger.info("ETL finished")
    print(f"ETL complete. Rows written: {row_counts}")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
