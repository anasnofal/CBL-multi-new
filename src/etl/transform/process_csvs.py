from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
from ..config import (
    ARCHIVES,
    CSV_CHUNK_SIZE,
    DATASET_TYPES,
    EXTRACTED_DIR,
    LOG_DIR,
    PARQUET_DIR,
    ArchiveConfig,
    ensure_directories,
)
from ..load.write_parquet import PartitionKey, write_parquet_chunk
from ..utils.file_utils import iter_csv_files
from ..utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def get_dataset_type(csv_path: Path) -> str | None:
    name = csv_path.name.lower()
    if "stop-and-search" in name:
        return "stop_and_search"
    if "outcomes" in name:
        return "outcomes"
    if "street" in name:
        return "street"
    return None


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_names = {}
    for column in df.columns:
        new_names[column] = str(column).replace("\ufeff", "").strip()

    df = df.rename(columns=new_names)

    for column in df.columns:
        if column.lower() == "month" and column != "Month":
            df = df.rename(columns={column: "Month"})
            break

    return df


def keep_archive_months(
    df: pd.DataFrame,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    df = df.copy()
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    return df[
        (df["Month"] >= pd.to_datetime(start_month))
        & (df["Month"] <= pd.to_datetime(end_month))
    ]


def add_year_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["Month"].dt.year.astype("int16")
    df["month"] = df["Month"].dt.strftime("%m")
    return df


def empty_row_counts() -> dict[str, int]:
    return {dataset_type: 0 for dataset_type in DATASET_TYPES}


def process_csv_file(
    csv_path: Path,
    archive: ArchiveConfig,
    rewritten_partitions: set[PartitionKey],
) -> tuple[str | None, int, set[str]]:
    dataset_type = get_dataset_type(csv_path)
    if dataset_type is None:
        return None, 0, set()

    rows_written = 0
    months_seen: set[str] = set()
    logger.info("Processing %s", csv_path)

    chunks = pd.read_csv(
        csv_path,
        chunksize=CSV_CHUNK_SIZE,
        dtype=str,
        encoding="utf-8-sig",
        on_bad_lines="warn",
    )

    for chunk in chunks:
        chunk = clean_columns(chunk)

        if (
            "Month" not in chunk.columns
        ):  # Month here is column with data so it has both year and month
            logger.warning("Skipping file without Month column: %s", csv_path)
            return dataset_type, rows_written, months_seen

        chunk = keep_archive_months(chunk, archive.start_month, archive.end_month)
        if chunk.empty:
            continue

        chunk = chunk.drop_duplicates()
        chunk = add_year_month(
            chunk
        )  # this add year and month as separate columns which are needed for partitioning in parquet.

        months_seen.update(chunk["Month"].dt.strftime("%Y-%m").dropna().unique())
        rows_written += write_parquet_chunk(
            chunk,
            dataset_type,
            PARQUET_DIR,
            rewritten_partitions,
        )

    return dataset_type, rows_written, months_seen


def process_archive(
    archive: ArchiveConfig,
    rewritten_partitions: set[PartitionKey],
) -> tuple[dict[str, int], set[str]]:
    row_counts = empty_row_counts()
    months_seen: set[str] = set()

    for csv_path in iter_csv_files(EXTRACTED_DIR / archive.name):
        dataset_type, rows_written, file_months = process_csv_file(
            csv_path,
            archive,
            rewritten_partitions,
        )
        if dataset_type is not None:
            row_counts[dataset_type] += rows_written
            months_seen.update(file_months)

    return row_counts, months_seen


def transform_and_load_archives(
    archives: list[ArchiveConfig] | None = None,
) -> tuple[dict[str, int], set[str]]:
    total_counts = empty_row_counts()
    all_months: set[str] = set()
    rewritten_partitions: set[PartitionKey] = set()

    for archive in ARCHIVES if archives is None else archives:
        archive_counts, archive_months = process_archive(archive, rewritten_partitions)

        for dataset_type, count in archive_counts.items():
            total_counts[dataset_type] += count
        all_months.update(archive_months)

    return total_counts, all_months


def main() -> None:
    ensure_directories()
    setup_logging(LOG_DIR)
    row_counts, months = transform_and_load_archives()
    print(f"Transform/load complete. Rows: {row_counts}. Months: {len(months)}.")


if __name__ == "__main__":
    main()
