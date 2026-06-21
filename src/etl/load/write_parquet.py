from __future__ import annotations
import shutil
import uuid
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PartitionKey = tuple[str, int, str]


def write_parquet_chunk(
    df: pd.DataFrame,
    dataset_type: str,
    parquet_dir: Path,
    rewritten_partitions: set[PartitionKey],
) -> int:
    """Write a chunk to dataset/year/month partitions.

    The first time a partition is seen in a run, it is deleted and recreated.
    That keeps reruns from appending duplicate files.
    """

    rows_written = 0

    for (year, month), month_df in df.groupby(["year", "month"]):
        year_int = int(year)
        month_str = str(month).zfill(2)
        partition_key = (dataset_type, year_int, month_str)
        partition_dir = (
            parquet_dir / dataset_type / f"year={year_int}" / f"month={month_str}"
        )

        if partition_key not in rewritten_partitions:
            if partition_dir.exists():
                shutil.rmtree(partition_dir)
            partition_dir.mkdir(parents=True, exist_ok=True)
            rewritten_partitions.add(partition_key)

        output_file = partition_dir / f"part-{uuid.uuid4().hex}.parquet"
        table = pa.Table.from_pandas(month_df, preserve_index=False)
        pq.write_table(table, output_file, compression="snappy")
        rows_written += len(month_df)

    return rows_written
