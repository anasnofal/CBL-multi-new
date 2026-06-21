"""
A script to combine all CSV files of a given type (street, outcomes, stop-and-search) into a 3 single files.
you can do download the data from https://data.police.uk/data/ and then run this script to combine the data into 3 files.
Usage:
    python combine_police_data.py -i /path/to/downloaded/csvs -o /path/to/save/combined/files
If you don't specify the input and output directories, it will default to the script's directory.
"""

import os
import glob
import pandas as pd
from pathlib import Path
import argparse

SCRIPT_DIR = Path(__file__).parent


FILE_TYPES = {
    "street": "all_street.csv",
    "outcomes": "all_outcomes.csv",
    "stop-and-search": "all_stop_and_search.csv",
}


def find_files(input_dir: Path, suffix: str) -> list[Path]:
    """Return all CSVs whose name ends with -{suffix}.csv (any depth)."""
    pattern = str(input_dir / "**" / f"*-{suffix}.csv")
    return sorted(glob.glob(pattern, recursive=True))


def combine(files: list[Path], label: str) -> pd.DataFrame:
    """Read and concatenate all files; add a 'source_file' column."""
    chunks = []
    for i, fp in enumerate(files, 1):
        try:
            df = pd.read_csv(fp, dtype=str, keep_default_na=False)
            df.insert(0, "source_file", Path(fp).name)
            chunks.append(df)
            if i % 100 == 0 or i == len(files):
                print(f"  [{label}] read {i}/{len(files)} files …")
        except Exception as e:
            print(f"  WARNING — could not read {fp}: {e}")

    if not chunks:
        print(f"  No files found for type '{label}'.")
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True, sort=False)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="A script that saves files to a specified output directory."
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=SCRIPT_DIR,
        help="Path to the output directory where results will be saved. Defaults to the script's directory.",
    )
    parser.add_argument(
        "-i",
        "--inputdir",
        type=Path,
        default=SCRIPT_DIR,
        help="Path to the input directory containing CSV files. Defaults to the script's directory.",
    )

    args = parser.parse_args()

    OUTPUT_DIR = args.outdir
    INPUT_DIR = args.inputdir

    print(f"Input directory : {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for suffix, out_name in FILE_TYPES.items():
        print(f"── Processing '{suffix}' files ──")
        files = find_files(INPUT_DIR, suffix)
        print(f"  Found {len(files)} files.")

        if not files:
            continue

        df = combine(files, suffix)
        out_path = OUTPUT_DIR / out_name
        df.to_csv(out_path, index=False)

        print(f"  Saved {len(df):,} rows → {out_path}\n")

    print("Done! All data combined.")


if __name__ == "__main__":
    main()
