"""
Aggregate raw UK police street-crime CSV into monthly counts ready for the
training pipeline.

The input is the combined all_street.csv produced by combine_police_data.py
(or any raw police download with the standard columns).

Usage
-----
    python -m src.scripts.aggregate_crime_data \
        --input  data/all_street.csv \
        --output data/monthly_counts.csv \
        --start  2020-01 \
        --end    2023-12

    # Restrict to specific LSOAs:
    python -m src.scripts.aggregate_crime_data \
        --input  data/all_street.csv \
        --output data/monthly_counts.csv \
        --start  2020-01 --end 2023-12 \
        --lsoas  E01000001 E01000002

Output columns
--------------
    Month, LSOA code, LSOA name, Crime type, crime_count
"""

import argparse
from pathlib import Path

import pandas as pd


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=["Month", "LSOA code", "LSOA name", "Crime type"],
        dtype=str,
        keep_default_na=False,
    )


def aggregate(
    raw: pd.DataFrame,
    start: str,
    end: str,
    lsoa_codes: list[str] | None = None,
) -> pd.DataFrame:
    df = raw.copy()

    # drop rows missing key fields
    df = df[
        df["Month"].str.strip().ne("")
        & df["LSOA code"].str.strip().ne("")
        & df["Crime type"].str.strip().ne("")
    ]

    # parse Month (raw format is "YYYY-MM") → month-start timestamp
    df["Month"] = pd.to_datetime(df["Month"].str.strip(), format="%Y-%m")

    # date filter
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.offsets.MonthEnd(0)
    df = df[(df["Month"] >= start_ts) & (df["Month"] <= end_ts)]

    if lsoa_codes:
        df = df[df["LSOA code"].isin(lsoa_codes)]

    if df.empty:
        return pd.DataFrame(
            columns=["Month", "LSOA code", "LSOA name", "Crime type", "crime_count"]
        )

    # keep last known LSOA name per code (some files have slight name variants)
    lsoa_names = (
        df.sort_values("Month")
        .drop_duplicates("LSOA code", keep="last")
        .set_index("LSOA code")["LSOA name"]
    )

    counts = (
        df.groupby(["Month", "LSOA code", "Crime type"], as_index=False)
        .size()
        .rename(columns={"size": "crime_count"})
    )
    counts["LSOA name"] = counts["LSOA code"].map(lsoa_names)
    counts = counts[["Month", "LSOA code", "LSOA name", "Crime type", "crime_count"]]
    counts = counts.sort_values(["LSOA code", "Crime type", "Month"]).reset_index(drop=True)
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate raw UK police street-crime CSV into monthly counts."
    )
    parser.add_argument("--input", required=True, type=Path, help="Raw all_street.csv")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument(
        "--start", required=True, metavar="YYYY-MM",
        help="First month to include (e.g. 2020-01)",
    )
    parser.add_argument(
        "--end", required=True, metavar="YYYY-MM",
        help="Last month to include (e.g. 2023-12)",
    )
    parser.add_argument(
        "--lsoas", nargs="+", metavar="CODE",
        help="Optional: restrict to these LSOA codes only",
    )
    args = parser.parse_args()

    print(f"Loading {args.input} …")
    raw = load_raw(args.input)
    print(f"  {len(raw):,} raw rows")

    counts = aggregate(raw, args.start, args.end, args.lsoas)
    print(f"  {len(counts):,} aggregated rows across {counts['LSOA code'].nunique()} LSOAs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
