"""
Fetch monthly average temperature for every LSOA in a crime-counts CSV and
produce a merged output file ready to pass directly to the training pipeline:

    python -m src.train --data <output> --all-lsoas [--use-extra-features]

Usage
-----
Supply LSOA coordinates via a dedicated file:

    python src/scripts/fetch_temperature.py \\
        --counts crime_counts.csv \\
        --coords lsoa_coords.csv \\
        --output crime_with_temperature.csv

Or extract coordinates automatically from the raw police street-crime CSV:

    python src/scripts/fetch_temperature.py \\
        --counts crime_counts.csv \\
        --raw-crime data/all_street.csv \\
        --output crime_with_temperature.csv

Coordinate file format  (--coords)
------------------------------------
    LSOA code,latitude,longitude
    E01035716,51.5074,-0.1278
    E01035717,51.5091,-0.1301
    ...

Input counts CSV  (--counts)
------------------------------
    Month,LSOA code,LSOA name,Crime type,crime_count
    2020-01-01,E01035716,Westminster 013G,Burglary,3
    ...

Output
------
Same as input counts CSV with an  avg_temperature_c  column appended.
A temperature cache CSV is saved alongside the output so re-runs skip
already-fetched LSOAs.
"""

import argparse
import calendar
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
BATCH_SIZE = 25
SLEEP_SECONDS = 0.15


# ── helpers ───────────────────────────────────────────────────────────────────


def _month_start(series):
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def _api_date_range(months):
    """Return (start_date_str, end_date_str) covering the full range of months."""
    months = _month_start(months)
    index = pd.date_range(months.min(), months.max(), freq="MS")
    start = index.min().strftime("%Y-%m-%d")
    last = index.max()
    end = last.replace(day=calendar.monthrange(last.year, last.month)[1]).strftime(
        "%Y-%m-%d"
    )
    return start, end


def _fetch_batch(locations, start_date, end_date):
    """
    Call Open-Meteo for up to BATCH_SIZE locations and return a DataFrame
    with columns: Month, LSOA code, avg_temperature_c.
    """
    params = {
        "latitude": ",".join(f"{v:.6f}" for v in locations["latitude"]),
        "longitude": ",".join(f"{v:.6f}" for v in locations["longitude"]),
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/London",
        "temperature_unit": "celsius",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=90)
    resp.raise_for_status()

    payload = resp.json()
    items = payload if isinstance(payload, list) else [payload]
    rows = []

    for (_, place), item in zip(locations.iterrows(), items):
        daily = item.get("daily", {})
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(daily.get("time", [])),
                "temp": pd.to_numeric(
                    pd.Series(daily.get("temperature_2m_mean", [])), errors="coerce"
                ),
            }
        ).dropna()
        df["Month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        monthly = df.groupby("Month", as_index=False)["temp"].mean()
        for row in monthly.itertuples(index=False):
            rows.append(
                {
                    "Month": row.Month,
                    "LSOA code": place["LSOA code"],
                    "avg_temperature_c": round(float(row.temp), 4),
                }
            )

    return pd.DataFrame(rows)


# ── coordinate loading ────────────────────────────────────────────────────────


def load_coords_from_file(path):
    """
    Load LSOA coordinates from a CSV with columns:
        LSOA code, latitude, longitude
    """
    df = pd.read_csv(path, dtype={"LSOA code": str})
    required = {"LSOA code", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Coordinate file is missing columns: {missing}")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df[["LSOA code", "latitude", "longitude"]].dropna()


def load_coords_from_crime_csv(crime_csv_path, lsoa_codes, chunk_size=750_000):
    """
    Derive mean centroid coordinates for each LSOA by reading the raw UK
    police street-crime CSV (which has Latitude and Longitude columns).
    """
    code_filter = set(lsoa_codes)
    chunks = []

    reader = pd.read_csv(
        crime_csv_path,
        usecols=["LSOA code", "Latitude", "Longitude"],
        dtype=str,
        chunksize=chunk_size,
        encoding="utf-8-sig",
    )
    for chunk in tqdm(reader, desc="Reading crime CSV for coordinates"):
        chunk = chunk[chunk["LSOA code"].isin(code_filter)].dropna(
            subset=["LSOA code", "Latitude", "Longitude"]
        )
        if chunk.empty:
            continue
        chunk["latitude"] = pd.to_numeric(chunk["Latitude"], errors="coerce")
        chunk["longitude"] = pd.to_numeric(chunk["Longitude"], errors="coerce")
        chunks.append(chunk[["LSOA code", "latitude", "longitude"]])

    if not chunks:
        raise RuntimeError("No matching LSOA coordinates found in the crime CSV.")

    coords = pd.concat(chunks, ignore_index=True)
    return (
        coords.groupby("LSOA code", as_index=False)
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .sort_values("LSOA code")
    )


# ── temperature fetching ──────────────────────────────────────────────────────


def fetch_temperature(locations, start_date, end_date, cache_path=None):
    """
    Fetch monthly temperature for all rows in *locations* (LSOA code, latitude,
    longitude), reusing *cache_path* if supplied.

    Returns a DataFrame with columns: Month, LSOA code, avg_temperature_c.
    """
    cache = pd.DataFrame(columns=["Month", "LSOA code", "avg_temperature_c"])
    if cache_path and Path(cache_path).exists():
        cache = pd.read_csv(cache_path, parse_dates=["Month"])
        cache["Month"] = _month_start(cache["Month"])
        print(
            f"Loaded temperature cache: {cache_path}  ({len(cache['LSOA code'].unique())} LSOAs)"
        )

    already_done = set(cache["LSOA code"].unique())
    missing = locations[~locations["LSOA code"].isin(already_done)]

    if missing.empty:
        print("All LSOAs already in cache — skipping API calls.")
    else:
        print(f"Fetching temperature for {len(missing):,} LSOA(s)…")
        new_frames = []
        for start in tqdm(
            range(0, len(missing), BATCH_SIZE), desc="Open-Meteo batches"
        ):
            batch = missing.iloc[start : start + BATCH_SIZE]
            new_frames.append(_fetch_batch(batch, start_date, end_date))
            time.sleep(SLEEP_SECONDS)

        cache = pd.concat([cache, *new_frames], ignore_index=True)
        cache = cache.drop_duplicates(["Month", "LSOA code"], keep="last").sort_values(
            ["LSOA code", "Month"]
        )

        if cache_path:
            cache.to_csv(cache_path, index=False)
            print(f"Saved temperature cache: {cache_path}")

    return cache[cache["LSOA code"].isin(set(locations["LSOA code"]))]


# ── merge and save ────────────────────────────────────────────────────────────


def merge_temperature(counts, temperature):
    """
    Left-join monthly temperature into the crime counts DataFrame.
    Missing temperature values are left as NaN; build_lsoa_frame will
    interpolate them during training.
    """
    temperature = temperature.copy()
    temperature["Month"] = _month_start(temperature["Month"])
    counts = counts.copy()
    counts["Month"] = _month_start(counts["Month"])
    return counts.merge(
        temperature[["Month", "LSOA code", "avg_temperature_c"]],
        on=["Month", "LSOA code"],
        how="left",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch monthly temperature for a list of LSOAs and merge into a crime-counts CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--counts",
        required=True,
        type=Path,
        help="Pre-aggregated crime counts CSV (Month, LSOA code, LSOA name, Crime type, crime_count).",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Path for the merged output CSV."
    )

    coord_group = parser.add_mutually_exclusive_group(required=True)
    coord_group.add_argument(
        "--coords", type=Path, help="CSV with columns: LSOA code, latitude, longitude."
    )
    coord_group.add_argument(
        "--raw-crime",
        type=Path,
        help="Raw UK police street-crime CSV to extract LSOA centroids from.",
    )

    parser.add_argument(
        "--lsoas",
        nargs="+",
        help="Subset of LSOA codes to process. Defaults to all codes in --counts.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=750_000,
        help="Row chunk size when reading the raw crime CSV (default: 750 000).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # load crime counts
    counts = pd.read_csv(args.counts, parse_dates=["Month"])
    counts["Month"] = _month_start(counts["Month"])

    # filter to requested LSOAs
    all_lsoa_codes = counts["LSOA code"].unique().tolist()
    lsoa_codes = args.lsoas if args.lsoas else all_lsoa_codes
    counts = counts[counts["LSOA code"].isin(lsoa_codes)].copy()

    if counts.empty:
        raise SystemExit(
            "No rows remain after filtering by --lsoas. Check the codes and try again."
        )

    print(
        f"Processing {len(lsoa_codes):,} LSOA(s) over {counts['Month'].nunique()} months."
    )

    # load coordinates
    if args.coords:
        locations = load_coords_from_file(args.coords)
        locations = locations[locations["LSOA code"].isin(lsoa_codes)]
    else:
        locations = load_coords_from_crime_csv(
            args.raw_crime, lsoa_codes, args.chunk_size
        )

    missing_coords = set(lsoa_codes) - set(locations["LSOA code"])
    if missing_coords:
        print(
            f"Warning: no coordinates found for {len(missing_coords)} LSOA(s) — they will have no temperature data."
        )
        print(
            f"  Missing: {sorted(missing_coords)[:10]}{' …' if len(missing_coords) > 10 else ''}"
        )

    # fetch temperature
    start_date, end_date = _api_date_range(counts["Month"])
    cache_path = args.output.with_name(args.output.stem + "_temperature_cache.csv")
    temperature = fetch_temperature(
        locations, start_date, end_date, cache_path=cache_path
    )

    # merge and save
    output = merge_temperature(counts, temperature)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    n_with_temp = output["avg_temperature_c"].notna().sum()
    n_total = len(output)
    print(f"\nSaved: {args.output}")
    print(
        f"Rows: {n_total:,}  |  Rows with temperature: {n_with_temp:,}  |  Missing: {n_total - n_with_temp:,}"
    )
    print(
        f"\nTo train:  python -m src.train --data {args.output} --all-lsoas --use-extra-features"
    )


if __name__ == "__main__":
    main()
