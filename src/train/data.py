from pathlib import Path

import pandas as pd

from .config import BASE_COLUMNS
from .helper import month_start


def choose_lsoas(
    data, lsoa_codes=None, all_lsoas=False, max_lsoas=None, min_total_crimes=0
):
    """Select which LSOAs to train on, ordered by total crime count descending."""
    totals = (
        data.groupby("LSOA code", as_index=False)["crime_count"]
        .sum()
        .sort_values("crime_count", ascending=False)
    )
    if min_total_crimes:
        totals = totals[totals["crime_count"] >= min_total_crimes]

    if all_lsoas:
        selected = totals["LSOA code"].tolist()
    else:
        selected = [
            code for code in (lsoa_codes or []) if code in set(totals["LSOA code"])
        ]

    return selected[:max_lsoas] if max_lsoas else selected


def build_lsoa_frame(data, lsoa_code):
    """
    Extract one LSOA from the pre-aggregated data, fill month gaps with zero crime counts,
    and interpolate any missing values in extra feature columns.
    """
    lsoa_data = data[data["LSOA code"].eq(lsoa_code)].copy()
    lsoa_name = lsoa_data["LSOA name"].dropna().iloc[0]
    extra_cols = [c for c in data.columns if c not in BASE_COLUMNS]

    months = pd.date_range(
        lsoa_data["Month"].min(), lsoa_data["Month"].max(), freq="MS"
    )
    crime_types = sorted(lsoa_data["Crime type"].dropna().unique())
    grid = pd.MultiIndex.from_product(
        [months, crime_types], names=["Month", "Crime type"]
    ).to_frame(index=False)

    agg = {"crime_count": ("crime_count", "sum")}
    for col in extra_cols:
        agg[col] = (col, "mean")
    monthly = lsoa_data.groupby(["Month", "Crime type"], as_index=False).agg(**agg)
    monthly = grid.merge(monthly, on=["Month", "Crime type"], how="left")
    monthly["crime_count"] = monthly["crime_count"].fillna(0)
    monthly["LSOA code"] = lsoa_code
    monthly["LSOA name"] = lsoa_name

    for col in extra_cols:
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce")
        monthly[col] = (
            monthly.sort_values("Month")
            .groupby("Crime type")[col]
            .transform(lambda s: s.interpolate().ffill().bfill())
        )

    return monthly.sort_values(["Crime type", "Month"]).reset_index(drop=True)


def load_data(path, date_col="Month"):
    """Load a pre-aggregated CSV. Normalises the date column to month-start timestamps."""
    df = pd.read_csv(path, parse_dates=[date_col])
    df[date_col] = month_start(df[date_col])
    return df
