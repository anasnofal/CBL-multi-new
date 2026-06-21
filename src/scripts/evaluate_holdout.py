"""
Compare the 12-month forecast against March and April 2026 actuals.

These two months were never included in training (cutoff was March 2025),
so this is a true out-of-sample evaluation.

Usage:
    uv run python -m src.scripts.evaluate_holdout
"""

import glob
from pathlib import Path

import pandas as pd

from src.train.helper import regression_scores

RAW_DIR = Path("/Users/anas/Downloads/3e1fc65f1d2850d53c164fb4da0c7be79942e4f7")
FORECAST_CSV = Path("src/dashboard/app/data/12month_forecast.csv")
TEST_MONTHS = ["2026-03", "2026-04"]

# ── 1. read and aggregate raw actuals ────────────────────────────────────────

files = glob.glob(str(RAW_DIR / "**" / "*-street.csv"), recursive=True)
print(f"Reading {len(files)} raw files …")

raw = pd.concat(
    [
        pd.read_csv(
            f,
            usecols=["Month", "LSOA code", "LSOA name", "Crime type"],
            dtype=str,
            keep_default_na=False,
        )
        for f in files
    ],
    ignore_index=True,
)

raw = raw[
    raw["Month"].str.strip().ne("")
    & raw["LSOA code"].str.strip().ne("")
    & raw["Crime type"].str.strip().ne("")
]
raw["Month"] = pd.to_datetime(raw["Month"].str.strip(), format="%Y-%m")

actuals = (
    raw.groupby(["Month", "LSOA code", "LSOA name", "Crime type"], as_index=False)
    .size()
    .rename(columns={"size": "actual"})
)

print(
    f"Actuals: {len(actuals):,} rows, "
    f"{actuals['LSOA code'].nunique():,} LSOAs, "
    f"months: {sorted(actuals['Month'].dt.strftime('%Y-%m').unique())}"
)

# ── 2. load forecast, filter to test months ───────────────────────────────────

fc = pd.read_csv(FORECAST_CSV, parse_dates=["Month"])
fc = fc[fc["Month"].dt.strftime("%Y-%m").isin(TEST_MONTHS)][
    ["LSOA code", "Crime type", "Month", "model", "predicted"]
].copy()

print(f"Forecast rows for test months: {len(fc):,}")

# ── 3. join ────────────────────────────────────────────────────────────────────

merged = fc.merge(
    actuals[["LSOA code", "Crime type", "Month", "actual"]],
    on=["LSOA code", "Crime type", "Month"],
    how="inner",
)

print(
    f"Matched rows (forecast ∩ actuals): {len(merged):,} "
    f"({merged['LSOA code'].nunique():,} LSOAs)\n"
)

# ── 4. metrics overall ────────────────────────────────────────────────────────

scores = regression_scores(merged["actual"], merged["predicted"])
print("=== Overall (Mar + Apr 2026) ===")
print(f"  MAE  : {scores['mae']:.3f}")
print(f"  RMSE : {scores['rmse']:.3f}")
print(f"  SMAPE: {scores['smape']:.1f}%")
print(f"  N    : {len(merged):,}")

# ── 5. per-month ──────────────────────────────────────────────────────────────

print("\n=== Per month ===")
for month, grp in merged.groupby("Month"):
    s = regression_scores(grp["actual"], grp["predicted"])
    print(
        f"  {month.strftime('%Y-%m')}  MAE={s['mae']:.3f}  RMSE={s['rmse']:.3f}  "
        f"SMAPE={s['smape']:.1f}%  N={len(grp):,}"
    )

# ── 6. per-model ──────────────────────────────────────────────────────────────

print("\n=== Per model ===")
for model, grp in merged.groupby("model"):
    s = regression_scores(grp["actual"], grp["predicted"])
    print(
        f"  {model:<10}  MAE={s['mae']:.3f}  RMSE={s['rmse']:.3f}  "
        f"SMAPE={s['smape']:.1f}%  N={len(grp):,}"
    )

# ── 7. per-crime-type ─────────────────────────────────────────────────────────

print("\n=== Per crime type (XGBoost only, sorted by MAE) ===")
xgb = merged[merged["model"] == "XGBoost"]
rows = []
for ct, grp in xgb.groupby("Crime type"):
    s = regression_scores(grp["actual"], grp["predicted"])
    rows.append({"crime_type": ct, "n": len(grp), **s})
ct_df = pd.DataFrame(rows).sort_values("mae")
for _, r in ct_df.iterrows():
    print(f"  {r['crime_type']:<45}  MAE={r['mae']:.3f}  N={int(r['n']):,}")
