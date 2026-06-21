# `src/scripts` — Data Prep & Utility Scripts

Standalone scripts for building the inputs to the [training pipeline](../train)
and for evaluating its outputs. Run any of them as a module from the repo root:

```bash
uv run python -m src.scripts.<name> [options]
```

## Typical order

1. **`combine_police_data`** — merge a folder of raw monthly police downloads
   (from [data.police.uk](https://data.police.uk/data/)) into single
   `all_street.csv` / `all_outcomes.csv` / `all_stop_and_search.csv` files.
2. **`aggregate_crime_data`** — turn `all_street.csv` into monthly LSOA counts
   (`Month, LSOA code, LSOA name, Crime type, crime_count`) ready for training.
3. *(optional)* **`fetch_temperature`** — add monthly average temperature per LSOA
   as an extra feature column for `--use-extra-features`.
4. Train with [`src/train`](../train), then **`evaluate_holdout`** to score the
   forecast against out-of-sample actuals.

## Scripts

| Script | What it does |
|--------|--------------|
| `combine_police_data` | Combine raw per-month police CSVs into 3 dataset-wide files. |
| `aggregate_crime_data` | Aggregate `all_street.csv` into monthly LSOA × crime-type counts. Supports `--start`/`--end` and `--lsoas`. |
| `fetch_temperature` | Fetch monthly avg temperature per LSOA and merge it into the counts CSV. Coords via `--coords` or extracted from `--raw-crime`. |
| `identify_hotspots` | Compute crime hotspots with Local Moran's I (per force/month or stable across the dataset). Needs force boundaries + LSOA shapefiles. |
| `hotspot_to_lsoa_file` | Convert a hotspot dictionary JSON into an LSOA list file for `python -m src.train --lsoa-file …`. |
| `evaluate_holdout` | Score the published 12-month forecast against true out-of-sample actuals (MAE/RMSE). See [FORECAST_VALIDATION.md](../../FORECAST_VALIDATION.md). |
| `download_data_kaggle` | Download the prepared UK police crime dataset from Kaggle into `data/`. |
| `scrape_police_station/` | Helpers for scraping police station locations. |

</content>
