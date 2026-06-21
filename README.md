# CBL — Crime Forecasting Pipeline

End-to-end pipeline for forecasting monthly crime counts at **LSOA** level across
England & Wales, with an interactive dashboard to explore the results. Models:
SARIMA, XGBoost, and Prophet.

The project has three stages: **prepare data** (scripts) → **train models**
(`src/train`) → **explore forecasts** (`src/dashboard`).

## Setup

```bash
uv sync
```

## Project structure

| Path | What it is |
|------|------------|
| [src/train/](src/train) | The training pipeline — rolling cross-validation, model selection, 12-month forecasts, and explanations. **See [src/train/README.md](src/train/README.md).** |
| [src/dashboard/](src/dashboard) | FastAPI + Vue dashboard for exploring the forecasts. **See [src/dashboard/README.md](src/dashboard/README.md).** |
| [src/scripts/](src/scripts) | Standalone data-prep & utility scripts (aggregate raw police data, fetch temperature, identify hotspots, evaluate holdout). Run with `uv run python -m src.scripts.<name>`. **See [src/scripts/README.md](src/scripts/README.md).** |
| [notebooks/](notebooks) | Exploratory analysis & evaluation notebooks (EDA, CV evaluation, explainability, historical analysis). |


## How to run it

### 1. Prepare the data (`src/scripts`)

Aggregate raw police street data into monthly LSOA counts, and optionally enrich
with temperature:

```bash
# Aggregate (raw data from https://data.police.uk/data/)
uv run python -m src.scripts.aggregate_crime_data \
  --input data/all_street.csv --output data/monthly_counts.csv \
  --start 2023-04 --end 2026-04

# Optional: add temperature as an extra feature
uv run python -m src.scripts.fetch_temperature \
  --counts data/monthly_counts.csv --output data/monthly_counts_with_temp.csv
```

### 2. Train the models (`src/train`)

```bash
uv run python -m src.train --data data/monthly_counts.csv --all-lsoas --output-dir results/
```

Full options, outputs, and resume behaviour are documented in
**[src/train/README.md](src/train/README.md)**.

### 3. Explore the forecasts (`src/dashboard`)

Run the FastAPI backend and the Vue frontend (two terminals). Full setup and run
instructions are in **[src/dashboard/README.md](src/dashboard/README.md)**.
</content>
