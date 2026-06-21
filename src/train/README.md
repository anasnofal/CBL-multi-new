# `src/train` — Crime Forecasting Training Pipeline

This module trains time-series models that forecast monthly crime counts per
**LSOA** (Lower-layer Super Output Area) and **crime type**. It runs rolling
cross-validation to pick the best model for each series, produces a 12-month
forecast with uncertainty bands, and explains each forecast with the drivers
behind it.

## What it does

For every selected LSOA, and for each crime type within it, the pipeline:

1. **Builds a clean monthly series** — fills *internal* month gaps within each
   LSOA's observed date range with zero counts (it does not back-fill history
   outside that range) and interpolates any extra feature columns ([data.py](data.py)).
2. **Rolling cross-validation** — trains on a sliding window and tests on the
   next few months, repeated across the history, scoring each model on
   **MAE**, **RMSE**, and **sMAPE** ([evaluate.py](evaluate.py)).
3. **Picks the best model per crime type** — by the metric you choose
   (`--best-by`, default MAE).
4. **Forecasts 12 months ahead** using the winning model, with confidence
   intervals ([worker.py](worker.py)).
5. **Explains the forecast** — SHAP drivers for XGBoost, component/CI
   breakdown for SARIMA ([explain.py](explain.py)).

Work is parallelised across LSOAs and is **resumable**: each finished LSOA drops
a sentinel file in `lsoa_cache/`, so an interrupted run picks up where it left
off on the next invocation.

## Models

Registered in [models/__init__.py](models/__init__.py):

| Name      | Notes |
|-----------|-------|
| `XGBoost` | Default. Global model with lag/rolling/seasonal features; optional fast hyperparameter pre-scan. |
| `SARIMA`  | Default. Auto-selects (p,d,q)(P,D,Q,12) order from a small grid. |
| `Prophet` | Available but off by default — enable with `--models XGBoost SARIMA Prophet`. |

## Input data

A **pre-aggregated CSV** with these base columns:

```
Month, LSOA code, LSOA name, Crime type, crime_count
```

Any additional columns are treated as **extra features** (exogenous regressors)
when you pass `--use-extra-features`.

## Usage

Run as a module from the repo root:

```bash
# Train all LSOAs, full output
python -m src.train --data path/to/data.csv --all-lsoas --output-dir results/

# Train specific LSOAs
python -m src.train --data path/to/data.csv --lsoas E01012480 E01017991

# Read LSOA codes from a file (one per line)
python -m src.train --data path/to/data.csv --lsoa-file lsoas.txt

# Retrain everything, ignoring previous progress
python -m src.train --data path/to/data.csv --all-lsoas --no-resume

# Limit parallel workers
python -m src.train --data path/to/data.csv --all-lsoas --n-jobs 4
```

Exactly one of `--lsoas`, `--all-lsoas`, or `--lsoa-file` is required.

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *(required)* | Path to the pre-aggregated CSV. |
| `--output-dir` | `notebooks/forecast_outputs` | Where results are written. |
| `--lsoas` / `--all-lsoas` / `--lsoa-file` | *(one required)* | Which LSOAs to train. |
| `--models` | `XGBoost SARIMA` | Models to compare. |
| `--crime-types` | all | Restrict to specific crime types. |
| `--max-lsoas` | none | Cap on number of LSOAs (highest-volume first). |
| `--min-total-crimes` | 0 | Skip LSOAs below this total crime count. |
| `--use-extra-features` | off | Use non-base columns as regressors. |
| `--rolling-train-months` | 24 | CV training window length. |
| `--rolling-test-months` | 3 | CV test window length. |
| `--rolling-step-months` | 12 | Step between CV folds. |
| `--forecast-horizon` | 12 | Months to forecast ahead. |
| `--best-by` | `mae` | Metric for picking the best model (`mae`/`rmse`/`smape`). |
| `--n-jobs` | -1 (all cores) | Parallel workers across LSOAs. |
| `--no-resume` | off | Ignore sentinels and retrain everything. |
| `--skip-xgb-tuning` | off | Skip the XGBoost hyperparameter pre-scan. |
| `--skip-plots` | off | Don't generate per-LSOA plots. |
| `--lean` | off | Save only the combined forecast CSV (much less disk). |
| `--forecast-cutoff` | none | Force all LSOAs to forecast from the same month (`YYYY-MM`). |

## Outputs

Written to `--output-dir`:

| File | Contents |
|------|----------|
| `12month_forecast.csv` | 12-month forecasts for all LSOAs, with confidence bands. |
| `forecast_explanations.csv` | Per-forecast drivers / CI (non-lean mode). |
| `{run}_aggregate_mean_metrics.csv` | Mean CV metrics per model. |
| `{run}_aggregate_pooled_metrics.csv` | Pooled CV metrics across folds. |
| `{run}_combined_model_comparison.csv` | All per-fold CV metrics. |
| `{run}_best_models.csv` | Winning model per LSOA / crime type. |
| `{lsoa}_*.csv` | Per-LSOA detail files (non-lean mode). |
| `xgb_tuned_params.json` | Selected XGBoost hyperparameters (when tuning runs). |
| `lsoa_cache/` | Resume sentinels — one empty file per completed LSOA. |

`{run}` is `all_lsoas`, a single LSOA code, or `selected_N_lsoas`.

### Lean mode

`--lean` writes a single combined `12month_forecast.csv` (predictions + CI +
drivers) and skips per-LSOA files, aggregate metric tables, and plots —
dramatically cutting disk usage for large runs.

## Resume behaviour

Each completed LSOA touches `lsoa_cache/{lsoa_code}`. On the next run, those
LSOAs are skipped (and their results reloaded in non-lean mode). Use
`--no-resume` to wipe the cache and start fresh. A `Ctrl+C` mid-run keeps every
LSOA that already finished.

## Module layout

| File | Role |
|------|------|
| [main.py](main.py) | CLI argument parsing and entry point. |
| [pipeline.py](pipeline.py) | Orchestration: LSOA selection, parallelism, resume, aggregation. |
| [worker.py](worker.py) | Per-LSOA training, forecasting, explanation, output. |
| [data.py](data.py) | Loading, LSOA selection, monthly frame construction. |
| [evaluate.py](evaluate.py) | Rolling CV folds, scoring, best-model selection. |
| [explain.py](explain.py) | SHAP (XGBoost) and component (SARIMA) explanations. |
| [tune.py](tune.py) | XGBoost hyperparameter pre-scan. |
| [models/](models/) | `BaseForecaster` + XGBoost / SARIMA / Prophet implementations. |
| [plots.py](plots.py) | Per-LSOA and aggregate report figures. |
| [config.py](config.py) | Defaults, column schemas, model grids, seeding. |
| [progress.py](progress.py) | Legacy JSON progress tracker. |
</content>
</invoke>
