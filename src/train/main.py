"""
CLI entry point for the crime forecasting pipeline.

Usage
-----
    python -m src.train --data path/to/data.csv --all-lsoas --output-dir results/
    python -m src.train --data path/to/data.csv --lsoas E01012480 E01017991
    python -m src.train --data path/to/data.csv --all-lsoas --no-resume   # retrain all
    python -m src.train --data path/to/data.csv --all-lsoas --n-jobs 4    # limit cores

Outputs (in --output-dir)
-------------------------
    progress.json                              — tracks which LSOAs are trained
    12month_forecast.csv                       — 12-month ahead predictions, all LSOAs
    {run_name}_aggregate_mean_metrics.csv
    {run_name}_aggregate_pooled_metrics.csv
    {lsoa}_crime_type_model_comparison.csv
    {lsoa}_crime_type_forecasts.csv
    {lsoa}_12month_forecast.csv
    {lsoa}_best_model_by_crime_type.csv
"""

import argparse
from pathlib import Path

from .config import (
    DEFAULT_OUTPUT_DIR,
    FORECAST_HORIZON,
    MODEL_NAMES,
    ROLLING_STEP_MONTHS,
    ROLLING_TEST_MONTHS,
    ROLLING_TRAIN_MONTHS,
)
import pandas as pd

from .helper import read_lsoa_file
from .pipeline import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train crime forecasting models.")
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Pre-aggregated CSV: Month, LSOA code, LSOA name, Crime type, crime_count, [extra features...]",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--use-extra-features",
        action="store_true",
        help="Use columns beyond the base five as exogenous regressors.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lsoas", nargs="+")
    group.add_argument("--all-lsoas", action="store_true")
    group.add_argument("--lsoa-file", type=Path)

    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--crime-types", nargs="+")
    parser.add_argument("--max-lsoas", type=int)
    parser.add_argument("--min-total-crimes", type=int, default=0)
    parser.add_argument(
        "--rolling-train-months", type=int, default=ROLLING_TRAIN_MONTHS
    )
    parser.add_argument("--rolling-test-months", type=int, default=ROLLING_TEST_MONTHS)
    parser.add_argument("--rolling-step-months", type=int, default=ROLLING_STEP_MONTHS)
    parser.add_argument("--forecast-horizon", type=int, default=FORECAST_HORIZON)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--best-by",
        choices=["mae", "rmse", "smape"],
        default="mae",
        help="Metric used to select the best model per crime type.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for the LSOA loop. -1 = all CPU cores.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore progress.json and retrain all LSOAs from scratch.",
    )
    parser.add_argument(
        "--skip-xgb-tuning",
        action="store_true",
        help="Skip the fast XGBoost hyperparameter pre-scan and use fixed defaults.",
    )
    parser.add_argument(
        "--lean",
        action="store_true",
        help=(
            "Save only the final 12month_forecast.csv. "
            "Skips per-LSOA CSVs, aggregate metric files, and plots. "
            "Cuts disk usage dramatically."
        ),
    )
    parser.add_argument(
        "--forecast-cutoff",
        type=str,
        default=None,
        help=(
            "Force all LSOAs to use this month as the last training month "
            "(YYYY-MM). Overrides each LSOA's own data end date so every "
            "LSOA forecasts the same window. Example: --forecast-cutoff 2026-02"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    lsoas = list(args.lsoas or [])
    if args.lsoa_file:
        lsoas.extend(read_lsoa_file(args.lsoa_file))

    forecast_cutoff = None
    if args.forecast_cutoff:
        forecast_cutoff = pd.to_datetime(
            args.forecast_cutoff, format="%Y-%m"
        ) + pd.offsets.MonthEnd(0)

    result = train(
        data=args.data,
        output_dir=args.output_dir,
        use_extra_features=args.use_extra_features,
        models=args.models,
        lsoa_codes=lsoas or None,
        all_lsoas=args.all_lsoas,
        max_lsoas=args.max_lsoas,
        min_total_crimes=args.min_total_crimes,
        crime_types=args.crime_types,
        train_months=args.rolling_train_months,
        test_months=args.rolling_test_months,
        step_months=args.rolling_step_months,
        forecast_horizon=args.forecast_horizon,
        best_by=args.best_by,
        save_plots=not args.skip_plots,
        n_jobs=args.n_jobs,
        resume=not args.no_resume,
        tune_xgb=not args.skip_xgb_tuning,
        lean=args.lean,
        forecast_cutoff=forecast_cutoff,
    )

    print(f"\nDone. Output: {result['output_dir']}")
    print(f"12-month forecast: {result['forecast']}")
    if not result["mean_metrics_table"].empty:
        print("\nCV metrics summary:")
        print(result["mean_metrics_table"].to_string(index=False))
