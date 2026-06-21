"""
Training pipeline orchestration.

Handles LSOA selection, parallelism, resume logic, and aggregate output.
"""

from contextlib import contextmanager
from pathlib import Path

import joblib
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .config import (
    BASE_COLUMNS,
    CV_FORECAST_COLUMNS,
    DEFAULT_OUTPUT_DIR,
    FORECAST_HORIZON,
    METRIC_COLUMNS,
    MODEL_NAMES,
    ROLLING_STEP_MONTHS,
    ROLLING_TEST_MONTHS,
    ROLLING_TRAIN_MONTHS,
    seed_everything,
)
from .data import choose_lsoas, load_data
from .helper import (
    accumulate_cv_forecasts,
    accumulate_cv_metrics,
    append_csv,
    cv_metrics_to_frame,
    pooled_forecasts_to_frame,
)
from .plots import plot_aggregate, plot_report
from .tune import tune_xgb_params
from .worker import load_lsoa_result, process_lsoa


@contextmanager
def _tqdm_joblib(tqdm_bar):
    """Hook a tqdm bar into joblib so it updates on task completion, not submission."""

    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_bar.close()


def train(
    data,
    output_dir=DEFAULT_OUTPUT_DIR,
    use_extra_features=False,
    models=None,
    lsoa_codes=None,
    all_lsoas=False,
    max_lsoas=None,
    min_total_crimes=0,
    crime_types=None,
    train_months=ROLLING_TRAIN_MONTHS,
    test_months=ROLLING_TEST_MONTHS,
    step_months=ROLLING_STEP_MONTHS,
    forecast_horizon=FORECAST_HORIZON,
    best_by="mae",
    save_plots=True,
    n_jobs=-1,
    resume=True,
    tune_xgb=True,
    lean=False,
    forecast_cutoff=None,
):
    """
    Train crime forecasting models on pre-aggregated data.

    Parameters
    ----------
    data : str | Path | pd.DataFrame
        Pre-aggregated CSV with columns:
        Month, LSOA code, LSOA name, Crime type, crime_count, [extra features...].
    use_extra_features : bool
        Pass extra columns to models as exogenous regressors.
    n_jobs : int
        Parallel workers for the LSOA loop. -1 = all CPU cores.
    resume : bool
        Skip LSOAs already recorded in progress.json and reload their CSV results.
        Pass resume=False (or --no-resume on the CLI) to retrain everything.
    """
    seed_everything()

    if not isinstance(data, pd.DataFrame):
        data = load_data(data)

    models = models or MODEL_NAMES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_features = (
        [c for c in data.columns if c not in BASE_COLUMNS] if use_extra_features else []
    )
    if crime_types:
        data = data[data["Crime type"].isin(crime_types)].copy()

    selected_lsoas = choose_lsoas(
        data, lsoa_codes, all_lsoas, max_lsoas, min_total_crimes
    )
    data = data[data["LSOA code"].isin(selected_lsoas)].copy()

    run_name = (
        "all_lsoas"
        if all_lsoas
        else (
            selected_lsoas[0]
            if len(selected_lsoas) == 1
            else f"selected_{len(selected_lsoas)}_lsoas"
        )
    )

    # ── output paths ──────────────────────────────────────────────────────────
    combined_metrics_path = output_dir / f"{run_name}_combined_model_comparison.csv"
    combined_cv_path = output_dir / f"{run_name}_combined_cv_forecasts.csv"
    best_models_path = output_dir / f"{run_name}_best_models.csv"
    mean_metrics_path = output_dir / f"{run_name}_aggregate_mean_metrics.csv"
    pooled_metrics_path = output_dir / f"{run_name}_aggregate_pooled_metrics.csv"
    forecast_path = output_dir / "12month_forecast.csv"
    cv_forecast_path = output_dir / "cv_forecasts.csv" if lean else None
    # In lean mode forecast_path IS the combined file (predictions + CI + explanation).
    # In non-lean mode a separate explanations file is written alongside the forecast.
    explain_path = None if lean else output_dir / "forecast_explanations.csv"

    # ── resume: sentinel files in lsoa_cache/ replace progress.json ──────────
    # Each worker touches lsoa_cache/{lsoa_code} on completion, so results

    cache_dir = output_dir / "lsoa_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not resume:
        # Full reset: wipe sentinels so every LSOA is re-trained
        for f in cache_dir.iterdir():
            f.unlink()
        # Clear shared CSVs so we start fresh
        for p in [forecast_path, cv_forecast_path, explain_path]:
            if p and p.exists():
                p.unlink()

    done = {c for c in selected_lsoas if (cache_dir / c).exists()}
    pending = [c for c in selected_lsoas if c not in done]

    # In lean mode the forecast CSV is the only persistent output for done LSOAs.

    if lean and done and not forecast_path.exists():
        print(
            f"  Warning: {len(done)} sentinel(s) found but {forecast_path.name} is missing. "
            "Clearing sentinels and retraining all LSOAs."
        )
        for f in cache_dir.iterdir():
            f.unlink()
        done = set()
        pending = list(selected_lsoas)

    if done:
        print(
            f"Resuming: {len(done)} LSOA(s) already trained, {len(pending)} to train."
        )

    if not lean:
        # Non-lean aggregate files are always rebuilt from loaded + new results,
        # so clear them every run to avoid duplicate rows on resume.
        for path in [
            combined_metrics_path,
            combined_cv_path,
            best_models_path,
            mean_metrics_path,
            pooled_metrics_path,
            forecast_path,
            explain_path,
        ]:
            if path.exists():
                path.unlink()

    # Single groupby — one pass over the data instead of one filter per LSOA
    lsoa_datasets = {
        code: group.reset_index(drop=True)
        for code, group in data.groupby("LSOA code", sort=False)
        if code in set(selected_lsoas)
    }

    # ── XGBoost hyperparameter tuning ─────────────────────────────────────────
    # Tuning uses ALL CV folds on a dedicated pool of LSOAs spread across the
    # volume distribution.  Those LSOAs are then excluded from the main loop so
    # their reported metrics are not biased by the param selection.
    xgb_params = None
    tuning_lsoas = frozenset()
    if tune_xgb and "XGBoost" in models and lsoa_datasets:
        print("Tuning XGBoost hyperparameters (full CV, multi-LSOA search)…")
        xgb_params, tuning_lsoas = tune_xgb_params(
            lsoa_datasets,
            extra_features,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
        )
        print(f"  Best XGBoost params: {xgb_params}")
        print(f"  {len(tuning_lsoas)} tuning LSOA(s) excluded from reported metrics.")
        import json

        (output_dir / "xgb_tuned_params.json").write_text(
            json.dumps(xgb_params, indent=2)
        )

        # Remove tuning LSOAs from all downstream data structures
        for code in tuning_lsoas:
            lsoa_datasets.pop(code, None)
        selected_lsoas = [c for c in selected_lsoas if c not in tuning_lsoas]
        done = done - tuning_lsoas
        pending = [c for c in pending if c not in tuning_lsoas]

    # In lean mode, done LSOAs' forecasts are already in the shared CSVs.
    # In non-lean mode, reload their per-LSOA CSV files.
    # Computed after tuning exclusion so we don't load results for tuning LSOAs.
    loaded_results = (
        [] if lean else [load_lsoa_result(code, output_dir) for code in done]
    )

    # Train pending LSOAs in parallel — each worker writes its own output and
    # touches its sentinel file, so a Ctrl+C preserves all completed work.
    if pending:
        with _tqdm_joblib(tqdm(total=len(pending), desc="Training LSOAs")):
            new_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(process_lsoa)(
                    lsoa_code,
                    lsoa_datasets[lsoa_code],
                    models,
                    extra_features,
                    train_months,
                    test_months,
                    step_months,
                    forecast_horizon,
                    best_by,
                    False if lean else save_plots,
                    output_dir,
                    xgb_params,
                    lean,
                    forecast_path,
                    cv_forecast_path,
                    explain_path,
                    forecast_cutoff,
                )
                for lsoa_code in pending
            )
    else:
        new_results = []

    # ── sequential aggregate writes (no race conditions) ─────────────────────
    cv_metric_totals = {}
    cv_forecast_totals = {}

    for r in loaded_results + new_results:
        if r is None:
            continue
        if not r["forecast_rows"].empty:
            append_csv(r["forecast_rows"], forecast_path)
        if not lean:
            if not r["best"].empty:
                append_csv(r["best"], best_models_path)
            append_csv(r["metrics"], combined_metrics_path, METRIC_COLUMNS)
            append_csv(r["cv_forecasts"], combined_cv_path, CV_FORECAST_COLUMNS)
            accumulate_cv_metrics(cv_metric_totals, r["metrics"])
            accumulate_cv_forecasts(cv_forecast_totals, r["cv_forecasts"])

    if not lean:
        mean_metrics = cv_metrics_to_frame(cv_metric_totals)
        pooled_metrics = pooled_forecasts_to_frame(cv_forecast_totals)
        mean_metrics.to_csv(mean_metrics_path, index=False)
        pooled_metrics.to_csv(pooled_metrics_path, index=False)
        report_path = plot_report(output_dir, run_name)
        if report_path:
            print(f"Report figure: {report_path}")
    else:
        mean_metrics = pd.DataFrame()
        pooled_metrics = pd.DataFrame()

    return {
        "output_dir": output_dir,
        "lsoas": selected_lsoas,
        "forecast": forecast_path,
        "mean_metrics": mean_metrics_path,
        "pooled_metrics": pooled_metrics_path,
        "mean_metrics_table": mean_metrics,
        "pooled_metrics_table": pooled_metrics,
    }
