"""
Per-LSOA training worker.

process_lsoa  — trains one LSOA, persists results, marks itself done.
load_lsoa_result — reloads per-LSOA CSV files for non-lean resume.
"""

import fcntl
import os
from pathlib import Path

import pandas as pd

from .data import build_lsoa_frame
from .evaluate import cross_validate_lsoa, summarize_cv_results
from .explain import build_explanation_df, explain_sarima_forecast, explain_xgb_forecast
from .models import fit_forecaster
from .models.xgboost import XGBoostGlobalForecaster


def _append_csv_locked(df: pd.DataFrame, path: Path) -> None:
    """
    Append df rows to a shared CSV file using an exclusive OS-level lock.
    Safe to call from multiple parallel worker processes simultaneously.
    """
    if df.empty:
        return
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            write_header = os.fstat(fh.fileno()).st_size == 0
            df.to_csv(fh, header=write_header, index=False)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def process_lsoa(
    lsoa_code: str,
    lsoa_subset: pd.DataFrame,
    models: list,
    extra_features: list,
    train_months: int,
    test_months: int,
    step_months: int,
    forecast_horizon: int,
    best_by: str,
    save_plots: bool,
    output_dir: Path,
    xgb_params: dict | None = None,
    lean: bool = False,
    forecast_path: Path | None = None,
    cv_path: Path | None = None,
    explain_path: Path | None = None,
    forecast_cutoff: pd.Timestamp | None = None,
) -> dict | None:
    """
    Run the full pipeline for one LSOA.

    In lean mode all output is written directly to the shared CSVs (forecast_path,
    cv_path) under a file lock, so results are durable even if the run is interrupted.
    A zero-byte sentinel  output_dir/lsoa_cache/{lsoa_code}  is created on success
    and is the only resume indicator needed.

    In non-lean mode per-LSOA CSV files are written as before; the sentinel is
    still created so resume works on the next run without progress.json.
    """
    try:
        lsoa_table = build_lsoa_frame(lsoa_subset, lsoa_code)
        lsoa_name = lsoa_table["LSOA name"].iloc[0]

        # ── rolling cross-validation ──────────────────────────────────────
        metrics, cv_forecasts = cross_validate_lsoa(
            lsoa_table,
            models,
            extra_features,
            train_months,
            test_months,
            step_months,
            xgb_params=xgb_params,
        )
        summary, best = summarize_cv_results(metrics, sort_by=best_by)

        if best.empty and metrics.empty:
            months_avail = lsoa_table["Month"].nunique()
            print(
                f"  [no folds] {lsoa_code}: {months_avail} months available, "
                f"need {train_months + test_months} for one CV fold — skipping."
            )
            (output_dir / "lsoa_cache" / lsoa_code).touch()
            return {
                "metrics": metrics,
                "cv_forecasts": cv_forecasts,
                "best": best,
                "forecast_rows": pd.DataFrame(),
            }

        if not lean:
            metrics.to_csv(
                output_dir / f"{lsoa_code}_crime_type_model_comparison.csv", index=False
            )
            cv_forecasts.to_csv(
                output_dir / f"{lsoa_code}_crime_type_forecasts.csv", index=False
            )
            if not summary.empty:
                summary.to_csv(
                    output_dir / f"{lsoa_code}_crime_type_model_summary.csv",
                    index=False,
                )
            if not best.empty:
                best.to_csv(
                    output_dir / f"{lsoa_code}_best_model_by_crime_type.csv",
                    index=False,
                )

        # ── 12-month forecast ─────────────────────────────────────────────
        # Use forecast_cutoff if provided so all LSOAs share the same window.
        # Without it each LSOA starts from its own last data month, which
        # causes window misalignment when LSOAs have different data end dates.
        forecast_start = (
            forecast_cutoff
            if forecast_cutoff is not None
            else lsoa_table["Month"].max()
        )
        future_months = pd.date_range(
            forecast_start + pd.offsets.MonthBegin(1),
            periods=forecast_horizon,
            freq="MS",
        )
        forecast_rows = []
        best_by_crime = best.set_index("Crime type")["model"] if not best.empty else {}

        xgb_crime_types = [ct for ct, m in best_by_crime.items() if m == "XGBoost"]
        global_forecaster = None
        if xgb_crime_types and "XGBoost" in models:
            try:
                global_forecaster = XGBoostGlobalForecaster.fit(
                    lsoa_table, extra_features, xgb_params=xgb_params
                )
            except Exception as err:
                print(f"  [skip global xgb] {lsoa_code}: {err}")

        # Collect SARIMA forecasters so we can explain them after the forecast loop
        sarima_forecasters: dict = {}  # {crime_type: (forecaster, crime_series)}

        for crime_type, crime_frame in lsoa_table.groupby("Crime type", sort=True):
            best_model = best_by_crime.get(crime_type)
            if best_model is None:
                continue
            try:
                if best_model == "XGBoost" and global_forecaster is not None:
                    preds = global_forecaster.predict_for_crime_type(
                        crime_type, forecast_horizon
                    )
                else:
                    forecaster = fit_forecaster(best_model, crime_frame, extra_features)
                    preds = forecaster.predict(forecast_horizon)
                    if best_model == "SARIMA":
                        sarima_forecasters[crime_type] = (
                            forecaster,
                            crime_frame.set_index("Month")["crime_count"]
                            .astype(float)
                            .asfreq("MS"),
                        )
            except Exception as err:
                print(
                    f"  [skip forecast] {lsoa_code} / {crime_type} / {best_model}: {err}"
                )
                preds = [float("nan")] * forecast_horizon

            for month, pred in zip(future_months, preds):
                forecast_rows.append(
                    {
                        "LSOA code": lsoa_code,
                        "LSOA name": lsoa_name,
                        "Crime type": crime_type,
                        "model": best_model,
                        "Month": month,
                        "predicted": float(pred),
                    }
                )

        forecast_df = pd.DataFrame(forecast_rows) if forecast_rows else pd.DataFrame()
        # Drop rows where the model failed — NaN predictions are useless in output
        if not forecast_df.empty:
            forecast_df = forecast_df.dropna(subset=["predicted"])

        # ── explanation generation ────────────────────────────────────────────
        all_explanations = []
        if global_forecaster is not None:
            try:
                all_explanations.extend(
                    explain_xgb_forecast(
                        global_forecaster,
                        forecast_horizon,
                        cv_forecasts=cv_forecasts,
                        forecast_cutoff=forecast_cutoff,
                    )
                )
            except Exception as err:
                print(f"  [skip xgb explain] {lsoa_code}: {err}")

        for ct, (sarima_f, crime_series) in sarima_forecasters.items():
            try:
                all_explanations.extend(
                    explain_sarima_forecast(
                        sarima_f,
                        crime_series,
                        ct,
                        forecast_horizon,
                        forecast_cutoff=forecast_cutoff,
                    )
                )
            except Exception as err:
                print(f"  [skip sarima explain] {lsoa_code} / {ct}: {err}")

        explain_df = build_explanation_df(all_explanations, lsoa_code, lsoa_name)

        # Merge ci_lower / ci_upper from explanations into the forecast so the
        # main output file has predictions + uncertainty bands in one place.
        if not explain_df.empty and not forecast_df.empty:
            ci_cols = explain_df[
                ["LSOA code", "Crime type", "model", "Month", "ci_lower", "ci_upper"]
            ].copy()
            forecast_df = forecast_df.merge(
                ci_cols,
                on=["LSOA code", "Crime type", "model", "Month"],
                how="left",
            )

        # ── write output files ────────────────────────────────────────────────
        if lean:
            # ONE combined file: explain_df already contains predicted + CI + drivers.
            # Fall back to forecast_df (prediction-only) if SHAP was unavailable.
            lean_out = explain_df if not explain_df.empty else forecast_df
            if forecast_path and not lean_out.empty:
                _append_csv_locked(lean_out, forecast_path)
            if cv_path and not cv_forecasts.empty:
                _append_csv_locked(cv_forecasts, cv_path)
        else:
            # Non-lean: forecast (+ CI) to shared forecast file, full explanations
            # to a separate shared explanations file, plus per-LSOA detail files.
            if forecast_path and not forecast_df.empty:
                _append_csv_locked(forecast_df, forecast_path)
            if explain_path and not explain_df.empty:
                _append_csv_locked(explain_df, explain_path)
            if not forecast_df.empty:
                forecast_df.to_csv(
                    output_dir / f"{lsoa_code}_12month_forecast.csv", index=False
                )
            if not explain_df.empty:
                explain_df.to_csv(
                    output_dir / f"{lsoa_code}_forecast_explanations.csv", index=False
                )
            if save_plots:
                from .plots import plot_lsoa

                plot_lsoa(lsoa_code, lsoa_table, metrics, cv_forecasts, output_dir)

        # ── sentinel: marks this LSOA done for resume ─────────────────────
        (output_dir / "lsoa_cache" / lsoa_code).touch()

        return {
            "metrics": metrics,
            "cv_forecasts": cv_forecasts,
            "best": best,
            "forecast_rows": pd.DataFrame(),
        }

    except Exception as err:
        print(f"  [ERROR] {lsoa_code}: {err}")
        return None


def load_lsoa_result(lsoa_code: str, output_dir: Path) -> dict:
    """Reload a previously trained LSOA's results from per-LSOA CSV files (non-lean mode)."""

    def _read(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    return {
        "metrics": _read(output_dir / f"{lsoa_code}_crime_type_model_comparison.csv"),
        "cv_forecasts": _read(output_dir / f"{lsoa_code}_crime_type_forecasts.csv"),
        "best": _read(output_dir / f"{lsoa_code}_best_model_by_crime_type.csv"),
        "forecast_rows": _read(output_dir / f"{lsoa_code}_12month_forecast.csv"),
    }
