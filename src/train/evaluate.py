import numpy as np
import pandas as pd

from .config import CV_FORECAST_COLUMNS, METRIC_COLUMNS
from .helper import regression_scores
from .models import detect_sarima_order, predict, xgboost_global_predict


def rolling_folds(row_count, train_months, test_months, step_months):
    folds, start, fold_number = [], 0, 1
    while start + train_months + test_months <= row_count:
        train_rows = np.arange(start, start + train_months)
        test_rows = np.arange(start + train_months, start + train_months + test_months)
        folds.append((fold_number, train_rows, test_rows))
        start += step_months
        fold_number += 1
    return folds


def cross_validate_crime_type(
    crime_frame,
    crime_type,
    lsoa_code,
    lsoa_name,
    models,
    extra_features,
    train_months,
    test_months,
    step_months,
    sarima_order=None,
):
    crime_frame = crime_frame.sort_values("Month").reset_index(drop=True)
    folds = rolling_folds(len(crime_frame), train_months, test_months, step_months)
    uses_extra = bool(extra_features)
    metric_rows, forecast_rows = [], []

    for fold, train_index, test_index in folds:
        train_frame = crime_frame.iloc[train_index].copy()
        test_frame = crime_frame.iloc[test_index].copy()
        actual = test_frame["crime_count"].to_numpy(dtype=float)

        for model_name in models:
            base = {
                "LSOA code": lsoa_code,
                "LSOA name": lsoa_name,
                "Crime type": crime_type,
                "model": model_name,
                "uses_extra_features": uses_extra,
                "fold": fold,
                "train_start": train_frame["Month"].min(),
                "train_end": train_frame["Month"].max(),
                "test_start": test_frame["Month"].min(),
                "test_end": test_frame["Month"].max(),
            }
            try:
                extra_kwargs = {}
                if model_name == "SARIMA" and sarima_order is not None:
                    extra_kwargs["cached_order"] = sarima_order
                prediction = predict(
                    model_name, train_frame, test_frame, extra_features, **extra_kwargs
                )
                predicted = np.clip(prediction.to_numpy(dtype=float), 0, None)
                metric_rows.append(
                    {
                        **base,
                        **regression_scores(actual, predicted),
                        "status": "ok",
                        "error": "",
                    }
                )
                for month, obs, pred in zip(test_frame["Month"], actual, predicted):
                    forecast_rows.append(
                        {
                            **base,
                            "Month": month,
                            "actual": float(obs),
                            "predicted": float(pred),
                            "status": "ok",
                            "error": "",
                        }
                    )
            except Exception as err:
                metric_rows.append(
                    {
                        **base,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "smape": np.nan,
                        "status": "failed",
                        "error": str(err),
                    }
                )

    return metric_rows, forecast_rows


def cross_validate_lsoa_xgb_global(
    lsoa_table, extra_features, train_months, test_months, step_months, xgb_params=None
):
    """
    One XGBoost model trained per fold on ALL crime types together.
    Folds are over unique months (shared timeline), not per-crime-type row counts.
    """
    lsoa_code = lsoa_table["LSOA code"].iloc[0]
    lsoa_name = lsoa_table["LSOA name"].iloc[0]
    uses_extra = bool(extra_features)

    months = sorted(lsoa_table["Month"].unique())
    folds = rolling_folds(len(months), train_months, test_months, step_months)
    metric_rows, forecast_rows = [], []

    for fold, train_idx, test_idx in folds:
        train_month_set = {months[i] for i in train_idx}
        test_month_set = {months[i] for i in test_idx}

        train_frame = lsoa_table[lsoa_table["Month"].isin(train_month_set)]
        test_frame = lsoa_table[lsoa_table["Month"].isin(test_month_set)]

        fold_meta = {
            "LSOA code": lsoa_code,
            "LSOA name": lsoa_name,
            "model": "XGBoost",
            "uses_extra_features": uses_extra,
            "fold": fold,
            "train_start": train_frame["Month"].min(),
            "train_end": train_frame["Month"].max(),
            "test_start": test_frame["Month"].min(),
            "test_end": test_frame["Month"].max(),
        }

        try:
            preds_by_crime = xgboost_global_predict(
                train_frame, test_frame, extra_features, xgb_params=xgb_params
            )
        except Exception as err:
            for ct in lsoa_table["Crime type"].unique():
                metric_rows.append(
                    {
                        **fold_meta,
                        "Crime type": ct,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "smape": np.nan,
                        "status": "failed",
                        "error": str(err),
                    }
                )
            continue

        for ct, pred_series in preds_by_crime.items():
            ct_test = test_frame[test_frame["Crime type"] == ct].sort_values("Month")
            actual = ct_test["crime_count"].to_numpy(dtype=float)
            predicted = np.clip(pred_series.to_numpy(dtype=float), 0, None)
            base = {**fold_meta, "Crime type": ct}
            metric_rows.append(
                {**base, **regression_scores(actual, predicted), "status": "ok", "error": ""}
            )
            for month, obs, pred_val in zip(ct_test["Month"], actual, predicted):
                forecast_rows.append(
                    {
                        **base,
                        "Month": month,
                        "actual": float(obs),
                        "predicted": float(pred_val),
                        "status": "ok",
                        "error": "",
                    }
                )

    return metric_rows, forecast_rows


def cross_validate_lsoa(
    lsoa_table, models, extra_features, train_months, test_months, step_months,
    xgb_params=None,
):
    lsoa_code = lsoa_table["LSOA code"].iloc[0]
    lsoa_name = lsoa_table["LSOA name"].iloc[0]
    metric_rows, forecast_rows = [], []

    # Global XGBoost: one model per fold trained on all crime types
    if "XGBoost" in models:
        m, f = cross_validate_lsoa_xgb_global(
            lsoa_table, extra_features, train_months, test_months, step_months,
            xgb_params=xgb_params,
        )
        metric_rows.extend(m)
        forecast_rows.extend(f)

    # Per-crime-type models (SARIMA, Prophet, …)
    other_models = [m for m in models if m != "XGBoost"]
    if other_models:
        # Detect SARIMA order on the TRAINING portion of the first fold only,
        # so no future test data influences the order selection.
        sarima_order = None
        if "SARIMA" in other_models:
            months = sorted(lsoa_table["Month"].unique())
            first_folds = rolling_folds(len(months), train_months, test_months, step_months)
            if first_folds:
                train_month_set = {months[i] for i in first_folds[0][1]}
                train_slice = lsoa_table[lsoa_table["Month"].isin(train_month_set)]
                aggregate = train_slice.groupby("Month")["crime_count"].sum().sort_index()
            else:
                aggregate = lsoa_table.groupby("Month")["crime_count"].sum().sort_index()
            sarima_order = detect_sarima_order(aggregate)

        for crime_type, crime_frame in lsoa_table.groupby("Crime type", sort=True):
            m, f = cross_validate_crime_type(
                crime_frame,
                crime_type,
                lsoa_code,
                lsoa_name,
                other_models,
                extra_features,
                train_months,
                test_months,
                step_months,
                sarima_order=sarima_order,
            )
            metric_rows.extend(m)
            forecast_rows.extend(f)

    return (
        pd.DataFrame(metric_rows).reindex(columns=METRIC_COLUMNS),
        pd.DataFrame(forecast_rows).reindex(columns=CV_FORECAST_COLUMNS),
    )


def summarize_cv_results(metrics, sort_by="mae"):
    ok = metrics[metrics["status"].eq("ok")]
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        ok.groupby(["LSOA code", "Crime type", "model"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            smape=("smape", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values(["LSOA code", "Crime type", sort_by])
    )
    best = summary.drop_duplicates(["LSOA code", "Crime type"], keep="first")
    return summary, best
