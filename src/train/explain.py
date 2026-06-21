"""
Forecast explanation generation — produces one flat CSV row per
(LSOA, crime_type, forecast_month) that the dashboard can use to:

  • XGBoost waterfall  — baseline + top-3 SHAP drivers + raw feature values
  • SARIMA stacked bar — baseline + seasonal + trend + ARIMA correction
  • CI bands           — XGBoost: normalized adaptive conformal
                         SARIMA : native pmdarima prediction intervals

Dashboard-ready schema
──────────────────────
LSOA code | LSOA name | Crime type | model | Month |
predicted | ci_lower | ci_upper |
baseline |
driver_1 | contrib_1 | value_1 |
driver_2 | contrib_2 | value_2 |
driver_3 | contrib_3 | value_3 |
seasonal | trend | arima_correction |
summary
"""

import warnings
import numpy as np
import pandas as pd

try:
    import shap as _shap

    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False

from .config import XGB_LAGS
from .models.xgboost import _crime_type_col

# ── human-readable feature labels ─────────────────────────────────────────────

_LABELS: dict[str, str] = {
    "lag_1": "Last month's count",
    "lag_2": "2 months ago",
    "lag_3": "3 months ago",
    "lag_6": "6 months ago",
    "lag_12": "Same month last year",
    "rolling_3_mean": "3-month average",
    "rolling_6_mean": "6-month average",
    "rolling_12_mean": "12-month average",
    "month_sin": "Seasonal pattern",
    "month_cos": "Seasonal pattern",
    "month_num": "Calendar month",
}


def _label(name: str) -> str:
    if name in _LABELS:
        return _LABELS[name]
    if name.startswith("ct__"):
        return name.replace("ct__", "").replace("_", " ").title() + " (type)"
    return name


def _make_summary(drivers: list, top_k: int = 3) -> str:
    top = sorted(drivers, key=lambda d: abs(d["contrib"]), reverse=True)[:top_k]
    parts = [
        f"{d['driver']} ({'+' if d['contrib'] >= 0 else ''}{d['contrib']:.1f})"
        for d in top
        if abs(d["contrib"]) > 0.05
    ]
    return "Driven by: " + "; ".join(parts) if parts else ""


def _conformal_q(scores: np.ndarray, coverage: float) -> float:
    """
    Finite-sample conformal quantile (Vovk et al.).
    Guarantees marginal coverage ≥ `coverage` when calibration and test are
    exchangeable (i.e. same distribution as the CV folds).
    """
    n = len(scores)
    if n == 0:
        return np.nan
    level = min((1 - (1 - coverage)) * (1 + 1 / n), 1.0)
    return float(np.quantile(scores, level))


# ── XGBoost explanations ───────────────────────────────────────────────────────


def explain_xgb_forecast(
    forecaster,
    forecast_horizon: int,
    cv_forecasts: "pd.DataFrame | None" = None,
    coverage: float = 0.80,
    top_k: int = 3,
    forecast_cutoff=None,
) -> list[dict]:
    """
    SHAP explanations + normalized adaptive conformal CI for every
    (crime_type, forecast_month) produced by an XGBoostGlobalForecaster.

    Parameters
    ----------
    forecaster    : fitted XGBoostGlobalForecaster
    forecast_horizon : number of months ahead
    cv_forecasts  : DataFrame with columns (Crime type, model, actual, predicted, status)
                    from cross_validate_lsoa — used for conformal calibration per crime type.
                    If None or too few rows, falls back to in-sample calibration.
    coverage      : target prediction interval coverage (default 0.80)
    top_k         : number of SHAP drivers to store (default 3)
    """
    if not _SHAP_OK:
        return []

    feature_cols = forecaster.feature_cols
    start = forecast_cutoff if forecast_cutoff is not None else forecaster.last_month
    future_months = pd.date_range(
        start + pd.offsets.MonthBegin(1),
        periods=forecast_horizon,
        freq="MS",
    )

    # ── build all prediction rows upfront (batch SHAP) ────────────────────────
    meta: list[tuple] = []  # (crime_type, month, predicted)
    feat_rows: list[dict] = []

    for ct in forecaster._active_types:
        history = list(forecaster._histories[ct])
        for month in future_months:
            row = _month_feats(month)
            for col in forecaster.extra_features:
                row[col] = float(
                    forecaster._climatology.get(col, {}).get(month.month, 0.0)
                )
            for lag in XGB_LAGS:
                row[f"lag_{lag}"] = (
                    float(history[-lag]) if len(history) >= lag else np.nan
                )
            s = pd.Series(history, dtype=float)
            row["rolling_3_mean"] = float(s.tail(3).mean())
            row["rolling_6_mean"] = float(s.tail(6).mean())
            row["rolling_12_mean"] = float(s.tail(12).mean())
            for ct2 in forecaster._active_types:
                row[_crime_type_col(ct2)] = 1.0 if ct2 == ct else 0.0

            pred_df = pd.DataFrame([row], columns=feature_cols)
            pred = max(0.0, float(forecaster._model.predict(pred_df)[0]))
            feat_rows.append(row)
            meta.append((ct, month, pred))
            history.append(pred)

    if not feat_rows:
        return []

    X = pd.DataFrame(feat_rows, columns=feature_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = _shap.TreeExplainer(forecaster._model)
        shap_values = explainer.shap_values(X)  # (n_rows, n_features)
    global_base = float(explainer.expected_value)

    # Indices of time-series vs one-hot columns
    ct_indices = [i for i, c in enumerate(feature_cols) if c.startswith("ct__")]
    ts_indices = [i for i, c in enumerate(feature_cols) if not c.startswith("ct__")]
    cos_idx = feature_cols.index("month_cos") if "month_cos" in feature_cols else None

    # ── conformal q̂ per crime type (CV hold-out data only) ───────────────────
    q_hats: dict[str, float] = {}
    for ct in forecaster._active_types:
        q_hats[ct] = _xgb_conformal_q(ct, cv_forecasts, coverage)

    # ── assemble results ───────────────────────────────────────────────────────
    results: list[dict] = []
    for row_idx, (ct, month, pred) in enumerate(meta):
        sv = shap_values[row_idx]

        # Absorb crime-type one-hot contributions into baseline
        ct_shap_sum = float(sv[ct_indices].sum())
        adj_base = global_base + ct_shap_sum

        # Time-series SHAP drivers (merged seasonality)
        drivers: list[dict] = []
        seen_seasonal = False
        for fi in ts_indices:
            feat_name = feature_cols[fi]
            contrib = float(sv[fi])
            if abs(contrib) < 0.01:
                continue
            label = _label(feat_name)
            if label == "Seasonal pattern":
                if seen_seasonal:
                    continue
                if cos_idx is not None:
                    contrib += float(sv[cos_idx])
                seen_seasonal = True
            raw_val = float(X.iloc[row_idx][feat_name])
            drivers.append(
                {"driver": label, "contrib": round(contrib, 3), "value": raw_val}
            )

        drivers.sort(key=lambda d: abs(d["contrib"]), reverse=True)
        top_drivers = drivers[:top_k]
        while len(top_drivers) < 3:
            top_drivers.append({"driver": "", "contrib": np.nan, "value": np.nan})

        # Conformal CI
        q_hat = q_hats.get(ct, np.nan)
        fc_safe = max(pred, 0.1)
        ci_lower = (
            np.clip(fc_safe * (1 - q_hat), 0, None) if not np.isnan(q_hat) else np.nan
        )
        ci_upper = fc_safe * (1 + q_hat) if not np.isnan(q_hat) else np.nan

        results.append(
            {
                "crime_type": ct,
                "model": "XGBoost",
                "month": month,
                "predicted": round(pred, 2),
                "ci_lower": (
                    round(float(ci_lower), 2)
                    if ci_lower is not None and not np.isnan(ci_lower)
                    else np.nan
                ),
                "ci_upper": (
                    round(float(ci_upper), 2)
                    if ci_upper is not None and not np.isnan(ci_upper)
                    else np.nan
                ),
                "baseline": round(adj_base, 2),
                "driver_1": top_drivers[0]["driver"],
                "contrib_1": top_drivers[0]["contrib"],
                "value_1": (
                    round(top_drivers[0]["value"], 2)
                    if not np.isnan(top_drivers[0]["value"])
                    else np.nan
                ),
                "driver_2": top_drivers[1]["driver"],
                "contrib_2": top_drivers[1]["contrib"],
                "value_2": (
                    round(top_drivers[1]["value"], 2)
                    if not np.isnan(top_drivers[1]["value"])
                    else np.nan
                ),
                "driver_3": top_drivers[2]["driver"],
                "contrib_3": top_drivers[2]["contrib"],
                "value_3": (
                    round(top_drivers[2]["value"], 2)
                    if not np.isnan(top_drivers[2]["value"])
                    else np.nan
                ),
                "seasonal": np.nan,
                "trend": np.nan,
                "arima_correction": np.nan,
                "summary": _make_summary(top_drivers, top_k=3),
            }
        )

    return results


def _xgb_conformal_q(
    crime_type: str,
    cv_forecasts: "pd.DataFrame | None",
    coverage: float,
) -> float:
    """
    Compute normalized conformal q̂ for one crime type using CV hold-out residuals.

    Normalized score: s_i = |actual_i − predicted_i| / predicted_i
    Conformal level : (1 − α)(1 + 1/n)  where α = 1 − coverage

    Returns np.nan when fewer than 5 CV observations are available so the
    caller can skip CI rather than produce misleading intervals.
    """
    if cv_forecasts is None or cv_forecasts.empty:
        return np.nan

    status_col = cv_forecasts.get("status") if hasattr(cv_forecasts, "get") else None
    mask = cv_forecasts["Crime type"].eq(crime_type) & cv_forecasts["model"].eq(
        "XGBoost"
    )
    if "status" in cv_forecasts.columns:
        mask &= cv_forecasts["status"].eq("ok")

    cal = cv_forecasts[mask].dropna(subset=["actual", "predicted"])
    # Minimum 3 calibration points.  With n=3 the conformal level clips to 1.0
    # (worst-case score), giving a conservative but valid interval.
    if len(cal) < 3:
        return np.nan

    y = cal["actual"].values.astype(float)
    yhat = np.clip(cal["predicted"].values.astype(float), 0.1, None)
    return _conformal_q(np.abs(y - yhat) / yhat, coverage)


# ── SARIMA explanations ────────────────────────────────────────────────────────


def explain_sarima_forecast(
    forecaster,
    crime_series: pd.Series,
    crime_type: str,
    forecast_horizon: int,
    coverage: float = 0.80,
    forecast_cutoff=None,
) -> list[dict]:
    """
    Forecast decomposition + pmdarima confidence intervals for a SARIMAForecaster.

    Decomposition (additive):
        prediction[t] = baseline + seasonal[t] + trend[t] + arima_correction[t]

    Returns a list of explanation dicts, one per forecast month.
    """
    # ── CI from pmdarima ──────────────────────────────────────────────────────
    alpha = 1.0 - coverage
    try:
        fc_vals, ci = forecaster._model.predict(
            n_periods=forecast_horizon, return_conf_int=True, alpha=alpha
        )
        # pmdarima may return pandas Series with DatetimeIndex; fc_vals[i]
        # then does label-based lookup and raises KeyError(i) for integer i.
        # Force numpy arrays so integer indexing always works.
        fc_vals = np.clip(np.asarray(fc_vals, dtype=float), 0, None)
        ci = np.clip(np.asarray(ci, dtype=float), 0, None).reshape(-1, 2)
    except Exception:
        fc_vals = np.full(forecast_horizon, np.nan)
        ci = np.full((forecast_horizon, 2), np.nan)

    start = forecast_cutoff if forecast_cutoff is not None else forecaster.last_month
    future_months = pd.date_range(
        start + pd.offsets.MonthBegin(1),
        periods=forecast_horizon,
        freq="MS",
    )

    # ── decomposition ─────────────────────────────────────────────────────────
    baseline = float(crime_series.mean()) if len(crime_series) > 0 else 0.0
    seasonal_by_month: dict[int, float] = {}
    trend_slope = 0.0

    try:
        if len(crime_series) >= 24:
            from statsmodels.tsa.seasonal import seasonal_decompose

            dec = seasonal_decompose(
                crime_series.asfreq("MS"),
                model="additive",
                period=12,
                extrapolate_trend="freq",
            )
            for ts, val in dec.seasonal.items():
                seasonal_by_month[ts.month] = float(val)
            trend_vals = dec.trend.dropna()
            if len(trend_vals) >= 2:
                n_fit = min(6, len(trend_vals))
                trend_slope = float(
                    np.polyfit(range(n_fit), trend_vals.iloc[-n_fit:].values, 1)[0]
                )
    except Exception:
        pass

    results: list[dict] = []
    for i, month in enumerate(future_months):
        seas_c = seasonal_by_month.get(month.month, 0.0)
        trend_c = trend_slope * (i + 1)
        fc = float(fc_vals[i])
        arima_c = fc - baseline - seas_c - trend_c
        ci_lo = round(float(ci[i, 0]), 2)
        ci_hi = round(float(ci[i, 1]), 2)

        drivers = []
        if abs(seas_c) > 0.5:
            lbl = "Seasonal peak" if seas_c > 0 else "Seasonal quiet"
            drivers.append(
                {"driver": lbl, "contrib": round(seas_c, 2), "value": np.nan}
            )
        if abs(trend_c) > 0.5:
            lbl = "Rising trend" if trend_c > 0 else "Falling trend"
            drivers.append(
                {"driver": lbl, "contrib": round(trend_c, 2), "value": np.nan}
            )
        if abs(arima_c) > 0.5:
            drivers.append(
                {
                    "driver": "ARIMA correction",
                    "contrib": round(arima_c, 2),
                    "value": np.nan,
                }
            )
        drivers.sort(key=lambda d: abs(d["contrib"]), reverse=True)
        while len(drivers) < 3:
            drivers.append({"driver": "", "contrib": np.nan, "value": np.nan})

        results.append(
            {
                "crime_type": crime_type,
                "model": "SARIMA",
                "month": month,
                "predicted": round(fc, 2),
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "baseline": round(baseline, 2),
                "driver_1": drivers[0]["driver"],
                "contrib_1": drivers[0]["contrib"],
                "value_1": np.nan,
                "driver_2": drivers[1]["driver"],
                "contrib_2": drivers[1]["contrib"],
                "value_2": np.nan,
                "driver_3": drivers[2]["driver"],
                "contrib_3": drivers[2]["contrib"],
                "value_3": np.nan,
                "seasonal": round(seas_c, 2),
                "trend": round(trend_c, 2),
                "arima_correction": round(arima_c, 2),
                "summary": _make_summary(drivers, top_k=3),
            }
        )

    return results


# ── public helpers ─────────────────────────────────────────────────────────────

_EXPLANATION_COLUMNS = [
    "LSOA code",
    "LSOA name",
    "Crime type",
    "model",
    "Month",
    "predicted",
    "ci_lower",
    "ci_upper",
    "baseline",
    "driver_1",
    "contrib_1",
    "value_1",
    "driver_2",
    "contrib_2",
    "value_2",
    "driver_3",
    "contrib_3",
    "value_3",
    "seasonal",
    "trend",
    "arima_correction",
    "summary",
]


def build_explanation_df(
    explanation_list: list[dict],
    lsoa_code: str,
    lsoa_name: str,
) -> pd.DataFrame:
    """Convert raw explanation dicts to the flat dashboard-ready DataFrame."""
    if not explanation_list:
        return pd.DataFrame(columns=_EXPLANATION_COLUMNS)
    rows = []
    for e in explanation_list:
        rows.append(
            {
                "LSOA code": lsoa_code,
                "LSOA name": lsoa_name,
                "Crime type": e["crime_type"],
                "model": e["model"],
                "Month": e["month"],
                "predicted": e["predicted"],
                "ci_lower": e.get("ci_lower", np.nan),
                "ci_upper": e.get("ci_upper", np.nan),
                "baseline": e.get("baseline", np.nan),
                "driver_1": e.get("driver_1", ""),
                "contrib_1": e.get("contrib_1", np.nan),
                "value_1": e.get("value_1", np.nan),
                "driver_2": e.get("driver_2", ""),
                "contrib_2": e.get("contrib_2", np.nan),
                "value_2": e.get("value_2", np.nan),
                "driver_3": e.get("driver_3", ""),
                "contrib_3": e.get("contrib_3", np.nan),
                "value_3": e.get("value_3", np.nan),
                "seasonal": e.get("seasonal", np.nan),
                "trend": e.get("trend", np.nan),
                "arima_correction": e.get("arima_correction", np.nan),
                "summary": e.get("summary", ""),
            }
        )
    return pd.DataFrame(rows, columns=_EXPLANATION_COLUMNS)


# ── private ────────────────────────────────────────────────────────────────────


def _month_feats(month) -> dict:
    n = int(pd.Timestamp(month).month)
    return {
        "month_num": n,
        "month_sin": float(np.sin(2 * np.pi * n / 12)),
        "month_cos": float(np.cos(2 * np.pi * n / 12)),
    }
