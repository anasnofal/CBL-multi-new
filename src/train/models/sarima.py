import warnings

import numpy as np
import pandas as pd
import pmdarima as pm

from .base import BaseForecaster, build_climatology

_SARIMA_FALLBACK_ORDER = (1, 1, 0)
_SARIMA_FALLBACK_SEASONAL = (1, 1, 0, 12)
_MIN_NONZERO = 6  # series with fewer non-zero months is skipped for SARIMA


def _check_series_quality(train_y):
    """Raise early when a series is too sparse for SARIMA to fit reliably."""
    nonzero = int((train_y.fillna(0) > 0).sum())
    if nonzero < _MIN_NONZERO:
        raise ValueError(
            f"Only {nonzero} non-zero months — too sparse for SARIMA (need {_MIN_NONZERO}). "
            "XGBoost will be used instead."
        )


def _to_numpy_exog(exog):
    """Ensure exog is a 2-D numpy float array (or None) for statsmodels/pmdarima."""
    if exog is None:
        return None
    arr = np.asarray(exog, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _fit_best_sarimax(train_y, train_exog):
    """Stepwise auto-ARIMA search — typically 4-8 fits instead of an 18-combo grid."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pm.auto_arima(
            train_y,
            X=train_exog,
            seasonal=True,
            m=12,
            stepwise=True,
            information_criterion="aic",
            start_p=0, start_q=0,
            max_p=1, max_q=1,
            start_P=0, start_Q=0,
            max_P=1, max_Q=1,
            max_D=1,
            suppress_warnings=True,
            error_action="ignore",
        )


class _StatsmodelsWrapper:
    """Thin adapter so statsmodels SARIMAX result has the same predict API as pmdarima."""

    def __init__(self, result):
        self._result = result

    def predict(self, n_periods, X=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._result.forecast(steps=n_periods, exog=_to_numpy_exog(X))
        return np.asarray(fc, dtype=float)


def _fit_sarimax_fixed(train_y, train_exog, order, seasonal_order):
    """Fit SARIMAX with known orders — skips auto_arima search entirely."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SARIMAX(
            train_y,
            exog=_to_numpy_exog(train_exog),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    return _StatsmodelsWrapper(result)


def detect_sarima_order(aggregate_series):
    """
    Run auto_arima once on the LSOA aggregate crime series to find the best
    (p,d,q)(P,D,Q,m).  Returns ((p,d,q), (P,D,Q,m)) or the fallback if it fails.
    Called once per LSOA — not per crime type or per fold.
    """
    agg = aggregate_series.astype(float)
    if not isinstance(agg.index, pd.DatetimeIndex):
        agg.index = pd.to_datetime(agg.index)
    try:
        agg = agg.asfreq("MS")
    except Exception:
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(
                agg,
                seasonal=True,
                m=12,
                stepwise=True,
                information_criterion="aic",
                start_p=0, start_q=0,
                max_p=1, max_q=1,
                start_P=0, start_Q=0,
                max_P=1, max_Q=1,
                max_D=1,
                suppress_warnings=True,
                error_action="ignore",
            )
        return model.order, model.seasonal_order
    except Exception:
        return _SARIMA_FALLBACK_ORDER, _SARIMA_FALLBACK_SEASONAL


class SARIMAForecaster(BaseForecaster):

    def __init__(self, model, extra_features, last_month, climatology):
        self._model = model
        self.extra_features = list(extra_features)
        self.last_month = pd.Timestamp(last_month)
        self._climatology = climatology

    @classmethod
    def fit(cls, train_frame, extra_features=None):
        extra_features = list(extra_features or [])
        train = train_frame.sort_values("Month")
        last_month = train["Month"].iloc[-1]
        train_y = train.set_index("Month")["crime_count"].astype(float).asfreq("MS")
        _check_series_quality(train_y)
        climatology = build_climatology(train, extra_features)
        train_exog = (
            _to_numpy_exog(
                train.set_index("Month")[extra_features].astype(float).asfreq("MS")
            )
            if extra_features
            else None
        )
        model = _fit_best_sarimax(train_y, train_exog)
        return cls(model, extra_features, last_month, climatology)

    def predict(self, steps, future_exog_frame=None):
        exog = self._build_exog(steps, future_exog_frame)
        fc = self._model.predict(n_periods=steps, X=exog)
        return np.clip(fc.astype(float), 0, None)

    def _build_exog(self, steps, future_exog_frame):
        if not self.extra_features:
            return None
        months = pd.date_range(
            self.last_month + pd.offsets.MonthBegin(1), periods=steps, freq="MS"
        )
        rows = []
        for i, month in enumerate(months):
            row = {}
            for col in self.extra_features:
                if future_exog_frame is not None and col in future_exog_frame.columns:
                    row[col] = float(future_exog_frame.iloc[i][col])
                else:
                    row[col] = float(
                        self._climatology.get(col, {}).get(month.month, 0.0)
                    )
            rows.append(row)
        return _to_numpy_exog(pd.DataFrame(rows, columns=self.extra_features))


# ── low-level CV function ─────────────────────────────────────────────────────


def sarima_predict(train_frame, future_frame, extra_features=None, cached_order=None):
    """
    Used by the rolling CV loop.
    If cached_order=(order, seasonal_order) is provided, skips auto_arima and
    uses _fit_sarimax_fixed — much faster when called many times per LSOA.
    """
    extra_features = list(extra_features or [])
    train = train_frame.sort_values("Month")
    future = future_frame.sort_values("Month")
    future_months = pd.DatetimeIndex(future["Month"])

    train_y = train.set_index("Month")["crime_count"].astype(float).asfreq("MS")
    _check_series_quality(train_y)
    train_exog = (
        _to_numpy_exog(
            train.set_index("Month")[extra_features].astype(float).asfreq("MS")
        )
        if extra_features
        else None
    )
    future_exog = (
        _to_numpy_exog(future.set_index("Month")[extra_features].astype(float))
        if extra_features
        else None
    )

    if cached_order is not None:
        order, seasonal_order = cached_order
        model = _fit_sarimax_fixed(train_y, train_exog, order, seasonal_order)
    else:
        model = _fit_best_sarimax(train_y, train_exog)

    preds = model.predict(n_periods=len(future_months), X=future_exog)
    return pd.Series(np.clip(preds, 0, None), index=future_months)
