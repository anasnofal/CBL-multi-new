import re

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..config import RANDOM_STATE, XGB_BASE_FEATURES, XGB_LAGS
from ..helper import month_features
from .base import BaseForecaster, build_climatology

_XGB_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": 4,       # raised from 2 — one-hot crime type needs more depth
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
}


def _crime_type_col(ct: str) -> str:
    """Stable column name for a crime type one-hot feature."""
    return "ct__" + re.sub(r"[^a-z0-9]+", "_", ct.lower()).strip("_")


def get_xgb_features(extra_features=None):
    """Base time-series features + any active extra columns."""
    return XGB_BASE_FEATURES + list(extra_features or [])


def get_xgb_global_features(extra_features=None, crime_types=None):
    """Base features + exog + one binary column per crime type."""
    ct_cols = [_crime_type_col(ct) for ct in (crime_types or [])]
    return get_xgb_features(extra_features) + ct_cols


def build_xgb_training_frame(frame, extra_features=None):
    extra_features = list(extra_features or [])
    table = frame[["Month", "crime_count"] + extra_features].copy().sort_values("Month")
    for lag in XGB_LAGS:
        table[f"lag_{lag}"] = table["crime_count"].shift(lag)
    table["rolling_3_mean"] = (
        table["crime_count"].shift(1).rolling(3, min_periods=1).mean()
    )
    table["rolling_6_mean"] = (
        table["crime_count"].shift(1).rolling(6, min_periods=1).mean()
    )
    table["rolling_12_mean"] = (
        table["crime_count"].shift(1).rolling(12, min_periods=1).mean()
    )
    table = pd.concat(
        [table, table["Month"].apply(month_features).apply(pd.Series)], axis=1
    )
    return table.dropna(subset=get_xgb_features(extra_features) + ["crime_count"])


def build_xgb_global_frame(lsoa_frame, extra_features=None):
    """
    Build a single training frame covering all crime types.
    Each crime type gets a dedicated one-hot column (ct__<name>) instead of a
    single integer label, so the model can learn fully independent intercepts
    per crime type without imposing a false ordinal relationship.

    Crime types whose training frame is empty after lag/rolling dropna are
    excluded so the model is never asked to predict for a class it has not seen.

    Returns (frame, active_types) where active_types is the ordered list of
    crime types that appear in the frame.
    """
    extra_features = list(extra_features or [])
    crime_types = sorted(lsoa_frame["Crime type"].unique())

    # First pass: keep only types with enough data for lag features
    active_types = []
    type_frames = {}
    for ct in crime_types:
        sub = lsoa_frame[lsoa_frame["Crime type"] == ct]
        tbl = build_xgb_training_frame(sub, extra_features)
        if not tbl.empty:
            active_types.append(ct)
            type_frames[ct] = tbl.copy()

    if not active_types:
        return pd.DataFrame(), []

    # Second pass: add one-hot columns (one per active type) to each frame
    pieces = []
    for ct in active_types:
        tbl = type_frames[ct]
        for ct2 in active_types:
            tbl[_crime_type_col(ct2)] = 1 if ct2 == ct else 0
        pieces.append(tbl)

    return pd.concat(pieces, ignore_index=True), active_types


def _prediction_row(history, month, future_row, extra_features):
    row = month_features(month)
    for col in extra_features:
        row[col] = float(getattr(future_row, col))
    for lag in XGB_LAGS:
        row[f"lag_{lag}"] = float(history[-lag]) if len(history) >= lag else np.nan
    series = pd.Series(history, dtype=float)
    row["rolling_3_mean"] = float(series.tail(3).mean())
    row["rolling_6_mean"] = float(series.tail(6).mean())
    row["rolling_12_mean"] = float(series.tail(12).mean())
    return row


def _make_model(params=None):
    p = {**_XGB_DEFAULTS, **(params or {})}
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        colsample_bytree=p.get("colsample_bytree", _XGB_DEFAULTS["colsample_bytree"]),
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


class XGBoostForecaster(BaseForecaster):

    def __init__(
        self, model, feature_cols, history, extra_features, last_month, climatology
    ):
        self._model = model
        self.feature_cols = feature_cols
        self._history = list(history)
        self.extra_features = list(extra_features)
        self.last_month = pd.Timestamp(last_month)
        self._climatology = climatology

    @classmethod
    def fit(cls, train_frame, extra_features=None):
        extra_features = list(extra_features or [])
        train = train_frame.sort_values("Month")
        last_month = train["Month"].iloc[-1]
        climatology = build_climatology(train, extra_features)

        features = get_xgb_features(extra_features)
        table = build_xgb_training_frame(train, extra_features)
        model = _make_model()
        model.fit(table[features], table["crime_count"])
        history = train["crime_count"].astype(float).tolist()
        return cls(model, features, history, extra_features, last_month, climatology)

    def predict(self, steps, future_exog_frame=None):
        history = list(self._history)
        months = pd.date_range(
            self.last_month + pd.offsets.MonthBegin(1), periods=steps, freq="MS"
        )
        predictions = []
        for i, month in enumerate(months):
            row = month_features(month)
            for col in self.extra_features:
                if future_exog_frame is not None and col in future_exog_frame.columns:
                    row[col] = float(future_exog_frame.iloc[i][col])
                else:
                    row[col] = float(
                        self._climatology.get(col, {}).get(month.month, 0.0)
                    )
            for lag in XGB_LAGS:
                row[f"lag_{lag}"] = (
                    float(history[-lag]) if len(history) >= lag else np.nan
                )
            series = pd.Series(history, dtype=float)
            row["rolling_3_mean"] = float(series.tail(3).mean())
            row["rolling_6_mean"] = float(series.tail(6).mean())
            row["rolling_12_mean"] = float(series.tail(12).mean())
            pred = max(
                0.0,
                float(
                    self._model.predict(pd.DataFrame([row], columns=self.feature_cols))[0]
                ),
            )
            predictions.append(pred)
            history.append(pred)
        return np.array(predictions)


class XGBoostGlobalForecaster:
    """
    One XGBoost model trained on all crime types for a single LSOA.
    Crime type is one-hot encoded so each type gets its own independent
    intercept without imposing any ordinal relationship.
    """

    def __init__(self, model, feature_cols, histories, extra_features,
                 last_month, active_types, climatology):
        self._model = model
        self.feature_cols = feature_cols
        self._histories = histories       # dict: crime_type -> list[float]
        self.extra_features = list(extra_features)
        self.last_month = pd.Timestamp(last_month)
        self._active_types = list(active_types)
        self._climatology = climatology

    @classmethod
    def fit(cls, lsoa_frame, extra_features=None, xgb_params=None):
        extra_features = list(extra_features or [])
        last_month = lsoa_frame["Month"].max()
        climatology = build_climatology(lsoa_frame, extra_features)

        train_table, active_types = build_xgb_global_frame(lsoa_frame, extra_features)
        if train_table.empty:
            raise ValueError("No training data after feature construction.")

        features = get_xgb_global_features(extra_features, active_types)
        model = _make_model(xgb_params)
        model.fit(train_table[features], train_table["crime_count"])

        histories = {
            ct: lsoa_frame[lsoa_frame["Crime type"] == ct]
            .sort_values("Month")["crime_count"]
            .astype(float)
            .tolist()
            for ct in active_types
        }
        return cls(model, features, histories, extra_features, last_month, active_types, climatology)

    def predict_for_crime_type(self, crime_type, steps, future_exog_frame=None):
        """Iterative forecast for one crime type using the shared model."""
        if crime_type not in self._active_types:
            raise ValueError(f"Crime type '{crime_type}' not seen during fit.")

        history = list(self._histories[crime_type])
        months = pd.date_range(
            self.last_month + pd.offsets.MonthBegin(1), periods=steps, freq="MS"
        )
        predictions = []
        for i, month in enumerate(months):
            row = month_features(month)
            for col in self.extra_features:
                if future_exog_frame is not None and col in future_exog_frame.columns:
                    row[col] = float(future_exog_frame.iloc[i][col])
                else:
                    row[col] = float(self._climatology.get(col, {}).get(month.month, 0.0))
            for lag in XGB_LAGS:
                row[f"lag_{lag}"] = float(history[-lag]) if len(history) >= lag else np.nan
            series = pd.Series(history, dtype=float)
            row["rolling_3_mean"] = float(series.tail(3).mean())
            row["rolling_6_mean"] = float(series.tail(6).mean())
            row["rolling_12_mean"] = float(series.tail(12).mean())
            for ct in self._active_types:
                row[_crime_type_col(ct)] = 1.0 if ct == crime_type else 0.0
            pred = max(0.0, float(
                self._model.predict(pd.DataFrame([row], columns=self.feature_cols))[0]
            ))
            predictions.append(pred)
            history.append(pred)
        return np.array(predictions)


# ── low-level CV functions ────────────────────────────────────────────────────


def xgboost_predict(train_frame, future_frame, extra_features=None):
    """Per-crime-type CV predict — kept for backward compatibility."""
    extra_features = list(extra_features or [])
    features = get_xgb_features(extra_features)
    train_table = build_xgb_training_frame(train_frame, extra_features)

    model = _make_model()
    model.fit(train_table[features], train_table["crime_count"])

    history = train_frame.sort_values("Month")["crime_count"].astype(float).tolist()
    predictions = []
    for future_row in future_frame.sort_values("Month").itertuples(index=False):
        row = _prediction_row(history, future_row.Month, future_row, extra_features)
        pred = max(0.0, float(model.predict(pd.DataFrame([row], columns=features))[0]))
        predictions.append(pred)
        history.append(pred)
    return pd.Series(predictions, index=pd.DatetimeIndex(future_frame["Month"]))


def xgboost_global_predict(lsoa_frame, future_frame, extra_features=None, xgb_params=None):
    """
    Train one XGBoost on all crime types, predict per crime type.
    Returns dict {crime_type: pd.Series of predictions}.
    """
    extra_features = list(extra_features or [])
    train_table, active_types = build_xgb_global_frame(lsoa_frame, extra_features)

    if train_table.empty:
        return {}

    features = get_xgb_global_features(extra_features, active_types)
    model = _make_model(xgb_params)
    model.fit(train_table[features], train_table["crime_count"])

    results = {}
    for ct in active_types:
        ct_train = lsoa_frame[lsoa_frame["Crime type"] == ct].sort_values("Month")
        ct_future = future_frame[future_frame["Crime type"] == ct].sort_values("Month")
        if ct_future.empty:
            continue
        history = ct_train["crime_count"].astype(float).tolist()
        predictions = []
        for future_row in ct_future.itertuples(index=False):
            row = _prediction_row(history, future_row.Month, future_row, extra_features)
            for ct2 in active_types:
                row[_crime_type_col(ct2)] = 1.0 if ct2 == ct else 0.0
            pred = max(0.0, float(model.predict(pd.DataFrame([row], columns=features))[0]))
            predictions.append(pred)
            history.append(pred)
        results[ct] = pd.Series(predictions, index=pd.DatetimeIndex(ct_future["Month"]))

    return results
