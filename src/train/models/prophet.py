import numpy as np
import pandas as pd

from .base import BaseForecaster, build_climatology

_Prophet = None


def _load_prophet():  # Lazy import to avoid hard dependency for users who don't need it. looks complex but is actually just a module-level singleton pattern.
    global _Prophet
    if _Prophet is None:
        from prophet import Prophet

        _Prophet = Prophet
    return _Prophet


class ProphetForecaster(BaseForecaster):

    def __init__(self, model, extra_features, last_month, climatology):
        self._model = model
        self.extra_features = list(extra_features)
        self.last_month = pd.Timestamp(last_month)
        self._climatology = climatology

    @classmethod
    def fit(cls, train_frame, extra_features=None):
        Prophet = _load_prophet()
        extra_features = list(extra_features or [])
        train = train_frame.sort_values("Month")
        last_month = train["Month"].iloc[-1]
        climatology = build_climatology(train, extra_features)

        prophet_train = train[["Month", "crime_count"] + extra_features].rename(
            columns={"Month": "ds", "crime_count": "y"}
        )
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            n_changepoints=5,
        )
        for col in extra_features:
            model.add_regressor(col)
        model.fit(prophet_train)
        return cls(model, extra_features, last_month, climatology)

    def predict(self, steps, future_exog_frame=None):
        months = pd.date_range(
            self.last_month + pd.offsets.MonthBegin(1), periods=steps, freq="MS"
        )
        future_df = pd.DataFrame({"ds": months})
        for col in self.extra_features:
            if future_exog_frame is not None and col in future_exog_frame.columns:
                future_df[col] = future_exog_frame[col].values
            else:
                future_df[col] = [
                    float(self._climatology.get(col, {}).get(m.month, 0.0))
                    for m in months
                ]
        fc = self._model.predict(future_df)
        return np.clip(fc["yhat"].to_numpy(dtype=float), 0, None)


# ── low-level CV function ─────────────────────────────────────────────────────


def prophet_predict(train_frame, future_frame, extra_features=None):
    """Used by the rolling CV loop — receives actual test-window exog values."""
    Prophet = _load_prophet()
    extra_features = list(extra_features or [])
    train = train_frame.sort_values("Month").copy()
    future = future_frame.sort_values("Month").copy()

    prophet_train = train[["Month", "crime_count"] + extra_features].rename(
        columns={"Month": "ds", "crime_count": "y"}
    )
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    for col in extra_features:
        model.add_regressor(col)
    model.fit(prophet_train)

    prophet_future = future[["Month"] + extra_features].rename(columns={"Month": "ds"})
    forecast = model.predict(prophet_future)
    return pd.Series(
        forecast["yhat"].to_numpy(), index=pd.DatetimeIndex(future["Month"])
    )
