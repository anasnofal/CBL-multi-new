"""
Model registry. To add a new model note this just a pythonic way of implementing the factory design pattern, in the future if it becomes more complex this files showed moved to factory.py file so:
  1. Create src/train/models/<name>.py with a class inheriting BaseForecaster.
  2. Add it to MODEL_REGISTRY below. Nothing else needs to change.
"""

from .base import BaseForecaster
from .prophet import ProphetForecaster, prophet_predict
from .sarima import SARIMAForecaster, detect_sarima_order, sarima_predict
from .xgboost import (
    XGBoostForecaster,
    XGBoostGlobalForecaster,
    get_xgb_features,
    xgboost_global_predict,
    xgboost_predict,
)

MODEL_REGISTRY: dict[str, type[BaseForecaster]] = {
    "SARIMA": SARIMAForecaster,
    "Prophet": ProphetForecaster,
    "XGBoost": XGBoostForecaster,
}

_CV_PREDICT = {
    "SARIMA": sarima_predict,
    "Prophet": prophet_predict,
    "XGBoost": xgboost_predict,
}


def predict(model_name, train_frame, future_frame, extra_features=None, **kwargs):
    """
    Dispatch to the correct CV predict function for `model_name`.
    Extra kwargs (e.g. cached_order for SARIMA) are forwarded to the model function.
    """
    if model_name not in _CV_PREDICT:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(_CV_PREDICT)}"
        )
    return _CV_PREDICT[model_name](train_frame, future_frame, extra_features, **kwargs)


def fit_forecaster(model_name, train_frame, extra_features=None):
    """Fit the named model on the full training frame and return a picklable Forecaster."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_name].fit(train_frame, extra_features)
