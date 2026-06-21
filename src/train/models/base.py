import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def build_climatology(frame, extra_features):
    """Per-calendar-month averages for each extra feature. Used as fallback when future values are unknown."""
    return {
        col: dict(frame.groupby(frame["Month"].dt.month)[col].mean())
        for col in extra_features
    }


class BaseForecaster(ABC):
    """
    Interface every forecasting model must satisfy.

    To add a new model
    ------------------
    1. Create  src/train/models/<name>.py  with a class that subclasses this.
    2. Implement  fit()  and  predict().
    3. Add one entry to MODEL_REGISTRY in  src/train/models/__init__.py.
    That is all — no other file needs to change.
    """

    @classmethod
    @abstractmethod
    def fit(cls, train_frame: pd.DataFrame, extra_features=None) -> "BaseForecaster":
        """Fit on the full training frame and return a ready-to-predict instance."""

    @abstractmethod
    def predict(self, steps: int, future_exog_frame=None) -> np.ndarray:
        """Return a non-negative float array of length `steps`."""

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "BaseForecaster":
        with open(path, "rb") as f:
            return pickle.load(f)
