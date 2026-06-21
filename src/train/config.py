import random
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "notebooks" / "forecast_outputs"

RANDOM_STATE = 42


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
MODEL_NAMES = ["XGBoost", "SARIMA"]  # add Prophet via --models XGBoost SARIMA Prophet

# Columns that are always present in the input data — anything else is an extra feature.
BASE_COLUMNS = ["Month", "LSOA code", "LSOA name", "Crime type", "crime_count"]

ROLLING_TRAIN_MONTHS = 24
ROLLING_TEST_MONTHS = 3
ROLLING_STEP_MONTHS = 12
FORECAST_HORIZON = 12  # months ahead for the saved forecast output

SARIMA_ORDERS = [
    (0, 0, 1),
    (1, 0, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
]

SARIMA_SEASONAL_ORDERS = [
    (0, 0, 0, 12),
    (1, 0, 0, 12),
    (0, 1, 1, 12),
]

XGB_LAGS = [1, 2, 3, 6, 12]

# Base XGBoost features — always included regardless of extra features.
# Extra feature columns are appended at runtime based on what is in the data.
XGB_BASE_FEATURES = [
    "month_num",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_3_mean",
    "rolling_6_mean",
    "rolling_12_mean",
]

COUNT_COLUMNS = BASE_COLUMNS  # alias kept for data.py backward compat

METRIC_COLUMNS = [
    "LSOA code",
    "LSOA name",
    "Crime type",
    "model",
    "uses_extra_features",
    "fold",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "mae",
    "rmse",
    "smape",
    "status",
    "error",
]

CV_FORECAST_COLUMNS = [
    "LSOA code",
    "LSOA name",
    "Crime type",
    "model",
    "uses_extra_features",
    "fold",
    "Month",
    "actual",
    "predicted",
    "status",
    "error",
]

FUTURE_COLUMNS = [
    "LSOA code",
    "LSOA name",
    "Crime type",
    "model",
    "Month",
    "predicted",
]
