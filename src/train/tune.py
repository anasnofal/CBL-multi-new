"""
XGBoost hyperparameter search.

Samples a dedicated pool of LSOAs spread across the crime-volume distribution,
runs a random search using ALL CV folds (not just one) on each, and returns
the best params together with the set of LSOAs that were used — so the caller
can exclude them from the main training loop and avoid reporting biased metrics.
"""

import itertools
import random

import numpy as np

from .config import RANDOM_STATE
from .data import build_lsoa_frame
from .evaluate import rolling_folds
from .models.xgboost import _XGB_DEFAULTS, xgboost_global_predict

# Search space: max_depth starts at 3 because one-hot crime-type encoding
# needs enough depth to split on individual type columns.
_SEARCH_SPACE = {
    "n_estimators":     [200, 300, 500],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.02, 0.05, 0.1],
    "subsample":        [0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9],
}


def _all_candidates():
    keys = list(_SEARCH_SPACE.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*_SEARCH_SPACE.values())]


def _eval_candidate(params, lsoa_subset, extra_features,
                    train_months, test_months, step_months):
    """
    Evaluate one param set across ALL CV folds of one LSOA.
    Returns mean MAE over all folds × all crime types, or inf on failure.
    """
    lsoa_code = lsoa_subset["LSOA code"].iloc[0]
    try:
        lsoa_table = build_lsoa_frame(lsoa_subset, lsoa_code)
    except Exception:
        return float("inf")

    months = sorted(lsoa_table["Month"].unique())
    folds = rolling_folds(len(months), train_months, test_months, step_months)
    if not folds:
        return float("inf")

    all_maes = []
    for _, train_idx, test_idx in folds:
        train_month_set = {months[i] for i in train_idx}
        test_month_set  = {months[i] for i in test_idx}
        train_frame = lsoa_table[lsoa_table["Month"].isin(train_month_set)]
        test_frame  = lsoa_table[lsoa_table["Month"].isin(test_month_set)]

        try:
            preds_by_crime = xgboost_global_predict(
                train_frame, test_frame, extra_features, xgb_params=params
            )
        except Exception:
            return float("inf")

        for ct, pred_series in preds_by_crime.items():
            ct_test  = test_frame[test_frame["Crime type"] == ct].sort_values("Month")
            actual   = ct_test["crime_count"].to_numpy(dtype=float)
            predicted = np.clip(pred_series.to_numpy(dtype=float), 0, None)
            if len(actual) > 0:
                all_maes.append(float(np.mean(np.abs(actual - predicted))))

    return float(np.mean(all_maes)) if all_maes else float("inf")


def tune_xgb_params(
    lsoa_datasets: dict,
    extra_features: list,
    n_lsoas: int = 5,
    n_trials: int = 50,
    train_months: int = 24,
    test_months: int = 3,
    step_months: int = 12,
) -> tuple[dict, frozenset]:
    """
    XGBoost hyperparameter search.

    Samples `n_lsoas` LSOAs evenly spread across the crime-volume distribution,
    evaluates `n_trials` random candidates across ALL CV folds on those LSOAs,
    and returns (best_params, tuning_lsoa_codes).

    The tuning LSOAs must be excluded from the main training loop by the caller —
    their CV metrics would be biased because their hyperparameters were selected
    using their own data.

    Parameters
    ----------
    lsoa_datasets : dict  {lsoa_code: DataFrame}
    extra_features : list  exogenous column names
    n_lsoas : int  number of tuning LSOAs (default 5)
    n_trials : int  random candidates to try (default 50)
    train_months, test_months, step_months : int  CV window config

    Returns
    -------
    (best_params dict, frozenset of tuning LSOA codes)
    """
    if not lsoa_datasets:
        return dict(_XGB_DEFAULTS), frozenset()

    # Evenly spread across the volume distribution for representativeness
    totals      = {code: df["crime_count"].sum() for code, df in lsoa_datasets.items()}
    sorted_codes = sorted(totals, key=totals.get)
    n = len(sorted_codes)

    if n <= n_lsoas:
        sample_codes = sorted_codes
    else:
        step = n / n_lsoas
        sample_codes = [sorted_codes[int(i * step)] for i in range(n_lsoas)]

    rng        = random.Random(RANDOM_STATE)
    candidates = rng.sample(_all_candidates(), min(n_trials, len(_all_candidates())))

    best_params = dict(_XGB_DEFAULTS)
    best_mae    = float("inf")

    for params in candidates:
        maes = []
        for code in sample_codes:
            mae = _eval_candidate(
                params, lsoa_datasets[code], extra_features,
                train_months, test_months, step_months,
            )
            if mae < float("inf"):
                maes.append(mae)

        if not maes:
            continue

        avg_mae = float(np.mean(maes))
        if avg_mae < best_mae:
            best_mae    = avg_mae
            best_params = dict(params)

    return best_params, frozenset(sample_codes)
