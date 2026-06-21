import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def month_start(values):
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def smape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    bottom = np.abs(actual) + np.abs(predicted)
    score = np.divide(
        2 * np.abs(predicted - actual),
        bottom,
        out=np.zeros_like(actual, dtype=float),
        where=bottom != 0,
    )
    return float(score.mean() * 100)


def regression_scores(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "smape": smape(actual, predicted),
    }


def month_features(month):
    n = pd.Timestamp(month).month
    return {
        "month_num": n,
        "month_sin": float(np.sin(2 * np.pi * n / 12)),
        "month_cos": float(np.cos(2 * np.pi * n / 12)),
    }


def safe_file_name(text):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    return re.sub(r"_+", "_", text).strip("_")


def read_lsoa_file(path):
    """
    Read LSOA codes from a file.  Supported formats:

    JSON  (.json)
        ["E01000001", "E01000002"]          — plain list
        {"lsoas": ["E01000001", ...]}        — object with "lsoas" key

    Plain text  (any other extension)
        One code per line, or comma / whitespace separated.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(c).strip() for c in data if str(c).strip()]
        if isinstance(data, dict) and "lsoas" in data:
            return [str(c).strip() for c in data["lsoas"] if str(c).strip()]
        raise ValueError(
            f"{path}: JSON must be a list or an object with a 'lsoas' key, got {type(data).__name__}"
        )
    codes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        codes.extend(part.strip() for part in re.split(r"[,\s]+", line) if part.strip())
    return codes


def append_csv(frame, path, columns=None):
    if frame.empty:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns:
        frame = frame.reindex(columns=columns)
    # Check header need after resolving the path so we never write a header-only file
    # when frame turns empty after reindex, and the exists() check is stable
    # (sequential writers only — parallel writes use _append_csv_locked instead).
    if frame.empty:
        return
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def accumulate_cv_metrics(totals, metrics):
    ok = metrics[metrics["status"].eq("ok")]
    for key, frame in ok.groupby(["model", "uses_extra_features"]):
        item = totals.setdefault(
            key,
            {
                "mae": 0.0,
                "rmse": 0.0,
                "smape": 0.0,
                "folds": 0,
                "lsoas": set(),
                "crime_types": set(),
            },
        )
        item["mae"] += frame["mae"].sum()
        item["rmse"] += frame["rmse"].sum()
        item["smape"] += frame["smape"].sum()
        item["folds"] += len(frame)
        item["lsoas"].update(frame["LSOA code"].astype(str).unique())
        item["crime_types"].update(frame["Crime type"].astype(str).unique())


def cv_metrics_to_frame(totals):
    rows = [
        {
            "model": model,
            "uses_extra_features": uses_extra,
            "mae": item["mae"] / item["folds"],
            "rmse": item["rmse"] / item["folds"],
            "smape": item["smape"] / item["folds"],
            "evaluated_folds": item["folds"],
            "lsoa_count": len(item["lsoas"]),
            "crime_type_count": len(item["crime_types"]),
        }
        for (model, uses_extra), item in totals.items()
    ]
    return pd.DataFrame(rows).sort_values("mae") if rows else pd.DataFrame()


def accumulate_cv_forecasts(totals, forecasts):
    ok = forecasts[forecasts["status"].eq("ok")].copy()
    if ok.empty:
        return
    ok["absolute_error"] = (ok["predicted"] - ok["actual"]).abs()
    ok["squared_error"] = (ok["predicted"] - ok["actual"]) ** 2
    bottom = ok["predicted"].abs() + ok["actual"].abs()
    ok["smape_contrib"] = np.where(
        bottom != 0, 2 * ok["absolute_error"] / bottom * 100, 0
    )

    for key, frame in ok.groupby(["model", "uses_extra_features"]):
        item = totals.setdefault(
            key,
            {
                "absolute_error": 0.0,
                "squared_error": 0.0,
                "smape_contrib": 0.0,
                "observations": 0,
                "lsoas": set(),
                "crime_types": set(),
            },
        )
        item["absolute_error"] += frame["absolute_error"].sum()
        item["squared_error"] += frame["squared_error"].sum()
        item["smape_contrib"] += frame["smape_contrib"].sum()
        item["observations"] += len(frame)
        item["lsoas"].update(frame["LSOA code"].astype(str).unique())
        item["crime_types"].update(frame["Crime type"].astype(str).unique())


def pooled_forecasts_to_frame(totals):
    rows = [
        {
            "model": model,
            "uses_extra_features": uses_extra,
            "mae": item["absolute_error"] / item["observations"],
            "rmse": np.sqrt(item["squared_error"] / item["observations"]),
            "smape": item["smape_contrib"] / item["observations"],
            "observations": item["observations"],
            "lsoa_count": len(item["lsoas"]),
            "crime_type_count": len(item["crime_types"]),
        }
        for (model, uses_extra), item in totals.items()
    ]
    return pd.DataFrame(rows).sort_values("mae") if rows else pd.DataFrame()
