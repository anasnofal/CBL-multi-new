import pandas as pd
from .BaseController import BaseController


class AlertController(BaseController):

    def __init__(self):
        super().__init__()

    def _empty_result(self):
        return {
            "total_alerts": 0,
            "alerts": [],
            "alert_levels": {},
        }

    def _compute_z_scores(
        self,
        lsoa_data: pd.DataFrame,
        window: int = 3
    ) -> pd.DataFrame:

        df = lsoa_data.sort_values("Month").copy()
        df["predicted"] = df["predicted"].round(0)
        shifted = df["predicted"].shift(1)

        df["rolling_mean"] = (shifted.rolling(window=window, min_periods=2).mean())
        df["rolling_std"] = (shifted.rolling(window=window, min_periods=2).std())
        df["z_score"] = ((df["predicted"] - df["rolling_mean"]) / df["rolling_std"].replace(0, float("nan")))

        return df

    def _compute_z_scores_for_lsoa(
        self,
        lsoa_code: str,
        lsoa_data: pd.DataFrame,
        window: int = 3,
        crime_type: str = None,
    ) -> pd.DataFrame:

        df = self._compute_z_scores(lsoa_data, window=window)
        df["LSOA code"] = lsoa_code
        if crime_type:
            df["Crime type"] = crime_type

        return df

    def _compute_grouped_z_scores(
        self,
        base: pd.DataFrame,
        group_columns: list[str],
        window: int = 3,
    ) -> pd.DataFrame:

        df = base.sort_values([*group_columns, "Month"]).copy()
        df["predicted"] = df["predicted"].round(0)
        grouped = df.groupby(group_columns, sort=False)["predicted"]
        shifted = grouped.shift(1)
        rolling = shifted.groupby([df[col] for col in group_columns], sort=False)

        df["rolling_mean"] = rolling.rolling(window=window, min_periods=2).mean().reset_index(level=group_columns, drop=True)
        df["rolling_std"] = rolling.rolling(window=window, min_periods=2).std().reset_index(level=group_columns, drop=True)
        df["z_score"] = ((df["predicted"] - df["rolling_mean"]) / df["rolling_std"].clip(lower = 0.1))

        return df

    def get_alerts(
        self,
        police_force: str,
        crime_type: str,
        time_start: str,
        time_end: str,
        df: pd.DataFrame,
        lsoa_map: dict,
        window: int = 3,
        z_threshold: float = 1.5,
    ):
        force_key = police_force.strip().lower().replace(" ", "-")
        hotspot_lsoas = set(lsoa_map.get(force_key, {}).get("hotspots", []))

        if not hotspot_lsoas:
            return self._empty_result()

        forecast_df = df.copy()

        start_date = pd.to_datetime(time_start)
        end_date = pd.to_datetime(time_end)
        include_all_crime_types = crime_type.strip().lower() == "all"

        base = forecast_df[
            (forecast_df["LSOA code"].isin(hotspot_lsoas)) &
            (forecast_df["Month"] <= end_date)
        ].copy()

        if not include_all_crime_types:
            base = base[base["Crime type"] == crime_type].copy()

        if base.empty:
            return self._empty_result()

        group_columns = ["LSOA code", "Crime type"] if include_all_crime_types else ["LSOA code"]
        scored = self._compute_grouped_z_scores(base, group_columns=group_columns, window=window)

        alerts_df = scored[
            (scored["Month"] >= start_date) &
            (scored["Month"] <= end_date) &
            (scored["z_score"] > z_threshold)
        ].copy()

        if alerts_df.empty:
            return self._empty_result()

        alerts_df["month"] = alerts_df["Month"].dt.strftime("%Y-%m")
        alerts_df["severity"] = "high"
        alerts_df = alerts_df.sort_values(["month", "z_score"], ascending=[True, False])

        alerts = [
            {
                "lsoa_code": row["LSOA code"],
                "crime_type": row["Crime type"],
                "month": row["month"],
                "predicted_count": round(row["predicted"], 2),
                "rolling_mean": round(row["rolling_mean"], 2),
                "z_score": round(row["z_score"], 2),
                "severity": row["severity"],
            }
            for _, row in alerts_df.iterrows()
        ]
        
        SEVERITY_LEVELS = {"high": 2, "medium": 1}

        alert_levels = {}
        for row in alerts:
            code = row["lsoa_code"]
            current_level = SEVERITY_LEVELS[row["severity"]]
            entry = alert_levels.setdefault(code, {"level": current_level, "dates": set(), "alerts": []})
            entry["level"] = max(entry["level"], current_level)
            entry["dates"].add(row["month"])
            entry["alerts"].append({
                "crime_type": row["crime_type"],
                "month": row["month"],
                "predicted_count": row["predicted_count"],
                "severity": row["severity"],
            })

        for entry in alert_levels.values():
            entry["dates"] = sorted(list(entry["dates"]))
            entry["alerts"] = sorted(entry["alerts"], key=lambda a: (a["month"], a["crime_type"]))

        return {
            "total_alerts": len(alerts),
            "alerts": alerts,
            "alert_levels": alert_levels,
        }