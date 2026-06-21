import pandas as pd
from .BaseController import BaseController


class ExplainabilityController(BaseController):

    def get_explain(
        self,
        df: pd.DataFrame,
        lsoa_code: str,
        crime_type: str,
        month: str,
    ) -> dict | None:
        mask = (
            (df["LSOA code"] == lsoa_code) &
            (df["Crime type"] == crime_type) &
            (df["Month"] == pd.to_datetime(month))
        )
        rows = df.loc[mask]
        if rows.empty:
            return None

        row = rows.iloc[0]
        model = row["model"]

        result = {
            "model": model,
            "month": str(row["Month"])[:7],
            "baseline": round(float(row["baseline"]), 3) if pd.notna(row.get("baseline")) else None,
            "summary": str(row["summary"]) if pd.notna(row.get("summary")) else None,
        }

        if model == "XGBoost":
            drivers = []
            for i in (1, 2, 3):
                name = row.get(f"driver_{i}")
                contrib = row.get(f"contrib_{i}")
                value = row.get(f"value_{i}")
                if pd.notna(name) and pd.notna(contrib):
                    drivers.append({
                        "name": str(name),
                        "contribution": round(float(contrib), 3),
                        "feature_value": round(float(value), 2) if pd.notna(value) else None,
                    })
            result["drivers"] = drivers
        else:
            components = []
            for label, col in [
                ("Seasonal", "seasonal"),
                ("Trend", "trend"),
                ("ARIMA correction", "arima_correction"),
            ]:
                v = row.get(col)
                if pd.notna(v):
                    components.append({"name": label, "contribution": round(float(v), 3)})
            result["components"] = components

        return result
