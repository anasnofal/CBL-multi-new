from .BaseController import BaseController
import pandas as pd


class ForecastController(BaseController):

    def filter_forecast_df(
        self,
        df: pd.DataFrame,
        lsoa_map: dict,
        police_station: str,
        LSOA: str = None,
        crime_type: str = None,
        time_start: str = None,
        time_end: str = None,
    ):
        lsoas = self.get_lsoa_from_police_station(police_station, lsoa_map)
        mask = df["LSOA code"].isin(lsoas)

        if LSOA:
            mask &= df["LSOA code"] == LSOA

        if crime_type:
            mask &= df["Crime type"] == crime_type

        if time_start:
            mask &= df["Month"] >= pd.to_datetime(time_start)

        if time_end:
            mask &= df["Month"] <= pd.to_datetime(time_end)

        return df.loc[mask]
