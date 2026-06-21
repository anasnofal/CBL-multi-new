from .BaseController import BaseController
import pandas as pd


class HistoricalController(BaseController):

    def filter_historical_df(
        self,
        df: pd.DataFrame,
        lsoa_map: dict,
        police_station: str,
        lsoa_code: str = None,
        crime_type: str = None,
    ) -> pd.DataFrame:
        lsoas = self.get_lsoa_from_police_station(police_station, lsoa_map)
        mask = df["LSOA code"].isin(lsoas)

        if lsoa_code:
            mask &= df["LSOA code"] == lsoa_code

        if crime_type:
            mask &= df["Crime type"] == crime_type

        return df.loc[mask]
