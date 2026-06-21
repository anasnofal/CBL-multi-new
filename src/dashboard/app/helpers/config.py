from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List


class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str
    DATA_DIR: str
    LSOA_DATA_DIR: str
    GEOJSON_PATH: str
    HISTORICAL_ANALYSIS_PATH: str
    CRIME_COSTS_DIR: str

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)  # make it cache only one in order to make a singleton class
def get_settings():
    return Settings()
