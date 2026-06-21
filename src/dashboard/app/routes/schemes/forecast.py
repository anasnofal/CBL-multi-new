from pydantic import BaseModel
from typing import Optional


class ForecastRequest(BaseModel):
    police_station: str
    LSOA: Optional[str] = None
    crime_type: Optional[str] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
