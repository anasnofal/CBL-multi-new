from fastapi import APIRouter, Request
from ..Controllers.HistoricalController import HistoricalController

historical_router = APIRouter(
    prefix="/api/v1/historical",
    tags=["historical"],
)

_controller = HistoricalController()

_COLUMN_MAP = {
    "LSOA code": "lsoa_code",
    "LSOA name": "lsoa_name",
    "Crime type": "crime_type",
}


@historical_router.get("/{policeforce}/{lsoa_code}/{crime_type}")
async def get_lsoa_historical(
    policeforce: str,
    lsoa_code: str,
    crime_type: str,
    request: Request,
):
    """
    GET /api/v1/historical/{policeforce}/{lsoa_code}/{crime_type}
    Returns historical crime stats for the given LSOA and crime type.
    """
    filtered_df = _controller.filter_historical_df(
        df=request.app.state.HISTORICAL_DF,
        lsoa_map=request.app.state.LSOA_MAP,
        police_station=policeforce,
        lsoa_code=lsoa_code,
        crime_type=crime_type,
    )

    records = filtered_df.rename(columns=_COLUMN_MAP).to_dict(orient="records")

    return {
        "police_force": policeforce,
        "lsoa_code": lsoa_code,
        "crime_type": crime_type,
        "count": len(records),
        "data": records,
    }


@historical_router.get("/{policeforce}/{lsoa_code}")
async def get_lsoa_historical_all(
    policeforce: str,
    lsoa_code: str,
    request: Request,
):
    """
    GET /api/v1/historical/{policeforce}/{lsoa_code}
    Returns historical crime stats for all crime types for the given LSOA.
    """
    filtered_df = _controller.filter_historical_df(
        df=request.app.state.HISTORICAL_DF,
        lsoa_map=request.app.state.LSOA_MAP,
        police_station=policeforce,
        lsoa_code=lsoa_code,
    )

    records = filtered_df.rename(columns=_COLUMN_MAP).to_dict(orient="records")

    return {
        "police_force": policeforce,
        "lsoa_code": lsoa_code,
        "crime_type": None,
        "count": len(records),
        "data": records,
    }
