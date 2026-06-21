from typing import Optional
from fastapi import APIRouter, Request
from ..Controllers.ForecastController import ForecastController

forecast_router = APIRouter(
    prefix="/api/v1/forecast",
    tags=["forecast"],
)

_controller = ForecastController()


@forecast_router.get("/{policeforce}/{lsoa_code}/{crime_type}")
async def get_lsoa_crime_forecast(
    policeforce: str,
    lsoa_code: str,
    crime_type: str,
    request: Request,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
):
    """
    GET /api/v1/forecast/{policeforce}/{lsoa_code}/{crime_type}?time_start=2026-03&time_end=2026-06
    """
    filtered_df = _controller.filter_forecast_df(
        df=request.app.state.FORECAST_DF,
        lsoa_map=request.app.state.LSOA_MAP,
        police_station=policeforce,
        LSOA=lsoa_code,
        crime_type=crime_type,
        time_start=time_start,
        time_end=time_end,
    )

    return {
        "lsoa_code": lsoa_code,
        "crime_type": crime_type,
        "time_start": time_start,
        "time_end": time_end,
        "predictions": filtered_df[["Month", "predicted", "ci_lower", "ci_upper", "model"]].to_dict(orient="records"),
    }
