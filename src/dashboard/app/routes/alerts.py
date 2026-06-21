from fastapi import APIRouter, Request, Query
from ..Controllers.AlertController import AlertController

alerts_router = APIRouter(
    prefix="/api/v1",
    tags=["alerts"],
)

_controller = AlertController()


@alerts_router.get("/alerts/{policeforce}")
async def get_alerts(
    policeforce: str,
    request: Request,
    crime_type: str = Query(...),
    time_start: str = Query(...),
    time_end: str = Query(...),
    window: int = Query(default=3),
    z_threshold: float = Query(default=1.5),
):
    result = _controller.get_alerts(
        police_force=policeforce,
        crime_type=crime_type,
        time_start=time_start,
        time_end=time_end,
        df=request.app.state.FORECAST_DF,
        lsoa_map=request.app.state.LSOA_MAP,
        window=window,
        z_threshold=z_threshold,
        )

    return result