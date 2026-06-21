from fastapi import APIRouter, Request, HTTPException
from ..Controllers.ExplainabilityController import ExplainabilityController

explain_router = APIRouter(prefix="/api/v1", tags=["explain"])

_controller = ExplainabilityController()


@explain_router.get("/explain/{policeforce}/{lsoa_code}/{crime_type}/{month}")
async def get_explain(
    policeforce: str,
    lsoa_code: str,
    crime_type: str,
    month: str,
    request: Request,
):
    """
    GET /api/v1/explain/{policeforce}/{lsoa}/{crime_type}/{month}
    month format: YYYY-MM  (e.g. 2026-03)
    Returns the waterfall decomposition for a single forecast month.
    """
    result = _controller.get_explain(
        df=request.app.state.FORECAST_DF,
        lsoa_code=lsoa_code,
        crime_type=crime_type,
        month=month,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="No explanation data for this selection")
    return result
