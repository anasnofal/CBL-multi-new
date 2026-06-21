from fastapi import APIRouter, Request
from ..Controllers.SavingsController import SavingsController

savings_router = APIRouter(
    prefix="/api/v1/savings",
    tags=["savings"],
)

_controller = SavingsController()


@savings_router.get("/{crime_type}/{predicted_count}")
async def get_crime_savings(
    crime_type: str,
    predicted_count: int,
    request: Request,
):
    """
    GET /api/v1/savings/{crime_type}/{predicted_count}
    """

    savings_data = _controller.calculate_crime_savings(
        crime_costs_df=request.app.state.CRIME_COSTS,
        crime_type=crime_type,
        predicted_count=predicted_count,
    )

    return savings_data