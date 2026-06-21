from fastapi import APIRouter, Request
from datetime import datetime
from ..Controllers.BaseController import BaseController

base_router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"],
)

_controller = BaseController()


@base_router.get("/")
async def welcome():
    return {
        "app_name": "My App",
        "app_version": "1.0.0",
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@base_router.get("/lsoa/{policeforce}")
async def get_lsoa_codes(policeforce: str, request: Request):
    lsoas = _controller.get_lsoa_from_police_station(
        policeforce, request.app.state.LSOA_MAP
    )
    return {"police_force": policeforce, "lsoas": lsoas}


@base_router.get("/geojson/{policeforce}")
async def get_force_geojson(policeforce: str, request: Request):
    lsoa_status = _controller.get_lsoa_status_from_police_station(
        policeforce, request.app.state.LSOA_MAP
        )

    features = []

    for code, feature in request.app.state.GEOJSON_INDEX.items():
        if code not in lsoa_status:
            continue

        feature_copy = {
            "type": feature["type"],
            "properties": dict(feature["properties"]),
            "geometry": feature["geometry"]
        }

        feature_copy["properties"]["is_hotspot"] = lsoa_status[code]

        features.append(feature_copy)

    return {
            "type": "FeatureCollection",
            "features": features
        }
