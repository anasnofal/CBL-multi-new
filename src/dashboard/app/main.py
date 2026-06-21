import json
from pathlib import Path
from .helpers.config import get_settings
import pandas as pd
from .routes import base, forecast, alerts, historical, savings, explain
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

settings = get_settings()


@asynccontextmanager
async def startup_event(app: FastAPI):
    print("Starting up the dashboard server...")
    crime_costs = pd.read_csv(settings.CRIME_COSTS_DIR, sep=";")
    crime_costs.columns = crime_costs.columns.str.strip()
    for col in crime_costs.select_dtypes(include="object").columns:
        crime_costs[col] = crime_costs[col].str.strip()
    app.state.CRIME_COSTS = crime_costs
    app.state.FORECAST_DF = pd.read_csv(settings.DATA_DIR)
    app.state.HISTORICAL_DF = pd.read_csv(settings.HISTORICAL_ANALYSIS_PATH)
    app.state.FORECAST_DF["Month"] = pd.to_datetime(app.state.FORECAST_DF["Month"])
    with open(settings.LSOA_DATA_DIR, encoding="utf-8") as f:
        app.state.LSOA_MAP = json.load(f)
    geojson_data = json.loads(Path(settings.GEOJSON_PATH).read_text(encoding="utf-8"))
    app.state.GEOJSON_INDEX = {
        f["properties"]["LSOA21CD"]: f for f in geojson_data["features"]
    }
    print(f"GeoJSON index built: {len(app.state.GEOJSON_INDEX)} features")

    yield


app = FastAPI(lifespan=startup_event)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base.base_router)
app.include_router(forecast.forecast_router)
app.include_router(alerts.alerts_router)
app.include_router(historical.historical_router)
app.include_router(savings.savings_router)
app.include_router(explain.explain_router)
