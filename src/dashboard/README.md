# `src/dashboard` — Crime Forecast Dashboard

Interactive dashboard for the 12-month crime forecasts produced by
[`src/train`](../train). A **FastAPI** backend serves forecast, alert, historical,
savings, and explanation data; a **Vue + Vite** frontend renders the map and charts.

## Layout

| Path | Role |
|------|------|
| `app/` | FastAPI backend (`main.py`, `routes/`, `Controllers/`, `helpers/`). |
| `app/data/` | CSV/JSON inputs: forecasts, historical analysis, GeoJSON, dictionaries. |
| `frontend/` | Vue 3 + Vite SPA (Leaflet map, Chart.js). |
| `.env` | Paths to the data files (copy from `.env.example`). |

## Setup

1. Copy `.env.example` to `.env` and point the paths at your data files
   (`DATA_DIR`, `LSOA_DATA_DIR`, `GEOJSON_PATH`, etc.).

   you can find the data at Kaagle  https://www.kaggle.com/datasets/anasnofal/uk-police-crime-dataset-mar-2023-feb-2025
3. Install frontend deps once:

   ```bash
   cd frontend && npm install
   ```

## Run

The backend and frontend run as two processes, so use **two terminals** from
`src/dashboard`:

```bash
# Terminal 1 — backend (FastAPI on http://localhost:8000)
uv run uvicorn app.main:app --reload

# Terminal 2 — frontend (Vite on http://localhost:5173)
cd frontend && npm run dev
```

Open **http://localhost:5173**. CORS is configured for that origin; the frontend
calls the backend on port 8000.

> Note: `run.sh` lists both commands, but `uvicorn` blocks, so the `npm run dev`
> lines never execute. Run the two commands in separate terminals as shown above.
</content>
