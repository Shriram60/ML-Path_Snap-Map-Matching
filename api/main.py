"""PathSnap API, runs on Laptop 1.

Start from the repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health   {"status": "ok"}
    GET  /trips    {"trips": [...]}  trip folders found in the evaluation folder
    POST /predict  {"trip_id": "..."} runs the full pipeline for that trip
"""
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api import config
from api.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")

TRIP_ID = re.compile(r"^[0-9A-Za-z_-]{1,64}$")
state = {}


def list_trips():
    folder = config.eval_dir()
    if not folder.is_dir():
        raise RuntimeError(f"Evaluation folder not found: {folder}")
    return sorted(p.name for p in folder.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


@asynccontextmanager
async def lifespan(app):
    pipeline = Pipeline()
    trips = list_trips()
    log.info("=" * 70)
    log.info("PathSnap API starting")
    log.info("evaluation folder: %s (%d trips)", config.eval_dir(), len(trips))
    log.info("pipeline:\n%s", pipeline.describe())
    if pipeline.uses_placeholder:
        log.warning("!!! PLACEHOLDER PIPELINE IN USE. Not the trained model. Do not demo or report. !!!")
    if trips:
        pipeline.warm_up(config.eval_dir() / trips[0])
    log.info("=" * 70)
    state["pipeline"] = pipeline
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)

# The UI is served from Laptop 2, a different origin, so the browser needs CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-PathSnap-Pipeline"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    client = request.client.host if request.client else "?"
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    log.info("%s %s %s -> %d (%.0f ms)", client, request.method, request.url.path, response.status_code, ms)
    pipeline = state.get("pipeline")
    if pipeline is not None:
        response.headers["X-PathSnap-Pipeline"] = "placeholder" if pipeline.uses_placeholder else "trained"
    return response


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    return {"trips": list_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    trip_id = req.trip_id.strip()
    # Only trip folders that exist in the evaluation folder are accepted,
    # so a request can never point the server at another path.
    if not TRIP_ID.match(trip_id) or trip_id not in list_trips():
        log.warning("unknown trip_id %r", req.trip_id)
        raise HTTPException(status_code=404, detail=f"Unknown trip_id: {req.trip_id}")

    pipeline = state["pipeline"]
    log.info("predict trip %s: running pipeline%s", trip_id,
             "  (PLACEHOLDER, not the trained model)" if pipeline.uses_placeholder else "")
    try:
        return pipeline.predict(config.eval_dir() / trip_id, log_prefix="predict")
    except Exception as exc:
        log.exception("predict trip %s failed", trip_id)
        raise HTTPException(status_code=500, detail=f"Pipeline failed on trip {trip_id}: {type(exc).__name__}: {exc}")
