from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text

from app.db import engine, init_db, telemetry

START_TIME = time.time()
LOCK = threading.Lock()
STATS = {
    "http_requests": 0,
    "telemetry_requests": 0,
    "predict_requests": 0,
    "errors_4xx": 0,
    "errors_5xx": 0,
    "predict_latency_sum_ms": 0.0,
}

FIELDS = ["robot_id", "timestamp", "x", "y", "speed", "battery",
          "motor_temp", "vibration", "motor_current", "obstacle_distance"]

MODEL = None
MODEL_FEATURES: list[str] = []
MODEL_VERSION = "v1"


def load_model() -> None:
    global MODEL, MODEL_FEATURES, MODEL_VERSION
    path = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        return
    bundle = joblib.load(path)
    if isinstance(bundle, dict):  # formato del script de entrenamiento
        MODEL = bundle["pipeline"]
        MODEL_FEATURES = bundle["features"]
        MODEL_VERSION = bundle.get("version", "v1")
    else:
        MODEL = bundle
        MODEL_FEATURES = ["speed", "battery", "motor_temp", "vibration",
                          "motor_current", "obstacle_distance"]
    # Una fila por solicitud: un solo hilo evita el costo de lanzar hilos
    try:
        MODEL.set_params(clf__n_jobs=1)
    except ValueError:
        pass  # el modelo no tiene ese parámetro (p. ej. regresión logística)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    load_model()
    yield


app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="1.0.0",
    description="Servicio de telemetría persistente con predicción de condición.",
    lifespan=lifespan,
)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception:
        with LOCK:
            STATS["http_requests"] += 1
            STATS["errors_5xx"] += 1
        raise
    with LOCK:
        STATS["http_requests"] += 1
        if response.status_code >= 500:
            STATS["errors_5xx"] += 1
        elif response.status_code >= 400:
            STATS["errors_4xx"] += 1
    return response


class Telemetry(BaseModel):
    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float = Field(allow_inf_nan=False)
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    speed: float = Field(ge=0, allow_inf_nan=False)
    battery: float = Field(ge=0, le=100, allow_inf_nan=False)
    motor_temp: float = Field(ge=-40, le=250, allow_inf_nan=False)
    vibration: float = Field(ge=0, allow_inf_nan=False)
    motor_current: float = Field(ge=0, allow_inf_nan=False)
    obstacle_distance: float = Field(ge=0, allow_inf_nan=False)


class PredictionRequest(Telemetry):
    pass


def row_to_dict(row) -> dict:
    return {k: row._mapping[k] for k in FIELDS}


@app.get("/health")
def health() -> dict:
    db_status, robots = "ok", None
    try:
        with engine.connect() as conn:
            robots = conn.execute(
                select(func.count(func.distinct(telemetry.c.robot_id)))
            ).scalar()
    except Exception:
        db_status = "error"
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "database": db_status,
        "model_loaded": MODEL is not None,
        "robots_stored": robots,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:
    with LOCK:
        STATS["telemetry_requests"] += 1
    with engine.begin() as conn:
        conn.execute(telemetry.insert().values(**sample.model_dump()))
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/robots")
def list_robots() -> list[dict]:
    q = (
        select(telemetry)
        .distinct(telemetry.c.robot_id)
        .order_by(telemetry.c.robot_id, telemetry.c.timestamp.desc(), telemetry.c.id.desc())
    )
    with engine.connect() as conn:
        return [row_to_dict(r) for r in conn.execute(q)]


@app.get("/robots/{robot_id}/history")
def robot_history(robot_id: str) -> list[dict]:
    q = (
        select(telemetry)
        .where(telemetry.c.robot_id == robot_id)
        .order_by(telemetry.c.timestamp.asc(), telemetry.c.id.asc())
    )
    with engine.connect() as conn:
        rows = [row_to_dict(r) for r in conn.execute(q)]
    if not rows:
        raise HTTPException(status_code=404, detail="Robot not found")
    return rows


@app.post("/predict")
def predict(req: PredictionRequest) -> dict:
    with LOCK:
        STATS["predict_requests"] += 1
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.perf_counter()
    X = pd.DataFrame([[getattr(req, f) for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
    proba = MODEL.predict_proba(X)[0]
    idx = int(proba.argmax())
    label = str(MODEL.classes_[idx])
    ms = round((time.perf_counter() - t0) * 1000, 2)
    with LOCK:
        STATS["predict_latency_sum_ms"] += ms
    return {
        "robot_id": req.robot_id,
        "predicted_status": label,
        "confidence": round(float(proba[idx]), 4),
        "model_version": MODEL_VERSION,
        "inference_ms": ms,
    }


@app.get("/metrics")
def metrics() -> dict:
    with LOCK:
        s = dict(STATS)
    n = s["predict_requests"]
    return {
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "http_requests": s["http_requests"],
        "telemetry_requests": s["telemetry_requests"],
        "predict_requests": n,
        "errors_4xx": s["errors_4xx"],
        "errors_5xx": s["errors_5xx"],
        "avg_predict_latency_ms": round(s["predict_latency_sum_ms"] / n, 2) if n else 0.0,
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION,
        "persistence": "POSTGRESQL",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")