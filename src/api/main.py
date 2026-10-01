"""FastAPI prediction service.

Run locally:  uvicorn src.api.main:app --reload
Docs:         http://localhost:8000/docs

Request flow for POST /predict:
  metrics middleware (start timer)
  -> API-key check (only if API_KEY is set)
  -> Pydantic validation (bad input -> 422, model never sees it)
  -> model.predict_proba -> apply threshold -> response
  -> input stored in the drift window
  -> middleware records latency + status for Prometheus
"""

import logging
import secrets
import socket
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.logging_utils import setup_logging
from src.api.metrics import (
    DRIFT_DETECTED,
    DRIFT_SHARE,
    DRIFT_WINDOW_SIZE,
    HTTP_LATENCY,
    HTTP_REQUESTS,
    MALIGNANT_PROBABILITY,
    MODEL_INFO,
    PREDICTIONS,
)
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    PatientFeatures,
    Prediction,
    PredictionResponse,
)
from src.api.settings import get_settings
from src.models.bundle import load_bundle
from src.monitoring.drift import DriftDetector

logger = logging.getLogger("cancer_api")
HOSTNAME = socket.gethostname()  # the pod name in Kubernetes, the container id in Docker

MODEL_INFO_KEYS = [
    "model_name",
    "model_version",
    "model_type",
    "threshold",
    "test_metrics",
    "test_metrics_95ci",
    "trained_at",
    "mlflow_run_id",
    "git_sha",
    "sklearn_version",
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs ONCE at startup, before any request is accepted.

    The model is loaded here (not per request: loading takes ~100ms+, predicting
    takes ~1ms). If loading fails, the exception stops the server from starting:
    fail fast. A server that is "up" but has no model is worse than one that is
    clearly down, because health checks and load balancers would send it traffic.
    """
    settings = get_settings()
    setup_logging(settings.log_level)
    bundle = load_bundle(settings.model_dir)

    app.state.settings = settings
    app.state.bundle = bundle
    app.state.drift_detector = DriftDetector(
        bundle.reference[bundle.feature_columns],
        alpha=settings.drift_alpha,
        min_samples=settings.drift_min_samples,
    )
    # Ring buffer of recent inputs: when full, the oldest row falls out.
    app.state.drift_window = deque(maxlen=settings.drift_window)

    MODEL_INFO.info(
        {
            "name": bundle.metadata["model_name"],
            "version": bundle.version,
            "type": bundle.metadata["model_type"],
        }
    )
    logger.info(
        "model loaded",
        extra={"extra_fields": {"model_version": bundle.version, "threshold": bundle.threshold}},
    )
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Cancer Prediction API",
    description="Breast cancer (malignant/benign) prediction from 30 cell-nucleus features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        # Which container/pod answered: makes load balancing visible to clients.
        response.headers["X-Served-By"] = HOSTNAME
        return response
    finally:
        elapsed = time.perf_counter() - start
        # Use the route TEMPLATE ("/predict"), not the raw URL, to keep label values bounded.
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        if path != "/metrics":  # don't measure Prometheus scraping itself
            HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            HTTP_LATENCY.labels(request.method, path).observe(elapsed)
            logger.info(
                "request",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "path": path,
                        "status": status,
                        "duration_ms": round(elapsed * 1000, 2),
                    }
                },
            )


def verify_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.settings.api_key
    if expected is None:
        return  # auth disabled (local dev)
    # compare_digest takes the same time whether the first or last character is wrong,
    # so an attacker cannot guess the key one character at a time by measuring response time.
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _predict(request: Request, rows: list[PatientFeatures]) -> list[Prediction]:
    bundle = request.app.state.bundle
    # Column ORDER must match training; build the frame from the bundle's feature list.
    X = pd.DataFrame([r.model_dump() for r in rows])[bundle.feature_columns]
    proba = bundle.model.predict_proba(X)[:, 1]

    request.app.state.drift_window.extend(X.to_dict(orient="records"))
    DRIFT_WINDOW_SIZE.set(len(request.app.state.drift_window))

    results = []
    for p in proba:
        diagnosis: Literal["malignant", "benign"] = (
            "malignant" if p >= bundle.threshold else "benign"
        )
        PREDICTIONS.labels(diagnosis).inc()
        MALIGNANT_PROBABILITY.observe(float(p))
        results.append(
            Prediction(
                diagnosis=diagnosis,
                malignant_probability=round(float(p), 4),
                threshold=round(bundle.threshold, 4),
            )
        )
    return results


@app.get("/")
def root() -> dict:
    return {"service": "cancer-prediction-api", "docs": "/docs", "served_by": HOSTNAME}


@app.get("/health")
def health() -> dict:
    """Liveness: 'is the process alive?' If this fails, Kubernetes RESTARTS the container."""
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> JSONResponse:
    """Readiness: 'can I take traffic?'

    If this fails, Kubernetes stops SENDING traffic to the pod (but does not restart it).
    """
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return JSONResponse({"status": "not ready"}, status_code=503)
    return JSONResponse({"status": "ready", "model_version": bundle.version})


@app.get("/model/info")
def model_info(request: Request) -> dict:
    m = request.app.state.bundle.metadata
    return {k: m.get(k) for k in MODEL_INFO_KEYS}


# Plain `def` (not `async def`): predict_proba is CPU work. FastAPI runs sync endpoints
# in a thread pool, so one slow prediction doesn't freeze the event loop for everyone.
@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def predict(request: Request, features: PatientFeatures) -> PredictionResponse:
    result = _predict(request, [features])[0]
    return PredictionResponse(**result.model_dump(), model_version=request.app.state.bundle.version)


@app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(verify_api_key)])
def predict_batch(request: Request, batch: BatchRequest) -> BatchResponse:
    results = _predict(request, batch.instances)
    return BatchResponse(
        predictions=results, count=len(results), model_version=request.app.state.bundle.version
    )


@app.get("/drift")
def drift(request: Request) -> dict:
    """Compare recent inputs (this replica's window) against the training data."""
    window = list(request.app.state.drift_window)
    detector: DriftDetector = request.app.state.drift_detector
    if len(window) < detector.min_samples:
        return {
            "status": "insufficient_data",
            "n_current": len(window),
            "min_samples": detector.min_samples,
        }
    report = detector.detect(pd.DataFrame(window))
    DRIFT_SHARE.set(report.share_drifted)
    DRIFT_DETECTED.set(int(report.dataset_drift))
    return {"status": "ok", **report.to_dict()}


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrapes this endpoint every few seconds (pull model)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
