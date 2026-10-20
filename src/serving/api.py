"""FastAPI application for the trained trajectory forecaster."""

from __future__ import annotations

import logging
import json
import os
import time
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.serving.inference import (InferenceError, InvalidForecastError, TrajectoryPredictor,
                                   default_checkpoint_path)
from src.monitoring.serving_monitor import ServingMonitor, load_drift_baseline


logger = logging.getLogger(__name__)
MAX_BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 512
Coordinate = tuple[float, float]
Trajectory = Annotated[list[Coordinate], Field(min_length=1, max_length=MAX_SEQUENCE_LENGTH)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trajectory: Trajectory = Field(
        description="Exactly input_length [latitude, longitude] pairs in degrees.",
        examples=[[[37.0, -122.0], [37.1, -121.9]]],
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trajectories: Annotated[list[Trajectory], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


class PredictionResponse(BaseModel):
    forecast: list[Coordinate] = Field(description="Forecast [latitude, longitude] pairs in degrees.")
    coordinate_order: str = "latitude,longitude"
    coordinate_units: str = "degrees"


class BatchPredictionResponse(BaseModel):
    forecasts: list[list[Coordinate]]
    coordinate_order: str = "latitude,longitude"
    coordinate_units: str = "degrees"


def _log_event(level: int, event: dict) -> None:
    """Emit one JSON log record; observability failures never affect serving."""
    try:
        logger.log(level, json.dumps(event, sort_keys=True, separators=(",", ":")))
    except Exception:
        pass


def create_app(checkpoint_path: str | None = None) -> FastAPI:
    """Create the service and attempt one model load; health remains available on failure."""
    app = FastAPI(
        title="Migration Intelligence Trajectory Forecasting API",
        version="1.0.0",
        description=(
            "Forecasts future GPS coordinates from raw latitude/longitude sequences. "
            "Inputs and outputs use degrees and preserve latitude, longitude ordering."
        ),
    )
    path = checkpoint_path or os.environ.get("MODEL_CHECKPOINT") or str(default_checkpoint_path())
    app.state.predictor = None
    app.state.model_load_error = None
    baseline = load_drift_baseline(os.environ.get("DRIFT_BASELINE_PATH"))
    try:
        app.state.predictor = TrajectoryPredictor(path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        app.state.model_load_error = exc
    app.state.monitor = ServingMonitor(baseline=baseline, model_ready=app.state.predictor is not None)
    _log_event(logging.INFO, {"event": "service_startup", "model_ready": app.state.predictor is not None,
                              "drift_baseline_configured": baseline is not None})

    @app.middleware("http")
    async def observe_business_request(request: Request, call_next):
        endpoint = request.url.path if request.url.path in ("/predict", "/predict/batch") else None
        started = time.perf_counter()
        response = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            if endpoint is not None:
                duration_ms = (time.perf_counter() - started) * 1000.0
                batch_size = getattr(request.state, "batch_size", None)
                _record_request_safely(app.state.monitor, endpoint, 500, 0, batch_size)
                _log_event(logging.ERROR, _request_log_fields(request, endpoint, 500, duration_ms, 0,
                                                               "server_error", batch_size))
            raise
        finally:
            if endpoint is not None and response is not None:
                duration_ms = (time.perf_counter() - started) * 1000.0
                predictions = getattr(request.state, "prediction_count", 0)
                batch_size = getattr(request.state, "batch_size", None)
                failure_category = _failure_category(status_code, request)
                _record_request_safely(app.state.monitor, endpoint, status_code, predictions, batch_size)
                _log_event(logging.WARNING if status_code >= 400 else logging.INFO,
                           _request_log_fields(request, endpoint, status_code, duration_ms,
                                              predictions, failure_category, batch_size))

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError):
        # Do not echo the supplied coordinate values or internal Pydantic context.
        request.state.failure_category = "request_schema"
        errors = [{"loc": error.get("loc", ()), "msg": error.get("msg", "Invalid request"),
                   "type": error.get("type", "value_error")} for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    def predictor_or_503() -> TrajectoryPredictor:
        predictor = app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Prediction model is not ready.")
        return predictor

    @app.get("/health", tags=["service"])
    def health():
        """Report that the HTTP process is responsive; this does not check model readiness."""
        return {"status": "ok"}

    @app.get("/metrics", tags=["monitoring"])
    def metrics():
        """Return process-local operational counters and an optional input-drift status."""
        return app.state.monitor.snapshot()

    @app.get("/ready", tags=["service"])
    def ready():
        """Report whether a compatible checkpoint has loaded and can serve predictions."""
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="Prediction model is not ready.")
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(request: PredictionRequest, http_request: Request):
        http_request.state.batch_size = 1
        predictor = predictor_or_503()
        started = time.perf_counter()
        try:
            forecast = predictor.predict(request.trajectory)
        except InferenceError as exc:
            http_request.state.failure_category = _inference_validation_category(str(exc))
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InvalidForecastError as exc:
            http_request.state.failure_category = "invalid_prediction_output"
            _record_invalid_outputs_safely(app.state.monitor, 1)
            _log_event(logging.ERROR, {"event": "invalid_prediction_output", "request_type": "single"})
            raise HTTPException(status_code=500, detail="Prediction failed.") from exc
        except Exception as exc:
            _log_event(logging.ERROR, {"event": "inference_failure", "request_type": "single"})
            raise HTTPException(status_code=500, detail="Prediction failed.") from exc
        finally:
            _record_inference_latency_safely(app.state.monitor, (time.perf_counter() - started) * 1000.0)
        _record_prediction_summaries(app.state.monitor, [request.trajectory], [forecast])
        http_request.state.prediction_count = 1
        return PredictionResponse(forecast=forecast)

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
    def predict_batch(request: BatchPredictionRequest, http_request: Request):
        http_request.state.batch_size = len(request.trajectories)
        predictor = predictor_or_503()
        started = time.perf_counter()
        try:
            forecasts = predictor.predict_batch(request.trajectories)
        except InferenceError as exc:
            http_request.state.failure_category = _inference_validation_category(str(exc))
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InvalidForecastError as exc:
            http_request.state.failure_category = "invalid_prediction_output"
            _record_invalid_outputs_safely(app.state.monitor, len(request.trajectories))
            _log_event(logging.ERROR, {"event": "invalid_prediction_output", "request_type": "batch"})
            raise HTTPException(status_code=500, detail="Batch prediction failed.") from exc
        except Exception as exc:
            _log_event(logging.ERROR, {"event": "inference_failure", "request_type": "batch"})
            raise HTTPException(status_code=500, detail="Batch prediction failed.") from exc
        finally:
            _record_inference_latency_safely(app.state.monitor, (time.perf_counter() - started) * 1000.0)
        _record_prediction_summaries(app.state.monitor, request.trajectories, forecasts)
        http_request.state.prediction_count = len(forecasts)
        return BatchPredictionResponse(forecasts=forecasts)

    return app


def _failure_category(status_code: int, request: Request) -> str | None:
    if status_code < 400:
        return None
    if status_code == 422:
        return getattr(request.state, "failure_category", "input_validation")
    if status_code == 503:
        return "model_unavailable"
    if status_code == 500:
        return getattr(request.state, "failure_category", "inference_failure")
    return "inference_failure" if status_code >= 500 else "http_error"


def _inference_validation_category(message: str) -> str:
    lowered = message.lower()
    if "numeric" in lowered:
        return "coordinate_type"
    if "finite" in lowered or "non-finite" in lowered:
        return "non_finite_coordinate"
    if "latitude" in lowered or "longitude" in lowered:
        return "coordinate_bounds"
    if "exactly" in lowered:
        return "sequence_length"
    return "coordinate_validation"


def _request_log_fields(request: Request, endpoint: str, status_code: int, duration_ms: float,
                        prediction_count: int, failure_category: str | None,
                        batch_size: int | None) -> dict:
    return {
        "event": "request_completed", "method": request.method, "endpoint": endpoint,
        "request_type": "batch" if endpoint == "/predict/batch" else "single",
        "batch_size": batch_size,
        "status_code": status_code, "status_category": f"{status_code // 100}xx",
        "failure_category": failure_category, "duration_ms": round(max(0.0, duration_ms), 3),
    }


def _record_request_safely(monitor: ServingMonitor, endpoint: str, status_code: int,
                           prediction_count: int, batch_size: int | None) -> None:
    try:
        monitor.record_request(endpoint, status_code, prediction_count, batch_size)
    except Exception:
        pass


def _record_inference_latency_safely(monitor: ServingMonitor, duration_ms: float) -> None:
    try:
        monitor.record_inference_latency(duration_ms)
    except Exception:
        pass


def _record_invalid_outputs_safely(monitor: ServingMonitor, count: int) -> None:
    try:
        monitor.record_invalid_prediction_outputs(count)
    except Exception:
        pass


def _record_prediction_summaries(monitor: ServingMonitor, trajectories: list, forecasts: list) -> None:
    try:
        monitor.record_successful_predictions(trajectories, forecasts)
    except Exception:
        # Monitoring is deliberately non-critical to prediction correctness.
        pass


app = create_app()
