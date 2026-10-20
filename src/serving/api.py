"""FastAPI application for the trained trajectory forecaster."""

from __future__ import annotations

import logging
import os
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.serving.inference import InferenceError, TrajectoryPredictor, default_checkpoint_path


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
    try:
        app.state.predictor = TrajectoryPredictor(path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        app.state.model_load_error = exc
        logger.error("Trajectory model is unavailable; configure a compatible checkpoint.")

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_request: Request, exc: RequestValidationError):
        # Do not echo the supplied coordinate values or internal Pydantic context.
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

    @app.get("/ready", tags=["service"])
    def ready():
        """Report whether a compatible checkpoint has loaded and can serve predictions."""
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="Prediction model is not ready.")
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(request: PredictionRequest):
        predictor = predictor_or_503()
        try:
            forecast = predictor.predict(request.trajectory)
        except InferenceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Trajectory prediction failed.")
            raise HTTPException(status_code=500, detail="Prediction failed.") from exc
        return PredictionResponse(forecast=forecast)

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
    def predict_batch(request: BatchPredictionRequest):
        predictor = predictor_or_503()
        try:
            forecasts = predictor.predict_batch(request.trajectories)
        except InferenceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Batch trajectory prediction failed.")
            raise HTTPException(status_code=500, detail="Batch prediction failed.") from exc
        return BatchPredictionResponse(forecasts=forecasts)

    return app


app = create_app()
