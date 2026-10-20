"""Explicit ground-truth forecast quality evaluation for serving observations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from src.monitoring.metrics import haversine_distance_km, validate_coordinate_tensor


def evaluate_forecast_quality(
    predictions: object,
    ground_truth: object,
    *,
    ade_threshold_km: float | None = None,
    fde_threshold_km: float | None = None,
) -> dict[str, Any]:
    """Compute ADE/FDE only for explicitly supplied matching truth trajectories."""
    prediction_tensor = _coordinates(predictions, "predictions")
    truth_tensor = _coordinates(ground_truth, "ground_truth")
    if prediction_tensor.shape != truth_tensor.shape:
        raise ValueError("Predictions and ground_truth must have matching (samples, horizon, 2) shapes.")
    if prediction_tensor.ndim != 3 or prediction_tensor.shape[0] < 1 or prediction_tensor.shape[1] < 1:
        raise ValueError("Predictions and ground_truth must have non-empty (samples, horizon, 2) shapes.")
    _validate_threshold(ade_threshold_km, "ade_threshold_km")
    _validate_threshold(fde_threshold_km, "fde_threshold_km")

    distances = haversine_distance_km(prediction_tensor, truth_tensor)
    ade = float(distances.mean().item())
    fde = float(distances[:, -1].mean().item())
    breached = []
    if ade_threshold_km is not None and ade > ade_threshold_km:
        breached.append("ade")
    if fde_threshold_km is not None and fde > fde_threshold_km:
        breached.append("fde")
    threshold_status = ("threshold_breach" if breached else
                        "within_thresholds" if ade_threshold_km is not None or fde_threshold_km is not None else
                        "evaluated")
    return {
        "status": threshold_status,
        "sample_count": int(prediction_tensor.shape[0]),
        "horizon": int(prediction_tensor.shape[1]),
        "ade_km": ade,
        "fde_km": fde,
        "thresholds_km": {"ade": ade_threshold_km, "fde": fde_threshold_km},
        "breached_metrics": breached,
    }


def unavailable_forecast_quality() -> dict[str, Any]:
    """Explicitly represent that no ground truth was supplied."""
    return {"status": "unavailable", "reason": "ground_truth_not_supplied"}


def _coordinates(value: object, name: str) -> torch.Tensor:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates.") from exc
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (samples, horizon, 2).")
    tensor = torch.as_tensor(array, dtype=torch.float64)
    try:
        validate_coordinate_tensor(tensor, name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains invalid geographic coordinates: {exc}.") from exc
    return tensor


def _validate_threshold(value: float | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number or None.")
