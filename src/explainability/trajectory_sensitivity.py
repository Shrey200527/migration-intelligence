"""Controlled coordinate perturbations for trajectory LSTM predictions.

This measures prediction sensitivity to a specified coordinate change. It is
not a causal explanation, an estimate of uncertainty, or a geographic quality
metric against observed future locations.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from src.monitoring.metrics import haversine_distance_km


FEATURES = (("latitude", 0, 90.0), ("longitude", 1, 180.0))
METHOD = "one-coordinate-at-a-time symmetric perturbation"


def _dimensions(predictor: Any) -> tuple[int, int]:
    input_length = getattr(predictor, "input_length", None)
    horizon = getattr(predictor, "forecast_horizon", None)
    if type(input_length) is not int or input_length < 1:
        raise ValueError("Predictor must expose a positive integer input_length.")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Predictor must expose a positive integer forecast_horizon.")
    model = getattr(predictor, "model", None)
    config = getattr(model, "cfg", None)
    if config is not None and (
        getattr(config, "input_size", None) != 2
        or getattr(config, "forecast_horizon", None) != horizon
    ):
        raise ValueError("Predictor model configuration is incompatible with latitude/longitude forecasts.")
    if not callable(getattr(predictor, "predict", None)):
        raise ValueError("Predictor must provide a predict(trajectory) method.")
    return input_length, horizon


def _trajectory(value: object, input_length: int) -> np.ndarray:
    try:
        points = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Trajectory must contain numeric [latitude, longitude] pairs.") from exc
    if points.shape != (input_length, 2):
        raise ValueError(f"Trajectory must have shape ({input_length}, 2) in [latitude, longitude] order.")
    if not np.isfinite(points).all():
        raise ValueError("Trajectory coordinates must be finite.")
    if np.any(np.abs(points[:, 0]) > 90) or np.any(np.abs(points[:, 1]) > 180):
        raise ValueError("Trajectory contains coordinates outside latitude [-90, 90] or longitude [-180, 180].")
    return points


def _prediction(predictor: Any, trajectory: np.ndarray, horizon: int) -> np.ndarray:
    try:
        result = np.asarray(predictor.predict(trajectory.tolist()), dtype=np.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Predictor failed for a valid trajectory.") from exc
    if result.shape != (horizon, 2):
        raise ValueError(f"Predictor must return shape ({horizon}, 2) in [latitude, longitude] order.")
    if not np.isfinite(result).all():
        raise ValueError("Predictor returned non-finite forecast coordinates.")
    if np.any(np.abs(result[:, 0]) > 90) or np.any(np.abs(result[:, 1]) > 180):
        raise ValueError("Predictor returned coordinates outside valid latitude/longitude bounds.")
    return result


@contextmanager
def _evaluation_mode(predictor: Any) -> Iterator[None]:
    """Disable dropout for a supported PyTorch predictor and restore its mode."""
    model = getattr(predictor, "model", None)
    if not isinstance(model, torch.nn.Module):
        yield
        return
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _displacements(base: np.ndarray, changed: np.ndarray) -> list[float]:
    base_tensor = torch.as_tensor(base, dtype=torch.float64)
    changed_tensor = torch.as_tensor(changed, dtype=torch.float64)
    return [float(value) for value in haversine_distance_km(base_tensor, changed_tensor).tolist()]


def explain_trajectory_sensitivity(
    predictor: Any,
    trajectory: object,
    *,
    perturbation_degrees: float = 0.01,
) -> dict[str, Any]:
    """Return baseline and symmetric per-coordinate perturbed forecasts.

    Every input timestep and coordinate feature is perturbed by plus and minus
    ``perturbation_degrees``. Out-of-range perturbations are reported as skipped
    rather than wrapped or clamped. Forecast displacement is the corresponding
    horizon-wise Haversine distance from the original prediction.
    """
    if isinstance(perturbation_degrees, bool) or not isinstance(perturbation_degrees, (int, float)):
        raise ValueError("perturbation_degrees must be a finite positive number.")
    try:
        perturbation = float(perturbation_degrees)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("perturbation_degrees must be a finite positive number.") from exc
    if not math.isfinite(perturbation) or perturbation <= 0:
        raise ValueError("perturbation_degrees must be a finite positive number.")
    input_length, horizon = _dimensions(predictor)
    original = _trajectory(trajectory, input_length)
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    with _evaluation_mode(predictor):
        baseline = _prediction(predictor, original, horizon)
        for timestep in range(input_length):
            for feature, column, bound in FEATURES:
                for direction, sign in (("positive", 1.0), ("negative", -1.0)):
                    candidate = original.copy()
                    candidate[timestep, column] += sign * perturbation
                    if abs(candidate[timestep, column]) > bound:
                        skipped.append({
                            "timestep_index": timestep,
                            "feature": feature,
                            "direction": direction,
                            "reason": "perturbation_out_of_geographic_bounds",
                        })
                        continue
                    forecast = _prediction(predictor, candidate, horizon)
                    displacement = _displacements(baseline, forecast)
                    completed.append({
                        "timestep_index": timestep,
                        "feature": feature,
                        "direction": direction,
                        "delta_degrees": sign * perturbation,
                        "prediction": forecast.tolist(),
                        "forecast_displacement_km_by_horizon": displacement,
                        "mean_forecast_displacement_km": float(np.mean(displacement)),
                        "max_forecast_displacement_km": float(np.max(displacement)),
                    })

    return {
        "schema_version": 1,
        "method": METHOD,
        "interpretation": {
            "measures": "Prediction sensitivity to the listed coordinate perturbation.",
            "does_not_establish": [
                "causality", "general feature importance", "model uncertainty", "forecast accuracy"
            ],
            "distance_definition": "Haversine displacement between matching forecast steps, in kilometres.",
            "temporal_note": "Input rows are indexed by timestep; timestamps and sampling intervals are not model inputs.",
        },
        "reproducibility": {
            "random_sampling": False,
            "torch_model_mode": "eval during calculation when predictor exposes a PyTorch model; prior mode restored",
            "scope": "Repeatable for a deterministic predictor, fixed checkpoint, and compatible software/hardware environment.",
        },
        "ground_truth_evaluation": {
            "status": "unavailable",
            "reason": "ground_truth_not_supplied",
        },
        "dimensions": {
            "input_shape": [input_length, 2],
            "output_shape": [horizon, 2],
            "coordinate_order": ["latitude", "longitude"],
            "coordinate_units": "degrees",
        },
        "perturbation": {
            "design": "one coordinate at one timestep at a time; positive and negative directions",
            "magnitude_degrees": perturbation,
            "boundary_policy": "skip out-of-bounds perturbations; do not clamp or wrap",
        },
        "baseline_prediction": baseline.tolist(),
        "completed_perturbations": completed,
        "skipped_perturbations": skipped,
    }


def write_sensitivity_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a strict JSON report, rejecting non-JSON numeric values."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a coordinate-perturbation sensitivity report.")
    parser.add_argument("--checkpoint", required=True, help="Compatible trajectory LSTM checkpoint.")
    parser.add_argument("--trajectory-json", required=True,
                        help="JSON array of input_length [latitude, longitude] coordinate pairs.")
    parser.add_argument("--perturbation-degrees", type=float, default=0.01)
    parser.add_argument("--output", help="Report path; writes JSON to stdout when omitted.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from src.serving.inference import TrajectoryPredictor

    try:
        trajectory = json.loads(args.trajectory_json)
        predictor = TrajectoryPredictor(args.checkpoint, device=args.device)
        report = explain_trajectory_sensitivity(
            predictor, trajectory, perturbation_degrees=args.perturbation_degrees
        )
        if args.output:
            write_sensitivity_report(report, args.output)
        else:
            print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
