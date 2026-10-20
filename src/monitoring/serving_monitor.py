"""Bounded, privacy-conscious operational and input-drift monitoring."""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np


FEATURE_NAMES = ("mean_latitude", "mean_longitude", "mean_step_distance_km")
MAX_RECENT_TRAJECTORIES = 256


def load_drift_baseline(path: str | Path | None) -> dict[str, Any] | None:
    """Load a user-supplied aggregate baseline; no repository baseline is assumed."""
    if path is None:
        return None
    try:
        with Path(path).expanduser().open("r", encoding="utf-8") as stream:
            baseline = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Drift baseline could not be read as JSON.") from exc
    validate_drift_baseline(baseline)
    return baseline


def validate_drift_baseline(baseline: object) -> None:
    if not isinstance(baseline, dict):
        raise ValueError("Drift baseline must be a JSON object.")
    if (not isinstance(baseline.get("version"), str) or not baseline["version"].strip()
            or not isinstance(baseline.get("source"), str) or not baseline["source"].strip()):
        raise ValueError("Drift baseline must identify its version and source.")
    threshold = baseline.get("z_threshold")
    minimum = baseline.get("min_samples")
    features = baseline.get("features")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold <= 0):
        raise ValueError("Drift baseline z_threshold must be a positive finite number.")
    if type(minimum) is not int or not 2 <= minimum <= MAX_RECENT_TRAJECTORIES:
        raise ValueError(f"Drift baseline min_samples must be between 2 and {MAX_RECENT_TRAJECTORIES}.")
    if not isinstance(features, dict) or set(features) != set(FEATURE_NAMES):
        raise ValueError(f"Drift baseline features must contain exactly: {', '.join(FEATURE_NAMES)}.")
    for name in FEATURE_NAMES:
        values = features[name]
        if not isinstance(values, dict) or set(values) != {"mean", "std"}:
            raise ValueError(f"Drift baseline feature {name} must contain mean and std.")
        mean, std = values["mean"], values["std"]
        if (isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean)
                or isinstance(std, bool) or not isinstance(std, (int, float))
                or not math.isfinite(std) or std < 1e-12):
            raise ValueError(f"Drift baseline feature {name} requires finite mean and std >= 1e-12.")
        if ((name == "mean_latitude" and not -90 <= mean <= 90)
                or (name == "mean_longitude" and not -180 <= mean <= 180)
                or (name == "mean_step_distance_km" and mean < 0)):
            raise ValueError(f"Drift baseline feature {name} has an invalid geographic summary mean.")


def summarize_trajectory(trajectory: object) -> dict[str, float | None]:
    """Return scalar summaries for a valid trajectory without retaining its coordinates."""
    try:
        points = np.asarray(trajectory, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Trajectory summary requires numeric coordinates.") from exc
    if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 2:
        raise ValueError("Trajectory summary requires shape (sequence_length, 2).")
    if not np.isfinite(points).all():
        raise ValueError("Trajectory summary coordinates must be finite.")
    if (np.any(np.abs(points[:, 0]) > 90) or np.any(np.abs(points[:, 1]) > 180)):
        raise ValueError("Trajectory summary coordinates are outside geographic bounds.")
    # Circular averaging avoids an artificial mean near zero for points around +/-180 degrees.
    mean_longitude = _circular_mean_degrees(points[:, 1])
    if len(points) > 1:
        lat_a, lat_b = np.deg2rad(points[:-1, 0]), np.deg2rad(points[1:, 0])
        lon_a, lon_b = np.deg2rad(points[:-1, 1]), np.deg2rad(points[1:, 1])
        delta_lat, delta_lon = lat_b - lat_a, lon_b - lon_a
        haversine = np.sin(delta_lat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2) ** 2
        distances = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        mean_step = float(distances.mean())
    else:
        mean_step = 0.0
    return {"mean_latitude": float(points[:, 0].mean()),
            "mean_longitude": mean_longitude,
            "mean_step_distance_km": mean_step}


def _circular_mean_degrees(values: list[float | None] | np.ndarray) -> float | None:
    radians = np.deg2rad(np.asarray(values, dtype=np.float64))
    sine_mean = float(np.sin(radians).mean())
    cosine_mean = float(np.cos(radians).mean())
    if math.hypot(sine_mean, cosine_mean) <= 1e-12:
        return None
    mean = math.degrees(math.atan2(sine_mean, cosine_mean))
    # Normalize to [-180, 180), making equivalent antimeridian values consistent.
    return (mean + 180.0) % 360.0 - 180.0


@dataclass
class _EndpointCounts:
    requests: int = 0
    succeeded: int = 0
    failed: int = 0


class ServingMonitor:
    """Thread-safe counters and a fixed-size window of trajectory-level summaries."""

    def __init__(self, *, baseline: Mapping[str, Any] | None = None, model_ready: bool = False):
        baseline_snapshot = deepcopy(dict(baseline)) if baseline is not None else None
        if baseline_snapshot is not None:
            validate_drift_baseline(baseline_snapshot)
        self._baseline = baseline_snapshot
        self._model_ready = bool(model_ready)
        self._lock = threading.Lock()
        self._endpoints = {"/predict": _EndpointCounts(), "/predict/batch": _EndpointCounts()}
        self._status_counts: Counter[str] = Counter()
        self._batch_sizes: Counter[int] = Counter()
        self._sequence_lengths: Counter[int] = Counter()
        self._validation_failures = 0
        self._prediction_count = 0
        self._invalid_prediction_count = 0
        self._latency_count = 0
        self._latency_sum_ms = 0.0
        self._latency_max_ms = 0.0
        self._summaries: deque[dict[str, float]] = deque(maxlen=MAX_RECENT_TRAJECTORIES)

    def set_model_ready(self, ready: bool) -> None:
        with self._lock:
            self._model_ready = bool(ready)

    def record_request(self, endpoint: str, status_code: int,
                       prediction_count: int = 0, batch_size: int | None = None) -> None:
        if endpoint not in self._endpoints:
            return
        with self._lock:
            counts = self._endpoints[endpoint]
            counts.requests += 1
            if 200 <= status_code < 400:
                counts.succeeded += 1
                self._prediction_count += max(0, prediction_count)
            else:
                counts.failed += 1
            if status_code >= 400:
                self._status_counts[f"{status_code // 100}xx"] += 1
            if status_code == 422:
                self._validation_failures += 1
            if type(batch_size) is int and 1 <= batch_size <= 64:
                self._batch_sizes[batch_size] += 1

    def record_inference_latency(self, duration_ms: float) -> None:
        """Record duration of the predictor call only, in milliseconds."""
        latency = max(0.0, float(duration_ms))
        with self._lock:
            self._latency_count += 1
            self._latency_sum_ms += latency
            self._latency_max_ms = max(self._latency_max_ms, latency)

    def record_invalid_prediction_outputs(self, count: int) -> None:
        if type(count) is not int or count < 0:
            return
        with self._lock:
            self._invalid_prediction_count += count

    def record_successful_predictions(self, trajectories: list[object], forecasts: list[object]) -> None:
        """Record aggregate features only after inference has fully succeeded."""
        if len(trajectories) != len(forecasts):
            return
        prepared: list[dict[str, Any]] = []
        lengths: list[int] = []
        invalid_outputs = 0
        for trajectory, forecast in zip(trajectories, forecasts):
            try:
                prepared.append(summarize_trajectory(trajectory))
                sequence_length = int(np.asarray(trajectory).shape[0])
                if 1 <= sequence_length <= 512:
                    lengths.append(sequence_length)
                output = np.asarray(forecast, dtype=np.float64)
                if (output.ndim != 2 or output.shape[1] != 2 or not np.isfinite(output).all()
                        or np.any(np.abs(output[:, 0]) > 90) or np.any(np.abs(output[:, 1]) > 180)):
                    invalid_outputs += 1
            except (TypeError, ValueError, OverflowError):
                # Monitoring must not change inference outcomes.
                continue
        with self._lock:
            self._summaries.extend(prepared)
            self._sequence_lengths.update(lengths)
            self._invalid_prediction_count += invalid_outputs

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            endpoint_data = {name: {"requests": counts.requests, "succeeded": counts.succeeded,
                                    "failed": counts.failed}
                             for name, counts in self._endpoints.items()}
            status_counts = dict(sorted(self._status_counts.items()))
            batch_sizes = {str(size): self._batch_sizes[size] for size in sorted(self._batch_sizes)}
            sequence_lengths = {str(size): self._sequence_lengths[size]
                                for size in sorted(self._sequence_lengths)}
            total = sum(item["requests"] for item in endpoint_data.values())
            succeeded = sum(item["succeeded"] for item in endpoint_data.values())
            failed = sum(item["failed"] for item in endpoint_data.values())
            prediction_count = self._prediction_count
            invalid_predictions = self._invalid_prediction_count
            latency_count, latency_sum, latency_max = (self._latency_count, self._latency_sum_ms,
                                                        self._latency_max_ms)
            validation_failures = self._validation_failures
            model_ready = self._model_ready
            summaries = list(self._summaries)

        drift = self._drift_summary(summaries)
        return {
            "requests": {"total": total, "succeeded": succeeded, "failed": failed,
                         "by_endpoint": endpoint_data, "http_errors_by_status_class": status_counts,
                         "validation_failures": validation_failures,
                         "batch_size_distribution": batch_sizes},
            "predictions": {"count": prediction_count, "invalid_output_trajectories": invalid_predictions},
            "input_summary": {"sampled_trajectories": sum(sequence_lengths.values()),
                              "sequence_length_distribution": sequence_lengths},
            "inference_latency_ms": {
                "count": latency_count,
                "mean": (latency_sum / latency_count) if latency_count else None,
                "max": latency_max if latency_count else None,
            },
            "model": {"ready": model_ready},
            "drift": drift,
        }

    def _drift_summary(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        ambiguous_features = (["mean_longitude"] if any(
            item.get("mean_longitude") is None for item in summaries) else [])
        if self._baseline is None:
            return {"status": "baseline_unavailable", "samples": len(summaries),
                    "baseline_version": None, "threshold": None,
                    "drifted_features": [], "ambiguous_features": ambiguous_features}
        minimum = self._baseline["min_samples"]
        common = {"samples": len(summaries), "baseline_version": self._baseline["version"],
                  "threshold": self._baseline["z_threshold"]}
        if len(summaries) < minimum:
            return {"status": "insufficient_data", **common, "min_samples": minimum,
                    "drifted_features": [], "ambiguous_features": ambiguous_features}
        z_scores: dict[str, float] = {}
        for name in FEATURE_NAMES:
            values = [item[name] for item in summaries]
            if name == "mean_longitude":
                if name in ambiguous_features:
                    continue
                observed_mean = _circular_mean_degrees(values)
                if observed_mean is None:
                    ambiguous_features.append(name)
                    continue
            else:
                observed_mean = sum(values) / len(values)
            reference = self._baseline["features"][name]
            difference = observed_mean - reference["mean"]
            if name == "mean_longitude":
                difference = (difference + 180.0) % 360.0 - 180.0
            z_scores[name] = abs(difference) / reference["std"]
        threshold = self._baseline["z_threshold"]
        drifted = sorted(name for name, score in z_scores.items() if score > threshold)
        status = "drift_detected" if drifted else "ambiguous_data" if ambiguous_features else "no_drift_detected"
        return {"status": status,
                **common, "min_samples": minimum, "z_scores": z_scores,
                "drifted_features": drifted, "ambiguous_features": ambiguous_features}
