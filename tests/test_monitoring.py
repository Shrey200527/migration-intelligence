import json
import logging
import copy
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch

from src.monitoring.forecast_quality import evaluate_forecast_quality, unavailable_forecast_quality
from src.monitoring.serving_monitor import (
    MAX_RECENT_TRAJECTORIES,
    ServingMonitor,
    load_drift_baseline,
    summarize_trajectory,
    validate_drift_baseline,
)
from src.serving.api import create_app
from tests.test_serving import asgi_request, write_checkpoint


def baseline(min_samples=3):
    return {
        "version": "synthetic-test-v1",
        "source": "explicit synthetic test fixture",
        "z_threshold": 3.0,
        "min_samples": min_samples,
        "features": {
            "mean_latitude": {"mean": 10.0, "std": 1.0},
            "mean_longitude": {"mean": 20.0, "std": 1.0},
            "mean_step_distance_km": {"mean": 0.0, "std": 0.25},
        },
    }


class ServingMonitorTests(unittest.TestCase):
    def test_request_counter_semantics_latency_units_and_operational_exclusions(self):
        monitor = ServingMonitor(model_ready=False)
        monitor.record_request("/predict", 200, prediction_count=1, batch_size=1)
        monitor.record_request("/predict", 422)
        monitor.record_request("/predict/batch", 503, batch_size=3)
        for latency in (10.0, 30.0, 20.0):
            monitor.record_inference_latency(latency)
        result = monitor.snapshot()
        self.assertEqual(result["requests"]["total"], 3)
        self.assertEqual(result["requests"]["succeeded"], 1)
        self.assertEqual(result["requests"]["failed"], 2)
        self.assertEqual(result["requests"]["validation_failures"], 1)
        self.assertEqual(result["requests"]["http_errors_by_status_class"], {"4xx": 1, "5xx": 1})
        self.assertEqual(result["predictions"]["count"], 1)
        self.assertEqual(result["inference_latency_ms"], {"count": 3, "mean": 20.0, "max": 30.0})
        self.assertFalse(result["model"]["ready"])
        monitor.record_request("/health", 200)
        monitor.record_request("/metrics", 200)
        self.assertEqual(monitor.snapshot()["requests"]["total"], 3)

    def test_counters_are_thread_safe(self):
        monitor = ServingMonitor()
        workers = 8
        increments = 250

        def record_many():
            for _ in range(increments):
                monitor.record_request("/predict", 200, prediction_count=1, batch_size=1)
                monitor.record_inference_latency(2.0)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda _: record_many(), range(workers)))
        snapshot = monitor.snapshot()
        expected = workers * increments
        self.assertEqual(snapshot["requests"]["total"], expected)
        self.assertEqual(snapshot["predictions"]["count"], expected)
        self.assertEqual(snapshot["inference_latency_ms"]["count"], expected)

    def test_input_summary_is_scalar_and_longitude_wraparound_aware(self):
        summary = summarize_trajectory([[10, 179], [10, -179]])
        self.assertEqual(summary["mean_latitude"], 10.0)
        self.assertAlmostEqual(abs(summary["mean_longitude"]), 180.0)
        self.assertLess(summary["mean_step_distance_km"], 250)
        self.assertEqual(set(summary), {"mean_latitude", "mean_longitude", "mean_step_distance_km"})

    def test_individual_trajectory_with_ambiguous_longitude_mean_is_explicit(self):
        summary = summarize_trajectory([[10, 0], [10, 180]])
        self.assertIsNone(summary["mean_longitude"])

    def test_bounded_window_and_nonfinite_prediction_detection(self):
        monitor = ServingMonitor()
        trajectory = [[10, 20], [10.01, 20.01]]
        for _ in range(MAX_RECENT_TRAJECTORIES + 10):
            monitor.record_successful_predictions([trajectory], [[[float("nan"), 20]]])
        result = monitor.snapshot()
        self.assertEqual(result["drift"]["samples"], MAX_RECENT_TRAJECTORIES)
        self.assertEqual(result["predictions"]["invalid_output_trajectories"], MAX_RECENT_TRAJECTORIES + 10)

    def test_drift_states_for_absent_baseline_insufficient_and_shifted_data(self):
        no_baseline = ServingMonitor().snapshot()["drift"]
        self.assertEqual(no_baseline["status"], "baseline_unavailable")

        stable = ServingMonitor(baseline=baseline())
        shifted = ServingMonitor(baseline=baseline())
        stable_trajectory = [[10.0, 20.0], [10.0, 20.0]]
        shifted_trajectory = [[50.0, 20.0], [50.0, 20.0]]
        for _ in range(2):
            stable.record_successful_predictions([stable_trajectory], [[[10.0, 20.0]]])
        self.assertEqual(stable.snapshot()["drift"]["status"], "insufficient_data")
        for _ in range(3):
            stable.record_successful_predictions([stable_trajectory], [[[10.0, 20.0]]])
            shifted.record_successful_predictions([shifted_trajectory], [[[30.0, 20.0]]])
        self.assertEqual(stable.snapshot()["drift"]["status"], "no_drift_detected")
        result = shifted.snapshot()["drift"]
        self.assertEqual(result["status"], "drift_detected")
        self.assertIn("mean_latitude", result["drifted_features"])
        self.assertNotIn("50.0", json.dumps(result))

    def test_circular_window_mean_handles_antimeridian_in_both_orders(self):
        reference = baseline(min_samples=2)
        reference["features"]["mean_longitude"]["mean"] = 180.0
        results = []
        for longitudes in ((179.0, -179.0), (-179.0, 179.0)):
            monitor = ServingMonitor(baseline=copy.deepcopy(reference))
            for longitude in longitudes:
                monitor.record_successful_predictions([[[10.0, longitude]]], [[[10.0, longitude]]])
            results.append(monitor.snapshot()["drift"])
        self.assertEqual(results[0]["status"], "no_drift_detected")
        self.assertEqual(results[0]["drifted_features"], [])
        self.assertAlmostEqual(results[0]["z_scores"]["mean_longitude"], 0.0, delta=1e-9)
        self.assertEqual(results[0]["status"], results[1]["status"])
        self.assertAlmostEqual(results[0]["z_scores"]["mean_longitude"],
                               results[1]["z_scores"]["mean_longitude"], delta=1e-12)

    def test_circular_window_mean_flags_a_real_longitude_shift(self):
        reference = baseline()
        reference["features"]["mean_longitude"]["mean"] = 180.0
        monitor = ServingMonitor(baseline=reference)
        for _ in range(3):
            monitor.record_successful_predictions([[[10.0, 150.0]]], [[[10.0, 150.0]]])
        result = monitor.snapshot()["drift"]
        self.assertEqual(result["status"], "drift_detected")
        self.assertIn("mean_longitude", result["drifted_features"])

    def test_circular_window_ambiguity_is_reported_without_a_longitude_score(self):
        monitor = ServingMonitor(baseline=baseline(min_samples=2))
        for longitude in (0.0, 180.0):
            monitor.record_successful_predictions([[[10.0, longitude]]], [[[10.0, longitude]]])
        result = monitor.snapshot()["drift"]
        self.assertEqual(result["status"], "ambiguous_data")
        self.assertEqual(result["ambiguous_features"], ["mean_longitude"])
        self.assertNotIn("mean_longitude", result["z_scores"])

    def test_caller_mutation_cannot_change_validated_baseline_snapshot(self):
        caller_baseline = baseline()
        monitor = ServingMonitor(baseline=caller_baseline)
        caller_baseline["version"] = "mutated-version"
        caller_baseline["features"]["mean_latitude"]["mean"] = 50.0
        caller_baseline["features"]["mean_latitude"]["std"] = 1e-20
        for _ in range(3):
            monitor.record_successful_predictions([[[10.0, 20.0]]], [[[10.0, 20.0]]])
        result = monitor.snapshot()["drift"]
        self.assertEqual(result["baseline_version"], "synthetic-test-v1")
        self.assertEqual(result["status"], "no_drift_detected")

    def test_zero_and_near_zero_baseline_standard_deviations_are_rejected(self):
        for std in (0.0, 0.5e-12):
            invalid = baseline()
            invalid["features"]["mean_longitude"]["std"] = std
            with self.subTest(std=std), self.assertRaisesRegex(ValueError, "std >= 1e-12"):
                ServingMonitor(baseline=invalid)
        boundary = baseline()
        boundary["features"]["mean_longitude"]["std"] = 1e-12
        self.assertEqual(ServingMonitor(baseline=boundary).snapshot()["drift"]["status"],
                         "insufficient_data")

    def test_baseline_validation_and_loading_errors_are_actionable(self):
        malformed = baseline()
        malformed["features"].pop("mean_latitude")
        with self.assertRaisesRegex(ValueError, "features must contain exactly"):
            validate_drift_baseline(malformed)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "baseline.json"
            path.write_text("{bad", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "could not be read as JSON"):
                load_drift_baseline(path)

    def test_monitoring_endpoint_and_request_logs_are_safe_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            app = create_app(str(Path(temp) / "missing.pt"))
            status, health = asgi_request(app, "GET", "/health")
            self.assertEqual((status, health), (200, {"status": "ok"}))
            status, ready = asgi_request(app, "GET", "/ready")
            self.assertEqual(status, 503)
            self.assertEqual(ready["detail"], "Prediction model is not ready.")
            status, metrics = asgi_request(app, "GET", "/metrics")
            self.assertEqual(status, 200)
            self.assertFalse(metrics["model"]["ready"])
            self.assertEqual(metrics["drift"]["status"], "baseline_unavailable")
            self.assertEqual(metrics["requests"]["total"], 0)

            body = {"trajectory": [[10, 20], [10, 20]]}
            with self.assertLogs("src.serving.api", level=logging.WARNING) as captured:
                status, _ = asgi_request(app, "POST", "/predict", body)
            self.assertEqual(status, 503)
            log_records = [json.loads(line.split(":", 2)[-1]) for line in captured.output]
            completion = next(record for record in log_records if record.get("event") == "request_completed")
            self.assertEqual(completion["failure_category"], "model_unavailable")
            self.assertIn("duration_ms", completion)
            self.assertNotIn("trajectory", json.dumps(completion))
            self.assertNotIn("10", json.dumps(completion))
            self.assertNotIn(temp, json.dumps(completion))

    def test_successful_inference_updates_business_metrics_and_emits_json_log(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.pt"
            write_checkpoint(checkpoint)
            app = create_app(str(checkpoint))
            sequence = [[20, 30], [20.1, 30.1], [20.2, 30.2], [20.3, 30.3]]
            with self.assertLogs("src.serving.api", level=logging.INFO) as captured:
                status, _ = asgi_request(app, "POST", "/predict", {"trajectory": sequence})
            self.assertEqual(status, 200)
            record = next(json.loads(line.split(":", 2)[-1]) for line in captured.output
                          if json.loads(line.split(":", 2)[-1]).get("event") == "request_completed")
            self.assertEqual(record["status_category"], "2xx")
            self.assertEqual(record["request_type"], "single")
            self.assertEqual(record["batch_size"], 1)
            metrics = asgi_request(app, "GET", "/metrics")[1]
            self.assertEqual(metrics["requests"]["total"], 1)
            self.assertEqual(metrics["predictions"]["count"], 1)
            self.assertEqual(metrics["drift"]["samples"], 1)
            self.assertEqual(metrics["requests"]["batch_size_distribution"], {"1": 1})
            self.assertEqual(metrics["input_summary"]["sequence_length_distribution"], {"4": 1})

    def test_coordinate_validation_logs_category_without_coordinates_or_duplicate_handlers(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.pt"
            write_checkpoint(checkpoint)
            logger = logging.getLogger("src.serving.api")
            handlers_before = tuple(logger.handlers)
            app = create_app(str(checkpoint))
            create_app(str(checkpoint))
            self.assertEqual(tuple(logger.handlers), handlers_before)
            body = {"trajectory": [[91, 20], [91, 20], [91, 20], [91, 20]]}
            with self.assertLogs("src.serving.api", level=logging.WARNING) as captured:
                status, _ = asgi_request(app, "POST", "/predict", body)
            self.assertEqual(status, 422)
            record = next(json.loads(line.split(":", 2)[-1]) for line in captured.output
                          if json.loads(line.split(":", 2)[-1]).get("event") == "request_completed")
            self.assertEqual(record["failure_category"], "coordinate_bounds")
            self.assertNotIn("91", json.dumps(record))

    def test_invalid_model_output_is_counted_and_returns_generic_error(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.pt"
            write_checkpoint(checkpoint)
            app = create_app(str(checkpoint))
            sequence = [[20, 30], [20.1, 30.1], [20.2, 30.2], [20.3, 30.3]]
            bad_output = torch.full((1, 3, 2), 91.0)
            with patch.object(app.state.predictor.model, "forward", return_value=bad_output):
                with self.assertLogs("src.serving.api", level=logging.WARNING) as captured:
                    status, payload = asgi_request(app, "POST", "/predict", {"trajectory": sequence})
            self.assertEqual(status, 500)
            self.assertEqual(payload["detail"], "Prediction failed.")
            self.assertNotIn(temp, json.dumps(payload))
            logs = [json.loads(line.split(":", 2)[-1]) for line in captured.output]
            completion = next(record for record in logs if record.get("event") == "request_completed")
            self.assertEqual(completion["failure_category"], "invalid_prediction_output")
            metrics = asgi_request(app, "GET", "/metrics")[1]
            self.assertEqual(metrics["predictions"]["invalid_output_trajectories"], 1)
            self.assertEqual(metrics["predictions"]["count"], 0)


class ForecastQualityTests(unittest.TestCase):
    def test_known_quality_and_threshold_breach(self):
        truth = [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]
        perfect = evaluate_forecast_quality(truth, truth, ade_threshold_km=0.1, fde_threshold_km=0.1)
        self.assertEqual(perfect["status"], "within_thresholds")
        self.assertEqual(perfect["sample_count"], 2)
        self.assertAlmostEqual(perfect["ade_km"], 0.0)
        shifted = [[[1.0, 0.0], [1.0, 0.0]], [[2.0, 1.0], [2.0, 1.0]]]
        result = evaluate_forecast_quality(shifted, truth, ade_threshold_km=10, fde_threshold_km=10)
        self.assertEqual(result["status"], "threshold_breach")
        self.assertEqual(result["breached_metrics"], ["ade", "fde"])
        self.assertGreater(result["fde_km"], 100)

    def test_invalid_shapes_coordinates_and_thresholds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            evaluate_forecast_quality([[[0, 0]]], [[[0, 0], [0, 0]]])
        with self.assertRaisesRegex(ValueError, "invalid geographic"):
            evaluate_forecast_quality([[[float("nan"), 0]]], [[[0, 0]]])
        with self.assertRaisesRegex(ValueError, "threshold"):
            evaluate_forecast_quality([[[0, 0]]], [[[0, 0]]], ade_threshold_km=-1)
        self.assertEqual(unavailable_forecast_quality(), {
            "status": "unavailable", "reason": "ground_truth_not_supplied"})


if __name__ == "__main__":
    unittest.main()
