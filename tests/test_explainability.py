import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.explainability.trajectory_sensitivity import (
    explain_trajectory_sensitivity,
    write_sensitivity_report,
)
from src.monitoring.metrics import EARTH_RADIUS_KM
from src.models.model import build_model
from src.serving.inference import TrajectoryPredictor


class EchoLastPointPredictor:
    input_length = 1
    forecast_horizon = 2

    def predict(self, trajectory):
        return [trajectory[-1], trajectory[-1]]


class TrajectorySensitivityTests(unittest.TestCase):
    def test_valid_output_shapes_and_finite_values(self):
        result = explain_trajectory_sensitivity(EchoLastPointPredictor(), [[10.0, 20.0]])
        self.assertEqual(result["dimensions"]["input_shape"], [1, 2])
        self.assertEqual(result["dimensions"]["output_shape"], [2, 2])
        self.assertEqual(len(result["baseline_prediction"]), 2)
        self.assertEqual(len(result["completed_perturbations"]), 4)
        serialized = json.dumps(result, allow_nan=False)
        self.assertIn('"status": "unavailable"', serialized)
        self.assertTrue(all(math.isfinite(value)
                            for item in result["completed_perturbations"]
                            for value in item["forecast_displacement_km_by_horizon"]))

    def test_sensitivity_is_reproducible(self):
        trajectory = [[10.0, 20.0]]
        first = explain_trajectory_sensitivity(EchoLastPointPredictor(), trajectory)
        second = explain_trajectory_sensitivity(EchoLastPointPredictor(), trajectory)
        self.assertEqual(first, second)

    def test_input_is_not_mutated(self):
        trajectory = [[10.0, 20.0]]
        original = copy.deepcopy(trajectory)
        explain_trajectory_sensitivity(EchoLastPointPredictor(), trajectory)
        self.assertEqual(trajectory, original)

    def test_haversine_displacement_matches_known_synthetic_coordinate_change(self):
        delta = 0.01
        result = explain_trajectory_sensitivity(
            EchoLastPointPredictor(), [[0.0, 0.0]], perturbation_degrees=delta
        )
        north = next(item for item in result["completed_perturbations"]
                     if item["feature"] == "latitude" and item["direction"] == "positive")
        expected_km = EARTH_RADIUS_KM * math.radians(delta)
        self.assertAlmostEqual(north["forecast_displacement_km_by_horizon"][0], expected_km, places=5)
        self.assertAlmostEqual(north["mean_forecast_displacement_km"], expected_km, places=5)

    def test_out_of_bounds_perturbations_are_reported_as_skipped(self):
        result = explain_trajectory_sensitivity(EchoLastPointPredictor(), [[90.0, 180.0]])
        skipped = {(item["feature"], item["direction"]) for item in result["skipped_perturbations"]}
        self.assertEqual(skipped, {("latitude", "positive"), ("longitude", "positive")})
        self.assertEqual(len(result["completed_perturbations"]), 2)

    def test_antimeridian_positive_longitude_perturbation_is_skipped_but_negative_is_evaluated(self):
        result = explain_trajectory_sensitivity(
            EchoLastPointPredictor(), [[10.0, 180.0]], perturbation_degrees=0.01
        )
        longitude_skips = [item for item in result["skipped_perturbations"]
                           if item["feature"] == "longitude"]
        self.assertEqual(longitude_skips, [{
            "timestep_index": 0,
            "feature": "longitude",
            "direction": "positive",
            "reason": "perturbation_out_of_geographic_bounds",
        }])
        longitude_evaluations = [item for item in result["completed_perturbations"]
                                 if item["feature"] == "longitude"]
        self.assertEqual(len(longitude_evaluations), 1)
        self.assertEqual(longitude_evaluations[0]["timestep_index"], 0)
        self.assertEqual(longitude_evaluations[0]["direction"], "negative")
        self.assertEqual(longitude_evaluations[0]["delta_degrees"], -0.01)

    def test_input_validation_rejects_shape_nonfinite_and_bounds(self):
        predictor = EchoLastPointPredictor()
        for trajectory, message in (
            ([[1.0, 2.0], [3.0, 4.0]], "shape"),
            ([[float("nan"), 2.0]], "finite"),
            ([[91.0, 2.0]], "outside"),
            ([[1.0, 181.0]], "outside"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                explain_trajectory_sensitivity(predictor, trajectory)

    def test_invalid_perturbation_configuration_and_incompatible_predictors_fail_clearly(self):
        for amount in (0, -1, float("nan"), float("inf"), True, 10 ** 400):
            with self.subTest(amount=amount), self.assertRaisesRegex(ValueError, "perturbation_degrees"):
                explain_trajectory_sensitivity(EchoLastPointPredictor(), [[1, 2]],
                                               perturbation_degrees=amount)
        incompatible = EchoLastPointPredictor()
        incompatible.model = SimpleNamespace(cfg=SimpleNamespace(input_size=3, forecast_horizon=2))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            explain_trajectory_sensitivity(incompatible, [[1, 2]])
        with self.assertRaisesRegex(ValueError, "input_length"):
            explain_trajectory_sensitivity(SimpleNamespace(predict=lambda value: value), [[1, 2]])

    def test_prediction_contract_and_json_schema_are_checked(self):
        class BadShapePredictor(EchoLastPointPredictor):
            def predict(self, trajectory):
                return [[1.0, 2.0]]

        with self.assertRaisesRegex(ValueError, "return shape"):
            explain_trajectory_sensitivity(BadShapePredictor(), [[1.0, 2.0]])
        report = explain_trajectory_sensitivity(EchoLastPointPredictor(), [[10.0, 20.0]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "explanation.json"
            write_sensitivity_report(report, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded, report)
        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["method"], "one-coordinate-at-a-time symmetric perturbation")
        self.assertNotIn("input_trajectory", loaded)

    def test_temporary_trajectory_lstm_checkpoint_uses_production_predictor(self):
        torch.manual_seed(7)
        model, config = build_model(hidden_size=4, num_layers=1, dropout=0.0, forecast_horizon=2)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint_path = Path(temp) / "synthetic.pt"
            torch.save({
                "model_state": model.state_dict(),
                "model_cfg": config.__dict__,
                "traj_cfg": {"input_length": 3, "forecast_horizon": 2},
            }, checkpoint_path)
            predictor = TrajectoryPredictor(checkpoint_path, device="cpu")
            predictor.model.train()
            trajectory = [[40.0, -3.0], [40.01, -2.99], [40.02, -2.98]]
            first = explain_trajectory_sensitivity(predictor, trajectory)
            second = explain_trajectory_sensitivity(predictor, trajectory)
        self.assertEqual(first, second)
        self.assertEqual(first["dimensions"]["output_shape"], [2, 2])
        self.assertEqual(len(first["baseline_prediction"]), 2)
        self.assertTrue(predictor.model.training)


if __name__ == "__main__":
    unittest.main()
