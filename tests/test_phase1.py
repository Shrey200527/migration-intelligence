"""Small CPU-only Phase 1 checks runnable with unittest or pytest."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.data_prep import TrajectoryConfig, build_trajectory_sequences
from src.data.validation import validate_dataset, write_quality_report
from src.models.baseline import persistence_forecast
from src.models.train import _comparison, build_parser, evaluate_ade, evaluate_fde, run_training
from src.monitoring.metrics import haversine_distance_km


def rows_for(count=120, animals=("A",)):
    result = []
    for a_idx, animal in enumerate(animals):
        for i in range(count):
            result.append({"timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i),
                           "individual-local-identifier": animal,
                           "location-long": 10 + a_idx + i * .01,
                           "location-lat": 40 + i * .01})
    return result


class Fixed(torch.nn.Module):
    def forward(self, past):
        return torch.tensor([[[1., 0.], [1., 0.]]], device=past.device).expand(past.shape[0], -1, -1)


class InvalidPrediction(torch.nn.Module):
    def __init__(self, coords):
        super().__init__()
        self.coords = coords

    def forward(self, past):
        return self.coords.to(past.device).expand(past.shape[0], -1, -1)


class PhaseOneTests(unittest.TestCase):
    def test_validation_schema_invalids_duplicates_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gps.csv"
            data = rows_for(120)
            data += [dict(data[0]), {"timestamp": "bad", "individual-local-identifier": "A",
                                     "location-long": 0, "location-lat": 91},
                    {"timestamp": "2025-01-01", "individual-local-identifier": "",
                     "location-long": 0, "location-lat": 0},
                    {"timestamp": "2025-01-02", "individual-local-identifier": "A",
                     "location-long": "not-a-number", "location-lat": 0},
                    {"timestamp": "2025-01-03", "individual-local-identifier": "A",
                     "location-long": 0, "location-lat": "inf"},
                    {"timestamp": "2025-01-04", "individual-local-identifier": "A",
                     "location-long": 181, "location-lat": 0}]
            pd.DataFrame(data).to_csv(path, index=False)
            report = validate_dataset(path, TrajectoryConfig(2, 1, 3, None), chunksize=11)
            self.assertTrue(report["usable"])
            self.assertEqual(report["counts"]["duplicate_records"], 1)
            self.assertEqual(report["counts"]["selected_duplicate_records"], 1)
            self.assertEqual(report["counts"]["invalid_timestamps"], 1)
            self.assertEqual(report["counts"]["invalid_coordinates"], 4)
            self.assertEqual(report["counts"]["missing_identifiers"], 1)
            self.assertEqual(report["counts"]["invalid_records"], 5)
            out = Path(tmp) / "quality.json"
            write_quality_report(report, out)
            self.assertEqual(json.loads(out.read_text())["individual_count"], 1)
            bad = Path(tmp) / "bad.csv"
            pd.DataFrame({"lat": [0]}).to_csv(bad, index=False)
            self.assertFalse(validate_dataset(bad)["usable"])

    def test_exact_duplicates_within_and_across_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicates.csv"
            duplicate = rows_for(1)[0]
            pd.DataFrame([duplicate, duplicate, *rows_for(40)[1:]]).to_csv(path, index=False)
            cfg = TrajectoryConfig(2, 1, 3, None)
            within_chunk = validate_dataset(path, cfg, chunksize=20)
            across_chunks = validate_dataset(path, cfg, chunksize=1)
            self.assertEqual(within_chunk["counts"]["duplicate_records"], 1)
            self.assertEqual(across_chunks["counts"]["duplicate_records"], 1)
            self.assertEqual(within_chunk["valid_observations"], across_chunks["valid_observations"])

    def test_validation_uses_same_max_animal_selection_as_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.csv"
            rows = rows_for(8, ("early_short",)) + rows_for(100, ("later_long",))
            # Give the later animal later first appearance, matching CSV order.
            pd.DataFrame(rows).to_csv(path, index=False)
            first_only = TrajectoryConfig(3, 2, 5, 1)
            rejected = validate_dataset(path, first_only, chunksize=6)
            self.assertFalse(rejected["usable"])
            self.assertEqual(rejected["selection"]["selected_individual_count"], 1)
            self.assertEqual(rejected["observations_per_individual"]["max"], 8)
            self.assertNotIn("early_short", json.dumps(rejected))
            with self.assertRaisesRegex(ValueError, "Could not construct"):
                build_trajectory_sequences(str(path), first_only)

            both = TrajectoryConfig(3, 2, 5, 2)
            accepted = validate_dataset(path, both, chunksize=6)
            self.assertTrue(accepted["usable"])
            self.assertEqual(accepted["selection"]["selected_individual_count"], 2)
            self.assertEqual(accepted["tracks_below_minimum_for_splits"], 1)
            train, val, test = build_trajectory_sequences(str(path), both)
            self.assertTrue(train and val and test)

    def test_split_integrity_shapes_and_short_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gps.csv"
            pd.DataFrame(rows_for(120)).to_csv(path, index=False)
            cfg = TrajectoryConfig(4, 2, 10, None)
            tr, va, te = build_trajectory_sequences(str(path), cfg)
            for split in (tr, va, te):
                self.assertTrue(split)
                for past, future in split:
                    self.assertEqual(past.shape, (4, 2)); self.assertEqual(future.shape, (2, 2))
                    self.assertTrue(np.isfinite(past).all() and np.isfinite(future).all())
            # The partition coordinate ranges are strictly chronological and disjoint.
            self.assertLess(max(x[1][-1, 0] for x in tr), min(x[0][0, 0] for x in va))
            self.assertLess(max(x[1][-1, 0] for x in va), min(x[0][0, 0] for x in te))
            short = Path(tmp) / "short.csv"
            pd.DataFrame(rows_for(10)).to_csv(short, index=False)
            with self.assertRaisesRegex(ValueError, "Could not construct"):
                build_trajectory_sequences(str(short), cfg)

    def test_geographic_metrics_baseline_and_validation(self):
        identical = torch.zeros((2, 3, 2))
        self.assertTrue(torch.equal(haversine_distance_km(identical, identical), torch.zeros((2, 3))))
        north = torch.tensor([[0., 0.]])
        one_degree = torch.tensor([[1., 0.]])
        self.assertAlmostEqual(float(haversine_distance_km(north, one_degree)[0]), 111.195, delta=.1)
        past = torch.tensor([[[0., 0.], [1., 2.]], [[3., 4.], [5., 6.]]])
        pred = persistence_forecast(past, 3)
        self.assertEqual(tuple(pred.shape), (2, 3, 2))
        self.assertTrue(torch.equal(pred[:, 0], past[:, -1]))
        self.assertIsNone(_comparison(0.0, 0.0)["percentage_improvement"])
        with self.assertRaises(ValueError): persistence_forecast(torch.tensor([[[float("nan"), 0.]]]), 2)
        with self.assertRaises(ValueError): haversine_distance_km(torch.tensor([[91., 0.]]), torch.zeros((1, 2)))
        loader = DataLoader(TensorDataset(torch.zeros((2, 1, 2)), torch.zeros((2, 2, 2))), batch_size=1)
        self.assertGreater(evaluate_ade(Fixed(), loader, torch.device("cpu")), 0)
        self.assertGreater(evaluate_fde(Fixed(), loader, torch.device("cpu")), 0)
        with self.assertRaises(ValueError): evaluate_ade(Fixed(), DataLoader(TensorDataset(torch.empty((0, 1, 2)), torch.empty((0, 2, 2)))), torch.device("cpu"))
        with self.assertRaises(ValueError): evaluate_fde(Fixed(), DataLoader(TensorDataset(torch.empty((0, 1, 2)), torch.empty((0, 2, 2)))), torch.device("cpu"))
        valid_targets = DataLoader(TensorDataset(torch.zeros((1, 1, 2)), torch.zeros((1, 2, 2))), batch_size=1)
        with self.assertRaisesRegex(ValueError, "Model predictions.*latitude.*outside"):
            evaluate_ade(InvalidPrediction(torch.tensor([[[91., 0.], [91., 0.]]])), valid_targets, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "Model predictions.*non-finite"):
            evaluate_fde(InvalidPrediction(torch.tensor([[[float("nan"), 0.], [0., 0.]]])), valid_targets, torch.device("cpu"))

    def test_tiny_training_checkpoint_reload_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / "tiny.csv"
            pd.DataFrame(rows_for(100, ("A", "B"))).to_csv(data, index=False)
            args = build_parser().parse_args(["--data-path", str(data), "--input-length", "3",
                "--forecast-horizon", "2", "--min-points", "5", "--max-animals", "2",
                "--batch-size", "8", "--epochs", "1", "--hidden-size", "4", "--num-layers", "1",
                "--dropout", "0", "--device", "cpu", "--out-dir", str(root / "runs"), "--data-kind", "synthetic"])
            with patch("src.models.train._load_mlflow", return_value=None):
                run_dir = run_training(args)
                repeat_run_dir = run_training(args)
            for name in ("trajectory_lstm.pt", "dataset_quality.json", "experiment_summary.json", "experiment_report.md"):
                self.assertTrue((run_dir / name).is_file())
            summary = json.loads((run_dir / "experiment_summary.json").read_text())
            quality = json.loads((run_dir / "dataset_quality.json").read_text())
            self.assertEqual(summary["experiment_type"], "dataset_training")
            self.assertEqual(summary["data_kind"], "synthetic")
            self.assertEqual(quality["selection"]["max_animals"], args.max_animals)
            self.assertIn("lstm_ade_km", summary["test_metrics"])
            self.assertEqual(summary["training_validation_history"][0]["epoch"], 1)
            checkpoint = torch.load(run_dir / "trajectory_lstm.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["seed"], args.seed)
            self.assertEqual(checkpoint["model_cfg"]["forecast_horizon"], args.forecast_horizon)
            self.assertGreaterEqual(checkpoint["best_validation_loss_mse_degrees_squared"], 0)
            first_summary = summary
            second_summary = json.loads((repeat_run_dir / "experiment_summary.json").read_text())
            self.assertEqual(first_summary["test_metrics"], second_summary["test_metrics"])
            self.assertEqual(first_summary["training_validation_history"][0]["training_loss_mse_degrees_squared"],
                             second_summary["training_validation_history"][0]["training_loss_mse_degrees_squared"])
            self.assertEqual(first_summary["training_validation_history"][0]["validation_loss_mse_degrees_squared"],
                             second_summary["training_validation_history"][0]["validation_loss_mse_degrees_squared"])


if __name__ == "__main__":
    unittest.main()
