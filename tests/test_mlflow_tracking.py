"""Offline MLflow integration checks using a small in-memory tracking client."""
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.data_prep import TrajectoryConfig, build_trajectory_sequences, save_prepared_sequences
from src.models.train import (build_parser, run_training, normalize_local_tracking_uri,
                              local_mlflow_artifact_uri)
from src.pipeline import main as pipeline_main


def _write_fixture(path: Path) -> None:
    rows = []
    for animal in ("fixture-a", "fixture-b"):
        for i in range(80):
            rows.append({"timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i),
                         "individual-local-identifier": animal,
                         "location-long": 10 + i * .001,
                         "location-lat": 40 + i * .001})
    pd.DataFrame(rows).to_csv(path, index=False)


class _FakeMLflow:
    def __init__(self):
        self.runs = []
        self.tracking_uri = None
        self.experiment_name = None

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment_name = name
        return type("Experiment", (), {"experiment_id": "offline-test"})()

    def active_run(self):
        return None

    @contextmanager
    def start_run(self, **kwargs):
        run = {"id": f"run-{len(self.runs) + 1}", "params": {}, "tags": {},
               "metrics": [], "artifacts": []}
        self.runs.append(run)
        self.current = run
        yield run

    def log_params(self, params):
        self.current["params"].update(params)

    def set_tags(self, tags):
        self.current["tags"].update(tags)

    def log_metrics(self, metrics, step=None):
        self.current["metrics"].append((dict(metrics), step))

    def log_artifact(self, path, artifact_path=None):
        self.current["artifacts"].append((Path(path).name, artifact_path))


class MlflowTrackingTests(unittest.TestCase):
    def test_sqlite_uri_resolution_and_local_only_restriction(self):
        repo_root = Path(__file__).resolve().parents[1]
        resolved = normalize_local_tracking_uri("sqlite:///mlruns/mlflow.db", base_dir=repo_root)
        self.assertEqual(resolved, "sqlite:///" + (repo_root / "mlruns" / "mlflow.db").resolve().as_posix())
        artifact_uri = local_mlflow_artifact_uri(resolved)
        self.assertTrue(artifact_uri.startswith("file:"))
        self.assertTrue(artifact_uri.endswith("/mlruns/artifacts"))
        with self.assertRaisesRegex(ValueError, "hosted tracking is disabled"):
            normalize_local_tracking_uri("https://tracking.example")
        with self.assertRaisesRegex(ValueError, "Only local SQLite"):
            normalize_local_tracking_uri("file:///tmp/mlruns")

    def test_training_logs_distinct_runs_parameters_metrics_and_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv = root / "fixture.csv"
            _write_fixture(csv)
            cfg = TrajectoryConfig(3, 2, 5, 2)
            prepared = root / "prepared"
            sequences = build_trajectory_sequences(str(csv), cfg, test_size=.15, val_size=.15)
            save_prepared_sequences(prepared, sequences, cfg, .15, .15, random_state=17)
            fake = _FakeMLflow()
            args = build_parser().parse_args([
                "--prepared-dir", str(prepared), "--input-length", "3", "--forecast-horizon", "2",
                "--min-points", "5", "--max-animals", "2", "--batch-size", "8", "--epochs", "1",
                "--lr", "0.002", "--hidden-size", "4", "--num-layers", "1", "--dropout", "0",
                "--device", "cpu", "--seed", "17", "--canonical-output", str(root / "evaluation"),
                "--mlflow-tracking-uri", str(root / "mlruns" / "mlflow.db"),
                "--mlflow-experiment-name", "fixture-experiment",
            ])
            with patch("src.models.train._load_mlflow", return_value=fake), \
                    patch("src.models.train._local_experiment", return_value=type(
                        "Experiment", (), {"experiment_id": "offline-test"})()) as get_experiment:
                run_training(args)
                run_training(args)

            self.assertEqual([run["id"] for run in fake.runs], ["run-1", "run-2"])
            self.assertEqual(get_experiment.call_count, 2)
            self.assertEqual(get_experiment.call_args.args[2], "fixture-experiment")
            self.assertEqual(fake.tracking_uri, "sqlite:///" + (root / "mlruns" / "mlflow.db").as_posix())
            run = fake.runs[0]
            for key in ("seed", "batch_size", "epochs", "learning_rate", "hidden_size", "num_layers",
                        "dropout", "input_length", "forecast_horizon", "validation_fraction",
                        "test_fraction", "model.hidden_size", "model.num_layers", "model.dropout"):
                self.assertIn(key, run["params"])
            self.assertEqual(run["params"]["seed"], 17)
            self.assertEqual(run["params"]["learning_rate"], .002)
            self.assertNotIn("mlflow_tracking_uri", run["params"])
            self.assertEqual(run["metrics"][0][1], 1)
            self.assertIn("validation_loss_mse_degrees_squared", run["metrics"][0][0])
            final_metrics = run["metrics"][-1][0]
            self.assertIn("best_validation_loss_mse_degrees_squared", final_metrics)
            self.assertIn("test_lstm_ade_km", final_metrics)
            self.assertNotIn("validation_ade_km", final_metrics)
            self.assertEqual({name for name, _ in run["artifacts"]}, {
                "trajectory_lstm.pt", "experiment_summary.json", "experiment_report.md"})
            summary = json.loads((root / "evaluation" / "experiment_summary.json").read_text())
            self.assertEqual(summary["data_kind"], "unspecified")

    def test_pipeline_passes_configured_local_tracking_settings_to_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "pipeline.json").read_text())
            config["data"].update({"prepared_dir": str(root / "prepared"), "max_animals": 2})
            config["trajectory"].update({"input_length": 3, "forecast_horizon": 2,
                                         "min_points_per_track": 5})
            config["training"].update({"evaluation_dir": str(root / "evaluation")})
            config["mlflow"].update({"tracking_uri": "sqlite:///local-runs/mlflow.db",
                                     "experiment_name": "configured-name"})
            config_path = root / "pipeline.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            captured = {}

            def capture(args):
                captured["args"] = args

            with patch("src.pipeline.run_training", side_effect=capture):
                self.assertEqual(pipeline_main(["train", "--config", str(config_path)]), 0)
            self.assertEqual(captured["args"].mlflow_experiment_name, "configured-name")
            self.assertEqual(captured["args"].mlflow_tracking_uri,
                             "sqlite:///" + (Path(__file__).resolve().parents[1] / "local-runs" / "mlflow.db").as_posix())


if __name__ == "__main__":
    unittest.main()
