"""Mocked and native offline MLflow integration checks."""
import json
import subprocess
import sys
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


def _assert_isolated_process_succeeded(test_case: unittest.TestCase, script: str, *args: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script, *args], cwd=project_root,
        capture_output=True, text=True, check=False,
    )
    test_case.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def _args_to_cli(args) -> list[str]:
    options = {
        "data_path": "--data-path", "prepared_dir": "--prepared-dir",
        "canonical_output": "--canonical-output", "diagnostics_dir": "--diagnostics-dir",
        "input_length": "--input-length", "forecast_horizon": "--forecast-horizon",
        "min_points": "--min-points", "max_animals": "--max-animals",
        "batch_size": "--batch-size", "epochs": "--epochs", "lr": "--lr",
        "hidden_size": "--hidden-size", "num_layers": "--num-layers",
        "dropout": "--dropout", "num_workers": "--num-workers", "device": "--device",
        "out_dir": "--out-dir", "seed": "--seed", "test_size": "--test-size",
        "val_size": "--val-size", "data_kind": "--data-kind",
        "mlflow_tracking_uri": "--mlflow-tracking-uri",
        "mlflow_experiment_name": "--mlflow-experiment-name",
    }
    result = []
    for name, value in vars(args).items():
        if name == "strict_determinism":
            if value:
                result.append("--strict-determinism")
        elif value is not None:
            result.extend((options[name], str(value)))
    return result


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
    def test_native_mlflow_client_run_logs_and_retrieves_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tracking_dir = root / "tracking"
            tracking_dir.mkdir()
            tracking_uri = normalize_local_tracking_uri(str(tracking_dir / "mlflow.db"))
            script = r"""
import sys
from pathlib import Path
from mlflow.tracking import MlflowClient
from src.models.train import local_mlflow_artifact_uri

root = Path(sys.argv[1])
tracking_uri = sys.argv[2]
client = MlflowClient(tracking_uri=tracking_uri)
experiment_id = client.create_experiment(
    "native-client-test", artifact_location=local_mlflow_artifact_uri(tracking_uri))
run_id = client.create_run(experiment_id).info.run_id
artifact = root / "sample.txt"
artifact.write_text("native artifact payload", encoding="utf-8")
try:
    client.log_param(run_id, "seed", "17")
    client.log_metric(run_id, "validation_loss", 0.25, step=1)
    client.log_artifact(run_id, str(artifact))
except BaseException:
    client.set_terminated(run_id, status="FAILED")
    raise
else:
    client.set_terminated(run_id, status="FINISHED")

finished = client.get_run(run_id)
assert finished.info.status == "FINISHED"
assert finished.info.experiment_id == experiment_id
assert client.get_experiment(experiment_id).name == "native-client-test"
assert finished.data.params["seed"] == "17"
assert finished.data.metrics["validation_loss"] == 0.25
downloaded = client.download_artifacts(run_id, "sample.txt", str(root / "downloads"))
assert Path(downloaded).read_text(encoding="utf-8") == "native artifact payload"
"""
            _assert_isolated_process_succeeded(self, script, str(root), tracking_uri)

    def test_native_training_logs_evaluation_and_retrievable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv = root / "fixture.csv"
            _write_fixture(csv)
            prepared = root / "prepared"
            config = TrajectoryConfig(3, 2, 5, 2)
            sequences = build_trajectory_sequences(str(csv), config, test_size=.15, val_size=.15)
            save_prepared_sequences(prepared, sequences, config, .15, .15, random_state=17)
            database = root / "tracking" / "mlflow.db"
            database.parent.mkdir()
            tracking_uri = normalize_local_tracking_uri(str(database))
            args = build_parser().parse_args([
                "--prepared-dir", str(prepared), "--input-length", "3", "--forecast-horizon", "2",
                "--min-points", "5", "--max-animals", "2", "--batch-size", "16", "--epochs", "1",
                "--lr", "0.002", "--hidden-size", "4", "--num-layers", "1", "--dropout", "0",
                "--device", "cpu", "--seed", "17", "--data-kind", "synthetic",
                "--canonical-output", str(root / "evaluation"), "--mlflow-tracking-uri", str(database),
                "--mlflow-experiment-name", "native-training-test",
            ])
            script = r"""
import json
import sys
from pathlib import Path
from mlflow.tracking import MlflowClient
from src.models.train import build_parser, run_training

arguments = json.loads(sys.argv[1])
tracking_uri = sys.argv[2]
run_training(build_parser().parse_args(arguments))
client = MlflowClient(tracking_uri=tracking_uri)
experiment = client.get_experiment_by_name("native-training-test")
assert experiment is not None
runs = client.search_runs([experiment.experiment_id])
assert len(runs) == 1
run = runs[0]
assert run.info.status == "FINISHED"
assert run.info.experiment_id == experiment.experiment_id
assert run.data.params["seed"] == "17"
assert "validation_loss_mse_degrees_squared" in run.data.metrics
assert "test_lstm_ade_km" in run.data.metrics
assert "test_persistence_ade_km" in run.data.metrics
artifacts = client.list_artifacts(run.info.run_id, "model")
assert "trajectory_lstm.pt" in {item.path.rsplit("/", 1)[-1] for item in artifacts}
checkpoint = client.download_artifacts(
    run.info.run_id, "model/trajectory_lstm.pt", str(Path(arguments[arguments.index("--canonical-output") + 1]).parent / "downloads"))
assert Path(checkpoint).stat().st_size > 0
"""
            _assert_isolated_process_succeeded(self, script, json.dumps(_args_to_cli(args)), tracking_uri)

    def test_native_training_exception_marks_mlflow_run_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = root / "prepared"
            prepared.mkdir()
            database = root / "tracking" / "mlflow.db"
            database.parent.mkdir()
            tracking_uri = normalize_local_tracking_uri(str(database))
            args = build_parser().parse_args([
                "--prepared-dir", str(prepared), "--device", "cpu", "--seed", "17",
                "--mlflow-tracking-uri", str(database), "--mlflow-experiment-name", "native-failure-test",
            ])
            script = r"""
import json
import sys
from unittest.mock import patch
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from src.models.train import build_parser, run_training

arguments = json.loads(sys.argv[1])
tracking_uri = sys.argv[2]
with patch("src.models.train._run_training", side_effect=RuntimeError("controlled failure")):
    try:
        run_training(build_parser().parse_args(arguments))
    except RuntimeError as exc:
        assert str(exc) == "controlled failure"
    else:
        raise AssertionError("injected training exception was swallowed")
client = MlflowClient(tracking_uri=tracking_uri)
experiment = client.get_experiment_by_name("native-failure-test")
assert experiment is not None
runs = client.search_runs([experiment.experiment_id], run_view_type=ViewType.ALL)
assert len(runs) == 1
assert runs[0].info.status == "FAILED"
assert runs[0].data.params["seed"] == "17"
"""
            _assert_isolated_process_succeeded(self, script, json.dumps(_args_to_cli(args)), tracking_uri)

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
