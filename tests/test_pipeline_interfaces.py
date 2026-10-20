"""Tests for reusable, pickle-free pipeline artifacts and stage interfaces."""
import json
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_prep import (TrajectoryConfig, build_trajectory_sequences,
                                load_prepared_sequences, save_prepared_sequences)
from src.pipeline import load_config, main as pipeline_main
from src.models.train import (_validate_args, build_parser, main as train_main,
                              run_training)


def write_fixture(path: Path, observations: int = 120) -> None:
    rows = []
    for animal in ("A", "B"):
        for i in range(observations):
            rows.append({"timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i),
                         "individual-local-identifier": animal, "location-long": 10 + i * .01,
                         "location-lat": 40 + i * .01})
    pd.DataFrame(rows).to_csv(path, index=False)


class PipelineInterfaceTests(unittest.TestCase):
    def test_standalone_training_argument_validation_boundaries_and_invalid_values(self):
        valid = build_parser().parse_args([])
        valid.seed = 0
        valid.batch_size = 1
        valid.epochs = 1
        valid.lr = 1e-12
        valid.hidden_size = 1
        valid.num_layers = 1
        valid.dropout = 0.0
        valid.num_workers = 0
        valid.input_length = 1
        valid.forecast_horizon = 1
        valid.min_points = 2
        valid.max_animals = 1
        _validate_args(valid)

        invalid_values = [
            ("seed", -1, "--seed must be non-negative"),
            ("lr", 0.0, "--lr must be a finite positive number"),
            ("lr", -0.1, "--lr must be a finite positive number"),
            ("lr", float("nan"), "--lr must be a finite positive number"),
            ("lr", float("inf"), "--lr must be a finite positive number"),
            ("lr", float("-inf"), "--lr must be a finite positive number"),
            ("epochs", 0, "--epochs must be positive"),
            ("batch_size", 0, "--batch-size must be positive"),
            ("hidden_size", 0, "--hidden-size must be positive"),
            ("num_layers", 0, "--num-layers must be positive"),
            ("input_length", 0, "input_length must be at least 1"),
            ("max_animals", 0, "max_animals must be positive or None"),
            ("val_size", float("nan"), "val_size and test_size must be positive"),
            ("dropout", -0.01, "--dropout must be finite and in \[0, 1\)"),
            ("dropout", 1.0, "--dropout must be finite and in \[0, 1\)"),
            ("dropout", float("nan"), "--dropout must be finite and in \[0, 1\)"),
            ("dropout", float("inf"), "--dropout must be finite and in \[0, 1\)"),
            ("num_workers", -1, "--num-workers cannot be negative"),
        ]
        for field, value, message in invalid_values:
            with self.subTest(field=field, value=value):
                args = build_parser().parse_args([])
                setattr(args, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    _validate_args(args)

        for option, value, message in (("--seed", "-1", "--seed must be non-negative"),
                                       ("--lr", "inf", "--lr must be a finite positive number")):
            with self.subTest(cli_option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    train_main(["--data-path", "unused.csv", option, value])
                self.assertEqual(raised.exception.code, 2, message)

    def test_artifacts_round_trip_and_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); csv = root / "gps.csv"; write_fixture(csv)
            cfg = TrajectoryConfig(3, 2, 5, 2)
            splits = build_trajectory_sequences(str(csv), cfg, .15, .15)
            first = root / "prepared1"; second = root / "prepared2"
            manifest = save_prepared_sequences(first, splits, cfg, .15, .15)
            save_prepared_sequences(second, splits, cfg, .15, .15)
            self.assertEqual(manifest["splits"]["train"]["feature_order"], ["latitude", "longitude"])
            for name in ("train.npz", "validation.npz", "test.npz", "manifest.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            loaded, loaded_manifest = load_prepared_sequences(first, cfg)
            self.assertEqual([len(part) for part in loaded], [len(part) for part in splits])
            self.assertEqual(loaded_manifest["format_version"], 1)
            self.assertEqual(loaded[0][0][0].dtype, np.float32)
            self.assertNotIn('"A"', json.dumps(loaded_manifest))
            self.assertLess(max(seq[1][-1, 0] for seq in loaded[0]), min(seq[0][0, 0] for seq in loaded[1]))
            self.assertLess(max(seq[1][-1, 0] for seq in loaded[1]), min(seq[0][0, 0] for seq in loaded[2]))

    def test_loader_rejects_missing_corrupt_and_inconsistent_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(FileNotFoundError, "manifest"):
                load_prepared_sequences(root)
            csv = root / "gps.csv"; write_fixture(csv)
            cfg = TrajectoryConfig(3, 2, 5, 2)
            output = root / "prepared"
            save_prepared_sequences(output, build_trajectory_sequences(str(csv), cfg), cfg, .15, .15)
            (output / "test.npz").unlink()
            with self.assertRaisesRegex(ValueError, "test artifact"):
                load_prepared_sequences(output)
            save_prepared_sequences(output, build_trajectory_sequences(str(csv), cfg), cfg, .15, .15)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["splits"]["train"]["sequence_count"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest metadata"):
                load_prepared_sequences(output)
            save_prepared_sequences(output, build_trajectory_sequences(str(csv), cfg), cfg, .15, .15)
            archive_path = output / "validation.npz"
            archive_path.write_bytes(archive_path.read_bytes()[:48])
            with self.assertRaisesRegex(ValueError, "Invalid prepared data"):
                load_prepared_sequences(output)
            save_prepared_sequences(output, build_trajectory_sequences(str(csv), cfg), cfg, .15, .15)
            with self.assertRaisesRegex(ValueError, "test_size"):
                load_prepared_sequences(output, cfg, expected_test_size=.2, expected_val_size=.15)

    def test_training_uses_prepared_arrays_after_raw_data_is_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); csv = root / "gps.csv"; write_fixture(csv)
            cfg = TrajectoryConfig(3, 2, 5, 2)
            prepared = root / "prepared"
            save_prepared_sequences(prepared, build_trajectory_sequences(str(csv), cfg), cfg, .15, .15)
            csv.unlink()
            args = build_parser().parse_args(["--prepared-dir", str(prepared), "--input-length", "3",
                "--forecast-horizon", "2", "--min-points", "5", "--max-animals", "2",
                "--batch-size", "8", "--epochs", "1", "--hidden-size", "4", "--num-layers", "1",
                "--dropout", "0", "--device", "cpu", "--canonical-output", str(root / "evaluation")])
            with patch("src.models.train._load_mlflow", return_value=None):
                result = run_training(args)
            self.assertTrue((result / "trajectory_lstm.pt").is_file())
            summary = json.loads((result / "experiment_summary.json").read_text(encoding="utf-8"))
            self.assertIn("persistence_ade_km", summary["test_metrics"])
            self.assertNotIn("total_training_seconds", summary)
            self.assertNotIn("elapsed_seconds", summary["training_validation_history"][0])
            self.assertTrue((root / "evaluation_diagnostics" / "runtime_diagnostics.json").is_file())
            self.assertFalse((result / "dataset_quality.json").exists())
            first_checkpoint = (result / "trajectory_lstm.pt").read_bytes()
            first_summary = (result / "experiment_summary.json").read_bytes()
            first_report = (result / "experiment_report.md").read_bytes()
            with patch("src.models.train._load_mlflow", return_value=None):
                run_training(args)
            self.assertEqual(first_checkpoint, (result / "trajectory_lstm.pt").read_bytes())
            self.assertEqual(first_summary, (result / "experiment_summary.json").read_bytes())
            self.assertEqual(first_report, (result / "experiment_report.md").read_bytes())

    def test_config_validation_and_missing_input_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "pipeline.json").read_text())
            config["trajectory"]["val_size"] = .9
            bad = root / "bad.json"; bad.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sum to less"):
                load_config(bad)
            config["trajectory"]["val_size"] = .15
            config["trajectory"]["input_length"] = 0
            bad.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input_length"):
                load_config(bad)
            config["trajectory"]["input_length"] = 20
            invalid_settings = [
                ("data", "max_animals", 0, "data.max_animals"),
                ("trajectory", "forecast_horizon", 0, "forecast_horizon"),
                ("training", "batch_size", 0, "training.batch_size"),
                ("training", "seed", -1, "training.seed"),
                ("training", "learning_rate", float("inf"), "training.learning_rate"),
                ("training", "dropout", 1, "training.dropout"),
                ("training", "strict_determinism", "false", "training.strict_determinism"),
            ]
            for section, key, value, error in invalid_settings:
                original = config[section][key]
                config[section][key] = value
                bad.write_text(json.dumps(config), encoding="utf-8")
                with self.subTest(setting=error), self.assertRaisesRegex(ValueError, error):
                    load_config(bad)
                config[section][key] = original
            config["data"]["raw_path"] = str(root / "missing.csv")
            good = root / "config.json"; good.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(pipeline_main(["validate", "--config", str(good)]), 2)
            self.assertEqual(pipeline_main(["prepare", "--config", str(good)]), 2)

    def test_pipeline_train_stage_does_not_require_raw_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); csv = root / "gps.csv"; write_fixture(csv)
            cfg = TrajectoryConfig(3, 2, 5, 2); prepared = root / "prepared"
            save_prepared_sequences(prepared, build_trajectory_sequences(str(csv), cfg), cfg, .15, .15)
            config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "pipeline.json").read_text())
            config["data"].update({"raw_path": str(root / "missing.csv"), "max_animals": 2,
                                   "prepared_dir": str(prepared)})
            config["trajectory"].update({"input_length": 3, "forecast_horizon": 2,
                                         "min_points_per_track": 5})
            config["training"].update({"epochs": 1, "batch_size": 8, "hidden_size": 4,
                                       "num_layers": 1, "dropout": 0,
                                       "evaluation_dir": str(root / "evaluation")})
            config_path = root / "config.json"; config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch("src.models.train._load_mlflow", return_value=None):
                self.assertEqual(pipeline_main(["train", "--config", str(config_path)]), 0)
            self.assertTrue((root / "evaluation" / "experiment_summary.json").is_file())

    def test_prepare_reuses_matching_validation_stage_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); csv = root / "gps.csv"; write_fixture(csv)
            config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "pipeline.json").read_text())
            config["data"].update({"raw_path": str(csv), "max_animals": 2,
                "validation_report": str(root / "quality.json"),
                "validation_summary": str(root / "quality.md"),
                "prepared_dir": str(root / "prepared")})
            config["trajectory"].update({"input_length": 3, "forecast_horizon": 2,
                "min_points_per_track": 5, "val_size": .15, "test_size": .15})
            path = root / "pipeline.json"; path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(pipeline_main(["validate", "--config", str(path)]), 0)
            with patch("src.pipeline.validate_dataset", side_effect=AssertionError("revalidated")):
                self.assertEqual(pipeline_main(["prepare", "--config", str(path)]), 0)
            self.assertTrue((root / "prepared" / "train.npz").is_file())
            self.assertTrue((root / "prepared" / "dataset_quality.md").is_file())

    def test_prepare_rejects_validation_report_for_changed_csv_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); csv = root / "gps.csv"; write_fixture(csv)
            config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "pipeline.json").read_text())
            config["data"].update({"raw_path": str(csv), "max_animals": 2,
                "validation_report": str(root / "quality.json"),
                "validation_summary": str(root / "quality.md"),
                "prepared_dir": str(root / "prepared")})
            config["trajectory"].update({"input_length": 3, "forecast_horizon": 2,
                "min_points_per_track": 5, "val_size": .15, "test_size": .15})
            path = root / "pipeline.json"; path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(pipeline_main(["validate", "--config", str(path)]), 0)
            report_text = (root / "quality.json").read_text(encoding="utf-8")
            original_report = json.loads(report_text)
            self.assertEqual(original_report["source_fingerprint"]["algorithm"], "sha256")
            self.assertNotIn('"A"', report_text)
            self.assertNotIn("2025-01-01T00:00:00,A", report_text)
            with csv.open("a", encoding="utf-8") as stream:
                stream.write("2025-01-06T00:00:00,A,12.5,42.5\n")
            error = io.StringIO()
            with redirect_stderr(error):
                self.assertEqual(pipeline_main(["prepare", "--config", str(path)]), 2)
            self.assertIn("stale", error.getvalue())
            self.assertFalse((root / "prepared" / "manifest.json").exists())

    def test_legacy_raw_csv_training_cli_still_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); csv = root / "gps.csv"; write_fixture(csv)
            with patch("src.models.train._load_mlflow", return_value=None):
                code = train_main(["--data-path", str(csv), "--input-length", "3",
                "--forecast-horizon", "2", "--min-points", "5", "--max-animals", "2",
                "--batch-size", "8", "--epochs", "1", "--hidden-size", "4", "--num-layers", "1",
                "--dropout", "0", "--device", "cpu", "--out-dir", str(root / "runs"),
                "--data-kind", "synthetic"])
            self.assertEqual(code, 0)
            self.assertEqual(len(list((root / "runs").glob("run_*/experiment_summary.json"))), 1)

    def test_script_entrypoint_resolves_paths_outside_repository_cwd(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); csv = root / "gps.csv"; write_fixture(csv, observations=300)
            output = root / "quality.json"
            script = Path(__file__).resolve().parents[1] / "src" / "pipeline.py"
            result = subprocess.run([sys.executable, str(script), "validate", "--config",
                                    "configs/pipeline.json", "--data-path", str(csv),
                                    "--output", str(output)], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
