"""Structural checks for the local DVC graph (JSON is a YAML 1.2 subset)."""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.pipeline import _stage_parser

ROOT = Path(__file__).resolve().parents[1]


def _params(stage):
    return set(stage["params"][0]["configs/pipeline.json"])


class DvcStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = json.loads((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
        cls.config = json.loads((ROOT / "configs/pipeline.json").read_text(encoding="utf-8"))

    def test_three_stages_and_cli_commands_are_wired(self):
        stages = self.graph["stages"]
        self.assertEqual(set(stages), {"validate", "prepare", "train"})
        for name in stages:
            command = shlex.split(stages[name]["cmd"])
            self.assertEqual(command[:2], ["python", "src/pipeline.py"])
            parsed = _stage_parser().parse_args(command[2:])
            self.assertEqual(parsed.stage, name)
            self.assertEqual(parsed.config, "configs/pipeline.json")
        self.assertIn("${data.validation_report}", stages["prepare"]["deps"])
        self.assertIn("${data.prepared_dir}", stages["train"]["deps"])

    def test_source_dependencies_are_complete_for_each_stage(self):
        deps = {name: set(stage["deps"]) for name, stage in self.graph["stages"].items()}
        self.assertEqual(deps["validate"], {"${data.raw_path}", "src/pipeline.py",
            "src/data/validation.py", "src/data/data_prep.py", "src/models/train.py", "requirements.txt"})
        self.assertEqual(deps["prepare"], {"${data.raw_path}", "${data.validation_report}",
            "src/pipeline.py", "src/data/validation.py", "src/data/data_prep.py", "src/models/train.py", "requirements.txt"})
        self.assertEqual(deps["train"], {"${data.prepared_dir}", "src/pipeline.py",
            "src/data/data_prep.py", "src/data/validation.py", "src/models/train.py",
            "src/models/model.py", "src/models/baseline.py", "src/monitoring/metrics.py",
            "requirements.txt"})

    def test_declared_outputs_match_canonical_configuration(self):
        training = self.config["training"]
        self.assertEqual(self.graph["stages"]["validate"]["outs"],
                         ["${data.validation_report}", "${data.validation_summary}"])
        self.assertEqual(self.graph["stages"]["prepare"]["outs"], ["${data.prepared_dir}"])
        self.assertEqual(self.graph["stages"]["train"]["outs"], [
            "${training.evaluation_dir}/trajectory_lstm.pt",
            "${training.evaluation_dir}/experiment_summary.json",
            "${training.evaluation_dir}/experiment_report.md",
        ])
        self.assertNotIn(training["diagnostics_dir"], str(self.graph["stages"]["train"]["outs"]))

    def test_dependencies_resolve_and_parameters_are_declared(self):
        for stage in self.graph["stages"].values():
            for dep in stage["deps"]:
                if dep.startswith("${"):
                    continue
                self.assertTrue((ROOT / dep).exists(), dep)
            self.assertTrue(stage["params"])
        self.assertEqual(_params(self.graph["stages"]["validate"]), {
            "data.raw_path", "data.max_animals", "data.validation_report", "data.validation_summary",
            "trajectory.input_length", "trajectory.forecast_horizon", "trajectory.min_points_per_track",
            "trajectory.val_size", "trajectory.test_size"})
        self.assertEqual(_params(self.graph["stages"]["prepare"]), {
            "data.raw_path", "data.max_animals", "data.validation_report", "data.prepared_dir",
            "trajectory.input_length", "trajectory.forecast_horizon", "trajectory.min_points_per_track",
            "trajectory.val_size", "trajectory.test_size", "training.seed"})
        self.assertEqual(_params(self.graph["stages"]["train"]), {
            "data.prepared_dir", "data.max_animals", "trajectory.input_length", "trajectory.forecast_horizon",
            "trajectory.min_points_per_track", "trajectory.val_size", "trajectory.test_size",
            "training.seed", "training.batch_size", "training.epochs", "training.learning_rate",
            "training.hidden_size", "training.num_layers", "training.dropout", "training.num_workers",
            "training.device", "training.data_kind", "training.strict_determinism",
            "training.evaluation_dir", "training.diagnostics_dir",
            "mlflow.tracking_uri", "mlflow.experiment_name"})

    def test_graph_has_only_expected_json_yaml_fields(self):
        self.assertEqual(set(self.graph), {"vars", "stages"})
        self.assertEqual(self.graph["vars"], ["configs/pipeline.json"])
        allowed_stage_fields = {"cmd", "deps", "params", "outs"}
        for stage in self.graph["stages"].values():
            self.assertEqual(set(stage), allowed_stage_fields)
            self.assertIsInstance(stage["cmd"], str)
            self.assertTrue(all(isinstance(item, str) for item in stage["deps"] + stage["outs"]))

    @unittest.skipUnless(__import__("importlib").util.find_spec("dvc"), "DVC is not installed")
    def test_isolated_dvc_reproduction_and_input_invalidation(self):
        with tempfile.TemporaryDirectory(prefix="migration-dvc-integration-") as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            shutil.copy2(ROOT / "dvc.yaml", workspace / "dvc.yaml")
            shutil.copy2(ROOT / "requirements.txt", workspace / "requirements.txt")
            shutil.copytree(ROOT / "src", workspace / "src",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            shutil.copytree(ROOT / "configs", workspace / "configs")
            data_dir = workspace / "data" / "raw"
            data_dir.mkdir(parents=True)
            config = json.loads((workspace / "configs" / "pipeline.json").read_text(encoding="utf-8"))
            config["data"].update({"raw_path": "data/raw/synthetic.csv", "max_animals": 2,
                "validation_report": "artifacts/dataset_quality.json",
                "validation_summary": "artifacts/dataset_quality.md",
                "prepared_dir": "data/processed/prepared"})
            config["trajectory"].update({"input_length": 3, "forecast_horizon": 2,
                "min_points_per_track": 5, "val_size": .15, "test_size": .15})
            config["training"].update({"seed": 17, "batch_size": 32, "epochs": 1,
                "hidden_size": 4, "num_layers": 1, "dropout": 0.0, "num_workers": 0,
                "device": "cpu", "evaluation_dir": "artifacts/pipeline/evaluation",
                "diagnostics_dir": "artifacts/pipeline/diagnostics"})
            config["mlflow"].update({"tracking_uri": "sqlite:///mlruns/mlflow.db",
                                     "experiment_name": "isolated-dvc-test"})
            (workspace / "configs" / "pipeline.json").write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8")

            def write_csv(extra=False):
                with (data_dir / "synthetic.csv").open("w", encoding="utf-8", newline="") as stream:
                    stream.write("timestamp,individual-local-identifier,location-long,location-lat\n")
                    for animal in ("fixture-1", "fixture-2"):
                        for index in range(40 + int(extra)):
                            stream.write(f"2025-01-{1 + index // 24:02d}T{index % 24:02d}:00:00,{animal},{10 + index * .001},{40 + index * .001}\n")

            write_csv()
            env = os.environ.copy()
            env["DVC_GLOBAL_CONFIG_DIR"] = str(workspace / ".dvc-global")
            env["DVC_SITE_CACHE_DIR"] = str(workspace / ".dvc-site")
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")

            def dvc(*arguments):
                result = subprocess.run([sys.executable, "-m", "dvc", *arguments], cwd=workspace,
                                        env=env, text=True, capture_output=True, timeout=180)
                self.assertEqual(result.returncode, 0,
                                 f"dvc {' '.join(arguments)} failed:\n{result.stdout}\n{result.stderr}")
                return result.stdout + result.stderr

            dvc("init", "--no-scm", "--quiet")
            dvc("dag")
            first = dvc("repro")
            for stage in ("validate", "prepare", "train"):
                self.assertIn(stage, first)
            self.assertTrue((workspace / "data/processed/prepared/manifest.json").is_file())
            self.assertTrue((workspace / "artifacts/pipeline/evaluation/trajectory_lstm.pt").is_file())
            self.assertIn("up to date", dvc("repro").lower())

            config["training"]["epochs"] = 2
            (workspace / "configs" / "pipeline.json").write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8")
            parameter_change = dvc("repro")
            self.assertIn("train", parameter_change)

            with (workspace / "src/models/train.py").open("a", encoding="utf-8") as stream:
                stream.write("\n# isolated dependency invalidation check\n")
            source_change = dvc("repro")
            for stage in ("validate", "prepare", "train"):
                self.assertIn(stage, source_change)

            write_csv(extra=True)
            csv_change = dvc("repro")
            for stage in ("validate", "prepare", "train"):
                self.assertIn(stage, csv_change)
            self.assertTrue((workspace / "data/processed/prepared/manifest.json").is_file())

    def test_local_only_configuration_and_narrow_pointer_ignores(self):
        config = (ROOT / ".dvc/config").read_text(encoding="utf-8").lower()
        self.assertIn("analytics = false", config)
        self.assertIn("check_update = false", config)
        self.assertIn("autostage = false", config)
        self.assertNotIn("remote", config)
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("!data/raw/*.dvc", gitignore)
        self.assertIn("!data/processed/*.dvc", gitignore)
        self.assertIn("artifacts/pipeline/**", gitignore)
        self.assertIn("!artifacts/pipeline/**/", gitignore)
        self.assertIn("!artifacts/pipeline/**/*.dvc", gitignore)
        self.assertIn("*.pt", gitignore)
        self.assertIn("data/raw/*", gitignore)
        self.assertIn("data/processed/*", gitignore)
        local_rules = (ROOT / ".dvc/.gitignore").read_text(encoding="utf-8")
        self.assertIn("/cache", local_rules)
        self.assertIn("/tmp", local_rules)


if __name__ == "__main__":
    unittest.main()
