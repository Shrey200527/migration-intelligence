"""Structural checks for the local DVC graph (JSON is a YAML 1.2 subset)."""
import json
import shlex
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
            "src/data/validation.py", "src/data/data_prep.py", "requirements.txt"})
        self.assertEqual(deps["prepare"], {"${data.raw_path}", "${data.validation_report}",
            "src/pipeline.py", "src/data/validation.py", "src/data/data_prep.py", "requirements.txt"})
        self.assertEqual(deps["train"], {"${data.prepared_dir}", "src/pipeline.py",
            "src/data/data_prep.py", "src/data/validation.py", "src/models/train.py",
            "src/models/model.py", "src/models/baseline.py", "src/monitoring/metrics.py",
            "requirements.txt"})

    def test_declared_outputs_match_canonical_configuration(self):
        data, training = self.config["data"], self.config["training"]
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
