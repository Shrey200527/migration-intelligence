"""Thin local stage entry points for validation, preparation, and training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline.json"
sys.path.insert(0, str(ROOT))

from src.data.data_prep import (TrajectoryConfig, build_trajectory_sequences,
                                save_prepared_sequences)
from src.data.validation import validate_dataset, write_quality_report
from src.models.train import (build_parser as build_training_parser,
                              normalize_local_tracking_uri, run_training,
                              validate_training_parameters)

def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256_file(path: Path) -> str:
    """Return a streaming content fingerprint without loading GPS data into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict, Path]:
    config_candidate = Path(path)
    config_path = config_candidate if config_candidate.is_absolute() else ROOT / config_candidate
    config_path = config_path.resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read pipeline configuration {config_path}: {exc}") from exc
    try:
        if not isinstance(config, dict): raise ValueError("configuration root must be an object")
        data, traj, training, mlflow = (config["data"], config["trajectory"],
                                        config["training"], config["mlflow"])
        if not all(isinstance(section, dict) for section in (data, traj, training, mlflow)):
            raise ValueError("data, trajectory, training, and mlflow sections must be objects")
        if not isinstance(mlflow.get("tracking_uri"), str) or not mlflow["tracking_uri"]:
            raise ValueError("mlflow.tracking_uri must be a non-empty local path or file URI")
        if not isinstance(mlflow.get("experiment_name"), str) or not mlflow["experiment_name"].strip():
            raise ValueError("mlflow.experiment_name must be a non-empty string")
        try:
            normalize_local_tracking_uri(mlflow["tracking_uri"], base_dir=ROOT)
        except ValueError as exc:
            raise ValueError(f"mlflow.tracking_uri is invalid: {exc}") from exc
        if not isinstance(data.get("raw_path"), str) or not data["raw_path"]:
            raise ValueError("data.raw_path must be a non-empty path string")
        for key in ("validation_report", "validation_summary", "prepared_dir"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError(f"data.{key} must be a non-empty path string")
        for key in ("evaluation_dir", "diagnostics_dir"):
            if not isinstance(training.get(key), str) or not training[key]:
                raise ValueError(f"training.{key} must be a non-empty path string")
        for key in ("input_length", "forecast_horizon", "min_points_per_track"):
            if type(traj.get(key)) is not int:
                raise ValueError(f"trajectory.{key} must be an integer")
        max_animals = data.get("max_animals")
        if max_animals is not None and (type(max_animals) is not int or max_animals < 1):
            raise ValueError("data.max_animals must be a positive integer or null")
        for key in ("val_size", "test_size"):
            value = traj.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"trajectory.{key} must be a number")
        cfg = TrajectoryConfig(traj["input_length"], traj["forecast_horizon"],
                               traj["min_points_per_track"], max_animals)
        from src.data.data_prep import _validate_config, _validate_split_sizes
        _validate_config(cfg)
        try:
            _validate_split_sizes(traj["test_size"], traj["val_size"])
        except ValueError as exc:
            raise ValueError(f"trajectory.val_size and trajectory.test_size are invalid: {exc}") from exc
        validate_training_parameters(training)
        if type(training.get("strict_determinism")) is not bool:
            raise ValueError("training.strict_determinism must be a boolean")
        if training["data_kind"] not in ("unspecified", "real", "synthetic"):
            raise ValueError("training.data_kind must be unspecified, real, or synthetic")
        if not isinstance(training.get("device"), str) or not training["device"]:
            raise ValueError("training.device must be a non-empty device string")
        try:
            import torch
            torch.device(training["device"])
        except (ImportError, RuntimeError, ValueError) as exc:
            raise ValueError(f"training.device is invalid: {training.get('device')}") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pipeline configuration: {exc}") from exc
    return config, ROOT


def _stage_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible Migration Intelligence pipeline stage")
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("validate", "prepare", "train"):
        command = sub.add_parser(stage)
        command.add_argument("--config", default=str(DEFAULT_CONFIG))
        if stage == "validate":
            command.add_argument("--data-path")
            command.add_argument("--output")
            command.add_argument("--summary-output")
        elif stage == "prepare":
            command.add_argument("--data-path")
            command.add_argument("--output-dir")
        else:
            command.add_argument("--prepared-dir")
            command.add_argument("--output-dir")
            command.add_argument("--diagnostics-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _stage_parser()
    args = parser.parse_args(argv)
    try:
        config, root = load_config(args.config)
        data, traj, training, mlflow = (config["data"], config["trajectory"],
                                        config["training"], config["mlflow"])
        cfg = TrajectoryConfig(traj["input_length"], traj["forecast_horizon"],
                               traj["min_points_per_track"], data["max_animals"])
        data_path = _path(root, getattr(args, "data_path", None) or data["raw_path"])
        if args.stage in ("validate", "prepare") and not data_path.is_file():
            raise FileNotFoundError(f"Raw dataset CSV not found: {data_path}")
        if args.stage == "validate":
            output = _path(root, args.output or data["validation_report"])
            summary = _path(root, args.summary_output or data["validation_summary"])
            report = validate_dataset(data_path, cfg, val_size=traj["val_size"],
                                      test_size=traj["test_size"], max_animals=cfg.max_animals)
            report["source_fingerprint"] = {"algorithm": "sha256", "digest": _sha256_file(data_path)}
            write_quality_report(report, output, summary)
            print(json.dumps(report, indent=2))
            return 0 if report["usable"] else 2
        if args.stage == "prepare":
            output = _path(root, args.output_dir or data["prepared_dir"])
            report_path = _path(root, data["validation_report"])
            expected_config = {**cfg.__dict__, "max_animals": cfg.max_animals}
            expected_split = {"val_size": traj["val_size"], "test_size": traj["test_size"]}
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Could not read validation report {report_path}: {exc}") from exc
                if not isinstance(report, dict):
                    raise ValueError(f"Validation report {report_path} must contain a JSON object; rerun validate")
                expected_fingerprint = {"algorithm": "sha256", "digest": _sha256_file(data_path)}
                if (report.get("path_name") != data_path.name or report.get("config") != expected_config
                        or report.get("split") != expected_split
                        or report.get("source_fingerprint") != expected_fingerprint):
                    raise ValueError("Validation report is stale or does not match the configured dataset contents, trajectory settings, and split; rerun validate")
            else:
                report = validate_dataset(data_path, cfg, val_size=traj["val_size"],
                                          test_size=traj["test_size"], max_animals=cfg.max_animals)
                report["source_fingerprint"] = {"algorithm": "sha256", "digest": _sha256_file(data_path)}
            if not report.get("usable"):
                raise ValueError(f"Dataset is unusable: {report.get('error')}")
            sequences = build_trajectory_sequences(str(data_path), cfg,
                                                   test_size=traj["test_size"], val_size=traj["val_size"],
                                                   random_state=training["seed"])
            current_fingerprint = {"algorithm": "sha256", "digest": _sha256_file(data_path)}
            if current_fingerprint != report.get("source_fingerprint"):
                raise ValueError("Raw CSV changed while preparation was running; rerun validate and prepare")
            manifest = save_prepared_sequences(output, sequences, cfg, traj["test_size"],
                                               traj["val_size"], training["seed"],
                                               source_fingerprint=current_fingerprint,
                                               quality_report=report)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        prepared_dir = _path(root, args.prepared_dir or data["prepared_dir"])
        train_parser = build_training_parser()
        train_args = train_parser.parse_args([
            "--prepared-dir", str(prepared_dir), "--input-length", str(cfg.input_length),
            "--forecast-horizon", str(cfg.forecast_horizon), "--min-points", str(cfg.min_points_per_track),
            *( ["--max-animals", str(cfg.max_animals)] if cfg.max_animals is not None else [] ),
            "--batch-size", str(training["batch_size"]),
            "--epochs", str(training["epochs"]), "--lr", str(training["learning_rate"]),
            "--hidden-size", str(training["hidden_size"]), "--num-layers", str(training["num_layers"]),
            "--dropout", str(training["dropout"]), "--num-workers", str(training["num_workers"]),
            "--device", str(training["device"]), "--seed", str(training["seed"]),
            "--val-size", str(traj["val_size"]), "--test-size", str(traj["test_size"]),
            "--data-kind", str(training["data_kind"]), "--canonical-output",
            str(_path(root, args.output_dir or training["evaluation_dir"])),
            "--mlflow-tracking-uri", normalize_local_tracking_uri(
                mlflow["tracking_uri"], base_dir=root),
            "--mlflow-experiment-name", mlflow["experiment_name"],
            "--diagnostics-dir", str(_path(root, args.diagnostics_dir or training["diagnostics_dir"])),
            *( ["--strict-determinism"] if training["strict_determinism"] else [] )])
        if cfg.max_animals is None:
            train_args.max_animals = None
        run_training(train_args)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Pipeline {args.stage} failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
