"""Reproducible CPU/GPU training and fair persistence-baseline evaluation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import io
import json
import math
import os
import platform
import random
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.data.data_prep import (TrajectoryConfig, TrajectoryDataset, build_trajectory_sequences,
                                load_prepared_sequences)
from src.data.validation import validate_dataset, write_quality_report
from src.models.baseline import persistence_forecast
from src.models.model import build_model
from src.monitoring.metrics import haversine_distance_km, validate_coordinate_tensor


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without changing process-wide backend policy."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def normalize_local_tracking_uri(value: str, base_dir: str | Path | None = None) -> str:
    """Normalize a local SQLite URI/path and reject non-SQLite or hosted stores."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("MLflow tracking URI must be a non-empty local SQLite URI or database path")
    value = value.strip()
    if value.startswith("sqlite:///"):
        parsed = urlparse(value)
        if parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("MLflow SQLite tracking URI must identify a local database path only")
        raw_path = value[len("sqlite:///"):]
        if not raw_path:
            raise ValueError("MLflow SQLite tracking URI must include a database path")
        path = Path(raw_path).expanduser()
        if os.name == "nt" and len(raw_path) >= 3 and raw_path[1:3] == ":/":
            path = Path(raw_path)
    else:
        if "://" in value:
            raise ValueError("Only local SQLite MLflow tracking URIs are supported; hosted tracking is disabled")
        path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return "sqlite:///" + path.resolve().as_posix()


def local_mlflow_artifact_uri(tracking_uri: str) -> str:
    """Use a local artifact directory beside the configured SQLite database."""
    if not tracking_uri.startswith("sqlite:///"):
        raise ValueError("MLflow tracking URI must use a local SQLite database")
    raw_path = tracking_uri[len("sqlite:///"):]
    database_path = Path(raw_path)
    if os.name == "nt" and len(raw_path) >= 3 and raw_path[1:3] == ":/":
        database_path = Path(raw_path)
    return (database_path.resolve().parent / "artifacts").as_uri()


def _local_experiment(mlflow, tracking_uri: str, experiment_name: str):
    """Get or create an experiment with an explicit local artifact root."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(
            experiment_name,
            artifact_location=local_mlflow_artifact_uri(tracking_uri),
        )
        experiment = client.get_experiment_by_name(experiment_name)
    artifact_uri = urlparse(experiment.artifact_location) if experiment is not None else None
    if (experiment is None or artifact_uri.scheme != "file"
            or artifact_uri.netloc not in ("", "localhost")):
        raise ValueError("MLflow experiment artifacts must use a local file store")
    return experiment


def _load_mlflow():
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def _mlflow_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_workers": args.num_workers,
        "device": args.device,
        "strict_determinism": args.strict_determinism,
        "input_length": args.input_length,
        "forecast_horizon": args.forecast_horizon,
        "min_points_per_track": args.min_points,
        "max_animals": args.max_animals,
        "validation_fraction": args.val_size,
        "test_fraction": args.test_size,
        "data_kind_label": args.data_kind,
        "input_mode": "prepared" if args.prepared_dir else "raw_csv",
    }


@contextmanager
def _mlflow_run(args: argparse.Namespace):
    """Start a distinct local MLflow run, or preserve legacy training if absent."""
    mlflow = _load_mlflow()
    if mlflow is None:
        warnings.warn(
            "MLflow is not installed; training will continue without experiment tracking. "
            "Install the project requirements to record this execution.",
            RuntimeWarning,
            stacklevel=2,
        )
        yield None
        return
    tracking_uri = normalize_local_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = _local_experiment(mlflow, tracking_uri, args.mlflow_experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id,
                          nested=mlflow.active_run() is not None):
        mlflow.log_params(_mlflow_params(args))
        mlflow.set_tags({"data_kind": args.data_kind,
                         "input_mode": "prepared" if args.prepared_dir else "raw_csv"})
        yield mlflow


@contextmanager
def _deterministic_training(strict: bool = False):
    """Temporarily request deterministic kernels and restore global settings afterward."""
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    except RuntimeError as exc:
        if strict and "determin" in str(exc).lower():
            raise RuntimeError(
                "Strict deterministic training was requested, but PyTorch reported an unsupported "
                f"nondeterministic operation: {exc}"
            ) from exc
        raise
    finally:
        torch.use_deterministic_algorithms(previous_algorithms, warn_only=previous_warn_only)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark


def make_loaders(data_path: str, traj_cfg: TrajectoryConfig, batch_size: int,
                 num_workers: int = 0, test_size: float = .15, val_size: float = .15,
                 seed: Optional[int] = None
                 ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train, val, test = build_trajectory_sequences(data_path, traj_cfg, test_size=test_size, val_size=val_size)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return (
        DataLoader(TrajectoryDataset(train), batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, generator=generator),
        DataLoader(TrajectoryDataset(val), batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
        DataLoader(TrajectoryDataset(test), batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )


def _check_loss(loss: torch.Tensor, phase: str) -> float:
    value = float(loss.detach().item())
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {phase} loss encountered; stopping without saving this checkpoint.")
    return value


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module, device: torch.device) -> float:
    if len(loader.dataset) == 0: raise ValueError("Cannot train on an empty dataset")
    model.train(); total = 0.0
    for past, future in loader:
        past, future = past.to(device), future.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(past), future)
        _check_loss(loss, "training")
        loss.backward(); optimizer.step(); total += float(loss.item()) * past.size(0)
    result = total / len(loader.dataset)
    if not math.isfinite(result): raise ValueError("Non-finite epoch training loss")
    return result


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
               device: torch.device) -> float:
    if len(loader.dataset) == 0: raise ValueError("Cannot evaluate an empty dataset")
    model.eval(); total = 0.0
    for past, future in loader:
        predictions = model(past.to(device))
        targets = future.to(device)
        if predictions.shape != targets.shape:
            raise ValueError(
                "Model predictions during evaluation must match the GPS target shape; "
                f"got {tuple(predictions.shape)} vs {tuple(targets.shape)}."
            )
        validate_coordinate_tensor(targets, "GPS evaluation targets")
        validate_coordinate_tensor(predictions, "Model predictions during evaluation")
        loss = criterion(predictions, targets)
        total += _check_loss(loss, "evaluation") * past.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def _evaluate_distances(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    if len(loader.dataset) == 0: raise ValueError("Cannot evaluate an empty dataset")
    if hasattr(model, "eval"): model.eval()
    total_ade = 0.0; point_count = 0; total_fde = 0.0; sequence_count = 0
    for past, future in loader:
        past, future = past.to(device), future.to(device)
        predictions = model(past)
        if predictions.shape != future.shape:
            raise ValueError(
                "Model predictions must match the GPS evaluation target shape; "
                f"got {tuple(predictions.shape)} vs {tuple(future.shape)}."
            )
        validate_coordinate_tensor(future, "GPS evaluation targets")
        validate_coordinate_tensor(predictions, "Model predictions during geographic evaluation")
        distances = haversine_distance_km(predictions, future)
        total_ade += float(distances.sum().item()); point_count += distances.numel()
        total_fde += float(distances[:, -1].sum().item()); sequence_count += future.size(0)
    return total_ade / point_count, total_fde / sequence_count


@torch.no_grad()
def evaluate_ade(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    return _evaluate_distances(model, loader, device)[0]


@torch.no_grad()
def evaluate_fde(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    return _evaluate_distances(model, loader, device)[1]


def _atomic_torch_save(value: Any, path: Path) -> None:
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)); os.close(fd)
    try:
        # Saving through a buffer avoids embedding the random atomic-temp filename
        # in PyTorch's ZIP member names, allowing repeatable checkpoint bytes.
        buffer = io.BytesIO()
        torch.save(value, buffer)
        with open(temp, "wb") as stream: stream.write(buffer.getvalue())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def _atomic_json(value: Dict[str, Any], path: Path) -> None:
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def _comparison(baseline: float, lstm: float) -> Dict[str, Any]:
    absolute = baseline - lstm
    percentage = (absolute / baseline * 100.0) if baseline != 0 else None
    return {"absolute_improvement_km": absolute,
            "percentage_improvement": percentage,
            "percentage_status": "defined" if percentage is not None else "undefined: baseline is zero"}


def _validate_args(args: argparse.Namespace) -> None:
    for field in ("epochs", "batch_size", "lr", "hidden_size", "num_layers", "input_length", "forecast_horizon", "min_points"):
        if getattr(args, field) <= 0: raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0: raise ValueError("--num-workers cannot be negative")
    if not 0 <= args.dropout < 1: raise ValueError("--dropout must be in [0, 1)")
    if args.val_size <= 0 or args.test_size <= 0 or args.val_size + args.test_size >= 1:
        raise ValueError("--val-size and --test-size must be positive and sum to less than 1")
    if args.min_points < args.input_length + args.forecast_horizon:
        raise ValueError("--min-points must be at least input-length + forecast-horizon")
    if args.lr <= 0: raise ValueError("--lr must be positive")
    if not args.mlflow_experiment_name.strip():
        raise ValueError("--mlflow-experiment-name must be non-empty")
    normalize_local_tracking_uri(args.mlflow_tracking_uri)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device '{args.device}', but CUDA is unavailable; use --device cpu")
    try: torch.device(args.device)
    except (RuntimeError, ValueError) as exc: raise ValueError(f"Invalid device '{args.device}'") from exc


def run_training(args: argparse.Namespace) -> Path:
    _validate_args(args)
    if args.prepared_dir:
        if not Path(args.prepared_dir).is_dir(): raise FileNotFoundError(f"Prepared data directory not found: {args.prepared_dir}")
    elif not args.data_path or not Path(args.data_path).is_file():
        raise FileNotFoundError(f"GPS dataset not found: {args.data_path}")
    set_seed(args.seed)
    with _deterministic_training(strict=args.strict_determinism):
        with _mlflow_run(args) as tracker:
            return _run_training(args, tracker)


def _run_training(args: argparse.Namespace, tracker=None) -> Path:
    device = torch.device(args.device)
    traj_cfg = TrajectoryConfig(args.input_length, args.forecast_horizon, args.min_points, args.max_animals)
    if args.prepared_dir:
        (train, val, test), manifest = load_prepared_sequences(
            args.prepared_dir, traj_cfg, expected_test_size=args.test_size,
            expected_val_size=args.val_size)
        quality_file = Path(args.prepared_dir) / "dataset_quality.json"
        report = json.loads(quality_file.read_text(encoding="utf-8")) if quality_file.is_file() else {
            "counts": {"rows": 0, "invalid_records": 0, "duplicate_records": 0},
            "valid_observations": 0, "individual_count": 0, "tracks_below_minimum_for_splits": 0,
            "selection": {"selected_individual_count": 0}, "prepared_manifest": manifest}
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(TrajectoryDataset(train), batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, generator=generator)
        val_loader = DataLoader(TrajectoryDataset(val), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)
        test_loader = DataLoader(TrajectoryDataset(test), batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        data_name = "prepared_sequences"
    else:
        report = validate_dataset(args.data_path, traj_cfg, val_size=args.val_size,
                                  test_size=args.test_size, max_animals=traj_cfg.max_animals)
        if not report.get("usable", False): raise ValueError(f"Dataset is unusable: {report.get('error', 'validation failed')}")
        train_loader, val_loader, test_loader = make_loaders(args.data_path, traj_cfg, args.batch_size,
                                                            args.num_workers, args.test_size, args.val_size,
                                                            seed=args.seed)
        data_name = Path(args.data_path).name
    # Canonical pipeline output is stable; legacy output remains per-run.
    if args.canonical_output:
        run_dir = Path(args.canonical_output); run_dir.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(args.out_dir); root.mkdir(parents=True, exist_ok=True)
        run_dir = root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"); run_dir.mkdir()
    if not args.prepared_dir:
        write_quality_report(report, run_dir / "dataset_quality.json", run_dir / "dataset_quality.md")
    model, model_cfg = build_model(hidden_size=args.hidden_size, num_layers=args.num_layers,
                                   dropout=args.dropout, forecast_horizon=args.forecast_horizon)
    model.to(device); criterion = nn.MSELoss(); optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf"); history = []; started = time.perf_counter(); checkpoint_path = run_dir / "trajectory_lstm.pt"
    safe_settings = vars(args).copy()
    # Tracking locations are local operational settings, not scientific run data.
    safe_settings.pop("mlflow_tracking_uri", None)
    safe_settings.pop("mlflow_experiment_name", None)
    safe_settings["data_path"] = Path(args.data_path).name if args.data_path else None
    safe_settings["prepared_dir"] = Path(args.prepared_dir).name if args.prepared_dir else None
    safe_settings["out_dir"] = Path(args.out_dir).name
    safe_settings["canonical_output"] = Path(args.canonical_output).name if args.canonical_output else None
    safe_settings["diagnostics_dir"] = Path(args.diagnostics_dir).name if args.diagnostics_dir else None
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        entry = {"epoch": epoch, "training_loss_mse_degrees_squared": train_loss,
                 "validation_loss_mse_degrees_squared": val_loss,
                 "elapsed_seconds": time.perf_counter() - epoch_start}
        history.append(entry)
        if tracker is not None:
            tracker.log_metrics({"training_loss_mse_degrees_squared": train_loss,
                                 "validation_loss_mse_degrees_squared": val_loss}, step=epoch)
        print(f"Epoch {epoch:03d} | train MSE degrees squared={train_loss:.6f} | validation MSE degrees squared={val_loss:.6f} | {entry['elapsed_seconds']:.2f}s", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            payload = {"model_state": model.state_dict(), "model_cfg": model_cfg.__dict__,
                       "traj_cfg": traj_cfg.__dict__, "seed": args.seed,
                       "best_validation_loss_mse_degrees_squared": best_val,
                       "training_settings": safe_settings}
            _atomic_torch_save(payload, checkpoint_path)
            print(f"Saved best validation checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if tracker is not None:
        tracker.log_params({f"model.{key}": value for key, value in model_cfg.__dict__.items()})
    test_mse = eval_epoch(model, test_loader, criterion, device)
    lstm_ade, lstm_fde = _evaluate_distances(model, test_loader, device)
    base_ade, base_fde = _evaluate_distances(lambda x: persistence_forecast(x, args.forecast_horizon), test_loader, device)
    canonical_output = bool(args.canonical_output)
    if canonical_output:
        diagnostics_dir = Path(args.diagnostics_dir) if args.diagnostics_dir else run_dir.with_name(run_dir.name + "_diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json({"epoch_elapsed_seconds": [entry["elapsed_seconds"] for entry in history],
                      "total_training_seconds": time.perf_counter() - started},
                     diagnostics_dir / "runtime_diagnostics.json")
    summary = {
        "experiment_type": "dataset_training", "data_kind": args.data_kind,
        "data_kind_note": "Not independently verified; user must identify whether the input is real or synthetic.",
        "data_file_name": data_name,
        "configuration": safe_settings, "seed": args.seed, "device": str(device),
        "python_version": platform.python_version(), "pytorch_version": torch.__version__,
        "model_configuration": model_cfg.__dict__, "trajectory_configuration": traj_cfg.__dict__,
        "training_validation_history": ([{k: v for k, v in entry.items() if k != "elapsed_seconds"}
                                          for entry in history] if canonical_output else history),
        "best_validation_loss_mse_degrees_squared": best_val,
        "test_metrics": {"lstm_mse_degrees_squared": test_mse, "lstm_ade_km": lstm_ade,
                         "lstm_fde_km": lstm_fde, "persistence_ade_km": base_ade,
                         "persistence_fde_km": base_fde,
                         "ade_comparison": _comparison(base_ade, lstm_ade),
                         "fde_comparison": _comparison(base_fde, lstm_fde)},
    }
    if not canonical_output:
        summary["total_training_seconds"] = time.perf_counter() - started
    _atomic_json(summary, run_dir / "experiment_summary.json")
    m = summary["test_metrics"]
    md = ("# Trajectory forecasting experiment\n\n"
          f"Dataset: `{data_name}` (source label: {args.data_kind}).\n\n"
          f"Data profile: {report['counts']['rows']} rows, {report.get('valid_observations', 0)} valid observations, "
          f"{report.get('individual_count', 0)} individuals, {report['counts']['invalid_records']} invalid records, "
          f"{report['counts']['duplicate_records']} duplicate records, and "
          f"{report.get('tracks_below_minimum_for_splits', 0)} tracks too short for all splits.\n\n"
          f"Best validation MSE: {best_val:.6f} degrees squared.\n\n"
          f"LSTM test MSE: {test_mse:.6f} degrees squared; ADE: {lstm_ade:.3f} km; FDE: {lstm_fde:.3f} km.\n\n"
          f"Persistence test ADE: {base_ade:.3f} km; FDE: {base_fde:.3f} km.\n\n"
          f"ADE improvement (baseline minus LSTM): {m['ade_comparison']['absolute_improvement_km']:.3f} km; "
          f"{m['ade_comparison']['percentage_improvement'] if m['ade_comparison']['percentage_improvement'] is not None else 'undefined (zero baseline)'}%.\n\n"
          f"FDE improvement (baseline minus LSTM): {m['fde_comparison']['absolute_improvement_km']:.3f} km; "
          f"{m['fde_comparison']['percentage_improvement'] if m['fde_comparison']['percentage_improvement'] is not None else 'undefined (zero baseline)'}%.\n\n"
          "Limitations: the data-kind label is user-supplied and unverified. Evaluation uses strict chronological partitions; test inputs are entirely within the test period. Results describe this dataset and run only.\n")
    (run_dir / "experiment_report.md").write_text(md, encoding="utf-8")
    if tracker is not None:
        tracker.log_metrics({
            "best_validation_loss_mse_degrees_squared": best_val,
            "test_lstm_mse_degrees_squared": test_mse,
            "test_lstm_ade_km": lstm_ade,
            "test_lstm_fde_km": lstm_fde,
            "test_persistence_ade_km": base_ade,
            "test_persistence_fde_km": base_fde,
        })
        tracker.log_artifact(str(checkpoint_path), artifact_path="model")
        tracker.log_artifact(str(run_dir / "experiment_summary.json"), artifact_path="evaluation")
        tracker.log_artifact(str(run_dir / "experiment_report.md"), artifact_path="evaluation")
    print(f"Test LSTM MSE: {test_mse:.6f} degrees squared | ADE: {lstm_ade:.3f} km | FDE: {lstm_fde:.3f} km")
    print(f"Persistence ADE: {base_ade:.3f} km | FDE: {base_fde:.3f} km")
    print(f"Experiment artifacts: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an LSTM trajectory forecaster and compare with persistence")
    parser.add_argument("--data-path"); parser.add_argument("--prepared-dir")
    parser.add_argument("--canonical-output", help="Stable output directory for pipeline use; replaces files in that directory.")
    parser.add_argument("--diagnostics-dir", help="Separate directory for nondeterministic runtime timing diagnostics.")
    parser.add_argument("--input-length", type=int, default=20)
    parser.add_argument("--forecast-horizon", type=int, default=5); parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument("--max-animals", type=int, default=10); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128); parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=.2); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="artifacts/experiments"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=.15); parser.add_argument("--val-size", type=float, default=.15)
    parser.add_argument("--data-kind", choices=("unspecified", "real", "synthetic"), default="unspecified",
                         help="Label the input source in reports; this is not independently verified.")
    parser.add_argument("--strict-determinism", action="store_true",
                         help="Fail if PyTorch encounters an operation without deterministic support.")
    default_mlflow_db = Path(__file__).resolve().parents[2] / "mlruns" / "mlflow.db"
    parser.add_argument("--mlflow-tracking-uri", default=str(default_mlflow_db),
                        help="Local SQLite tracking URI or database path (default: repository mlruns/mlflow.db).")
    parser.add_argument("--mlflow-experiment-name", default="migration-intelligence",
                        help="MLflow experiment name (default: migration-intelligence).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(); args = parser.parse_args(argv)
    if bool(args.data_path) == bool(args.prepared_dir):
        parser.error("provide exactly one of --data-path or --prepared-dir")
    try: run_training(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
