"""Reproducible CPU/GPU training and fair persistence-baseline evaluation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import platform
import random
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.data.data_prep import TrajectoryConfig, TrajectoryDataset, build_trajectory_sequences
from src.data.validation import validate_dataset, write_quality_report
from src.models.baseline import persistence_forecast
from src.models.model import build_model
from src.monitoring.metrics import haversine_distance_km, validate_coordinate_tensor


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without changing process-wide backend policy."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


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
    try: torch.save(value, temp); os.replace(temp, path)
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
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device '{args.device}', but CUDA is unavailable; use --device cpu")
    try: torch.device(args.device)
    except (RuntimeError, ValueError) as exc: raise ValueError(f"Invalid device '{args.device}'") from exc


def run_training(args: argparse.Namespace) -> Path:
    _validate_args(args)
    if not Path(args.data_path).is_file(): raise FileNotFoundError(f"GPS dataset not found: {args.data_path}")
    set_seed(args.seed)
    with _deterministic_training(strict=args.strict_determinism):
        return _run_training(args)


def _run_training(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    traj_cfg = TrajectoryConfig(args.input_length, args.forecast_horizon, args.min_points, args.max_animals)
    report = validate_dataset(
        args.data_path,
        traj_cfg,
        val_size=args.val_size,
        test_size=args.test_size,
        max_animals=traj_cfg.max_animals,
    )
    if not report.get("usable", False): raise ValueError(f"Dataset is unusable: {report.get('error', 'validation failed')}")
    train_loader, val_loader, test_loader = make_loaders(args.data_path, traj_cfg, args.batch_size,
                                                        args.num_workers, args.test_size, args.val_size,
                                                        seed=args.seed)
    # Use a unique run subdirectory so prior artifacts are never silently replaced.
    root = Path(args.out_dir); root.mkdir(parents=True, exist_ok=True)
    run_dir = root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"); run_dir.mkdir()
    write_quality_report(report, run_dir / "dataset_quality.json", run_dir / "dataset_quality.md")
    model, model_cfg = build_model(hidden_size=args.hidden_size, num_layers=args.num_layers,
                                   dropout=args.dropout, forecast_horizon=args.forecast_horizon)
    model.to(device); criterion = nn.MSELoss(); optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf"); history = []; started = time.perf_counter(); checkpoint_path = run_dir / "trajectory_lstm.pt"
    safe_settings = vars(args).copy()
    safe_settings["data_path"] = Path(args.data_path).name
    safe_settings["out_dir"] = Path(args.out_dir).name
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        entry = {"epoch": epoch, "training_loss_mse_degrees_squared": train_loss,
                 "validation_loss_mse_degrees_squared": val_loss,
                 "elapsed_seconds": time.perf_counter() - epoch_start}
        history.append(entry)
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
    test_mse = eval_epoch(model, test_loader, criterion, device)
    lstm_ade, lstm_fde = _evaluate_distances(model, test_loader, device)
    base_ade, base_fde = _evaluate_distances(lambda x: persistence_forecast(x, args.forecast_horizon), test_loader, device)
    summary = {
        "experiment_type": "dataset_training", "data_kind": args.data_kind,
        "data_kind_note": "Not independently verified; user must identify whether the input is real or synthetic.",
        "data_file_name": Path(args.data_path).name,
        "configuration": safe_settings, "seed": args.seed, "device": str(device),
        "python_version": platform.python_version(), "pytorch_version": torch.__version__,
        "model_configuration": model_cfg.__dict__, "trajectory_configuration": traj_cfg.__dict__,
        "training_validation_history": history, "best_validation_loss_mse_degrees_squared": best_val,
        "test_metrics": {"lstm_mse_degrees_squared": test_mse, "lstm_ade_km": lstm_ade,
                         "lstm_fde_km": lstm_fde, "persistence_ade_km": base_ade,
                         "persistence_fde_km": base_fde,
                         "ade_comparison": _comparison(base_ade, lstm_ade),
                         "fde_comparison": _comparison(base_fde, lstm_fde)},
        "total_training_seconds": time.perf_counter() - started,
    }
    _atomic_json(summary, run_dir / "experiment_summary.json")
    m = summary["test_metrics"]
    md = ("# Trajectory forecasting experiment\n\n"
          f"Dataset: `{Path(args.data_path).name}` (source label: {args.data_kind}).\n\n"
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
    print(f"Test LSTM MSE: {test_mse:.6f} degrees squared | ADE: {lstm_ade:.3f} km | FDE: {lstm_fde:.3f} km")
    print(f"Persistence ADE: {base_ade:.3f} km | FDE: {base_fde:.3f} km")
    print(f"Experiment artifacts: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an LSTM trajectory forecaster and compare with persistence")
    parser.add_argument("--data-path", required=True); parser.add_argument("--input-length", type=int, default=20)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(); args = parser.parse_args(argv)
    try: run_training(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
