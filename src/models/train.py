import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.data.data_prep import (
    TrajectoryConfig,
    TrajectoryDataset,
    build_trajectory_sequences,
)
from src.models.model import build_model


def make_loaders(
    data_path: str,
    traj_cfg: TrajectoryConfig,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_seqs, val_seqs, test_seqs = build_trajectory_sequences(data_path, traj_cfg)

    train_ds = TrajectoryDataset(train_seqs)
    val_ds = TrajectoryDataset(val_seqs)
    test_ds = TrajectoryDataset(test_seqs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for past, future in loader:
        past = past.to(device)
        future = future.to(device)

        optimizer.zero_grad()
        preds = model(past)
        loss = criterion(preds, future)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * past.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for past, future in loader:
        past = past.to(device)
        future = future.to(device)
        preds = model(past)
        loss = criterion(preds, future)
        total_loss += loss.item() * past.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM trajectory forecaster on Movebank GPS data.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to GPS CSV file.")
    parser.add_argument("--input-length", type=int, default=20)
    parser.add_argument("--forecast-horizon", type=int, default=5)
    parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument("--max-animals", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device(args.device)

    traj_cfg = TrajectoryConfig(
        input_length=args.input_length,
        forecast_horizon=args.forecast_horizon,
        min_points_per_track=args.min_points,
        max_animals=args.max_animals,
    )

    train_loader, val_loader, test_loader = make_loaders(
        data_path=args.data_path,
        traj_cfg=traj_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Data loaded: {len(train_loader.dataset)} train | {len(val_loader.dataset)} val | {len(test_loader.dataset)} test sequences", flush=True)

    model, model_cfg = build_model(
        input_size=2,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        forecast_horizon=args.forecast_horizon,
    )
    model.to(device)
    print(f"Model ready on {device}. Starting training...", flush=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(out_dir, "trajectory_lstm.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": model_cfg.__dict__,
                    "traj_cfg": traj_cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"  Saved new best model to {ckpt_path}", flush=True)

    # Final test evaluation
    test_loss = eval_epoch(model, test_loader, criterion, device)
    print(f"Test loss (MSE on coordinates): {test_loss:.4f}")


if __name__ == "__main__":
    main()

