import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


@dataclass
class TrajectoryConfig:
    input_length: int = 20
    forecast_horizon: int = 5
    min_points_per_track: int = 50
    max_animals: Optional[int] = 20  # limit for quick experiments


class TrajectoryDataset(Dataset):
    """
    Dataset of sliding-window trajectories.

    Each item:
      - past:  (T, F) tensor of past features (here lat/long)
      - future: (H, 2) tensor of future lat/long
    """

    def __init__(
        self,
        sequences: List[Tuple[np.ndarray, np.ndarray]],
        device: Optional[torch.device] = None,
    ) -> None:
        self.sequences = sequences
        self.device = device

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        past, future = self.sequences[idx]
        past_t = torch.from_numpy(past).float()
        future_t = torch.from_numpy(future).float()
        if self.device is not None:
            past_t = past_t.to(self.device)
            future_t = future_t.to(self.device)
        return past_t, future_t


def _load_movebank_gps(
    csv_path: str,
    max_animals: Optional[int] = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load a Movebank-style GPS CSV in a memory-conscious way.

    Expected columns:
      - timestamp
      - individual-local-identifier
      - location-long
      - location-lat
    """
    usecols = [
        "timestamp",
        "individual-local-identifier",
        "location-long",
        "location-lat",
    ]

    chunks: List[pd.DataFrame] = []
    animal_order: List[str] = []

    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        parse_dates=["timestamp"] if parse_dates else None,
        chunksize=100_000,
    ):
        # Drop rows with missing coordinates
        chunk = chunk.dropna(subset=["location-long", "location-lat"])
        if chunk.empty:
            continue

        # Track order of animals, and optionally limit
        for aid in chunk["individual-local-identifier"].unique():
            if aid not in animal_order:
                animal_order.append(aid)

        if max_animals is not None and len(animal_order) > max_animals:
            keep = set(animal_order[:max_animals])
            chunk = chunk[chunk["individual-local-identifier"].isin(keep)]

        chunks.append(chunk)

        if max_animals is not None and len(animal_order) >= max_animals:
            # We have seen enough animals — stop reading further chunks
            break

    if not chunks:
        raise ValueError("No data loaded. Check CSV path and column names.")

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["individual-local-identifier", "timestamp"])
    return df


def _to_sequences_for_animal(
    df_animal: pd.DataFrame,
    cfg: TrajectoryConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    coords = df_animal[["location-lat", "location-long"]].to_numpy(dtype=np.float32)

    if len(coords) < cfg.min_points_per_track:
        return []

    T = cfg.input_length
    H = cfg.forecast_horizon
    seqs: List[Tuple[np.ndarray, np.ndarray]] = []

    # Sliding window
    for start in range(0, len(coords) - T - H + 1):
        past = coords[start : start + T]
        future = coords[start + T : start + T + H]
        seqs.append((past, future))

    return seqs


def build_trajectory_sequences(
    csv_path: str,
    cfg: TrajectoryConfig,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """
    High-level helper:
      1) Load Movebank GPS CSV
      2) Build sliding-window sequences for each animal
      3) Split into train/val/test
    """
    df = _load_movebank_gps(csv_path, max_animals=cfg.max_animals)

    sequences: List[Tuple[np.ndarray, np.ndarray]] = []
    for aid, df_animal in df.groupby("individual-local-identifier"):
        sequences.extend(_to_sequences_for_animal(df_animal, cfg))

    if not sequences:
        raise ValueError("No sequences created. Try lowering min_points_per_track.")

    # Build indices for splitting
    indices = np.arange(len(sequences))
    train_idx, temp_idx = train_test_split(
        indices, test_size=test_size + val_size, random_state=random_state, shuffle=True
    )
    rel_val_size = val_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1 - rel_val_size, random_state=random_state, shuffle=True
    )

    train_seqs = [sequences[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]
    test_seqs = [sequences[i] for i in test_idx]

    return train_seqs, val_seqs, test_seqs


def main():
    parser = argparse.ArgumentParser(description="Inspect Movebank GPS CSV and build sequences.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to GPS CSV file.")
    parser.add_argument("--input-length", type=int, default=20)
    parser.add_argument("--forecast-horizon", type=int, default=5)
    parser.add_argument("--max-animals", type=int, default=5)
    args = parser.parse_args()

    cfg = TrajectoryConfig(
        input_length=args.input_length,
        forecast_horizon=args.forecast_horizon,
        max_animals=args.max_animals,
    )
    train_seqs, val_seqs, test_seqs = build_trajectory_sequences(args.data_path, cfg)
    print(f"Train sequences: {len(train_seqs)}")
    print(f"Val sequences:   {len(val_seqs)}")
    print(f"Test sequences:  {len(test_seqs)}")


if __name__ == "__main__":
    main()

