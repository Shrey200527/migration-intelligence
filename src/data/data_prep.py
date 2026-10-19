
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


Sequence = Tuple[np.ndarray, np.ndarray]


@dataclass
class TrajectoryConfig:
    """Configuration for constructing trajectory forecasting examples."""

    input_length: int = 20
    forecast_horizon: int = 5
    min_points_per_track: int = 50
    max_animals: Optional[int] = 20


class TrajectoryDataset(Dataset):
    """
    PyTorch dataset of trajectory windows.

    Each item contains:
        past:   (input_length, 2) latitude/longitude coordinates
        future: (forecast_horizon, 2) latitude/longitude coordinates
    """

    def __init__(
        self,
        sequences: List[Sequence],
        device: Optional[torch.device] = None,
    ) -> None:
        self.sequences = sequences
        self.device = device

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        past, future = self.sequences[idx]

        past_tensor = torch.from_numpy(past).float()
        future_tensor = torch.from_numpy(future).float()

        if self.device is not None:
            past_tensor = past_tensor.to(self.device)
            future_tensor = future_tensor.to(self.device)

        return past_tensor, future_tensor


def _validate_config(cfg: TrajectoryConfig) -> None:
    """Reject invalid sequence configuration before processing data."""

    if cfg.input_length < 1:
        raise ValueError("input_length must be at least 1.")

    if cfg.forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1.")

    if cfg.min_points_per_track < cfg.input_length + cfg.forecast_horizon:
        raise ValueError(
            "min_points_per_track must be at least "
            "input_length + forecast_horizon."
        )

    if cfg.max_animals is not None and cfg.max_animals < 1:
        raise ValueError("max_animals must be positive or None.")


def _select_animals_in_order(
    animal_ids,
    selected_animals: set,
    max_animals: Optional[int],
) -> None:
    """Mutate selection state using the first valid animal IDs encountered."""
    if max_animals is None:
        selected_animals.update(animal_ids)
        return
    for animal_id in pd.unique(animal_ids):
        if animal_id in selected_animals:
            continue
        if len(selected_animals) >= max_animals:
            break
        selected_animals.add(animal_id)


def _load_movebank_gps(
    csv_path: str,
    max_animals: Optional[int] = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load Movebank-style GPS data in chunks.

    Required CSV columns:
        timestamp
        individual-local-identifier
        location-long
        location-lat

    When max_animals is specified, retain the first animals encountered
    but continue reading the file so their later observations are included.
    """

    if max_animals is not None and max_animals < 1:
        raise ValueError("max_animals must be positive or None.")

    required_columns = [
        "timestamp",
        "individual-local-identifier",
        "location-long",
        "location-lat",
    ]

    chunks: List[pd.DataFrame] = []
    selected_animals = set()

    try:
        reader = pd.read_csv(
            csv_path,
            usecols=required_columns,
            parse_dates=["timestamp"] if parse_dates else None,
            chunksize=100_000,
        )

        for chunk in reader:
            # Parse timestamps explicitly: pandas' parse_dates may leave an
            # object column when even one malformed value is present.
            chunk["timestamp"] = pd.to_datetime(
                chunk["timestamp"], errors="coerce", utc=True
            )

            # Missing identifiers or invalid timestamps cannot define a track.
            chunk = chunk.dropna(
                subset=[
                    "timestamp",
                    "individual-local-identifier",
                    "location-long",
                    "location-lat",
                ]
            )
            valid_identifiers = chunk["individual-local-identifier"].astype(str).str.strip().ne("")
            chunk = chunk.loc[valid_identifiers].copy()

            if chunk.empty:
                continue

            # Convert coordinates explicitly and discard invalid values.
            chunk["location-long"] = pd.to_numeric(
                chunk["location-long"], errors="coerce"
            )
            chunk["location-lat"] = pd.to_numeric(
                chunk["location-lat"], errors="coerce"
            )

            chunk = chunk.dropna(
                subset=["location-long", "location-lat"]
            )

            valid_coordinates = (
                chunk["location-lat"].between(-90, 90)
                & chunk["location-long"].between(-180, 180)
            )
            chunk = chunk.loc[valid_coordinates].copy()

            if chunk.empty:
                continue

            # Select animals deterministically in order of first appearance.
            if max_animals is not None:
                _select_animals_in_order(
                    chunk["individual-local-identifier"],
                    selected_animals,
                    max_animals,
                )

                chunk = chunk[
                    chunk["individual-local-identifier"].isin(
                        selected_animals
                    )
                ]

            if not chunk.empty:
                chunks.append(chunk)

    except ValueError as exc:
        raise ValueError(
            "Could not read the GPS CSV. Confirm that it contains the "
            "required columns: timestamp, individual-local-identifier, "
            "location-long, and location-lat."
        ) from exc

    if not chunks:
        raise ValueError(
            "No valid GPS observations were loaded. Check the CSV path, "
            "required columns, identifiers, timestamps, and coordinates."
        )

    df = pd.concat(chunks, ignore_index=True)

    # Remove exact duplicate records, but retain distinct observations
    # even when an animal has multiple records with the same timestamp.
    df = df.drop_duplicates(subset=required_columns)

    df = df.sort_values(
        ["individual-local-identifier", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    return df


def _to_sequences_for_animal(
    df_animal: pd.DataFrame,
    cfg: TrajectoryConfig,
) -> List[Sequence]:
    """
    Create sliding windows within a single already-partitioned track segment.

    This helper does not split data. Its caller must supply observations
    belonging entirely to one split.
    """

    coords = df_animal[
        ["location-lat", "location-long"]
    ].to_numpy(dtype=np.float32)

    T = cfg.input_length
    H = cfg.forecast_horizon

    if len(coords) < T + H:
        return []

    if not np.isfinite(coords).all():
        raise ValueError("Trajectory coordinates contain non-finite values.")

    windows: List[Sequence] = []

    for start in range(len(coords) - T - H + 1):
        past = coords[start : start + T]
        future = coords[start + T : start + T + H]

        windows.append((past.copy(), future.copy()))

    return windows


def build_trajectory_sequences(
    csv_path: str,
    cfg: TrajectoryConfig,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Sequence], List[Sequence], List[Sequence]]:
    """
    Build training, validation, and test sequences using chronological splits.

    For each animal independently:
        1. Sort observations by timestamp.
        2. Partition observations into train, validation, and test periods.
        3. Generate windows independently inside each period.

    Default proportions:
        Train:      70%
        Validation: 15%
        Test:       15%

    Because windows are generated independently within disjoint raw-data
    partitions, no input or forecast target can cross a split boundary.
    No extra gap is required for this non-overlap guarantee.

    Tracks that cannot produce at least one window in every split are
    excluded from all three splits.

    random_state is retained for backward compatibility. Chronological
    splitting is deterministic and does not use random_state.
    """

    _validate_config(cfg)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")

    if val_size + test_size >= 1:
        raise ValueError(
            "val_size + test_size must be less than 1."
        )

    df = _load_movebank_gps(
        csv_path,
        max_animals=cfg.max_animals,
    )

    train_sequences: List[Sequence] = []
    val_sequences: List[Sequence] = []
    test_sequences: List[Sequence] = []

    eligible_animals = 0
    skipped_short_tracks = 0
    skipped_insufficient_splits = 0

    for _, df_animal in df.groupby(
        "individual-local-identifier",
        sort=False,
    ):
        coords = df_animal[
            ["location-lat", "location-long"]
        ].to_numpy(dtype=np.float32)

        n = len(coords)

        if n < cfg.min_points_per_track:
            skipped_short_tracks += 1
            continue

        # Partition the ordered observations before creating any windows.
        train_end = int(n * (1.0 - val_size - test_size))
        val_end = int(n * (1.0 - test_size))

        train_coords = coords[:train_end]
        val_coords = coords[train_end:val_end]
        test_coords = coords[val_end:]

        required_length = cfg.input_length + cfg.forecast_horizon

        # Every split must support at least one complete input/forecast pair.
        if min(
            len(train_coords),
            len(val_coords),
            len(test_coords),
        ) < required_length:
            skipped_insufficient_splits += 1
            continue

        # Generate windows independently so no window crosses a boundary.
        train_windows = _to_sequences_for_animal(
            pd.DataFrame(
                train_coords,
                columns=["location-lat", "location-long"],
            ),
            cfg,
        )
        val_windows = _to_sequences_for_animal(
            pd.DataFrame(
                val_coords,
                columns=["location-lat", "location-long"],
            ),
            cfg,
        )
        test_windows = _to_sequences_for_animal(
            pd.DataFrame(
                test_coords,
                columns=["location-lat", "location-long"],
            ),
            cfg,
        )

        # Do not let an animal contribute to only some splits.
        if not train_windows or not val_windows or not test_windows:
            skipped_insufficient_splits += 1
            continue

        train_sequences.extend(train_windows)
        val_sequences.extend(val_windows)
        test_sequences.extend(test_windows)
        eligible_animals += 1

    if not train_sequences or not val_sequences or not test_sequences:
        raise ValueError(
            "Could not construct non-empty train, validation, and test "
            "splits. Increase the number of GPS observations per animal, "
            "lower input_length or forecast_horizon, or adjust val_size "
            "and test_size. Each contributing animal must have at least "
            "input_length + forecast_horizon observations in every split. "
            f"Eligible animals: {eligible_animals}; "
            f"tracks below min_points_per_track: {skipped_short_tracks}; "
            f"tracks with insufficient split lengths: "
            f"{skipped_insufficient_splits}."
        )

    print(f"Animals contributing to all splits: {eligible_animals}")
    print(f"Tracks skipped for insufficient total length: {skipped_short_tracks}")
    print(
        "Tracks skipped because a split was too short: "
        f"{skipped_insufficient_splits}"
    )

    return train_sequences, val_sequences, test_sequences


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Movebank GPS data and build chronological "
            "trajectory sequences."
        )
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the GPS CSV file.",
    )
    parser.add_argument("--input-length", type=int, default=20)
    parser.add_argument("--forecast-horizon", type=int, default=5)
    parser.add_argument("--min-points-per-track", type=int, default=50)
    parser.add_argument("--max-animals", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)

    args = parser.parse_args()

    cfg = TrajectoryConfig(
        input_length=args.input_length,
        forecast_horizon=args.forecast_horizon,
        min_points_per_track=args.min_points_per_track,
        max_animals=args.max_animals,
    )

    train_seqs, val_seqs, test_seqs = build_trajectory_sequences(
        args.data_path,
        cfg,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    print(f"Train sequences: {len(train_seqs)}")
    print(f"Validation sequences: {len(val_seqs)}")
    print(f"Test sequences: {len(test_seqs)}")


if __name__ == "__main__":
    main()
