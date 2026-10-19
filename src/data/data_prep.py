
import argparse
from dataclasses import asdict, dataclass
import io
import json
import os
import tempfile
import zipfile
import zlib
from pathlib import Path
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

    for name in ("input_length", "forecast_horizon", "min_points_per_track"):
        if type(getattr(cfg, name)) is not int:
            raise ValueError(f"{name} must be an integer.")
    if cfg.max_animals is not None and type(cfg.max_animals) is not int:
        raise ValueError("max_animals must be a positive integer or None.")

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


def _validate_split_sizes(test_size: float, val_size: float) -> None:
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1.")


def save_prepared_sequences(output_dir: str | Path, splits, cfg: TrajectoryConfig,
                            test_size: float, val_size: float, random_state: int = 42) -> dict:
    """Persist window arrays without pickle and write a deterministic, aggregate-only manifest."""
    _validate_config(cfg)
    _validate_split_sizes(test_size, val_size)
    if len(splits) != 3:
        raise ValueError("splits must contain train, validation, and test sequences")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = ("train", "validation", "test")
    counts = {}
    for name, sequences in zip(names, splits):
        if not sequences:
            raise ValueError(f"Prepared {name} split is empty.")
        past = np.stack([pair[0] for pair in sequences]).astype(np.float32, copy=False)
        future = np.stack([pair[1] for pair in sequences]).astype(np.float32, copy=False)
        expected = ((len(sequences), cfg.input_length, 2), (len(sequences), cfg.forecast_horizon, 2))
        if past.shape != expected[0] or future.shape != expected[1] or not np.isfinite(past).all() or not np.isfinite(future).all():
            raise ValueError(f"Prepared {name} arrays have invalid shapes or non-finite coordinates.")
        if (np.any((past[:, :, 0] < -90) | (past[:, :, 0] > 90))
                or np.any((future[:, :, 0] < -90) | (future[:, :, 0] > 90))
                or np.any((past[:, :, 1] < -180) | (past[:, :, 1] > 180))
                or np.any((future[:, :, 1] < -180) | (future[:, :, 1] > 180))):
            raise ValueError(f"Prepared {name} arrays contain out-of-range latitude or longitude.")
        fd, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=str(output))
        try:
            with os.fdopen(fd, "wb") as stream:
                with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for array_name, array in (("past", past), ("future", future)):
                        buffer = io.BytesIO()
                        np.save(buffer, array, allow_pickle=False)
                        info = zipfile.ZipInfo(f"{array_name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                        info.compress_type = zipfile.ZIP_DEFLATED
                        info.external_attr = 0o600 << 16
                        archive.writestr(info, buffer.getvalue())
            os.replace(temporary, output / f"{name}.npz")
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        counts[name] = {"sequence_count": len(sequences), "past_shape": list(past.shape),
                        "future_shape": list(future.shape), "dtype": "float32",
                        "feature_order": ["latitude", "longitude"]}
    manifest = {"format_version": 1, "coordinate_order": ["latitude", "longitude"],
                "coordinate_units": "degrees", "config": asdict(cfg),
                "split": {"policy": "chronological per individual before window generation; windows never cross split boundaries",
                          "val_size": val_size, "test_size": test_size, "random_state_unused": random_state},
                "splits": counts}
    manifest_path = output / "manifest.json"
    fd, temporary = tempfile.mkstemp(prefix="manifest.", suffix=".tmp", dir=str(output))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        os.replace(temporary, manifest_path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return manifest


def load_prepared_sequences(input_dir: str | Path, expected_config: TrajectoryConfig | None = None,
                            expected_test_size: float | None = None,
                            expected_val_size: float | None = None):
    """Load and validate the safe NPZ/JSON prepared-data format (pickle disabled)."""
    source = Path(input_dir)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file(): raise FileNotFoundError(f"Prepared-data manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict): raise ValueError("manifest root must be a JSON object")
        if manifest.get("format_version") != 1: raise ValueError("unsupported format_version")
        if not isinstance(manifest.get("config"), dict): raise ValueError("manifest config must be an object")
        if not isinstance(manifest.get("split"), dict): raise ValueError("manifest split metadata must be an object")
        if not isinstance(manifest.get("splits"), dict): raise ValueError("manifest splits metadata must be an object")
        cfg = TrajectoryConfig(**manifest["config"])
        _validate_config(cfg)
        _validate_split_sizes(manifest["split"]["test_size"], manifest["split"]["val_size"])
        if expected_config is not None and asdict(expected_config) != asdict(cfg):
            raise ValueError("prepared-data configuration does not match requested training configuration")
        if expected_test_size is not None and manifest["split"]["test_size"] != expected_test_size:
            raise ValueError("prepared-data test_size does not match requested training configuration")
        if expected_val_size is not None and manifest["split"]["val_size"] != expected_val_size:
            raise ValueError("prepared-data val_size does not match requested training configuration")
        if manifest.get("coordinate_order") != ["latitude", "longitude"]:
            raise ValueError("unsupported coordinate order")
        if manifest.get("coordinate_units") != "degrees":
            raise ValueError("unsupported coordinate units")
        if manifest["split"].get("policy") != "chronological per individual before window generation; windows never cross split boundaries":
            raise ValueError("unsupported chronological split policy")
        result = []
        for name in ("train", "validation", "test"):
            path = source / f"{name}.npz"
            if not path.is_file(): raise FileNotFoundError(f"Prepared {name} artifact not found: {path}")
            if not zipfile.is_zipfile(path):
                raise ValueError(f"Prepared {name} artifact is truncated or is not a valid NPZ archive")
            with zipfile.ZipFile(path) as archive_check:
                corrupt_member = archive_check.testzip()
                if corrupt_member is not None:
                    raise ValueError(f"Prepared {name} artifact has a corrupt member: {corrupt_member}")
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"past", "future"}: raise ValueError(f"{name}.npz must contain past and future arrays")
                past, future = archive["past"], archive["future"]
            if past.dtype != np.float32 or future.dtype != np.float32: raise ValueError(f"{name} arrays must have dtype float32")
            if past.ndim != 3 or future.ndim != 3 or past.shape[1:] != (cfg.input_length, 2) or future.shape[1:] != (cfg.forecast_horizon, 2) or len(past) != len(future) or len(past) == 0:
                raise ValueError(f"{name} arrays have incompatible or empty shapes")
            if not np.isfinite(past).all() or not np.isfinite(future).all(): raise ValueError(f"{name} arrays contain non-finite coordinates")
            if (np.any((past[:, :, 0] < -90) | (past[:, :, 0] > 90))
                    or np.any((future[:, :, 0] < -90) | (future[:, :, 0] > 90))
                    or np.any((past[:, :, 1] < -180) | (past[:, :, 1] > 180))
                    or np.any((future[:, :, 1] < -180) | (future[:, :, 1] > 180))):
                raise ValueError(f"{name} arrays contain out-of-range latitude or longitude")
            validate_meta = manifest["splits"][name]
            if not isinstance(validate_meta, dict): raise ValueError(f"{name} manifest metadata must be an object")
            if validate_meta.get("sequence_count") != len(past) or validate_meta.get("past_shape") != list(past.shape) or validate_meta.get("future_shape") != list(future.shape) or validate_meta.get("dtype") != "float32" or validate_meta.get("feature_order") != ["latitude", "longitude"]:
                raise ValueError(f"{name} arrays do not match manifest metadata")
            result.append(list(zip(past, future)))
        return tuple(result), manifest
    except (KeyError, TypeError, json.JSONDecodeError, OSError, EOFError, ValueError,
            zipfile.BadZipFile, zipfile.LargeZipFile, zlib.error) as exc:
        raise ValueError(f"Invalid prepared data at {source}: {exc}") from exc


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
