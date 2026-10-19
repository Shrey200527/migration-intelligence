"""Chunked dataset validation and aggregate profiling for Movebank GPS CSVs."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import logging
import numbers
import os
import sqlite3
import tempfile
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.data.data_prep import (
    TrajectoryConfig,
    _select_animals_in_order,
    _validate_config,
)

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = ("timestamp", "individual-local-identifier", "location-long", "location-lat")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _record_identifier(value: Any) -> str:
    """Encode IDs consistently with pandas equality used by drop_duplicates/groupby."""
    if isinstance(value, numbers.Number) and not isinstance(value, complex):
        return f"number:{Decimal(str(value)).normalize()}"
    return f"{type(value).__name__}:{value!r}"


def validate_dataset(
    csv_path: str | Path,
    config: Optional[TrajectoryConfig] = None,
    chunksize: int = 100_000,
    val_size: float = .15,
    test_size: float = .15,
    max_animals: Optional[int] = None,
) -> Dict[str, Any]:
    """Profile a CSV using preprocessing's deterministic first-valid-animal selection.

    Raw rows are streamed. Exact duplicate keys are stored in a temporary SQLite
    index: disk use scales with distinct valid records, while Python memory is
    bounded by the current chunk and per-animal aggregate counts. SQLite checks
    all key fields for equality (the hash-only approach could misclassify a
    collision). The temporary database is removed on success or failure.
    """
    path = Path(csv_path)
    cfg = config or TrajectoryConfig()
    _validate_config(cfg)
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1")
    effective_max = cfg.max_animals if max_animals is None else max_animals
    if effective_max is not None and effective_max < 1:
        raise ValueError("max_animals must be positive or None")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {path}")

    counts: Dict[str, int] = {
        "rows": 0,  # all raw CSV rows
        "missing_values": 0,  # across all raw rows and required columns
        "invalid_records": 0,  # all raw rows; overlapping reason counts follow
        "invalid_timestamps": 0,
        "invalid_coordinates": 0,
        "missing_identifiers": 0,
        "duplicate_records": 0,  # all valid raw rows
        "selected_rows_before_duplicate_removal": 0,
        "selected_duplicate_records": 0,
    }
    observations: Dict[str, int] = {}
    selected_animals: set = set()
    min_time = max_time = None
    schema_failure = None

    try:
        with tempfile.TemporaryDirectory(prefix="migration_validation_") as temp_dir:
            db_path = Path(temp_dir) / "duplicate_index.sqlite3"
            with closing(sqlite3.connect(str(db_path))) as connection, connection:
                connection.execute(
                    "CREATE TABLE seen (timestamp_ns INTEGER, animal TEXT, longitude REAL, latitude REAL, "
                    "UNIQUE(timestamp_ns, animal, longitude, latitude))"
                )
                try:
                    with pd.read_csv(path, chunksize=chunksize) as reader:
                        first = reader.get_chunk()
                        missing_columns = [col for col in REQUIRED_COLUMNS if col not in first.columns]
                        if missing_columns:
                            counts["rows"] += len(first)
                            for chunk in reader:
                                counts["rows"] += len(chunk)
                            schema_failure = missing_columns
                        else:
                            def process_chunk(chunk: pd.DataFrame) -> None:
                                nonlocal min_time, max_time
                                counts["rows"] += len(chunk)
                                counts["missing_values"] += int(chunk[list(REQUIRED_COLUMNS)].isna().sum().sum())

                                ts = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
                                ids = chunk["individual-local-identifier"]
                                missing_id = ids.isna() | ids.astype(str).str.strip().eq("")
                                lat = pd.to_numeric(chunk["location-lat"], errors="coerce")
                                lon = pd.to_numeric(chunk["location-long"], errors="coerce")
                                invalid_coord = (
                                    lat.isna() | lon.isna()
                                    | ~lat.between(-90, 90) | ~lon.between(-180, 180)
                                )
                                invalid_timestamp = ts.isna()
                                invalid_record = invalid_timestamp | missing_id | invalid_coord
                                counts["invalid_timestamps"] += int(invalid_timestamp.sum())
                                counts["missing_identifiers"] += int(missing_id.sum())
                                counts["invalid_coordinates"] += int(invalid_coord.sum())
                                counts["invalid_records"] += int(invalid_record.sum())
                                valid = ~invalid_record
                                if not valid.any():
                                    return

                                # Preprocessing chooses animals after invalid rows are filtered,
                                # in first-appearance order, before removing duplicates.
                                _select_animals_in_order(ids[valid], selected_animals, effective_max)
                                selected = valid & ids.isin(selected_animals)
                                counts["selected_rows_before_duplicate_removal"] += int(selected.sum())
                                valid_ts_ns = ts[valid].astype("int64").tolist()
                                valid_ids = ids[valid].tolist()
                                valid_lons = lon[valid].astype(float).tolist()
                                valid_lats = lat[valid].astype(float).tolist()
                                selected_flags = selected[valid].tolist()
                                valid_timestamps = ts[selected]
                                if not valid_timestamps.empty:
                                    lo, hi = valid_timestamps.min(), valid_timestamps.max()
                                    min_time = lo if min_time is None or lo < min_time else min_time
                                    max_time = hi if max_time is None or hi > max_time else max_time

                                for i, timestamp_ns in enumerate(valid_ts_ns):
                                    key = (
                                        int(timestamp_ns),
                                        _record_identifier(valid_ids[i]),
                                        valid_lons[i],
                                        valid_lats[i],
                                    )
                                    cursor = connection.execute(
                                        "INSERT OR IGNORE INTO seen VALUES (?, ?, ?, ?)", key
                                    )
                                    is_duplicate = cursor.rowcount == 0
                                    if is_duplicate:
                                        counts["duplicate_records"] += 1
                                        if selected_flags[i]:
                                            counts["selected_duplicate_records"] += 1
                                    elif selected_flags[i]:
                                        animal = _record_identifier(valid_ids[i])
                                        observations[animal] = observations.get(animal, 0) + 1

                            process_chunk(first)
                            for chunk in reader:
                                process_chunk(chunk)
                            connection.commit()
                except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
                    raise ValueError(f"Could not parse dataset CSV: {exc}") from exc
    except sqlite3.Error as exc:
        raise ValueError(f"Could not build temporary exact-duplicate index: {exc}") from exc

    if schema_failure is not None:
        return {
            "usable": False,
            "path_name": path.name,
            "required_columns": list(REQUIRED_COLUMNS),
            "missing_columns": schema_failure,
            "error": "Required dataset columns are missing.",
            "counts": counts,
            "config": {**asdict(cfg), "max_animals": effective_max},
        }

    required_points = cfg.input_length + cfg.forecast_horizon
    minimum_for_splits = max(cfg.min_points_per_track, required_points)
    while (
        int(minimum_for_splits * (1 - val_size - test_size)) < required_points
        or int(minimum_for_splits * (1 - test_size))
        - int(minimum_for_splits * (1 - val_size - test_size)) < required_points
        or minimum_for_splits - int(minimum_for_splits * (1 - test_size)) < required_points
    ):
        minimum_for_splits += 1
    short = sum(n < minimum_for_splits for n in observations.values())
    result = {
        "usable": bool(observations and short < len(observations)),
        "path_name": path.name,
        "required_columns": list(REQUIRED_COLUMNS),
        "missing_columns": [],
        "counts": counts,
        "selection": {
            "policy": "first valid individual identifiers encountered in CSV order",
            "max_animals": effective_max,
            "selected_individual_count": len(selected_animals),
        },
        "valid_observations": sum(observations.values()),
        "individual_count": len(observations),
        "observations_per_individual": {
            "min": min(observations.values()) if observations else 0,
            "max": max(observations.values()) if observations else 0,
            "mean": sum(observations.values()) / len(observations) if observations else 0.0,
        },
        "timestamp_range_utc": {
            "min": min_time.isoformat() if min_time is not None else None,
            "max": max_time.isoformat() if max_time is not None else None,
        },
        "tracks_below_minimum_for_splits": int(short),
        "minimum_points_per_track_for_splits": minimum_for_splits,
        "config": {**asdict(cfg), "max_animals": effective_max},
        "count_semantics": {
            "rows": "all raw CSV rows",
            "invalid_*": "all raw CSV rows; reason counts can overlap",
            "duplicate_records": "exact duplicates among all valid rows, before animal selection",
            "selected_rows_before_duplicate_removal": "valid rows belonging to selected animals, including duplicates",
            "selected_duplicate_records": "duplicate rows excluded from selected animals",
            "valid_observations_and_track_lengths": "selected valid rows after duplicate removal",
            "tracks_below_minimum_for_splits": "selected individuals excluded by configured minimum/split policy",
        },
        "duplicate_strategy": "temporary disk-backed SQLite unique index over normalized timestamp, identifier, longitude, and latitude; exact composite-key comparison",
        "notes": "No raw GPS records or individual identifiers are persisted in the report.",
    }
    if not result["usable"]:
        result["error"] = "No selected tracks can form all configured chronological splits."
    return result


def write_quality_report(
    report: Dict[str, Any],
    json_path: str | Path,
    summary_path: Optional[str | Path] = None,
) -> None:
    _atomic_json(Path(json_path), report)
    if summary_path:
        lines = [
            "# Dataset quality summary", "", f"Usable: {report.get('usable', False)}",
            f"Raw CSV rows: {report.get('counts', {}).get('rows', 0)}",
            f"Selected individuals: {report.get('selection', {}).get('selected_individual_count', 0)}",
            f"Selected valid observations: {report.get('valid_observations', 0)}",
            f"Selected duplicate exclusions: {report.get('counts', {}).get('selected_duplicate_records', 0)}",
            f"Tracks below split minimum: {report.get('tracks_below_minimum_for_splits', 'unknown')}",
        ]
        Path(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and profile a Movebank GPS CSV")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--report", default="dataset_quality.json")
    parser.add_argument("--input-length", type=int, default=20)
    parser.add_argument("--forecast-horizon", type=int, default=5)
    parser.add_argument("--min-points-per-track", type=int, default=50)
    parser.add_argument("--max-animals", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args(argv)
    if args.test_size <= 0 or args.val_size <= 0 or args.test_size + args.val_size >= 1:
        parser.error("test-size and val-size must be positive and sum to less than 1")
    cfg = TrajectoryConfig(
        args.input_length, args.forecast_horizon, args.min_points_per_track, args.max_animals
    )
    try:
        report = validate_dataset(args.data_path, cfg, val_size=args.val_size, test_size=args.test_size)
    except (OSError, ValueError) as exc:
        LOGGER.error("Dataset validation failed: %s", exc)
        return 2
    write_quality_report(report, args.report)
    print(json.dumps(report, indent=2))
    return 0 if report["usable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
