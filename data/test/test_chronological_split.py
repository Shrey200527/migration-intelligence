import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.data_prep import TrajectoryConfig, build_trajectory_sequences


def main():
    rows = []
    for animal in ["bird_A", "bird_B"]:
        for t in range(300):
            rows.append({
                "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=t),
                "individual-local-identifier": animal,
                "location-long": 10.0 + t * 0.01,
                "location-lat": 50.0 + t * 0.01,
            })
    cfg = TrajectoryConfig(20, 5, 50, None)
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "synthetic_gps.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        train, val, test = build_trajectory_sequences(str(csv_path), cfg)
        assert train and val and test, "A split is empty!"

        def mean_future_latitude(sequences):
            return np.mean([future[:, 0].mean() for _, future in sequences])

        train_mean, val_mean, test_mean = map(mean_future_latitude, (train, val, test))
        assert train_mean < val_mean < test_mean, "Forecasts are not in chronological order."
        print("Train sequences:", len(train))
        print("Validation sequences:", len(val))
        print("Test sequences:", len(test))
        print("Mean future latitude:", train_mean, val_mean, test_mean)
        print("PASS: all splits are non-empty and chronologically ordered.")


if __name__ == "__main__":
    main()
