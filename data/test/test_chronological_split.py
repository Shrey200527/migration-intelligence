import numpy as np
import pandas as pd

from src.data.data_prep import TrajectoryConfig, build_trajectory_sequences

rows = []
for animal in ["bird_A", "bird_B"]:
    for t in range(300):
        rows.append({
            "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=t),
            "individual-local-identifier": animal,
            "location-long": 10.0 + t * 0.01,
            "location-lat": 50.0 + t * 0.01,
        })

csv_path = "data/test/synthetic_gps.csv"
pd.DataFrame(rows).to_csv(csv_path, index=False)

cfg = TrajectoryConfig(
    input_length=20,
    forecast_horizon=5,
    min_points_per_track=50,
    max_animals=None,
)

train, val, test = build_trajectory_sequences(csv_path, cfg)

assert train and val and test, "A split is empty!"

def mean_future_latitude(sequences):
    return np.mean([future[:, 0].mean() for _, future in sequences])

train_mean = mean_future_latitude(train)
val_mean = mean_future_latitude(val)
test_mean = mean_future_latitude(test)

print("Train sequences:", len(train))
print("Validation sequences:", len(val))
print("Test sequences:", len(test))
print("Mean future latitude:", train_mean, val_mean, test_mean)

assert train_mean < val_mean < test_mean, (
    "Forecasts are not in chronological order."
)

print("PASS: all splits are non-empty and chronologically ordered.")
