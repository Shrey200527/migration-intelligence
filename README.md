# Migration Intelligence — Phase 1

Phase 1 provides a scientific baseline for GPS trajectory forecasting: dataset checks, chronological sequence construction, an LSTM, a persistence baseline, and geographic evaluation. It does not include a serving API, deployment stack, orchestration, DVC, or MLflow.

## Dataset

Supply a local Movebank-style CSV with these columns:

| Column | Meaning |
|---|---|
| `timestamp` | Observation time, parseable as a date/time |
| `individual-local-identifier` | Animal/track identifier |
| `location-long` | Longitude in degrees, from -180 to 180 |
| `location-lat` | Latitude in degrees, from -90 to 90 |

Preprocessing rejects missing or blank identifiers, missing or unparseable timestamps, non-numeric or out-of-range coordinates, and exact duplicate records. Validation applies the same first-valid-individuals-in-CSV-order selection as training (`--max-animals`), then reports selected track lengths and split-policy exclusions. Its `rows` count covers all raw CSV rows; invalid reason counts cover all rows and can overlap; duplicate totals cover valid rows before animal selection; selected-row and selected-duplicate counts cover the training subset; valid observation totals and track lengths cover selected rows after duplicate removal. Reports contain aggregate information only.

Exact duplicates are counted with a temporary SQLite unique index using normalized timestamp, identifier, longitude, and latitude fields. This performs disk I/O proportional to distinct valid records while keeping Python memory bounded by the current input chunk and per-individual aggregates. The temporary index is deleted after profiling.

## Validate and profile

From the repository root, using the existing Windows virtual environment:

```powershell
.\.venv\Scripts\python.exe -m src.data.validation --data-path .\data\raw\movebank.csv --report .\artifacts\dataset_quality.json
```

The command returns a nonzero exit code when the schema is missing, no usable track exists, or the dataset cannot provide each split with at least one input/forecast window.
It accepts `--max-animals`; the default is the first five valid individual IDs, matching the validation CLI configuration. Training defaults to ten; pass the same `--max-animals` value to validation when profiling for a training run.

## Tests

`pytest` is not included in the existing environment. Run the automated suite with the standard library:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe data\test\test_chronological_split.py
.\.venv\Scripts\python.exe data\test\test_ade.py
```

The synthetic suite uses temporary directories and does not require the real dataset.

## Train and compare

```powershell
.\.venv\Scripts\python.exe -m src.models.train --data-path .\data\raw\movebank.csv --device cpu --data-kind real --out-dir .\artifacts\experiments
```

The CLI retains options such as `--input-length`, `--forecast-horizon`, `--epochs`, `--batch-size`, and `--seed`. Validation and preprocessing use the same trajectory config, including `--max-animals`; selection is deterministic by first valid appearance in CSV order. Set `--data-kind` to `real` or `synthetic` to label the report; the label is supplied by the user and is not independently verified. Each run creates a unique subdirectory to preserve earlier artifacts. It stores the best validation checkpoint, dataset-quality JSON and Markdown, experiment-summary JSON, and an experiment report. The best checkpoint is reloaded before held-out test evaluation. The LSTM and persistence baseline use the same test sequences.

Training seeds Python, NumPy, and PyTorch and supplies an explicit seed to the shuffled training sampler. It temporarily enables PyTorch deterministic algorithms in warning mode by default and restores prior backend settings after training. `--strict-determinism` makes unsupported nondeterministic operations fail. These settings improve repeatability for supported operations on the same software/hardware setup; they do not promise bit-for-bit agreement across devices, library versions, or platforms.

## Metrics and split policy

- **MSE in degrees squared** is coordinate-space mean squared error over latitude/longitude outputs. It is not a geographic distance.
- **ADE in kilometres** is the mean Haversine distance across every sequence and forecast step.
- **FDE in kilometres** is the mean Haversine distance at each sequence’s final forecast step.

Evaluation rejects non-finite or out-of-bounds GPS targets and model predictions with contextual errors. Predictions are not clamped; failed geographic evaluation stops before writing a completed experiment summary or report.

Observations are sorted and split chronologically per individual before windows are generated. No input/target pair crosses a boundary and no observation is shared between partitions. Consequently, test inputs are drawn entirely from the test partition; the evaluation does not borrow preceding validation observations to form test histories. This is a stricter evaluation and can change comparability with protocols that use historical context from before the test boundary.

## Limitations

No real Movebank dataset is checked into this repository. Synthetic tests validate the implementation, but do not establish real-world model performance. Run validation and training on an identified real dataset before drawing performance conclusions. Results depend on sampling cadence and data quality; the current model consumes raw degree coordinates and does not model irregular time gaps.
