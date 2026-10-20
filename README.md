# Migration Intelligence - Phase 2 pipeline interfaces

The project provides a scientific baseline for GPS trajectory forecasting and local scriptable stage interfaces: dataset checks, chronological sequence construction, persisted split artifacts, an LSTM, a persistence baseline, and geographic evaluation. DVC orchestrates the local validation, preparation, and training stages. MLflow records each executed training run locally; no hosted tracking service or credentials are required.

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
The legacy validation CLI accepts `--max-animals` and defaults to five. The shared pipeline configuration defaults to ten, matching training; use `python -m src.pipeline validate` to validate with the shared settings.

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

Prepared-data manifest version 2 records the source CSV SHA-256 digest and each split archive's SHA-256 digest and byte size, without GPS rows or animal identifiers. Loading verifies checksums, shapes, dtypes, coordinate bounds, counts, and configuration. Older manifests without integrity metadata are rejected; rerun validation and preparation to regenerate them. Preparation builds and validates a complete candidate directory before publishing it by directory rename. The previous generation is retained as a hidden sibling for manual recovery; failed publication rolls back without mixing generations.

Canonical training validates `dataset_quality.json` before training and writes to a unique sibling candidate directory. It publishes canonical results only after evaluation, report validation, and MLflow logging succeed. At the start of an attempted canonical run, an existing result directory is moved to a hidden `.previous-*` sibling; if training, evaluation, MLflow logging, or publication fails, the canonical path remains absent rather than showing stale results as the latest run. The legacy standalone output remains a unique per-run directory.

### Programmatic prepared-data generation

`save_prepared_sequences(output_dir, splits, cfg, test_size, val_size, random_state=42, *, source_csv=None, source_fingerprint=None, quality_report=None)` requires either `source_csv` or a `source_fingerprint` dictionary containing `{"algorithm": "sha256", "digest": "<64 lowercase hex characters>"}`. A source fingerprint is mandatory and is written into the manifest. For prepared data that will be used by training, also pass an aggregate `quality_report` produced by `validate_dataset` with the same CSV, trajectory configuration, and split settings. That report must be usable and include the required aggregate counts and selection fields; training rejects a prepared directory that lacks a valid `dataset_quality.json`.

Migrate callers of the former positional form `save_prepared_sequences(out, splits, cfg, test_size, val_size, random_state)` by adding the source and report keywords:

```python
from src.data.data_prep import save_prepared_sequences
from src.data.validation import validate_dataset

quality = validate_dataset(
    csv_path, cfg, val_size=val_size, test_size=test_size,
    max_animals=cfg.max_animals,
)
save_prepared_sequences(
    out, splits, cfg, test_size, val_size, random_state,
    source_csv=csv_path, quality_report=quality,
)
```

If the CSV is not available, supply a previously verified SHA-256 `source_fingerprint` instead. The prior positional arguments remain in the same order; callers must add one of these provenance keywords.

## Metrics and split policy

- **MSE in degrees squared** is coordinate-space mean squared error over latitude/longitude outputs. It is not a geographic distance.
- **ADE in kilometres** is the mean Haversine distance across every sequence and forecast step.
- **FDE in kilometres** is the mean Haversine distance at each sequence’s final forecast step.

Evaluation rejects non-finite or out-of-bounds GPS targets and model predictions with contextual errors. Predictions are not clamped; failed geographic evaluation stops before writing a completed experiment summary or report.

Observations are sorted and split chronologically per individual before windows are generated. No input/target pair crosses a boundary and no observation is shared between partitions. Consequently, test inputs are drawn entirely from the test partition; the evaluation does not borrow preceding validation observations to form test histories. This is a stricter evaluation and can change comparability with protocols that use historical context from before the test boundary.

## Limitations

No real Movebank dataset is checked into this repository. Synthetic tests validate the implementation, but do not establish real-world model performance. Run validation and training on an identified real dataset before drawing performance conclusions. Results depend on sampling cadence and data quality; the current model consumes raw degree coordinates and does not model irregular time gaps.

## Reproducible pipeline interfaces

`configs/pipeline.json` is the shared pipeline configuration. Relative data and output paths are resolved from the repository root, independent of the shell's current directory. Defaults preserve Phase 1 model and split settings; the shared animal limit is 10, matching the training CLI. Edit this JSON or pass stage-specific path overrides. Supported settings are:

- `data.raw_path`, `data.max_animals`, `data.validation_report`, `data.validation_summary`, `data.prepared_dir`
- `trajectory.input_length`, `trajectory.forecast_horizon`, `trajectory.min_points_per_track`, `trajectory.val_size`, `trajectory.test_size`
- `training.seed`, `batch_size`, `epochs`, `learning_rate`, `hidden_size`, `num_layers`, `dropout`, `num_workers`, `device`, `data_kind`, `strict_determinism`, `evaluation_dir`, and `diagnostics_dir`
- `mlflow.tracking_uri` (local SQLite URI such as `sqlite:///mlruns/mlflow.db`, or a local database path) and `mlflow.experiment_name`. `file:` tracking URIs and hosted tracking URIs are rejected.

Run from the repository root with the existing environment:

```powershell
.\.venv\Scripts\python.exe -m src.pipeline validate --config configs/pipeline.json
.\.venv\Scripts\python.exe -m src.pipeline prepare --config configs/pipeline.json
.\.venv\Scripts\python.exe -m src.pipeline train --config configs/pipeline.json
```

To invoke the script from another working directory, use absolute paths for the interpreter and script; relative configuration and output paths still resolve from the repository root:

```powershell
D:\migration-intelligence\.venv\Scripts\python.exe D:\migration-intelligence\src\pipeline.py validate --config configs\pipeline.json
```

The explicit defaults can be overridden per stage, for example `--data-path C:\datasets\movebank.csv`, `--output .\artifacts\quality.json` on `validate`; `--output-dir .\data\processed\prepared` on `prepare`; and `--prepared-dir .\data\processed\prepared --output-dir .\artifacts\pipeline\evaluation` on `train`. All relative overrides remain repository-root-relative. Each command returns a nonzero status on failure. Validation and preparation report a missing raw CSV without printing its contents.

Preparation writes `train.npz`, `validation.npz`, `test.npz`, and `manifest.json` to `data.prepared_dir`. Each compressed NPZ contains only `past` and `future` float32 arrays, with shapes `(sequence_count, input_length, 2)` and `(sequence_count, forecast_horizon, 2)` and feature order latitude then longitude. Loading uses `allow_pickle=False` and checks shapes, dtypes, finite/ranged coordinates, manifest counts, and configuration. The manifest contains split settings and aggregate sequence metadata, not GPS rows or animal identifiers. NPZ members use fixed ZIP timestamps so identical arrays produce stable files. Preparation also writes the aggregate validation report beside the arrays for convenient lineage.

The train stage consumes these prepared files without opening or preprocessing the raw CSV. It writes canonical pipeline outputs to `artifacts/pipeline/evaluation` (checkpoint, JSON summary, and Markdown report). Runtime timings are written separately to `artifacts/pipeline/diagnostics/runtime_diagnostics.json`; they are excluded from canonical summaries. Repeated canonical CPU training with identical prepared files and settings produced byte-identical checkpoints in the regression test for the installed software environment. This is not a cross-version or cross-platform guarantee. The legacy `python -m src.models.train --data-path ...` workflow remains available and continues to create unique per-run directories under `artifacts/experiments`, retaining elapsed-time fields in its experiment reports.

Training uses a local SQLite backend by default at `mlruns/mlflow.db`, resolved from the repository root for pipeline runs and standalone CLI defaults. Experiment artifacts are stored in the local `mlruns/artifacts/` directory. The experiment name defaults to `migration-intelligence`. Override `mlflow.tracking_uri` and `mlflow.experiment_name` in `configs/pipeline.json`; standalone training also accepts `--mlflow-tracking-uri` and `--mlflow-experiment-name`. Tracking accepts local SQLite database paths or SQLite URIs and rejects filesystem tracking URIs and hosted services. MLflow records actual training settings and model configuration, per-epoch training and validation MSE, the best validation loss, and explicitly named held-out test metrics (including ADE/FDE when evaluation completes). It logs the selected checkpoint and existing summary/report files, but no row-level GPS data or dataset-quality report. MLflow is optional at runtime for backward compatibility: install `requirements.txt` to enable tracking; without MLflow, training still runs and emits a warning.

Install the declared dependencies in the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then run `dvc repro` to execute and track the pipeline. A DVC cache hit skips training, so it does not create an MLflow run; use `dvc repro --force train` when you intentionally want to execute training again. To run the legacy standalone interface with local tracking:

```powershell
.\.venv\Scripts\python.exe -m src.models.train --data-path .\data\raw\movebank.csv --device cpu --mlflow-tracking-uri .\mlruns --mlflow-experiment-name migration-intelligence
```

Open the local SQLite-backed experiment UI in a second terminal:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000
```

Visit `http://127.0.0.1:5000`. DVC remains responsible for stage dependencies, cache, and declared pipeline outputs; MLflow records parameters, metrics, and useful artifacts for executions that actually run. Their local stores are independent, and no DVC remote or hosted MLflow service is configured.

The pipeline stages are validation (raw CSV to quality JSON/Markdown), preparation (raw CSV and validation report to split arrays, manifest, and copied quality report), and training/evaluation (prepared arrays to checkpoint and evaluation report). Pipeline validation adds a streaming SHA-256 fingerprint of the source CSV to the aggregate report. Preparation compares the current CSV fingerprint, filename, trajectory settings, and split ratios before reusing a report; an old report without a fingerprint is rejected. When run by itself without a report, preparation validates the CSV in memory and writes its quality report only inside the prepared output directory. DVC also tracks the CSV and validation JSON dependency, so it schedules validation before preparation when the raw file changes.

## Local DVC workflow

DVC 3.x is declared in `requirements.txt`. DVC 3.67.1 successfully parsed this repository's graph with `dvc dag` during the latest audit, using a temporary DVC cache/config location to avoid a machine-level cache permission error. That graph check does not verify stage execution; a full `dvc repro` was not run during that audit. Use the project environment from the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
dvc dag
dvc repro
dvc status
Get-Content .\artifacts\pipeline\evaluation\experiment_summary.json
```

The graph runs `validate`, `prepare`, then `train`. DVC tracks the stage definitions and parameters in `dvc.yaml` and `dvc.lock`; it caches declared outputs in the local `.dvc/cache`. `.dvc/config` disables analytics, update checks, and automatic Git staging. No DVC remote is configured, so the cache is local to this checkout and is not a backup or a transfer mechanism. Runtime diagnostics under `artifacts/pipeline/diagnostics` are deliberately excluded from canonical DVC outputs; the checkpoint, summary, and report are declared outputs. Checkpoint byte stability is limited to the tested software environment described above.

To track a supplied raw CSV with DVC, run `dvc add data/raw/movebank.csv`. This creates `data/raw/movebank.csv.dvc`; the CSV itself stays ignored by Git. Keep DVC metadata and pointer files under Git using your normal manual Git workflow, for example `.dvc/config`, `.dvc/.gitignore`, `dvc.yaml`, `dvc.lock`, and the `.dvc` pointer file. Do not stage the raw CSV with Git. The ignore rules allow these pointer files without allowing generated data or pipeline outputs into Git. No Git commands are run by the pipeline or DVC configuration.

Edit `configs/pipeline.json` to change stage parameters, then run `dvc repro`; DVC will rerun affected downstream stages. Use `dvc status` to inspect changed dependencies and `dvc dag` to inspect stage order. The machine-readable evaluation metrics are in `artifacts/pipeline/evaluation/experiment_summary.json`. The graph has been parsed with DVC, but this does not establish that the current configuration can reproduce stages; full reproduction requires a supplied raw CSV and has not been repeated as part of the latest audit.
