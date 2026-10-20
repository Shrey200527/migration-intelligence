# Model card: trajectory LSTM

## Status and evidence

This card describes the implementation in this repository. It does not report
real-data performance: the real Movebank CSV and canonical trained checkpoint
are absent. Synthetic fixtures exercise software behavior only. Dataset origin,
collection protocol, licensing, access terms, and attribution requirements have
not been verified from authoritative dataset records and remain unknown.

## Intended purpose and users

The model maps a fixed window of past GPS coordinates to a fixed-length
coordinate forecast. It is a research and engineering baseline for people
evaluating trajectory forecasting workflows. Outputs are estimated locations,
not observations. They are not suitable as the sole basis for ecological,
conservation, animal-welfare, enforcement, or other consequential decisions.

## Inputs, outputs, and architecture

- Input: `(batch, input_length, 2)` raw coordinate values, ordered
  `[latitude, longitude]`, in degrees. The default configured input length is
  20 observations.
- Output: `(batch, forecast_horizon, 2)` predicted coordinates in the same
  order and units. The default horizon is 5 observations.
- The input has no timestamp, elapsed-time, altitude, or other feature. The
  model therefore cannot distinguish different time intervals between
  observations. The forecast horizon is measured in observations, not hours or
  days.
- `TrajectoryLSTM` is a batch-first PyTorch LSTM encoder. It uses the last
  timestep's LSTM output and a linear head to produce `forecast_horizon * 2`
  values. Defaults are input size 2, hidden size 128, two layers, and dropout
  0.2 between recurrent layers. These values are configurable.
- Coordinates are not normalized, standardized, wrapped, or clamped by the
  model. Serving rejects invalid inputs and rejects malformed, non-finite, or
  out-of-bounds outputs.

## Data handling and training

The pipeline expects a Movebank-style CSV with timestamp, individual
identifier, longitude, and latitude columns. Validation profiles malformed or
missing values, invalid timestamps and coordinates, and exact duplicate rows;
preprocessing excludes invalid records and removes exact duplicates. Training
uses latitude and longitude only. Data-quality reports contain aggregate
counts, not raw rows or identifiers. Prepared-data manifests store a source
SHA-256 fingerprint and split-file checksums, not GPS records or identifiers.

Observations are ordered chronologically within each individual. The timeline
is partitioned into train, validation, and test regions before windows are
created. A window never crosses a partition boundary, and an individual
contributes only when it can provide examples in all three partitions. Default
validation and test fractions are 0.15 each. This protocol does not use a
validation-period history to form test-period inputs.

Training defaults are seed 42, batch size 64, 10 epochs, Adam learning rate
0.001, hidden size 128, two LSTM layers, dropout 0.2, and zero data-loader
workers. Python, NumPy, and PyTorch are seeded; the shuffled loader receives an
explicit generator seed. PyTorch deterministic algorithms are requested in
warning mode by default, with an opt-in strict mode. Repeatability is not
guaranteed bit-for-bit across hardware, software versions, or unsupported
operations.

The checkpoint is selected by the lowest validation mean squared error (MSE)
over coordinate outputs, measured in degrees squared. The selected checkpoint
is reloaded before held-out test evaluation. A persistence baseline repeats
each test input's last coordinate for every forecast step and is evaluated on
the same test examples.

## Evaluation metrics

- Coordinate MSE averages squared latitude/longitude coordinate errors and is
  reported in degrees squared. It is not a geographic distance.
- Haversine distance uses a spherical Earth radius of 6371.0088 km.
- ADE is the mean Haversine distance over all sequences and forecast steps, in
  kilometres.
- FDE is the mean Haversine distance at the final forecast step for each
  sequence, in kilometres.
- Test metrics compare the selected LSTM checkpoint with persistence on the
  same held-out test sequences. Validation MSE selects a checkpoint; it is not
  a test metric.

No real-data metrics are currently available in this repository. Successful
synthetic tests demonstrate code behavior, not forecasting skill or biological
validity.

## Data provenance

The repository calls its expected input a Movebank-style GPS CSV and uses the
default path `data/raw/movebank.csv`. It does not contain that dataset or
records that identify a particular Movebank study, dataset version, account,
collection protocol, or download date. The `data_kind` run label is supplied
by the user and is not independently checked. A content fingerprint supports
file identity and lineage; it does not establish origin or permission to use.

Dataset license, attribution, access restrictions, consent or permits, and
retention requirements are unknown here and must be verified against the
specific dataset terms before use or distribution.

## Limitations and risks

- Geographic and temporal generalization are untested. Geographic clustering,
  sampling cadence, seasonal coverage, and study-population representativeness
  are unknown until the actual data and its collection documentation are
  reviewed.
- Missing and invalid observations are excluded rather than imputed. Irregular
  sampling intervals are not represented, even though they can change the
  meaning of a fixed observation horizon.
- Raw degree coordinates are not a local metric coordinate system. The model
  may produce invalid or implausible paths; predictions are not clamped.
- Longitude wraparound, polar geometry, coordinate noise, distribution shifts,
  and extrapolation can affect interpretation. Operational drift alerts do not
  establish model degradation or ecological change.
- Sensitivity results describe response to specified input perturbations.
  They do not establish causal mechanisms, calibrated uncertainty, or accuracy.
- The perturbation analysis changes one coordinate at one input index. It may
  create a geographically valid but physically implausible path; no timestamp
  or speed constraint is available to test that plausibility.
- Forecasts must not be treated as actual animal locations. Require domain
  review and independent validation before any use that could affect animals,
  habitat, conservation priorities, or people.

## Before making performance claims

Identify and document the exact dataset and its version, lawful access and
attribution terms, collection and sampling limits, and privacy controls. Run
the complete pipeline with that data, verify its validation report and
provenance, inspect temporal split suitability, and report held-out results
alongside persistence with enough context to assess limitations. Synthetic
tests and this repository's code alone do not meet these conditions.
