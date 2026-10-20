# Responsible use and data governance

## Appropriate and inappropriate use

This repository provides a trajectory forecasting research baseline. Forecasts
are estimates from past coordinate sequences. Do not use them as observed animal
locations or as the sole basis for conservation, ecological, animal-welfare,
enforcement, or other consequential decisions. A qualified domain user should
review inputs, outputs, and limitations.

## Geographic, temporal, and sampling limits

The model consumes raw latitude/longitude pairs and a fixed number of
observations. It receives no timestamps or elapsed intervals. Irregular sampling
can therefore make an identical observation horizon represent different real
durations. Missing observations are excluded; the pipeline does not impute them.
The repository does not establish geographic coverage, sampling bias, seasonal
coverage, or population representativeness. Behavior outside the training
distribution and geographic extrapolation are unvalidated.

## Data provenance, rights, and privacy

The expected input is described as a Movebank-style CSV, but no particular
study, source account, dataset version, collection protocol, license, access
restriction, or attribution requirement is verified in this repository. Check
the terms attached to the exact data source before access, analysis, sharing,
or publication; this document makes no legal or licensing determination.

Animal tracking coordinates can reveal sensitive locations. Restrict access to
raw files and checkpoints, store them only in approved locations, avoid putting
coordinates or identifiers in logs and issue reports, and use aggregate
artifacts where possible. Dataset-quality reports and prepared manifests are
designed to omit raw records and animal identifiers, but model forecasts and
explanation JSON can still contain sensitive location information. Treat them
accordingly and prefer synthetic inputs for demonstrations.

## Monitoring and explanation

Serving drift summaries are bounded operational indicators based on scalar
trajectory summaries. A drift alert is not proof of quality degradation, data
error, or ecological change; a stable status is not proof of validity. Forecast
quality metrics require explicitly supplied aligned ground truth and do not
create a trusted baseline by themselves.

The sensitivity utility varies one input coordinate at one timestep by a
specified degree amount and measures displacement between corresponding model
forecasts with Haversine distance. It is local to that input and perturbation.
It does not establish causality, feature importance independent of the chosen
perturbation, or model uncertainty. Equal degree changes in latitude and
longitude do not represent equal ground distances at all locations. The model
does not encode timestamps, so timestep indices do not imply elapsed time.
For input `x`, a selected timestep/coordinate index `i`, forecast step `h`,
and perturbation `δ`, the reported value is `d_h = Haversine(f(x)_h,
f(x + δ e_i)_h)` in kilometres. The per-perturbation mean is the arithmetic
mean of `d_h` over forecast steps. Values can be compared only with their input,
location, coordinate, and δ in view; this is not a global feature-importance
ranking.

## Reproducibility and traceability

Record the source dataset identity/version and terms outside sensitive logs,
retain its content fingerprint and configuration lineage, preserve prepared
artifact manifests, and track model checkpoints with their configuration and
evaluation artifacts. DVC tracks pipeline inputs and outputs; local MLflow
records training runs. Neither tool proves that data rights, source attribution,
or scientific assumptions are correct. A user-supplied `real` label is not
independent evidence that a dataset is real.

## Conditions for real-world claims

Before presenting performance or scientific conclusions, establish the exact
data source and permissions, characterize missingness and sampling, review
geographic and temporal coverage, validate a suitable held-out evaluation
design, compare against persistence, and disclose limitations and uncertainty
methods actually evaluated. This repository currently lacks the real CSV and a
canonical trained checkpoint, so it supports no real-world accuracy or drift
claim.
