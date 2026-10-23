# PR: Add ExperimentDesigner for posterior-aware lift test design

Closes #2355

## Issue Summary

Add an `ExperimentDesigner` to PyMC-Marketing that recommends which marketing experiment to run (which channel, at what spend level, for how long) based on a fitted MMM's posterior uncertainty about channel response functions. The v1 scope is national-level experiments analysed via Interrupted Time Series (ITS).

## Root Cause

No existing tool makes upstream experiment design decisions (which channel to test, at what spend level) based on model uncertainty. GeoLift requires the user to guess the expected effect size; the `ExperimentDesigner` replaces that guess with the posterior distribution of predicted lift.

## Solution

Implemented a complete `experiment_design` subpackage within `pymc_marketing.mmm` containing:

- **`ExperimentDesigner`** class with two constructors: `__init__(mmm)` for fitted MMM objects and `from_idata()` for saved InferenceData fixtures
- **`ExperimentRecommendation`** dataclass with all metrics (lift, HDI, SNR, assurance, ramp fraction, cost, score, rationale)
- **Adstock-aware lift prediction** accounting for geometric adstock ramp-up over experiment duration
- **Posterior-predictive power (Bayesian assurance)** — expected power averaged over the posterior
- **Weighted composite scoring** with configurable weights across 5 dimensions (uncertainty, correlation, gradient, assurance, cost efficiency)
- **5 plotting methods**: channel diagnostics, power-cost scatter, lift distributions, saturation curves, adstock ramp
- **Fixture generator** with both fast synthetic posteriors and full MCMC fitting

## Changes Made

- `pymc_marketing/mmm/experiment_design/__init__.py`: Public API exports
- `pymc_marketing/mmm/experiment_design/designer.py`: Main `ExperimentDesigner` class with `recommend()`, scoring, and plotting
- `pymc_marketing/mmm/experiment_design/recommendation.py`: `ExperimentRecommendation` dataclass and rationale template
- `pymc_marketing/mmm/experiment_design/functions.py`: Lightweight numpy `logistic_saturation` function
- `pymc_marketing/mmm/experiment_design/fixture.py`: `generate_experiment_fixture()` utility
- `pymc_marketing/mmm/__init__.py`: Added `ExperimentDesigner` and `ExperimentRecommendation` to package exports
- `tests/mmm/test_experiment_design/test_designer.py`: 33 tests covering init, adstock ramp, lift prediction, assurance, recommend, scoring, plotting
- `tests/mmm/test_experiment_design/test_functions.py`: 9 tests for response function correctness and broadcasting
- `tests/mmm/test_experiment_design/test_fixture.py`: 14 tests for fixture generation and `from_idata()` round-trip

## Testing

- [x] All 65 new tests pass
- [x] Pre-commit (ruff, mypy, formatting) all pass
- [x] Existing tests unaffected

## Notes

- v1 supports `LogisticSaturation` + `GeometricAdstock` with `adstock_first=True` only
- The `.nc` fixture file is not included — `generate_experiment_fixture(fit_model=True)` can create one offline
- Slow simulation-based calibration tests (assurance calibration check) are deferred — the framework supports them but they require fitting an actual MMM
- v2 scope (geo designs, pulse/switchback, Fisher Information, Pareto frontier) is documented in the issue
