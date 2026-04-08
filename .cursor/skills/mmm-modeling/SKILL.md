---
name: mmm-modeling
description: >
  Media Mix Modeling with PyMC-Marketing. Use when building MMMs, specifying adstock/saturation
  transformations, setting priors, fitting multidimensional (geo-level) models, computing channel
  contributions, ROAS, running budget optimization, calibrating with lift tests, or performing
  sensitivity analysis. Covers the MMM class, GeometricAdstock, LogisticSaturation,
  MultiDimensionalBudgetOptimizerWrapper, and ArviZ diagnostics for marketing models.
---

# Media Mix Modeling with PyMC-Marketing

Bayesian Media Mix Modeling workflow using the PyMC-Marketing `MMM` class.

> **PyMC prerequisite:** This skill assumes familiarity with PyMC's core modeling API (coords/dims, priors, MCMC diagnostics, HSGP). For foundational patterns, see the [pymc-modeling skill](https://github.com/pymc-labs/python-analytics-skills/tree/main/skills/pymc-modeling).

LLMs understand Bayesian inference, MCMC, and hierarchical models in general. But getting from those concepts to a correctly specified, well-diagnosed, and actionable PyMC-Marketing MMM requires domain-specific knowledge: which `Prior` to use for saturation beta informed by spend shares, how `dims=("geo",)` activates multidimensional partial pooling, why the final model must be fit on the **full dataset** (time-slice CV is only for stability assessment), how `add_lift_test_measurements()` resolves causal identification, and how `MultiDimensionalBudgetOptimizerWrapper` translates posterior uncertainty into optimal allocations.

This skill encodes those patterns. Without it, an LLM might hold out test data for the final fit (wrong -- use all data, validate with time-slice CV), use flat priors on saturation parameters (causes divergences), skip `add_original_scale_contribution_variable` (then contributions are on scaled space), or call `BudgetOptimizer` directly instead of `MultiDimensionalBudgetOptimizerWrapper` (misses geo-level allocation).

## Quick Start

```python
import arviz as az
import numpy as np
import pandas as pd
from pymc_extras.prior import Prior

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM

# Load data
data_df = pd.read_csv("data.csv", parse_dates=["date"])
X = data_df.drop(columns=["y"])
y = data_df["y"]

# Specify model
mmm = MMM(
    date_column="date",
    channel_columns=["tv", "radio", "social"],
    target_column="y",
    adstock=GeometricAdstock(l_max=6),
    saturation=LogisticSaturation(),
    yearly_seasonality=5,
)

# Build and fit on FULL dataset
mmm.build_model(X, y)
mmm.fit(X=X, y=y, nuts_sampler="nutpie", target_accept=0.9, random_seed=42)
mmm.sample_posterior_predictive(X=X, random_seed=42)
```

## Model Specification

The `MMM` class is the central entry point:

```python
mmm = MMM(
    date_column="date",
    channel_columns=channel_columns,
    target_column="y",
    adstock=GeometricAdstock(l_max=6),
    saturation=LogisticSaturation(),
    dims=("geo",),                        # multidimensional
    scaling={"channel": {"method": "max", "dims": ()},
             "target": {"method": "max", "dims": ()}},
    model_config=model_config,
    control_columns=control_columns,
    yearly_seasonality=5,
    time_varying_intercept=False,
    time_varying_media=False,
)
```

Priors are customized via `model_config` using `pymc_extras.prior.Prior`:

```python
model_config = {
    "intercept": Prior("Normal", mu=0.2, sigma=0.05),
    "saturation_beta": Prior("HalfNormal", sigma=spend_shares, dims="channel"),
    "gamma_control": Prior("Normal", mu=0, sigma=1, dims="control"),
    "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
    "likelihood": Prior("TruncatedNormal", lower=0, sigma=Prior("HalfNormal", sigma=1)),
}
```

See [references/model_specification.md](references/model_specification.md) for full constructor reference, hierarchical prior patterns (partial/full/no pooling), and prior predictive checks.

## Data Preparation

Data must have a date column, one or more channel (spend) columns, and a target column. For multidimensional models, include a geo/region column and provide data in long format.

**Critical**: The final model is always fit on the **full dataset**. Train/test splits are only used for model stability assessment via time-slice cross-validation.

See [references/data_analysis.md](references/data_analysis.md) for EDA patterns, data format requirements, and spend share computation.

## Inference and Diagnostics

Always fit on the full dataset, then run diagnostics:

```python
mmm.fit(X=X, y=y, target_accept=0.9, chains=6, draws=800,
        tune=1_500, nuts_sampler="nutpie", random_seed=rng)
mmm.sample_posterior_predictive(X=X, random_seed=rng)

# Divergences (must be 0)
mmm.idata["sample_stats"]["diverging"].sum().item()

# R-hat (must be < 1.01)
az.summary(data=mmm.idata, var_names=[...])["r_hat"].describe()

# Trace plots
az.plot_trace(data=mmm.fit_result, var_names=[...], compact=True)
```

Use `TimeSliceCrossValidator` to assess model stability **before** the final fit:

```python
from pymc_marketing.mmm.time_slice_cross_validation import TimeSliceCrossValidator

cv = TimeSliceCrossValidator(n_init=163, forecast_horizon=12, date_column="date", step_size=1)
results = cv.run(X, y, sampler_config={...}, yaml_path=...)
cv.plot.param_stability(results, parameter=["adstock_alpha"], dims={...})
cv.plot.cv_predictions(results)
cv.plot.cv_crps(results)
```

See [references/model_fit.md](references/model_fit.md) for the full diagnostics checklist, time-slice CV workflow, and common fit issues.

## Media Analysis

After fitting, analyze channel contributions, ROAS, and saturation:

```python
# Waterfall decomposition
mmm.plot.waterfall_components_decomposition()

# Contributions over time
mmm.plot.contributions_over_time(
    var=["channel_contribution_original_scale",
         "control_contribution_original_scale",
         "intercept_contribution_original_scale"],
    combine_dims=True, hdi_prob=0.94,
)

# Incremental ROAS (preferred -- accounts for adstock carryover)
roas = mmm.incrementality.contribution_over_spend(frequency="all_time")
az.plot_forest(roas, combined=True)

# Saturation curves
mmm.plot.saturation_scatterplot(original_scale=True)

# Sensitivity analysis
sweeps = np.linspace(0, 1.5, 16)
mmm.sensitivity.run_sweep(
    sweep_values=sweeps, var_input="channel_data",
    var_names="channel_contribution_original_scale", extend_idata=True,
)
mmm.plot.sensitivity_analysis(hue_dim="channel", x_sweep_axis="relative")
```

See [references/media_deep_dive.md](references/media_deep_dive.md) for ROAS computation, incremental analysis, saturation/adstock curves, sensitivity analysis, and contribution share plots.

## Time-Varying Parameters

The `MMM` class supports GP-based time-varying intercept and time-varying media multiplier via Hilbert Space Gaussian Processes (HSGP):

```python
from pymc_marketing.hsgp_kwargs import HSGPKwargs

mmm = MMM(
    ...,
    time_varying_intercept=True,
    time_varying_media=True,
    model_config={
        "intercept_tvp_config": HSGPKwargs(m=500, L=188, eta_lam=5.0, ls_mu=5.0, ls_sigma=10.0),
        "media_tvp_config": HSGPKwargs(ls_mu=11.0, ls_sigma=5.0),
    },
)
```

Use TVP when residuals show **irregular, non-repeating** temporal variation not explained by seasonality, trend, or controls. The GP is primarily useful for in-sample decomposition; it reverts to the prior mean out of sample.

See [references/time_varying_parameters.md](references/time_varying_parameters.md) for `HSGPKwargs` reference, parameterization tips, diagnostics, and code examples.

## Custom Models

When the `MMM` class cannot express your model structure (non-standard hierarchies, spline baselines, custom likelihoods), build a custom model by combining PyMC-Marketing components with plain PyMC:

```python
import pymc as pm
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation

with pm.Model(coords=coords) as custom_mmm:
    channel_data_ = pm.Data("channel_data", channel_scaled, dims=("date", "geo", "channel"))
    adstocked = adstock.apply(channel_data_, core_dim="date")
    channel_contribution = saturation.apply(adstocked, core_dim="date")
    # ... add intercept, controls, seasonality, likelihood
```

Custom models gain full flexibility but lose built-in scaling, plotting, budget optimization, lift test integration, and save/load.

See [references/custom_model.md](references/custom_model.md) for standalone component usage, hierarchical prior patterns, spline-based intercepts, custom events, and a complete geo-hierarchical example.

## Budget Optimization

```python
from pymc_marketing.mmm.multidimensional import MultiDimensionalBudgetOptimizerWrapper

optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
    model=mmm, start_date=str(start_date), end_date=str(end_date),
)

allocation, result = optimizable_model.optimize_budget(
    budget=budget_per_period, budget_bounds=budget_bounds_xr,
    minimize_kwargs={"method": "SLSQP", "options": {"ftol": 1e-4, "maxiter": 10_000}},
)

response = optimizable_model.sample_response_distribution(
    allocation_strategy=allocation,
    include_last_observations=True, include_carryover=True,
)
optimizable_model.plot.budget_allocation(samples=response)
```

See [references/budget_optimization.md](references/budget_optimization.md) for budget bounds setup, custom constraints, budget sweeps, and channel fixing.

## Lift Test Calibration

Lift tests resolve causal identification when channels are correlated:

```python
# After build_model, before fit
mmm.build_model(X, y)
mmm.add_lift_test_measurements(df_lift_test)
mmm.fit(X=X, y=y, nuts_sampler="nutpie", ...)
```

The lift test DataFrame requires columns: `channel`, `x`, `delta_x`, `delta_y`, `sigma` (plus `geo` for geo-level).
When `time_varying_media=True`, include `date` so each lift measurement maps to the correct `media_temporal_latent_multiplier` time coordinate.

See [references/liftest_calibration.md](references/liftest_calibration.md) for data format, calibrated vs uncalibrated comparison, geo-level patterns, and sigma estimation.

## Saving, Loading, and YAML Specification

```python
# Save / load fitted model
mmm.save("mmm_model.nc", engine="h5netcdf")
loaded_mmm = MMM.load("mmm_model.nc")

# Build model from a YAML specification
from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml

mmm = build_mmm_from_yaml("model_spec.yaml", X=X, y=y)
```

The YAML builder (`build_mmm_from_yaml`) enables declarative model specification -- useful for reproducible experiments, `TimeSliceCrossValidator` integration (via `yaml_path`), and MLflow tracking.

## Incremental Analysis and Summary

### `mmm.incrementality`

Counterfactual analysis with proper adstock carryover handling. Preferred approach for ROAS/CAC computation:

```python
roas = mmm.incrementality.contribution_over_spend(frequency="quarterly")
cac = mmm.incrementality.spend_over_contribution(frequency="quarterly")
marginal_roas = mmm.incrementality.marginal_contribution_over_spend(frequency="all_time")
```

See [references/media_deep_dive.md](references/media_deep_dive.md#incremental-analysis-mmmincementality) for details.

### `mmm.summary`

DataFrame generation for key metrics:

```python
mmm.summary.posterior_predictive()       # mean, median, HDI, observed
mmm.summary.contributions()              # per-channel/control/seasonality contributions
mmm.summary.roas()                       # ROAS with HDI
mmm.summary.channel_spend()              # raw spend per channel/date
mmm.summary.saturation_curves()          # saturation response curves
mmm.summary.adstock_curves()             # adstock decay curves
mmm.summary.total_contribution()         # summed contributions by component type
mmm.summary.change_over_time()           # percentage change between periods
```

### `mmm.data`

Validated access to `idata` with convenience methods:

```python
mmm.data.get_target()                    # observed target values
mmm.data.get_contributions(original_scale=True)
mmm.data.filter_dates("2024-01-01", "2024-12-31")
```

## Plotting Methods Quick Reference

All visualization methods are accessed via the `mmm.plot` namespace. See [references/plot.md](references/plot.md) for the complete API with exact signatures.

| Method | Description |
|--------|-------------|
| `prior_predictive(var=..., hdi_prob=0.85)` | Prior predictive HDI bands |
| `posterior_predictive(var=..., hdi_prob=0.85)` | Posterior predictive HDI bands |
| `residuals_over_time(hdi_prob=...)` | Residuals with HDI bands |
| `residuals_posterior_distribution(aggregation=...)` | Residual distribution |
| `contributions_over_time(var=..., combine_dims=..., hdi_prob=...)` | Contributions over time with HDI |
| `waterfall_components_decomposition(split_by=...)` | Mean component decomposition (waterfall) |
| `channel_contribution_share_hdi(hdi_prob=0.94)` | Channel contribution shares (forest plot) |
| `posterior_distribution(var=..., plot_dim=...)` | Violin plots of a posterior variable |
| `channel_parameter(param_name=...)` | Posterior of a specific channel parameter |
| `prior_vs_posterior(var=..., plot_dim=...)` | Prior vs posterior KDE comparison |
| `saturation_scatterplot(original_scale=...)` | Observed spend vs. saturated effect |
| `saturation_curves(curve, original_scale=...)` | Smooth posterior saturation curves |
| `sensitivity_analysis(hue_dim=..., x_sweep_axis=...)` | Counterfactual response under spend scaling |
| `uplift_curve(hue_dim=...)` | Precomputed uplift curves |
| `marginal_curve(hue_dim=...)` | Precomputed marginal effects |
| `budget_allocation(samples=...)` | Optimal allocation summary |
| `allocated_contribution_by_channel_over_time(samples=...)` | Contributions under optimal allocation |
| `cv_predictions(results)` | Posterior predictive across CV folds |
| `param_stability(results, parameter=...)` | Parameter stability across CV folds |
| `cv_crps(results)` | CRPS scores across CV folds |

## Typical MMM Workflow

```
EDA & Data Prep
    ↓
Model Specification (priors, adstock, saturation, dims)
    ↓
Build Model (mmm.build_model)
    ↓
Prior Predictive Checks
    ↓
[Optional] Add Lift Test / Cost-Per-Target Calibration
    ↓
Fit on FULL Dataset (mmm.fit)
    ↓
Diagnostics (divergences, R-hat, ESS, trace plots)
    ↓
Posterior Predictive Checks
    ↓
Media Deep Dive (contributions, ROAS, saturation, sensitivity)
    ↓
[Optional] Time-Slice Cross-Validation (stability assessment)
    ↓
Budget Optimization
```
