---
name: pymc-marketing-mmm
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

# Always import MMM from multidimensional (the newer unified class).
# `from pymc_marketing.mmm import MMM` resolves to the legacy class with
# a different constructor (no dims, no target_column, no cost_per_unit).
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
mmm.fit(X=X, y=y, target_accept=0.9, random_seed=42)
mmm.sample_posterior_predictive(X=X, random_seed=42)
```

## Model Specification

The `MMM` class is the central entry point. **Reserved dim names** (`date`, `channel`, `control`, `fourier_mode`) cannot be used as custom dims:

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
    adstock_first=True,                   # adstock before saturation (default)
    cost_per_unit=None,                   # pd.DataFrame to convert monetary to model units
)
```

### Adstock Transformations

All adstock classes share the interface `(l_max, normalize=True, mode=ConvMode.After, priors=None)`:

| Class | Key Params | Use Case |
|-------|-----------|----------|
| `GeometricAdstock` | `alpha` (decay rate) | Default choice -- exponential decay |
| `DelayedAdstock` | `alpha` (decay), `theta` (peak delay) | Channels with delayed peak effect |
| `WeibullPDFAdstock` | `lam`, `k` | Flexible shape via Weibull PDF |
| `WeibullCDFAdstock` | `lam`, `k` | Flexible shape via Weibull CDF |
| `BinomialAdstock` | `alpha` | Binomial-based decay |
| `NoAdstock` | (none) | Bypass adstock entirely |

### Saturation Transformations

| Class | Key Params | Use Case |
|-------|-----------|----------|
| `LogisticSaturation` | `lam`, `beta` | Default -- S-shaped diminishing returns |
| `HillSaturation` | `slope`, `kappa`, `beta` | Generalized Hill function |
| `MichaelisMentenSaturation` | `alpha`, `lam` | Enzyme kinetics-inspired curve |
| `TanhSaturation` | `b`, `c` | Hyperbolic tangent |
| `TanhSaturationBaselined` | `x0`, `gain`, `r`, `beta` | Tanh with offset |
| `HillSaturationSigmoid` | `sigma`, `beta`, `lam` | Sigmoid Hill variant |
| `InverseScaledLogisticSaturation` | `lam`, `beta` | Inverse-scaled logistic |
| `RootSaturation` | `alpha`, `beta` | Power-law transformation |
| `NoSaturation` | `beta` | Scaling only, no saturation |

All are imported from `pymc_marketing.mmm` and accept custom `priors` dicts.

### Priors

**Best practice**: set adstock and saturation priors directly on the transformation objects via their `priors` argument. This keeps transformation-specific priors co-located with the component and is the recommended approach:

```python
adstock = GeometricAdstock(
    l_max=6,
    priors={"alpha": Prior("Beta", alpha=2, beta=5, dims="channel")},
)
saturation = LogisticSaturation(
    priors={
        "lam": Prior("Gamma", alpha=3, beta=1, dims="channel"),
        "beta": Prior("HalfNormal", sigma=spend_shares, dims="channel"),
    },
)
```

Use `model_config` for non-component priors (intercept, controls, seasonality, likelihood):

```python
model_config = {
    "intercept": Prior("Normal", mu=0.2, sigma=0.05),
    "gamma_control": Prior("Normal", mu=0, sigma=1, dims="control"),
    "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
    "likelihood": Prior("TruncatedNormal", lower=0, sigma=Prior("HalfNormal", sigma=1)),
}
```

See [references/model_specification.md](references/model_specification.md) for full constructor reference (including `dag`/`treatment_nodes`/`outcome_node` for causal graph integration), hierarchical prior patterns (partial/full/no pooling), adstock/saturation default priors, and prior predictive checks.

## Prior Predictive Checks

Sample from the prior predictive distribution **after** `build_model` and **before** `fit` to verify that priors produce plausible target ranges:

```python
mmm.build_model(X, y)

# Returns an xr.Dataset (az.extract output), not InferenceData
prior_samples = mmm.sample_prior_predictive(X=X, random_seed=42)
prior_samples["y"]  # access from returned Dataset

# Or equivalently, access from idata (populated via extend_idata=True default)
mmm.idata.prior_predictive["y"]
```

**Common pitfall**: The prior predictive variable is always `"y"` because `MMM.output_var = "y"` is hardcoded as a class attribute (the PyMC likelihood variable name). It does **not** follow `target_column`. Using the target column name (e.g., `mmm.idata.prior_predictive["revenue"]`) will raise a `KeyError`.

Plot with the built-in method:

```python
mmm.plot.prior_predictive(var="y", hdi_prob=0.85)
```

## Data Preparation

Data must have a date column, one or more channel (spend) columns, and a target column. For multidimensional models, include a geo/region column and provide data in long format.

**Critical**: The final model is always fit on the **full dataset**. Train/test splits are only used for model stability assessment via time-slice cross-validation.

See [references/data_analysis.md](references/data_analysis.md) for EDA patterns, data format requirements, and spend share computation.

## Inference and Diagnostics

Always fit on the full dataset, then run diagnostics:

```python
mmm.fit(X=X, y=y, target_accept=0.9, chains=6, draws=800,
        tune=1_500, random_seed=rng)
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
results = cv.run(
    X, y, sampler_config={...},
    yaml_path=...,                        # or pass mmm=mmm_instance directly
    df_lift_test=None,                    # optional lift test for each fold
    original_scale_vars=None,             # variables to rescale
    return_models=False,                  # True to get fitted models per fold
)
mmm.plot.cv_predictions(results)
mmm.plot.param_stability(results, parameter=["adstock_alpha"], dims={...})
mmm.plot.cv_crps(results)
```

See [references/model_fit.md](references/model_fit.md) for the full diagnostics checklist, time-slice CV workflow, and common fit issues.

## Model Variables and Original Scale

`build_model` creates variables on the **scaled** space. The key model variables are:

| Variable | Present when |
|----------|-------------|
| `channel_contribution` | Always |
| `intercept_contribution` | Always |
| `control_contribution` | `control_columns` set |
| `yearly_seasonality_contribution` | `yearly_seasonality` set |
| `baseline_channel_contribution` | `time_varying_media=True` |
| `y` | Always (predicted target) |

`total_media_contribution_original_scale` is created automatically. All other `*_original_scale` variants require an explicit call to `add_original_scale_contribution_variable` **after** `build_model` and **before** `sample_posterior_predictive` or `sample_prior_predictive`:

```python
mmm.build_model(X, y)

mmm.add_original_scale_contribution_variable(
    var=["channel_contribution", "control_contribution",
         "intercept_contribution", "yearly_seasonality_contribution", "y"]
)

mmm.fit(X=X, y=y, target_accept=0.9, random_seed=rng)
mmm.sample_posterior_predictive(X=X, random_seed=rng)
```

This multiplies each variable by `target_scale` and registers it as `{var}_original_scale` (e.g. `channel_contribution_original_scale`). Without this call, plotting and analysis methods that reference `*_original_scale` names will fail.

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
    var_input="channel_data", sweep_values=sweeps,
    var_names="channel_contribution_original_scale", extend_idata=True,
)
mmm.plot.sensitivity_analysis(hue_dim="channel", x_sweep_axis="relative")
```

See [references/media_deep_dive.md](references/media_deep_dive.md) for ROAS computation, incremental analysis, saturation/adstock curves, sensitivity analysis, and contribution share plots.

## Time-Varying Parameters

The `MMM` class supports GP-based time-varying intercept and time-varying media multiplier via Hilbert Space Gaussian Processes (HSGP).

**Option A** -- `bool` + `HSGPKwargs` in `model_config` (simple):

```python
from pymc_marketing.hsgp_kwargs import HSGPKwargs

mmm = MMM(
    ...,
    time_varying_intercept=True,
    time_varying_media=True,
    model_config={
        "intercept_tvp_config": HSGPKwargs(
            m=500, L=188, eta_lam=5.0, ls_mu=5.0, ls_sigma=10.0,
        ),
        "media_tvp_config": HSGPKwargs(ls_mu=11.0, ls_sigma=5.0),
    },
)
```

**Option B** -- pass an `HSGPBase` instance directly (full control over dims and priors):

```python
from pymc_marketing.mmm.hsgp import HSGPBase

mmm = MMM(
    ...,
    time_varying_intercept=HSGPBase(m=500, ...),
    time_varying_media=HSGPBase(m=200, ...),
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
    adstocked = adstock.apply(channel_data_, dims=("geo", "channel"))
    channel_contribution = saturation.apply(adstocked, dims=("geo", "channel"))
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
    budget=budget_per_period,
    budget_bounds=budget_bounds_xr,
    budgets_to_optimize=None,             # xr.DataArray mask to fix certain channels
    budget_distribution_over_period=None,  # temporal distribution (flighting)
    cost_per_unit=None,                   # convert monetary to model units
    constraints=(),                       # custom Constraint objects
    minimize_kwargs={"method": "SLSQP", "options": {"ftol": 1e-4, "maxiter": 10_000}},
)

response = optimizable_model.sample_response_distribution(
    allocation_strategy=allocation,
    additional_var_names=["channel_contribution_original_scale"],
    include_last_observations=False, include_carryover=True,
    noise_level=0.05,
)
optimizable_model.plot.budget_allocation(samples=response)
```

For impression-based models, use `mmm.set_cost_per_unit(cost_df)` after fitting or pass `cost_per_unit` to the constructor to convert monetary budgets to model units.

See [references/budget_optimization.md](references/budget_optimization.md) for budget bounds setup, custom constraints, budget sweeps, channel fixing, and temporal distribution.

## Lift Test Calibration

Lift tests resolve causal identification when channels are correlated:

```python
# After build_model, before fit
mmm.build_model(X, y)
mmm.add_lift_test_measurements(df_lift_test)
mmm.fit(X=X, y=y, ...)
```

The lift test DataFrame requires columns: `channel`, `x`, `delta_x`, `delta_y`, `sigma` (plus `geo` for geo-level).
When `time_varying_media=True`, include `date` so each lift measurement maps to the correct `media_temporal_latent_multiplier` time coordinate.

## Cost-Per-Target Calibration

When you have cost-efficiency benchmarks instead of full lift tests, use CPT calibration. It adds an observed Normal likelihood constraining `mean(spend) / mean(contribution)` toward a target CPT:

```python
calibration_data = pd.DataFrame({
    "channel": ["tv", "social"],
    "cost_per_target": [25.0, 12.0],
    "sigma": [5.0, 3.0],
    # include one column per model dim, e.g. "geo"
})

mmm.build_model(X, y)
mmm.add_cost_per_target_calibration(data=X, calibration_data=calibration_data)
mmm.fit(X=X, y=y, ...)
```

See [references/liftest_calibration.md](references/liftest_calibration.md) for lift test data format, CPT calibration details, calibrated vs uncalibrated comparison, geo-level patterns, and sigma estimation.

## Saving, Loading, and YAML Specification

```python
# Save / load fitted model
mmm.save("mmm_model.nc", engine="h5netcdf")
loaded_mmm = MMM.load("mmm_model.nc")

# Build model from a YAML specification
from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml

mmm = build_mmm_from_yaml(config_path="model_spec.yaml", X=X, y=y)
```

The YAML builder (`build_mmm_from_yaml`) enables declarative model specification -- useful for reproducible experiments, `TimeSliceCrossValidator` integration (via its `yaml_path` parameter), and MLflow tracking.

## Incremental Analysis and Summary

### `mmm.incrementality`

Counterfactual analysis with proper adstock carryover handling. Preferred approach for ROAS/CAC computation:

```python
roas = mmm.incrementality.contribution_over_spend(frequency="quarterly")
cac = mmm.incrementality.spend_over_contribution(frequency="quarterly")
marginal_roas = mmm.incrementality.marginal_contribution_over_spend(frequency="all_time")
```

See [references/media_deep_dive.md](references/media_deep_dive.md#incremental-analysis-mmmincrementality) for details.

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

Validated access to `idata` via `MMMIDataWrapper` with convenience methods:

```python
mmm.data.get_target()                    # observed target values
mmm.data.get_channel_contributions(original_scale=True)
mmm.data.get_contributions(original_scale=True, include_baseline=True)
mmm.data.get_channel_spend()             # channel spend in monetary units
mmm.data.get_target_scale()              # scaling factor used during fit
mmm.data.validate_or_raise()             # validate idata structure

# Chained filtering and aggregation
monthly = mmm.data.filter_dates("2024-01-01", "2024-12-31").aggregate_time("monthly")
```

## Event Effects

Add holiday or campaign effects before building the model via the `MuEffect` system:

```python
from pymc_marketing.mmm.events import EventEffect

df_events = pd.DataFrame({
    "name": ["black_friday", "christmas"],
    "start_date": ["2024-11-29", "2024-12-25"],
    "end_date": ["2024-11-29", "2024-12-25"],
})

mmm.add_events(df_events, prefix="holiday", effect=EventEffect(dims=("holiday",)))
mmm.build_model(X, y)
```

Events are registered as `EventAdditiveEffect` instances in the model's `mu_effects` list. The `LinearTrendEffect` and `FourierEffect` also use this system. See [references/custom_model.md](references/custom_model.md) for advanced additive effects.

## Plotting Methods Quick Reference

All visualization methods are accessed via the `mmm.plot` namespace (matplotlib). For interactive Plotly plots, use `mmm.plot_interactive` (requires `pip install pymc-marketing[plotly]`).

See [references/plot.md](references/plot.md) for the complete API with exact signatures.

| Method | Description |
|--------|-------------|
| `prior_predictive(var=..., hdi_prob=0.85)` | Prior predictive HDI bands |
| `posterior_predictive(var=..., hdi_prob=0.85)` | Posterior predictive HDI bands |
| `residuals_over_time(hdi_prob=[0.94])` | Residuals with HDI bands (`hdi_prob` is `list[float]`) |
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
| `uplift_curve(hdi_prob=..., aggregation=...)` | Precomputed uplift curves |
| `marginal_curve(hdi_prob=..., aggregation=...)` | Precomputed marginal effects |
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
[Optional] Add Events (mmm.add_events, before build_model)
    ↓
Build Model (mmm.build_model)
    ↓
Prior Predictive Checks
    ↓
[Optional] Time-Slice Cross-Validation (stability assessment, before final fit)
    ↓
[Optional] Add Lift Test / Cost-Per-Target Calibration (after build, before fit)
    ↓
Fit on FULL Dataset (mmm.fit)
    ↓
Diagnostics (divergences, R-hat, ESS, trace plots)
    ↓
Posterior Predictive Checks
    ↓
Media Deep Dive (contributions, ROAS, saturation, sensitivity)
    ↓
Budget Optimization
```
