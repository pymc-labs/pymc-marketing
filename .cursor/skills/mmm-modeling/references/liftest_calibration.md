# Lift Test Calibration

## Table of Contents
- [Why Calibrate](#why-calibrate)
- [Lift Test Data Format](#lift-test-data-format)
- [Time-Varying Media and Date Alignment](#time-varying-media-and-date-alignment)
- [Adding Lift Tests to the Model](#adding-lift-tests-to-the-model)
- [Calibrated vs Uncalibrated Comparison](#calibrated-vs-uncalibrated-comparison)
- [How It Works Internally](#how-it-works-internally)
- [Geo-Level Lift Tests](#geo-level-lift-tests)
- [Practical Guidelines](#practical-guidelines)

## Why Calibrate

MMMs fit on observational data alone cannot fully resolve causal identification when:

- **Channels are correlated**: TV and social campaigns often run simultaneously, making it hard to attribute effects to individual channels.
- **Unobserved confounders exist**: Seasonality, competitor activity, or economic conditions affect both spend and sales, creating spurious associations.
- **The model is under-identified**: Multiple parameter combinations can explain the observed data equally well.

Lift tests provide **experimental evidence** -- a known change in spend (`delta_x`) and the measured causal effect (`delta_y`) -- which anchors the model's saturation curve to ground truth. This dramatically narrows posteriors and reduces bias.

## Lift Test Data Format

### National (Single-Geo) Lift Tests

```python
import pandas as pd

df_lift_test = pd.DataFrame({
    "channel": ["tv", "social"],
    "x": [500.0, 300.0],          # baseline spend level
    "delta_x": [100.0, 50.0],     # incremental spend change
    "delta_y": [20.0, 8.0],       # observed causal lift in target
    "sigma": [3.0, 2.0],          # measurement uncertainty
})
```

| Column | Description |
|--------|-------------|
| `channel` | Channel name (must match `channel_columns`) |
| `x` | Baseline spend at which the test was run |
| `delta_x` | Change in spend during the experiment |
| `delta_y` | Measured causal effect on the target variable |
| `sigma` | Uncertainty in `delta_y` (standard deviation) |

### Geo-Level Lift Tests

Add a `geo` column for multidimensional models:

```python
df_lift_test = pd.DataFrame({
    "channel": ["tv", "tv", "social"],
    "geo": ["geo_a", "geo_b", "geo_a"],
    "x": [500.0, 400.0, 300.0],
    "delta_x": [100.0, 80.0, 50.0],
    "delta_y": [20.0, 15.0, 8.0],
    "sigma": [3.0, 4.0, 2.0],
})
```

## Time-Varying Media and Date Alignment

When `time_varying_media=True`, lift test rows should include `date` in addition to the standard columns (`channel`, `x`, `delta_x`, `delta_y`, `sigma`).

Why this matters:

- With static media effects, the predicted lift is `sat(x + delta_x) - sat(x)`.
- With time-varying media, the predicted lift is scaled by a time-specific multiplier:
  `media_temporal_latent_multiplier[t] * (sat(x + delta_x) - sat(x))`.
- The same `x` and `delta_x` can imply different expected `delta_y` at different dates, so calibration rows must be tied to the correct time coordinate.

Use this pattern:

```python
df_lift_test = pd.DataFrame({
    "channel": ["tv"],
    "date": [pd.Timestamp("2024-06-03")],  # required for time-varying media
    "x": [500.0],
    "delta_x": [100.0],
    "delta_y": [20.0],
    "sigma": [3.0],
})
```

For multidimensional models, include one column per dimension as well (for example `geo`).

Legacy note: older MMM behavior may auto-fill missing `date` with the first model date for `time_varying_media=True`. Avoid relying on this fallback; pass explicit experiment dates.

## Adding Lift Tests to the Model

Lift tests are added **after** `build_model()` and **before** `fit()`:

```python
# 1. Build the model graph
mmm.build_model(X, y)

# 2. Add lift test measurements
mmm.add_lift_test_measurements(df_lift_test)

# 3. Fit
mmm.fit(X=X, y=y, nuts_sampler="nutpie", target_accept=0.9, random_seed=rng)
```

### Multiple Lift Test Groups

You can add lift tests from different experiments as separate groups:

```python
mmm.build_model(X, y)
mmm.add_lift_test_measurements(df_lift_test_q1)
mmm.add_lift_test_measurements(df_lift_test_q2, name="q2_lift_measurements")
mmm.fit(X=X, y=y, ...)
```

Each call adds a distinct likelihood term, so multiple experiments are jointly used for calibration.

## Calibrated vs Uncalibrated Comparison

Always compare a calibrated model against an uncalibrated baseline to quantify the lift test's impact:

```python
import arviz as az
from pymc_marketing.mmm.multidimensional import MMM

# Uncalibrated model
mmm_uncalibrated = MMM(
    date_column="date",
    channel_columns=channel_columns,
    target_column="y",
    adstock=adstock,
    saturation=saturation,
    model_config=model_config,
)
mmm_uncalibrated.build_model(X, y)
mmm_uncalibrated.fit(X=X, y=y, nuts_sampler="nutpie", target_accept=0.9, random_seed=rng)

# Calibrated model
mmm_calibrated = MMM(
    date_column="date",
    channel_columns=channel_columns,
    target_column="y",
    adstock=adstock,
    saturation=saturation,
    model_config=model_config,
)
mmm_calibrated.build_model(X, y)
mmm_calibrated.add_lift_test_measurements(df_lift_test)
mmm_calibrated.fit(X=X, y=y, nuts_sampler="nutpie", target_accept=0.9, random_seed=rng)
```

### Compare Posteriors

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

az.plot_forest(
    [mmm_uncalibrated.idata, mmm_calibrated.idata],
    var_names=["saturation_beta"],
    combined=True,
    model_names=["Uncalibrated", "Calibrated"],
    ax=axes[0],
)
axes[0].set_title("Saturation Beta")

az.plot_forest(
    [mmm_uncalibrated.idata, mmm_calibrated.idata],
    var_names=["adstock_alpha"],
    combined=True,
    model_names=["Uncalibrated", "Calibrated"],
    ax=axes[1],
)
axes[1].set_title("Adstock Alpha")
```

**What to expect**: The calibrated model should have narrower posteriors (less uncertainty) and parameter values closer to the true causal effects. If the posteriors shift substantially, the observational data alone was biased.

## How It Works Internally

The `add_lift_test_measurements()` method adds a Gamma likelihood that constrains the saturation curve:

```
saturation(x + delta_x) - saturation(x) ~ Gamma(mu=delta_y, sigma=sigma)
```

This means: the model's predicted incremental effect of changing spend from `x` to `x + delta_x` must be consistent with the experimentally observed lift `delta_y`, within uncertainty `sigma`.

The Gamma distribution is used because lift effects are non-negative (more spend should not decrease the target).

## Geo-Level Lift Tests

### Hierarchical Information Flow

In multidimensional models with partial pooling, calibrating **some** geos also improves estimates for **uncalibrated** geos through shared hyperpriors:

```
Treated geo (has lift test) → updates geo-level parameter
→ updates shared hyperprior (mu, sigma)
→ improves control geo estimates (through the hyperprior)
```

This means you do not need lift tests for every geo. Running experiments in a subset of geos provides information that propagates to all geos via the hierarchical structure.

### Sigma Scaling for Geo-Level Tests

Larger geos have more stable measurements. Scale sigma inversely with geo size:

```python
import numpy as np

median_sales = data_df.groupby("geo")["y"].median().median()
geo_sales = data_df.groupby("geo")["y"].median()

base_sigma = 3.0
df_lift_test["sigma"] = base_sigma * np.sqrt(median_sales / geo_sales[df_lift_test["geo"]].values)
```

This gives larger geos tighter uncertainty (smaller sigma) and smaller geos wider uncertainty, reflecting that larger geos provide more reliable experimental estimates.

## Cost-Per-Target Calibration

An alternative to lift test calibration is `add_cost_per_target_calibration`, which constrains the model's cost-per-target (CPT) ratio via an **observed Normal likelihood**:

```python
import pandas as pd

calibration_data = pd.DataFrame({
    "channel": ["tv", "social"],
    "cost_per_target": [25.0, 12.0],   # desired CPT
    "sigma": [5.0, 3.0],               # tolerance (larger = weaker constraint)
})

mmm.build_model(X, y)
mmm.add_cost_per_target_calibration(
    data=X,                        # feature DataFrame with spend in original units
    calibration_data=calibration_data,
)
mmm.fit(X=X, y=y, nuts_sampler="nutpie", target_accept=0.9, random_seed=rng)
```

Internally, for each calibration row the method computes `mean(spend) / mean(contribution)` over the date dimension and adds:

```
Normal(mu=cpt_mean, sigma=sigma, observed=cost_per_target)
```

This constrains the model's CPT ratio toward the calibration target within the specified uncertainty. Useful when you have cost-efficiency benchmarks but not full lift test data.

**Prerequisite**: `channel_contribution_original_scale` must exist in the model graph. Call `add_original_scale_contribution_variable` before `add_cost_per_target_calibration` if it is not already present.

The `calibration_data` DataFrame requires columns: `channel`, `cost_per_target`, `sigma`, plus one column per model dimension (e.g., `geo`).

### Geo-Level CPT Calibration

For multidimensional models, include dimension columns:

```python
calibration_data = pd.DataFrame({
    "channel": ["tv", "tv", "social"],
    "geo": ["US", "UK", "US"],
    "cost_per_target": [25.0, 30.0, 12.0],
    "sigma": [5.0, 6.0, 3.0],
})
```

## Practical Guidelines

### Estimating `delta_y`

| Source | Method |
|--------|--------|
| **Randomized experiment** | Treatment mean - Control mean |
| **Synthetic control** | Use CausalPy or similar to estimate the counterfactual; `delta_y = observed - counterfactual` |
| **Geo-lift test** | Difference-in-differences or synthetic control across treated/control geos |
| **Incrementality test** | Hold-out region with zero spend; `delta_y = other_geos_mean - holdout_mean` |

### Estimating `sigma`

| Approach | Formula |
|----------|---------|
| **Standard error** | `sigma = se(delta_y)` from the experiment |
| **Conservative** | `sigma = 0.2 * abs(delta_y)` (20% relative uncertainty) |
| **From confidence interval** | `sigma = (upper - lower) / (2 * 1.96)` for 95% CI |

### When to Use Lift Tests

| Scenario | Recommendation |
|----------|----------------|
| Channels highly correlated | Lift tests essential -- observational data alone cannot separate effects |
| Suspected confounders | Lift tests reduce bias from unobserved variables |
| New channel with no history | Even a single lift test provides a strong anchor |
| Model posteriors very wide | Lift tests narrow uncertainty by adding information |
| Prior and posterior nearly identical | Data is not informative; lift tests break the tie |
| Time-varying media models | Include `date` so lift constraints align to the right latent multiplier |

### When Not to Use Lift Tests

- If the experiment was poorly designed (high contamination, short duration)
- If `sigma` is very large relative to `delta_y` (the test is uninformative)
- If the experiment was run at a very different spend level than the model's operating range

---

## See Also

- [model_specification.md](model_specification.md) -- Model setup (build_model must be called before adding lift tests)
- [model_fit.md](model_fit.md) -- Fitting and diagnostics after calibration
- [media_deep_dive.md](media_deep_dive.md) -- Compare calibrated vs uncalibrated ROAS
