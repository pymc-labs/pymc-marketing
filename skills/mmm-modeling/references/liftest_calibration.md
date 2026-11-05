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
from pymc_marketing.mmm.mmm import MMM

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

`add_lift_test_measurements()` adds one observed likelihood term per row, centred on the model's incremental effect of moving spend from `x` to `x + delta_x`:

```
delta_y ~ Normal(mu=saturation(x + delta_x) - saturation(x), sigma=sigma)
```

The observation is the signed `delta_y` as measured, `sigma` comes from the `sigma` column, and `Normal` is the default (`likelihood=None` resolves to `Prior("Normal")`). Parameter draws whose implied lift sits far from the experiment score badly, so the posterior is pulled toward it; the sales likelihood is still in play and `sigma` sets the exchange rate. A negative `delta_y` (a spend decrease that lowered the target, or a null result) is accepted as is.

The family is a `Prior` passed as `likelihood`, and the alternatives are opt-in:

```python
from pymc_extras.prior import Prior

mmm.add_lift_test_measurements(df_lift_test, likelihood=Prior("StudentT", nu=4))  # heavier tails
mmm.add_lift_test_measurements(df_lift_test, likelihood=Prior("Gamma"))           # positive support
```

- `StudentT` needs `nu`; it lets the model walk away from an experiment that may itself be wrong.
- `Gamma` is only valid when every `delta_y` and `delta_x` in the table is strictly positive, and the model-implied lift must be positive as well: every free variable upstream of it needs a positive-support prior and the lift must be positive at the initial point. A spend decrease or a null result is refused under `Gamma` and accepted under the default `Normal`.
- `StudentT`'s `nu` is the only parameter the `Prior` takes: the model, the `sigma` column and the rows decide the others, so they, `dims` (also inside `nu`) and `centered=False` are refused, as is a `nu` that is a non-positive number or has a prior that can go negative (`Normal`, `StudentT`, a `Uniform` or `TruncatedNormal` with a lower bound below zero, ...). A `transform` only renames the observed variable to `{name}_raw`. A `Prior`-valued `nu` adds a free variable named `{name}_nu`.

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

`add_cost_per_target_calibration` calibrates a whole-period ratio instead of a point on the saturation curve: cost per target (CPA, cost per conversion) or, with `target_per_cost=True`, target per cost (ROAS). The model side is `mean(spend) / mean(contribution)` over the fitted dates, and each row of the table becomes an observed likelihood term on that ratio, the same mechanism as a lift test with less information (one number per channel, averaged over the whole period):

```python
import pandas as pd
from pymc_extras.prior import Prior

cpa_from_lift_test = pd.DataFrame({
    "channel": ["tv"],
    "cost_per_target": [25.0],          # CPA read off a lift test
    "sigma": [5.0],                     # its standard error, in the same units
})
cpa_relative = pd.DataFrame({
    "channel": ["social"],
    "cost_per_target": [12.0],          # CPA reported as 12 +/- 25% (relative)
    "sigma": [3.0],
})

mmm.build_model(X, y)
# A CPA from a lift test has its error on the ROAS scale: invert it and keep
# the default Normal, the experiment's sampling model.
cpa = cpa_from_lift_test["cost_per_target"]
mmm.add_cost_per_target_calibration(
    data=X,                        # spend in original units, same layout as X
    calibration_data=cpa_from_lift_test.assign(
        roas=1 / cpa, sigma=cpa_from_lift_test["sigma"] / cpa**2
    ),
    name_prefix="tv_roas",
    target_column="roas",
    target_per_cost=True,
)
# A CPA with a relative interval: measure the error on the log scale. Under
# the default Normal a CPA table is capped where an overstated channel goes.
mmm.add_cost_per_target_calibration(
    data=X,
    calibration_data=cpa_relative,
    name_prefix="social_cpa",
    likelihood=Prior("LogNormal"),
)
mmm.fit(X=X, y=y, nuts_sampler="nutpie", target_accept=0.9, random_seed=rng)
```

The `calibration_data` DataFrame requires `channel`, the value column (`cost_per_target` by default, or `target_column`), `sigma`, plus one column per model dimension (e.g., `geo`). `sigma` is the standard error of the value in the table's units under every family.

`likelihood` picks the family of the term; `Normal` is the default. A Normal on a ratio is capped on one side: its penalty tends to `t**2 / (2 * sigma**2)` as the ratio goes to zero (8 nats at `sigma = t / 4`), so a CPA table is a floor on ROAS but not a ceiling, and a ROAS table is the reverse. `Prior("LogNormal")` measures the error on the log scale (`mu = log(ratio)`, `sigma / target` as the log-scale width), so a 2x miss costs the same in either direction and the penalty keeps growing; under it the table direction only decides how the table is written. `Prior("StudentT", nu=...)` caps lower on purpose, for an experiment that may itself be wrong.

Pick the family by the error the experiment reports:

| Calibration value | Family |
|---|---|
| ROAS read off a lift test (Normal standard error on the incremental outcome, known spend) | `Normal` on a ROAS table (`target_per_cost=True`): it is the sampling model, and it pulls harder than `LogNormal` against an overstated channel |
| CPA from a lift test | Invert it to a ROAS table (`roas = 1 / cpa`, `sigma_roas = sigma_cpa / cpa**2`) and use the row above. Exact for the delta-method standard error of `spend / delta_y`, close while `sigma_cpa / cpa` is about 0.3 or less; a CPA given as bootstrap replicates or posterior draws is inverted draw by draw (mean and SD of `1 / draws`) |
| CPA with a relative, asymmetric or log-scale interval, or of unknown provenance | `LogNormal`: the CPA-table `Normal` is capped on the side an overstated channel moves toward. Inverting first changes nothing under `LogNormal` (the two terms differ by a constant) |
| Other value reported with a relative, asymmetric or log-scale interval | `LogNormal`; a log-scale standard error `s` goes in as `sigma = s * target` |
| Experiment you only partly trust | `StudentT` |

`LogNormal`'s `sigma / target` width translates an additive standard error to first order, which is accurate while `sigma / target` is about 0.3 or less; at 0.5 it allows ratios from 0.38x to 2.66x the target where the Normal allows 0.02x to 1.98x. It requires positive values, a positive mean spend and mean contribution for every calibrated row at the initial point (a channel with no spend over the dates is refused), and contributions that stay positive while sampling (positive-support channel priors); past a `1e-12` floor a negative contribution gets a large finite penalty with no gradient. The family is saved with the model and restored on load.

Use the mean scale (`central_tendency="mean"`, the default) when the values come from an experiment; see the [ROAS calibration notebook](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas_calibration.html) for the full argument, including a comparison of the families on a confounded dataset.

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
