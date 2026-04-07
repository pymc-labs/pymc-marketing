# Model Specification

## Table of Contents
- [MMM Constructor Reference](#mmm-constructor-reference)
- [Adstock Transformations](#adstock-transformations)
- [Saturation Transformations](#saturation-transformations)
- [Prior Specification with Prior](#prior-specification-with-prior)
- [Model Config Dictionary](#model-config-dictionary)
- [Multidimensional / Hierarchical Models](#multidimensional--hierarchical-models)
- [Scaling Configuration](#scaling-configuration)
- [Prior Predictive Checks](#prior-predictive-checks)
- [Building the Model](#building-the-model)

## MMM Constructor Reference

```python
from pymc_marketing.mmm.multidimensional import MMM

mmm = MMM(
    date_column="date",
    channel_columns=["tv", "radio", "social"],
    target_column="y",
    adstock=GeometricAdstock(l_max=6),
    saturation=LogisticSaturation(),
    dims=None,                              # tuple[str, ...] for multidimensional
    scaling=None,                           # dict or Scaling object
    model_config=None,                      # dict of Prior objects
    sampler_config=None,                    # dict of sampler kwargs
    control_columns=None,                   # list[str] for control variables
    yearly_seasonality=None,                # int: number of Fourier terms
    adstock_first=True,                     # adstock before saturation
    time_varying_intercept=False,           # bool or HSGPBase
    time_varying_media=False,               # bool or HSGPBase
    cost_per_unit=None,                     # pd.DataFrame for monetary→model unit conversion
    dag=None,                               # causal graph string (DOT format)
    treatment_nodes=None,                   # list[str] of treatment nodes in the DAG
    outcome_node=None,                      # outcome node name in the DAG
)
```

**Reserved dim names**: `date`, `channel`, `control`, `fourier_mode` are used internally and cannot be passed as custom `dims`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `date_column` | `str` | Name of the date column |
| `channel_columns` | `list[str]` | Media channel column names |
| `target_column` | `str` | Response variable name (default `"y"`) |
| `adstock` | `AdstockTransformation` | Adstock component (e.g., `GeometricAdstock`) |
| `saturation` | `SaturationTransformation` | Saturation component (e.g., `LogisticSaturation`) |
| `dims` | `tuple[str, ...] \| None` | Dimensions for multidimensional models (e.g., `("geo",)`) |
| `scaling` | `dict \| Scaling \| None` | How to scale channel/target data |
| `model_config` | `dict \| None` | Prior specification for model parameters |
| `control_columns` | `list[str] \| None` | Non-media predictor columns |
| `yearly_seasonality` | `int \| None` | Number of Fourier pairs for yearly seasonality |
| `time_varying_intercept` | `bool \| HSGPBase` | Gaussian process intercept |
| `time_varying_media` | `bool \| HSGPBase` | Gaussian process media effects |
| `adstock_first` | `bool` | Apply adstock before saturation (default `True`) |
| `cost_per_unit` | `pd.DataFrame \| None` | Maps monetary spend to model units (impressions, clicks). Also settable post-fit via `mmm.set_cost_per_unit()` |
| `dag` | `str \| None` | Causal DAG in DOT format for `CausalGraphModel` integration |
| `treatment_nodes` | `list[str] \| None` | Treatment nodes in the causal DAG |
| `outcome_node` | `str \| None` | Outcome node in the causal DAG |

## Adstock Transformations

All adstock classes share the constructor `(l_max, normalize=True, mode=ConvMode.After, priors=None, prefix=None)`. The `mode` parameter controls alignment: `After` (default, causal), `Before`, or `Overlap`.

**Best practice**: set adstock priors directly via the `priors` argument on the transformation object rather than through `model_config`. This keeps transformation-specific priors co-located with the component they belong to. Use `model_config` only for non-component parameters (intercept, controls, seasonality, likelihood).

### GeometricAdstock

Exponential decay carryover effect. The most common choice:

```python
from pymc_marketing.mmm import GeometricAdstock

adstock = GeometricAdstock(
    l_max=6,           # maximum lag (weeks)
    normalize=True,    # normalize weights to sum to 1
)
```

Default prior: `alpha ~ Beta(1, 3)` -- favors fast decay.

Custom priors per channel:

```python
from pymc_extras.prior import Prior

adstock = GeometricAdstock(
    l_max=6,
    priors={"alpha": Prior("Beta", alpha=2, beta=5, dims="channel")},
)
```

For multidimensional models with per-channel-per-geo priors:

```python
adstock = GeometricAdstock(
    l_max=6,
    priors={"alpha": Prior("Beta", alpha=2, beta=5, dims=("channel", "geo"))},
)
```

**Rule of thumb for `l_max`**: Set to the maximum plausible carryover duration. For weekly data, `l_max=6` (6 weeks) is typical. For daily data, use `l_max=14` or higher. Too large wastes computation; too small truncates real effects.

### DelayedAdstock

Adds a `theta` parameter for peak delay on top of geometric decay. Useful when the effect peaks days/weeks after exposure:

```python
from pymc_marketing.mmm import DelayedAdstock

adstock = DelayedAdstock(l_max=10)
```

Default priors: `alpha ~ Beta(1, 3)`, `theta ~ HalfNormal(sigma=1)`.

### Other Adstock Types

| Class | Lookup | Default Priors | Notes |
|-------|--------|----------------|-------|
| `BinomialAdstock` | `"binomial"` | `alpha ~ Beta(1, 3)` | Binomial-based decay |
| `WeibullPDFAdstock` | `"weibull_pdf"` | `lam ~ Gamma(mu=2, sigma=1)`, `k ~ Gamma(mu=3, sigma=1)` | Flexible shape via Weibull PDF |
| `WeibullCDFAdstock` | `"weibull_cdf"` | `lam ~ Gamma(mu=2, sigma=2.5)`, `k ~ Gamma(mu=2, sigma=2.5)` | Flexible shape via Weibull CDF |
| `NoAdstock` | `"no_adstock"` | (none) | Bypasses adstock entirely |

All are imported from `pymc_marketing.mmm` and use the `adstock_from_dict()` factory for deserialization.

## Saturation Transformations

**Best practice**: set saturation priors directly via the `priors` argument on the transformation object. The `beta` parameter on saturation classes controls the maximum reachable effect per channel -- setting it proportional to spend shares is a common, effective pattern.

### LogisticSaturation

S-shaped saturation curve (diminishing returns):

```python
from pymc_marketing.mmm import LogisticSaturation

saturation = LogisticSaturation()
```

Default priors: `lam ~ Gamma(3, 1)`, `beta ~ HalfNormal(2)`.

Custom priors:

```python
saturation = LogisticSaturation(
    priors={
        "lam": Prior("Gamma", alpha=3, beta=1, dims="channel"),
        "beta": Prior("HalfNormal", sigma=spend_shares, dims="channel"),
    },
)
```

The `beta` parameter controls the maximum reachable effect per channel. Setting its prior proportional to spend shares encodes the belief that channels with higher spend should have proportionally larger effects.

### Alternative Saturation Functions

| Class | Description | Default Priors |
|-------|-------------|----------------|
| `LogisticSaturation` | S-shaped logistic curve (most common) | `lam ~ Gamma(3, 1)`, `beta ~ HalfNormal(2)` |
| `InverseScaledLogisticSaturation` | Inverse-scaled logistic | `lam ~ Gamma(0.5, 1)`, `beta ~ HalfNormal(2)` |
| `MichaelisMentenSaturation` | Enzyme kinetics-inspired curve | `alpha ~ Gamma(mu=2, sigma=1)`, `lam ~ HalfNormal(1)` |
| `HillSaturation` | Generalized Hill function | `slope ~ HalfNormal(1.5)`, `kappa ~ HalfNormal(1.5)`, `beta ~ HalfNormal(1.5)` |
| `HillSaturationSigmoid` | Sigmoid variant of Hill | `sigma ~ HalfNormal(1.5)`, `beta ~ HalfNormal(1.5)`, `lam ~ HalfNormal(1.5)` |
| `TanhSaturation` | Hyperbolic tangent | `b ~ HalfNormal(1)`, `c ~ HalfNormal(1)` |
| `TanhSaturationBaselined` | Baselined tanh with offset | `x0 ~ HalfNormal(1)`, `gain ~ HalfNormal(1)`, `r ~ HalfNormal(1)`, `beta ~ HalfNormal(1)` |
| `RootSaturation` | Power-law (root) transformation | `alpha ~ Beta(1, 2)`, `beta ~ Gamma(mu=1, sigma=1)` |
| `NoSaturation` | Identity (no saturation, scaling only) | `beta ~ HalfNormal(1)` |

```python
from pymc_marketing.mmm import (
    HillSaturation,
    HillSaturationSigmoid,
    LogisticSaturation,
    MichaelisMentenSaturation,
    RootSaturation,
    TanhSaturation,
)

saturation = MichaelisMentenSaturation()
saturation = HillSaturation()
```

All saturation functions share the same interface: pass to the `MMM` constructor via the `saturation` parameter, and customize priors via the `priors` kwarg.

## Prior Specification with Prior

All priors in `model_config` and component `priors` use `pymc_extras.prior.Prior`:

```python
from pymc_extras.prior import Prior

# Simple prior
Prior("Normal", mu=0, sigma=1)

# Prior with dims
Prior("HalfNormal", sigma=0.5, dims="channel")

# Prior with multiple dims
Prior("Beta", alpha=2, beta=5, dims=("channel", "geo"))

# Hierarchical (non-centered) prior -- uses LogNormalPrior from special_priors
from pymc_marketing.special_priors import LogNormalPrior

LogNormalPrior(
    mean=Prior("Gamma", mu=1.0, sigma=1.0),
    std=Prior("HalfNormal", sigma=1.0),
    dims=("channel", "geo"),
    centered=False,
)
```

The `dims` argument determines broadcasting across model dimensions and controls pooling behavior in hierarchical models.

## Model Config Dictionary

The `model_config` dictionary is for **non-component** priors (intercept, controls, seasonality, likelihood). Adstock and saturation priors should be set directly on the transformation objects via their `priors` argument (see above).

```python
from pymc_extras.prior import Prior

model_config = {
    # Intercept
    "intercept": Prior("Normal", mu=0.2, sigma=0.05),

    # Control variable coefficients
    "gamma_control": Prior("Normal", mu=0, sigma=1, dims="control"),

    # Fourier seasonality coefficients
    "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),

    # Likelihood (default is Normal; TruncatedNormal is a common custom choice for non-negative targets)
    "likelihood": Prior("TruncatedNormal", lower=0, sigma=Prior("HalfNormal", sigma=1)),
}
```

| Config Key | Description | Default | Typical Custom Prior |
|------------|-------------|---------|----------------------|
| `intercept` | Baseline response | `Normal(mu=0, sigma=2, dims=self.dims)` | `Normal(mu=0.2, sigma=0.05)` |
| `gamma_control` | Control coefficients | `Normal(mu=0, sigma=2, dims=(*dims, "control"))` | `Normal(0, 1, dims="control")` |
| `gamma_fourier` | Seasonality coefficients | `Laplace(mu=0, b=1, dims=(*dims, "fourier_mode"))` | `Laplace(0, 1)` |
| `likelihood` | Observation noise | `Normal(sigma=HalfNormal(sigma=2))` | `TruncatedNormal(lower=0, sigma=HalfNormal(1))` |

Note: `saturation_beta` can still be set via `model_config` for backward compatibility, but placing it on the saturation object via `priors={"beta": ...}` is preferred.

## Multidimensional / Hierarchical Models

Activate multidimensional modeling with `dims=("geo",)`:

```python
mmm = MMM(
    date_column="date",
    channel_columns=channel_columns,
    target_column="y",
    adstock=adstock,
    saturation=saturation,
    dims=("geo",),
    scaling=scaling_config,
    model_config=model_config,
)
```

### Pooling Strategies

**Partial pooling** (recommended) -- parameters vary by geo but share hyperpriors:

```python
from pymc_marketing.special_priors import LogNormalPrior

model_config = {
    "saturation_beta": LogNormalPrior(
        mean=Prior("Gamma", mu=1.0, sigma=1.0),
        std=Prior("HalfNormal", sigma=1.0),
        dims=("channel", "geo"),
        centered=False,
    ),
}
```

**Full pooling** -- same parameters across all geos:

```python
model_config = {
    "saturation_beta": Prior("HalfNormal", sigma=spend_shares, dims="channel"),
}
```

**No pooling** -- independent parameters per geo (requires substantial data per geo):

```python
model_config = {
    "saturation_beta": Prior("HalfNormal", sigma=0.5, dims=("geo", "channel")),
}
```

### Which Pooling to Choose

| Strategy | When to Use |
|----------|-------------|
| Partial pooling | Default choice. Geos share information but can differ. Best when some geos have sparse data. |
| Full pooling | All geos are believed to behave identically (rare in practice). Simplest model. |
| No pooling | Each geo has abundant data and behaviors genuinely differ. Risk of overfitting with sparse data. |

## Scaling Configuration

For multidimensional models, control whether scaling is computed globally or per geo:

```python
scaling_config = {
    "channel": {"method": "max", "dims": ()},    # global max across geos
    "target": {"method": "max", "dims": ()},      # global max across geos
}

# Alternatively, scale per geo:
scaling_config = {
    "channel": {"method": "max", "dims": ("geo",)},
    "target": {"method": "max", "dims": ("geo",)},
}
```

Global scaling (empty `dims`) is generally preferred -- it keeps channels comparable across geos.

Pre-built scaling classes:

```python
from pymc_marketing.mmm.preprocessing import MaxAbsScaleChannels, MaxAbsScaleTarget
```

## Prior Predictive Checks

Always run prior predictive checks before fitting to verify priors produce plausible predictions:

```python
mmm.sample_prior_predictive(X, y, samples=4_000, random_seed=42)
```

### Visualize Prior Predictions

```python
import arviz as az
import matplotlib.pyplot as plt

prior_pred = mmm.idata["prior_predictive"]["y"]

fig, ax = plt.subplots()
az.plot_hdi(
    X[date_column],
    prior_pred,
    hdi_prob=0.94,
    color="C0", fill_kwargs={"alpha": 0.3},
    ax=ax,
)
ax.plot(X[date_column], y, color="black")
ax.set_title("Prior Predictive Check")
```

### Prior Channel Contribution Share

Check that prior contributions are not concentrated on a single channel. **Note**: `add_original_scale_contribution_variable` must be called before `sample_prior_predictive` for `*_original_scale` variables to be available:

```python
mmm.add_original_scale_contribution_variable(
    var=["channel_contribution", "control_contribution",
         "intercept_contribution", "yearly_seasonality_contribution", "y"]
)
mmm.sample_prior_predictive(X, y, samples=4_000, random_seed=42)

prior_contrib = mmm.idata["prior_predictive"]["channel_contribution_original_scale"]
total = prior_contrib.sum(dim=["date", "channel"])

shares = prior_contrib.sum(dim="date") / total

az.plot_forest(shares, combined=True)
```

If one channel dominates in the prior, widen or equalize the `saturation_beta` priors.

## Building the Model

Build the model graph without fitting to inspect the structure:

```python
mmm.build_model(X, y)
```

### Inspecting the Model

```python
mmm.graphviz()          # DAG visualization (requires graphviz)
mmm.table()             # summary table of all variables, dims, and expressions
```

`mmm.table()` produces a Rich `Table` showing every variable's name, distribution, shape, and dimensions -- useful for verifying the model was constructed as intended.

After building, you can also call `mmm.add_lift_test_measurements(...)` before fitting to incorporate experimental calibration data. See [liftest_calibration.md](liftest_calibration.md).

---

## See Also

- [data_analysis.md](data_analysis.md) -- Data preparation and EDA
- [model_fit.md](model_fit.md) -- Fitting and diagnostics
- [liftest_calibration.md](liftest_calibration.md) -- Lift test calibration (applied between build and fit)
