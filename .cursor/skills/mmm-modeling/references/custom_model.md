# Custom Models with PyMC and PyMC-Marketing Components

## Table of Contents
- [When to Go Custom](#when-to-go-custom)
- [Prerequisites](#prerequisites)
- [Using Components Standalone](#using-components-standalone)
- [Adding Parameter Dimensions](#adding-parameter-dimensions)
- [Assembling a Custom Model](#assembling-a-custom-model)
- [Example: Spline-Based Intercept](#example-spline-based-intercept)
- [Custom Events](#custom-events)
- [Trade-Offs vs the MMM Class](#trade-offs-vs-the-mmm-class)

## When to Go Custom

The `MMM` class handles common workflows (scaling, plotting, budget optimization, lift test calibration, save/load). Build a custom model only when the `MMM` class cannot express your model structure:

| Reason | Example |
|--------|---------|
| Non-standard hierarchical structure | Partial pooling across geo **and** product line simultaneously |
| Custom time-varying baseline | Spline-based intercept, changepoint models, or other non-GP approaches |
| Custom likelihood | Student-t, ZeroInflatedPoisson, or a mixture likelihood |
| Components not yet in `MMM` | Custom covariate transformations, non-linear interactions |
| Full control over parameterization | Manual centering, sum-to-zero constraints on seasonality |
| Multi-equation models | Linking MMM to a demand model or pricing model |

The key idea: **PyMC-Marketing components (adstock, saturation, Fourier) work inside any `pm.Model`**. You combine them with arbitrary PyMC code to build whatever structure you need. The examples below illustrate this general pattern.

If none of these reasons apply, prefer the `MMM` class -- you get built-in scaling, contribution plotting, budget optimization, lift test integration, and model serialization for free.

## Prerequisites

Custom models require fluency with PyMC's core modeling API: `pm.Model`, coords/dims, `pm.Data`, `pm.Deterministic`, prior specification, and MCMC diagnostics. For foundational PyMC patterns (coords/dims, parameterization, HSGP, diagnostics), see the [pymc-modeling skill](https://github.com/pymc-labs/python-analytics-skills/tree/main/skills/pymc-modeling).

## Using Components Standalone

PyMC-Marketing exposes adstock, saturation, and Fourier components as standalone objects. Each has `apply()`, `sample_prior()`, `sample_curve()`, and `plot_curve()` methods.

### Saturation

```python
from pymc_marketing import mmm

saturation = mmm.MichaelisMentenSaturation()
saturation.function_priors
# {'alpha': Prior("Gamma", mu=2, sigma=1), 'lam': Prior("HalfNormal", sigma=1)}

parameters = saturation.sample_prior(random_seed=rng)
curve = saturation.sample_curve(parameters, max_value=5)
saturation.plot_curve(curve)
```

### Adstock

```python
adstock = mmm.GeometricAdstock(l_max=10)
adstock.default_priors
# {'alpha': Prior("Beta", alpha=1, beta=3)}
```

### Fourier Seasonality

```python
yearly = mmm.YearlyFourier(n_order=2)
prior = yearly.sample_prior()
curve = yearly.sample_curve(prior)
yearly.plot_curve(curve)
```

### Applying inside a `pm.Model`

Each component's `.apply()` method registers its priors in the enclosing `pm.Model` context and returns a PyTensor expression:

```python
import pymc as pm

with pm.Model(coords=coords) as model:
    saturated_spends = saturation.apply(df_spends, dims="channel")
```

The posterior can be passed to `sample_curve` and `plot_curve` instead of the prior, and any additional coordinates from the parameters are handled automatically.

## Adding Parameter Dimensions

By default, component priors are scalar. To estimate separate parameters per channel (or per geo, etc.), set `dims` on each prior:

### Manual dim assignment

```python
for dist in saturation.function_priors.values():
    dist.dims = "channel"

# Now sample_prior needs coords
prior = saturation.sample_prior(coords=coords, random_seed=rng)
```

### Hierarchical priors on components

```python
from pymc_extras.prior import Prior

hierarchical_lam = Prior(
    "HalfNormal",
    sigma=Prior("HalfNormal", sigma=1),
    dims="channel",
)
common_alpha = Prior("Gamma", mu=2, sigma=1)

saturation = mmm.MichaelisMentenSaturation(
    priors={"lam": hierarchical_lam, "alpha": common_alpha}
)
```

### Multi-dimensional (geo x channel)

```python
hierarchical_alpha = Prior(
    "Gamma",
    mu=Prior("HalfNormal", sigma=1, dims="geo"),
    sigma=Prior("HalfNormal", sigma=1, dims="geo"),
    dims=("channel", "geo"),
)
common_lam = Prior("HalfNormal", sigma=1, dims="channel")

saturation = mmm.MichaelisMentenSaturation(
    priors={"alpha": hierarchical_alpha, "lam": common_lam}
)

with pm.Model(coords=geo_coords) as geo_model:
    geo_data = pm.Data("geo_data", geo_spends.to_numpy(), dims=("date", "channel", "geo"))
    saturated_geo_spends = pm.Deterministic(
        "saturated_geo_spends",
        saturation.apply(geo_data, dims=("channel", "geo")),
        dims=("date", "channel", "geo"),
    )
```

### Convenience helper

Use `set_dims_for_all_priors("channel")` to add a dimension to every prior at once (when available on the component).

## Assembling a Custom Model

A custom MMM follows this general pattern:

1. Define coords and create `pm.Model(coords=...)`.
2. Wrap observed data in `pm.Data(...)` with explicit dims.
3. Apply component `.apply()` calls for media and seasonality.
4. Build `mu` by combining components -- the structure is entirely up to you (additive, multiplicative, hierarchical, multi-equation, etc.).
5. Define a likelihood with `observed=target` -- any PyMC distribution works.

The example below shows a geo-hierarchical model, but the same principles apply to any custom structure. You can add or remove components, change the hierarchy, introduce interactions, use different likelihoods, or combine PyMC-Marketing components with arbitrary PyMC code.

### Geo-hierarchical example

```python
import pymc as pm
from pymc_extras.prior import Prior
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm import YearlyFourier

adstock = GeometricAdstock(
    l_max=8,
    priors={
        "alpha": Prior(
            "Beta",
            alpha=Prior("HalfNormal", sigma=8.0),
            beta=Prior("HalfNormal", sigma=8.0),
            dims=("geo", "channel"),
        ),
    },
)

saturation = LogisticSaturation(
    priors={
        "lam": Prior("Gamma", alpha=3, beta=1, dims="channel"),
        "beta": Prior("HalfNormal", sigma=3, dims="channel"),
    }
)

yearly_fourier = YearlyFourier(
    n_order=2,
    prefix="fourier_mode",
    variable_name="gamma_fourier",
    prior=Prior(
        "ZeroSumNormal",
        sigma=Prior("HalfNormal", sigma=0.1),
        dims=("geo", "fourier_mode"),
    ),
)

with pm.Model(coords=coords) as custom_mmm:
    channel_data_ = pm.Data(
        "channel_data", channel_scaled.transpose("date", "geo", "channel"),
        dims=("date", "geo", "channel"),
    )
    control_data_ = pm.Data(
        "control_data", control_data.transpose("date", "geo", "control"),
        dims=("date", "geo", "control"),
    )
    target_data_ = pm.Data(
        "target_data", target_scaled.transpose("date", "geo"),
        dims=("date", "geo"),
    )
    dayofyear_ = pm.Data(
        "dayofyear", pd.DatetimeIndex(date_index).dayofyear.values, dims="date"
    )

    # Intercept
    intercept = Prior("Normal", mu=2.5, sigma=0.25, dims="geo").create_variable("intercept")

    # Media: adstock -> saturation
    adstocked_media = adstock.apply(channel_data_, dims=("geo", "channel"))
    channel_contribution = pm.Deterministic(
        "channel_contribution",
        saturation.apply(adstocked_media, dims=("geo", "channel")),
        dims=("date", "geo", "channel"),
    )
    total_media = channel_contribution.sum(axis=-1)

    # Controls: hierarchical over geos
    gamma_control_mu = pm.Normal("gamma_control_mu", mu=0, sigma=1, dims="control")
    gamma_control_sigma = pm.HalfNormal("gamma_control_sigma", sigma=1, dims="control")
    gamma_control_offset = pm.Normal("gamma_control_offset", 0, 1, dims=("geo", "control"))
    gamma_control = pm.Deterministic(
        "gamma_control",
        gamma_control_mu + gamma_control_offset * gamma_control_sigma,
        dims=("geo", "control"),
    )
    total_control = (control_data_ * gamma_control).sum(axis=-1)

    # Seasonality
    yearly_contribution = yearly_fourier.apply(dayofyear_)

    mu = pm.Deterministic(
        "mu",
        intercept + total_media + total_control + yearly_contribution,
        dims=("date", "geo"),
    )

    sigma = pm.HalfNormal("y_sigma", sigma=1, dims="geo")
    pm.TruncatedNormal("y", mu=mu, sigma=sigma, lower=0,
                        observed=target_data_, dims=("date", "geo"))
```

### Fitting and diagnostics

```python
with custom_mmm:
    idata = pm.sample(
        chains=8, tune=2_000, draws=400,
        nuts_sampler="nutpie", target_accept=0.95, random_seed=rng,
    )
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

az.summary(idata, var_names=["gamma_control", "y_sigma", "saturation_lam"])
```

## Example: Spline-Based Intercept

The following illustrates one way to extend a custom model beyond what the `MMM` class supports -- a spline-based time-varying intercept. This is just **one example** of the kind of flexibility custom models provide. You could equally replace this with changepoint models, random-walk intercepts, or any other PyMC-compatible structure. The core pattern is the same: define your custom component as PyTensor math inside `pm.Model`, then add it to `mu`.

### Building the basis

```python
import numpy as np
from patsy import dmatrix

time_idx = np.arange(len(date_index), dtype=float)
n_knots = 4
knot_list = np.percentile(time_idx, np.linspace(0, 100, n_knots + 2))[1:-1]

spline_basis = np.asarray(
    dmatrix(
        "bs(t, knots=knots, degree=3, include_intercept=True) - 1",
        {"t": time_idx, "knots": knot_list},
    )
)
```

### Hierarchical spline weights

```python
import pytensor.tensor as pt

with pm.Model(coords=coords) as model:
    spline_basis_ = pm.Data("spline_basis", spline_basis, dims=("date", "spline"))

    spline_mu = pm.Laplace("spline_mu", mu=0, b=0.10, dims="spline")
    spline_sigma = pm.HalfNormal("spline_sigma", sigma=0.08, dims="spline")
    spline_offset = pm.Normal("spline_offset", mu=0, sigma=1, dims=("geo", "spline"))
    spline_w = pm.Deterministic(
        "spline_w",
        spline_mu + spline_offset * spline_sigma,
        dims=("geo", "spline"),
    )

    time_varying_intercept_raw = pm.math.dot(spline_basis_, spline_w.T)
    time_varying_intercept = pm.Deterministic(
        "time_varying_intercept",
        pt.softplus(time_varying_intercept_raw),
        dims=("date", "geo"),
    )
```

**Key design choices:**

- **Fewer knots** (4--6) emphasize long-run baseline movement rather than short-term wiggles.
- **Softplus** ensures the intercept stays positive (same idea as `SoftPlusHSGP` in PyMC-Marketing).
- **Spline regularization** (`sigma=0.08`--`0.10`) prevents the spline from absorbing seasonal patterns meant for the Fourier term.

**Confounding with Fourier:** Spline baselines and Fourier harmonics can both explain smooth temporal patterns. Mitigate this by using a `ZeroSumNormal` prior on Fourier coefficients (prevents Fourier from absorbing level shifts) and stronger spline regularization.

## Custom Events

PyMC-Marketing provides basis functions for modeling event effects with flexible temporal profiles:

| Class | Description |
|-------|-------------|
| `GaussianBasis` | Symmetric bell curve centered on the event window |
| `HalfGaussianBasis` | One-sided effect (before or after the event) |
| `AsymmetricGaussianBasis` | Different spread before and after the event |

### Usage pattern

```python
from pymc_extras.prior import Prior
from pymc_marketing.mmm.events import (
    EventEffect,
    GaussianBasis,
    HalfGaussianBasis,
    AsymmetricGaussianBasis,
)

half_after = HalfGaussianBasis(
    priors={"sigma": Prior("Gamma", mu=7, sigma=1, dims="event")},
    mode="after",
    include_event=False,
)

effect_size = Prior("Normal", mu=1, sigma=1, dims="event")
effect = EventEffect(basis=half_after, effect_size=effect_size, dims=("event",))

with pm.Model(coords=coords):
    event_curve = pm.Deterministic("effect", effect.apply(X), dims=("date", "event"))
```

The basis matrix `X` contains day offsets relative to each event window. See the [MMM components notebook](https://www.pymc-marketing.io/en/latest/notebooks/mmm/mmm_components.html) for the full `create_basis_matrix` helper.

### AsymmetricGaussianBasis

For events with different pre- and post-event dynamics:

```python
asymmetric = AsymmetricGaussianBasis(
    priors={
        "sigma_before": Prior("Gamma", mu=3, sigma=1, dims="event"),
        "sigma_after": Prior("Gamma", mu=7, sigma=2, dims="event"),
        "a_after": Prior("Normal", mu=3, sigma=0.5, dims="event"),
    },
    event_in="after",  # "after", "before", or "exclude"
)
```

## Trade-Offs vs the MMM Class

| Feature | `MMM` class | Custom `pm.Model` |
|---------|-------------|---------------------|
| Built-in scaling (channel & target) | Yes | Manual |
| `plot.waterfall_components_decomposition()` | Yes | Manual |
| `plot.contributions_over_time()` | Yes | Manual |
| `MultiDimensionalBudgetOptimizerWrapper` | Yes | Not available |
| `add_lift_test_measurements()` | Yes | Manual implementation required |
| `save()` / `load()` | Yes | Use `arviz.to_netcdf(idata)` |
| `TimeSliceCrossValidator` | Yes | Manual loop |
| Arbitrary hierarchical structure | Limited to supported dims | Full flexibility |
| Custom likelihood | Any `Prior` via `model_config["likelihood"]` (default: `TruncatedNormal`) | Any PyMC distribution |
| Custom intercept (splines, changepoints, etc.) | Not built-in (HSGP TVP only) | Full control |
| Sum-to-zero constraints | Not built-in | `pm.ZeroSumNormal` |
| Multi-equation / linked models | Not supported | Any structure expressible in PyMC |

**Recommendation:** Start with the `MMM` class. Move to a custom model only when the class cannot express your model structure. If you switch to a custom model, be prepared to reimplement scaling, contribution decomposition, and optimization manually.

---

## See Also

- [model_specification.md](model_specification.md) -- `MMM` class constructor and `model_config` reference
- [time_varying_parameters.md](time_varying_parameters.md) -- GP-based TVP within the `MMM` class (alternative to splines)
- For foundational PyMC patterns (coords/dims, parameterization, HSGP, diagnostics), see the [pymc-modeling skill](https://github.com/pymc-labs/python-analytics-skills/tree/main/skills/pymc-modeling).
