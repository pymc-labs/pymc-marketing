# Time-Varying Parameters

## Table of Contents
- [When to Use Time-Varying Parameters](#when-to-use-time-varying-parameters)
- [Time-Varying Intercept](#time-varying-intercept)
- [Time-Varying Media](#time-varying-media)
- [HSGPKwargs Reference](#hsgpkwargs-reference)
- [Parameterization Tips](#parameterization-tips)
- [Out-of-Sample Behavior](#out-of-sample-behavior)
- [Diagnostics](#diagnostics)

## When to Use Time-Varying Parameters

The `MMM` class supports two GP-based time-varying components: a time-varying intercept and a time-varying media multiplier. Both use Hilbert Space Gaussian Processes (HSGP) for computational efficiency.

| Component | What It Captures | When to Use |
|-----------|-----------------|-------------|
| Time-varying intercept | Baseline demand shifts not explained by media, controls, or seasonality | Unexplained spikes/dips from events like competitor launches, pandemics, supply shocks |
| Time-varying media | Temporal fluctuations in overall media effectiveness | Media ROI changes over time due to market dynamics, creative fatigue, competitive pressure |

**Decision guide:**

- If residuals show **periodic** patterns, prefer Fourier seasonality (`yearly_seasonality=N`) or a linear control variable over a GP.
- If residuals show **persistent trend** (steady growth), prefer a linear trend control variable.
- If residuals show **irregular, non-repeating** temporal variation that no other component explains, use a time-varying parameter.
- GPs are best at capturing patterns that are **hard to extrapolate** (random events, regime shifts). They are not the most efficient tool for simple seasonality or linear trends.

## Time-Varying Intercept

Enable with `time_varying_intercept=True`. The GP models percentage deviations from a baseline intercept, constrained to `mu=1` and then multiplied by the baseline.

```python
from pymc_extras.prior import Prior

from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM

model_config = {
    "intercept": Prior("LogitNormal", mu=0, sigma=0.1),
    "intercept_tvp_config": HSGPKwargs(
        m=500,
        L=188,
        eta_lam=5.0,
        ls_mu=5.0,
        ls_sigma=10.0,
    ),
}

mmm = MMM(
    date_column="date",
    target_column="y",
    channel_columns=channel_columns,
    control_columns=control_columns,
    adstock=GeometricAdstock(l_max=10),
    saturation=LogisticSaturation(),
    time_varying_intercept=True,
    model_config=model_config,
)
```

After fitting, the posterior contains:
- `intercept_contribution_original_scale`: the baseline intercept value (should match domain expectation).
- `intercept_contribution`: the time-varying intercept contribution over time.

To visualize the time-varying intercept against a known ground truth or expectation:

```python
mmm.sample_posterior_predictive(
    X_full,
    extend_idata=True,
    var_names=[
        "channel_contribution",
        "control_contribution",
        "intercept_contribution",
        "y_original_scale",
        "intercept_baseline",
    ],
)
```

## Time-Varying Media

Enable with `time_varying_media=True`. A single latent process `lambda_t` multiplies **all** channels simultaneously, capturing shared temporal fluctuations in media effectiveness:

```
y_t = alpha + lambda_t * sum_m(beta_m * f(x_m,t)) + sum_c(gamma_c * z_c,t) + epsilon_t
```

```python
from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm import GeometricAdstock, MichaelisMentenSaturation
from pymc_marketing.mmm.multidimensional import MMM

hsgp_kwargs = HSGPKwargs(
    ls_mu=11.0,
    ls_sigma=5.0,
)

mmm = MMM(
    date_column="date",
    target_column="y",
    channel_columns=["x1", "x2"],
    control_columns=["event_1", "event_2"],
    yearly_seasonality=2,
    adstock=GeometricAdstock(l_max=8),
    saturation=MichaelisMentenSaturation(),
    time_varying_media=True,
    model_config={"media_tvp_config": hsgp_kwargs},
)
```

After fitting, the posterior contains:
- `media_temporal_latent_multiplier`: the recovered latent process `lambda_t` (centered around 1.0).
- `baseline_channel_contribution`: channel contributions *before* the time-varying multiplier.
- `channel_contribution`: channel contributions *after* the time-varying multiplier.

To inspect the recovered latent process:

```python
media_latent = mmm.fit_result["media_temporal_latent_multiplier"]
media_latent_q = media_latent.quantile([0.025, 0.50, 0.975], dim=["chain", "draw"])

fig, ax = plt.subplots()
ax.plot(dates, media_latent_q.sel(quantile=0.5), label="Posterior median")
ax.fill_between(
    dates,
    media_latent_q.sel(quantile=0.025),
    media_latent_q.sel(quantile=0.975),
    alpha=0.3,
)
ax.set(title="Recovered media latent multiplier", ylabel="Multiplier")
```

## HSGPKwargs Reference

Both `intercept_tvp_config` and `media_tvp_config` accept an `HSGPKwargs` instance:

```python
from pymc_marketing.hsgp_kwargs import HSGPKwargs

config = HSGPKwargs(
    m=200,          # number of basis functions
    L=None,         # half-length of the approximation domain (auto-computed if None)
    eta_lam=1.0,    # marginal standard deviation prior scale
    ls_mu=5.0,      # length-scale prior mean (InverseGamma)
    ls_sigma=5.0,   # length-scale prior sigma (InverseGamma)
    cov_func=None,  # custom covariance function (overrides eta/ls priors)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `m` | `200` | Number of HSGP basis functions. Higher values capture more frequency detail at the cost of more parameters and slower sampling. |
| `L` | `None` | Half-length of the approximation domain. If `None`, computed automatically. Manual formula: `L = (1 + margin) * n_dates / 2` where `margin` is typically 0.2--0.3. |
| `eta_lam` | `1.0` | Scale for the marginal standard deviation prior. Controls the amplitude of the GP. |
| `ls_mu` | Default varies | Mean of the length-scale prior. Controls the smoothness timescale of the GP. |
| `ls_sigma` | Default varies | Standard deviation of the length-scale prior. |
| `cov_func` | `None` | Optional custom covariance function. If provided, overrides `eta_lam`, `ls_mu`, `ls_sigma`. Useful for multi-scale patterns (sum of two kernels with different length-scales). |

## Parameterization Tips

### Length-scale prior (`ls_mu`)

The length-scale controls the timescale of variation the GP can capture:

- **Long length-scale** (e.g., `ls_mu=104` for ~2 years at weekly granularity): captures slow, broad trends. Appropriate when unexplained variation happens on seasonal or annual timescales.
- **Short length-scale** (e.g., `ls_mu=5`--`52`): captures faster, event-like fluctuations. Appropriate for modeling irregular events (competitor launches, supply disruptions).

If the GP is oversmoothing and missing short-timescale events, lower `ls_mu`. If it is fitting noise, raise `ls_mu`.

### Number of basis functions (`m`)

Higher `m` lets the HSGP represent higher-frequency variation. The default of 200 is generous for most weekly datasets. For very long time series or short length-scales, increase `m` (e.g., 500). For shorter series, 50--100 may suffice.

### Domain half-length (`L`)

`L` defines the "box" `[-L, L]` over which the GP approximation is valid. If `L` is too small, the approximation degrades near the boundaries. The automatic computation is usually sufficient, but for out-of-sample prediction you may need to manually set `L` large enough to cover the prediction window.

### Multi-scale events

For events operating on two distinct timescales (e.g., both quarterly and sporadic), supply a custom `cov_func` that is the **sum of two covariance functions**, each with its own length-scale prior:

```python
import pymc as pm

ls_short = 10.0   # captures event-like fluctuations (~10 weeks)
ls_long = 52.0    # captures quarterly/annual patterns (~1 year)

cov_func = (
    pm.gp.cov.ExpQuad(input_dim=1, ls=ls_short)
    + pm.gp.cov.ExpQuad(input_dim=1, ls=ls_long)
)

config = HSGPKwargs(m=300, L=200, cov_func=cov_func)
```

## Out-of-Sample Behavior

GPs revert to their prior mean out of sample, and uncertainty grows rapidly beyond the training window. This means:

- **In-sample decomposition** is the primary use case for TVP. The GP explains temporal variation within the observed period.
- **Short-horizon forecasts** (up to ~3 months for weekly data) may remain reasonable, depending on the length-scale.
- **Long-horizon forecasts** will see the TVP component collapse toward its mean with wide uncertainty bands.

For scenario planning or forecasting beyond a few months, do not rely on the GP component. Instead, use Fourier seasonality for periodic patterns and linear controls for trends.

## Diagnostics

### Time-varying intercept

1. Check `intercept_contribution_original_scale` mean: it should match your expectation for baseline demand.
2. Plot the posterior intercept against the observed data to see if the GP captures unexplained variation.
3. Verify out-of-sample: the intercept uncertainty should widen (not stay artificially tight).

### Time-varying media

1. Plot `media_temporal_latent_multiplier` over time. It should be centered around 1.0.
2. Values > 1 indicate periods of above-average media effectiveness; < 1 indicates below-average.
3. Compare `baseline_channel_contribution` vs `channel_contribution` to see the multiplicative effect.
4. If the model without TVP shows divergences but the TVP model does not, this is evidence that time-varying media is needed.

### General HSGP checks

- Verify `r_hat < 1.01` for `media_temporal_latent_multiplier_raw_eta`, `media_temporal_latent_multiplier_raw_ls`, and the HSGP coefficients.
- Check that the recovered length-scale posterior is consistent with the expected timescale of variation.

---

## See Also

- [model_specification.md](model_specification.md) -- `time_varying_intercept` and `time_varying_media` constructor parameters
- [model_fit.md](model_fit.md) -- General diagnostics checklist
- For foundational GP and HSGP patterns, see the [pymc-modeling skill](https://github.com/pymc-labs/python-analytics-skills/tree/main/skills/pymc-modeling).
