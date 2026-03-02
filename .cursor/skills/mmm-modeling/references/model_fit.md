# Model Fitting, Diagnostics, and Validation

## Table of Contents
- [Fitting on Full Data](#fitting-on-full-data)
- [Posterior Predictive Sampling](#posterior-predictive-sampling)
- [Diagnostics Checklist](#diagnostics-checklist)
- [Posterior Predictive Checks](#posterior-predictive-checks)
- [Time-Slice Cross-Validation](#time-slice-cross-validation)
- [Out-of-Sample Forecasting](#out-of-sample-forecasting)
- [Common Fit Issues](#common-fit-issues)

## Fitting on Full Data

The final production model is always fit on the **entire dataset**. Never hold out data for the final model. More data means better parameter estimates.

```python
import numpy as np

rng = np.random.default_rng(42)

X = data_df.drop(columns=[target_column])
y = data_df[target_column]

mmm.fit(
    X=X,
    y=y,
    target_accept=0.9,
    chains=6,
    draws=800,
    tune=1_500,
    nuts_sampler="nutpie",
    random_seed=rng,
)
```

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| `target_accept` | `0.9` | Increase to `0.95` or `0.99` if divergences persist |
| `chains` | `4`--`6` | More chains help diagnose convergence |
| `draws` | `500`--`1000` | Per-chain posterior draws |
| `tune` | `1000`--`2000` | Warm-up iterations (discarded) |
| `nuts_sampler` | `"nutpie"` | Fastest sampler for most MMMs |

## Posterior Predictive Sampling

After fitting, register original-scale contribution variables and then sample posterior predictions:

```python
mmm.add_original_scale_contribution_variable(
    var=["channel_contribution", "control_contribution",
         "intercept_contribution", "yearly_seasonality_contribution", "y"]
)
mmm.sample_posterior_predictive(X=X, random_seed=rng)
```

`add_original_scale_contribution_variable` creates `pm.Deterministic` nodes that multiply contributions by the target scaler. Without this call, `*_original_scale` variables will not exist in `posterior_predictive`.

This adds `posterior_predictive` to `mmm.idata`, which is needed for posterior predictive checks and component decomposition.

## Diagnostics Checklist

Run these checks after every fit:

| Diagnostic | How to Compute | Threshold |
|------------|---------------|-----------|
| **Divergences** | `mmm.idata["sample_stats"]["diverging"].sum().item()` | Must be **0** |
| **R-hat** | `az.summary(data=mmm.idata, var_names=[...])["r_hat"].describe()` | All < **1.01** |
| **ESS bulk** | `az.summary(...)["ess_bulk"].describe()` | All > **400** |
| **ESS tail** | `az.summary(...)["ess_tail"].describe()` | All > **400** |

### Divergence Check

```python
n_divergences = mmm.idata["sample_stats"]["diverging"].sum().item()
print(f"Number of divergences: {n_divergences}")
assert n_divergences == 0, "Model has divergences -- fix before proceeding!"
```

### R-hat and ESS

Check convergence on the **stochastic** (sampled) parameters. Note: `"intercept"` is the raw parameter; `"intercept_contribution"` is the deterministic transformation used in decomposition -- both exist in the posterior.

```python
import arviz as az

var_names = [
    "intercept", "saturation_beta", "adstock_alpha", "saturation_lam",
    "gamma_control", "gamma_fourier",
]

summary = az.summary(
    data=mmm.idata,
    var_names=var_names,
)
print(summary[["r_hat", "ess_bulk", "ess_tail"]].describe())
```

### Trace Plots

Visual check for chain mixing:

```python
az.plot_trace(
    data=mmm.fit_result,
    var_names=["intercept", "saturation_beta", "adstock_alpha", "saturation_lam"],
    compact=True,
)
```

Look for: chains overlapping (good), stationary distributions (good), sticky chains or multimodality (bad).

## Posterior Predictive Checks

### HDI Bands Over Time

```python
import matplotlib.pyplot as plt
import arviz as az

posterior_pred = mmm.idata["posterior_predictive"]["y"]
dates = X[date_column]

fig, ax = plt.subplots(figsize=(15, 5))
az.plot_hdi(dates, posterior_pred, hdi_prob=0.94, color="C0",
            fill_kwargs={"alpha": 0.3}, ax=ax)
ax.plot(dates, y, color="black", label="Observed")
ax.set(xlabel="Date", ylabel=target_column,
       title="Posterior Predictive Check")
ax.legend()
```

### Error Analysis

```python
posterior_pred_mean = posterior_pred.mean(dim=("chain", "draw"))
errors = y.values - posterior_pred_mean.values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(dates, errors)
axes[0].axhline(0, color="black", linestyle="--")
axes[0].set(title="Residuals Over Time", xlabel="Date", ylabel="Error")

axes[1].hist(errors, bins=30, edgecolor="white")
axes[1].set(title="Error Distribution", xlabel="Error", ylabel="Count")
```

Look for: no systematic time trends in residuals, roughly symmetric error distribution.

## Time-Slice Cross-Validation

Time-slice CV is the **only** appropriate use of train/test splits in MMM. It assesses model stability and out-of-sample prediction quality, but the final production model is always fit on the full dataset.

### Setup

```python
from pymc_marketing.mmm.time_slice_cross_validation import TimeSliceCrossValidator

cv = TimeSliceCrossValidator(
    n_init=163,             # initial training window size
    forecast_horizon=12,    # number of periods to forecast
    date_column="date",
    step_size=1,            # advance by 1 period per slice
)
```

| Parameter | Description |
|-----------|-------------|
| `n_init` | Number of initial observations in the first training window |
| `forecast_horizon` | How many future periods to predict in each slice |
| `date_column` | Name of the date column |
| `step_size` | How many periods to advance the window between slices |

### Running Cross-Validation

```python
results = cv.run(
    X, y,
    sampler_config={"nuts_sampler": "nutpie", "target_accept": 0.9},
    yaml_path="model_spec.yaml",  # optional: model specification as YAML
)
```

### Diagnostics Plots

```python
# Parameter stability across time slices
cv.plot.param_stability(results, parameter=["adstock_alpha"], dims={"channel": "tv"})

# Out-of-sample predictions
cv.plot.cv_predictions(results)

# CRPS scores
cv.plot.cv_crps(results)
```

**What to look for:**

- **Parameter stability**: Parameters should not jump abruptly between adjacent time slices. Gradual drift is acceptable; sudden shifts indicate model instability.
- **CRPS scores**: Lower is better. Stable CRPS across slices indicates consistent predictive performance.
- **Predictions**: Forecasted intervals should cover the observed values.

### Data Leakage Warning

Compute any data-derived quantities (spend shares, scaling factors) independently within each cross-validation fold. Do not use statistics from the full dataset when building per-fold models, as this introduces data leakage.

## Out-of-Sample Forecasting

After fitting the final model on the full dataset, forecast future periods:

```python
X_future = pd.DataFrame({
    "date": pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=12, freq="W"),
    "tv": planned_tv_spend,
    "radio": planned_radio_spend,
    "social": planned_social_spend,
})

forecast = mmm.sample_posterior_predictive(
    X_future,
    extend_idata=False,
    include_last_observations=True,
    random_seed=rng,
)
```

The `include_last_observations=True` ensures adstock carryover from the last observed period is included.

### Evaluating Forecasts (When Actuals Become Available)

```python
from pymc_marketing.metrics import crps

posterior_pred = forecast["posterior_predictive"]["y"].values
score = crps(y_true=y_actual.to_numpy(), y_pred=posterior_pred.T)
```

## Common Fit Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Divergences > 0 | Step size too large or poor geometry | Increase `target_accept` to `0.95`/`0.99`; reparameterize priors (e.g., `centered=False`) |
| Low ESS (< 100) | Slow mixing | Increase `tune` and `draws`; use `centered=False` for hierarchical priors |
| R-hat > 1.01 | Chains not converged | Run more `chains` and `tune`; check for multimodality in trace plots |
| Posterior predictive misses trend | Missing control variable or seasonality | Add `yearly_seasonality` or `time_varying_intercept` |
| Saturating too early / late | Poor `lam` prior | Adjust `lam` prior based on domain knowledge; check `saturation_scatterplot` |
| Adstock too slow / fast | Poor `alpha` prior | Tighten `alpha` prior (e.g., `Beta(2, 5)` for faster decay) |
| All channels look the same | Full pooling when partial needed | Switch to `dims=("channel", "geo")` with hierarchical priors |

---

## See Also

- [model_specification.md](model_specification.md) -- Prior configuration and constructor reference
- [media_deep_dive.md](media_deep_dive.md) -- Post-fit media analysis
- [data_analysis.md](data_analysis.md) -- Data preparation
