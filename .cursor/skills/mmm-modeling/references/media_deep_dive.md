# Media Deep Dive: Contributions, ROAS, and Diagnostics

## Table of Contents
- [Channel Contributions Over Time](#channel-contributions-over-time)
- [Waterfall Decomposition](#waterfall-decomposition)
- [ROAS Computation](#roas-computation)
- [Contribution Share vs ROAS](#contribution-share-vs-roas)
- [Incremental Analysis](#incremental-analysis-mmmincementality)
- [Saturation Curves](#saturation-curves)
- [Adstock Curves](#adstock-curves)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Channel Contribution Share](#channel-contribution-share)

## Channel Contributions Over Time

Visualize how each channel contributes to the target over the observation period:

```python
mmm.plot.contributions_over_time(
    var=[
        "channel_contribution_original_scale",
        "control_contribution_original_scale",
        "intercept_contribution_original_scale",
    ],
    combine_dims=True,
    hdi_prob=0.94,
)
```

For manual control (e.g., specific channels or custom styling):

```python
import arviz as az
import matplotlib.pyplot as plt

channel_contrib = mmm.idata["posterior"]["channel_contribution_original_scale"]
dates = X[date_column]

fig, axes = plt.subplots(len(channel_columns), 1,
                          figsize=(15, 3 * len(channel_columns)),
                          sharex=True, layout="constrained")

for i, channel in enumerate(channel_columns):
    ax = axes[i]
    contrib = channel_contrib.sel(channel=channel)
    if "geo" in contrib.dims:
        contrib = contrib.sum(dim="geo")
    az.plot_hdi(dates, contrib, hdi_prob=0.94, color=f"C{i}",
                fill_kwargs={"alpha": 0.3}, ax=ax)
    ax.set(title=channel, ylabel="Contribution")
```

### Original Scale Contributions

Channel contributions are stored on the scaled space by default. To get original-scale contributions, call `add_original_scale_contribution_variable` **before** `sample_posterior_predictive`:

```python
mmm.add_original_scale_contribution_variable(
    var=["channel_contribution", "control_contribution",
         "intercept_contribution", "yearly_seasonality_contribution", "y"]
)
mmm.sample_posterior_predictive(X=X, random_seed=rng)
```

This registers `pm.Deterministic` variables that multiply by the target scaler:

- `channel_contribution_original_scale` -- channel media contributions
- `control_contribution_original_scale` -- control variable contributions
- `intercept_contribution_original_scale` -- baseline
- `y_original_scale` -- predicted target on original scale

Without this call, `*_original_scale` variables will not exist in the posterior predictive.

## Waterfall Decomposition

Decompose the total target into component contributions:

```python
mmm.plot.waterfall_components_decomposition()
```

This shows the mean contribution of each component (intercept, channels, controls, seasonality) as a stacked waterfall chart.

## ROAS Computation

Return on Ad Spend = total channel contribution / total channel spend:

```python
import arviz as az
import xarray as xr

channel_contrib = mmm.idata["posterior"]["channel_contribution_original_scale"]

# Sum contributions over time (and geo if multidimensional)
sum_dims = ["date"]
if "geo" in channel_contrib.dims:
    sum_dims.append("geo")

contrib_total = channel_contrib.sum(dim=sum_dims)

# Compute total spend per channel
spend_sum = X[channel_columns].sum(axis=0).values
spend_xr = xr.DataArray(spend_sum, dims=["channel"], coords={"channel": channel_columns})

roas_samples = contrib_total / spend_xr
```

### ROAS Forest Plot

```python
az.plot_forest(roas_samples, combined=True, hdi_prob=0.94)
plt.xlabel("ROAS")
plt.title("Return on Ad Spend by Channel")
```

### ROAS Summary Table

```python
roas_summary = az.summary(roas_samples, hdi_prob=0.94)
print(roas_summary[["mean", "hdi_3%", "hdi_97%"]])
```

## Contribution Share vs ROAS

A scatter plot combining contribution share (x-axis) with ROAS (y-axis) reveals which channels are both large contributors and efficient:

```python
import numpy as np

contrib_share_mean = (contrib_total / contrib_total.sum(dim="channel")).mean(dim=("chain", "draw"))
roas_mean = roas_samples.mean(dim=("chain", "draw"))
roas_hdi = az.hdi(roas_samples, hdi_prob=0.94)

fig, ax = plt.subplots(figsize=(10, 7))
for i, ch in enumerate(channel_columns):
    ax.errorbar(
        contrib_share_mean.sel(channel=ch).item(),
        roas_mean.sel(channel=ch).item(),
        yerr=np.array([[
            roas_mean.sel(channel=ch).item() - roas_hdi.sel(channel=ch).values[0],
            roas_hdi.sel(channel=ch).values[1] - roas_mean.sel(channel=ch).item(),
        ]]).T,
        fmt="o", markersize=10, capsize=5, label=ch,
    )
ax.set(xlabel="Contribution Share", ylabel="ROAS",
       title="Contribution Share vs ROAS")
ax.legend()
```

**Interpretation**: Channels in the upper-left (low share, high ROAS) may be under-invested. Channels in the lower-right (high share, low ROAS) may be over-invested.

## Incremental Analysis (`mmm.incrementality`)

The `Incrementality` class provides counterfactual analysis that correctly accounts for adstock carryover effects. This is the preferred approach for computing ROAS and CAC, as it handles temporal attribution properly: spend at time *t* affects outcomes at *t, t+1, ..., t+l_max*.

Access via the `mmm.incrementality` property:

```python
roas = mmm.incrementality.contribution_over_spend(
    frequency="quarterly",
    start_date="2024-01-01",
    end_date="2024-12-31",
)
```

### Available Methods

| Method | Description | Counterfactual |
|--------|-------------|----------------|
| `compute_incremental_contribution(frequency, ...)` | Raw incremental contribution per channel | Zero-out (default) or custom factor |
| `contribution_over_spend(frequency, ...)` | Contribution / spend (ROAS when target is revenue) | Zero-out |
| `spend_over_contribution(frequency, ...)` | Spend / contribution (CAC when target is acquisitions) | Zero-out |
| `marginal_contribution_over_spend(frequency, ..., spend_increase_pct=0.01)` | Marginal efficiency at current spend level | Small perturbation (+1% default) |

### Core Method Signature

```python
mmm.incrementality.compute_incremental_contribution(
    frequency: str,                         # "original", "weekly", "monthly", "quarterly", "yearly", "all_time"
    start_date: str | None = None,          # evaluation window start
    end_date: str | None = None,            # evaluation window end
    include_carryover: bool = True,         # account for adstock carry-in and carry-out
    num_samples: int | None = None,         # subsample posterior draws (None = all)
    random_state: RandomState | None = None,
    counterfactual_spend_factor: float = 0.0,   # 0.0 = zero-out, 0.5 = halve spend, etc.
) -> xr.DataArray
```

### Total vs Marginal ROAS

```python
total_roas = mmm.incrementality.contribution_over_spend(
    frequency="all_time",
)

marginal_roas = mmm.incrementality.marginal_contribution_over_spend(
    frequency="all_time",
    spend_increase_pct=0.01,
)
```

**Total ROAS** (zero-out counterfactual) answers: "What is the overall return per dollar for this channel?" A heavily invested channel can have a high total ROAS.

**Marginal ROAS** (small perturbation) answers: "What is the return on the *next* dollar?" This captures diminishing returns -- a channel at saturation will have a low marginal ROAS even if its total ROAS is high. This is the metric that directly informs budget reallocation.

### Element-Wise vs Counterfactual ROAS

The [ROAS section above](#roas-computation) computes ROAS element-wise as `contribution / spend`. This is a quick approximation but does not account for adstock carryover. The `mmm.incrementality` approach is more rigorous: it evaluates the model under actual vs. counterfactual spend and properly attributes carryover effects.

## Saturation Curves

There are two distinct views of saturation. Understanding the difference is critical for correct interpretation.

### Saturation Scatterplot (Direct/Marginal Contribution)

```python
mmm.plot.saturation_scatterplot(original_scale=True)
```

Shows actual spend vs. saturated effect **at each observed time point** with posterior uncertainty. Each point is one (date, channel) observation. This visualizes the **direct (marginal)** contribution: how much each unit of spend contributed at that specific spend level.

### Sampled Saturation Curves

Generate smooth saturation curves from the posterior:

```python
curve = mmm.saturation.sample_curve(mmm.idata.posterior, max_value=2)
mmm.plot.saturation_curves(curve, original_scale=True)
```

Or use the convenience method:

```python
sat_curve = mmm.sample_saturation_curve(max_value=1.0, num_points=100)
```

### Sensitivity Analysis (Total/Counterfactual Contribution)

`sensitivity_analysis` sweeps spend from 0 to N times the current level and shows the **total contribution** at each hypothetical spend level. This is a **counterfactual** view: "What would total contribution be if we scaled spend to X%?"

The key distinction:
- **Scatterplot**: shows point-wise marginal contributions at observed spend levels.
- **Sensitivity analysis**: shows total (integrated) contributions under hypothetical spend scaling -- the basis for budget reallocation decisions.

**Interpretation**: Steeper initial slope = higher marginal return at low spend. Flat tail = diminishing returns at high spend. If the curve hasn't flattened, the channel may benefit from increased investment.

## Adstock Curves

Visualize the temporal decay profile of each channel:

```python
adstock_curve = mmm.sample_adstock_curve(amount=1.0, num_samples=500)
```

This shows how a single unit of spend decays over `l_max` periods. Channels with slow decay (high `alpha`) have longer carryover effects.

## Sensitivity Analysis

Sensitivity analysis measures how channel contributions change when spend is scaled up or down:

```python
import numpy as np

sweeps = np.linspace(0, 1.5, 16)

mmm.sensitivity.run_sweep(
    sweep_values=sweeps,
    var_input="channel_data",
    var_names="channel_contribution_original_scale",
    extend_idata=True,
)
```

### Sensitivity Plot

```python
mmm.plot.sensitivity_analysis(
    xlabel="Relative Spend",
    ylabel="Channel Contribution",
    hue_dim="channel",
    x_sweep_axis="relative",
)
```

**Interpretation**: The slope at `x=1.0` (current spend) indicates marginal efficiency. Steeper slope = more responsive to spend changes. Flatter slope = closer to saturation.

### Multidimensional Sensitivity

For geo-level models, the sensitivity plot automatically creates subplots for each geo when the sweep data has a `geo` dimension -- no explicit faceting parameter is needed. Use `aggregation` to group geos before plotting:

```python
mmm.sensitivity.run_sweep(
    sweep_values=sweeps,
    var_input="channel_data",
    var_names="channel_contribution_original_scale",
    extend_idata=True,
)

mmm.plot.sensitivity_analysis(
    hue_dim="channel",
    x_sweep_axis="relative",
)

mmm.plot.sensitivity_analysis(
    hue_dim="channel",
    aggregation={"geo": ("US", "UK")},
    x_sweep_axis="relative",
)
```

See [plot.md](plot.md) for the full `sensitivity_analysis` signature and related methods (`uplift_curve`, `marginal_curve`).

## Channel Contribution Share

Compare prior vs. posterior contribution shares to assess how much the data informed the model.

**Prerequisite**: `add_original_scale_contribution_variable` must be called before both `sample_prior_predictive` and `sample_posterior_predictive` for `*_original_scale` variables to exist in the respective groups.

```python
# Posterior share
posterior_share = contrib_total / contrib_total.sum(dim="channel")

# Prior share (from prior predictive -- requires add_original_scale_contribution_variable
# to have been called before sample_prior_predictive)
prior_contrib = mmm.idata["prior_predictive"]["channel_contribution_original_scale"]
prior_total = prior_contrib.sum(dim=sum_dims)
prior_share = prior_total / prior_total.sum(dim="channel")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
az.plot_forest(prior_share, combined=True, ax=axes[0])
axes[0].set_title("Prior Contribution Share")
az.plot_forest(posterior_share, combined=True, ax=axes[1])
axes[1].set_title("Posterior Contribution Share")
```

If prior and posterior shares are nearly identical, the data may not be informative enough to distinguish channels -- consider adding lift test calibration (see [liftest_calibration.md](liftest_calibration.md)).

---

## See Also

- [model_fit.md](model_fit.md) -- Fitting and diagnostics (must pass before media analysis)
- [budget_optimization.md](budget_optimization.md) -- Translate ROAS insights into optimal allocations
- [liftest_calibration.md](liftest_calibration.md) -- Resolve identification issues revealed by media analysis
