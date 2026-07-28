# Plotting Reference (`MMMPlotSuite`)

## Table of Contents
- [Access Pattern](#access-pattern)
- [Common Parameters](#common-parameters)
- [Model Diagnostics](#model-diagnostics)
- [Contributions and Decomposition](#contributions-and-decomposition)
- [Parameter Inspection](#parameter-inspection)
- [Saturation and Media](#saturation-and-media)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Budget Optimization](#budget-optimization)
- [Cross-Validation](#cross-validation)

## Access Pattern

All visualization methods live on `MMMPlotSuite`, accessed via the `mmm.plot` property:

```python
fig, axes = mmm.plot.posterior_predictive()
fig, axes = mmm.plot.waterfall_components_decomposition()
```

The property creates a fresh `MMMPlotSuite(idata=self.idata)` on every access. The model must be built and `idata` must exist (i.e., `fit()` or `sample_prior_predictive()` has been called).

`TimeSliceCrossValidator` also exposes a `.plot` property returning the same `MMMPlotSuite`, so cross-validation plots are called as `cv.plot.cv_predictions(results)`.

`MMMPlotSuite` can also be used standalone with any `az.InferenceData`:

```python
from pymc_marketing.mmm.plot import MMMPlotSuite

suite = MMMPlotSuite(idata=idata)
fig, axes = suite.posterior_predictive()
```

## Common Parameters

### `dims` parameter

Many methods accept a `dims: dict[str, str | int | list] | None` parameter for filtering and faceting across model dimensions. When provided, only the selected slices are plotted:

```python
mmm.plot.contributions_over_time(
    var=["channel_contribution_original_scale"],
    dims={"geo": ["US", "UK"]},
)

mmm.plot.posterior_distribution(
    var="saturation_beta",
    dims={"geo": "US"},
)
```

Keys must match model dimension names. Values can be a single coordinate value, an integer index, or a list of values (which generates multiple panels).

### `split_by` parameter

Some methods accept `split_by: str | list[str] | None` to create separate subplots along a dimension:

```python
mmm.plot.waterfall_components_decomposition(split_by="geo")
```

### Return types

Most methods return `tuple[Figure, NDArray[Axes]]`. Some return a single `Axes` when there is only one panel, or `Figure` alone. The exact return type is documented per method below.

---

## Model Diagnostics

### `posterior_predictive`

```python
mmm.plot.posterior_predictive(
    var: list[str] | None = None,       # default: ["y"]
    idata: xr.Dataset | None = None,    # default: self.idata.posterior_predictive
    hdi_prob: float = 0.85,
) -> tuple[Figure, NDArray[Axes]]
```

Plots time series from the posterior predictive distribution with median line and HDI band. Creates one subplot per combination of non-date dimensions. `var` is a **list** of variable names to overlay.

```python
mmm.plot.posterior_predictive(var=["y"], hdi_prob=0.94)
```

### `prior_predictive`

```python
mmm.plot.prior_predictive(
    var: str | None = None,             # default: "y" -- NOTE: single str, not list
    idata: xr.Dataset | None = None,    # default: self.idata.prior_predictive
    hdi_prob: float = 0.85,
) -> tuple[Figure, NDArray[Axes]]
```

Plots time series from the prior predictive distribution. Same layout as `posterior_predictive` but `var` is a **single string**, not a list.

```python
mmm.plot.prior_predictive(var="y", hdi_prob=0.94)
```

### `residuals_over_time`

```python
mmm.plot.residuals_over_time(
    hdi_prob: list[float] | None = None,    # default: [0.94]
) -> tuple[Figure, NDArray[Axes]]
```

Plots residuals (target - predicted) over time with HDI bands. Requires `y_original_scale` in `posterior_predictive` and `target_data` in `constant_data`. Multiple HDI probabilities can be overlaid:

```python
mmm.plot.residuals_over_time(hdi_prob=[0.5, 0.85, 0.94])
```

### `residuals_posterior_distribution`

```python
mmm.plot.residuals_posterior_distribution(
    quantiles: list[float] | None = None,   # default: [0.25, 0.5, 0.75]
    aggregation: str | None = None,         # "mean", "sum", or None
) -> tuple[Figure, NDArray[Axes]]
```

Plots the posterior distribution of residuals. With `aggregation=None` (default), creates one panel per dimension combination. With `"mean"` or `"sum"`, aggregates across all dimensions into a single panel.

```python
mmm.plot.residuals_posterior_distribution(
    quantiles=[0.05, 0.5, 0.95], aggregation="mean"
)
```

---

## Contributions and Decomposition

### `contributions_over_time`

```python
mmm.plot.contributions_over_time(
    var: list[str],                                     # REQUIRED
    hdi_prob: float = 0.85,
    dims: dict[str, str | int | list] | None = None,
    combine_dims: bool = False,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, NDArray[Axes]]
```

Plots time-series contributions from the posterior for each variable in `var`, with median line and HDI band. Creates one subplot per dimension combination. With `combine_dims=True`, overlays all dimension combinations on a single axis.

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

### `waterfall_components_decomposition`

```python
mmm.plot.waterfall_components_decomposition(
    var: list[str] | None = None,                       # auto-detects if None
    original_scale: bool = True,
    dims: dict[str, str | int | list] | None = None,
    split_by: str | list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[Figure, Axes] | tuple[Figure, NDArray[Axes]]
```

Horizontal waterfall chart of mean response decomposition by components. Auto-detects contribution variables when `var=None`. Use `split_by` to facet by a dimension (e.g., `split_by="geo"`).

```python
mmm.plot.waterfall_components_decomposition(original_scale=True)

mmm.plot.waterfall_components_decomposition(
    split_by="geo", dims={"geo": ["US", "UK"]}
)
```

### `channel_contribution_share_hdi`

```python
mmm.plot.channel_contribution_share_hdi(
    hdi_prob: float = 0.94,
    dims: dict[str, str | int | list] | None = None,
    figsize: tuple[float, float] = (10, 6),
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]
```

Forest plot of channel contribution shares (percentages) with HDI intervals. Uses `az.plot_forest` internally.

```python
mmm.plot.channel_contribution_share_hdi(hdi_prob=0.94)
```

---

## Parameter Inspection

### `posterior_distribution`

```python
mmm.plot.posterior_distribution(
    var: str,                                           # REQUIRED
    plot_dim: str = "channel",
    orient: str = "h",
    dims: dict[str, str | int | list] | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, NDArray[Axes]]
```

Violin plots of a posterior variable across a specified dimension. One subplot per additional dimension combination.

```python
mmm.plot.posterior_distribution(var="saturation_beta", plot_dim="channel")
```

### `channel_parameter`

```python
mmm.plot.channel_parameter(
    param_name: str,                                    # REQUIRED
    orient: str = "h",
    dims: dict[str, str | int | list] | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Figure                                             # NOTE: returns Figure only, not a tuple
```

Violin plots of a channel parameter's posterior. Returns only `Figure`, not `(Figure, Axes)`.

```python
fig = mmm.plot.channel_parameter(param_name="adstock_alpha")
```

### `prior_vs_posterior`

```python
mmm.plot.prior_vs_posterior(
    var: str,                                           # REQUIRED
    plot_dim: str = "channel",
    alphabetical_sort: bool = True,
    dims: dict[str, str | int | list] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, NDArray[Axes]]
```

KDE plots comparing prior and posterior distributions. One subplot per `plot_dim` value. Handles scalar variables (no `plot_dim`) gracefully.

```python
mmm.plot.prior_vs_posterior(var="saturation_beta", plot_dim="channel")
```

---

## Saturation and Media

### `saturation_scatterplot`

```python
mmm.plot.saturation_scatterplot(
    original_scale: bool = False,
    dims: dict[str, str | int | list] | None = None,
    **kwargs,                                           # width_per_col, height_per_row
) -> tuple[Figure, NDArray[Axes]]
```

Scatter plots of channel contributions vs. channel data at each observed time point. One subplot per channel x dimension combination. This visualizes **direct (marginal)** contributions.

```python
mmm.plot.saturation_scatterplot(original_scale=True)
```

### `saturation_curves`

```python
mmm.plot.saturation_curves(
    curve: xr.DataArray,                                # REQUIRED: posterior saturation curve
    original_scale: bool = False,
    n_samples: int = 10,
    hdi_probs: float | list[float] | None = None,
    random_seed: np.random.Generator | None = None,
    colors: Iterable[str] | None = None,
    subplot_kwargs: dict | None = None,
    rc_params: dict | None = None,
    dims: dict[str, str | int | list] | None = None,
    **plot_kwargs,
) -> tuple[plt.Figure, np.ndarray]
```

Overlays saturation curve sample lines and HDI bands on scatter data. The `curve` argument is a `DataArray` from `mmm.sample_saturation_curve(...)` or `saturation.sample_curve(posterior)`.

```python
curve = mmm.sample_saturation_curve(max_value=2.0, num_points=100)
mmm.plot.saturation_curves(curve, original_scale=True, hdi_probs=[0.5, 0.94])
```

### `saturation_curves_scatter` (DEPRECATED)

```python
mmm.plot.saturation_curves_scatter(
    original_scale: bool = False,
    **kwargs,
) -> tuple[Figure, NDArray[Axes]]
```

Deprecated. Delegates to `saturation_scatterplot`. Use `saturation_scatterplot` instead.

---

## Sensitivity Analysis

### `sensitivity_analysis`

```python
mmm.plot.sensitivity_analysis(
    hdi_prob: float = 0.94,
    ax: plt.Axes | None = None,
    aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    *,
    plot_kwargs: dict[str, Any] | None = None,
    ylabel: str = "Effect",
    xlabel: str = "Sweep",
    title: str | None = None,
    add_figure_title: bool = False,
    subplot_title_fallback: str = "Sensitivity Analysis",
    hue_dim: str | None = None,
    legend: bool | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    x_sweep_axis: Literal["relative", "absolute"] = "relative",
) -> tuple[Figure, NDArray[Axes]] | plt.Axes
```

Plots sensitivity analysis sweep results from `idata.sensitivity_analysis`. Returns a single `Axes` when there is only one panel, or `(Figure, NDArray[Axes])` for multiple.

Requires `mmm.sensitivity.run_sweep(..., extend_idata=True)` to have been called first. Multi-dimensional models automatically create subplots for each additional dimension combination -- no explicit faceting parameter is needed.

```python
import numpy as np

sweeps = np.linspace(0, 1.5, 16)
mmm.sensitivity.run_sweep(
    sweep_values=sweeps,
    var_input="channel_data",
    var_names="channel_contribution_original_scale",
    extend_idata=True,
)

mmm.plot.sensitivity_analysis(
    hue_dim="channel",
    x_sweep_axis="relative",
    xlabel="Relative Spend",
    ylabel="Channel Contribution",
)
```

The `aggregation` parameter allows grouping dimensions before plotting:

```python
mmm.plot.sensitivity_analysis(
    hue_dim="channel",
    aggregation={"geo": ("US", "UK")},
)
```

### `uplift_curve`

```python
mmm.plot.uplift_curve(
    hdi_prob: float = 0.94,
    ax: plt.Axes | None = None,
    aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    *,
    plot_kwargs: dict[str, Any] | None = None,
    ylabel: str = "Uplift",
    xlabel: str = "Sweep",
    title: str | None = "Uplift curve",
    add_figure_title: bool = True,
) -> tuple[Figure, NDArray[Axes]] | plt.Axes
```

Plots precomputed uplift curves from `idata.sensitivity_analysis["uplift_curve"]`. Delegates to `sensitivity_analysis` internally with uplift-specific defaults.

Requires `mmm.sensitivity.compute_uplift_curve_respect_to_base(..., extend_idata=True)` to have been called.

```python
results = mmm.sensitivity.run_sweep(
    sweep_values=sweeps, var_input="channel_data",
    var_names="channel_contribution_original_scale",
)
ref = results.sel(sweep=1.0)
mmm.sensitivity.compute_uplift_curve_respect_to_base(results, ref=ref, extend_idata=True)

mmm.plot.uplift_curve(hue_dim="channel")
```

### `marginal_curve`

```python
mmm.plot.marginal_curve(
    hdi_prob: float = 0.94,
    ax: plt.Axes | None = None,
    aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    *,
    plot_kwargs: dict[str, Any] | None = None,
    ylabel: str = "Marginal effect",
    xlabel: str = "Sweep",
    title: str | None = "Marginal effects",
    add_figure_title: bool = True,
) -> tuple[Figure, NDArray[Axes]] | plt.Axes
```

Plots precomputed marginal effects from `idata.sensitivity_analysis["marginal_effects"]`. Delegates to `sensitivity_analysis` internally.

Requires `mmm.sensitivity.compute_marginal_effects(..., extend_idata=True)` to have been called.

```python
results = mmm.sensitivity.run_sweep(
    sweep_values=sweeps, var_input="channel_data",
    var_names="channel_contribution_original_scale",
)
mmm.sensitivity.compute_marginal_effects(results, extend_idata=True)

mmm.plot.marginal_curve(hue_dim="channel")
```

---

## Budget Optimization

### `budget_allocation`

```python
mmm.plot.budget_allocation(
    samples: xr.Dataset,                                # REQUIRED: output from sample_response_distribution
    scale_factor: float | None = None,
    figsize: tuple[float, float] = (12, 6),
    ax: plt.Axes | None = None,
    original_scale: bool = True,
    dims: dict[str, str | int | list] | None = None,
) -> tuple[Figure, plt.Axes] | tuple[Figure, np.ndarray]
```

Twin-axis bar chart of allocated spend (left axis) vs. expected channel contribution (right axis).

```python
response = optimizable_model.sample_response_distribution(
    allocation_strategy=allocation,
    include_last_observations=True,
    include_carryover=True,
)
mmm.plot.budget_allocation(samples=response)
```

### `allocated_contribution_by_channel_over_time`

```python
mmm.plot.allocated_contribution_by_channel_over_time(
    samples: xr.Dataset | az.InferenceData,             # REQUIRED
    hdi_prob: float = 0.94,
    dims: dict[str, str | int | list] | None = None,
    split_by: str | list[str] | None = None,
    original_scale: bool = True,
    scale_factor: float | None = None,
    figsize: tuple[float, float] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[Figure, Axes] | tuple[Figure, NDArray[Axes]]
```

Time-series of allocated channel contributions with HDI bands per channel. Returns single `Axes` for one panel, `NDArray[Axes]` for multiple.

```python
mmm.plot.allocated_contribution_by_channel_over_time(
    samples=response, hdi_prob=0.94, split_by="geo"
)
```

---

## Cross-Validation

These methods are typically called via `cv.plot.<method>(results)` where `cv` is a `TimeSliceCrossValidator` and `results` is the output of `cv.run(...)`.

### `cv_predictions`

```python
mmm.plot.cv_predictions(
    results: az.InferenceData,                          # REQUIRED: output from cv.run()
    dims: dict[str, str | int | list] | None = None,
) -> tuple[Figure, NDArray[Axes]]
```

Plots posterior predictive across CV folds, with train/test HDI bands, observed values, and a vertical marker at the train-test boundary.

```python
cv.plot.cv_predictions(results)
```

### `param_stability`

```python
mmm.plot.param_stability(
    results: az.InferenceData,                          # REQUIRED
    parameter: list[str],                               # REQUIRED: parameter names to inspect
    dims: dict[str, list[str]] | None = None,           # NOTE: values are list[str] specifically
) -> tuple[Figure, NDArray[Axes]]
```

Forest plots of parameter stability across CV folds. Shows how parameter posteriors shift between folds. Note: `dims` values here are specifically `list[str]`, not the general `str | int | list` used elsewhere.

```python
cv.plot.param_stability(
    results, parameter=["adstock_alpha"], dims={"channel": ["tv", "radio"]}
)
```

### `cv_crps`

```python
mmm.plot.cv_crps(
    results: az.InferenceData,                          # REQUIRED
    dims: dict[str, str | int | list] | None = None,
) -> tuple[Figure, NDArray[Axes]]
```

Plots CRPS scores for train and test sets across CV splits. Returns axes with shape `(n_panels, 2)` (train column, test column).

```python
cv.plot.cv_crps(results)
```

---

## See Also

- [model_fit.md](model_fit.md) -- Fitting and diagnostics workflow
- [media_deep_dive.md](media_deep_dive.md) -- Media analysis patterns that use these plots
- [budget_optimization.md](budget_optimization.md) -- Budget optimization plots
