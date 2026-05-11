# Migration Guide Rewrite — Design Spec
**Date:** 2026-05-11
**File:** `docs/source/notebooks/mmm/mmm_plot_suite_migration_guide.ipynb`

---

## Goal

Complete rewrite of the existing migration guide notebook. The current notebook only covers method renames and namespace structure. It is missing:

- arviz-plots as the rendering engine
- Documentation of the new common argument set (most args changed)
- Non-matplotlib backend caveats
- Per-namespace argument change tables

The new notebook uses **reference (non-executing) code** throughout.

---

## Structure (Option A — linear progressive disclosure)

Sections appear in this order. Each is one or more notebook cells (markdown + code as noted).

### 1. Introduction
**Markdown only.**

One paragraph explaining what changed and why: the monolithic `MMMPlotSuite` has been replaced with a modular, namespace-based API. The new API is built on top of **arviz-plots** (`PlotCollection`), which handles subplot layout, dimension-aware faceting, and multi-backend rendering. Link to arviz-plots docs.

### 2. Opting In
**Markdown + code cell.**

Show `mmm.plot_suite = "new"` and `cv.plot_suite = "new"`. No change from current notebook.

```python
from pymc_marketing.mmm import MMM, TimeSliceCrossValidator

mmm = MMM(...)
mmm.plot_suite = "new"

cv = TimeSliceCrossValidator(n_init=100, forecast_horizon=12, date_column="date")
cv.plot_suite = "new"
```

Note: until set to `"new"`, accessing `mmm.plot` emits a `FutureWarning`.

### 3. Common Arguments
**Markdown + code cell.**

Largest new section. Explains every shared parameter once; subsequent sections reference these by name without re-explaining.

#### Argument table

| Argument | Type | Default | Description |
|---|---|---|---|
| `idata` | `az.InferenceData \| None` | `None` | Override the model's fitted data for a single call. When `None`, uses the data stored on the model instance. |
| `hdi_prob` | `float` | `0.94` | Credible interval width. Replaces old `hdi_probs: list[float]` — only one level per call. |
| `dims` | `dict[str, Any] \| None` | `None` | Subset coordinates, e.g. `{"channel": ["tv", "radio"]}`. Size-1 dims are preserved as facets rather than squeezed out. |
| `figsize` | `tuple[float, float] \| None` | `None` | Shorthand for `figure_kwargs={"figsize": ...}` passed to `PlotCollection`. |
| `backend` | `str \| None` | `None` (matplotlib) | Rendering backend. Supported values: `"matplotlib"`, `"plotly"`, `"bokeh"`. |
| `return_as_pc` | `bool` | `False` | Return a `PlotCollection` instead of `(Figure, NDArray[Axes])`. Required when `backend` is not `"matplotlib"`. |
| `**pc_kwargs` | — | — | Forwarded to `PlotCollection.wrap()`. Controls `col_wrap`, figure layout, aesthetic mappings. |
| `*_kwargs` | `dict \| None` | `None` | Per-visual-element kwargs. Method-specific names like `line_kwargs`, `hdi_kwargs`, `scatter_kwargs`, forwarded to the underlying arviz-plots visual primitive. |

#### Reference example

```python
fig, axes = mmm.plot.diagnostics.posterior_predictive(
    hdi_prob=0.89,
    dims={"channel": ["tv"]},
    figsize=(10, 4),
    line_kwargs={"color": "blue"},
    hdi_kwargs={"alpha": 0.3},
)
```

### 4. Non-Matplotlib Backends
**Markdown + code cell.**

Backends other than `matplotlib` require `return_as_pc=True`, which returns a `PlotCollection` rather than a `(Figure, axes)` tuple. This is necessary because Plotly and Bokeh figures are not Matplotlib `Figure` objects.

> **Note:** Non-matplotlib backend support has not been fully tested and is likely to contain issues. Use at your own risk.

```python
# Plotly backend — requires return_as_pc=True
pc = mmm.plot.diagnostics.posterior_predictive(
    backend="plotly",
    return_as_pc=True,
)
pc.show()
```

Note: `waterfall()` in the decomposition namespace always returns a matplotlib `(Figure, axes)` tuple and does not support the `backend` or `return_as_pc` parameters.

### 5. Namespace Map

Four subsections, one per namespace. Each has:
- A table of old → new **method names**
- A table of old → new **argument names** for that namespace
- One reference code snippet

#### 5a. Diagnostics (`mmm.plot.diagnostics`)

**Method names:**

| Legacy | New |
|---|---|
| `mmm.plot.posterior_predictive(...)` | `mmm.plot.diagnostics.posterior_predictive(...)` |
| `mmm.plot.prior_predictive(...)` | `mmm.plot.diagnostics.prior_predictive(...)` |
| `mmm.plot.residuals_over_time(...)` | `mmm.plot.diagnostics.residuals_over_time(...)` |
| `mmm.plot.residuals_posterior_distribution(...)` | `mmm.plot.diagnostics.residuals_distribution(...)` |
| `mmm.plot.posterior_distribution(...)` | `mmm.plot.diagnostics.posterior(...)` |
| `mmm.plot.prior_vs_posterior(...)` | `mmm.plot.diagnostics.prior_vs_posterior(...)` |

**Argument changes:**

| Old | New | Notes |
|---|---|---|
| `hdi_probs: list[float]` | `hdi_prob: float` | Single level per call |
| `var: list[str]` (on `posterior_distribution`) | `var_names: list[str] \| str \| None` | Also accepts a single string |
| — | `group: str = "posterior"` | New on `posterior()` — can pass `"prior"` to plot prior instead |
| — | `kind: str = "kde"` | New on `posterior()` and `prior_vs_posterior()` — controls plot type |
| — | `aggregation` | New on `residuals_distribution()` — dimension to aggregate over |
| — | `quantiles` | New on `residuals_distribution()` — quantile lines to overlay |

**Example:**

```python
fig, axes = mmm.plot.diagnostics.posterior_predictive(hdi_prob=0.94)
fig, axes = mmm.plot.diagnostics.posterior(var_names=["alpha", "beta"], kind="kde")
fig, axes = mmm.plot.diagnostics.residuals_distribution(quantiles=[0.025, 0.5, 0.975])
```

#### 5b. Decomposition (`mmm.plot.decomposition`)

**Method names:**

| Legacy | New |
|---|---|
| `mmm.plot.contributions_over_time(...)` | `mmm.plot.decomposition.contributions_over_time(...)` |
| `mmm.plot.waterfall_components_decomposition(...)` | `mmm.plot.decomposition.waterfall(...)` |
| `mmm.plot.channel_parameter(...)` | `mmm.plot.decomposition.channel_share_hdi(...)` |

**Argument changes:**

| Old | New | Notes |
|---|---|---|
| `hdi_probs: list[float]` | `hdi_prob: float` | Single level per call |
| `original_scale: bool = False` | `original_scale: bool = True` | Default flipped to True |
| — | `include: list[Literal["channels", "baseline", "controls", "seasonality"]] \| None` | New on `contributions_over_time()` — filter which components appear |

Note: `waterfall()` does not support `backend` or `return_as_pc` — it always returns `(Figure, NDArray[Axes])`.

**Example:**

```python
fig, axes = mmm.plot.decomposition.contributions_over_time(
    include=["channels", "baseline"],
    original_scale=True,
)
fig, axes = mmm.plot.decomposition.waterfall(original_scale=True)
fig, axes = mmm.plot.decomposition.channel_share_hdi(hdi_prob=0.89)
```

#### 5c. Sensitivity (`mmm.plot.sensitivity`)

**Method names:**

| Legacy | New |
|---|---|
| `mmm.plot.sensitivity_analysis(...)` | `mmm.plot.sensitivity.analysis(...)` |
| `mmm.plot.uplift_curve(...)` | `mmm.plot.sensitivity.uplift(...)` |
| `mmm.plot.marginal_curve(...)` | `mmm.plot.sensitivity.marginal(...)` |

**Argument changes:**

| Old | New | Notes |
|---|---|---|
| `hdi_probs: list[float]` | `hdi_prob: float` | Single level per call |
| — | `x_sweep_axis: Literal["relative", "absolute"] = "relative"` | New — controls x-axis scale |
| — | `apply_cost_per_unit: bool = True` | New — scale spend axis by cost-per-unit |
| — | `aggregation: dict[str, str \| list[str]] \| None` | New — aggregate over dimensions before plotting |

**Example:**

```python
fig, axes = mmm.plot.sensitivity.analysis(
    x_sweep_axis="relative",
    apply_cost_per_unit=True,
    hdi_prob=0.94,
)
```

#### 5d. Transformation (`mmm.plot.transformation`)

**Method names:**

| Legacy | New |
|---|---|
| `mmm.plot.saturation_scatterplot(...)` | `mmm.plot.transformation.saturation_scatterplot(...)` |
| `mmm.plot.saturation_curves(...)` | `mmm.plot.transformation.saturation_curves(...)` |

**Argument changes:**

| Old | New | Notes |
|---|---|---|
| `original_scale: bool = False` | `original_scale: bool = True` | Default flipped to True |
| `hdi_probs: list[float]` | `hdi_prob: float \| None = 0.94` | Single level; pass `None` to suppress HDI band |
| — | `apply_cost_per_unit: bool = True` | New on both methods |
| — | `n_samples: int = 10` | New on `saturation_curves()` — number of posterior draws to overlay |
| — | `random_seed` | New on `saturation_curves()` — for reproducible sample selection |
| — | `mean_curve_kwargs` | New on `saturation_curves()` — style the mean curve separately |
| — | `sample_curves_kwargs` | New on `saturation_curves()` — style individual sample curves |
| — | `curves: xr.DataArray` | Now required positional arg on `saturation_curves()` |

**Example:**

```python
fig, axes = mmm.plot.transformation.saturation_scatterplot(original_scale=True)
fig, axes = mmm.plot.transformation.saturation_curves(
    curves=saturation_curve_data,
    n_samples=20,
    hdi_prob=0.94,
)
```

### 6. Budget Plots
**Markdown + code cell.**

Budget plots have moved from `mmm.plot` to `optimizer.plot`. The `BudgetPlots` namespace is accessed via `BudgetOptimizerWrapper` and is stateless — all data is passed per-call via `samples`.

```python
from pymc_marketing.mmm import BudgetOptimizerWrapper, MMM

mmm = MMM(...)
mmm.plot_suite = "new"
optimizer = BudgetOptimizerWrapper(model=mmm, start_date="2024-01-01", end_date="2024-12-31")
samples = optimizer.allocate_budget(...)

fig, axes = optimizer.plot.allocation_roas(samples=samples, hdi_prob=0.94)
fig, axes = optimizer.plot.contribution_over_time(samples=samples, hdi_prob=0.94)
```

**Argument changes from legacy:**

| Old | New | Notes |
|---|---|---|
| `hdi_probs: list[float]` | `hdi_prob: float` | Single level per call |
| — | `samples: xr.Dataset` | Required arg — output of `allocate_budget()` |

### 7. Cross-Validation Plots
**Markdown + code cell.**

CV plots use `MMMCVPlotSuite` when `cv.plot_suite = "new"`.

```python
cv = TimeSliceCrossValidator(n_init=100, forecast_horizon=12, date_column="date")
cv.plot_suite = "new"
cv_idata = cv.run(X, y, mmm=mmm)

fig, axes = cv.plot.predictions(cv_idata)
fig, axes = cv.plot.param_stability(cv_idata, var_names=["alpha"])
fig, axes = cv.plot.crps(cv_idata)
```

**Argument changes from legacy:**

| Old | New | Notes |
|---|---|---|
| `hdi_probs: list[float]` | `hdi_prob: float` | Single level per call |
| — | `var_names: list[str] \| None` | New on `param_stability()` — filter which parameters appear |

### 8. Removal Timeline
**Markdown only.**

The legacy `MMMPlotSuite` (the default when `mmm.plot_suite` is not set) will be **removed in pymc-marketing 2.0.0**. Until then, accessing `mmm.plot` without opting in emits a `FutureWarning`.

To suppress it: set `mmm.plot_suite = "new"`.

---

## Out of scope

- Runnable code (notebook cells will not execute against real model fits)
- API reference docs (docstrings cover individual method details)
- Plotting gallery or worked examples (separate notebook)
