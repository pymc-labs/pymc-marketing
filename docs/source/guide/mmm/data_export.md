# Exporting MMM data for frontends

Production dashboards often render charts in JavaScript (React, Dash, Streamlit, custom UIs) rather than serving static matplotlib images from Python. PyMC-Marketing exposes **tabular summaries** with mean, median, and HDI bounds — the same statistics underlying the plot suite — so your frontend owns presentation.

Issue [#2715](https://github.com/pymc-labs/pymc-marketing/issues/2715) originally proposed a `return_data=True` flag on plot methods. The shipped design is the **`mmm.summary`** layer instead: one source of truth for both matplotlib/Plotly plots and JSON export.

## Quick start

After fitting an `MMM`:

```python
from pymc_marketing.mmm.summary import dataframe_to_json_records

# Pandas records (simple path)
records = mmm.summary.contributions().to_dict(orient="records")

# JSON-safe records (ISO dates, native Python scalars)
df = mmm.summary.contributions()
records = dataframe_to_json_records(df)
```

Pass `records` to `json.dumps` or your API response layer.

## Core API: `mmm.summary`

`mmm.summary` returns an `MMMSummaryFactory` with defaults for HDI levels and output format:

```python
df = mmm.summary.contributions()                    # default: pandas, 94% HDI
df = mmm.summary.roas(frequency="yearly")
df = mmm.summary.posterior_predictive(
    hdi_probs=[0.80, 0.94],
    frequency="monthly",
    output_format="polars",
)
```

Common parameters across methods:

- **`hdi_probs`** — HDI probability levels (default `(0.94,)`)
- **`output_format`** — `"pandas"` or `"polars"`
- **`frequency`** — time aggregation (`"weekly"`, `"monthly"`, `"yearly"`, etc.)
- **`dims`** — dimension filters, e.g. `{"geo": ["CA"]}`

See the full API: [`MMMSummaryFactory`](../../api/generated/pymc_marketing.mmm.summary.MMMSummaryFactory.html).

### Methods overview

| Method | What it summarizes |
|--------|-------------------|
| `contributions()` | Per-channel / control / seasonality contributions over time |
| `waterfall()` | Component totals and shares (waterfall decomposition) |
| `channel_share_hdi()` | Channel share of total contribution |
| `posterior_predictive()` | Posterior predictive vs observed |
| `prior_predictive()` | Prior predictive vs observed |
| `residuals_over_time()` | Residuals with HDI bands |
| `residuals_distribution()` | Residual distribution quantiles |
| `prior_vs_posterior()` | Prior vs posterior density grid |
| `roas()` | Return on ad spend |
| `channel_spend()` | Raw spend per channel/date |
| `saturation_curves()` | Saturation response curves |
| `adstock_curves()` | Adstock decay curves |
| `saturation_scatterplot()` | Spend vs saturated effect |
| `total_contribution()` | Summed contributions by component type |
| `change_over_time()` | Percentage change between periods |
| `sensitivity_analysis()` | Raw sensitivity sweep |
| `sensitivity_uplift()` | Uplift curves from sensitivity sweep |
| `sensitivity_marginal()` | Marginal effects from sensitivity sweep |

## Plot ↔ summary mapping

If you know the matplotlib plot API, use the matching summary method for tabular export:

| Plot entry point | Summary entry point |
|------------------|---------------------|
| `mmm.plot.decomposition.contributions_over_time` | `mmm.summary.contributions` |
| `mmm.plot.decomposition.waterfall` | `mmm.summary.waterfall` |
| `mmm.plot.decomposition.channel_share_hdi` | `mmm.summary.channel_share_hdi` |
| `mmm.plot.diagnostics.posterior_predictive` | `mmm.summary.posterior_predictive` |
| `mmm.plot.diagnostics.prior_predictive` | `mmm.summary.prior_predictive` |
| `mmm.plot.diagnostics.residuals_over_time` | `mmm.summary.residuals_over_time` |
| `mmm.plot.diagnostics.residuals_distribution` | `mmm.summary.residuals_distribution` |
| `mmm.plot.diagnostics.prior_vs_posterior` | `mmm.summary.prior_vs_posterior` |
| `mmm.plot.transformation.saturation_scatterplot` | `mmm.summary.saturation_scatterplot` |
| `mmm.plot.transformation.saturation_curves` | `mmm.summary.saturation_curves` |
| `mmm.plot.sensitivity.analysis` | `mmm.summary.sensitivity_analysis` |
| `mmm.plot.sensitivity.uplift` | `mmm.summary.sensitivity_uplift` |
| `mmm.plot.sensitivity.marginal` | `mmm.summary.sensitivity_marginal` |
| `optimizer.plot.allocation_roas(samples)` | `optimizer.summary.allocation_roas(samples)` |
| `optimizer.plot.contribution_over_time(samples)` | `optimizer.summary.contribution_over_time(samples)` |
| `cv.plot.predictions(cv_idata)` | `cv.summary.predictions()` |
| `cv.plot.param_stability(cv_idata)` | `cv.summary.param_stability()` |
| `cv.plot.crps(cv_idata)` | `cv.summary.crps()` |

## Budget allocation samples

Budget plots are stateless: pass allocation samples to both plot and summary namespaces.

```python
optimizer = BudgetOptimizerWrapper(model=mmm, start_date="2024-01-01", end_date="2024-12-31")
samples = optimizer.allocate_budget(...)

df_roas = optimizer.summary.allocation_roas(samples=samples)
df_ts = optimizer.summary.contribution_over_time(samples=samples)
records = df_roas.to_dict(orient="records")
```

Samples come from `allocate_budget()` or `mmm.sample_response_distribution(...)`.

## Cross-validation results

After `TimeSliceCrossValidator.run()`:

```python
cv_idata = cv.run(X, y, mmm=mmm)

df_pred = cv.summary.predictions()
df_stab = cv.summary.param_stability(var_names=["alpha"])
df_crps = cv.summary.crps()
```

When you already hold an `MMMCVPlotSuite`, `cv.plot.summary` exposes the same factory bound to the CV DataTree.

## Interactive Plotly path

`mmm.plot_interactive` reads from the same summary DataFrames. Use it for quick Plotly exploration, or call `mmm.summary` directly when you need raw tables for a custom frontend.

See the [Interactive MMM Visualizations notebook](../../notebooks/mmm/plot_interactive.html).

## Release notes (copy-paste)

Use these bullets in GitHub release notes under **New Features**:

- Expanded `mmm.summary` with decomposition, diagnostics, sensitivity, and transformation summaries as JSON-serializable DataFrames for custom frontends
- Added `optimizer.summary` and `cv.plot.summary` parity with budget and CV plot APIs
- New guide: [Exporting MMM data for frontends](data_export.html)
