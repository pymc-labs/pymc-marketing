# MMM Plotting Functionality - Executive Summary

## Quick Answer

**Are there missing plotting functions in `mmm.plot.MMMPlotSuite`?**

**Yes** - There are **4 key plotting methods** from the legacy MMM classes that are not yet available in `MMMPlotSuite`:

| # | Missing Method | Priority | What It Does |
|---|----------------|----------|--------------|
| 1 | `plot_grouped_contribution_breakdown_over_time` | ⚠️ HIGH | Stacked area chart of contributions over time with grouping |
| 2 | `plot_prior_vs_posterior` | ⚠️ HIGH | Side-by-side KDE plots comparing prior vs posterior |
| 3 | `plot_channel_contribution_grid` | 🔶 MEDIUM | Grid showing contributions at different spending levels |
| 4 | `plot_new_spend_contributions` | 🔶 MEDIUM | Temporal effects of new spend accounting for adstock |

## Full Details

See the comprehensive analysis: [MMM Plotting Functionality Comparison](./mmm_plotting_functionality_comparison.md)

## What's Already There

`MMMPlotSuite` has excellent coverage with **17 plotting methods**, including:
- ✅ Posterior/prior predictive distributions
- ✅ Residuals analysis (over time + distribution)
- ✅ Contribution analysis over time
- ✅ Saturation curves (scatter + curves)
- ✅ Budget allocation
- ✅ Waterfall decomposition
- ✅ Channel contribution share
- ✅ Sensitivity analysis (new!)
- ✅ Uplift & marginal curves (new!)

## Recommendation

**For high-priority use cases** requiring the 4 missing methods, continue using:
- `mmm.plot_grouped_contribution_breakdown_over_time()`
- `mmm.plot_prior_vs_posterior()`
- `mmm.plot_channel_contribution_grid()`
- `mmm.plot_new_spend_contributions()`

**For everything else**, use the modern `mmm.plot.*` interface, which provides:
- Better multi-dimensional support
- Consistent API
- Works with any PyMC model
- Active development

---

**Related Issue:** [Add missing plotting functionality](../issues/XXX)  
**Full Analysis:** [mmm_plotting_functionality_comparison.md](./mmm_plotting_functionality_comparison.md)  
**Last Updated:** 2025-12-09
