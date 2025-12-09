# MMM Plotting Functionality Comparison

> **Quick Summary:** See [MMM_PLOTTING_SUMMARY.md](./MMM_PLOTTING_SUMMARY.md) for an executive summary with just the key findings.

This document provides a comprehensive comparison of plotting functionality across the MMM codebase, specifically comparing what's available in `mmm.base.MMMModelBuilder`, `mmm.mmm.MMM`, and `mmm.plot.MMMPlotSuite`.

## Summary

The `MMMPlotSuite` class (in `mmm.plot.py`) was designed to provide a modern, flexible plotting interface that works with any PyMC model's `InferenceData` object. However, the legacy `MMM` class and its base classes still contain several plotting methods that have not been migrated to `MMMPlotSuite`.

## Comparison Table

| Plotting Method | mmm.base.MMMModelBuilder | mmm.mmm.MMM | mmm.plot.MMMPlotSuite | Notes |
|----------------|--------------------------|-------------|----------------------|-------|
| **Predictive Distributions** |
| `plot_prior_predictive` | ✅ | ✅ (inherited) | ✅ `prior_predictive` | Available in MMMPlotSuite with different signature |
| `plot_posterior_predictive` | ✅ | ✅ (inherited) | ✅ `posterior_predictive` | Available in MMMPlotSuite with different signature |
| **Errors/Residuals** |
| `plot_errors` | ✅ | ✅ (inherited) | ✅ `residuals_over_time` + `residuals_posterior_distribution` | Split into two methods in MMMPlotSuite |
| **Component Contributions** |
| `plot_components_contributions` | ✅ | ✅ (overridden) | ✅ `contributions_over_time` | Available but with different interface |
| `plot_grouped_contribution_breakdown_over_time` | ✅ | ✅ (inherited) | ❌ **MISSING** | Stacked area chart for contributions over time |
| **Channel Analysis** |
| `plot_channel_contribution_share_hdi` | ✅ | ✅ (inherited) | ✅ `channel_contribution_share_hdi` | Available in MMMPlotSuite |
| `plot_prior_vs_posterior` | ✅ | ✅ (inherited) | ❌ **MISSING** | KDE plots comparing prior vs posterior by channel |
| `plot_channel_parameter` | ❌ | ✅ | ✅ `posterior_distribution` | Similar functionality available |
| `plot_channel_contribution_grid` | ❌ | ✅ | ❌ **MISSING** | Grid of contributions for different spending scenarios |
| **Direct Contribution/Saturation Curves** |
| `plot_direct_contribution_curves` | ❌ | ✅ | ✅ `saturation_scatterplot` + `saturation_curves` | Split into scatter and curve plots |
| **Decomposition/Waterfall** |
| `plot_waterfall_components_decomposition` | ✅ | ✅ (inherited) | ✅ `waterfall_components_decomposition` | Available in MMMPlotSuite |
| **Budget Optimization** |
| `plot_new_spend_contributions` | ❌ | ✅ | ❌ **MISSING** | Upcoming sales for given spend |
| `plot_budget_allocation` | ❌ | ✅ | ✅ `budget_allocation` | Available in MMMPlotSuite |
| `plot_allocated_contribution_by_channel` | ❌ | ✅ | ✅ `allocated_contribution_by_channel_over_time` | Available in MMMPlotSuite |
| **Sensitivity Analysis** |
| N/A | ❌ | ❌ | ✅ `sensitivity_analysis` | New in MMMPlotSuite |
| N/A | ❌ | ❌ | ✅ `uplift_curve` | New in MMMPlotSuite |
| N/A | ❌ | ❌ | ✅ `marginal_curve` | New in MMMPlotSuite |

## Missing Functionality Details

### 1. `plot_grouped_contribution_breakdown_over_time` (HIGH PRIORITY)

**Location:** `mmm.base.MMMModelBuilder`

**Description:** Creates a stacked area chart showing how different model components (channels, controls, intercept, seasonality) contribute to the target over time. Allows grouping variables together for cleaner visualization.

**Key Features:**
- Stacked area chart visualization
- Ability to group components using `stack_groups` parameter
- Works in both original and scaled space
- Useful for understanding the relative importance of different components over time

**Example Usage:**
```python
mmm.plot_grouped_contribution_breakdown_over_time(
    stack_groups={
        "Baseline": ["intercept"],
        "Offline": ["TV", "Radio"],
        "Online": ["Banners"]
    },
    original_scale=True
)
```

**Why Missing:** This is a valuable visualization for understanding component contributions over time, especially when there are many channels. The `contributions_over_time` method in `MMMPlotSuite` doesn't provide the stacked area chart visualization.

---

### 2. `plot_prior_vs_posterior` (HIGH PRIORITY)

**Location:** `mmm.base.MMMModelBuilder`

**Description:** Creates KDE plots comparing prior predictive and posterior distributions for a specified parameter (e.g., `adstock_alpha`, `saturation_beta`) across all channels.

**Key Features:**
- Side-by-side KDE plots for each channel
- Shows prior mean and posterior mean with difference
- Can sort by alphabetical or by largest difference
- Helps assess model learning and parameter updates

**Example Usage:**
```python
mmm.plot_prior_vs_posterior(
    var_name='adstock_alpha',
    alphabetical_sort=False  # Sort by largest difference
)
```

**Why Missing:** This is crucial for model diagnostics and understanding how much the model learned from the data. There's no equivalent in `MMMPlotSuite`.

---

### 3. `plot_channel_contribution_grid` (MEDIUM PRIORITY)

**Location:** `mmm.mmm.MMM`

**Description:** Plots a grid of channel contributions for different spending scenarios (scaled by a multiplier δ). Shows how contributions would change if spending was increased/decreased.

**Key Features:**
- Shows contribution vs spending multiplier
- HDI bands for uncertainty
- Can plot in absolute or relative (percentage) scale
- Useful for "what-if" scenarios

**Example Usage:**
```python
mmm.plot_channel_contribution_grid(
    start=0.5,  # 50% of current spend
    stop=1.5,   # 150% of current spend
    num=10,     # 10 points
    absolute_xrange=False
)
```

**Why Missing:** While `saturation_curves` in `MMMPlotSuite` provides similar functionality, it requires more setup and doesn't have the same grid-based interface.

---

### 4. `plot_new_spend_contributions` (MEDIUM PRIORITY)

**Location:** `mmm.mmm.MMM`

**Description:** Visualizes the effect of a new spend amount over time, accounting for adstock effects (carry-over). Shows how a one-time or continuous spend impacts sales.

**Key Features:**
- Shows time-since-spend on x-axis
- Accounts for adstock lag effects
- Can plot one-time or continuous spending
- Confidence intervals for uncertainty
- Separate lines for each channel

**Example Usage:**
```python
mmm.plot_new_spend_contributions(
    spend_amount=100,
    one_time=True,  # One-time spend
    lower=0.025,
    upper=0.975,
    channels=['TV', 'Radio']
)
```

**Why Missing:** This is specific to the `MMM` class's `new_spend_contributions` method and provides valuable insights into the temporal dynamics of marketing effects. There's no equivalent in `MMMPlotSuite`.

---

## Recommendations

### Short Term (High Priority)
1. **Add `plot_grouped_contribution_breakdown_over_time` equivalent**
   - Implement as `contributions_breakdown_area` or enhance existing `contributions_over_time` with `plot_type='area'` and `stack_groups` parameter
   - This is widely used for stakeholder presentations

2. **Add `plot_prior_vs_posterior` equivalent**
   - Implement as `prior_posterior_comparison` 
   - Essential for model diagnostics and validation

### Medium Term
3. **Add `plot_channel_contribution_grid` equivalent**
   - Could be enhanced version of existing `saturation_curves` 
   - Or implement as separate `contribution_grid` method

4. **Add `plot_new_spend_contributions` equivalent**
   - Implement as `new_spend_effects` or `temporal_contribution_analysis`
   - Would require coordination with the budget optimization functionality

### Long Term
5. **Deprecation Strategy**
   - Consider deprecating plot methods in `mmm.base` and `mmm.mmm` once equivalents are in `MMMPlotSuite`
   - Ensure backward compatibility during transition period
   - Update documentation to guide users to `MMMPlotSuite`

## Design Considerations

### Why `MMMPlotSuite` is Better

1. **Separation of Concerns:** Plotting logic is separate from model logic
2. **Works with Any PyMC Model:** Not tied to specific MMM implementation
3. **Consistent Interface:** All plots follow similar patterns
4. **Better Multi-dimensional Support:** Handles arbitrary dimensions naturally
5. **Cleaner Code:** No plot-specific state in model classes

### Migration Challenges

1. **Different Signatures:** `MMMPlotSuite` methods have different parameters than legacy methods
2. **Data Requirements:** Some legacy plots rely on model-specific attributes that aren't in `InferenceData`
3. **User Familiarity:** Existing users may expect the old interface
4. **Documentation:** Need to update examples and tutorials

## Testing Coverage

Current test coverage for plotting:
- `tests/mmm/test_plotting.py` - Tests for legacy MMM plotting methods
- Need tests for MMMPlotSuite methods for missing functionality

## Conclusion

The `MMMPlotSuite` provides excellent coverage for most common plotting needs, but there are 4 key methods from the legacy classes that are either missing or not easily accessible:

1. ❌ **Missing:** `plot_grouped_contribution_breakdown_over_time` (stacked area chart)
2. ❌ **Missing:** `plot_prior_vs_posterior` (prior vs posterior KDE comparison)
3. ❌ **Missing:** `plot_channel_contribution_grid` (contribution vs spending grid)
4. ❌ **Missing:** `plot_new_spend_contributions` (temporal effects of new spend)

Adding these would provide complete feature parity and allow for eventual deprecation of the plotting methods in the base classes.

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-09  
**Author:** GitHub Copilot Analysis
