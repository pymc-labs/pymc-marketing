---
date: 2025-12-09T15:34:51+00:00
researcher: Claude Sonnet 4.5
git_commit: 021b29b8f5743dd087153b2bdade19652111c1a1
branch: work-issue-2128
repository: pymc-marketing
topic: "Missing plotting functionality in mmm.multidimension.MMM / mmm.plot.MMMPlotSuite"
tags: [research, codebase, mmm, plotting, visualization, issue-2128]
status: complete
last_updated: 2025-12-09
last_updated_by: Claude Sonnet 4.5
issue_number: 2128
---

# Research: Missing plotting functionality in mmm.multidimension.MMM / mmm.plot.MMMPlotSuite

**Date**: 2025-12-09T15:34:51+00:00
**Researcher**: Claude Sonnet 4.5
**Git Commit**: 021b29b8f5743dd087153b2bdade19652111c1a1
**Branch**: work-issue-2128
**Repository**: pymc-marketing
**Issue**: #2128

## Research Question

What plotting functionality exists in `mmm.base` and `mmm.mmm` modules that didn't make it into the `mmm.plot.MMMPlotSuite`? Create a comparison table to understand what is missing and how to handle it.

## Summary

The pymc-marketing library has two MMM implementations:
1. **Legacy MMM** (`mmm/mmm.py`) - inherits plotting methods from `mmm/base.py` directly
2. **Multidimensional MMM** (`mmm/multidimensional.py`) - uses `MMMPlotSuite` as a property

Analysis reveals **4 plotting methods are missing** from `MMMPlotSuite`:
1. `plot_grouped_contribution_breakdown_over_time` (from base.py)
2. `plot_prior_vs_posterior` (from base.py)
3. `plot_channel_contribution_grid` (from mmm.py)
4. `plot_new_spend_contributions` (from mmm.py)

The multidimensional MMM implementation uses MMMPlotSuite as a property (`mmm.plot`), making it incompatible with these methods that were available in the legacy implementation.

## Detailed Findings

### Architecture Overview

#### Legacy MMM Class (`mmm/mmm.py`)
- **Inheritance**: Multiple inheritance with mixins
- **Base classes**: `MaxAbsScaleTarget`, `MaxAbsScaleChannels`, `ValidateControlColumns`, `BaseMMM`
  - `BaseMMM` → `BaseValidateMMM` → `MMMModelBuilder` → `RegressionModelBuilder`
- **Plotting approach**: Methods inherited directly into class
- **MMMPlotSuite usage**: Not used

#### Multidimensional MMM Class (`multidimensional.py`)
- **Inheritance**: Single inheritance from `RegressionModelBuilder`
- **Plotting approach**: Composition pattern via `plot` property (`multidimensional.py:619-623`)
- **MMMPlotSuite usage**:
  ```python
  @property
  def plot(self) -> MMMPlotSuite:
      """Use the MMMPlotSuite to plot the results."""
      self._validate_model_was_built()
      self._validate_idata_exists()
      return MMMPlotSuite(idata=self.idata)
  ```

### Complete Functionality Comparison

| Functionality | base.py | mmm.py | MMMPlotSuite | Status |
|---------------|---------|--------|--------------|--------|
| **Prior predictive** | ✓ `plot_prior_predictive` (573-633) | - | ✓ `prior_predictive` (414-502) | ✓ Covered |
| **Posterior predictive** | ✓ `plot_posterior_predictive` (635-698) | - | ✓ `posterior_predictive` (504-591) | ✓ Covered |
| **Errors/Residuals** | ✓ `plot_errors` (761-815) | - | ✓ `residuals_over_time` (636-775)<br>✓ `residuals_posterior_distribution` (777-924) | ✓ Covered (enhanced) |
| **Components contributions** | ✓ `plot_components_contributions` (828-933) | ✓ Override (1681-1788) | ✓ `contributions_over_time` (926-1049) | ✓ Covered |
| **Grouped contribution breakdown** | ✓ `plot_grouped_contribution_breakdown_over_time` (1068-1138) | - | - | **✗ MISSING** |
| **Channel contribution share** | ✓ `plot_channel_contribution_share_hdi` (1161-1193) | - | ✓ `channel_contribution_share_hdi` (2761-2887) | ✓ Covered |
| **Prior vs posterior comparison** | ✓ `plot_prior_vs_posterior` (1225-1369) | - | - | **✗ MISSING** |
| **Waterfall decomposition** | ✓ `plot_waterfall_components_decomposition` (1371-1457) | - | ✓ `waterfall_components_decomposition` (2623-2759) | ✓ Covered |
| **Channel parameter distribution** | - | ✓ `plot_channel_parameter` (1531-1578) | ✓ `posterior_distribution` (1051-1234) | ✓ Covered (generalized) |
| **Channel contribution grid** | - | ✓ `plot_channel_contribution_grid` (1790-1874) | - | **✗ MISSING** |
| **New spend contributions** | - | ✓ `plot_new_spend_contributions` (1992-2091) | - | **✗ MISSING** |
| **Direct contribution curves** | - | ✓ `plot_direct_contribution_curves` (2277-2390) | ✓ `saturation_scatterplot` (1236-1388)<br>✓ `saturation_curves` (1390-1642) | ✓ Covered (similar) |
| **Budget allocation** | - | ✓ `plot_budget_allocation` (3030-3112) | ✓ `budget_allocation` (1683-1858) | ✓ Covered |
| **Allocated contribution by channel** | - | ✓ `plot_allocated_contribution_by_channel` (3114-3169) | ✓ `allocated_contribution_by_channel_over_time` (1925-2127) | ✓ Covered |
| **Sensitivity analysis** | - | - | ✓ `sensitivity_analysis` (2129-2364) | New in MMMPlotSuite |
| **Uplift curve** | - | - | ✓ `uplift_curve` (2366-2466) | New in MMMPlotSuite |
| **Marginal curve** | - | - | ✓ `marginal_curve` (2468-2569) | New in MMMPlotSuite |

### Missing Methods Detail

#### 1. plot_grouped_contribution_breakdown_over_time
- **Source**: `pymc_marketing/mmm/base.py:1068-1138`
- **Purpose**: Plot a time series area chart for all channel contributions with optional grouping
- **Key features**:
  - Groups variables together using `stack_groups` parameter
  - Creates stacked area chart showing contribution over time
  - Useful for crowded charts with many channels/controls
- **Signature**:
  ```python
  def plot_grouped_contribution_breakdown_over_time(
      self,
      stack_groups: dict[str, list[str]] | None = None,
      original_scale: bool = False,
      area_kwargs: dict[str, Any] | None = None,
      **plt_kwargs: Any,
  ) -> plt.Figure
  ```

#### 2. plot_prior_vs_posterior
- **Source**: `pymc_marketing/mmm/base.py:1225-1369`
- **Purpose**: Compare prior and posterior distributions for a specified variable
- **Key features**:
  - Creates KDE plots for each channel
  - Shows prior predictive and posterior distributions with means
  - Can sort alphabetically or by difference (posterior - prior mean)
  - Useful for understanding model learning
- **Signature**:
  ```python
  def plot_prior_vs_posterior(
      self,
      var_name: str,
      alphabetical_sort: bool = True,
      figsize: tuple[int, int] | None = None,
  ) -> plt.Figure
  ```

#### 3. plot_channel_contribution_grid
- **Source**: `pymc_marketing/mmm/mmm.py:1790-1874`
- **Purpose**: Plot a grid of scaled channel contributions for a given grid of share values
- **Key features**:
  - Shows how contributions change across spend scenarios
  - Creates line plots with HDI bands
  - X-axis can show relative percentage or absolute input units
- **Signature**:
  ```python
  def plot_channel_contribution_grid(
      self,
      start: float,
      stop: float,
      num: int,
      absolute_xrange: bool = False,
      **plt_kwargs: Any,
  ) -> plt.Figure
  ```

#### 4. plot_new_spend_contributions
- **Source**: `pymc_marketing/mmm/mmm.py:1992-2091`
- **Purpose**: Plot upcoming sales for a given spend amount
- **Key features**:
  - Visualizes contributions over time since spend
  - Supports one-time and continuous spend scenarios
  - Shows confidence intervals
  - Calls `new_spend_contributions` method internally
- **Signature**:
  ```python
  def plot_new_spend_contributions(
      self,
      spend_amount: float,
      one_time: bool = True,
      lower: float = 0.025,
      upper: float = 0.975,
      ylabel: str = "Sales",
      idx: slice | None = None,
      channels: list[str] | None = None,
      prior: bool = False,
      original_scale: bool = True,
      ax: plt.Axes | None = None,
      **sample_posterior_predictive_kwargs,
  ) -> plt.Axes
  ```

## Code References

### Base MMM Methods (mmm/base.py)
- `pymc_marketing/mmm/base.py:573-633` - plot_prior_predictive
- `pymc_marketing/mmm/base.py:635-698` - plot_posterior_predictive
- `pymc_marketing/mmm/base.py:761-815` - plot_errors
- `pymc_marketing/mmm/base.py:828-933` - plot_components_contributions
- `pymc_marketing/mmm/base.py:1068-1138` - plot_grouped_contribution_breakdown_over_time ⚠️
- `pymc_marketing/mmm/base.py:1161-1193` - plot_channel_contribution_share_hdi
- `pymc_marketing/mmm/base.py:1225-1369` - plot_prior_vs_posterior ⚠️
- `pymc_marketing/mmm/base.py:1371-1457` - plot_waterfall_components_decomposition

### Legacy MMM Methods (mmm/mmm.py)
- `pymc_marketing/mmm/mmm.py:1531-1578` - plot_channel_parameter
- `pymc_marketing/mmm/mmm.py:1681-1788` - plot_components_contributions [override]
- `pymc_marketing/mmm/mmm.py:1790-1874` - plot_channel_contribution_grid ⚠️
- `pymc_marketing/mmm/mmm.py:1992-2091` - plot_new_spend_contributions ⚠️
- `pymc_marketing/mmm/mmm.py:2277-2390` - plot_direct_contribution_curves
- `pymc_marketing/mmm/mmm.py:3030-3112` - plot_budget_allocation
- `pymc_marketing/mmm/mmm.py:3114-3169` - plot_allocated_contribution_by_channel

### MMMPlotSuite Methods (mmm/plot.py)
- `pymc_marketing/mmm/plot.py:414-502` - posterior_predictive
- `pymc_marketing/mmm/plot.py:504-591` - prior_predictive
- `pymc_marketing/mmm/plot.py:636-775` - residuals_over_time
- `pymc_marketing/mmm/plot.py:777-924` - residuals_posterior_distribution
- `pymc_marketing/mmm/plot.py:926-1049` - contributions_over_time
- `pymc_marketing/mmm/plot.py:1051-1234` - posterior_distribution
- `pymc_marketing/mmm/plot.py:1236-1388` - saturation_scatterplot
- `pymc_marketing/mmm/plot.py:1390-1642` - saturation_curves
- `pymc_marketing/mmm/plot.py:1683-1858` - budget_allocation
- `pymc_marketing/mmm/plot.py:1925-2127` - allocated_contribution_by_channel_over_time
- `pymc_marketing/mmm/plot.py:2129-2364` - sensitivity_analysis
- `pymc_marketing/mmm/plot.py:2366-2466` - uplift_curve
- `pymc_marketing/mmm/plot.py:2468-2569` - marginal_curve
- `pymc_marketing/mmm/plot.py:2623-2759` - waterfall_components_decomposition
- `pymc_marketing/mmm/plot.py:2761-2887` - channel_contribution_share_hdi

### Class Definitions
- `pymc_marketing/mmm/base.py:55` - MMMModelBuilder class
- `pymc_marketing/mmm/mmm.py:70` - BaseMMM class
- `pymc_marketing/mmm/mmm.py:1300` - MMM class (legacy)
- `pymc_marketing/mmm/multidimensional.py:219` - MMM class (multidimensional)
- `pymc_marketing/mmm/multidimensional.py:619` - plot property using MMMPlotSuite
- `pymc_marketing/mmm/plot.py:206` - MMMPlotSuite class

⚠️ = Missing from MMMPlotSuite

## Architecture Insights

### Design Pattern Evolution

The library is transitioning from an **inheritance-based** plotting approach to a **composition-based** approach:

**Old Pattern (mmm.py)**: Plot methods are inherited into the MMM class through the inheritance chain
```
MMM → BaseMMM → BaseValidateMMM → MMMModelBuilder
```

**New Pattern (multidimensional.py)**: Plot methods are accessed through a separate plotting suite
```
MMM.plot → MMMPlotSuite(idata=self.idata)
```

### Benefits of the New Pattern
1. **Separation of concerns**: Plotting logic is isolated from model logic
2. **Reusability**: Any PyMC model with compatible idata can use MMMPlotSuite
3. **Lazy initialization**: Plotting suite only created when needed
4. **Flexibility**: Easier to extend plotting without modifying model classes

### Migration Challenges
The composition pattern means that methods requiring model-specific attributes (like `self.channel_columns`, `self.X`, `self.y`) need to be refactored to work with inference data alone, or the model needs to pass additional context to MMMPlotSuite.

## Recommendations

### Option 1: Add Missing Methods to MMMPlotSuite
Port the 4 missing methods to MMMPlotSuite, refactoring them to work with `idata` only:
- `grouped_contribution_breakdown_over_time`
- `prior_vs_posterior`
- `channel_contribution_grid`
- `new_spend_contributions`

**Pros**: Complete feature parity, unified interface
**Cons**: May require refactoring to remove dependencies on model attributes

### Option 2: Keep Methods on Model Classes
Keep certain methods (like `plot_new_spend_contributions`) on the model class since they depend on model-specific computation methods.

**Pros**: Maintains access to model state and methods
**Cons**: Split plotting interface, users need to know which plots are on `mmm.plot` vs `mmm`

### Option 3: Hybrid Approach
- Port visualization-only methods to MMMPlotSuite
- Keep methods that require complex model computation on the model class
- Document the split clearly

**Pros**: Pragmatic balance
**Cons**: Still requires user awareness of the split

### Option 4: Pass Model Context to MMMPlotSuite
Extend MMMPlotSuite to optionally accept model context (channels, dates, etc.) in addition to idata.

**Pros**: Enables all plotting in one place
**Cons**: Increases coupling between model and plotting suite

## Open Questions

1. Should all plotting methods be in MMMPlotSuite, or is it acceptable to have some on the model class?
2. How should methods that depend on model-specific computation (like `new_spend_contributions`) be handled?
3. Is the goal to fully deprecate the legacy MMM in favor of multidimensional MMM?
4. Should MMMPlotSuite be extended to accept model context, or should methods be refactored to work with idata only?

## Related Research

This research relates to the architectural evolution of the pymc-marketing library from inheritance-based to composition-based design patterns for plotting functionality.
