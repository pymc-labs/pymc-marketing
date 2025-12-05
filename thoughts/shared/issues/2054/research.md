---
date: 2025-11-04T22:26:50Z
researcher: Claude Code
git_commit: 9537b9a08837a3c5dabcdee6244a0cd1c4688ea0
branch: work-issue-2054
repository: pymc-labs/pymc-marketing
topic: "Add Gradient to plot posterior predictive in Plot Suite"
tags: [research, codebase, plot-suite, gradient, visualization, mmm, issue-2054]
status: complete
last_updated: 2025-11-04
last_updated_by: Claude Code
issue_number: 2054
---

# Research: Add Gradient to plot posterior predictive in Plot Suite

**Date**: 2025-11-04T22:26:50Z
**Researcher**: Claude Code
**Git Commit**: 9537b9a08837a3c5dabcdee6244a0cd1c4688ea0
**Branch**: work-issue-2054
**Repository**: pymc-labs/pymc-marketing
**Issue**: #2054

## Research Question

How to add Gradient visualization functionality to the `plot_posterior_predictive` method in the Plot Suite to enable full migration from base model plotting methods?

## Summary

The gradient visualization feature currently exists in the **BaseValidateMMM** class but is **not available in the MMMPlotSuite**. This creates an incomplete migration path because:

1. **BaseValidateMMM.plot_posterior_predictive()** (`pymc_marketing/mmm/base.py:625`) supports `add_gradient` parameter
2. **MMMPlotSuite.posterior_predictive()** (`pymc_marketing/mmm/plot.py:375`) does NOT support gradient visualization
3. The gradient implementation in BaseValidateMMM uses `_add_gradient_to_plot()` (`pymc_marketing/mmm/base.py:362-433`)

To complete the migration, the gradient functionality needs to be added to the Plot Suite's `posterior_predictive()` method.

## Detailed Findings

### 1. Current Plot Suite Architecture

**Location**: `pymc_marketing/mmm/plot.py:187-1923`

The `MMMPlotSuite` class provides a comprehensive plotting API for Media Mix Models:

```python
class MMMPlotSuite:
    """Media Mix Model Plot Suite."""

    def __init__(self, idata: xr.Dataset | az.InferenceData):
        self.idata = idata
```

**Integration**: Exposed via property in `pymc_marketing/mmm/multidimensional.py:618-623`:

```python
@property
def plot(self) -> MMMPlotSuite:
    """Use the MMMPlotSuite to plot the results."""
    return MMMPlotSuite(idata=self.idata)
```

**Access Pattern**: Users call `mmm.plot.method_name()` on fitted models.

### 2. Posterior Predictive Plotting - Two APIs

#### API 1: MMMPlotSuite.posterior_predictive() (Plot Suite)
**Location**: `pymc_marketing/mmm/plot.py:375-463`

**Current Signature**:
```python
def posterior_predictive(
    self,
    var: list[str] | None = None,
    idata: xr.Dataset | None = None,
    hdi_prob: float = 0.85,
) -> tuple[Figure, NDArray[Axes]]:
```

**Features**:
- Multi-variable plotting
- Multi-dimensional subplot support
- HDI bands at configurable probability
- Median line visualization
- **Missing: Gradient visualization**

**Test Coverage**: `tests/mmm/test_plot.py:185`

#### API 2: BaseValidateMMM.plot_posterior_predictive() (Base Model)
**Location**: `pymc_marketing/mmm/base.py:625-682`

**Current Signature**:
```python
def plot_posterior_predictive(
    self,
    original_scale: bool = False,
    hdi_list: list[float] | None = None,
    add_mean: bool = True,
    add_gradient: bool = False,
    ax: plt.Axes | None = None,
    **plt_kwargs,
) -> plt.Figure:
```

**Features**:
- Scale transformation (`original_scale`)
- Multiple HDI levels (`hdi_list`)
- Mean line (`add_mean`)
- **Gradient visualization (`add_gradient`)** ← This is what needs to be migrated
- Custom axes support

**Test Coverage**: `tests/mmm/test_plotting.py:206-263` (extensive parametrized tests)

### 3. Gradient Implementation Details

**Core Implementation**: `pymc_marketing/mmm/base.py:362-433`

```python
def _add_gradient_to_plot(
    self,
    ax: plt.Axes,
    group: Literal["prior_predictive", "posterior_predictive"],
    original_scale: bool = False,
    n_percentiles: int = 30,
    palette: str = "Blues",
    **kwargs,
) -> plt.Axes:
    """
    Add a gradient representation of the prior or posterior predictive distribution.

    Creates a shaded area plot where color intensity represents
    the density of the posterior predictive distribution.
    """
```

**Algorithm**:
1. Retrieves posterior_predictive data and flattens samples
2. Computes percentile ranges (default: 30 ranges from 3rd to 97th percentile)
3. Creates layered `fill_between()` calls with varying colors and alpha
4. Middle percentiles use higher alpha (denser distribution)
5. Outer percentiles use lower alpha (sparser distribution)
6. Color mapping via matplotlib colormap (default "Blues")

**Visual Effect**: Creates a smooth gradient visualization showing full distribution density.

**Usage in Base Model** (`pymc_marketing/mmm/base.py:534-541`):
```python
if add_gradient:
    ax = self._add_gradient_to_plot(
        ax=ax,
        group=group,
        original_scale=original_scale,
        n_percentiles=30,
        palette="Blues",
    )
```

### 4. Test Coverage for Gradient Feature

**Location**: `tests/mmm/test_plotting.py:206-263`

Tests include combinations of:
- `add_gradient: True` with various other parameters
- Prior predictive plots (lines 160, 181, 189, 197)
- Posterior predictive plots (lines 219, 240, 248, 256)
- Combinations with `add_mean`, `original_scale`, `hdi_list`

Example test cases:
```python
("plot_posterior_predictive", {"add_gradient": True}),
("plot_posterior_predictive", {"add_gradient": True, "original_scale": True}),
("plot_posterior_predictive", {"add_gradient": True, "add_mean": False}),
```

### 5. Migration Context

**Current State**:
- BaseValidateMMM methods: Full-featured but older API pattern
- MMMPlotSuite: Modern API with better multi-dimensional support but missing gradient

**Migration Goal**: Enable users to get all functionality through the Plot Suite API:
- Before: `mmm.plot_posterior_predictive(add_gradient=True)`
- After: `mmm.plot.posterior_predictive(add_gradient=True)` or similar

**Blockers for Full Migration**:
1. Gradient visualization not available in Plot Suite
2. `original_scale` parameter not in Plot Suite
3. Multiple HDI levels (`hdi_list`) not in Plot Suite (currently single `hdi_prob`)

## Code References

### Key Implementation Files
- `pymc_marketing/mmm/plot.py:187` - MMMPlotSuite class
- `pymc_marketing/mmm/plot.py:375` - MMMPlotSuite.posterior_predictive() method
- `pymc_marketing/mmm/base.py:362` - _add_gradient_to_plot() implementation
- `pymc_marketing/mmm/base.py:625` - BaseValidateMMM.plot_posterior_predictive()
- `pymc_marketing/mmm/multidimensional.py:618` - Plot Suite property accessor

### Helper Methods (Reusable in Implementation)
- `pymc_marketing/mmm/plot.py:200` - `_init_subplots()` - Subplot grid initialization
- `pymc_marketing/mmm/plot.py:247` - `_get_additional_dim_combinations()` - Dimension handling
- `pymc_marketing/mmm/plot.py:269` - `_reduce_and_stack()` - Data reduction
- `pymc_marketing/mmm/plot.py:286` - `_get_posterior_predictive_data()` - Data retrieval
- `pymc_marketing/mmm/plot.py:306` - `_add_median_and_hdi()` - Add median/HDI to plot

### Test Files
- `tests/mmm/test_plot.py:185` - Basic posterior_predictive test
- `tests/mmm/test_plotting.py:206-263` - Parametrized tests with gradient
- `tests/mmm/test_base.py:358` - Error handling test

## Architecture Insights

### Design Patterns in Plot Suite

1. **Dimension-Aware Subplots**: Plot methods automatically create subplots for each combination of non-ignored dimensions
2. **Helper Method Composition**: Complex plotting logic decomposed into reusable helpers
3. **xarray Integration**: Heavy use of xarray for multi-dimensional data manipulation
4. **Tuple Returns**: Methods return `(Figure, NDArray[Axes])` for flexibility
5. **No State Mutation**: Plot Suite is stateless, only operates on InferenceData

### Gradient Implementation Pattern

The gradient visualization follows a **layered percentile** approach:
- **Conceptual**: Stack many thin HDI bands with varying opacity
- **Visual**: Creates smooth density gradient from sparse (edges) to dense (center)
- **Technical**: Uses `np.percentile()` + `ax.fill_between()` in loop
- **Customization**: Configurable via `n_percentiles` and `palette`

### Integration Approach for Plot Suite

**Recommended Pattern**: Add `add_gradient` parameter to `MMMPlotSuite.posterior_predictive()`

```python
def posterior_predictive(
    self,
    var: list[str] | None = None,
    idata: xr.Dataset | None = None,
    hdi_prob: float = 0.85,
    add_gradient: bool = False,  # NEW
    n_percentiles: int = 30,      # NEW
    palette: str = "Blues",       # NEW
) -> tuple[Figure, NDArray[Axes]]:
```

**Implementation Strategy**:
1. Extract gradient logic from `BaseValidateMMM._add_gradient_to_plot()` into standalone function
2. Adapt to work with xarray DataArrays (Plot Suite uses xarray, base uses Dataset)
3. Integrate gradient plotting into dimension loop in `posterior_predictive()`
4. Place gradient layer BEFORE median/HDI visualization (background layer)
5. Add conditional logic: `if add_gradient:` section before `_add_median_and_hdi()`

**Key Adaptation**: The base model gradient method works with dates directly from Dataset, while Plot Suite works with dimension-sliced DataArrays. Need to handle this in the adaptation.

## Implementation Checklist

To add gradient to Plot Suite's `posterior_predictive()`:

1. **Create Helper Method**: Add `_add_gradient_to_axes()` in MMMPlotSuite
   - Adapt `BaseValidateMMM._add_gradient_to_plot()` logic
   - Accept xarray DataArray instead of Dataset
   - Work with generic dimensions (not just "date")

2. **Modify `posterior_predictive()` Method**:
   - Add parameters: `add_gradient`, `n_percentiles`, `palette`
   - Add gradient rendering in dimension loop (before median/HDI)
   - Ensure gradient is background layer (drawn first)

3. **Update Tests**:
   - Add test cases in `tests/mmm/test_plot.py`
   - Test with single dimension
   - Test with multiple dimensions
   - Test with various `n_percentiles` and `palette` values

4. **Documentation**:
   - Update method docstring
   - Add parameter descriptions
   - Include example in docstring or documentation

5. **Validation**:
   - Visual comparison with base model gradient output
   - Ensure color/alpha mapping matches
   - Test with real-world model fits

## Related Research

No previous research documents found (no `thoughts/` directory in repository).

## Open Questions

1. **Should gradient replace or complement HDI bands?**
   - Current base model: Users choose either HDI OR gradient
   - Plot Suite option: Allow both simultaneously (gradient + HDI overlay)?

2. **Dimension-specific gradients?**
   - Should gradient settings be customizable per dimension?
   - Or single global setting for all subplots?

3. **Parameter naming consistency?**
   - Should we match base model parameter names exactly?
   - Or adapt to Plot Suite conventions?

4. **original_scale support?**
   - Should we also add `original_scale` parameter to Plot Suite?
   - This would require access to model's scale transformation logic

5. **hdi_list vs hdi_prob?**
   - Base model supports multiple HDI levels via `hdi_list`
   - Plot Suite currently has single `hdi_prob`
   - Should we unify these?

## Next Steps

1. **Implement `_add_gradient_to_axes()` helper method** in MMMPlotSuite
2. **Modify `posterior_predictive()` method** to accept gradient parameters
3. **Add comprehensive tests** covering gradient visualization
4. **Visual validation** with example models
5. **Consider migrating other base model features** (`original_scale`, `hdi_list`)

## Additional Context

### Similar Patterns in Codebase

The saturation curves method (`pymc_marketing/mmm/plot.py:744-996`) demonstrates a similar pattern of complex layered visualization:
- Uses `plot_samples()` for sample curves
- Uses `plot_hdi()` for HDI bands
- Layers visualization elements in specific order
- Could serve as reference for gradient implementation

### Color Palette Options

The gradient implementation uses matplotlib colormaps. Common options:
- "Blues" (default) - Blue gradient
- "Reds" - Red gradient
- "Greens" - Green gradient
- "viridis", "plasma", "inferno" - Perceptually uniform colormaps

### Performance Considerations

Gradient rendering with 30 percentile ranges creates 29 `fill_between()` calls per subplot. For models with many dimensions, this could impact rendering performance. Consider:
- Caching computed percentiles
- Reducing default `n_percentiles` if needed
- Providing `add_gradient=False` as default for backward compatibility
