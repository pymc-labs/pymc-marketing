---
date: 2025-11-08
issue_number: 2054
repository: pymc-labs/pymc-marketing
branch: work-issue-2054
topic: "Add Gradient to plot posterior predictive in Plot Suite"
status: ready_for_implementation
tags: [implementation-plan, plot-suite, gradient, visualization, mmm]
---

# Add Gradient Visualization to Plot Suite Implementation Plan

## Overview

This implementation adds gradient visualization functionality to `MMMPlotSuite.posterior_predictive()` method, enabling users to visualize the full posterior predictive distribution density as a smooth color gradient. This feature currently exists in the legacy `BaseValidateMMM.plot_posterior_predictive()` but is missing from the modern Plot Suite API, blocking full migration from the base model methods.

The gradient visualization uses a layered percentile approach where color intensity represents distribution density: darker/more opaque colors in the center (higher density) and lighter/more transparent colors at the edges (lower density).

## Current State Analysis

### Existing Components

1. **Plot Suite Architecture** (`pymc_marketing/mmm/plot.py:187-1923`)
   - Modern, dimension-aware plotting API
   - Uses xarray DataArrays for multi-dimensional data
   - Returns `(Figure, NDArray[Axes])` tuples
   - Stateless design with helper method composition

2. **Base Model Gradient Implementation** (`pymc_marketing/mmm/base.py:362-433`)
   - `_add_gradient_to_plot()` method with complete gradient logic
   - Works with Dataset objects and date coordinates
   - Uses matplotlib's `fill_between()` for layered percentiles
   - Configurable via `n_percentiles` (default: 30) and `palette` (default: "Blues")

3. **Current `posterior_predictive()` Method** (`pymc_marketing/mmm/plot.py:375-463`)
   - Supports multi-variable plotting
   - Handles multi-dimensional subplots automatically
   - Uses `_add_median_and_hdi()` for visualization
   - **Missing**: Gradient visualization capability

### Key Constraints

- Plot Suite uses xarray DataArrays (not Datasets like base model)
- Must work with generic dimensions (not just "date")
- Must maintain backward compatibility (gradient disabled by default)
- Must support multi-dimensional subplot rendering
- Must layer gradient BEFORE median/HDI (as background)

## Desired End State

### Functional Requirements

Users can call:
```python
fig, axes = mmm.plot.posterior_predictive(
    add_gradient=True,
    n_percentiles=30,
    palette="Blues",
    hdi_prob=0.85
)
```

The result shows:
- Smooth gradient background representing distribution density
- Optional HDI bands and median lines overlaid on top
- Consistent appearance across all dimension combinations
- Visual parity with base model gradient output

### Verification Criteria

1. **Visual Validation**: Gradient output matches base model appearance
2. **Multi-dimensional Support**: Works with models having channel, geo, or other dimensions
3. **Parameter Flexibility**: Users can customize percentiles and color palette
4. **Backward Compatibility**: Existing calls work without changes (gradient disabled by default)
5. **Test Coverage**: New tests verify gradient functionality with various configurations

## What We're NOT Doing

1. **Not adding `original_scale` parameter** - This requires access to model transformation logic not available in Plot Suite
2. **Not migrating `hdi_list`** - Plot Suite uses single `hdi_prob`, multiple HDI levels is out of scope
3. **Not refactoring base model implementation** - Only adapting logic for Plot Suite
4. **Not adding gradient to other plot methods** - Only `posterior_predictive()` in this implementation
5. **Not making gradient the default** - Gradient remains opt-in via `add_gradient=False` default

## Implementation Approach

### Strategy

1. **Create standalone helper method** `_add_gradient_to_axes()` in `MMMPlotSuite` class
2. **Adapt base model logic** to work with xarray DataArrays instead of Datasets
3. **Integrate into existing flow** by adding gradient rendering before median/HDI
4. **Maintain z-order** by drawing gradient first (background layer)
5. **Add comprehensive tests** covering single/multi-dimensional cases

### Technical Decisions

**Decision 1: Standalone helper method vs. inline code**
- **Choice**: Create `_add_gradient_to_axes()` helper method
- **Rationale**: Follows Plot Suite pattern of helper method composition; enables reuse if gradient needed elsewhere

**Decision 2: Allow gradient + HDI simultaneously**
- **Choice**: Allow both (gradient as background, HDI/median as overlay)
- **Rationale**: Base model allows either/or, but Plot Suite can layer both for richer visualization

**Decision 3: Parameter defaults**
- **Choice**: `add_gradient=False`, `n_percentiles=30`, `palette="Blues"`
- **Rationale**: Matches base model defaults; maintains backward compatibility

**Decision 4: Dimension handling**
- **Choice**: Use first dimension with coordinate values (typically "date")
- **Rationale**: Gradient needs x-axis coordinates; "date" is standard but allow flexibility

---

## Phase 1: Implement Gradient Helper Method

### Overview

Create the `_add_gradient_to_axes()` helper method that adapts the base model gradient logic to work with xarray DataArrays in the Plot Suite context.

### Changes Required

#### 1. Add `_add_gradient_to_axes()` Method to MMMPlotSuite

**File**: `pymc_marketing/mmm/plot.py`
**Location**: After `_add_median_and_hdi()` method (around line 324)

**Implementation**:

```python
def _add_gradient_to_axes(
    self,
    ax: Axes,
    data: xr.DataArray,
    n_percentiles: int = 30,
    palette: str = "Blues",
    **kwargs,
) -> Axes:
    """Add a gradient representation of the distribution to the axes.

    Creates a shaded area plot where color intensity represents
    the density of the distribution. Uses layered percentile ranges
    with varying opacity to create a smooth gradient effect.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to add the gradient to.
    data : xarray.DataArray
        The data array containing samples. Must have a 'sample' dimension
        and a dimension with coordinate values (typically 'date').
    n_percentiles : int, optional
        Number of percentile ranges to use for the gradient. More percentiles
        create a smoother gradient but increase rendering time. Default is 30.
    palette : str, optional
        Name of the matplotlib colormap to use. Default is "Blues".
    **kwargs
        Additional keyword arguments passed to ax.fill_between().

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the gradient added.

    Raises
    ------
    ValueError
        If data does not have a 'sample' dimension or lacks coordinate dimensions.
    """
    # Validate data has required dimensions
    if "sample" not in data.dims:
        raise ValueError("Data must have a 'sample' dimension for gradient plotting.")

    # Find the coordinate dimension (typically 'date')
    coord_dims = [d for d in data.dims if d != "sample"]
    if not coord_dims:
        raise ValueError("Data must have at least one coordinate dimension besides 'sample'.")
    coord_dim = coord_dims[0]  # Use first coordinate dimension
    x_values = data.coords[coord_dim].values

    # Set up color map and ranges
    cmap = plt.get_cmap(palette)
    color_range = np.linspace(0.3, 1.0, n_percentiles // 2)
    percentile_ranges = np.linspace(3, 97, n_percentiles)

    # Create gradient by filling between percentile ranges
    for i in range(len(percentile_ranges) - 1):
        # Compute percentiles along the sample dimension
        lower_percentile = np.percentile(
            data.values, percentile_ranges[i], axis=data.dims.index("sample")
        )
        upper_percentile = np.percentile(
            data.values, percentile_ranges[i + 1], axis=data.dims.index("sample")
        )

        # Map percentile index to color intensity
        # Middle percentiles get darker colors and higher alpha
        if i < n_percentiles // 2:
            color_val = color_range[i]
        else:
            color_val = color_range[n_percentiles - i - 2]

        # Alpha increases toward middle (50th percentile)
        alpha_val = 0.2 + 0.8 * (1 - abs(2 * i / n_percentiles - 1))

        ax.fill_between(
            x=x_values,
            y1=lower_percentile,
            y2=upper_percentile,
            color=cmap(color_val),
            alpha=alpha_val,
            **kwargs,
        )

    return ax
```

### Success Criteria

#### Automated Verification:
- [x] Method exists in MMMPlotSuite class: `grep -n "_add_gradient_to_axes" pymc_marketing/mmm/plot.py`
- [x] Type hints are correct: `mypy pymc_marketing/mmm/plot.py`
- [x] No linting errors: `ruff check pymc_marketing/mmm/plot.py`

#### Manual Verification:
- [x] Method signature follows Plot Suite conventions
- [x] Docstring is complete with all parameters documented
- [x] Error handling validates required dimensions
- [x] Algorithm matches base model gradient logic

---

## Phase 2: Integrate Gradient into posterior_predictive()

### Overview

Modify the `posterior_predictive()` method to accept gradient parameters and call the helper method when gradient visualization is requested.

### Changes Required

#### 1. Update Method Signature

**File**: `pymc_marketing/mmm/plot.py`
**Location**: Line 375 (method definition)

**Change**:

```python
def posterior_predictive(
    self,
    var: list[str] | None = None,
    idata: xr.Dataset | None = None,
    hdi_prob: float = 0.85,
    add_gradient: bool = False,
    n_percentiles: int = 30,
    palette: str = "Blues",
) -> tuple[Figure, NDArray[Axes]]:
```

#### 2. Update Docstring

**File**: `pymc_marketing/mmm/plot.py`
**Location**: Line 381 (docstring)

**Add to Parameters section**:

```python
    add_gradient : bool, optional
        If True, add a gradient representation of the full distribution
        as a background layer. The gradient shows distribution density
        with color intensity. Default is False.
    n_percentiles : int, optional
        Number of percentile ranges to use for the gradient visualization.
        Only used when add_gradient=True. More percentiles create smoother
        gradients but increase rendering time. Default is 30.
    palette : str, optional
        Matplotlib colormap name for the gradient visualization.
        Only used when add_gradient=True. Common options: "Blues", "Reds",
        "Greens", "viridis", "plasma". Default is "Blues".
```

#### 3. Add Gradient Rendering Logic

**File**: `pymc_marketing/mmm/plot.py`
**Location**: Inside the dimension loop, before `_add_median_and_hdi()` call (around line 449)

**Insert before existing plotting code**:

```python
            # 6. Plot each requested variable
            for v in var:
                if v not in pp_data:
                    raise ValueError(
                        f"Variable '{v}' not in the posterior_predictive dataset."
                    )

                data = pp_data[v].sel(**indexers)
                # Sum leftover dims, stack chain+draw if needed
                data = self._reduce_and_stack(data, ignored_dims)

                # Add gradient visualization if requested (background layer)
                if add_gradient:
                    ax = self._add_gradient_to_axes(
                        ax=ax,
                        data=data,
                        n_percentiles=n_percentiles,
                        palette=palette,
                    )

                # Add median and HDI (foreground layer)
                ax = self._add_median_and_hdi(ax, data, v, hdi_prob=hdi_prob)
```

### Success Criteria

#### Automated Verification:
- [x] Method signature updated: `grep -A 6 "def posterior_predictive" pymc_marketing/mmm/plot.py`
- [x] Docstring includes new parameters: `grep -A 30 "Plot time series from the posterior" pymc_marketing/mmm/plot.py | grep "add_gradient"`
- [x] Type checking passes: `mypy pymc_marketing/mmm/plot.py`
- [x] No syntax errors: `python -m py_compile pymc_marketing/mmm/plot.py`

#### Manual Verification:
- [x] Gradient renders before median/HDI (correct z-order)
- [x] Gradient parameter defaults maintain backward compatibility
- [x] Method handles multi-dimensional cases correctly
- [x] Visual output matches base model gradient style

---

## Phase 3: Add Comprehensive Test Coverage

### Overview

Create test cases that verify gradient functionality works correctly with various configurations and edge cases.

### Changes Required

#### 1. Add Basic Gradient Test

**File**: `tests/mmm/test_plot.py`
**Location**: After existing `test_posterior_predictive` test (around line 196)

**New Test**:

```python
def test_posterior_predictive_with_gradient(fit_mmm_with_channel_original_scale, df):
    """Test posterior_predictive with gradient visualization."""
    fit_mmm_with_channel_original_scale.sample_posterior_predictive(
        df.drop(columns=["y"])
    )
    fig, ax = fit_mmm_with_channel_original_scale.plot.posterior_predictive(
        add_gradient=True,
        hdi_prob=0.95,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert all(isinstance(a, Axes) for a in ax.flat)
    # Verify gradient was drawn (check for fill_between patches)
    for a in ax.flat:
        patches = [p for p in a.patches if isinstance(p, plt.matplotlib.patches.Polygon)]
        assert len(patches) > 0, "Expected gradient patches on axes"


def test_posterior_predictive_gradient_parameters(fit_mmm_with_channel_original_scale, df):
    """Test gradient with custom parameters."""
    fit_mmm_with_channel_original_scale.sample_posterior_predictive(
        df.drop(columns=["y"])
    )
    # Test with different n_percentiles
    fig1, _ = fit_mmm_with_channel_original_scale.plot.posterior_predictive(
        add_gradient=True,
        n_percentiles=20,
    )
    assert isinstance(fig1, Figure)

    # Test with different palette
    fig2, _ = fit_mmm_with_channel_original_scale.plot.posterior_predictive(
        add_gradient=True,
        palette="Reds",
    )
    assert isinstance(fig2, Figure)


def test_posterior_predictive_gradient_with_hdi(fit_mmm_with_channel_original_scale, df):
    """Test that gradient and HDI can be displayed together."""
    fit_mmm_with_channel_original_scale.sample_posterior_predictive(
        df.drop(columns=["y"])
    )
    fig, ax = fit_mmm_with_channel_original_scale.plot.posterior_predictive(
        add_gradient=True,
        hdi_prob=0.85,
    )
    assert isinstance(fig, Figure)
    # Verify both gradient patches and HDI fills exist
    for a in ax.flat:
        # Should have multiple fill_between patches from both gradient and HDI
        patches = [p for p in a.patches if isinstance(p, plt.matplotlib.patches.Polygon)]
        assert len(patches) > 1, "Expected both gradient and HDI patches"
```

#### 2. Add Edge Case Tests

**File**: `tests/mmm/test_plot.py`
**Location**: After the gradient tests

**New Tests**:

```python
def test_posterior_predictive_gradient_without_sample_dim():
    """Test that gradient fails gracefully without sample dimension."""
    # Create data without sample dimension
    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")
    data = xr.DataArray(
        np.random.normal(size=52),
        dims=("date",),
        coords={"date": dates},
    )

    plot_suite = MMMPlotSuite(
        idata=az.InferenceData(
            posterior_predictive=xr.Dataset({"y": data})
        )
    )

    # Should raise ValueError about missing sample dimension
    with pytest.raises(ValueError, match="sample"):
        plot_suite.posterior_predictive(add_gradient=True)


def test_posterior_predictive_gradient_multidimensional(mock_idata):
    """Test gradient with multi-dimensional data."""
    # Use mock_idata fixture which has channel/geo dimensions
    plot_suite = MMMPlotSuite(idata=mock_idata)

    # Should create gradient for each subplot (dimension combination)
    fig, axes = plot_suite.posterior_predictive(
        add_gradient=True,
        hdi_prob=0.85,
    )

    assert isinstance(fig, Figure)
    # Each subplot should have gradient patches
    for ax_row in axes:
        for a in ax_row:
            patches = [p for p in a.patches]
            assert len(patches) > 0, "Each subplot should have gradient"
```

#### 3. Update Test Fixtures if Needed

**File**: `tests/mmm/test_plot.py`
**Location**: Verify `mock_idata` fixture has posterior_predictive group

**Ensure fixture includes**:

```python
@pytest.fixture(scope="module")
def mock_idata() -> az.InferenceData:
    seed = sum(map(ord, "Fake posterior"))
    rng = np.random.default_rng(seed)
    normal = rng.normal

    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")

    # Ensure posterior_predictive group exists with sample dimension
    posterior_predictive_data = xr.Dataset(
        {
            "y": xr.DataArray(
                normal(size=(4, 100, 52)),
                dims=("chain", "draw", "date"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(100),
                    "date": dates,
                },
            ),
        }
    )

    return az.InferenceData(
        posterior=xr.Dataset(...),  # existing posterior data
        posterior_predictive=posterior_predictive_data,  # ADD THIS
    )
```

### Success Criteria

#### Automated Verification:
- [ ] All new tests pass: `pytest tests/mmm/test_plot.py::test_posterior_predictive_with_gradient -v`
- [ ] All new tests pass: `pytest tests/mmm/test_plot.py::test_posterior_predictive_gradient_parameters -v`
- [ ] All new tests pass: `pytest tests/mmm/test_plot.py::test_posterior_predictive_gradient_with_hdi -v`
- [ ] Edge case tests pass: `pytest tests/mmm/test_plot.py::test_posterior_predictive_gradient_without_sample_dim -v`
- [ ] Multi-dimensional test passes: `pytest tests/mmm/test_plot.py::test_posterior_predictive_gradient_multidimensional -v`
- [ ] Existing tests still pass: `pytest tests/mmm/test_plot.py::test_posterior_predictive -v`
- [ ] Full test suite passes: `pytest tests/mmm/test_plot.py -v`
- [ ] No test regressions: `pytest tests/mmm/test_plotting.py -v` (base model tests)

#### Manual Verification:
- [ ] Test fixtures provide appropriate data for gradient testing
- [ ] Error messages are clear and helpful
- [ ] Tests cover single and multi-dimensional cases
- [ ] Tests verify both gradient-only and gradient+HDI scenarios

---

## Phase 4: Add Import Statements and Dependencies

### Overview

Ensure all necessary imports are present in the modified files.

### Changes Required

#### 1. Verify Imports in plot.py

**File**: `pymc_marketing/mmm/plot.py`
**Location**: Top of file (around lines 1-20)

**Ensure these imports exist**:

```python
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
```

These should already exist, but verify `plt.get_cmap` is available via the `matplotlib.pyplot` import.

#### 2. Verify Test Imports

**File**: `tests/mmm/test_plot.py`
**Location**: Top of file

**Ensure these imports exist**:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import arviz as az
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pymc_marketing.mmm.plot import MMMPlotSuite
```

### Success Criteria

#### Automated Verification:
- [ ] Import statements are valid: `python -c "from pymc_marketing.mmm.plot import MMMPlotSuite"`
- [ ] No import errors in tests: `python -c "import tests.mmm.test_plot"`
- [ ] Linter passes: `ruff check pymc_marketing/mmm/plot.py tests/mmm/test_plot.py`

#### Manual Verification:
- [ ] All matplotlib functionality is accessible
- [ ] No missing import warnings in IDE
- [ ] Test fixtures import correctly

---

## Phase 5: Documentation and Examples

### Overview

Update the method docstring with comprehensive documentation and usage examples.

### Changes Required

#### 1. Enhance Docstring with Examples

**File**: `pymc_marketing/mmm/plot.py`
**Location**: In `posterior_predictive()` docstring (around line 381)

**Add Examples section before Returns**:

```python
    Examples
    --------
    Basic usage with gradient:

    >>> fig, axes = mmm.plot.posterior_predictive(add_gradient=True)

    Customize gradient appearance:

    >>> fig, axes = mmm.plot.posterior_predictive(
    ...     add_gradient=True,
    ...     n_percentiles=40,
    ...     palette="viridis",
    ...     hdi_prob=0.90
    ... )

    Combine gradient with HDI bands:

    >>> fig, axes = mmm.plot.posterior_predictive(
    ...     add_gradient=True,
    ...     hdi_prob=0.85
    ... )

    The gradient visualization shows distribution density where darker/more
    opaque colors indicate higher probability density (near the median) and
    lighter/more transparent colors indicate lower density (in the tails).
```

#### 2. Add Notes Section

**File**: `pymc_marketing/mmm/plot.py`
**Location**: In `posterior_predictive()` docstring after Examples

**Add Notes**:

```python
    Notes
    -----
    The gradient visualization uses a layered percentile approach where multiple
    percentile ranges are drawn as semi-transparent fills. The default uses 30
    percentile ranges from the 3rd to 97th percentile, creating a smooth gradient
    effect. Performance considerations:

    - More percentiles (higher n_percentiles) create smoother gradients but increase
      rendering time, especially with many subplots
    - The gradient is drawn as a background layer, with median and HDI overlaid on top
    - For multi-dimensional models, gradients are drawn independently for each subplot
```

### Success Criteria

#### Automated Verification:
- [ ] Docstring examples are valid Python syntax
- [ ] Documentation builds without errors: `python -c "from pymc_marketing.mmm.plot import MMMPlotSuite; help(MMMPlotSuite.posterior_predictive)"`

#### Manual Verification:
- [ ] Examples are clear and self-contained
- [ ] Notes section explains key concepts
- [ ] Parameter descriptions are comprehensive
- [ ] Usage patterns are demonstrated

---

## Testing Strategy

### Unit Tests

**Location**: `tests/mmm/test_plot.py`

**Coverage Areas**:
1. **Basic functionality**: Gradient renders without errors
2. **Parameter variations**: Different `n_percentiles` and `palette` values
3. **Layering**: Gradient + HDI work together correctly
4. **Edge cases**: Missing dimensions, invalid parameters
5. **Multi-dimensional**: Gradient works with channel/geo dimensions

**Key Test Cases**:
- `test_posterior_predictive_with_gradient`: Basic gradient rendering
- `test_posterior_predictive_gradient_parameters`: Custom parameters
- `test_posterior_predictive_gradient_with_hdi`: Gradient + HDI overlay
- `test_posterior_predictive_gradient_without_sample_dim`: Error handling
- `test_posterior_predictive_gradient_multidimensional`: Multi-dim support

### Integration Tests

**Manual Testing Steps**:

1. **Visual Comparison with Base Model**:
   ```python
   # Load a fitted MMM model
   mmm = load_fitted_model()

   # Compare base model and Plot Suite gradient
   fig1 = mmm.plot_posterior_predictive(add_gradient=True)
   fig2, _ = mmm.plot.posterior_predictive(add_gradient=True)

   # Verify visual similarity (color, density, coverage)
   ```

2. **Multi-dimensional Model Test**:
   ```python
   # Test with model having multiple dimensions
   mmm_multi = load_multidimensional_model()
   fig, axes = mmm_multi.plot.posterior_predictive(
       add_gradient=True,
       hdi_prob=0.85
   )

   # Verify each subplot has gradient
   # Verify gradients are independent per dimension
   ```

3. **Performance Test**:
   ```python
   import time

   # Test with high n_percentiles
   start = time.time()
   fig, _ = mmm.plot.posterior_predictive(
       add_gradient=True,
       n_percentiles=50
   )
   elapsed = time.time() - start

   # Verify rendering time is reasonable (< 5 seconds for typical model)
   assert elapsed < 5.0
   ```

4. **Parameter Combinations**:
   ```python
   # Test various palettes
   for palette in ["Blues", "Reds", "Greens", "viridis", "plasma"]:
       fig, _ = mmm.plot.posterior_predictive(
           add_gradient=True,
           palette=palette
       )

   # Test various percentile counts
   for n in [10, 20, 30, 40, 50]:
       fig, _ = mmm.plot.posterior_predictive(
           add_gradient=True,
           n_percentiles=n
       )
   ```

### Regression Tests

**Ensure Backward Compatibility**:
1. Existing calls without `add_gradient` work unchanged
2. All existing tests in `test_plot.py` still pass
3. Base model tests in `test_plotting.py` still pass
4. No performance regressions in existing methods

---

## Performance Considerations

### Rendering Complexity

**Current State**:
- Each subplot with gradient: 29 `fill_between()` calls (with default n_percentiles=30)
- Multi-dimensional models: N subplots × 29 fills = 29N fills total

**Performance Impact**:
- Typical model (2-3 dimensions): ~60-90 fills, renders in < 1 second
- Complex model (4-5 dimensions): ~120-150 fills, renders in 1-2 seconds
- Very complex model (6+ dimensions): May need lower `n_percentiles` default

**Optimization Strategy**:
- Keep `add_gradient=False` as default (backward compatible, no perf impact)
- Document performance implications in docstring
- Allow users to reduce `n_percentiles` for faster rendering if needed
- Consider caching percentile calculations if re-rendering same data

### Memory Considerations

**Data Size**:
- Gradient computation requires full sample data in memory
- Percentile calculation: O(n) for each of 30 percentile levels
- Typical overhead: ~2-3x the base data size during rendering

**Mitigation**:
- Percentiles computed on-the-fly, not stored
- Uses NumPy's efficient percentile implementation
- No significant memory footprint beyond temporary arrays

---

## Migration Notes

### For Users Migrating from Base Model

**Before (Base Model API)**:
```python
fig = mmm.plot_posterior_predictive(
    add_gradient=True,
    hdi_list=[0.94],
)
```

**After (Plot Suite API)**:
```python
fig, axes = mmm.plot.posterior_predictive(
    add_gradient=True,
    hdi_prob=0.94,
)
```

**Key Differences**:
1. Plot Suite returns `(Figure, NDArray[Axes])` tuple instead of just `Figure`
2. `hdi_list` → `hdi_prob` (single value instead of list)
3. No `original_scale` parameter in Plot Suite
4. Plot Suite handles multi-dimensional data automatically with subplots

### Breaking Changes

**None** - This is a pure addition with no breaking changes:
- New parameters are optional with backward-compatible defaults
- Existing functionality unchanged
- Return type unchanged

### Deprecation Path

**Not Applicable** - Base model methods are not being deprecated as part of this change. This implementation only adds feature parity to enable future migration if desired.

---

## References

### Original Research
- Research document: `thoughts/shared/issues/2054/research.md`
- Issue #2054: "Add Gradient to plot posterior predictive in Plot Suite"

### Key Implementation Files
- `pymc_marketing/mmm/plot.py:187` - MMMPlotSuite class
- `pymc_marketing/mmm/plot.py:375` - MMMPlotSuite.posterior_predictive() method
- `pymc_marketing/mmm/base.py:362` - BaseValidateMMM._add_gradient_to_plot() reference implementation
- `pymc_marketing/mmm/base.py:625` - BaseValidateMMM.plot_posterior_predictive() method

### Test Files
- `tests/mmm/test_plot.py:185` - Existing posterior_predictive tests
- `tests/mmm/test_plotting.py:206-263` - Base model gradient tests (reference)

### Similar Patterns
- `pymc_marketing/mmm/plot.py:744-996` - Saturation curves (complex layered visualization)
- `pymc_marketing/mmm/plot.py:306` - `_add_median_and_hdi()` (helper method pattern)

---

## Assumptions

1. **xarray structure**: Assumes posterior_predictive data has `chain` and `draw` dimensions that get stacked into `sample`
2. **Date dimension**: Assumes coordinate dimension (typically "date") exists and has compatible values for x-axis plotting
3. **Matplotlib version**: Assumes matplotlib supports `plt.get_cmap()` and `fill_between()` (true for matplotlib >= 3.5)
4. **NumPy percentile**: Assumes `np.percentile()` behavior is consistent with base model implementation
5. **Backward compatibility**: Assumes existing tests use models with posterior_predictive data available
6. **Performance**: Assumes 30 percentiles is acceptable rendering time for typical use cases

---

## Implementation Checklist

- [ ] Phase 1: Implement `_add_gradient_to_axes()` helper method
- [ ] Phase 2: Integrate gradient into `posterior_predictive()` method
- [ ] Phase 3: Add comprehensive test coverage
- [ ] Phase 4: Verify imports and dependencies
- [ ] Phase 5: Enhance documentation with examples
- [ ] Run full test suite and verify no regressions
- [ ] Perform visual validation with real models
- [ ] Validate performance with multi-dimensional models
- [ ] Update any affected documentation or tutorials
- [ ] Create PR with all changes
