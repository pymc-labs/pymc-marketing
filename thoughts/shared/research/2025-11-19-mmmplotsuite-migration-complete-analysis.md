---
date: 2025-11-19T14:04:21+0000
researcher: Claude
git_commit: d6331a03727aa9c78ad16690aca25ce9cb869129
branch: feature/mmmplotsuite-arviz
repository: pymc-labs/pymc-marketing
topic: "MMMPlotSuite Migration - Complete Implementation Analysis and Requirements"
tags: [research, codebase, mmm, plotting, migration, backward-compatibility, testing, arviz-plots]
status: complete
last_updated: 2025-11-19
last_updated_by: Claude
---

# Research: MMMPlotSuite Migration - Complete Implementation Analysis and Requirements

**Date**: 2025-11-19T14:04:21+0000
**Researcher**: Claude
**Git Commit**: d6331a03727aa9c78ad16690aca25ce9cb869129
**Branch**: feature/mmmplotsuite-arviz
**Repository**: pymc-labs/pymc-marketing

## Research Question

The user is migrating MMMPlotSuite from matplotlib-based plotting to arviz_plots with multi-backend support. The legacy implementation is currently in `mmm/old_plot.py` and should be renamed to `mmm/legacy_plot.py`. To complete this migration, they need to:

1. Rename `old_plot.py` to `legacy_plot.py` and `OldMMMPlotSuite` to `LegacyMMMPlotSuite`
2. Support global backend configuration with per-function override capability
3. Implement backward compatibility with a flag to control legacy vs new behavior (default: legacy)
4. Add deprecation warning pointing to v0.20.0 removal
5. Review the new code implementation for quality issues
6. Create comprehensive tests for matplotlib, bokeh, and plotly backends

## Summary

Based on comprehensive codebase analysis, the migration is **75% complete** with critical gaps identified:

**✅ Already Implemented:**
- Backend configuration system with `mmm_config["plot.backend"]` supporting matplotlib/plotly/bokeh
- Complete new arviz_plots-based implementation returning `PlotCollection` objects
- Legacy matplotlib-based implementation preserved in `old_plot.py` (to be renamed `legacy_plot.py`)
- Per-method backend override via `backend` parameter on all plot methods

**❌ Missing Critical Components:**
- Rename `old_plot.py` to `legacy_plot.py` and `OldMMMPlotSuite` to `LegacyMMMPlotSuite`
- **Data Parameter Standardization**: All plotting methods should accept data as input parameters (some with fallback to `self.idata`, some without). Currently inconsistent across methods.
- **`_sensitivity_analysis_plot()` refactoring**: Must accept `data` as REQUIRED parameter (no fallback), and all callers (`sensitivity_analysis()`, `uplift_curve()`, `marginal_curve()`) must be updated to pass data explicitly.
- Backward compatibility flag (`use_v2`) to toggle between legacy/new suite
- Deprecation warning system for users
- Comprehensive backend testing for the new suite
- Compatibility test suite
- Documentation of breaking changes

**⚠️ Code Review Issues Found:**
- Return type documentation incomplete
- Breaking parameter type changes across all methods (intentional, no backward compatibility needed)
- Lost customization parameters (colors, subplot_kwargs, rc_params) - handled by arviz_plots
- **Deprecated method carried forward**: `saturation_curves_scatter()` is implemented in v2 but should be removed (already deprecated in v0.1.0)

## Detailed Findings

### 1. Current Architecture

#### 1.1 Class Definitions and Locations

**New Implementation:**
- **File**: [pymc_marketing/mmm/plot.py:187-1272](pymc_marketing/mmm/plot.py#L187-L1272)
- **Class**: `MMMPlotSuite`
- **Export**: `__all__ = ["MMMPlotSuite"]` at line 181
- **Technology**: arviz_plots library
- **Return Type**: `PlotCollection` (unified across all backends)

**Legacy Implementation:**
- **File**: [pymc_marketing/mmm/old_plot.py:191-1936](pymc_marketing/mmm/old_plot.py#L191-L1936) (to be renamed to `legacy_plot.py`)
- **Class**: `OldMMMPlotSuite` (to be renamed to `LegacyMMMPlotSuite`)
- **Export**: Not exported in any `__all__`
- **Technology**: matplotlib only
- **Return Type**: `tuple[Figure, NDArray[Axes]]` or `tuple[Figure, plt.Axes]`

**Integration Point:**
- **File**: [pymc_marketing/mmm/multidimensional.py:602-607](pymc_marketing/mmm/multidimensional.py#L602-L607)
- **Property**: `MMM.plot` returns `MMMPlotSuite(idata=self.idata)`
- **Issue**: Hardcoded to only return new suite, no version control

#### 1.2 Method Comparison Matrix

| Method | New Suite | Legacy Suite | API Compatible | Breaking Changes |
|--------|-----------|--------------|----------------|------------------|
| `__init__` | ✅ | ✅ | ✅ | None |
| `posterior_predictive()` | ✅ | ✅ | ❌ | `var: str` vs `list[str]`, return type |
| `contributions_over_time()` | ✅ | ✅ | ⚠️ | Return type only |
| `saturation_scatterplot()` | ✅ | ✅ | ⚠️ | Lost `**kwargs`, return type |
| `saturation_curves()` | ✅ | ✅ | ❌ | Lost colors, subplot_kwargs, rc_params |
| `saturation_curves_scatter()` | ⚠️ | ✅ | ⚠️ | **SHOULD BE REMOVED** - Currently in v2 but deprecated, delegates to saturation_scatterplot |
| `budget_allocation()` | ❌ | ✅ | ❌ | **REMOVED** - no replacement |
| `budget_allocation_roas()` | ✅ | ❌ | N/A | New method, different purpose |
| `allocated_contribution_by_channel_over_time()` | ✅ | ✅ | ❌ | Lost scale_factor, quantiles, figsize, ax |
| `sensitivity_analysis()` | ✅ | ✅ | ❌ | Lost ax, subplot_kwargs, plot_kwargs |
| `uplift_curve()` | ✅ | ✅ | ❌ | Lost ax, subplot_kwargs, plot_kwargs |
| `marginal_curve()` | ✅ | ✅ | ❌ | Lost ax, subplot_kwargs, plot_kwargs |

**Helper Methods:**
- New Suite: `_get_additional_dim_combinations()`, `_get_posterior_predictive_data()`, `_validate_dims()`, `_dim_list_handler()`, `_resolve_backend()`, `_sensitivity_analysis_plot()`
- Legacy Suite: `_init_subplots()`, `_build_subplot_title()`, `_reduce_and_stack()`, `_add_median_and_hdi()`, `_plot_budget_allocation_bars()` + shared helpers

### 2. Backend Configuration System ✅ **COMPLETE**

#### 2.1 Implementation

**File**: [pymc_marketing/mmm/config.py:21-66](pymc_marketing/mmm/config.py#L21-L66)

```python
VALID_BACKENDS = {"matplotlib", "plotly", "bokeh"}

class MMMConfig(dict):
    """Configuration dictionary for MMM plotting settings."""

    _defaults = {
        "plot.backend": "matplotlib",
        "plot.show_warnings": True,
    }

    def __setitem__(self, key, value):
        """Set config value with validation for backend."""
        if key == "plot.backend":
            if value not in VALID_BACKENDS:
                warnings.warn(
                    f"Invalid backend '{value}'. Valid backends are: {VALID_BACKENDS}. "
                    f"Setting anyway, but plotting may fail.",
                    UserWarning,
                    stacklevel=2,
                )
        super().__setitem__(key, value)

# Global config instance
mmm_config = MMMConfig()
```

#### 2.2 Backend Resolution

**File**: [pymc_marketing/mmm/plot.py:288-292](pymc_marketing/mmm/plot.py#L288-L292)

```python
def _resolve_backend(self, backend: str | None) -> str:
    """Resolve backend parameter to actual backend string."""
    from pymc_marketing.mmm.config import mmm_config
    return backend or mmm_config["plot.backend"]
```

#### 2.3 Usage Pattern

```python
from pymc_marketing.mmm import mmm_config

# Set global backend
mmm_config["plot.backend"] = "plotly"

# All plots use plotly
mmm.plot.posterior_predictive()

# Override for specific plot
mmm.plot.posterior_predictive(backend="matplotlib")
```

**Status**: ✅ No action needed - fully functional

### 3. Backward Compatibility ❌ **MISSING - CRITICAL**

#### 3.1 Current Gap

The `.plot` property currently only returns the new suite:

```python
# Current implementation in multidimensional.py:602-607
@property
def plot(self) -> MMMPlotSuite:
    """Use the MMMPlotSuite to plot the results."""
    self._validate_model_was_built()
    self._validate_idata_exists()
    return MMMPlotSuite(idata=self.idata)
```

#### 3.2 Required Implementation

**Step 1: Add flag to config.py**

```python
# File: pymc_marketing/mmm/config.py
_defaults = {
    "plot.backend": "matplotlib",
    "plot.show_warnings": True,
    "plot.use_v2": False,  # ← ADD THIS LINE
}
```

**Step 2: Implement version switching in multidimensional.py**

```python
# File: pymc_marketing/mmm/multidimensional.py:602-607
@property
def plot(self) -> MMMPlotSuite | LegacyMMMPlotSuite:
    """Use the MMMPlotSuite to plot the results."""
    from pymc_marketing.mmm.config import mmm_config
    from pymc_marketing.mmm.plot import MMMPlotSuite
    from pymc_marketing.mmm.legacy_plot import LegacyMMMPlotSuite
    import warnings

    self._validate_model_was_built()
    self._validate_idata_exists()

    # Check version flag
    if mmm_config.get("plot.use_v2", False):
        return MMMPlotSuite(idata=self.idata)
    else:
        # Show deprecation warning for legacy suite
        if mmm_config.get("plot.show_warnings", True):
            warnings.warn(
                "The current MMMPlotSuite will be deprecated in v0.20.0. "
                "The new version uses arviz_plots and supports multiple backends (matplotlib, plotly, bokeh). "
                "To use the new version: "
                ">>> from pymc_marketing.mmm.config import mmm_config\n"
                ">>> mmm_config['plot.use_v2'] = True\n"
                "To suppress this warning: mmm_config['plot.show_warnings'] = False\n"
                "See migration guide: https://docs.pymc-marketing.io/en/latest/mmm/plotting_migration.html",
                FutureWarning,
                stacklevel=2,
            )
        return LegacyMMMPlotSuite(idata=self.idata)
```

#### 3.3 Design Rationale

**Why `FutureWarning` instead of `DeprecationWarning`?**
- `DeprecationWarning` is for library developers (hidden by default in Python)
- `FutureWarning` is for end users (always shown)
- Our users are data scientists/analysts, not library developers
- Pattern found in [pymc_marketing/mlflow.py:180-185](pymc_marketing/mlflow.py#L180-L185)

**Why config flag instead of function parameter?**
- Consistent with existing backend configuration pattern
- Allows global setting affecting all plot calls
- Can be overridden per-session
- Pattern found throughout codebase (e.g., `plot.backend`)

**Why default to `False` (legacy suite)?**
- Non-breaking change in initial release
- Gives users time to migrate (1-2 releases)
- Prevents surprise breakage for existing code

### 4. Deprecation Patterns Research

Found **10 distinct patterns** used across the codebase:

#### Pattern 1: Parameter Name Deprecation with Helper
**Location**: [pymc_marketing/model_builder.py:60-77](pymc_marketing/model_builder.py#L60-L77)
**Test**: [tests/test_model_builder.py:530-554](tests/test_model_builder.py#L530-L554)

```python
def _handle_deprecate_pred_argument(value, name: str, kwargs: dict):
    name_pred = f"{name}_pred"

    if name_pred in kwargs and value is not None:
        raise ValueError(f"Both {name} and {name_pred} cannot be provided.")

    if name_pred in kwargs:
        warnings.warn(
            f"{name_pred} is deprecated, use {name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return kwargs.pop(name_pred)

    return value
```

#### Pattern 2: Method Deprecation with Delegation
**Location**: [pymc_marketing/mmm/plot.py:737-771](pymc_marketing/mmm/plot.py#L737-L771)
**Test**: [tests/mmm/test_plot.py:722-731](tests/mmm/test_plot.py#L722-L731)

```python
def saturation_curves_scatter(self, original_scale: bool = False, **kwargs) -> PlotCollection:
    """
    .. deprecated:: 0.1.0
       Will be removed in version 0.20.0. Use :meth:`saturation_scatterplot` instead.
    """
    import warnings
    warnings.warn(
        "saturation_curves_scatter is deprecated and will be removed in version 0.2.0. "
        "Use saturation_scatterplot instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.saturation_scatterplot(original_scale=original_scale, **kwargs)
```

#### Pattern 3: Config Key Renaming
**Location**: [pymc_marketing/clv/models/basic.py:49-59](pymc_marketing/clv/models/basic.py#L49-L59)

```python
deprecated_keys = [key for key in model_config if key.endswith("_prior")]
for key in deprecated_keys:
    new_key = key.replace("_prior", "")
    warnings.warn(
        f"The key '{key}' in model_config is deprecated. Use '{new_key}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    model_config[new_key] = model_config.pop(key)
```

#### Pattern 4: Module-Level Deprecation
**Location**: [pymc_marketing/deserialize.py:14-40](pymc_marketing/deserialize.py#L14-L40)

```python
warnings.warn(
    "The pymc_marketing.deserialize module is deprecated. "
    "Please use pymc_extras.deserialize instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

**Key Testing Pattern**: All deprecation warnings tested with `pytest.warns()`:

```python
def test_deprecation():
    with pytest.warns(DeprecationWarning, match=r"is deprecated"):
        result = deprecated_function()

    # Verify functionality still works
    assert isinstance(result, ExpectedType)
```

### 5. Code Review: Issues Found in New Implementation

#### Issue 1: Return Type Documentation ⚠️ **MINOR**

**Problem**: Method docstrings don't clearly state `PlotCollection` return type vs old `(Figure, Axes)` tuple.

**Location**: All methods in [plot.py:298-1272](pymc_marketing/mmm/plot.py#L298-L1272)

**Example** - `posterior_predictive()` docstring:
```python
def posterior_predictive(...) -> PlotCollection:
    """
    Plot posterior predictive distributions over time.

    Returns
    -------
    PlotCollection  # ← States type but doesn't explain what it is
    """
```

**Fix**: Add explanatory text:
```python
    Returns
    -------
    PlotCollection
        arviz_plots PlotCollection object containing the plot.
        Use .show() to display or .save("filename") to save.
        Unlike the old implementation which returned (Figure, Axes),
        this provides a unified interface across matplotlib, plotly, and bokeh backends.
```

#### Issue 2: Breaking Parameter Type Changes ✅ **INTENTIONAL - NO ACTION NEEDED**

**Status**: Many parameters have changed across all methods. Since this is a comprehensive migration to a new architecture (arviz_plots), these breaking changes are expected and documented.

**Examples of parameter changes**:

```python
# LEGACY (old_plot.py:387 - to be renamed to legacy_plot.py)
def posterior_predictive(
    self,
    var: list[str] | None = None,  # ← Accepts list
    ...
) -> tuple[Figure, NDArray[Axes]]:

# NEW (plot.py:300)
def posterior_predictive(
    self,
    var: str | None = None,  # ← Only accepts string
    ...
) -> PlotCollection:
```

**Rationale for no backward compatibility**:
- The entire API is changing (return types, parameters, behavior)
- Users switch to new suite explicitly via `mmm_config["plot.use_v2"] = True`
- Legacy suite remains available for those who need legacy parameter behavior
- Attempting to handle all parameter changes would add significant complexity for minimal benefit
- Migration guide will document all parameter changes with examples

**Action**: Document parameter changes in migration guide, let users adapt code when they opt into v2.

#### Issue 3: Missing Method ⚠️ **MAJOR**

**Problem**: `budget_allocation()` completely removed with no replacement.

**Legacy Method**: [old_plot.py:1049-1224](pymc_marketing/mmm/old_plot.py#L1049-L1224) (to be renamed to legacy_plot.py)
- Creates bar chart comparing allocated spend vs channel contributions
- Dual y-axis visualization

**New Method**: `budget_allocation_roas()` at [plot.py:773-874](pymc_marketing/mmm/plot.py#L773-L874)
- Completely different purpose (ROI distributions)
- Different parameters and output

**Impact**: Code using `mmm.plot.budget_allocation()` will fail with `AttributeError`.

**Recommendation**: Add stub method that raises helpful error:

```python
def budget_allocation(self, *args, **kwargs):
    """
    .. deprecated:: 0.18.0
       Removed in version 2.0. See budget_allocation_roas() for ROI distributions.

    Raises
    ------
    NotImplementedError
        This method was removed in MMMPlotSuite v2.
        For ROI distributions, use budget_allocation_roas().
        To use the old budget_allocation(), set mmm_config['plot.use_v2'] = False.
    """
    raise NotImplementedError(
        "budget_allocation() was removed in MMMPlotSuite v2. "
        "The new version uses arviz_plots which doesn't support this chart type. "
        "Options:\n"
        "  1. For ROI distributions: use budget_allocation_roas()\n"
        "  2. To use old method: set mmm_config['plot.use_v2'] = False\n"
        "  3. Implement custom bar chart using samples data"
    )
```

#### Issue 4: Backend Parameter Coverage ✅ **GOOD**

**Status**: All public methods have `backend` parameter:
- `posterior_predictive()` ✅
- `contributions_over_time()` ✅
- `saturation_scatterplot()` ✅
- `saturation_curves()` ✅
- `budget_allocation_roas()` ✅
- `allocated_contribution_by_channel_over_time()` ✅
- `sensitivity_analysis()` ✅
- `uplift_curve()` ✅
- `marginal_curve()` ✅
- ~~`saturation_curves_scatter()`~~ - **TO BE REMOVED** (deprecated method, see Issue 5)

**Pattern**: Consistent across all methods, properly resolves via `_resolve_backend()`.

#### Issue 5: Deprecated Method Should Be Removed ⚠️ **MINOR BUT IMPORTANT**

**Problem**: `saturation_curves_scatter()` is currently implemented in MMMPlotSuite v2 but is deprecated and just delegates to `saturation_scatterplot()`.

**Current implementation** (lines 737-771 in [plot.py](pymc_marketing/mmm/plot.py#L737-L771)):
```python
def saturation_curves_scatter(self, original_scale: bool = False, **kwargs) -> PlotCollection:
    """
    .. deprecated:: 0.1.0
       Will be removed in version 0.20.0. Use :meth:`saturation_scatterplot` instead.
    """
    import warnings
    warnings.warn(
        "saturation_curves_scatter is deprecated and will be removed in version 0.2.0. "
        "Use saturation_scatterplot instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.saturation_scatterplot(original_scale=original_scale, **kwargs)
```

**Rationale for removal**:
- Since MMMPlotSuite v2 is a completely new implementation, we should NOT carry forward deprecated methods
- The legacy suite (LegacyMMMPlotSuite) already has this method for users who need it
- Users who opt into v2 (`mmm_config["plot.use_v2"] = True`) should use the new, correct method name
- Keeping deprecated methods in v2 defeats the purpose of a clean migration
- The method was deprecated in v0.1.0, giving users ample time to migrate

**Recommendation**: **REMOVE** `saturation_curves_scatter()` from MMMPlotSuite (plot.py) entirely.

**Implementation**:
1. Delete the method from [pymc_marketing/mmm/plot.py:737-771](pymc_marketing/mmm/plot.py#L737-L771)
2. Keep it in LegacyMMMPlotSuite (legacy_plot.py) for backward compatibility
3. Document the removal in migration guide

**Alternative** (if keeping for one more release):
Add a note in the deprecation warning that it won't be available in v2 by default:
```python
warnings.warn(
    "saturation_curves_scatter is deprecated and will be removed in version 0.20.0. "
    "Use saturation_scatterplot instead. "
    "Note: This method is not available when using mmm_config['plot.use_v2'] = True.",
    DeprecationWarning,
    stacklevel=2,
)
```

**Preferred approach**: Clean removal from v2, keep only in legacy suite.

### 6. Testing Infrastructure ⚠️ **MAJOR GAPS**

#### 6.1 Current Test Coverage

**Test Files Found:**
1. [tests/mmm/test_plot.py](tests/mmm/test_plot.py) - 800+ lines
   - Contains ~28 test functions
   - Good fixture patterns
   - **Tests for LegacyMMMPlotSuite only** (currently using `old_plot.py`)
   - **NEW suite (plot.py) has NO test coverage**
   - **Needs new tests for the new MMMPlotSuite with all backends**

2. [tests/mmm/test_plot_backends.py](tests/mmm/test_plot_backends.py) - 255 lines
   - **EXPERIMENTAL FILE - SHOULD BE REMOVED**
   - Contains ~14 test functions
   - Only tests `posterior_predictive()` with multiple backends
   - Functionality should be merged into test_plot.py with parametrization

3. [tests/mmm/test_plotting.py](tests/mmm/test_plotting.py) - Legacy tests
   - Tests for old `BaseMMM` and `MMM` plotting
   - Not for MMMPlotSuite

**Test Coverage Analysis:**

| Method | Legacy Suite Tests | New Suite Tests | All Backends | Compatibility Tests |
|--------|-------------------|----------------|--------------|---------------------|
| `posterior_predictive()` | ✅ (matplotlib only) | ⚠️ (test_plot_backends.py only) | ❌ | ❌ |
| `contributions_over_time()` | ✅ (matplotlib only) | ❌ | ❌ | ❌ |
| `saturation_scatterplot()` | ✅ (matplotlib only) | ❌ | ❌ | ❌ |
| `saturation_curves()` | ✅ (matplotlib only) | ❌ | ❌ | ❌ |
| `budget_allocation()` | ✅ (matplotlib only) | N/A (removed) | ❌ | ❌ |
| `budget_allocation_roas()` | N/A (doesn't exist) | ❌ | ❌ | ❌ |
| `allocated_contribution_by_channel_over_time()` | ✅ (matplotlib only) | ❌ | ❌ | ❌ |
| `sensitivity_analysis()` | ✅ (matplotlib only) | ❌ | ❌ | ❌ |
| `uplift_curve()` | ✅ (matplotlib only) | ❌ | ❌ | ❌ |
| `marginal_curve()` | ✅ (matplotlib only) | ❌ | ❌ | ❌ |
| Config flag switching | ❌ | ❌ | ❌ | ❌ |
| Deprecation warnings | ❌ | ❌ | ❌ | ❌ |

**Coverage**:
- Legacy suite: ~80% (8 methods tested, matplotlib only)
- **New suite: ~1% (only 1 method partially tested in experimental file)**
- Compatibility tests: 0%

**Critical Gap**: The new MMMPlotSuite (plot.py) has essentially NO test coverage!

**Testing Strategy**:
- **Create new comprehensive tests for the new MMMPlotSuite**
- Parametrize all new tests to run against all backends (matplotlib, plotly, bokeh)
- Keep existing test_plot.py tests for legacy suite (will be removed in v0.20.0)
- Create separate compatibility test suite

#### 6.2 Available Test Fixtures

**From test_plot.py (for LegacyMMMPlotSuite):**
```python
@pytest.fixture(scope="module")
def mock_idata() -> az.InferenceData:
    """Basic mock InferenceData with posterior."""
    # Line 201

@pytest.fixture(scope="module")
def mock_idata_with_constant_data() -> az.InferenceData:
    """Mock InferenceData with constant_data for saturation plots."""
    # Line 315

@pytest.fixture(scope="module")
def mock_suite(mock_idata) -> LegacyMMMPlotSuite:
    """LegacyMMMPlotSuite instance with basic mock data."""
    # Line 290 - currently creates from old_plot

@pytest.fixture(scope="module")
def mock_suite_with_constant_data(mock_idata_with_constant_data) -> LegacyMMMPlotSuite:
    """LegacyMMMPlotSuite with constant data for saturation plots."""
    # Line 382 - currently creates from old_plot

@pytest.fixture
def mock_saturation_curve(mock_idata_with_constant_data) -> xr.DataArray:
    """Mock saturation curve DataArray."""
    # Line 388
```

**Pattern**: All fixtures use deterministic seeds for reproducibility.

**Note**: These fixtures will need to be adapted/duplicated for testing the new MMMPlotSuite.

#### 6.3 Required Test Implementation

**Strategy**: Create NEW comprehensive tests for the new MMMPlotSuite with multi-backend support

**Step 1: Keep existing test_plot.py for legacy suite**
- Rename test file to make it clear it's for legacy: `test_plot.py` → `test_legacy_plot.py`
- Update imports to use `legacy_plot.LegacyMMMPlotSuite`
- These tests will be removed in v0.20.0 along with the legacy suite

**Step 2: Create new test_plot.py for new MMMPlotSuite**

Create comprehensive tests with backend parametrization:

```python
"""Tests for new MMMPlotSuite with multi-backend support."""

import pytest
from arviz_plots import PlotCollection
from pymc_marketing.mmm import mmm_config
from pymc_marketing.mmm.plot import MMMPlotSuite

@pytest.fixture(scope="module")
def new_mock_suite(mock_idata) -> MMMPlotSuite:
    """New MMMPlotSuite instance with basic mock data."""
    return MMMPlotSuite(idata=mock_idata)

@pytest.fixture(scope="module")
def new_mock_suite_with_constant_data(mock_idata_with_constant_data) -> MMMPlotSuite:
    """New MMMPlotSuite with constant data for saturation plots."""
    return MMMPlotSuite(idata=mock_idata_with_constant_data)

# Parametrize all tests across all backends
@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_posterior_predictive(new_mock_suite, backend):
    """Test posterior_predictive works with all backends."""
    pc = new_mock_suite.posterior_predictive(backend=backend)
    assert isinstance(pc, PlotCollection)
    assert pc.backend == backend

@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_contributions_over_time(new_mock_suite, backend):
    """Test contributions_over_time works with all backends."""
    pc = new_mock_suite.contributions_over_time(
        var=["intercept"],
        backend=backend
    )
    assert isinstance(pc, PlotCollection)
    assert pc.backend == backend

@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_saturation_scatterplot(new_mock_suite_with_constant_data, backend):
    """Test saturation_scatterplot works with all backends."""
    pc = new_mock_suite_with_constant_data.saturation_scatterplot(backend=backend)
    assert isinstance(pc, PlotCollection)
    assert pc.backend == backend

@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_saturation_curves(
    new_mock_suite_with_constant_data, mock_saturation_curve, backend
):
    """Test saturation_curves works with all backends."""
    pc = new_mock_suite_with_constant_data.saturation_curves(
        curve=mock_saturation_curve,
        backend=backend
    )
    assert isinstance(pc, PlotCollection)
    assert pc.backend == backend

@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_budget_allocation_roas(new_mock_suite, backend):
    """Test budget_allocation_roas works with all backends."""
    # Note: This is a NEW method that doesn't exist in legacy suite
    pc = new_mock_suite.budget_allocation_roas(backend=backend)
    assert isinstance(pc, PlotCollection)
    assert pc.backend == backend

# ... Create tests for all 8 methods with 3 backends = 24 core tests ...
```

**Step 3: Remove experimental test_plot_backends.py**
```bash
rm tests/mmm/test_plot_backends.py
```

**Step 4: Add backend-specific tests**

```python
def test_backend_overrides_global_config(mock_suite):
    """Test that method backend parameter overrides global config."""
    original = mmm_config.get("plot.backend", "matplotlib")
    try:
        mmm_config["plot.backend"] = "matplotlib"

        # Override with plotly
        pc = mock_suite.contributions_over_time(
            var=["intercept"],
            backend="plotly"
        )
        assert pc.backend == "plotly"

        # Default should still be matplotlib
        pc2 = mock_suite.contributions_over_time(var=["intercept"])
        assert pc2.backend == "matplotlib"
    finally:
        mmm_config["plot.backend"] = original

def test_invalid_backend_warning(mock_suite):
    """Test that invalid backend shows warning but attempts plot."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mmm_config["plot.backend"] = "invalid_backend"

        assert len(w) == 1
        assert "Invalid backend" in str(w[0].message)
```

**Result**:
- New suite: ~8 methods × 3 backends = ~24 core test cases (plus backend-specific tests) - note: saturation_curves_scatter removed
- Legacy suite: ~28 existing test functions (matplotlib only, will be removed in v0.20.0)

**File 2: Create tests/mmm/test_plot_compatibility.py**

New file for backward compatibility:

```python
"""Tests for MMMPlotSuite backward compatibility and version switching."""

import pytest
import warnings
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from arviz_plots import PlotCollection

from pymc_marketing.mmm import mmm_config
from pymc_marketing.mmm.plot import MMMPlotSuite
from pymc_marketing.mmm.legacy_plot import LegacyMMMPlotSuite


class TestVersionSwitching:
    """Test mmm_config['plot.use_v2'] flag controls suite version."""

    def test_use_v2_false_returns_legacy_suite(self, mock_mmm):
        """Test that use_v2=False returns LegacyMMMPlotSuite."""
        original = mmm_config.get("plot.use_v2", False)
        try:
            mmm_config["plot.use_v2"] = False

            with pytest.warns(FutureWarning, match="deprecated in v0.20.0"):
                plot_suite = mock_mmm.plot

            assert isinstance(plot_suite, LegacyMMMPlotSuite)
            assert not isinstance(plot_suite, MMMPlotSuite)
        finally:
            mmm_config["plot.use_v2"] = original

    def test_use_v2_true_returns_new_suite(self, mock_mmm):
        """Test that use_v2=True returns MMMPlotSuite."""
        original = mmm_config.get("plot.use_v2", False)
        try:
            mmm_config["plot.use_v2"] = True

            # Should not warn
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Turn warnings into errors
                plot_suite = mock_mmm.plot

            assert isinstance(plot_suite, MMMPlotSuite)
        finally:
            mmm_config["plot.use_v2"] = original

    def test_default_is_legacy_suite(self, mock_mmm):
        """Test that default behavior uses legacy suite (backward compatible)."""
        # Reset to defaults
        mmm_config.reset()

        with pytest.warns(FutureWarning):
            plot_suite = mock_mmm.plot

        assert isinstance(plot_suite, LegacyMMMPlotSuite)


class TestDeprecationWarnings:
    """Test deprecation warning system."""

    def test_deprecation_warning_shown_by_default(self, mock_mmm):
        """Test that deprecation warning is shown when using legacy suite."""
        mmm_config["plot.use_v2"] = False
        mmm_config["plot.show_warnings"] = True

        with pytest.warns(FutureWarning, match=r"deprecated in v0\.20\.0"):
            plot_suite = mock_mmm.plot

        assert isinstance(plot_suite, LegacyMMMPlotSuite)

    def test_deprecation_warning_suppressible(self, mock_mmm):
        """Test that deprecation warning can be suppressed."""
        original_use_v2 = mmm_config.get("plot.use_v2", False)
        original_warnings = mmm_config.get("plot.show_warnings", True)

        try:
            mmm_config["plot.use_v2"] = False
            mmm_config["plot.show_warnings"] = False

            # Should not warn
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Turn warnings into errors
                plot_suite = mock_mmm.plot

            assert isinstance(plot_suite, LegacyMMMPlotSuite)
        finally:
            mmm_config["plot.use_v2"] = original_use_v2
            mmm_config["plot.show_warnings"] = original_warnings

    def test_warning_message_includes_migration_info(self, mock_mmm):
        """Test that warning provides clear migration instructions."""
        mmm_config["plot.use_v2"] = False
        mmm_config["plot.show_warnings"] = True

        with pytest.warns(FutureWarning) as warning_list:
            plot_suite = mock_mmm.plot

        warning_msg = str(warning_list[0].message)
        assert "v0.20.0" in warning_msg
        assert "mmm_config['plot.use_v2'] = True" in warning_msg
        assert "migration guide" in warning_msg.lower() or "documentation" in warning_msg.lower()

    def test_no_warning_when_using_new_suite(self, mock_mmm):
        """Test that no warning shown when using new suite."""
        mmm_config["plot.use_v2"] = True

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot_suite = mock_mmm.plot

        assert isinstance(plot_suite, MMMPlotSuite)


class TestReturnTypeCompatibility:
    """Test that both suites return expected types."""

    def test_legacy_suite_returns_tuple(self, mock_mmm_fitted):
        """Test legacy suite returns (Figure, Axes) tuple."""
        mmm_config["plot.use_v2"] = False

        with pytest.warns(FutureWarning):
            plot_suite = mock_mmm_fitted.plot
            result = plot_suite.posterior_predictive()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Figure)
        # result[1] can be Axes or ndarray of Axes
        if isinstance(result[1], np.ndarray):
            assert all(isinstance(ax, Axes) for ax in result[1].flat)
        else:
            assert isinstance(result[1], Axes)

    def test_new_suite_returns_plot_collection(self, mock_mmm_fitted):
        """Test new suite returns PlotCollection."""
        mmm_config["plot.use_v2"] = True

        plot_suite = mock_mmm_fitted.plot
        result = plot_suite.posterior_predictive()

        assert isinstance(result, PlotCollection)
        assert hasattr(result, 'backend')
        assert hasattr(result, 'show')

    def test_both_suites_produce_valid_plots(self, mock_mmm_fitted):
        """Test that both suites can successfully create plots."""
        # Legacy suite
        mmm_config["plot.use_v2"] = False
        with pytest.warns(FutureWarning):
            legacy_result = mock_mmm_fitted.plot.contributions_over_time(
                var=["intercept"]
            )
        assert legacy_result is not None

        # New suite
        mmm_config["plot.use_v2"] = True
        new_result = mock_mmm_fitted.plot.contributions_over_time(
            var=["intercept"]
        )
        assert new_result is not None


class TestMissingMethods:
    """Test handling of methods that exist in one suite but not the other."""

    def test_budget_allocation_exists_in_legacy_suite(self, mock_mmm_fitted, mock_allocation_samples):
        """Test that budget_allocation() works in legacy suite."""
        mmm_config["plot.use_v2"] = False

        with pytest.warns(FutureWarning):
            plot_suite = mock_mmm_fitted.plot

        # Should work (not raise AttributeError)
        result = plot_suite.budget_allocation(samples=mock_allocation_samples)
        assert isinstance(result, tuple)

    def test_budget_allocation_raises_in_new_suite(self, mock_mmm_fitted):
        """Test that budget_allocation() raises helpful error in new suite."""
        mmm_config["plot.use_v2"] = True
        plot_suite = mock_mmm_fitted.plot

        with pytest.raises(NotImplementedError, match="removed in MMMPlotSuite v2"):
            plot_suite.budget_allocation(samples=None)

    def test_budget_allocation_roas_exists_in_new_suite(
        self, mock_mmm_fitted, mock_allocation_samples
    ):
        """Test that budget_allocation_roas() works in new suite."""
        mmm_config["plot.use_v2"] = True
        plot_suite = mock_mmm_fitted.plot

        result = plot_suite.budget_allocation_roas(samples=mock_allocation_samples)
        assert isinstance(result, PlotCollection)

    def test_budget_allocation_roas_missing_in_legacy_suite(self, mock_mmm_fitted):
        """Test that budget_allocation_roas() doesn't exist in legacy suite."""
        mmm_config["plot.use_v2"] = False

        with pytest.warns(FutureWarning):
            plot_suite = mock_mmm_fitted.plot

        with pytest.raises(AttributeError):
            plot_suite.budget_allocation_roas(samples=None)


class TestParameterCompatibility:
    """Test parameter compatibility between suites."""

    def test_var_parameter_list_in_legacy_suite(self, mock_mmm_fitted):
        """Test that legacy suite accepts var as list."""
        mmm_config["plot.use_v2"] = False

        with pytest.warns(FutureWarning):
            plot_suite = mock_mmm_fitted.plot

        # Should accept list
        result = plot_suite.posterior_predictive(var=["y", "target"])
        assert isinstance(result, tuple)

    def test_var_parameter_list_warning_in_new_suite(self, mock_mmm_fitted):
        """Test that new suite warns when given list for var."""
        mmm_config["plot.use_v2"] = True
        plot_suite = mock_mmm_fitted.plot

        with pytest.warns(UserWarning, match="only supports single variable"):
            result = plot_suite.posterior_predictive(var=["y"])

        assert isinstance(result, PlotCollection)
```

**File 3: Additional fixtures in tests/conftest.py or tests/mmm/conftest.py**

```python
@pytest.fixture
def mock_mmm(mock_idata):
    """Mock MMM instance with idata."""
    from pymc_marketing.mmm.multidimensional import MMM

    mmm = Mock(spec=MMM)
    mmm.idata = mock_idata
    mmm._validate_model_was_built = Mock()
    mmm._validate_idata_exists = Mock()

    # Make .plot property work
    type(mmm).plot = MMM.plot

    return mmm

@pytest.fixture
def mock_allocation_samples():
    """Mock samples dataset for budget allocation tests."""
    import xarray as xr
    import numpy as np

    rng = np.random.default_rng(42)

    return xr.Dataset({
        "channel_contribution_original_scale": xr.DataArray(
            rng.normal(size=(4, 100, 52, 3)),
            dims=("chain", "draw", "date", "channel"),
            coords={
                "chain": np.arange(4),
                "draw": np.arange(100),
                "date": pd.date_range("2025-01-01", periods=52, freq="W"),
                "channel": ["TV", "Radio", "Digital"],
            },
        ),
        "allocation": xr.DataArray(
            rng.uniform(100, 1000, size=(3,)),
            dims=("channel",),
            coords={"channel": ["TV", "Radio", "Digital"]},
        ),
    })
```

#### 6.4 Test Execution Checklist

**Backend Testing:**
- [ ] Remove experimental test_plot_backends.py file
- [ ] Remove deprecated `saturation_curves_scatter()` from MMMPlotSuite
- [ ] Parametrize all tests in new test_plot.py with backend parameter (~8 methods)
- [ ] All ~24 parametrized tests pass (8 methods × 3 backends)
- [ ] Backend override test works correctly
- [ ] Invalid backend warning test passes

**Compatibility Testing:**
- [ ] Create new test_plot_compatibility.py file
- [ ] All 15+ compatibility tests pass
- [ ] Config flag switching works
- [ ] Deprecation warnings show correctly
- [ ] Warnings are suppressible
- [ ] Both suites produce valid output
- [ ] Missing method raises helpful errors

### 7. Import/Export Architecture

#### 7.1 Current Import Chain

```
User Code
    ↓
from pymc_marketing.mmm.multidimensional import MMM
    ↓
MMM.plot property (multidimensional.py:602-607)
    ↓
Imports: from pymc_marketing.mmm.plot import MMMPlotSuite
    ↓
Returns: MMMPlotSuite(idata=self.idata)
```

#### 7.2 Required Imports for Compatibility

**In multidimensional.py:**
```python
# Current (line 194)
from pymc_marketing.mmm.plot import MMMPlotSuite

# Need to add in .plot property
from pymc_marketing.mmm.legacy_plot import LegacyMMMPlotSuite  # Import locally in property
from pymc_marketing.mmm.config import mmm_config  # Import locally in property
```

**NOT in mmm/__init__.py:**
- MMMPlotSuite is **not** exported in [pymc_marketing/mmm/__init__.py](pymc_marketing/mmm/__init__.py#L69-L119)
- Users access it via `mmm.plot.method()`, not by importing directly
- This is good - no need to modify `__all__`

#### 7.3 User Usage Pattern

```python
# Users do this:
from pymc_marketing.mmm.multidimensional import MMM

mmm = MMM(...)
mmm.fit(...)

# Access via property - this is where version switching happens
mmm.plot.posterior_predictive()  # ← Property returns either old or new suite
```

### 8. Migration Timeline

#### Phase 1: v0.18.0 (Current/Next Release)
**Goal**: Introduce new suite with safe fallback

- ✅ Backend configuration (done)
- ✅ New suite implementation (done)
- ❌ Add `use_v2` flag to config (TODO)
- ❌ Implement version switching in `.plot` property (TODO)
- ❌ Add deprecation warning (TODO)
- ❌ Complete test coverage (TODO)
- ❌ Write migration guide documentation (TODO)

**User Experience**:
- Default behavior: legacy suite with warning
- Opt-in to new: `mmm_config["plot.use_v2"] = True`
- Clear migration path provided

#### Phase 2: v0.19.0
**Goal**: Encourage migration to new suite

- Change default: `"plot.use_v2": True`
- Keep legacy suite available via `use_v2=False`
- Strengthen warning when using legacy suite
- Monitor for issues

**User Experience**:
- Default behavior: new suite
- Opt-out to legacy: `mmm_config["plot.use_v2"] = False`
- Legacy suite shows stronger deprecation warning

#### Phase 3: v0.20.0
**Goal**: Complete migration

- Remove `LegacyMMMPlotSuite` class
- Remove `legacy_plot.py` file
- Remove `use_v2` flag
- Update all documentation
- Only new suite available

**User Experience**:
- Only new suite available
- Legacy code must update to new API

### 9. Breaking Changes Summary

#### 9.1 Return Type

**Legacy**: `tuple[Figure, NDArray[Axes]]` or `tuple[Figure, plt.Axes]`
```python
fig, axes = mmm.plot.posterior_predictive()
axes[0].set_title("Custom")
fig.savefig("plot.png")
```

**New**: `PlotCollection`
```python
pc = mmm.plot.posterior_predictive()
pc.show()  # Display
pc.save("plot.png")  # Save
```

#### 9.2 Parameter Changes

| Method | Parameter | Legacy | New | Fix |
|--------|-----------|--------|-----|-----|
| `posterior_predictive()` | `var` | `list[str]` | `str` | Call multiple times or use list with warning |
| `saturation_scatterplot()` | `**kwargs` | Accepted | Removed | Customize PlotCollection after |
| `saturation_curves()` | `colors` | Supported | Removed | Use PlotCollection API |
| `saturation_curves()` | `subplot_kwargs` | Supported | Removed | Use PlotCollection API |
| `saturation_curves()` | `rc_params` | Supported | Removed | Set before calling |
| All methods | `ax` | Supported | Removed | Use PlotCollection |
| All methods | `figsize` | Supported | Removed | Use PlotCollection |
| All methods | `backend` | N/A | Added | Override global config |

#### 9.3 Method Changes

| Method | Status | Replacement | Notes |
|--------|--------|-------------|-------|
| `saturation_curves_scatter()` | **REMOVED in v2** | `saturation_scatterplot()` | Deprecated in v0.1.0, not carried forward to v2 |
| `budget_allocation()` | **REMOVED** | None exact | Use legacy suite or custom plot |
| `budget_allocation_roas()` | **NEW** | N/A | Different purpose (ROI dist) |

### 10. Documentation Requirements

#### 10.1 Migration Guide (docs/source/guides/mmm_plotting_migration.rst)

Must include:

1. **Overview**
   - Why the change (arviz_plots benefits)
   - Timeline (v0.18.0 intro, v0.19.0 default, v0.20.0 removal)
   - How to opt-in/opt-out

2. **Quick Start**
   ```python
   # Use new suite
   from pymc_marketing.mmm import mmm_config
   mmm_config["plot.use_v2"] = True

   # Set backend
   mmm_config["plot.backend"] = "plotly"
   ```

3. **Return Type Migration**
   - Side-by-side examples
   - How to work with PlotCollection

4. **Method-by-Method Guide**
   - API changes table
   - Code examples for each method
   - Common issues and solutions

5. **Missing Features**
   - `budget_allocation()` alternatives
   - Lost customization parameters
   - Workarounds

6. **Backend Selection**
   - Pros/cons of each backend
   - When to use which
   - Examples

#### 10.2 Docstring Updates

All methods in new suite need:
```python
def method_name(...) -> PlotCollection:
    """
    Description.

    .. versionadded:: 0.18.0
       New arviz_plots-based implementation supporting multiple backends.

    Parameters
    ----------
    backend : str, optional
        Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
        If None, uses global config via mmm_config["plot.backend"].
        Default is "matplotlib".

    Returns
    -------
    PlotCollection
        arviz_plots PlotCollection object containing the plot.
        Use .show() to display or .save("filename") to save.
        Supports matplotlib, plotly, and bokeh backends.

        Unlike v1 which returned (Figure, Axes), this provides
        a unified interface across all backends.

    Examples
    --------
    Basic usage:

    >>> pc = mmm.plot.method_name()
    >>> pc.show()

    Save to file:

    >>> pc.save("output.png")

    Use different backend:

    >>> pc = mmm.plot.method_name(backend="plotly")
    >>> pc.show()
    """
```

## Code References

### Core Implementation Files
- [pymc_marketing/mmm/plot.py](pymc_marketing/mmm/plot.py) - New MMMPlotSuite (1272 lines)
- [pymc_marketing/mmm/old_plot.py](pymc_marketing/mmm/old_plot.py) - Legacy implementation (1936 lines) - **TO BE RENAMED to legacy_plot.py**
- [pymc_marketing/mmm/config.py:21-66](pymc_marketing/mmm/config.py#L21-L66) - Backend configuration
- [pymc_marketing/mmm/multidimensional.py:602-607](pymc_marketing/mmm/multidimensional.py#L602-L607) - Integration point (.plot property)

### Test Files
- [tests/mmm/test_plot.py](tests/mmm/test_plot.py) - Main plot tests (800+ lines)
- [tests/mmm/test_plot_backends.py](tests/mmm/test_plot_backends.py) - Backend tests (255 lines, incomplete)
- [tests/mmm/test_plotting.py](tests/mmm/test_plotting.py) - Legacy plotting tests

### Deprecation Patterns
- [pymc_marketing/model_builder.py:60-77](pymc_marketing/model_builder.py#L60-L77) - Parameter deprecation helper
- [pymc_marketing/mmm/plot.py:737-771](pymc_marketing/mmm/plot.py#L737-L771) - Method deprecation example
- [pymc_marketing/clv/models/basic.py:49-59](pymc_marketing/clv/models/basic.py#L49-L59) - Config key deprecation
- [tests/test_model_builder.py:530-554](tests/test_model_builder.py#L530-L554) - Deprecation test pattern

## Architecture Insights

1. **Config-Based Feature Flags**: The codebase uses dict-based configuration (`mmm_config`) for runtime behavior control, similar to matplotlib's `rcParams` or arviz's config system.

2. **Property-Based API**: Plot methods are accessed via `.plot` property that creates instances on-demand, enabling clean version switching at the access point.

3. **Backend Abstraction**: The new implementation achieves backend independence through arviz_plots' `PlotCollection`, which handles backend-specific rendering internally.

4. **Test Fixture Patterns**: All test fixtures use deterministic random seeds and module scope for performance, following pytest best practices.

5. **Deprecation Philosophy**: The codebase uses `DeprecationWarning` for library developers and `FutureWarning` for end users, with clear migration paths in all warnings.

6. **Incremental Migration**: Multiple patterns show support for gradual API transitions over several releases before removing old code.

## Data Parameter Standardization ⚠️ **CRITICAL - MUST IMPLEMENT**

### Summary

**Goal**: All plotting methods should accept data as input parameters for consistency, testability, and flexibility.

**Status**: Currently **inconsistent** - some methods accept data, others hard-code `self.idata` access.

**Impact**: Must be fixed BEFORE writing tests, as tests need to be written against the correct API.

**Time Estimate**: 4 hours

**Key Changes**:
- 7 methods need updates
- `_sensitivity_analysis_plot()` must accept `data` as REQUIRED parameter (no fallback)
- All other methods can have fallback to `self.idata`
- Removes monkey-patching in `uplift_curve()` and `marginal_curve()`

### Current State Analysis

The new MMMPlotSuite methods currently have **inconsistent data parameter patterns**:

**✅ Methods that already accept data as input:**
- `posterior_predictive(idata: xr.Dataset | None)` - With fallback to `self.idata.posterior_predictive`
- `budget_allocation_roas(samples: xr.Dataset)` - No fallback
- `allocated_contribution_by_channel_over_time(samples: xr.Dataset)` - No fallback

**❌ Methods that need data parameters added:**
- `contributions_over_time()` - Currently uses `self.idata.posterior` directly
- `saturation_scatterplot()` - Currently uses `self.idata.constant_data` and `self.idata.posterior`
- `saturation_curves()` - Accepts `curve` but still uses `self.idata.constant_data` and `self.idata.posterior` for scatter
- `_sensitivity_analysis_plot()` - Currently uses `self.idata.sensitivity_analysis` (**must accept data, NO fallback**)
- `sensitivity_analysis()` - Needs to accept and pass data to `_sensitivity_analysis_plot()`
- `uplift_curve()` - Needs to accept and pass data to `_sensitivity_analysis_plot()`
- `marginal_curve()` - Needs to accept and pass data to `_sensitivity_analysis_plot()`

### Required API Changes

#### 1. contributions_over_time() - Add data parameter with fallback

**Current signature** (line 387):
```python
def contributions_over_time(
    self,
    var: list[str],
    hdi_prob: float = 0.85,
    dims: dict[str, str | int | list] | None = None,
    backend: str | None = None,
) -> PlotCollection:
```

**New signature**:
```python
def contributions_over_time(
    self,
    var: list[str],
    data: xr.Dataset | None = None,  # ← ADD THIS
    hdi_prob: float = 0.85,
    dims: dict[str, str | int | list] | None = None,
    backend: str | None = None,
) -> PlotCollection:
    """Plot the time-series contributions for each variable in `var`.

    Parameters
    ----------
    var : list of str
        A list of variable names to plot from the posterior.
    data : xr.Dataset, optional
        Dataset containing posterior data. If None, uses self.idata.posterior.
    ...
    """
```

**Implementation changes** (lines 426-437):
```python
# OLD:
if not hasattr(self.idata, "posterior"):
    raise ValueError(...)
da = self.idata.posterior[var]

# NEW:
if data is None:
    if not hasattr(self.idata, "posterior"):
        raise ValueError(
            "No posterior data found in 'self.idata' and no 'data' argument provided. "
            "Please ensure 'self.idata' contains a 'posterior' group or provide 'data'."
        )
    data = self.idata.posterior
da = data[var]
```

#### 2. saturation_scatterplot() - Add data parameters with fallback

**Current signature** (line 493):
```python
def saturation_scatterplot(
    self,
    original_scale: bool = False,
    dims: dict[str, str | int | list] | None = None,
    backend: str | None = None,
) -> PlotCollection:
```

**New signature**:
```python
def saturation_scatterplot(
    self,
    original_scale: bool = False,
    constant_data: xr.Dataset | None = None,  # ← ADD THIS
    posterior_data: xr.Dataset | None = None,  # ← ADD THIS
    dims: dict[str, str | int | list] | None = None,
    backend: str | None = None,
) -> PlotCollection:
    """Plot the saturation curves for each channel.

    Parameters
    ----------
    original_scale : bool, optional
        Whether to plot the original scale contributions. Default is False.
    constant_data : xr.Dataset, optional
        Dataset containing constant_data group with 'channel_data' variable.
        If None, uses self.idata.constant_data.
    posterior_data : xr.Dataset, optional
        Dataset containing posterior group with channel contribution variables.
        If None, uses self.idata.posterior.
    ...
    """
```

**Implementation changes** (lines 524-562):
```python
# OLD:
if not hasattr(self.idata, "constant_data"):
    raise ValueError(...)
cdims = self.idata.constant_data.channel_data.dims
channel_data = self.idata.constant_data.channel_data
channel_contrib = self.idata.posterior[channel_contribution]

# NEW:
if constant_data is None:
    if not hasattr(self.idata, "constant_data"):
        raise ValueError(
            "No 'constant_data' found in 'self.idata' and no 'constant_data' argument provided. "
            "Please ensure 'self.idata' contains the constant_data group or provide 'constant_data'."
        )
    constant_data = self.idata.constant_data

if posterior_data is None:
    if not hasattr(self.idata, "posterior"):
        raise ValueError(
            "No 'posterior' found in 'self.idata' and no 'posterior_data' argument provided. "
            "Please ensure 'self.idata' contains the posterior group or provide 'posterior_data'."
        )
    posterior_data = self.idata.posterior

cdims = constant_data.channel_data.dims
channel_data = constant_data.channel_data
channel_contrib = posterior_data[channel_contribution]
```

#### 3. saturation_curves() - Update to use data parameters from saturation_scatterplot

**Current signature** (line 597):
```python
def saturation_curves(
    self,
    curve: xr.DataArray,
    original_scale: bool = False,
    n_samples: int = 10,
    hdi_probs: float | list[float] | None = None,
    random_seed: np.random.Generator | None = None,
    dims: dict[str, str | int | list] | None = None,
    backend: str | None = None,
) -> PlotCollection:
```

**New signature**:
```python
def saturation_curves(
    self,
    curve: xr.DataArray,
    original_scale: bool = False,
    constant_data: xr.Dataset | None = None,  # ← ADD THIS
    posterior_data: xr.Dataset | None = None,  # ← ADD THIS
    n_samples: int = 10,
    hdi_probs: float | list[float] | None = None,
    random_seed: np.random.Generator | None = None,
    dims: dict[str, str | int | list] | None = None,
    backend: str | None = None,
) -> PlotCollection:
    """Overlay saturation‑curve scatter‑plots with posterior‑predictive sample curves.

    Parameters
    ----------
    curve : xr.DataArray
        Posterior‑predictive curves (e.g. dims `("chain","draw","x","channel","geo")`).
    original_scale : bool, default=False
        Plot `channel_contribution_original_scale` if True, else `channel_contribution`.
    constant_data : xr.Dataset, optional
        Dataset containing constant_data group. If None, uses self.idata.constant_data.
    posterior_data : xr.Dataset, optional
        Dataset containing posterior group. If None, uses self.idata.posterior.
    ...
    """
```

**Implementation changes** (lines 645-696):
```python
# OLD:
if not hasattr(self.idata, "constant_data"):
    raise ValueError(...)
if original_scale:
    curve_data = curve * self.idata.constant_data.target_scale
    curve_data["x"] = curve_data["x"] * self.idata.constant_data.channel_scale
cdims = self.idata.constant_data.channel_data.dims
pc = self.saturation_scatterplot(original_scale=original_scale, dims=dims, backend=backend)

# NEW:
if constant_data is None:
    if not hasattr(self.idata, "constant_data"):
        raise ValueError(
            "No 'constant_data' found in 'self.idata' and no 'constant_data' argument provided."
        )
    constant_data = self.idata.constant_data

if posterior_data is None:
    if not hasattr(self.idata, "posterior"):
        raise ValueError(
            "No 'posterior' found in 'self.idata' and no 'posterior_data' argument provided."
        )
    posterior_data = self.idata.posterior

if original_scale:
    curve_data = curve * constant_data.target_scale
    curve_data["x"] = curve_data["x"] * constant_data.channel_scale
cdims = constant_data.channel_data.dims
pc = self.saturation_scatterplot(
    original_scale=original_scale,
    constant_data=constant_data,
    posterior_data=posterior_data,
    dims=dims,
    backend=backend
)
```

#### 4. _sensitivity_analysis_plot() - Accept data parameter WITHOUT fallback ⚠️ **CRITICAL**

**Current signature** (line 979):
```python
def _sensitivity_analysis_plot(
    self,
    hdi_prob: float = 0.94,
    aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
    backend: str | None = None,
) -> PlotCollection:
```

**New signature**:
```python
def _sensitivity_analysis_plot(
    self,
    data: xr.DataArray | xr.Dataset,  # ← ADD THIS (REQUIRED, NO DEFAULT)
    hdi_prob: float = 0.94,
    aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
    backend: str | None = None,
) -> PlotCollection:
    """Plot helper for sensitivity analysis results.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Sensitivity analysis data to plot. Must have 'sample' and 'sweep' dimensions.
        If Dataset, should contain 'x' variable. NO fallback to self.idata.
    ...
    """
```

**Implementation changes** (lines 1002-1007):
```python
# OLD:
if not hasattr(self.idata, "sensitivity_analysis"):
    raise ValueError("No sensitivity analysis results found. Run run_sweep() first.")
sa = self.idata.sensitivity_analysis
x = sa["x"] if isinstance(sa, xr.Dataset) else sa

# NEW:
# Validate input data
if data is None:
    raise ValueError(
        "data parameter is required for _sensitivity_analysis_plot. "
        "This is a helper method that should receive data explicitly."
    )

# Handle Dataset or DataArray
x = data["x"] if isinstance(data, xr.Dataset) else data
```

**Rationale for NO fallback:**
- This is a **private helper method** (prefixed with `_`)
- It should be a pure plotting function that operates on provided data
- The public methods (`sensitivity_analysis()`, `uplift_curve()`, `marginal_curve()`) handle data retrieval from `self.idata`
- This separation of concerns makes the code more testable and maintainable

#### 5. sensitivity_analysis() - Update to pass data

**Current implementation** (lines 1071-1116):
```python
def sensitivity_analysis(
    self,
    hdi_prob: float = 0.94,
    aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
    backend: str | None = None,
) -> PlotCollection:
    pc = self._sensitivity_analysis_plot(
        hdi_prob=hdi_prob, aggregation=aggregation, backend=backend
    )
    pc.map(azp.visuals.labelled_y, text="Contribution")
    return pc
```

**New implementation**:
```python
def sensitivity_analysis(
    self,
    data: xr.DataArray | xr.Dataset | None = None,  # ← ADD THIS
    hdi_prob: float = 0.94,
    aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
    backend: str | None = None,
) -> PlotCollection:
    """Plot sensitivity analysis results.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset, optional
        Sensitivity analysis data to plot. If None, uses self.idata.sensitivity_analysis.
    ...
    """
    # Retrieve data if not provided
    if data is None:
        if not hasattr(self.idata, "sensitivity_analysis"):
            raise ValueError(
                "No sensitivity analysis results found in 'self.idata' and no 'data' argument provided. "
                "Run 'mmm.sensitivity.run_sweep()' first or provide 'data'."
            )
        data = self.idata.sensitivity_analysis  # type: ignore

    pc = self._sensitivity_analysis_plot(
        data=data,  # ← PASS DATA
        hdi_prob=hdi_prob,
        aggregation=aggregation,
        backend=backend,
    )
    pc.map(azp.visuals.labelled_y, text="Contribution")
    return pc
```

#### 6. uplift_curve() - Update to pass data

**Current implementation** (lines 1158-1193):
```python
def uplift_curve(
    self,
    hdi_prob: float = 0.94,
    aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
    backend: str | None = None,
) -> PlotCollection:
    if not hasattr(self.idata, "sensitivity_analysis"):
        raise ValueError(...)

    sa_group = self.idata.sensitivity_analysis
    if isinstance(sa_group, xr.Dataset):
        if "uplift_curve" not in sa_group:
            raise ValueError(...)
        data_var = sa_group["uplift_curve"]
    else:
        raise ValueError(...)

    # Monkey-patch approach with temporary swap
    tmp_idata = xr.Dataset({"x": data_var})
    original_group = self.idata.sensitivity_analysis
    try:
        self.idata.sensitivity_analysis = tmp_idata
        pc = self._sensitivity_analysis_plot(...)
        ...
    finally:
        self.idata.sensitivity_analysis = original_group
```

**New implementation**:
```python
def uplift_curve(
    self,
    data: xr.DataArray | xr.Dataset | None = None,  # ← ADD THIS
    hdi_prob: float = 0.94,
    aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
    backend: str | None = None,
) -> PlotCollection:
    """Plot precomputed uplift curves.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset, optional
        Uplift curve data to plot. If Dataset, should contain 'uplift_curve' variable.
        If None, uses self.idata.sensitivity_analysis['uplift_curve'].
    ...
    """
    # Retrieve data if not provided
    if data is None:
        if not hasattr(self.idata, "sensitivity_analysis"):
            raise ValueError(
                "No sensitivity analysis results found in 'self.idata' and no 'data' argument provided. "
                "Run 'mmm.sensitivity.run_sweep()' first or provide 'data'."
            )

        sa_group = self.idata.sensitivity_analysis  # type: ignore
        if isinstance(sa_group, xr.Dataset):
            if "uplift_curve" not in sa_group:
                raise ValueError(
                    "Expected 'uplift_curve' in idata.sensitivity_analysis. "
                    "Use SensitivityAnalysis.compute_uplift_curve_respect_to_base(..., extend_idata=True)."
                )
            data = sa_group["uplift_curve"]
        else:
            raise ValueError(
                "sensitivity_analysis does not contain 'uplift_curve'. Did you persist it to idata?"
            )

    # Handle Dataset input
    if isinstance(data, xr.Dataset):
        if "uplift_curve" in data:
            data = data["uplift_curve"]
        elif "x" in data:
            data = data["x"]
        else:
            raise ValueError("Dataset must contain 'uplift_curve' or 'x' variable.")

    # Call helper with data (no more monkey-patching!)
    pc = self._sensitivity_analysis_plot(
        data=data,  # ← PASS DATA DIRECTLY
        hdi_prob=hdi_prob,
        aggregation=aggregation,
        backend=backend,
    )
    pc.map(azp.visuals.labelled_y, text="Uplift (%)")
    return pc
```

#### 7. marginal_curve() - Update to pass data

**Current implementation** (lines 1237-1271):
```python
def marginal_curve(
    self,
    hdi_prob: float = 0.94,
    aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
    backend: str | None = None,
) -> PlotCollection:
    if not hasattr(self.idata, "sensitivity_analysis"):
        raise ValueError(...)

    sa_group = self.idata.sensitivity_analysis
    # Similar monkey-patching as uplift_curve
```

**New implementation**:
```python
def marginal_curve(
    self,
    data: xr.DataArray | xr.Dataset | None = None,  # ← ADD THIS
    hdi_prob: float = 0.94,
    aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
    backend: str | None = None,
) -> PlotCollection:
    """Plot precomputed marginal effects.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset, optional
        Marginal effects data to plot. If Dataset, should contain 'marginal_effects' variable.
        If None, uses self.idata.sensitivity_analysis['marginal_effects'].
    ...
    """
    # Retrieve data if not provided
    if data is None:
        if not hasattr(self.idata, "sensitivity_analysis"):
            raise ValueError(
                "No sensitivity analysis results found in 'self.idata' and no 'data' argument provided. "
                "Run 'mmm.sensitivity.run_sweep()' first or provide 'data'."
            )

        sa_group = self.idata.sensitivity_analysis  # type: ignore
        if isinstance(sa_group, xr.Dataset):
            if "marginal_effects" not in sa_group:
                raise ValueError(
                    "Expected 'marginal_effects' in idata.sensitivity_analysis. "
                    "Use SensitivityAnalysis.compute_marginal_effects(..., extend_idata=True)."
                )
            data = sa_group["marginal_effects"]
        else:
            raise ValueError(
                "sensitivity_analysis does not contain 'marginal_effects'. Did you persist it to idata?"
            )

    # Handle Dataset input
    if isinstance(data, xr.Dataset):
        if "marginal_effects" in data:
            data = data["marginal_effects"]
        elif "x" in data:
            data = data["x"]
        else:
            raise ValueError("Dataset must contain 'marginal_effects' or 'x' variable.")

    # Call helper with data (no more monkey-patching!)
    pc = self._sensitivity_analysis_plot(
        data=data,  # ← PASS DATA DIRECTLY
        hdi_prob=hdi_prob,
        aggregation=aggregation,
        backend=backend,
    )
    pc.map(azp.visuals.labelled_y, text="Marginal Effect")
    return pc
```

### Benefits of This Standardization

1. **Consistency**: All methods follow the same pattern for data handling
2. **Flexibility**: Users can pass external data or use data from self.idata
3. **Testability**: Methods can be tested with mock data without needing full MMM setup
4. **Separation of Concerns**: `_sensitivity_analysis_plot()` is a pure plotting function
5. **No More Monkey-Patching**: The uplift_curve() and marginal_curve() methods no longer need to temporarily swap self.idata
6. **Better Error Messages**: Clear messages when data is missing

### Implementation Priority

This is **CRITICAL** and should be completed as **Priority 0** (along with file renaming) because:
- It's a fundamental API design issue
- It affects multiple methods
- It's easier to fix before the migration is complete
- Tests need to be written against the correct API

## Recommendations

### Priority 0: File Renaming and Data Parameter Standardization (Must Complete First)

0. **Rename files and classes** ✅ 30 minutes
   - Rename `pymc_marketing/mmm/old_plot.py` to `legacy_plot.py`
   - Rename class `OldMMMPlotSuite` to `LegacyMMMPlotSuite` throughout the file
   - Update any imports in existing code/tests
   - **This must be done BEFORE implementing other changes**

0b. **Data Parameter Standardization** ✅ 4 hours
   - Update `contributions_over_time()` to accept `data` parameter with fallback
   - Update `saturation_scatterplot()` to accept `constant_data` and `posterior_data` parameters with fallback
   - Update `saturation_curves()` to accept and pass `constant_data` and `posterior_data` parameters
   - Update `_sensitivity_analysis_plot()` to accept `data` parameter WITHOUT fallback (REQUIRED parameter)
   - Update `sensitivity_analysis()` to accept and pass `data` parameter with fallback
   - Update `uplift_curve()` to accept and pass `data` parameter with fallback (removes monkey-patching)
   - Update `marginal_curve()` to accept and pass `data` parameter with fallback (removes monkey-patching)
   - **This is critical for API consistency and must be done BEFORE writing tests**

### Priority 1: Critical (Must Complete for PR)

1. **Remove deprecated method from new suite** ✅ 15 minutes
   - Delete `saturation_curves_scatter()` from [pymc_marketing/mmm/plot.py:737-771](pymc_marketing/mmm/plot.py#L737-L771)
   - Keep it in LegacyMMMPlotSuite (will be in legacy_plot.py after renaming)
   - Document in migration guide that deprecated methods are not carried forward to v2

2. **Add backward compatibility flag** ✅ 2 hours
   - Modify `config.py` to add `"plot.use_v2": False`
   - Implement version switching in `multidimensional.py:602-607`
   - Import from `legacy_plot` module
   - Add deprecation warning with migration guide link
   - Test manual switching works

3. **Create comprehensive backend testing for new suite** ✅ 6 hours
   - Rename existing test_plot.py to test_legacy_plot.py
   - Update imports in legacy test file to use legacy_plot module
   - CREATE NEW test_plot.py for the new MMMPlotSuite
   - Write ~8 methods × 3 backends = ~24 parametrized tests (note: saturation_curves_scatter removed)
   - Remove experimental test_plot_backends.py file
   - Add backend override and invalid backend tests
   - Verify all new tests pass

4. **Create compatibility test suite** ✅ 3 hours
   - Create `test_plot_compatibility.py`
   - Test version switching (5 tests)
   - Test deprecation warnings (4 tests)
   - Test return types (3 tests)
   - Test missing methods (4 tests)
   - Test parameter compatibility (2 tests)

### Priority 2: Important (Before Merge)

5. **Update documentation** ⏱️ 4 hours
   - Update method docstrings with PlotCollection info
   - Add version directives (.. versionadded::)
   - Document backend parameter
   - Add usage examples

6. **Write migration guide** ⏱️ 6 hours
   - Create `docs/source/guides/mmm_plotting_migration.rst`
   - Document all breaking changes (including parameter type changes)
   - Provide side-by-side examples
   - List missing features and workarounds
   - Explain that parameter changes require code adaptation when switching to v2

### Priority 3: Nice to Have (Can Defer)

8. **Add usage examples to docstrings** ⏱️ 2 hours
   - Add Examples section to all methods
   - Show basic usage, saving, backend switching

9. **Create visual test notebook** ⏱️ 3 hours
   - Notebook comparing old vs new outputs
   - Demonstrates all backends
   - Helps verify visual equivalence

10. **Performance testing** ⏱️ 2 hours
    - Compare old vs new rendering times
    - Test with large datasets
    - Document any performance changes

## Open Questions

### Q1: When should default switch from old to new?

**Options**:
- A. v0.18.0 - Aggressive, breaks existing code immediately
- B. v0.19.0 - Conservative, gives 1 release for users to adapt
- C. v0.20.0 - Very conservative, 2 releases to adapt

**Recommendation**: Option B (v0.19.0)
- v0.18.0: Introduce with legacy default + warning
- v0.19.0: Switch to new default, keep legacy available
- v0.20.0: Remove legacy completely

### Q2: Should LegacyMMMPlotSuite be importable directly?

**Current**: Only via `.plot` property
**Alternative**: Export in `mmm/__init__.py`

**Recommendation**: Keep internal-only
- Encourages proper migration
- Reduces maintenance burden
- Users can still access via `use_v2=False`

### Q3: How to handle `budget_allocation()` removal?

**Options**:
- A. Keep in legacy suite, remove from new (current approach)
- B. Add adapter in new suite that approximates behavior
- C. Port to new suite with PlotCollection return type

**Recommendation**: Option A with stub that raises
- Clear error message guides users
- Avoids maintaining duplicate functionality
- Allows temporary use of legacy suite

### Q4: Should warnings be shown every time or once per session?

**Current Pattern**: Every call
**Alternative**: Once per session using warning filters

**Recommendation**: Every call (current)
- More visible, harder to ignore
- Consistent with other deprecation warnings
- Users can suppress globally if desired

### Q5: What about projects pinned to specific versions?

**Scenario**: User pins to v0.18.0, doesn't update

**Solution**:
- `use_v2=False` default in v0.18.0 ensures no breakage
- Warning provides clear timeline
- Projects can update at their own pace
- No forced migration until they upgrade to v0.20.0+

## Implementation Checklist

### Phase 1: Code Changes
- [ ] **Rename `old_plot.py` to `legacy_plot.py` and `OldMMMPlotSuite` to `LegacyMMMPlotSuite`**
- [ ] **Remove deprecated method from new suite:**
  - [ ] Delete `saturation_curves_scatter()` from pymc_marketing/mmm/plot.py (lines 737-771)
  - [ ] Keep it in LegacyMMMPlotSuite (legacy_plot.py) for backward compatibility
  - [ ] Add note in migration guide about deprecated methods not carried forward to v2
- [ ] **Data Parameter Standardization (CRITICAL - do before tests):**
  - [ ] Update `contributions_over_time()` - add `data` parameter with fallback
  - [ ] Update `saturation_scatterplot()` - add `constant_data` and `posterior_data` parameters with fallback
  - [ ] Update `saturation_curves()` - add and pass `constant_data` and `posterior_data` parameters
  - [ ] Update `_sensitivity_analysis_plot()` - add `data` parameter WITHOUT fallback (REQUIRED)
  - [ ] Update `sensitivity_analysis()` - add and pass `data` parameter with fallback
  - [ ] Update `uplift_curve()` - add and pass `data` parameter with fallback
  - [ ] Update `marginal_curve()` - add and pass `data` parameter with fallback
- [ ] Add `"plot.use_v2": False` to config.py defaults
- [ ] Modify multidimensional.py `.plot` property with version switching
- [ ] Add FutureWarning for legacy suite usage
- [ ] Update all docstrings to document PlotCollection return type and new data parameters

### Phase 2: Testing
- [ ] **Rename `tests/mmm/test_plot.py` to `test_legacy_plot.py` (tests for legacy suite)**
- [ ] **Update imports in renamed test file to use `legacy_plot.LegacyMMMPlotSuite`**
- [ ] **Create NEW `tests/mmm/test_plot.py` for new MMMPlotSuite**
- [ ] **Write ~8 methods × 3 backends = ~24 parametrized tests for new suite** (note: saturation_curves_scatter removed)
- [ ] Remove experimental `tests/mmm/test_plot_backends.py` file
- [ ] Remove deprecated `saturation_curves_scatter()` from pymc_marketing/mmm/plot.py
- [ ] Add backend override and invalid backend tests
- [ ] Create `tests/mmm/test_plot_compatibility.py` (15+ tests)
- [ ] Add mock_mmm fixture
- [ ] Add mock_allocation_samples fixture
- [ ] Verify all ~24 new suite backend tests pass
- [ ] Verify all 15 compatibility tests pass
- [ ] Test warning suppression works
- [ ] Test both suites produce valid output

### Phase 3: Documentation
- [ ] Create migration guide (docs/source/guides/mmm_plotting_migration.rst)
- [ ] Document breaking changes table
- [ ] Provide code examples for migration
- [ ] Update API reference
- [ ] Add versionadded directives
- [ ] Document backend selection
- [ ] List missing features and workarounds

### Phase 4: Review
- [ ] Code review for new implementation
- [ ] Test coverage review (aim for >95%)
- [ ] Documentation review
- [ ] Migration guide validation with sample code
- [ ] Timeline communication (v0.18.0 → v0.20.0)

## Related Research

- [CLAUDE.md](../../CLAUDE.md) - Project development guidelines
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Code style and testing requirements
- [pyproject.toml](../../pyproject.toml) - Test configuration and linting rules

## Appendix: Complete Implementation Template

### A. Config File Modification

**File**: `pymc_marketing/mmm/config.py`

```python
_defaults = {
    "plot.backend": "matplotlib",
    "plot.show_warnings": True,
    "plot.use_v2": False,  # ← ADD THIS LINE
}
```

### B. Property Modification

**File**: `pymc_marketing/mmm/multidimensional.py`

```python
@property
def plot(self) -> MMMPlotSuite | LegacyMMMPlotSuite:
    """Use the MMMPlotSuite to plot the results.

    The plot suite version is controlled by mmm_config["plot.use_v2"]:
    - False (default): Uses legacy matplotlib-based suite (will be deprecated)
    - True: Uses new arviz_plots-based suite with multi-backend support

    .. versionchanged:: 0.18.0
       Added version control via mmm_config["plot.use_v2"].
       The legacy suite will be removed in v0.20.0.

    Examples
    --------
    Use new plot suite:

    >>> from pymc_marketing.mmm import mmm_config
    >>> mmm_config["plot.use_v2"] = True
    >>> pc = mmm.plot.posterior_predictive()
    >>> pc.show()

    Returns
    -------
    MMMPlotSuite or LegacyMMMPlotSuite
        Plot suite instance for creating MMM visualizations.
    """
    from pymc_marketing.mmm.config import mmm_config
    from pymc_marketing.mmm.plot import MMMPlotSuite
    from pymc_marketing.mmm.legacy_plot import LegacyMMMPlotSuite
    import warnings

    self._validate_model_was_built()
    self._validate_idata_exists()

    # Check version flag
    if mmm_config.get("plot.use_v2", False):
        return MMMPlotSuite(idata=self.idata)
    else:
        # Show deprecation warning for legacy suite
        if mmm_config.get("plot.show_warnings", True):
            warnings.warn(
                "The current MMMPlotSuite will be deprecated in v0.20.0. "
                "The new version uses arviz_plots and supports multiple backends (matplotlib, plotly, bokeh). "
                "To use the new version: mmm_config['plot.use_v2'] = True\n"
                "To suppress this warning: mmm_config['plot.show_warnings'] = False\n"
                "See migration guide: https://docs.pymc-marketing.io/en/latest/mmm/plotting_migration.html",
                FutureWarning,
                stacklevel=2,
            )
        return LegacyMMMPlotSuite(idata=self.idata)
```

### C. Missing Method Stub

**File**: `pymc_marketing/mmm/plot.py` (add to MMMPlotSuite class)

```python
def budget_allocation(self, *args, **kwargs):
    """
    Create bar chart comparing allocated spend and channel contributions.

    .. deprecated:: 0.18.0
       This method was removed in MMMPlotSuite v2. The arviz_plots library
       used in v2 doesn't support this specific chart type. See alternatives below.

    Raises
    ------
    NotImplementedError
        This method is not available in MMMPlotSuite v2.

    Notes
    -----
    Alternatives:

    1. **For ROI distributions**: Use :meth:`budget_allocation_roas`
       (different purpose but related to budget allocation)

    2. **To use the old method**: Switch to legacy suite:

       >>> from pymc_marketing.mmm import mmm_config
       >>> mmm_config["plot.use_v2"] = False
       >>> mmm.plot.budget_allocation(samples)

    3. **Custom implementation**: Create bar chart using samples data:

       >>> import matplotlib.pyplot as plt
       >>> channel_contrib = samples["channel_contribution"].mean(...)
       >>> allocated_spend = samples["allocation"]
       >>> # Create custom bar chart with matplotlib

    See Also
    --------
    budget_allocation_roas : Plot ROI distributions by channel

    Examples
    --------
    Use legacy suite temporarily:

    >>> from pymc_marketing.mmm import mmm_config
    >>> original = mmm_config.get("plot.use_v2")
    >>> try:
    ...     mmm_config["plot.use_v2"] = False
    ...     fig, ax = mmm.plot.budget_allocation(samples)
    ...     fig.savefig("budget.png")
    ... finally:
    ...     mmm_config["plot.use_v2"] = original
    """
    raise NotImplementedError(
        "budget_allocation() was removed in MMMPlotSuite v2.\n\n"
        "The new arviz_plots-based implementation doesn't support this chart type.\n\n"
        "Alternatives:\n"
        "  1. For ROI distributions: use budget_allocation_roas()\n"
        "  2. To use old method: set mmm_config['plot.use_v2'] = False\n"
        "  3. Implement custom bar chart using the samples data\n\n"
        "See documentation: https://docs.pymc-marketing.io/en/latest/mmm/plotting_migration.html#budget-allocation"
    )
```

---

**End of Research Document**
