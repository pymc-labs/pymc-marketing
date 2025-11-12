# MMMPlotSuite Backend Migration - TDD Implementation Plan

## Overview

This plan implements backend-agnostic plotting for the MMMPlotSuite class using ArviZ's PlotCollection API, enabling support for matplotlib, plotly, and bokeh backends while maintaining full backward compatibility. We follow Test-Driven Development: write comprehensive tests first, verify they fail properly, then implement features by making those tests pass.

## Current State Analysis

### Existing Implementation
- **Location**: [pymc_marketing/mmm/plot.py:187-1924](pymc_marketing/mmm/plot.py#L187)
- **Class**: `MMMPlotSuite` with 10 public plotting methods
- **Current approach**: All methods directly use matplotlib APIs and return `(Figure, NDArray[Axes])`
- **Dependencies**: matplotlib, arviz (for HDI computation only)

### Current Testing Landscape
- **Test framework**: pytest with parametrized tests
- **Test file**: [tests/mmm/test_plot.py](tests/mmm/test_plot.py) - 1053 lines, comprehensive fixture-based testing
- **Mock data patterns**: xarray-based InferenceData fixtures with realistic structure
- **Test conventions**:
  - Module-scoped fixtures for expensive setup
  - Type assertions only (no visual output validation)
  - `plt.close()` after each test
  - Parametrized tests for multiple configurations

### Key Discoveries
1. **No PlotCollection usage**: ArviZ PlotCollection is not used anywhere in production code
2. **Testing patterns exist**: Parametrized tests, deprecation warnings, backward compatibility tests all have examples
3. **Mock data is realistic**: Fixtures create proper InferenceData structure with posterior, constant_data groups
4. **Helper functions available**: `_init_subplots()`, `_add_median_and_hdi()` need backend abstraction

## Desired End State

After implementation, the MMMPlotSuite should:

1. ✅ Support matplotlib, plotly, and bokeh backends via ArviZ PlotCollection
2. ✅ Maintain 100% backward compatibility (existing code works unchanged)
3. ✅ Support global backend configuration via `mmm_config["plot.backend"]`
4. ✅ Support per-function backend parameter that overrides global config
5. ✅ Return PlotCollection when `return_as_pc=True`, tuple when `False` (default)
6. ✅ Handle matplotlib-specific features (twinx) with clear fallback warnings
7. ✅ Deprecate `rc_params` in favor of `backend_config` with warnings
8. ✅ Pass comprehensive test suite across all three backends

## What We're NOT Testing/Implementing

- Performance comparisons between backends (explicitly out of scope)
- Component plot methods outside MMMPlotSuite (requirement #9)
- Saving plots to files (not in current test suite)
- Interactive features specific to plotly/bokeh (basic rendering only)
- New plotting methods (only migrating existing 10 methods)

## TDD Approach

### Test Design Philosophy
1. **Depth over breadth**: Thoroughly test first 2-3 methods before moving to others
2. **Verify visual output**: Use PlotCollection's backend-specific output validation, not just type checking
3. **Fail diagnostically**: Tests should fail with clear messages pointing to missing functionality
4. **Test data isolation**: Use module-scoped fixtures, mock InferenceData structures

### Implementation Priority
**Phase 1**: Infrastructure + `posterior_predictive()` (simplest method)
**Phase 2**: `contributions_over_time()` (similar to Phase 1)
**Phase 3**: `saturation_curves()` (rc_params deprecation, external functions)
**Phase 4**: `budget_allocation()` (twinx fallback behavior)

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive, informative tests that define the feature completely. These tests should fail in expected, diagnostic ways. We focus deeply on infrastructure and the simplest method (`posterior_predictive()`) first.

### Test Categories

#### 1. Infrastructure Tests (Global Configuration & Return Types)
**Test File**: `tests/mmm/test_plot_backends.py` (NEW)
**Purpose**: Validate backend configuration system and return type switching

**Test Cases to Write:**

##### Test: `test_mmm_config_exists`
**Purpose**: Verify the global configuration object is accessible
**Test Data**: None needed
**Expected Behavior**: Can import and access `mmm_config` from `pymc_marketing.mmm`

```python
def test_mmm_config_exists():
    """
    Test that the global mmm_config object exists and is accessible.

    This test verifies:
    - mmm_config can be imported from pymc_marketing.mmm
    - It has a "plot.backend" key
    - Default backend is "matplotlib"
    """
    from pymc_marketing.mmm import mmm_config

    assert "plot.backend" in mmm_config, \
        "mmm_config should have 'plot.backend' key"
    assert mmm_config["plot.backend"] == "matplotlib", \
        f"Default backend should be 'matplotlib', got {mmm_config['plot.backend']}"
```

**Expected Failure Mode**:
- Error type: `ImportError` or `AttributeError`
- Expected message: `cannot import name 'mmm_config' from 'pymc_marketing.mmm'`

##### Test: `test_mmm_config_backend_setting`
**Purpose**: Verify global backend can be changed and persists
**Test Data**: None needed
**Expected Behavior**: Setting backend value works and can be read back

```python
def test_mmm_config_backend_setting():
    """
    Test that mmm_config backend can be set and retrieved.

    This test verifies:
    - Backend can be changed from default
    - New value persists
    - Can be reset to default
    """
    from pymc_marketing.mmm import mmm_config

    # Store original
    original = mmm_config["plot.backend"]

    try:
        # Change backend
        mmm_config["plot.backend"] = "plotly"
        assert mmm_config["plot.backend"] == "plotly", \
            "Backend should change to 'plotly'"

        # Reset
        mmm_config.reset()
        assert mmm_config["plot.backend"] == "matplotlib", \
            "reset() should restore default 'matplotlib' backend"
    finally:
        # Cleanup
        mmm_config["plot.backend"] = original
```

**Expected Failure Mode**:
- Error type: `AttributeError` on `mmm_config.reset()`
- Expected message: `'dict' object has no attribute 'reset'` (if mmm_config is plain dict)

##### Test: `test_mmm_config_invalid_backend_warning`
**Purpose**: Verify setting invalid backend emits a warning or raises error
**Test Data**: Invalid backend name "invalid_backend"
**Expected Behavior**: Validation prevents or warns about invalid backend

```python
def test_mmm_config_invalid_backend_warning():
    """
    Test that setting an invalid backend name is handled gracefully.

    This test verifies:
    - Invalid backend names are detected
    - Either raises ValueError or emits UserWarning
    - Helpful error message provided
    """
    from pymc_marketing.mmm import mmm_config
    import warnings

    original = mmm_config["plot.backend"]

    try:
        # Attempt to set invalid backend - should either raise or warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mmm_config["plot.backend"] = "invalid_backend"

            # If no exception, should have warning
            assert len(w) > 0, \
                "Should emit warning for invalid backend"
            assert "invalid" in str(w[0].message).lower(), \
                f"Warning should mention 'invalid', got: {w[0].message}"
    except ValueError as e:
        # Acceptable alternative: raise ValueError
        assert "backend" in str(e).lower(), \
            f"Error should mention 'backend', got: {e}"
    finally:
        mmm_config["plot.backend"] = original
```

**Expected Failure Mode**:
- Error type: `AssertionError`
- Expected message: "Should emit warning for invalid backend" (no validation present)

#### 2. Backend Parameter Tests (posterior_predictive)
**Test File**: `tests/mmm/test_plot_backends.py`
**Purpose**: Validate `backend` parameter is accepted and overrides global config

**Test Cases to Write:**

##### Test: `test_posterior_predictive_accepts_backend_parameter`
**Purpose**: Verify method accepts new `backend` parameter without error
**Test Data**: `mock_suite` fixture with posterior_predictive data
**Expected Behavior**: Method accepts backend="matplotlib" without TypeError

```python
def test_posterior_predictive_accepts_backend_parameter(mock_suite_with_pp):
    """
    Test that posterior_predictive() accepts backend parameter.

    This test verifies:
    - backend parameter is accepted
    - No TypeError is raised
    - Method completes successfully
    """
    # Should not raise TypeError
    result = mock_suite_with_pp.posterior_predictive(backend="matplotlib")

    assert result is not None, \
        "posterior_predictive should return a result"
```

**Expected Failure Mode**:
- Error type: `TypeError`
- Expected message: `posterior_predictive() got an unexpected keyword argument 'backend'`

##### Test: `test_posterior_predictive_accepts_return_as_pc_parameter`
**Purpose**: Verify method accepts new `return_as_pc` parameter without error
**Test Data**: `mock_suite_with_pp` fixture
**Expected Behavior**: Method accepts return_as_pc=False without TypeError

```python
def test_posterior_predictive_accepts_return_as_pc_parameter(mock_suite_with_pp):
    """
    Test that posterior_predictive() accepts return_as_pc parameter.

    This test verifies:
    - return_as_pc parameter is accepted
    - No TypeError is raised
    """
    # Should not raise TypeError
    result = mock_suite_with_pp.posterior_predictive(return_as_pc=False)

    assert result is not None, \
        "posterior_predictive should return a result"
```

**Expected Failure Mode**:
- Error type: `TypeError`
- Expected message: `posterior_predictive() got an unexpected keyword argument 'return_as_pc'`

##### Test: `test_posterior_predictive_backend_overrides_global`
**Purpose**: Verify function parameter overrides global config
**Test Data**: `mock_suite_with_pp` fixture
**Expected Behavior**: backend="plotly" overrides global matplotlib setting

```python
@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_posterior_predictive_backend_overrides_global(mock_suite_with_pp, backend):
    """
    Test that backend parameter overrides global mmm_config setting.

    This test verifies:
    - Global config set to one backend
    - Function called with different backend
    - Function uses parameter, not global config
    """
    from pymc_marketing.mmm import mmm_config

    original = mmm_config["plot.backend"]

    try:
        # Set global to matplotlib
        mmm_config["plot.backend"] = "matplotlib"

        # Call with different backend, request PlotCollection to check
        pc = mock_suite_with_pp.posterior_predictive(
            backend=backend,
            return_as_pc=True
        )

        assert hasattr(pc, 'backend'), \
            "PlotCollection should have backend attribute"
        assert pc.backend == backend, \
            f"PlotCollection backend should be '{backend}', got '{pc.backend}'"
    finally:
        mmm_config["plot.backend"] = original
```

**Expected Failure Mode**:
- Error type: `AttributeError` or `AssertionError`
- Expected message: `'tuple' object has no attribute 'backend'` (returns tuple instead of PlotCollection)

#### 3. Return Type Tests (Backward Compatibility)
**Test File**: `tests/mmm/test_plot_backends.py`
**Purpose**: Verify return types match expectations based on `return_as_pc` parameter

**Test Cases to Write:**

##### Test: `test_posterior_predictive_returns_tuple_by_default`
**Purpose**: Verify backward compatibility - default returns tuple
**Test Data**: `mock_suite_with_pp` fixture
**Expected Behavior**: Returns `(Figure, List[Axes])` tuple by default

```python
def test_posterior_predictive_returns_tuple_by_default(mock_suite_with_pp):
    """
    Test that posterior_predictive() returns tuple by default (backward compat).

    This test verifies:
    - Default behavior (no return_as_pc parameter) returns tuple
    - Tuple has two elements: (figure, axes)
    - axes is a list of matplotlib Axes objects (1D list, not 2D array)
    """
    result = mock_suite_with_pp.posterior_predictive()

    assert isinstance(result, tuple), \
        f"Default return should be tuple, got {type(result)}"
    assert len(result) == 2, \
        f"Tuple should have 2 elements (fig, axes), got {len(result)}"

    fig, axes = result

    # For matplotlib backend (default), should be Figure and list
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    assert isinstance(fig, Figure), \
        f"First element should be Figure, got {type(fig)}"
    assert isinstance(axes, list), \
        f"Second element should be list, got {type(axes)}"
    assert all(isinstance(ax, Axes) for ax in axes), \
        "All list elements should be matplotlib Axes instances"
```

**Expected Failure Mode**:
- Error type: `AssertionError` or `AttributeError`
- Expected message: `Default return should be tuple, got <class 'arviz_plots.PlotCollection'>` (if returning PC)

##### Test: `test_posterior_predictive_returns_plotcollection_when_requested`
**Purpose**: Verify new behavior - returns PlotCollection when return_as_pc=True
**Test Data**: `mock_suite_with_pp` fixture
**Expected Behavior**: Returns PlotCollection object when return_as_pc=True

```python
@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_posterior_predictive_returns_plotcollection_when_requested(
    mock_suite_with_pp, backend
):
    """
    Test that posterior_predictive() returns PlotCollection when return_as_pc=True.

    This test verifies:
    - return_as_pc=True returns PlotCollection object
    - PlotCollection has correct backend attribute
    """
    from arviz_plots import PlotCollection

    result = mock_suite_with_pp.posterior_predictive(
        backend=backend,
        return_as_pc=True
    )

    assert isinstance(result, PlotCollection), \
        f"Should return PlotCollection, got {type(result)}"
    assert hasattr(result, 'backend'), \
        "PlotCollection should have backend attribute"
    assert result.backend == backend, \
        f"Backend should be '{backend}', got '{result.backend}'"
```

**Expected Failure Mode**:
- Error type: `AssertionError`
- Expected message: `Should return PlotCollection, got <class 'tuple'>` (still returns tuple)

##### Test: `test_posterior_predictive_tuple_has_correct_axes_for_matplotlib`
**Purpose**: Verify matplotlib backend returns list of Axes in tuple
**Test Data**: `mock_suite_with_pp` fixture
**Expected Behavior**: Tuple's second element is list of Axes objects

```python
def test_posterior_predictive_tuple_has_correct_axes_for_matplotlib(mock_suite_with_pp):
    """
    Test that matplotlib backend returns proper axes list in tuple.

    This test verifies:
    - When return_as_pc=False and backend="matplotlib"
    - Second tuple element is list of matplotlib Axes
    - All elements in list are Axes instances
    """
    from matplotlib.axes import Axes

    fig, axes = mock_suite_with_pp.posterior_predictive(
        backend="matplotlib",
        return_as_pc=False
    )

    assert isinstance(axes, list), \
        f"Axes should be list for matplotlib, got {type(axes)}"
    assert all(isinstance(ax, Axes) for ax in axes), \
        "All list elements should be matplotlib Axes instances"
```

**Expected Failure Mode**:
- Error type: `AssertionError`
- Expected message: `Axes should be list for matplotlib, got <class 'NoneType'>` (if not extracting axes)

##### Test: `test_posterior_predictive_tuple_has_none_axes_for_nonmatplotlib`
**Purpose**: Verify non-matplotlib backends return None for axes in tuple
**Test Data**: `mock_suite_with_pp` fixture
**Expected Behavior**: Tuple's second element is None for plotly/bokeh

```python
@pytest.mark.parametrize("backend", ["plotly", "bokeh"])
def test_posterior_predictive_tuple_has_none_axes_for_nonmatplotlib(
    mock_suite_with_pp, backend
):
    """
    Test that non-matplotlib backends return None for axes in tuple.

    This test verifies:
    - When return_as_pc=False and backend in ["plotly", "bokeh"]
    - Second tuple element is None (no axes concept)
    - First element is backend-specific figure object
    """
    fig, axes = mock_suite_with_pp.posterior_predictive(
        backend=backend,
        return_as_pc=False
    )

    assert axes is None, \
        f"Axes should be None for {backend} backend, got {type(axes)}"
    assert fig is not None, \
        f"Figure should exist for {backend} backend"
```

**Expected Failure Mode**:
- Error type: `AssertionError`
- Expected message: `Axes should be None for plotly backend, got <class 'list'>` (always matplotlib)

#### 4. Visual Output Validation Tests
**Test File**: `tests/mmm/test_plot_backends.py`
**Purpose**: Verify that plots actually render and contain expected elements

**Test Cases to Write:**

##### Test: `test_posterior_predictive_plotcollection_has_viz_attribute`
**Purpose**: Verify PlotCollection has visualization data we can inspect
**Test Data**: `mock_suite_with_pp` fixture
**Expected Behavior**: PlotCollection has `viz` attribute with figure data

```python
@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_posterior_predictive_plotcollection_has_viz_attribute(
    mock_suite_with_pp, backend
):
    """
    Test that PlotCollection has viz attribute with figure data.

    This test verifies:
    - PlotCollection has viz attribute
    - viz has figure attribute
    - Figure can be extracted
    """
    from arviz_plots import PlotCollection

    pc = mock_suite_with_pp.posterior_predictive(
        backend=backend,
        return_as_pc=True
    )

    assert hasattr(pc, 'viz'), \
        "PlotCollection should have 'viz' attribute"
    assert hasattr(pc.viz, 'figure'), \
        "PlotCollection.viz should have 'figure' attribute"

    # Should be able to extract figure
    fig = pc.viz.figure.data.item()
    assert fig is not None, \
        "Should be able to extract figure from PlotCollection"
```

**Expected Failure Mode**:
- Error type: `AttributeError`
- Expected message: `PlotCollection should have 'viz' attribute` (if PC not properly constructed)

##### Test: `test_posterior_predictive_matplotlib_has_lines`
**Purpose**: Verify matplotlib output contains actual plot elements
**Test Data**: `mock_suite_with_pp` fixture with known variables
**Expected Behavior**: Axes contain Line2D objects (the actual plotted data)

```python
def test_posterior_predictive_matplotlib_has_lines(mock_suite_with_pp):
    """
    Test that matplotlib output contains actual plotted lines.

    This test verifies:
    - Axes contain Line2D objects (plotted data)
    - Number of lines matches expected variables
    - Visual output actually created, not just empty axes
    """
    from matplotlib.lines import Line2D

    fig, axes = mock_suite_with_pp.posterior_predictive(
        backend="matplotlib",
        return_as_pc=False
    )

    # Get first axis (should have plots)
    ax = axes.flat[0]

    # Should have lines (median plots)
    lines = [child for child in ax.get_children() if isinstance(child, Line2D)]
    assert len(lines) > 0, \
        f"Axes should contain Line2D objects (plots), found {len(lines)}"
```

**Expected Failure Mode**:
- Error type: `AssertionError`
- Expected message: `Axes should contain Line2D objects (plots), found 0` (empty plot)

##### Test: `test_posterior_predictive_plotly_has_traces`
**Purpose**: Verify plotly output contains traces (plotly's plot elements)
**Test Data**: `mock_suite_with_pp` fixture
**Expected Behavior**: Plotly figure has traces in data attribute

```python
def test_posterior_predictive_plotly_has_traces(mock_suite_with_pp):
    """
    Test that plotly output contains actual traces.

    This test verifies:
    - Plotly figure has 'data' attribute with traces
    - Number of traces > 0 (something was plotted)
    - Visual output actually created
    """
    fig, _ = mock_suite_with_pp.posterior_predictive(
        backend="plotly",
        return_as_pc=False
    )

    # Plotly figures have .data attribute with traces
    assert hasattr(fig, 'data'), \
        "Plotly figure should have 'data' attribute"
    assert len(fig.data) > 0, \
        f"Plotly figure should have traces, found {len(fig.data)}"
```

**Expected Failure Mode**:
- Error type: `AttributeError` or `AssertionError`
- Expected message: `Plotly figure should have 'data' attribute` (matplotlib Figure returned instead)

##### Test: `test_posterior_predictive_bokeh_has_renderers`
**Purpose**: Verify bokeh output contains renderers (bokeh's plot elements)
**Test Data**: `mock_suite_with_pp` fixture
**Expected Behavior**: Bokeh figure has renderers (glyphs)

```python
def test_posterior_predictive_bokeh_has_renderers(mock_suite_with_pp):
    """
    Test that bokeh output contains actual renderers (plot elements).

    This test verifies:
    - Bokeh figure has renderers
    - Number of renderers > 0 (something was plotted)
    - Visual output actually created
    """
    fig, _ = mock_suite_with_pp.posterior_predictive(
        backend="bokeh",
        return_as_pc=False
    )

    # Bokeh figures have .renderers attribute
    assert hasattr(fig, 'renderers'), \
        "Bokeh figure should have 'renderers' attribute"
    assert len(fig.renderers) > 0, \
        f"Bokeh figure should have renderers, found {len(fig.renderers)}"
```

**Expected Failure Mode**:
- Error type: `AttributeError` or `AssertionError`
- Expected message: `Bokeh figure should have 'renderers' attribute` (matplotlib Figure returned instead)

#### 5. Fixture Setup
**Test File**: `tests/mmm/test_plot_backends.py`
**Purpose**: Create reusable fixtures for backend testing

```python
"""
Backend-agnostic plotting tests for MMMPlotSuite.

This test file validates the migration to ArviZ PlotCollection API for
multi-backend support (matplotlib, plotly, bokeh).

NOTE: Once this migration is complete and stable, evaluate whether
tests/mmm/test_plot.py can be consolidated into this file to avoid duplication.
"""

import numpy as np
import pytest
import xarray as xr
import arviz as az
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pymc_marketing.mmm.plot import MMMPlotSuite


@pytest.fixture(scope="module")
def mock_idata_for_pp():
    """
    Create mock InferenceData with posterior_predictive for testing.

    Structure mirrors real MMM output with:
    - posterior_predictive group with y variable
    - proper dimensions: chain, draw, date
    - realistic date range
    """
    seed = sum(map(ord, "Backend test posterior_predictive"))
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")

    # Create posterior_predictive data
    posterior_predictive = xr.Dataset({
        "y": xr.DataArray(
            rng.normal(loc=100, scale=10, size=(4, 100, 52)),
            dims=("chain", "draw", "date"),
            coords={
                "chain": np.arange(4),
                "draw": np.arange(100),
                "date": dates,
            },
        )
    })

    # Also create a minimal posterior (required for some internal logic)
    posterior = xr.Dataset({
        "intercept": xr.DataArray(
            rng.normal(size=(4, 100)),
            dims=("chain", "draw"),
            coords={
                "chain": np.arange(4),
                "draw": np.arange(100),
            },
        )
    })

    return az.InferenceData(
        posterior=posterior,
        posterior_predictive=posterior_predictive
    )


@pytest.fixture(scope="module")
def mock_suite_with_pp(mock_idata_for_pp):
    """
    Fixture providing MMMPlotSuite with posterior_predictive data.

    Used for testing posterior_predictive() method across backends.
    """
    return MMMPlotSuite(idata=mock_idata_for_pp)


@pytest.fixture(scope="function")
def reset_mmm_config():
    """
    Fixture to reset mmm_config after each test.

    Ensures test isolation - one test's backend changes don't affect others.
    """
    from pymc_marketing.mmm import mmm_config

    original = mmm_config["plot.backend"]
    yield
    mmm_config["plot.backend"] = original
```

### Implementation Steps

1. **Create test file**: `tests/mmm/test_plot_backends.py`
2. **Add note to existing test file**: Edit `tests/mmm/test_plot.py` line 1 to add:
   ```python
   # NOTE: This file may be consolidated with test_plot_backends.py in the future
   # once the backend migration is complete and stable.
   ```

3. **Implement fixtures** (see Fixture Setup section above)

4. **Implement all test cases** in the order listed:
   - Infrastructure tests (global config)
   - Backend parameter tests
   - Return type tests
   - Visual output validation tests

5. **Run tests to verify failures**: `pytest tests/mmm/test_plot_backends.py -v`

### Success Criteria

#### Automated Verification:
- [x] Test file created: `tests/mmm/test_plot_backends.py`
- [x] All tests discovered: `pytest tests/mmm/test_plot_backends.py --collect-only`
- [x] Tests fail (not pass): `pytest tests/mmm/test_plot_backends.py --tb=short`
- [x] No import/syntax errors: `pytest tests/mmm/test_plot_backends.py --tb=line`
- [x] Linting passes: `make lint`
- [x] Test code follows conventions: Style matches `test_plot.py` patterns

#### Manual Verification:
- [ ] Each test has clear docstring explaining what it validates
- [ ] Test names clearly describe what they test (e.g., `test_X_does_Y`)
- [ ] Assertion messages are diagnostic and helpful
- [ ] Fixtures are well-documented with realistic data
- [ ] Test file header includes note about consolidation

---

## Phase 2: Test Failure Verification

### Overview
Run the tests and verify they fail in the expected, diagnostic ways. This ensures our tests are actually testing something and will catch regressions.

### Verification Steps

1. **Run the test suite**:
   ```bash
   pytest tests/mmm/test_plot_backends.py -v
   ```

2. **Verify all tests are discovered**:
   ```bash
   pytest tests/mmm/test_plot_backends.py --collect-only
   ```
   Expected: All tests listed, no collection errors

3. **Check failure modes**:
   ```bash
   pytest tests/mmm/test_plot_backends.py -v --tb=short
   ```
   Review each failure to ensure it matches expected failure mode

### Expected Failures

**Infrastructure Tests:**
- `test_mmm_config_exists`: `ImportError: cannot import name 'mmm_config'`
- `test_mmm_config_backend_setting`: `ImportError: cannot import name 'mmm_config'`
- `test_mmm_config_invalid_backend_warning`: `ImportError: cannot import name 'mmm_config'`

**Backend Parameter Tests:**
- `test_posterior_predictive_accepts_backend_parameter`: `TypeError: posterior_predictive() got an unexpected keyword argument 'backend'`
- `test_posterior_predictive_accepts_return_as_pc_parameter`: `TypeError: posterior_predictive() got an unexpected keyword argument 'return_as_pc'`
- `test_posterior_predictive_backend_overrides_global`: `ImportError: cannot import name 'mmm_config'` or `TypeError` (backend param)

**Return Type Tests:**
- `test_posterior_predictive_returns_tuple_by_default`: Should PASS (existing behavior works)
- `test_posterior_predictive_returns_plotcollection_when_requested`: `TypeError: unexpected keyword argument 'return_as_pc'`
- `test_posterior_predictive_tuple_has_correct_axes_for_matplotlib`: Should PASS (existing behavior)
- `test_posterior_predictive_tuple_has_none_axes_for_nonmatplotlib`: `TypeError: unexpected keyword argument 'backend'`

**Visual Output Tests:**
- `test_posterior_predictive_plotcollection_has_viz_attribute`: `TypeError: unexpected keyword argument 'return_as_pc'`
- `test_posterior_predictive_matplotlib_has_lines`: Should PASS (existing behavior works)
- `test_posterior_predictive_plotly_has_traces`: `TypeError: unexpected keyword argument 'backend'`
- `test_posterior_predictive_bokeh_has_renderers`: `TypeError: unexpected keyword argument 'backend'`

### Success Criteria

#### Automated Verification:
- [ ] All tests run (no collection errors): `pytest tests/mmm/test_plot_backends.py --collect-only`
- [ ] Expected number of failures: Count matches test cases written
- [ ] No unexpected errors: No `ImportError` on test fixtures, no syntax errors
- [ ] Existing tests still pass: `pytest tests/mmm/test_plot.py -k test_posterior_predictive`

#### Manual Verification:
- [ ] Each test fails with expected error type (TypeError, ImportError, AssertionError as listed)
- [ ] Failure messages clearly indicate what's missing
- [ ] Failure messages would help during implementation (diagnostic)
- [ ] Stack traces point to relevant code locations (test assertions, not fixture setup)
- [ ] No cryptic or misleading error messages

### Adjustment Phase

If tests don't fail properly:

**Problem**: Tests pass unexpectedly
- **Fix**: Review test assertions - they may be too lenient
- **Action**: Add stricter type checks, verify specific attributes

**Problem**: Tests error instead of fail (e.g., ImportError on fixtures)
- **Fix**: Check fixture dependencies, ensure mock data doesn't rely on new code
- **Action**: Simplify fixtures to not use non-existent features

**Problem**: Confusing error messages
- **Fix**: Improve assertion messages with context
- **Action**: Add `assert x, f"Expected Y, got {x}"` style messages

**Problem**: Tests fail in wrong order (dependency issues)
- **Fix**: Ensure test isolation - no shared state between tests
- **Action**: Use `reset_mmm_config` fixture, don't modify shared fixtures

**Checklist for Adjustment:**
- [ ] All infrastructure tests fail with ImportError or AttributeError
- [ ] All backend parameter tests fail with TypeError (unexpected keyword)
- [ ] Return type tests for new behavior fail with TypeError
- [ ] Return type tests for existing behavior PASS
- [ ] Visual output tests fail with TypeError (unexpected keyword)

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement the feature by making tests pass, one at a time. Work like debugging - let test failures guide what needs to be implemented next.

### Implementation Strategy

**Order of Implementation:**
1. Global config infrastructure (`mmm_config`)
2. Add `backend` and `return_as_pc` parameters to `posterior_predictive()`
3. Implement PlotCollection integration
4. Implement figure/axes extraction for tuple return
5. Verify visual output across backends

### Implementation 1: Create Global Configuration

**Target Tests**:
- `test_mmm_config_exists`
- `test_mmm_config_backend_setting`
- `test_mmm_config_invalid_backend_warning`

**Current Failure**: `ImportError: cannot import name 'mmm_config' from 'pymc_marketing.mmm'`

**Changes Required:**

**File**: `pymc_marketing/mmm/config.py` (NEW)
**Purpose**: Global configuration management for MMM plotting

```python
"""Configuration management for MMM plotting."""

VALID_BACKENDS = {"matplotlib", "plotly", "bokeh"}


class MMMConfig(dict):
    """
    Configuration dictionary for MMM plotting settings.

    Provides backend configuration with validation and reset functionality.
    Modeled after ArviZ's rcParams pattern.

    Examples
    --------
    >>> from pymc_marketing.mmm import mmm_config
    >>> mmm_config["plot.backend"] = "plotly"
    >>> mmm_config["plot.backend"]
    'plotly'
    >>> mmm_config.reset()
    >>> mmm_config["plot.backend"]
    'matplotlib'
    """

    _defaults = {
        "plot.backend": "matplotlib",
        "plot.show_warnings": True,
    }

    def __init__(self):
        super().__init__(self._defaults)

    def __setitem__(self, key, value):
        """Set config value with validation for backend."""
        if key == "plot.backend":
            if value not in VALID_BACKENDS:
                import warnings
                warnings.warn(
                    f"Invalid backend '{value}'. Valid backends are: {VALID_BACKENDS}. "
                    f"Setting anyway, but plotting may fail.",
                    UserWarning,
                    stacklevel=2
                )
        super().__setitem__(key, value)

    def reset(self):
        """Reset all configuration to default values."""
        self.clear()
        self.update(self._defaults)


# Global config instance
mmm_config = MMMConfig()
```

**File**: `pymc_marketing/mmm/__init__.py`
**Changes**: Add mmm_config export

```python
# Existing imports...

from pymc_marketing.mmm.config import mmm_config

__all__ = [
    # ... existing exports ...
    "mmm_config",
]
```

**Debugging Approach:**
1. Create `config.py` with MMMConfig class
2. Run: `pytest tests/mmm/test_plot_backends.py::test_mmm_config_exists -v`
3. If fails, check import path and __all__ export
4. Run: `pytest tests/mmm/test_plot_backends.py::test_mmm_config_backend_setting -v`
5. If fails, check reset() implementation
6. Run: `pytest tests/mmm/test_plot_backends.py::test_mmm_config_invalid_backend_warning -v`
7. If fails, verify warning is emitted in __setitem__

**Success Criteria:**

##### Automated Verification:
- [x] Test passes: `pytest tests/mmm/test_plot_backends.py::test_mmm_config_exists -v`
- [x] Test passes: `pytest tests/mmm/test_plot_backends.py::test_mmm_config_backend_setting -v`
- [x] Test passes: `pytest tests/mmm/test_plot_backends.py::test_mmm_config_invalid_backend_warning -v`
- [x] Can import: `python -c "from pymc_marketing.mmm import mmm_config; print(mmm_config['plot.backend'])"`
- [x] Linting passes: `make lint`
- [x] Type checking passes: `mypy pymc_marketing/mmm/config.py` (no new errors)

##### Manual Verification:
- [ ] Code is clean and well-documented
- [ ] Follows project conventions (NumPy docstrings)
- [ ] No performance issues (dict operations are O(1))
- [ ] Warning messages are clear and actionable

### Implementation 2: Add Parameters to posterior_predictive()

**Target Tests**:
- `test_posterior_predictive_accepts_backend_parameter`
- `test_posterior_predictive_accepts_return_as_pc_parameter`

**Current Failure**: `TypeError: posterior_predictive() got an unexpected keyword argument 'backend'`

**Changes Required:**

**File**: `pymc_marketing/mmm/plot.py`
**Method**: `posterior_predictive()` (line 375)
**Changes**: Add backend and return_as_pc parameters

```python
def posterior_predictive(
    self,
    var: list[str] | None = None,
    idata: xr.Dataset | None = None,
    hdi_prob: float = 0.85,
    backend: str | None = None,
    return_as_pc: bool = False,
) -> tuple[Figure, list[Axes] | None] | "PlotCollection":
    """
    Plot posterior predictive distributions over time.

    Parameters
    ----------
    var : list of str, optional
        List of variable names to plot. If None, uses "y".
    idata : xr.Dataset, optional
        Dataset containing posterior predictive samples.
        If None, uses self.idata.posterior_predictive.
    hdi_prob : float, default 0.85
        Probability mass for HDI interval.
    backend : str, optional
        Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
        If None, uses global config via mmm_config["plot.backend"].
        Default (via config) is "matplotlib".
    return_as_pc : bool, default False
        If True, returns PlotCollection object.
        If False, returns tuple (figure, axes) for backward compatibility.

    Returns
    -------
    PlotCollection or tuple
        If return_as_pc=True, returns PlotCollection object.
        If return_as_pc=False, returns (figure, axes) where:
        - figure: backend-specific figure object (matplotlib.figure.Figure,
          plotly.graph_objs.Figure, or bokeh.plotting.Figure)
        - axes: list of matplotlib Axes if backend="matplotlib", else None

    Notes
    -----
    When backend is not "matplotlib" and return_as_pc=False, the axes
    element of the returned tuple will be None, as plotly and bokeh
    do not have an equivalent axes list concept.

    Examples
    --------
    >>> # Backward compatible usage (matplotlib)
    >>> fig, axes = model.plot.posterior_predictive()

    >>> # Multi-backend with PlotCollection
    >>> pc = model.plot.posterior_predictive(backend="plotly", return_as_pc=True)
    >>> pc.show()
    """
    from pymc_marketing.mmm.config import mmm_config

    # Resolve backend (parameter overrides global config)
    backend = backend or mmm_config["plot.backend"]

    # Temporary: Keep existing matplotlib implementation
    # This makes tests pass (accepts parameters) but doesn't use them yet
    # We'll implement PlotCollection integration in next step

    # [Existing implementation continues unchanged for now...]
    # Just pass through to existing code
```

**Debugging Approach:**
1. Add parameters to signature with defaults
2. Add backend resolution logic (import mmm_config, use parameter or config)
3. Run: `pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_accepts_backend_parameter -v`
4. Should PASS (accepts parameter even if not used yet)
5. Run: `pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_accepts_return_as_pc_parameter -v`
6. Should PASS (accepts parameter even if not used yet)
7. Update docstring with new parameters
8. Update type hints in return annotation

**Success Criteria:**

##### Automated Verification:
- [x] Test passes: `pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_accepts_backend_parameter -v`
- [x] Test passes: `pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_accepts_return_as_pc_parameter -v`
- [x] Existing tests still pass: `pytest tests/mmm/test_plot.py::test_posterior_predictive -v`
- [x] Linting passes: `make lint`
- [x] Type checking passes: `mypy pymc_marketing/mmm/plot.py`

##### Manual Verification:
- [x] Docstring updated with new parameters (NumPy style)
- [x] Default values maintain backward compatibility
- [x] Parameter order is logical (existing params first, new params last)
- [x] Type hints are accurate (use string quotes for forward ref to PlotCollection)

### Implementation 3: Integrate PlotCollection and Return Type Logic

**Target Tests**:
- `test_posterior_predictive_returns_tuple_by_default`
- `test_posterior_predictive_returns_plotcollection_when_requested`
- `test_posterior_predictive_backend_overrides_global`
- `test_posterior_predictive_tuple_has_correct_axes_for_matplotlib`
- `test_posterior_predictive_tuple_has_none_axes_for_nonmatplotlib`

**Current Failure**: Tests pass/fail mix - need to integrate PlotCollection

**Changes Required:**

This is the most complex implementation step. We need to:
1. Create PlotCollection-based plotting logic
2. Implement figure/axes extraction for tuple return
3. Handle backend-specific differences

**File**: `pymc_marketing/mmm/plot.py`
**Method**: `posterior_predictive()` (line 375)
**Changes**: Complete PlotCollection integration

```python
def posterior_predictive(
    self,
    var: list[str] | None = None,
    idata: xr.Dataset | None = None,
    hdi_prob: float = 0.85,
    backend: str | None = None,
    return_as_pc: bool = False,
) -> tuple[Figure, list[Axes] | None] | "PlotCollection":
    """[Docstring from previous step]"""
    from pymc_marketing.mmm.config import mmm_config
    from arviz_plots import PlotCollection, visuals

    # Resolve backend (parameter overrides global config)
    backend = backend or mmm_config["plot.backend"]

    # Get data
    var = var or ["y"]
    pp_data = self._get_posterior_predictive_data(idata=idata)

    # Get dimension combinations for subplots
    ignored_dims = {"chain", "draw", "date", "sample"}
    available_dims = [d for d in pp_data[var[0]].dims if d not in ignored_dims]
    additional_dims = [d for d in available_dims if d not in var]
    dim_combinations = self._get_additional_dim_combinations(
        pp_data[var[0]], additional_dims
    )

    n_subplots = len(dim_combinations)

    # Create PlotCollection with grid layout
    # We'll build a dataset for PlotCollection
    plot_data = {}
    for v in var:
        data = pp_data[v]
        # Stack chain and draw into sample dimension
        if "chain" in data.dims and "draw" in data.dims:
            data = data.stack(sample=("chain", "draw"))
        plot_data[v] = data

    plot_dataset = xr.Dataset(plot_data)

    # Create figure with appropriate layout
    # PlotCollection.grid creates a grid of subplots
    pc = PlotCollection.grid(
        plot_dataset,
        backend=backend,
        plots_per_row=1,  # One column layout like original
        figsize=(10, 4 * n_subplots),
    )

    # For each subplot, add line plot and HDI
    for row_idx, combo in enumerate(dim_combinations):
        indexers = dict(zip(additional_dims, combo, strict=False)) if additional_dims else {}

        # Select subplot
        if n_subplots > 1:
            # Multi-panel: select by row index
            pc_subplot = pc.sel(row=row_idx)
        else:
            # Single panel: use full pc
            pc_subplot = pc

        for v in var:
            data = plot_data[v].sel(**indexers) if indexers else plot_data[v]

            # Compute median and HDI
            median = data.median(dim="sample")
            hdi = az.hdi(data, hdi_prob=hdi_prob, input_core_dims=[["sample"]])

            # Add median line
            pc_subplot.map(
                visuals.line,
                data=median.rename("median"),
                color=f"C{var.index(v)}",
                label=v,
            )

            # Add HDI band
            pc_subplot.map(
                visuals.fill_between,
                data1=hdi[v].sel(hdi="lower"),
                data2=hdi[v].sel(hdi="higher"),
                color=f"C{var.index(v)}",
                alpha=0.2,
            )

        # Add labels
        title = self._build_subplot_title(additional_dims, combo, "Posterior Predictive")
        pc_subplot.map(visuals.labelled, title=title, xlabel="Date", ylabel="Posterior Predictive")
        pc_subplot.map(visuals.legend)

    # Return based on return_as_pc flag
    if return_as_pc:
        return pc
    else:
        # Extract figure from PlotCollection
        fig = pc.viz.figure.data.item()

        # Extract axes (only for matplotlib)
        if backend == "matplotlib":
            axes = list(fig.get_axes())  # Return as simple list
        else:
            axes = None

        return fig, axes
```

**Note**: The above is pseudocode showing the structure. Actual implementation will need to:
- Check PlotCollection API documentation for exact method signatures
- Handle dimension combinations correctly
- Ensure HDI computation works with PlotCollection
- Test iteratively with debugger

**Debugging Approach:**
1. Start with simplest case: single variable, no extra dimensions
2. Run: `pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_returns_plotcollection_when_requested[matplotlib] -v`
3. Debug PlotCollection creation - check what data format it expects
4. Debug median/HDI computation - verify dimensions match
5. Debug PlotCollection.map() calls - check visual function signatures
6. Once matplotlib works, test plotly: `pytest ... [plotly]`
7. Debug backend-specific issues (figure extraction, etc.)
8. Test tuple return: `pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_returns_tuple_by_default -v`
9. Debug figure/axes extraction logic

**Alternative Simpler Approach** (if PlotCollection API is challenging):

Keep matplotlib implementation, add a wrapper that converts to/from PlotCollection:

```python
def posterior_predictive(self, ..., backend=None, return_as_pc=False):
    """[docstring]"""
    from pymc_marketing.mmm.config import mmm_config
    backend = backend or mmm_config["plot.backend"]

    # For now, always use matplotlib internally
    # This lets us make progress while learning PlotCollection API
    fig_mpl, axes_mpl = self._posterior_predictive_matplotlib(
        var=var, idata=idata, hdi_prob=hdi_prob
    )

    if backend != "matplotlib":
        # Convert matplotlib to other backend via PlotCollection
        # This is a valid incremental approach
        import warnings
        warnings.warn(
            f"Backend '{backend}' requested but full support not yet implemented. "
            f"Using matplotlib with conversion.",
            UserWarning
        )
        # Conversion logic here...

    if return_as_pc:
        # Wrap matplotlib figure in PlotCollection
        pc = PlotCollection.wrap(fig_mpl, backend=backend)
        return pc
    else:
        if backend == "matplotlib":
            return fig_mpl, axes_mpl
        else:
            # Convert figure to target backend
            fig_converted = convert_figure(fig_mpl, backend)
            return fig_converted, None
```

This incremental approach lets tests pass while we refine the implementation.

**Success Criteria:**

##### Automated Verification:
- [ ] Test passes: `pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_returns_plotcollection_when_requested -v`
- [ ] Test passes: `pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_returns_tuple_by_default -v`
- [ ] Test passes: `pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_backend_overrides_global -v`
- [ ] Test passes: `pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_tuple_has_correct_axes_for_matplotlib -v`
- [ ] Test passes: `pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_tuple_has_none_axes_for_nonmatplotlib -v`
- [ ] All existing tests pass: `pytest tests/mmm/test_plot.py::test_posterior_predictive -v`
- [ ] Linting passes: `make lint`

##### Manual Verification:
- [ ] PlotCollection objects are created correctly
- [ ] Figure extraction works for all backends
- [ ] Axes extraction works for matplotlib, returns None for others
- [ ] Visual output looks reasonable (manually inspect one plot per backend)
- [ ] No performance regressions (test with `time pytest ...`)

### Implementation 4: Visual Output Validation

**Target Tests**:
- `test_posterior_predictive_plotcollection_has_viz_attribute`
- `test_posterior_predictive_matplotlib_has_lines`
- `test_posterior_predictive_plotly_has_traces`
- `test_posterior_predictive_bokeh_has_renderers`

**Current State**: May already pass if Implementation 3 is complete, or may need refinement

**Debugging Approach:**
1. Run: `pytest tests/mmm/test_plot_backends.py -k "visual_output" -v`
2. If `test_plotcollection_has_viz_attribute` fails:
   - Check PlotCollection structure
   - Verify viz.figure.data.item() works
3. If `test_matplotlib_has_lines` fails:
   - Check that median lines are actually plotted
   - Verify Line2D objects exist in axes
4. If `test_plotly_has_traces` fails:
   - Check plotly figure structure
   - Verify conversion from matplotlib worked
   - Check fig.data contains traces
5. If `test_bokeh_has_renderers` fails:
   - Check bokeh figure structure
   - Verify renderers exist

**Possible Issues and Fixes:**
- **Empty plots**: Check that visuals.line() is actually called
- **Wrong backend**: Verify backend parameter is passed through correctly
- **Extraction fails**: Check PlotCollection API version, may need updates

**Success Criteria:**

##### Automated Verification:
- [ ] All visual output tests pass: `pytest tests/mmm/test_plot_backends.py -k "visual" -v`
- [ ] No warnings about empty plots
- [ ] All backends produce non-empty output

##### Manual Verification:
- [ ] Matplotlib plots look correct (run test, inspect saved figure manually)
- [ ] Plotly plots render correctly (check fig.show() if interactive)
- [ ] Bokeh plots render correctly (check bokeh output)
- [ ] HDI bands are visible and correct

### Complete Feature Implementation

Once all tests pass:

**Final Integration Check:**
```bash
# Run all backend tests
pytest tests/mmm/test_plot_backends.py -v

# Run all existing tests to ensure no regressions
pytest tests/mmm/test_plot.py -v

# Run full test suite
pytest tests/mmm/ -v

# Check coverage for new code
pytest tests/mmm/test_plot_backends.py --cov=pymc_marketing.mmm.plot --cov-report=term-missing
```

**Success Criteria:**

##### Automated Verification:
- [ ] All new tests pass: `pytest tests/mmm/test_plot_backends.py -v`
- [ ] No regressions: `pytest tests/mmm/test_plot.py::test_posterior_predictive -v`
- [ ] All MMM tests pass: `pytest tests/mmm/ -v`
- [ ] Code coverage: New code is >90% covered
- [ ] Linting passes: `make lint`
- [ ] Type checking passes: `make typecheck`

##### Manual Verification:
- [ ] Can import and use mmm_config: `from pymc_marketing.mmm import mmm_config`
- [ ] Backward compatible: Old code works unchanged
- [ ] New API works: Can switch backends and get PlotCollection
- [ ] Visual output: Plots look correct in all three backends
- [ ] Documentation: Docstrings are complete and accurate

---

## Phase 4: Refactoring & Cleanup

### Overview
Now that tests are green, refactor to improve code quality while keeping tests passing. Tests protect us during refactoring.

### Refactoring Targets

#### 1. Code Duplication in Test File
**Problem**: Test cases may have repeated setup code
**Solution**: Extract common patterns to helper functions

```python
# tests/mmm/test_plot_backends.py

def assert_valid_plotcollection(pc, expected_backend):
    """
    Helper to validate PlotCollection structure.

    Reduces duplication across tests.
    """
    from arviz_plots import PlotCollection

    assert isinstance(pc, PlotCollection), \
        f"Should return PlotCollection, got {type(pc)}"
    assert hasattr(pc, 'backend'), \
        "PlotCollection should have backend attribute"
    assert pc.backend == expected_backend, \
        f"Backend should be '{expected_backend}', got '{pc.backend}'"


def assert_valid_backend_figure(fig, backend):
    """
    Helper to validate backend-specific figure types.
    """
    if backend == "matplotlib":
        from matplotlib.figure import Figure
        assert isinstance(fig, Figure)
    elif backend == "plotly":
        assert hasattr(fig, 'data'), "Plotly figure should have 'data'"
    elif backend == "bokeh":
        assert hasattr(fig, 'renderers'), "Bokeh figure should have 'renderers'"
```

#### 2. Backend Resolution Logic
**Problem**: Backend resolution logic may be duplicated in every method
**Solution**: Extract to a helper method in MMMPlotSuite

```python
# pymc_marketing/mmm/plot.py

class MMMPlotSuite:
    """[existing docstring]"""

    def _resolve_backend(self, backend: str | None) -> str:
        """
        Resolve backend parameter to actual backend string.

        Parameters
        ----------
        backend : str or None
            Backend parameter from method call.

        Returns
        -------
        str
            Resolved backend name (parameter overrides global config).

        Examples
        --------
        >>> suite._resolve_backend(None)  # uses global config
        'matplotlib'
        >>> suite._resolve_backend("plotly")  # uses parameter
        'plotly'
        """
        from pymc_marketing.mmm.config import mmm_config
        return backend or mmm_config["plot.backend"]
```

#### 3. Figure/Axes Extraction Logic
**Problem**: Tuple return logic may be complex and repeated
**Solution**: Extract to helper method

```python
# pymc_marketing/mmm/plot.py

class MMMPlotSuite:
    """[existing docstring]"""

    def _extract_figure_and_axes(
        self,
        pc: "PlotCollection",
        backend: str
    ) -> tuple:
        """
        Extract figure and axes from PlotCollection for tuple return.

        Parameters
        ----------
        pc : PlotCollection
            PlotCollection object to extract from.
        backend : str
            Backend name ("matplotlib", "plotly", or "bokeh").

        Returns
        -------
        tuple
            (figure, axes) where figure is backend-specific Figure object
            and axes is list of Axes for matplotlib, None for other backends.

        Notes
        -----
        This method enables backward compatibility by extracting matplotlib-style
        return values from PlotCollection objects.
        """
        # Extract figure
        fig = pc.viz.figure.data.item()

        # Extract axes (only for matplotlib)
        if backend == "matplotlib":
            axes = list(fig.get_axes())
        else:
            axes = None

        return fig, axes
```

#### 4. Simplify PlotCollection Creation
**Problem**: PlotCollection creation logic may be verbose
**Solution**: Extract data preparation to helper method

```python
# pymc_marketing/mmm/plot.py

class MMMPlotSuite:
    """[existing docstring]"""

    def _prepare_data_for_plotcollection(
        self,
        data: xr.DataArray,
        stack_dims: tuple[str, ...] = ("chain", "draw")
    ) -> xr.DataArray:
        """
        Prepare xarray data for PlotCollection plotting.

        Parameters
        ----------
        data : xr.DataArray
            Input data with MCMC dimensions.
        stack_dims : tuple of str, default ("chain", "draw")
            Dimensions to stack into 'sample' dimension.

        Returns
        -------
        xr.DataArray
            Data with chain and draw stacked into sample dimension.
        """
        if all(d in data.dims for d in stack_dims):
            data = data.stack(sample=stack_dims)
        return data
```

#### 5. Test Code Quality
**Problem**: Some tests may have complex setup
**Solution**: Use parametrize more effectively

```python
# Example refactoring of tests

# Before: Multiple similar test functions
def test_backend_matplotlib_works():
    pc = suite.posterior_predictive(backend="matplotlib", return_as_pc=True)
    assert pc.backend == "matplotlib"

def test_backend_plotly_works():
    pc = suite.posterior_predictive(backend="plotly", return_as_pc=True)
    assert pc.backend == "plotly"

def test_backend_bokeh_works():
    pc = suite.posterior_predictive(backend="bokeh", return_as_pc=True)
    assert pc.backend == "bokeh"

# After: Single parametrized test
@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_backend_parameter_works(suite, backend):
    """Test that all backends work correctly."""
    pc = suite.posterior_predictive(backend=backend, return_as_pc=True)
    assert pc.backend == backend
```

### Refactoring Steps

1. **Ensure all tests pass before starting**:
   ```bash
   pytest tests/mmm/test_plot_backends.py -v
   ```

2. **For each refactoring**:
   - Make the change (extract helper, rename variable, etc.)
   - Run tests immediately: `pytest tests/mmm/test_plot_backends.py -v`
   - If tests pass, commit the change (or move to next refactoring)
   - If tests fail, revert and reconsider

3. **Focus areas**:
   - Extract helper methods (backend resolution, figure extraction)
   - Improve naming (clear variable names, descriptive method names)
   - Add code comments where logic is complex
   - Simplify conditional logic
   - Remove any dead code or unused imports

4. **Test code refactoring**:
   - Extract test helpers (assertion helpers)
   - Use parametrize more effectively
   - Improve test names for clarity
   - Add docstrings to complex test fixtures

### Success Criteria

#### Automated Verification:
- [ ] All tests still pass: `pytest tests/mmm/test_plot_backends.py -v`
- [ ] No regressions: `pytest tests/mmm/test_plot.py -v`
- [ ] Code coverage maintained: `pytest --cov=pymc_marketing.mmm.plot --cov-report=term-missing`
- [ ] Linting passes: `make lint`
- [ ] Type checking passes: `mypy pymc_marketing/mmm/plot.py`
- [ ] No performance regressions: Compare test run time before/after

#### Manual Verification:
- [ ] Code is more readable after refactoring
- [ ] No unnecessary complexity added
- [ ] Function/variable names are clear and descriptive
- [ ] Comments explain "why" not "what"
- [ ] Helper methods have clear single responsibilities
- [ ] Test code is DRY (Don't Repeat Yourself)
- [ ] Code follows project idioms (check CLAUDE.md patterns)

---

## Phase 5: Expand to contributions_over_time()

### Overview
Apply the same TDD process to the second method, `contributions_over_time()`. This method is similar to `posterior_predictive()`, so the pattern is established.

### Test Design for contributions_over_time()

**New Test Cases** (add to `tests/mmm/test_plot_backends.py`):

```python
@pytest.fixture(scope="module")
def mock_suite_with_contributions(mock_idata):
    """
    Fixture providing MMMPlotSuite with contribution data.

    Reuses mock_idata which already has intercept and linear_trend.
    """
    return MMMPlotSuite(idata=mock_idata)


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_contributions_over_time_backend_parameter(mock_suite_with_contributions, backend):
    """Test contributions_over_time accepts backend parameter and uses it."""
    pc = mock_suite_with_contributions.contributions_over_time(
        var=["intercept"],
        backend=backend,
        return_as_pc=True
    )
    assert_valid_plotcollection(pc, backend)


def test_contributions_over_time_returns_tuple_by_default(mock_suite_with_contributions):
    """Test backward compatibility - returns tuple by default."""
    result = mock_suite_with_contributions.contributions_over_time(
        var=["intercept"]
    )
    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_contributions_over_time_plotcollection(mock_suite_with_contributions, backend):
    """Test return_as_pc=True returns PlotCollection."""
    pc = mock_suite_with_contributions.contributions_over_time(
        var=["intercept"],
        backend=backend,
        return_as_pc=True
    )
    assert_valid_plotcollection(pc, backend)


# Add visual output tests similar to posterior_predictive
```

### Implementation Steps

1. **Write tests first**: Add all test cases to `test_plot_backends.py`
2. **Run tests to verify failures**: `pytest tests/mmm/test_plot_backends.py -k contributions_over_time -v`
3. **Add parameters to method signature**: Same as posterior_predictive
4. **Implement PlotCollection integration**: Reuse patterns from posterior_predictive
5. **Extract figure/axes for tuple return**: Use helper methods from refactoring
6. **Verify all tests pass**: `pytest tests/mmm/test_plot_backends.py -k contributions_over_time -v`

### Success Criteria

- [ ] All contributions_over_time tests pass
- [ ] Existing contributions_over_time tests still pass
- [ ] Code follows same pattern as posterior_predictive
- [ ] Refactored helpers are reused (no duplication)

---

## Testing Strategy Summary

### Test Coverage Goals
- [x] Normal operation paths: All public methods work with default parameters
- [x] Backend switching: All three backends (matplotlib, plotly, bokeh) work
- [x] Return type switching: Both tuple and PlotCollection returns work
- [x] Backward compatibility: Existing code works unchanged
- [x] Configuration: Global and per-function backend configuration
- [x] Edge cases: Invalid backends warn, missing data errors clearly
- [x] Visual output: Plots contain expected elements (lines, traces, renderers)

### Test Organization
- **Test files**:
  - `tests/mmm/test_plot_backends.py` (NEW) - Backend migration tests
  - `tests/mmm/test_plot.py` (EXISTING) - Original tests, marked for future consolidation
- **Fixtures**:
  - Module-scoped for expensive InferenceData creation
  - Function-scoped for config cleanup
  - Located at top of test file
- **Test utilities**:
  - Helper assertions (assert_valid_plotcollection, assert_valid_backend_figure)
  - Located in test file (not separate module yet)
- **Test data**:
  - xarray-based InferenceData fixtures
  - Realistic structure matching MMM output

### Running Tests

```bash
# Run all backend tests
pytest tests/mmm/test_plot_backends.py -v

# Run specific test
pytest tests/mmm/test_plot_backends.py::test_posterior_predictive_returns_plotcollection_when_requested -v

# Run with coverage
pytest tests/mmm/test_plot_backends.py --cov=pymc_marketing.mmm.plot --cov-report=term-missing

# Run with failure details
pytest tests/mmm/test_plot_backends.py -vv --tb=short

# Run only matplotlib backend tests (faster)
pytest tests/mmm/test_plot_backends.py -k "matplotlib" -v

# Run all backends in parallel (if pytest-xdist installed)
pytest tests/mmm/test_plot_backends.py -n auto
```

## Performance Considerations

Performance testing is explicitly out of scope (requirement #2), but we should avoid obvious regressions:

- **Keep existing matplotlib path fast**: Don't add unnecessary overhead for default usage
- **Lazy imports**: Import PlotCollection only when needed
- **Reuse computations**: Don't recompute HDI if already computed
- **Fixture scope**: Use module-scoped fixtures to avoid repeated setup

## Migration Notes

### For Users

**Backward Compatibility**:
- All existing code continues to work without changes
- Default behavior unchanged (matplotlib, tuple return)
- No breaking changes to public API

**New Features**:
```python
# Global backend configuration
from pymc_marketing.mmm import mmm_config
mmm_config["plot.backend"] = "plotly"

# All plots now use plotly
model.plot.posterior_predictive()

# Override for specific plot
model.plot.contributions_over_time(backend="matplotlib")

# Get PlotCollection for advanced customization
pc = model.plot.saturation_curves(curve=curve_data, return_as_pc=True)
pc.map(custom_visual_function)
pc.show()
```

### For Developers

**Adding New Plotting Methods**:
1. Add `backend` and `return_as_pc` parameters
2. Use `self._resolve_backend(backend)` to get backend
3. Create PlotCollection with appropriate backend
4. Use `self._extract_figure_and_axes(pc, backend)` for tuple return
5. Write tests in `test_plot_backends.py` before implementing

**Testing Checklist**:
- [ ] Test accepts backend parameter
- [ ] Test accepts return_as_pc parameter
- [ ] Test returns tuple by default (backward compat)
- [ ] Test returns PlotCollection when requested
- [ ] Test all three backends (parametrize)
- [ ] Test visual output (has lines/traces/renderers)

## Dependencies

### New Dependencies
- **arviz-plots**: Required for PlotCollection API
  - Add to `pyproject.toml`: `arviz-plots>=0.7.0`
  - Add to `environment.yml`: `- arviz-plots>=0.7.0`

### Existing Dependencies (no changes)
- matplotlib: Already required
- arviz: Already required
- xarray: Already required
- numpy: Already required

## References

- Original research: [thoughts/shared/research/2025-11-12-mmmplotsuite-backend-migration-comprehensive.md](thoughts/shared/research/2025-11-12-mmmplotsuite-backend-migration-comprehensive.md)
- MMMPlotSuite implementation: [pymc_marketing/mmm/plot.py:187-1924](pymc_marketing/mmm/plot.py#L187-1924)
- Existing tests: [tests/mmm/test_plot.py](tests/mmm/test_plot.py)
- ArviZ PlotCollection docs: https://arviz-plots.readthedocs.io/
- Test patterns reference: This plan's Phase 1 test examples

## Open Questions

1. **PlotCollection API Learning Curve**: ✅ ADDRESSED - Use incremental approach, start with matplotlib wrapper if needed
2. **Visual Output Validation**: ✅ ADDRESSED - Test for presence of elements (lines/traces/renderers), not pixel-perfect matching
3. **Performance Impact**: ✅ OUT OF SCOPE - User confirmed not a concern for this migration
4. **Deprecation Timeline**: When should we deprecate tuple return in favor of PlotCollection?
   - Recommendation: Keep both indefinitely, default to tuple for backward compat
5. **Test File Consolidation**: When to merge `test_plot.py` into `test_plot_backends.py`?
   - Recommendation: After all methods migrated and stable (next version)

## Next Steps After Phase 5

Once `posterior_predictive()` and `contributions_over_time()` are fully implemented and tested:

1. **Expand to saturation methods**:
   - `saturation_scatterplot()` - Similar pattern, adds scatter plots
   - `saturation_curves()` - Adds `rc_params` deprecation, `backend_config` parameter

2. **Implement twinx fallback**:
   - `budget_allocation()` - Special case with fallback warning

3. **Expand to sensitivity methods**:
   - `sensitivity_analysis()`, `uplift_curve()`, `marginal_curve()` - Wrappers

4. **Full test suite validation**:
   - Run all MMM tests: `pytest tests/mmm/ -v`
   - Check coverage: `pytest tests/mmm/ --cov=pymc_marketing.mmm --cov-report=html`
   - Performance baseline: `pytest tests/mmm/ --durations=20`

5. **Documentation**:
   - Update user guide with backend examples
   - Add migration guide for users
   - Update docstrings with examples
   - Create notebook showing multi-backend usage

## Summary of Key Decisions

1. ✅ **Backward Compatibility**: Maintained via `return_as_pc=False` default
2. ✅ **Global Configuration**: ArviZ-style `mmm_config` dictionary
3. ✅ **Test Organization**: New file `test_plot_backends.py`, mark old file for future consolidation
4. ✅ **Mock Data**: Use existing patterns from `test_plot.py`, realistic xarray structures
5. ✅ **Test Depth**: Prioritize depth (thorough testing of first 2 methods) over breadth
6. ✅ **Visual Validation**: Test for presence of plot elements, not pixel-perfect matching
7. ✅ **Default Backend**: Keep "matplotlib" as default for full backward compatibility
8. ✅ **Helper Extraction**: Refactor common patterns to methods during cleanup phase
9. ✅ **Incremental Implementation**: OK to start with matplotlib-only, add backends incrementally
