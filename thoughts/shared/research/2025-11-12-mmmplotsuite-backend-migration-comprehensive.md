---
date: 2025-11-11T21:13:39-05:00
researcher: Claude
git_commit: e78e3afb259a33f0d2b09d0d6c7e409fe4ddc90d
branch: main
repository: pymc-marketing
topic: "MMMPlotSuite Migration to ArviZ PlotCollection with Backward Compatibility - Comprehensive Research"
tags: [research, codebase, plotting, visualization, arviz, matplotlib, plotly, bokeh, backend-agnostic, mmm, backward-compatibility]
status: complete
last_updated: 2025-11-11
last_updated_by: Claude
---

# Research: MMMPlotSuite Migration to ArviZ PlotCollection with Backward Compatibility

**Date**: 2025-11-11T21:13:39-05:00
**Researcher**: Claude
**Git Commit**: e78e3afb259a33f0d2b09d0d6c7e409fe4ddc90d
**Branch**: main
**Repository**: pymc-marketing

## Research Question

The user wants to rewrite the MMMPlotSuite class in [plot.py](pymc_marketing/mmm/plot.py) to support additional backends beyond matplotlib. Specifically, they want to rewrite the functions in that class to use ArviZ's PlotCollection API instead of matplotlib directly, making the methods return PlotCollection objects instead of matplotlib Figure and Axes objects.

### Updated Requirements (Corrected from previous research)

1. **Global Backend Configuration**: Support the ability to set the backend once, in a "global manner", and then all plots will use that backend. This is on top of the option of setting the backend to individual functions using a backend argument, which will override the global setting.

2. **Backward Compatibility**: The changes to MMMPlotSuite plotting functions should be backward compatible. The output to the function will be based on the backend argument, and there would also be an argument that would control whether to return a PlotCollection object instead.

3. **Backend-Specific Code**: Identify all the matplotlib-specific functions that do not have a direct equivalent in other backends and come up with specific code to handle them for all backends.

4. **RC Params Handling**: For the method `saturation_curves`, it uses `plt.rc_context(rc_params)` so we will need to change it. Use `backend="matplotlib", backend_config=None` as arguments. We are going to keep that `rc_params` parameter for backward compatibility, but emit a warning when using it.

5. **Twin Axes Fallback**: A function that uses matplotlib `twinx` cannot be currently written using arviz-plots. So if a different backend is chosen it needs to emit a warning and fallback to matplotlib.

6. **ArviZ-style rcParams**: Use the recommended "ArviZ-style rcParams with fallback" for Global Backend Configuration Implementation.

7. **Performance**: Performance is not a concern for this migration.

8. **Testing**: Testing of all functions should be across matplotlib, plotly and bokeh backends.

9. **Component Plot Methods**: Do not migrate component plot methods outside MMMPlotSuite. If MMMPlotSuite uses a plotting function that is defined in a different file we would need to create a new function instead.

## Summary

The current MMMPlotSuite implementation is tightly coupled to matplotlib, with all 10 public plotting methods returning `tuple[Figure, NDArray[Axes]]` or similar matplotlib objects. The class uses matplotlib-specific APIs throughout (`plt.subplots`, `ax.plot`, `ax.fill_between`, `ax.twinx`, etc.).

ArviZ's PlotCollection API (from arviz-plots) provides a backend-agnostic alternative that supports matplotlib, bokeh, plotly, and none backends. The codebase already uses ArviZ extensively for HDI computation (`az.hdi()`) and some plotting (`az.plot_hdi()`), but does not use PlotCollection anywhere in production code.

The migration must maintain backward compatibility, support global backend configuration with per-function overrides, handle matplotlib-specific features gracefully (particularly `ax.twinx()` which requires backend-specific implementations or fallback), and be tested across matplotlib, plotly, and bokeh backends.

## Detailed Findings

### Current MMMPlotSuite Architecture

#### Class Overview

**Location**: [pymc_marketing/mmm/plot.py:187-1924](pymc_marketing/mmm/plot.py#L187)

The MMMPlotSuite class is a standalone visualization class for MMM models:
- Initialized with `xr.Dataset` or `az.InferenceData`
- 10 public plotting methods (including 1 deprecated)
- Multiple helper methods for subplot creation and data manipulation
- All methods return matplotlib objects

#### Method Signatures and Return Types

| Method | Current Return Type | Lines | Usage |
|--------|-------------------|-------|-------|
| `posterior_predictive()` | `tuple[Figure, NDArray[Axes]]` | 375-463 | Plot posterior predictive time series |
| `contributions_over_time()` | `tuple[Figure, NDArray[Axes]]` | 465-588 | Plot contribution time series with HDI |
| `saturation_scatterplot()` | `tuple[Figure, NDArray[Axes]]` | 590-742 | Scatter plots of channel saturation |
| `saturation_curves()` | `tuple[plt.Figure, np.ndarray]` | 744-996 | Overlay scatter data with posterior curves |
| `saturation_curves_scatter()` | `tuple[Figure, NDArray[Axes]]` | 998-1035 | **Deprecated** - use `saturation_scatterplot()` |
| `budget_allocation()` | `tuple[Figure, plt.Axes] \| tuple[Figure, np.ndarray]` | 1037-1212 | Bar chart with dual y-axes |
| `allocated_contribution_by_channel_over_time()` | `tuple[Figure, plt.Axes \| NDArray[Axes]]` | 1279-1481 | Line plots with uncertainty bands |
| `sensitivity_analysis()` | `tuple[Figure, NDArray[Axes]] \| plt.Axes` | 1483-1718 | Plot sensitivity sweep results |
| `uplift_curve()` | `tuple[Figure, NDArray[Axes]] \| plt.Axes` | 1720-1820 | Wrapper around sensitivity_analysis for uplift |
| `marginal_curve()` | `tuple[Figure, NDArray[Axes]] \| plt.Axes` | 1822-1923 | Wrapper around sensitivity_analysis for marginal effects |

#### Matplotlib-Specific APIs Used

**Core matplotlib functions used across all methods:**
- `plt.subplots()` - Creating figure and axes grid
- `ax.plot()` - Line plots for medians
- `ax.fill_between()` - HDI/uncertainty bands
- `ax.scatter()` - Scatter plots for data points
- `ax.bar()` - Bar charts (budget allocation)
- `ax.twinx()` - Dual y-axes (budget allocation) - **CRITICAL FEATURE**
- `ax.set_title()`, `ax.set_xlabel()`, `ax.set_ylabel()` - Labeling
- `ax.legend()` - Legends
- `ax.set_visible()` - Hide unused axes
- `fig.tight_layout()` - Layout adjustment
- `fig.suptitle()` - Figure titles
- `plt.rc_context()` - Temporary matplotlib settings

### Matplotlib-Specific Features Analysis

#### Critical Feature: ax.twinx() - Dual Y-Axes

**Location**: [pymc_marketing/mmm/plot.py:1249](pymc_marketing/mmm/plot.py#L1249)

**Method**: `_plot_budget_allocation_bars()` → `budget_allocation()`

**What it does**: Creates a secondary y-axis with independent scale, used to compare allocated spend vs. channel contribution on the same plot with different y-scales.

**Implementation details**:
```python
# Line 1239-1246: Primary bars on primary axis
bars1 = ax.bar(index, allocated_spend, bar_width, color="C0", alpha=opacity, label="Allocated Spend")

# Line 1249: Create twin axis
ax2 = ax.twinx()

# Line 1252-1259: Secondary bars on secondary axis
bars2 = ax2.bar([i + bar_width for i in index], channel_contribution, bar_width,
                color="C1", alpha=opacity, label="Channel Contribution")
```

**Backends without native PlotCollection support**:
- **Bokeh**: No direct twin axes support in PlotCollection
- **Plotly**: Has secondary y-axes but requires different approach

**User Requirement**: "A function that uses matplotlib `twinx` cannot be currently written using arviz. So if a different backend is chosen it needs to emit a warning and fallback to matplotlib."

**Recommended Strategy**: Detect when non-matplotlib backend is requested, emit warning, and force fallback to matplotlib backend.

#### Medium Impact Feature: plt.rc_context()

**Location**: [pymc_marketing/mmm/plot.py:878-880](pymc_marketing/mmm/plot.py#L878-880)

**Method**: `saturation_curves()`

**Code**:
```python
rc_params = rc_params or {}
with plt.rc_context(rc_params):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **subkw)
```

**User Requirement**: "For the method `saturation_curves`, it uses `plt.rc_context(rc_params)` so we will need to change it. I want to use `backend="matplotlib", backend_config=None` as arguments. We are going to keep that `rc_params` parameter for backward compatibility, but emit a warning when using it."

**Recommended Strategy**:
- Add `backend_config` parameter
- Keep `rc_params` parameter with deprecation warning
- Only apply config when backend is matplotlib
- Warn if `backend_config` provided for non-matplotlib backends

### ArviZ Usage in the Codebase

#### Current ArviZ Integration

**ArviZ functions currently used:**
1. `az.hdi()` - HDI computation (used extensively)
2. `az.plot_hdi()` - HDI plotting (used in 2 methods)
3. `az.summary()` - Summary statistics
4. `az.InferenceData` - Primary data container (88+ references)

**Key Finding: No PlotCollection usage**
- Zero instances of `PlotCollection` in production code
- No imports from `arviz_plots` in production code
- All plotting is matplotlib-specific

#### External Plotting Functions Used by MMMPlotSuite

**From [pymc_marketing/plot.py](pymc_marketing/plot.py):**

Imported at line 799 in `saturation_curves()` method:
```python
from pymc_marketing.plot import plot_hdi, plot_samples
```

These functions use matplotlib directly and would need PlotCollection versions created per requirement #9.

### Recommendations

#### 1. Global Backend Configuration Implementation

**Pattern**: ArviZ-style rcParams (per user requirement #6)

**Implementation**:

```python
# pymc_marketing/mmm/config.py (new file)
class MMMConfig(dict):
    """Configuration dictionary for MMM plotting."""

    _defaults = {
        "plot.backend": "matplotlib",
        "plot.show_warnings": True,
    }

    def __init__(self):
        super().__init__(self._defaults)

    def reset(self):
        """Reset to defaults."""
        self.clear()
        self.update(self._defaults)

# Global config instance
mmm_config = MMMConfig()
```

**User API**:
```python
import pymc_marketing as pmm

# Set global backend
pmm.mmm.mmm_config["plot.backend"] = "plotly"

# All subsequent plots use plotly
model.plot.posterior_predictive()

# Override for specific plot
model.plot.saturation_curves(backend="matplotlib")

# Reset to defaults
pmm.mmm.mmm_config.reset()
```

#### 2. Backward Compatibility Strategy

**Method Signature Pattern**:
```python
def posterior_predictive(
    self,
    var: list[str] | None = None,
    idata: xr.Dataset | None = None,
    hdi_prob: float = 0.85,
    backend: str | None = None,
    return_as_pc: bool = False,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """
    Parameters
    ----------
    backend : str, optional
        Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
        If None, uses global config (default: "matplotlib").
    return_as_pc : bool, default False
        If True, returns PlotCollection object. If False, returns a tuple of
        (figure, axes) where figure is the backend-specific figure object and
        axes is an array of axes for matplotlib or None for other backends.

    Returns
    -------
    PlotCollection or tuple
        If return_as_pc=True, returns PlotCollection object.
        If return_as_pc=False, returns (figure, axes) where:
        - figure: backend-specific figure object (plt.Figure, plotly.graph_objs.Figure, etc.)
        - axes: np.ndarray of matplotlib Axes if backend="matplotlib", else None
    """
    # Resolve backend
    backend = backend or mmm_config["plot.backend"]

    # Create PlotCollection
    pc = PlotCollection.grid(data, backend=backend, ...)
    pc.map(plotting_function, ...)

    # Return based on return_as_pc flag
    if return_as_pc:
        return pc
    else:
        # Extract figure from PlotCollection
        fig = pc.viz.figure.data.item()

        # Only matplotlib has axes
        if backend == "matplotlib":
            axes = fig.get_axes()
        else:
            axes = None

        return fig, axes
```

#### 3. Twin Axes Fallback Strategy

**Implementation for `budget_allocation()`**:

```python
def budget_allocation(
    self,
    samples: xr.Dataset,
    backend: str | None = None,
    return_as_pc: bool = False,
    **kwargs
) -> tuple[Figure, plt.Axes] | tuple[Figure, np.ndarray] | PlotCollection:
    """
    Notes
    -----
    This method uses dual y-axes (matplotlib's twinx), which is not supported
    by PlotCollection. If a non-matplotlib backend is requested, a warning
    will be issued and the method will fallback to matplotlib.
    """
    # Resolve backend
    backend = backend or mmm_config["plot.backend"]

    # Check for twinx compatibility (per user requirement #5)
    if backend != "matplotlib":
        import warnings
        warnings.warn(
            f"budget_allocation() uses dual y-axes (ax.twinx()) which is not "
            f"supported by PlotCollection with backend='{backend}'. "
            f"Falling back to matplotlib.",
            UserWarning
        )
        backend = "matplotlib"

    # Proceed with implementation
    # ...
```

#### 4. RC Params Handling for saturation_curves()

**Implementation** (per user requirement #4):

```python
def saturation_curves(
    self,
    curve: xr.DataArray,
    rc_params: dict | None = None,  # DEPRECATED
    backend: str | None = None,
    backend_config: dict | None = None,
    return_as_pc: bool = False,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray] | PlotCollection:
    """
    Parameters
    ----------
    rc_params : dict, optional
        **DEPRECATED**: Use `backend_config` instead.
        Temporary `matplotlib.rcParams` for this plot (matplotlib backend only).
        A DeprecationWarning will be issued when using this parameter.
    backend : str, optional
        Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
        If None, uses global config (default: "matplotlib").
    backend_config : dict, optional
        Backend-specific configuration dictionary:
        - matplotlib: rcParams dict (same as deprecated rc_params)
        - plotly: layout configuration dict
        - bokeh: theme configuration dict
    """
    # Resolve backend
    backend = backend or mmm_config["plot.backend"]

    # Handle deprecated rc_params (per user requirement #4)
    if rc_params is not None:
        import warnings
        warnings.warn(
            "The 'rc_params' parameter is deprecated and will be removed in a "
            "future version. Use 'backend_config' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if backend_config is None:
            backend_config = rc_params

    # Apply backend-specific config if matplotlib
    if backend == "matplotlib" and backend_config:
        with plt.rc_context(backend_config):
            # ... create PlotCollection ...
    else:
        if backend_config and backend != "matplotlib":
            import warnings
            warnings.warn(
                f"backend_config only supported for matplotlib backend, "
                f"ignoring for backend='{backend}'",
                UserWarning
            )
        # ... create PlotCollection without rc_context ...
```

#### 5. Helper Function Migration Strategy

**Problem**: Helper functions in [pymc_marketing/plot.py](pymc_marketing/plot.py) use matplotlib directly.

**Approach**: Create new PlotCollection-compatible versions (per user requirement #9)

```python
# New backend-agnostic versions
def plot_hdi_pc(data, *, backend=None, plot_collection=None, **pc_kwargs):
    """Plot HDI using PlotCollection (backend-agnostic)."""
    backend = backend or mmm_config["plot.backend"]

    if plot_collection is None:
        pc = PlotCollection.grid(data, backend=backend, **pc_kwargs)
    else:
        pc = plot_collection

    pc.map(_plot_hdi_visual, data=data)
    return pc

# Keep existing matplotlib-specific version for backward compatibility
def plot_hdi(da, ax=None, **kwargs):
    """Plot HDI using matplotlib (legacy)."""
    # ... existing matplotlib implementation
```

#### 6. Testing Strategy

**Requirement**: Test across matplotlib, plotly and bokeh backends (user requirement #8).

**Implementation**:
```python
# tests/mmm/test_plotting_backends.py
import pytest

@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
class TestMMMPlotSuiteBackends:
    """Test all MMMPlotSuite methods across backends."""

    def test_posterior_predictive(self, mmm_model, backend):
        pc = mmm_model.plot.posterior_predictive(
            backend=backend,
            return_as_pc=True
        )
        assert isinstance(pc, PlotCollection)
        assert pc.backend == backend

    def test_backward_compatibility_matplotlib(self, mmm_model):
        """Test backward compatibility with matplotlib."""
        fig, axes = mmm_model.plot.posterior_predictive(
            backend="matplotlib",
            return_as_pc=False
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)

    def test_twinx_fallback(self, mmm_model):
        """Test that budget_allocation falls back to matplotlib for non-matplotlib backends."""
        with pytest.warns(UserWarning, match="Falling back to matplotlib"):
            result = mmm_model.plot.budget_allocation(
                samples=...,
                backend="plotly",
                return_as_pc=False
            )
            # Should return matplotlib objects despite requesting plotly
            assert isinstance(result[0], plt.Figure)

    def test_rc_params_deprecation(self, mmm_model):
        """Test that rc_params parameter issues deprecation warning."""
        with pytest.warns(DeprecationWarning, match="rc_params.*deprecated"):
            mmm_model.plot.saturation_curves(
                curve=...,
                rc_params={"xtick.labelsize": 12},
                backend="matplotlib"
            )

    def test_global_backend_config(self, mmm_model):
        """Test global backend configuration."""
        import pymc_marketing as pmm
        original_backend = pmm.mmm.mmm_config["plot.backend"]
        try:
            pmm.mmm.mmm_config["plot.backend"] = "plotly"
            pc = mmm_model.plot.posterior_predictive(return_as_pc=True)
            assert pc.backend == "plotly"
        finally:
            pmm.mmm.mmm_config["plot.backend"] = original_backend
```

## Migration Implementation Checklist

### Phase 1: Infrastructure Setup

- [ ] Add `arviz-plots` as a required dependency in `pyproject.toml`
- [ ] Create `pymc_marketing/mmm/config.py` with `MMMConfig` class and `mmm_config` instance
- [ ] Export `mmm_config` from `pymc_marketing/mmm/__init__.py`
- [ ] Create backend-agnostic plotting function templates

### Phase 2: Helper Functions

- [ ] Create `plot_hdi_pc()` PlotCollection version in `pymc_marketing/plot.py`
- [ ] Create `plot_samples_pc()` PlotCollection version in `pymc_marketing/plot.py`
- [ ] Implement backend detection logic in visual functions
- [ ] Keep existing `plot_hdi()` and `plot_samples()` for backward compatibility

### Phase 3: MMMPlotSuite Methods (Priority Order)

**High Priority (Simple methods)**:
1. [ ] `posterior_predictive()` - Add backend/return_as_pc parameters
2. [ ] `contributions_over_time()` - Add backend/return_as_pc parameters
3. [ ] `saturation_scatterplot()` - Add backend/return_as_pc parameters
4. [ ] `sensitivity_analysis()` - Add backend/return_as_pc parameters
5. [ ] `uplift_curve()` - Inherits from sensitivity_analysis
6. [ ] `marginal_curve()` - Inherits from sensitivity_analysis

**Medium Priority (Uses external functions)**:
7. [ ] `saturation_curves()` - Add backend/backend_config/return_as_pc, deprecate rc_params
8. [ ] `allocated_contribution_by_channel_over_time()` - Add backend/return_as_pc

**Low Priority (Requires twinx fallback)**:
9. [ ] `budget_allocation()` - Add backend/return_as_pc with twinx fallback logic

### Phase 4: Testing

- [ ] Create `tests/mmm/test_plotting_backends.py`
- [ ] Parametrized tests across matplotlib/plotly/bokeh
- [ ] Backward compatibility tests
- [ ] Global config tests
- [ ] Fallback behavior tests (twinx)
- [ ] Deprecation warning tests (rc_params)
- [ ] Return type validation tests

### Phase 5: Documentation

- [ ] Update all docstrings with new parameters
- [ ] Add migration guide for users
- [ ] Add examples showing new API
- [ ] Document backend limitations
- [ ] Update notebooks to show multi-backend usage

## Code References

### MMMPlotSuite Implementation
- Class definition: [pymc_marketing/mmm/plot.py:187](pymc_marketing/mmm/plot.py#L187)
- Twin axes usage: [pymc_marketing/mmm/plot.py:1249](pymc_marketing/mmm/plot.py#L1249)
- RC context usage: [pymc_marketing/mmm/plot.py:878-880](pymc_marketing/mmm/plot.py#L878-880)
- Helper methods: [pymc_marketing/mmm/plot.py:200-370](pymc_marketing/mmm/plot.py#L200-370)
- Main plotting methods: [pymc_marketing/mmm/plot.py:375-1923](pymc_marketing/mmm/plot.py#L375-1923)

### Helper Functions to Migrate
- plot_hdi: [pymc_marketing/plot.py:434](pymc_marketing/plot.py#L434)
- plot_samples: [pymc_marketing/plot.py:503](pymc_marketing/plot.py#L503)

### Dependencies
- Package configuration: `/Users/imrisofer/projects/pymc-marketing/pyproject.toml`
- Conda environment: `/Users/imrisofer/projects/pymc-marketing/environment.yml`

## Open Questions

1. **PlotCollection Figure/Axes Extraction**: ✅ RESOLVED - Use `pc.viz.figure.data.item()` to extract backend-specific figure object, then `fig.get_axes()` for matplotlib axes (returns None for non-matplotlib backends).

2. **Backend-Specific Styling**: Should we implement backend-specific styling translation for common use cases, or just warn users that `backend_config` only works for matplotlib?

3. **Helper Function Strategy**: Should we deprecate old `plot_hdi()` and `plot_samples()` or keep them indefinitely?

4. **Component Plotting**: While we're not migrating component plot methods per requirement #9, should we at least add documentation noting that they remain matplotlib-only?

5. **Version Number**: Should this migration be part of a major version bump (e.g., 0.x → 1.0 or 1.x → 2.0)?

## Summary of Key Decisions

1. **Backward Compatibility**: Maintained via `return_as_pc=False` default parameter
   - When `return_as_pc=False`, functions return `(figure, axes)` tuple
   - `figure` is extracted via `pc.viz.figure.data.item()`
   - `axes` is extracted via `fig.get_axes()` for matplotlib, `None` for other backends
2. **Global Configuration**: ArviZ-style `mmm_config` dictionary
3. **Twin Axes**: Fallback to matplotlib with warning for `budget_allocation()`
4. **RC Params**: Deprecate `rc_params`, add `backend_config` parameter
5. **Testing**: Parametrized tests across matplotlib, plotly, bokeh
6. **Helper Functions**: Create new PlotCollection versions, keep existing for compatibility
7. **Default Backend**: Keep "matplotlib" as default for full backward compatibility
