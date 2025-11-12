---
date: 2025-11-12T08:54:53+00:00
researcher: Claude (Sonnet 4.5)
git_commit: eb4daf44d6aaad2fe239ec13a74709f3de40f5d2
branch: work-issue-2083
repository: pymc-labs/pymc-marketing
topic: "MMMPlotSuite Multi-Backend Support Migration (Matplotlib, Plotly, Bokeh)"
tags: [research, codebase, mmm, plotting, backend-support, arviz, plottcollection, issue-2083]
status: complete
last_updated: 2025-11-12
last_updated_by: Claude (Sonnet 4.5)
issue_number: 2083
---

# Research: MMMPlotSuite Multi-Backend Support Migration

**Date**: 2025-11-12T08:54:53+00:00
**Researcher**: Claude (Sonnet 4.5)
**Git Commit**: eb4daf44d6aaad2fe239ec13a74709f3de40f5d2
**Branch**: work-issue-2083
**Repository**: pymc-labs/pymc-marketing
**Issue**: #2083

## Research Question

How to migrate the MMMPlotSuite class in `mmm/plot.py` to support multiple backends (matplotlib, plotly, bokeh) using ArviZ's PlotCollection API while maintaining backward compatibility, handling rc_params deprecation, implementing global backend configuration using ArviZ-style rcParams, and providing comprehensive testing across all backends?

### Requirements
1. Backend Support: Support matplotlib, plotly, and bokeh backends with ability to set backend globally or per-function
2. Backward Compatibility: Changes must be backward compatible - output format controlled by backend argument and a parameter to return PlotCollection vs traditional matplotlib objects
3. Matplotlib-Specific Functions: Identify functions without direct backend equivalents and create backend-specific code
4. rc_params Migration: Move from plt.rc_context(rc_params) to backend="matplotlib", backend_config=None arguments, keeping rc_params for backward compatibility with deprecation warning
5. Global Backend Configuration: Use ArviZ-style rcParams with fallback for global backend settings
6. Performance: Not a concern for this migration
7. Testing: Test all functions across matplotlib, plotly, and bokeh backends
8. Component Plot Methods: Do not migrate component plot methods outside MMMPlotSuite; create new functions if needed

## Summary

This research provides a comprehensive analysis of migrating MMMPlotSuite to support multiple plotting backends (matplotlib, plotly, bokeh) using ArviZ's PlotCollection API. The investigation reveals:

**Key Findings:**
- **Current State**: MMMPlotSuite has 12 public plotting methods, all matplotlib-only
- **Migration Complexity**: 4 methods are easy to migrate, 4 are medium difficulty, and 1 is hard (uses `twinx()`)
- **Critical Blockers**: Two matplotlib-specific features need special handling:
  - `plt.rc_context()` (used in `saturation_curves()`)
  - `ax.twinx()` (used in `budget_allocation()`)
- **ArviZ Architecture**: ArviZ uses a separation-of-concerns pattern with backend-agnostic root functions and backend-specific implementations dynamically loaded via importlib
- **Backward Compatibility**: Multiple established patterns exist in the codebase for parameter deprecation, automatic migration, and warning users

**Recommended Approach:**
1. Adopt ArviZ's architecture pattern with root-level functions and backend-specific implementations in `backends/` subdirectories
2. Use global rcParams with per-function backend parameter override
3. Implement deprecation warnings for `rc_params` parameter migration to `backend_config`
4. Return PlotCollection objects when backend != "matplotlib", else return matplotlib Figure/Axes for backward compatibility
5. Phase migration: Easy methods first (4), medium complexity second (4), hard methods last (1)

## Detailed Findings

### 1. ArviZ PlotCollection API and Multi-Backend Architecture

**Source**: ArviZ documentation and web research

#### PlotCollection Overview

ArviZ's PlotCollection is the central manager for creating visualizations across backends. It provides:

**Core Purpose**: "PlotCollection provides the logic to loop over each plot, assign the correct data, aesthetics, and other properties to each plot, and then render them in a single figure."

**Key Methods**:
- `aes()` - Define aesthetic-to-data mappings
- `aes_set()` - Apply aesthetics to specific visuals
- `map()` - Apply operations across plots
- `get_viz()` - Access backend-specific plot objects
- `allocate_artist()` - Reserve space for graphical elements
- `show()` / `savefig()` - Display or export visualizations
- `grid()` / `wrap()` - Layout management
- `add_legend()` - Legend management

**Example Usage**:
```python
import arviz_plots as azp
from arviz_base import load_arviz_data

schools = load_arviz_data("centered_eight")

# Create plot - returns PlotCollection
pc = azp.plot_dist(schools, var_names=["mu", "tau"], backend="matplotlib")

# Modify specific plot via backend access
pc.get_viz("plot", "mu").set_xlim(-5, 12)  # matplotlib-specific

# Display result
pc.show()
```

#### Backend Selection Architecture

**ArviZ uses three levels of backend configuration**:

1. **Configuration Files** (arvizrc) - Loaded on import
   ```
   plot.backend : bokeh
   ```

2. **Runtime rcParams** - Global settings
   ```python
   import arviz as az
   az.rcParams["plot.backend"] = "bokeh"
   ```

3. **Per-Function Parameter** - Override for specific plot
   ```python
   az.plot_posterior(data, backend="bokeh", backend_kwargs={...})
   ```

**Dynamic Backend Loading Pattern**:
```python
def get_plotting_function(plot_name, plot_module, backend):
    """Return plotting function for correct backend."""
    if backend is None:
        backend = rcParams["plot.backend"]

    # Perform import of plotting method
    module = importlib.import_module(f"arviz.plots.backends.{backend}.{plot_module}")
    plotting_method = getattr(module, plot_name)

    return plotting_method
```

**Directory Structure**:
```
arviz/
├── plots/
│   ├── posteriorplot.py          # Root-level (backend-agnostic)
│   ├── traceplot.py               # Root-level
│   └── backends/
│       ├── matplotlib/
│       │   ├── posteriorplot.py   # Matplotlib implementation
│       │   └── traceplot.py
│       └── bokeh/
│           ├── posteriorplot.py   # Bokeh implementation
│           └── traceplot.py
```

#### Backend-Specific Parameters

ArviZ uses `backend_kwargs` (not `backend_config`) to pass backend-specific configuration:

```python
# Matplotlib backend
az.plot_posterior(
    data,
    backend="matplotlib",
    backend_kwargs={
        "figsize": (10, 6),
        "dpi": 150,
        "tight_layout": True
    }
)

# Bokeh backend
az.plot_posterior(
    data,
    backend="bokeh",
    backend_kwargs={
        "width": 350,
        "background_fill_color": "#d3d0e3",
        "toolbar_location": "above"
    }
)
```

### 2. Current MMMPlotSuite Matplotlib Dependencies

**Source**: Analysis of `pymc_marketing/mmm/plot.py`

#### Method Inventory and Complexity Assessment

| Method | Lines | Return Type | Difficulty | Key Features | Blockers |
|--------|-------|-------------|------------|--------------|----------|
| `_init_subplots()` | 200-233 | `tuple[Figure, NDArray[Axes]]` | Easy | `plt.subplots` | None |
| `_add_median_and_hdi()` | 306-323 | `Axes` | Easy | `ax.plot`, `ax.fill_between` | None |
| `_plot_budget_allocation_bars()` | 1214-1277 | `None` | **Hard** | `ax.bar`, **`ax.twinx()`** | **twinx()** |
| `posterior_predictive()` | 375-463 | `tuple[Figure, NDArray[Axes]]` | Easy | Line plots + HDI | None |
| `contributions_over_time()` | 465-588 | `tuple[Figure, NDArray[Axes]]` | Easy | Line plots + HDI | None |
| `saturation_scatterplot()` | 590-742 | `tuple[Figure, NDArray[Axes]]` | Easy | Scatter plots | None |
| `saturation_curves()` | 744-996 | `tuple[plt.Figure, np.ndarray]` | Medium | **`plt.rc_context()`**, `plot_samples`, `plot_hdi` | rc_context, external deps |
| `budget_allocation()` | 1037-1212 | `tuple[Figure, plt.Axes \| np.ndarray]` | **Hard** | Calls `_plot_budget_allocation_bars` | **twinx()** |
| `allocated_contribution_by_channel_over_time()` | 1279-1481 | `tuple[Figure, plt.Axes \| NDArray[Axes]]` | Medium | xarray.plot(), fill_between | xarray plotting |
| `sensitivity_analysis()` | 1483-1718 | `tuple[Figure, NDArray[Axes]] \| plt.Axes` | Medium | `az.plot_hdi()`, complex grid | ArviZ dependency |
| `uplift_curve()` | 1720-1820 | `tuple[Figure, NDArray[Axes]] \| plt.Axes` | Medium | Delegates to sensitivity_analysis | Same as above |
| `marginal_curve()` | 1822-1923 | `tuple[Figure, NDArray[Axes]] \| plt.Axes` | Medium | Delegates to sensitivity_analysis | Same as above |

#### Critical Matplotlib-Only Features

**1. `plt.rc_context(rc_params)` - Line 879 in `saturation_curves()`**

Current usage:
```python
rc_params = rc_params or {}
with plt.rc_context(rc_params):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **subkw)
```

**Purpose**: Temporary matplotlib configuration for font sizes, tick sizes, etc.

**Migration Strategy**:
- Deprecate `rc_params` parameter with warning
- Add `backend_config` parameter for backend-specific configuration
- For matplotlib backend: Pass to `backend_kwargs` of `plt.subplots()`
- For plotly/bokeh: Apply equivalent styling via backend-specific methods

**2. `ax.twinx()` - Line 1249 in `_plot_budget_allocation_bars()`**

Current usage:
```python
# Create twin axis for contributions
ax2 = ax.twinx()

# Plot on secondary y-axis
bars2 = ax2.bar(
    [i + bar_width for i in index],
    channel_contribution,
    bar_width,
    color="C1",
    alpha=opacity,
    label="Channel Contribution",
)
```

**Purpose**: Dual y-axis for different scales (allocated spend vs. channel contributions)

**Migration Strategies**:

*Plotly Solution*:
```python
from plotly.subplots import make_subplots
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(y=allocated_spend, name="Allocated Spend"), secondary_y=False)
fig.add_trace(go.Bar(y=channel_contribution, name="Channel Contribution"), secondary_y=True)
fig.update_yaxes(title_text="Allocated Spend", secondary_y=False)
fig.update_yaxes(title_text="Channel Contributions", secondary_y=True)
```

*Bokeh Solution*:
```python
from bokeh.models import LinearAxis, Range1d
p = figure()
p.vbar(x=channels, top=allocated_spend, width=0.35, color="C0", legend_label="Allocated Spend")
p.extra_y_ranges = {"contrib": Range1d(start=min(channel_contribution), end=max(channel_contribution))}
p.add_layout(LinearAxis(y_range_name="contrib", axis_label="Channel Contributions"), 'right')
p.vbar(x=channels, top=channel_contribution, width=0.35, color="C1", y_range_name="contrib")
```

### 3. Existing Backend and rc_params Usage in Codebase

**Source**: Codebase analysis

#### Current rc_params Usage

**File**: `pymc_marketing/mmm/plot.py:753`
- **Method**: `saturation_curves()`
- **Parameter**: `rc_params: dict | None = None`
- **Usage**: `with plt.rc_context(rc_params):` (line 879)
- **Purpose**: Allows customization of matplotlib rendering parameters

**Test**: `tests/mmm/test_plot.py:611-616`
```python
def test_saturation_curves_rc_params():
    """Test that saturation_curves accepts rc_params."""
    rc_params = {"font.size": 12, "axes.labelsize": 10}
    fig, axes = mock_suite.saturation_curves(
        curve=mock_curve,
        rc_params=rc_params
    )
    assert isinstance(fig, Figure)
```

#### ArviZ Integration Points

Files using ArviZ plotting functions:
- `pymc_marketing/mmm/plot.py` - Uses `az.hdi()` at line 311, `az.plot_hdi()` at line 1689
- `pymc_marketing/mmm/base.py` - Uses `az.plot_forest()` at line 1163, `az.extract()`, `az.hdi()`
- `pymc_marketing/plot.py` - Core plotting utilities with ArviZ integration

**Current Pattern**: Direct ArviZ function calls without backend specification (uses ArviZ's default backend)

### 4. Testing Patterns for Multi-Backend Visualization

**Source**: Test file analysis

#### Pattern 1: Basic Figure and Axes Type Checking

**Location**: `tests/mmm/test_plot.py:104-110`

```python
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def test_saturation_curves_scatter_original_scale(fit_mmm):
    fig, ax = fit_mmm.plot.saturation_curves_scatter(original_scale=True)
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert all(isinstance(a, Axes) for a in ax.flat)
```

**Adaptation for Multi-Backend**:
```python
@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_saturation_curves_scatter_backends(fit_mmm, backend):
    result = fit_mmm.plot.saturation_curves_scatter(
        original_scale=True,
        backend=backend
    )

    if backend == "matplotlib":
        fig, ax = result
        assert isinstance(fig, Figure)
        assert isinstance(ax, np.ndarray)
    else:
        # PlotCollection for plotly/bokeh
        assert isinstance(result, PlotCollection)
        assert result.viz.backend == backend
```

#### Pattern 2: Parametrized Plot Testing

**Location**: `tests/mmm/test_plotting.py:141-286`

```python
@pytest.mark.parametrize(
    "func_plot_name, kwargs_plot",
    [
        ("plot_prior_predictive", {}),
        ("plot_prior_predictive", {"original_scale": True}),
        ("plot_posterior_predictive", {}),
        ("plot_errors", {}),
        # ... 20+ more variations
    ],
)
def test_plots(plotting_mmm, func_plot_name, kwargs_plot):
    func = plotting_mmm.__getattribute__(func_plot_name)
    assert isinstance(func(**kwargs_plot), plt.Figure)
    plt.close("all")
```

**Multi-Backend Adaptation**:
```python
@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
@pytest.mark.parametrize(
    "func_plot_name, kwargs_plot",
    [
        ("posterior_predictive", {}),
        ("contributions_over_time", {"var": ["intercept"]}),
        # ... other methods
    ],
)
def test_plots_all_backends(mmm_suite, func_plot_name, kwargs_plot, backend):
    func = mmm_suite.__getattribute__(func_plot_name)
    result = func(**kwargs_plot, backend=backend)

    if backend == "matplotlib":
        assert isinstance(result[0], plt.Figure)
    else:
        assert isinstance(result, PlotCollection)

    if backend == "matplotlib":
        plt.close("all")
```

#### Pattern 3: Mock InferenceData Fixtures

**Location**: `tests/mmm/test_plot.py:197-228`

```python
@pytest.fixture(scope="module")
def mock_idata() -> az.InferenceData:
    seed = sum(map(ord, "Fake posterior"))
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")
    return az.InferenceData(
        posterior=xr.Dataset({
            "intercept": xr.DataArray(
                rng.normal(size=(4, 100, 52, 3)),
                dims=("chain", "draw", "date", "country"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(100),
                    "date": dates,
                    "country": ["A", "B", "C"],
                },
            ),
        })
    )
```

**Key Aspects**:
- Deterministic seed for reproducibility
- Proper xarray.Dataset structure
- Module scope for reuse
- Realistic dimensions (chain, draw, date, etc.)

#### Pattern 4: Deprecation Warning Testing

**Location**: `tests/mmm/test_plot.py:719-728`

```python
def test_saturation_curves_scatter_deprecation_warning():
    with pytest.warns(
        DeprecationWarning,
        match=r"saturation_curves_scatter is deprecated"
    ):
        fig, axes = mock_suite.saturation_curves_scatter()

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
```

**For rc_params Migration**:
```python
def test_saturation_curves_rc_params_deprecation():
    with pytest.warns(
        DeprecationWarning,
        match=r"rc_params is deprecated.*Use backend_config"
    ):
        result = mock_suite.saturation_curves(
            curve=mock_curve,
            rc_params={"font.size": 12}
        )

    assert isinstance(result[0], Figure)
```

### 5. Deprecation Patterns and Backward Compatibility

**Source**: Codebase pattern analysis

#### Pattern 1: Parameter Deprecation with Automatic Migration

**Location**: `pymc_marketing/model_builder.py:53-77`

```python
def _handle_deprecate_pred_argument(
    value,
    name: str,
    kwargs: dict,
    none_allowed: bool = False,
):
    """Handle deprecated parameter with automatic migration."""
    name_pred = f"{name}_pred"

    # Check for conflicts
    if name_pred in kwargs and value is not None:
        raise ValueError(f"Both {name} and {name_pred} cannot be provided.")

    # Migrate from deprecated parameter
    if name_pred in kwargs:
        warnings.warn(
            f"{name_pred} is deprecated, use {name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return kwargs.pop(name_pred)

    return value
```

**Application to rc_params Migration**:
```python
def _handle_rc_params_deprecation(
    backend_config: dict | None,
    rc_params: dict | None,
    backend: str,
) -> dict:
    """Migrate rc_params to backend_config with deprecation warning."""
    if rc_params is not None and backend_config is not None:
        raise ValueError(
            "Cannot provide both 'rc_params' and 'backend_config'. "
            "Use 'backend_config' instead."
        )

    if rc_params is not None:
        warnings.warn(
            "'rc_params' is deprecated and will be removed in version 0.3.0. "
            "Use 'backend_config' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return rc_params

    return backend_config or {}
```

#### Pattern 2: Method Deprecation with Warning

**Location**: `pymc_marketing/mmm/plot.py:998-1035`

```python
def saturation_curves_scatter(
    self, original_scale: bool = False, **kwargs
) -> tuple[Figure, NDArray[Axes]]:
    """
    .. deprecated:: 0.1.0
       Will be removed in version 0.2.0. Use :meth:`saturation_scatterplot` instead.
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

**Key Aspects**:
- Docstring with `.. deprecated::` directive
- Clear version information (when deprecated, when removed)
- Redirect to new method
- `stacklevel=2` points warning to user's code

#### Pattern 3: Testing Deprecation Warnings

**Location**: `tests/test_model_builder.py:529-557`

```python
def test_handle_deprecate_pred_argument():
    # Test deprecated argument triggers warning
    kwargs = {"test_pred": "deprecated_value"}
    with pytest.warns(DeprecationWarning, match="test_pred is deprecated"):
        result = _handle_deprecate_pred_argument(None, "test", kwargs)
    assert result == "deprecated_value"
    assert "test_pred" not in kwargs  # Should be removed

    # Test both arguments provided raises error
    kwargs = {"test_pred": "deprecated_value"}
    with pytest.raises(
        ValueError,
        match=r"Both test and test_pred cannot be provided"
    ):
        _handle_deprecate_pred_argument("test_value", "test", kwargs)
```

## Architecture Recommendations

### Recommended Directory Structure

```
pymc_marketing/
├── mmm/
│   ├── plot.py                    # Root-level (backend-agnostic)
│   └── backends/
│       ├── __init__.py
│       ├── matplotlib/
│       │   ├── __init__.py
│       │   ├── plot.py            # Matplotlib implementations
│       │   └── utils.py           # Matplotlib-specific utilities
│       ├── plotly/
│       │   ├── __init__.py
│       │   ├── plot.py            # Plotly implementations
│       │   └── utils.py           # Plotly-specific utilities
│       └── bokeh/
│           ├── __init__.py
│           ├── plot.py            # Bokeh implementations
│           └── utils.py           # Bokeh-specific utilities
```

### Root-Level Function Pattern

**File**: `pymc_marketing/mmm/plot.py`

```python
class MMMPlotSuite:
    def __init__(
        self,
        idata: xr.Dataset | az.InferenceData,
        backend: str | None = None,
    ):
        self.idata = idata
        self._backend = backend

    def posterior_predictive(
        self,
        var: list[str] | None = None,
        idata: xr.Dataset | None = None,
        hdi_prob: float = 0.85,
        backend: str | None = None,
        backend_config: dict | None = None,
        return_type: str = "auto",  # "auto", "matplotlib", "collection"
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Plot posterior predictive time series.

        Parameters
        ----------
        backend : str, optional
            Backend to use for plotting. One of "matplotlib", "plotly", "bokeh".
            If None, uses the instance backend or default from rcParams.
        backend_config : dict, optional
            Backend-specific configuration parameters.
        return_type : str, default "auto"
            Return type control:
            - "auto": Returns matplotlib Figure/Axes if backend="matplotlib",
                     PlotCollection otherwise
            - "matplotlib": Always return Figure/Axes (only valid for matplotlib backend)
            - "collection": Always return PlotCollection
        """
        # Determine backend
        backend = self._resolve_backend(backend)

        # Get backend-specific implementation
        plot_func = self._get_plotting_function("posterior_predictive", backend)

        # Prepare kwargs
        plot_kwargs = {
            "var": var,
            "idata": idata,
            "hdi_prob": hdi_prob,
            "backend_config": backend_config,
        }

        # Call backend implementation
        result = plot_func(self.idata, **plot_kwargs)

        # Handle return type
        if return_type == "auto":
            if backend == "matplotlib":
                return result  # Figure, Axes
            else:
                return result  # PlotCollection
        elif return_type == "matplotlib":
            if backend != "matplotlib":
                raise ValueError(
                    "return_type='matplotlib' only valid with backend='matplotlib'"
                )
            return result
        elif return_type == "collection":
            if isinstance(result, tuple):
                # Wrap matplotlib result in PlotCollection
                return self._matplotlib_to_collection(result)
            return result
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

    def _resolve_backend(self, backend: str | None) -> str:
        """Resolve backend from parameter, instance, or rcParams."""
        if backend is not None:
            return backend.lower()
        if self._backend is not None:
            return self._backend.lower()
        # TODO: Implement rcParams fallback
        return "matplotlib"  # default

    def _get_plotting_function(self, plot_name: str, backend: str):
        """Dynamically import backend-specific implementation."""
        import importlib

        backend_map = {
            "matplotlib": "matplotlib",
            "mpl": "matplotlib",
            "plotly": "plotly",
            "bokeh": "bokeh",
        }

        backend = backend_map.get(backend, backend)

        if backend not in ["matplotlib", "plotly", "bokeh"]:
            raise ValueError(
                f"Backend '{backend}' not supported. "
                f"Choose from: {list(backend_map.values())}"
            )

        try:
            module = importlib.import_module(
                f"pymc_marketing.mmm.backends.{backend}.plot"
            )
            plot_func = getattr(module, plot_name)
        except (ImportError, AttributeError) as e:
            raise NotImplementedError(
                f"Backend '{backend}' does not implement '{plot_name}'"
            ) from e

        return plot_func
```

### Backend Implementation Pattern

**File**: `pymc_marketing/mmm/backends/matplotlib/plot.py`

```python
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

def posterior_predictive(
    idata,
    var: list[str] | None = None,
    hdi_prob: float = 0.85,
    backend_config: dict | None = None,
) -> tuple[Figure, NDArray[Axes]]:
    """Matplotlib implementation of posterior_predictive plot."""
    backend_config = backend_config or {}

    # Extract data
    pp_data = idata.posterior_predictive if hasattr(idata, "posterior_predictive") else idata
    if var is None:
        var = ["y"]

    # Create figure with backend_config
    figsize = backend_config.get("figsize", (10, 8))
    fig, axes = plt.subplots(
        nrows=len(var),
        ncols=1,
        figsize=figsize,
        **{k: v for k, v in backend_config.items() if k != "figsize"}
    )

    if len(var) == 1:
        axes = np.array([[axes]])

    # Plot each variable
    for idx, v in enumerate(var):
        ax = axes[idx][0]
        data = pp_data[v]

        # Compute statistics
        median = data.median(dim=["chain", "draw"])
        hdi = az.hdi(data, hdi_prob=hdi_prob)

        # Plot
        dates = data.coords["date"].values
        ax.plot(dates, median, label=v, alpha=0.9)
        ax.fill_between(dates, hdi[v][..., 0], hdi[v][..., 1], alpha=0.2)

        ax.set_title(f"Posterior Predictive: {v}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()

    fig.tight_layout()
    return fig, axes
```

**File**: `pymc_marketing/mmm/backends/plotly/plot.py`

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import arviz_plots as azp

def posterior_predictive(
    idata,
    var: list[str] | None = None,
    hdi_prob: float = 0.85,
    backend_config: dict | None = None,
):
    """Plotly implementation via arviz-plots PlotCollection."""
    backend_config = backend_config or {}

    # Use arviz-plots for plotly backend
    pc = azp.plot_dist(
        idata,
        var_names=var or ["y"],
        backend="plotly",
        hdi_prob=hdi_prob,
        plot_kwargs=backend_config,
    )

    return pc
```

### rc_params Migration Pattern

```python
def saturation_curves(
    self,
    curve: xr.DataArray,
    original_scale: bool = False,
    n_samples: int = 10,
    hdi_probs: float | list[float] | None = None,
    random_seed: np.random.Generator | None = None,
    colors: Iterable[str] | None = None,
    subplot_kwargs: dict | None = None,
    rc_params: dict | None = None,  # DEPRECATED
    backend: str | None = None,
    backend_config: dict | None = None,
    dims: dict[str, str | int | list] | None = None,
    **plot_kwargs,
):
    """Saturation curves with backend support.

    Parameters
    ----------
    rc_params : dict, optional
        .. deprecated:: 0.2.0
            Use `backend_config` instead. Will be removed in 0.3.0.
        Matplotlib-specific configuration (deprecated).
    backend_config : dict, optional
        Backend-specific configuration parameters. For matplotlib,
        passed to plt.subplots(). For plotly/bokeh, passed to
        respective figure creation functions.
    """
    # Handle deprecation
    if rc_params is not None and backend_config is not None:
        raise ValueError(
            "Cannot provide both 'rc_params' and 'backend_config'. "
            "Use 'backend_config' instead."
        )

    if rc_params is not None:
        warnings.warn(
            "'rc_params' is deprecated and will be removed in version 0.3.0. "
            "Use 'backend_config' instead with backend='matplotlib'.",
            DeprecationWarning,
            stacklevel=2,
        )
        backend_config = rc_params
        if backend is None:
            backend = "matplotlib"

    # Resolve backend
    backend = self._resolve_backend(backend)

    # Get backend implementation
    plot_func = self._get_plotting_function("saturation_curves", backend)

    # Call implementation
    return plot_func(
        self.idata,
        curve=curve,
        original_scale=original_scale,
        n_samples=n_samples,
        hdi_probs=hdi_probs,
        random_seed=random_seed,
        colors=colors,
        subplot_kwargs=subplot_kwargs,
        backend_config=backend_config,
        dims=dims,
        **plot_kwargs,
    )
```

### Global Backend Configuration

Create `pymc_marketing/rcparams.py`:

```python
"""Global configuration for pymc-marketing."""
import os
from pathlib import Path

# Default parameters
_default_params = {
    "plot.backend": "matplotlib",
    "plot.max_subplots": 50,
    "plot.point_estimate": "median",
}

class RCParams(dict):
    """Runtime configuration parameters for pymc-marketing."""

    def __init__(self):
        super().__init__(_default_params)
        self._load_config_files()

    def _load_config_files(self):
        """Load configuration from pymc_marketing_rc files."""
        search_paths = [
            Path.cwd() / "pymc_marketing_rc",  # Current directory
            Path.home() / ".config" / "pymc_marketing" / "pymc_marketing_rc",  # User config
        ]

        for config_path in search_paths:
            if config_path.exists():
                self._load_config_file(config_path)
                break

    def _load_config_file(self, path: Path):
        """Parse and load config file."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, value = line.split(":", 1)
                    self[key.strip()] = value.strip()

# Global instance
rcParams = RCParams()

def rc_context(params: dict):
    """Context manager for temporary rcParams changes."""
    import contextlib

    @contextlib.contextmanager
    def _rc_context():
        old_params = {k: rcParams[k] for k in params}
        rcParams.update(params)
        try:
            yield
        finally:
            rcParams.update(old_params)

    return _rc_context()
```

Usage:
```python
from pymc_marketing import rcParams, rc_context

# Global configuration
rcParams["plot.backend"] = "plotly"

# Temporary override
with rc_context({"plot.backend": "bokeh"}):
    mmm.plot.posterior_predictive()  # Uses bokeh
# Outside context, uses plotly
```

## Code References

### Current Implementation
- `pymc_marketing/mmm/plot.py` - MMMPlotSuite class (all plotting methods)
  - Line 879: `plt.rc_context()` usage in `saturation_curves()`
  - Line 1249: `ax.twinx()` usage in `_plot_budget_allocation_bars()`
  - Lines 375-463: `posterior_predictive()` method
  - Lines 465-588: `contributions_over_time()` method
  - Lines 590-742: `saturation_scatterplot()` method
  - Lines 744-996: `saturation_curves()` method
  - Lines 1037-1212: `budget_allocation()` method

### Testing
- `tests/mmm/test_plot.py` - Main plotting tests
  - Lines 104-110: Basic figure/axes type checking
  - Lines 197-228: Mock InferenceData fixture
  - Lines 312-381: Mock InferenceData with constant_data
  - Lines 611-616: rc_params functionality test
  - Lines 719-728: Deprecation warning test
- `tests/mmm/test_plotting.py` - Parametrized plotting tests
  - Lines 141-286: Comprehensive plot testing with 25+ variations

### Deprecation Patterns
- `pymc_marketing/model_builder.py:53-77` - Parameter deprecation utility
- `pymc_marketing/clv/models/basic.py:49-59` - Dictionary key deprecation
- `pymc_marketing/clv/models/basic.py:133-140` - Method parameter deprecation
- `tests/test_model_builder.py:529-557` - Deprecation testing patterns

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. Create backend directory structure
2. Implement global rcParams system
3. Add `_resolve_backend()` and `_get_plotting_function()` utilities
4. Set up testing infrastructure for multi-backend

### Phase 2: Easy Migrations (Week 3-4)
Migrate these 4 methods first:
- `posterior_predictive()` - Line plots with HDI
- `contributions_over_time()` - Similar to posterior_predictive
- `saturation_scatterplot()` - Simple scatter plots
- `_add_median_and_hdi()` - Helper method

**Deliverable**: Working multi-backend support for basic time-series plots

### Phase 3: Medium Migrations (Week 5-7)
Migrate these 4 methods:
- `saturation_curves()` - Handle rc_params deprecation, external plot functions
- `allocated_contribution_by_channel_over_time()` - Manual xarray data extraction
- `sensitivity_analysis()` - Manual HDI computation
- `uplift_curve()` and `marginal_curve()` - Follow sensitivity_analysis

**Deliverable**: Most plotting functions support all backends

### Phase 4: Hard Migrations (Week 8-9)
Migrate the final method:
- `budget_allocation()` and `_plot_budget_allocation_bars()` - Implement dual y-axis for each backend

**Deliverable**: Complete multi-backend support

### Phase 5: Testing & Documentation (Week 10-11)
1. Comprehensive backend testing across all methods
2. Update documentation with backend examples
3. Migration guide for users
4. Performance benchmarking (if needed)

**Deliverable**: Production-ready multi-backend plotting suite

### Phase 6: Deprecation Cleanup (Future Release)
1. Remove deprecated `rc_params` parameter (version 0.3.0)
2. Remove deprecated `saturation_curves_scatter()` method (version 0.2.0)

## Open Questions

1. **PlotCollection vs Native Returns**: Should we always return PlotCollection for consistency, or maintain backward compatibility with matplotlib Figure/Axes returns?
   - **Recommendation**: Use `return_type="auto"` parameter with default behavior of returning Figure/Axes for matplotlib, PlotCollection for others

2. **External Dependencies**: How to handle `pymc_marketing.plot.plot_samples()` and `plot_hdi()` which are used by `saturation_curves()`?
   - **Recommendation**: Check if these can be made backend-agnostic or call ArviZ equivalents

3. **ArviZ Backend Compatibility**: Do we need to ensure ArviZ is using the same backend as MMMPlotSuite?
   - **Recommendation**: Yes, set ArviZ rcParams when resolving backend

4. **Bokeh Backend Priority**: Should we implement bokeh backend given limited ArviZ support compared to plotly?
   - **Recommendation**: Prioritize plotly, make bokeh optional/future work

5. **Return Type Consistency**: How to handle methods that return different types (some return single Axes, some return arrays)?
   - **Recommendation**: PlotCollection provides consistent interface; for matplotlib backend, maintain current return types

## Related Research

This research is the first comprehensive investigation of MMMPlotSuite backend migration. No prior research documents exist in the repository on this topic.

## Conclusion

Migrating MMMPlotSuite to support multiple backends is feasible using ArviZ's architectural patterns. The key success factors are:

1. **Phased Approach**: Start with easy methods, build infrastructure, tackle hard methods last
2. **Backward Compatibility**: Use deprecation warnings, maintain matplotlib returns by default
3. **Clear Architecture**: Separate backend-agnostic logic from backend-specific implementations
4. **Comprehensive Testing**: Test all methods across all backends with parametrized tests
5. **User Migration Path**: Provide clear deprecation warnings and migration examples

The most challenging aspect is the `budget_allocation()` method due to `twinx()`, but this can be addressed with backend-specific implementations. The recommended timeline is 11 weeks for complete migration with testing and documentation.
