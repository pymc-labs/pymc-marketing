# DiagnosticsPlots Namespace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `DiagnosticsPlots` namespace class as PR 2 of the MMMPlotSuite v2 rewrite — 4 diagnostic plotting methods on a clean namespace class following the `TransformationPlots` (PR4) template.

**Architecture:** `DiagnosticsPlots` holds only `data: MMMIDataWrapper` and implements 4 methods: `posterior_predictive`, `prior_predictive`, `residuals`, and `residuals_distribution`. The first three use `PlotCollection` with xarray's `.azstats.hdi()`. `residuals_distribution` uses a matplotlib fallback (`az.plot_dist` has no arviz-plots equivalent). All methods follow the standardized API contract from the design doc.

**Tech Stack:** Python ≥3.12, xarray (`.azstats.hdi()`), arviz-plots (`PlotCollection`, `azp.visuals`), arviz (`az.plot_dist`), matplotlib, numpy, pymc-marketing (`MMMIDataWrapper`, `_helpers.py`)

**Spec:** `docs/plans/2026-03-11-mmmplotsuite-v2-attack-plan-design.md` (PR 2 section)

---

## Context: What PR1 and PR4 already delivered

**PR1 (done, on this branch)** added to `pymc_marketing/mmm/plotting/_helpers.py`:
- `_dims_to_sel_kwargs`, `_select_dims`, `_validate_dims`
- `_process_plot_params`, `_extract_matplotlib_result`

**PR4 (done, on this branch)** added `pymc_marketing/mmm/plotting/transformations.py`:
- `TransformationPlots` class — the reference template for every subsequent namespace PR
- Patterns to replicate: idata override resolution, `_process_plot_params`, `_select_dims`, `PlotCollection.grid/wrap`, `pc.map(azp.visuals.*)`, `_extract_matplotlib_result`

Read `pymc_marketing/mmm/plotting/transformations.py` in full before implementing.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `pymc_marketing/mmm/plotting/diagnostics.py` | **Create** | `DiagnosticsPlots` class + 4 module-level helpers |
| `tests/mmm/plotting/test_diagnostics.py` | **Create** | All unit tests |
| `pymc_marketing/mmm/plotting/__init__.py` | **Modify** | Add `DiagnosticsPlots` export |

### `diagnostics.py` internal structure

```
# Module-level private functions
_get_posterior_predictive(data: MMMIDataWrapper) -> xr.Dataset
_get_prior_predictive(data: MMMIDataWrapper) -> xr.Dataset
_compute_residuals(data: MMMIDataWrapper, pp_var: str = "y_original_scale") -> xr.DataArray
_zero_hline(ax: Axes, **kwargs) -> None     # PlotCollection-compatible zero reference line

# Namespace class
class DiagnosticsPlots:
    __init__(self, data: MMMIDataWrapper) -> None
    posterior_predictive(...)
    prior_predictive(...)
    residuals(...)
    residuals_distribution(...)
```

---

## Cross-cutting patterns (apply to every method)

These come directly from the PR4 template and **must** be followed in every public method:

1. **idata override resolution at the top of every method:**
   ```python
   data = (
       MMMIDataWrapper(idata, schema=self._data.schema)
       if idata is not None
       else self._data
   )
   ```
2. **Standard parameter wiring:**
   ```python
   pc_kwargs = _process_plot_params(figsize=figsize, backend=backend, return_as_pc=return_as_pc, **pc_kwargs)
   ```
3. **Return last line:**
   ```python
   return _extract_matplotlib_result(pc, return_as_pc)
   ```
4. **All subsequent data access uses the resolved local `data`, never `self._data` directly.**
5. **No `plt.show()` calls.**
6. **No nested functions** — all helpers are module-level.

---

## PlotCollection time-series strategy

For `posterior_predictive`, `prior_predictive`, and `residuals`, the data has shape
`(chain, draw, date[, geo, ...])`. The strategy:

```python
# 1. Get and filter the DataArray
da = ...  # (chain, draw, date[, extra_dims])
da = _select_dims(da, dims)

# 2. Compute statistics (aggregate over chain/draw)
mean_da = da.mean(dim=("chain", "draw"))          # (date[, extra_dims])
hdi_da  = da.azstats.hdi(hdi_prob)               # (date[, extra_dims], ci_bound)
#   .azstats is an arviz-base xarray accessor registered on import of arviz.
#   ci_bound coordinate has values "lower" and "upper".

# 3. Determine extra (faceting) dims
extra_dims = [d for d in mean_da.dims if d != "date"]

# 4. Build PlotCollection layout dataset from extra dims only (not date)
#    isel(date=0, drop=True) removes the date dim so PlotCollection facets
#    only on the extra dims. For simple models with no extra dims this
#    gives a 0-dim DataArray — PlotCollection.wrap handles this as single panel.
layout_ds = mean_da.isel(date=0, drop=True).to_dataset(name="y")

# 5. Create PlotCollection
pc = PlotCollection.wrap(layout_ds, cols=extra_dims, backend=backend, **pc_kwargs)

# 6. Map visual layers — PlotCollection slices x/y for each panel automatically
dates = da.coords["date"].values
pc.map(azp.visuals.line_xy, x=dates, y=mean_da, **(line_kwargs or {}))
pc.map(
    azp.visuals.fill_between_y,
    x=dates,
    y_bottom=hdi_da.sel(ci_bound="lower"),
    y_top=hdi_da.sel(ci_bound="upper"),
    **{"alpha": 0.2, **(hdi_kwargs or {})},
)
```

**Implementation note:** Verify `PlotCollection.wrap(zero_dim_ds, cols=[])` creates a
single panel during Task 2 testing. If not, use this workaround for the no-extra-dims case:
```python
if not extra_dims:
    layout_ds = mean_da.expand_dims({"_panel": [0]}).isel(date=0, drop=True).to_dataset(name="y")
    pc = PlotCollection.wrap(layout_ds, cols=["_panel"], backend=backend, **pc_kwargs)
else:
    layout_ds = mean_da.isel(date=0, drop=True).to_dataset(name="y")
    pc = PlotCollection.wrap(layout_ds, cols=extra_dims, backend=backend, **pc_kwargs)
```

---

## Task 1: Module-level helpers + class skeleton

**Files:**
- Create: `pymc_marketing/mmm/plotting/diagnostics.py`
- Create: `tests/mmm/plotting/test_diagnostics.py`

- [ ] **Step 1.1: Write failing tests for helpers and class skeleton**

```python
# tests/mmm/plotting/test_diagnostics.py
from __future__ import annotations

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.figure import Figure

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting.diagnostics import (
    DiagnosticsPlots,
    _compute_residuals,
    _get_posterior_predictive,
    _get_prior_predictive,
)

matplotlib.use("Agg")

SEED = sum(map(ord, "DiagnosticsPlots tests"))


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def simple_idata() -> az.InferenceData:
    """InferenceData with (chain, draw, date) dims — no extra dims."""
    rng = np.random.default_rng(SEED)
    n_chain, n_draw, n_date = 2, 50, 20
    dates = np.arange(n_date)
    coords = {"chain": np.arange(n_chain), "draw": np.arange(n_draw), "date": dates}
    base_shape = (n_chain, n_draw, n_date)

    pp = xr.Dataset(
        {
            "y": xr.DataArray(rng.normal(size=base_shape), dims=("chain", "draw", "date"), coords=coords),
            "y_original_scale": xr.DataArray(
                rng.normal(size=base_shape) * 100 + 500,
                dims=("chain", "draw", "date"),
                coords=coords,
            ),
        }
    )
    prior = xr.Dataset(
        {"y": xr.DataArray(rng.normal(size=base_shape), dims=("chain", "draw", "date"), coords=coords)}
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date,)), dims=("date",), coords={"date": dates}
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(posterior_predictive=pp, prior_predictive=prior, constant_data=const)


@pytest.fixture(scope="module")
def panel_idata() -> az.InferenceData:
    """InferenceData with extra 'geo' dim — (chain, draw, date, geo)."""
    rng = np.random.default_rng(SEED + 1)
    n_chain, n_draw, n_date = 2, 30, 15
    dates = np.arange(n_date)
    geos = ["CA", "NY"]
    coords = {
        "chain": np.arange(n_chain),
        "draw": np.arange(n_draw),
        "date": dates,
        "geo": geos,
    }
    base_shape = (n_chain, n_draw, n_date, 2)

    pp = xr.Dataset(
        {
            "y": xr.DataArray(rng.normal(size=base_shape), dims=("chain", "draw", "date", "geo"), coords=coords),
            "y_original_scale": xr.DataArray(
                rng.normal(size=base_shape) * 100 + 500,
                dims=("chain", "draw", "date", "geo"),
                coords=coords,
            ),
        }
    )
    prior = xr.Dataset(
        {
            "y": xr.DataArray(
                rng.normal(size=base_shape), dims=("chain", "draw", "date", "geo"), coords=coords
            )
        }
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date, 2)),
                dims=("date", "geo"),
                coords={"date": dates, "geo": geos},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(posterior_predictive=pp, prior_predictive=prior, constant_data=const)


@pytest.fixture(scope="module")
def simple_data(simple_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(simple_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def panel_data(panel_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(panel_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def simple_plots(simple_data) -> DiagnosticsPlots:
    return DiagnosticsPlots(simple_data)


@pytest.fixture(scope="module")
def panel_plots(panel_data) -> DiagnosticsPlots:
    return DiagnosticsPlots(panel_data)


# ============================================================================
# Helper tests
# ============================================================================


class TestGetPosteriorPredictive:
    def test_returns_dataset_with_y(self, simple_data):
        result = _get_posterior_predictive(simple_data)
        assert isinstance(result, xr.Dataset)
        assert "y" in result

    def test_raises_when_missing(self):
        data = MMMIDataWrapper(az.InferenceData(), validate_on_init=False)
        with pytest.raises(ValueError, match="posterior_predictive"):
            _get_posterior_predictive(data)


class TestGetPriorPredictive:
    def test_returns_dataset_with_y(self, simple_data):
        result = _get_prior_predictive(simple_data)
        assert isinstance(result, xr.Dataset)
        assert "y" in result

    def test_raises_when_missing(self):
        data = MMMIDataWrapper(az.InferenceData(), validate_on_init=False)
        with pytest.raises(ValueError, match="prior_predictive"):
            _get_prior_predictive(data)


class TestComputeResiduals:
    def test_returns_dataarray_named_residuals(self, simple_data):
        result = _compute_residuals(simple_data)
        assert isinstance(result, xr.DataArray)
        assert result.name == "residuals"

    def test_has_chain_draw_date_dims(self, simple_data):
        result = _compute_residuals(simple_data)
        assert {"chain", "draw", "date"}.issubset(result.dims)

    def test_custom_pp_var_fix_iv18(self, simple_data):
        """pp_var parameter allows non-hardcoded variable name (fix IV.18)."""
        result = _compute_residuals(simple_data, pp_var="y_original_scale")
        assert result.name == "residuals"

    def test_raises_on_missing_pp_var(self, simple_data):
        with pytest.raises(ValueError, match="y_nonexistent"):
            _compute_residuals(simple_data, pp_var="y_nonexistent")


class TestDiagnosticsPlotsConstructor:
    def test_stores_data(self, simple_data):
        plots = DiagnosticsPlots(simple_data)
        assert plots._data is simple_data
```

- [ ] **Step 1.2: Run tests — expect ImportError**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py -v --no-header 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'DiagnosticsPlots' from 'pymc_marketing.mmm.plotting.diagnostics'`

- [ ] **Step 1.3: Create `diagnostics.py` with helpers + skeleton**

```python
#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Diagnostics namespace — posterior/prior predictive and residual plots."""

from __future__ import annotations

import itertools
from typing import Any

import arviz as az
import arviz_plots as azp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting._helpers import (
    _extract_matplotlib_result,
    _process_plot_params,
    _select_dims,
)


def _get_posterior_predictive(data: MMMIDataWrapper) -> xr.Dataset:
    """Return the posterior_predictive group from *data*.

    Parameters
    ----------
    data : MMMIDataWrapper
        Wrapper holding the fitted model's InferenceData.

    Returns
    -------
    xr.Dataset
        The posterior_predictive group.

    Raises
    ------
    ValueError
        If posterior_predictive is absent from idata.
    """
    if (
        not hasattr(data.idata, "posterior_predictive")
        or data.idata.posterior_predictive is None
    ):
        raise ValueError(
            "No posterior_predictive data found in idata. "
            "Run MMM.sample_posterior_predictive() first."
        )
    return data.idata.posterior_predictive


def _get_prior_predictive(data: MMMIDataWrapper) -> xr.Dataset:
    """Return the prior_predictive group from *data*.

    Parameters
    ----------
    data : MMMIDataWrapper
        Wrapper holding the fitted model's InferenceData.

    Returns
    -------
    xr.Dataset
        The prior_predictive group.

    Raises
    ------
    ValueError
        If prior_predictive is absent from idata.
    """
    if (
        not hasattr(data.idata, "prior_predictive")
        or data.idata.prior_predictive is None
    ):
        raise ValueError(
            "No prior_predictive data found in idata. "
            "Run MMM.sample_prior_predictive() first."
        )
    return data.idata.prior_predictive


def _compute_residuals(
    data: MMMIDataWrapper,
    pp_var: str = "y_original_scale",
) -> xr.DataArray:
    """Compute residuals as target_data - posterior predictions.

    Parameters
    ----------
    data : MMMIDataWrapper
        Wrapper holding idata with posterior_predictive and constant_data.
    pp_var : str, default "y_original_scale"
        Variable in posterior_predictive to use as predictions.
        This parameter was added to fix issue IV.18 (previously hardcoded).

    Returns
    -------
    xr.DataArray
        Residuals named "residuals" with same dims as *pp_var*
        (typically ``(chain, draw, date[, extra_dims])``).

    Raises
    ------
    ValueError
        If *pp_var* not in posterior_predictive, or target_data not in constant_data.
    """
    pp_ds = _get_posterior_predictive(data)
    if pp_var not in pp_ds:
        raise ValueError(
            f"Variable '{pp_var}' not found in posterior_predictive. "
            f"Available: {list(pp_ds.data_vars)}"
        )
    predictions = pp_ds[pp_var]
    target = data.get_target(original_scale=True)
    residuals = target - predictions
    residuals.name = "residuals"
    return residuals


def _zero_hline(ax: Axes, **kwargs: Any) -> None:
    """Draw a zero reference line on an axes panel.

    Designed as a ``PlotCollection.map()``-compatible callback.
    """
    ax.axhline(y=0.0, **kwargs)


class DiagnosticsPlots:
    """Time-series diagnostic plots for fitted MMM models.

    Provides four methods to visualize model fit and residuals:

    - ``posterior_predictive`` — Posterior predictive time series with HDI.
    - ``prior_predictive``    — Prior predictive time series with HDI.
    - ``residuals``            — Residuals (target − predictions) over time.
    - ``residuals_distribution`` — Posterior distribution of residuals.

    Parameters
    ----------
    data : MMMIDataWrapper
        Validated wrapper around the fitted model's InferenceData.
    """

    def __init__(self, data: MMMIDataWrapper) -> None:
        self._data = data
```

- [ ] **Step 1.4: Run helper tests — expect all pass**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py::TestGetPosteriorPredictive tests/mmm/plotting/test_diagnostics.py::TestGetPriorPredictive tests/mmm/plotting/test_diagnostics.py::TestComputeResiduals tests/mmm/plotting/test_diagnostics.py::TestDiagnosticsPlotsConstructor -v --no-header
```

- [ ] **Step 1.5: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/diagnostics.py tests/mmm/plotting/test_diagnostics.py
```

Fix any linting issues, then re-run pre-commit until clean.

- [ ] **Step 1.6: Commit**

```bash
git add pymc_marketing/mmm/plotting/diagnostics.py tests/mmm/plotting/test_diagnostics.py
git commit -m "feat(plotting): add DiagnosticsPlots skeleton and module-level helpers"
```

---

## Task 2: `posterior_predictive()` method

**Files:**
- Modify: `pymc_marketing/mmm/plotting/diagnostics.py`
- Modify: `tests/mmm/plotting/test_diagnostics.py`

- [ ] **Step 2.1: Add failing tests**

Append to `tests/mmm/plotting/test_diagnostics.py`:

```python
# ============================================================================
# posterior_predictive tests
# ============================================================================


class TestPosteriorPredictiveBasic:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.posterior_predictive()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_single_panel_no_extra_dims(self, simple_plots):
        _, axes = simple_plots.posterior_predictive()
        assert axes.size == 1

    def test_return_as_pc(self, simple_plots):
        result = simple_plots.posterior_predictive(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_default_var_names_is_y(self, simple_plots):
        # Must not raise — default var_names=["y"] exists in simple_idata
        fig, _ = simple_plots.posterior_predictive()
        assert isinstance(fig, Figure)

    def test_explicit_var_names(self, simple_plots):
        fig, _ = simple_plots.posterior_predictive(var_names=["y"])
        assert isinstance(fig, Figure)

    def test_raises_on_missing_var(self, simple_plots):
        with pytest.raises(ValueError, match="nonexistent"):
            simple_plots.posterior_predictive(var_names=["nonexistent"])

    def test_hdi_prob_accepted(self, simple_plots):
        fig, _ = simple_plots.posterior_predictive(hdi_prob=0.50)
        assert isinstance(fig, Figure)


class TestPosteriorPredictiveDims:
    def test_panel_idata_creates_multiple_panels(self, panel_plots):
        _, axes = panel_plots.posterior_predictive()
        assert axes.size >= 2  # one per geo value

    def test_dims_filter_single_value(self, panel_plots):
        _, axes = panel_plots.posterior_predictive(dims={"geo": ["CA"]})
        assert axes.size == 1

    def test_invalid_dim_raises(self, panel_plots):
        with pytest.raises(ValueError, match="nonexistent"):
            panel_plots.posterior_predictive(dims={"nonexistent": "CA"})


class TestPosteriorPredictiveCustomization:
    def test_figsize_accepted(self, simple_plots):
        fig, _ = simple_plots.posterior_predictive(figsize=(10, 4))
        assert isinstance(fig, Figure)

    def test_non_matplotlib_backend_without_return_as_pc_raises(self, simple_plots):
        with pytest.raises(ValueError, match="return_as_pc"):
            simple_plots.posterior_predictive(backend="plotly")

    def test_line_kwargs_accepted(self, simple_plots):
        fig, _ = simple_plots.posterior_predictive(line_kwargs={"color": "blue"})
        assert isinstance(fig, Figure)

    def test_hdi_kwargs_accepted(self, simple_plots):
        fig, _ = simple_plots.posterior_predictive(hdi_kwargs={"alpha": 0.1})
        assert isinstance(fig, Figure)


class TestPosteriorPredictiveIdataOverride:
    def test_idata_override_uses_different_data(self, simple_plots, panel_idata):
        """panel_idata has a geo dim — should create more panels than simple_idata."""
        _, axes = simple_plots.posterior_predictive(idata=panel_idata)
        assert axes.size >= 2

    def test_idata_override_does_not_mutate_self(self, simple_plots, simple_idata, panel_idata):
        """self._data must not be mutated after an idata override call."""
        simple_plots.posterior_predictive(idata=panel_idata)
        assert simple_plots._data.idata is simple_idata
```

- [ ] **Step 2.2: Run to verify all fail**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py -k "PostPred" -v --no-header
```

Expected: `AttributeError: 'DiagnosticsPlots' object has no attribute 'posterior_predictive'`

- [ ] **Step 2.3: Implement `posterior_predictive()`**

Add this method to the `DiagnosticsPlots` class in `diagnostics.py`:

```python
def posterior_predictive(
    self,
    var_names: list[str] | None = None,
    hdi_prob: float = 0.94,
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    line_kwargs: dict[str, Any] | None = None,
    hdi_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Plot time series from the posterior predictive distribution.

    Creates one panel per extra-dimension combination (e.g. one per geo
    for geo-segmented models). Each panel shows the posterior median as a
    line and an HDI band. Multiple variables in *var_names* are overlaid
    as separate layers on the same panels.

    Parameters
    ----------
    var_names : list[str], optional
        Variable names from posterior_predictive to plot. Default ``["y"]``.
    hdi_prob : float, default 0.94
        Probability mass of the HDI band (renamed from ``hdi_probs`` — II.5).
    idata : az.InferenceData, optional
        Override instance data. Constructs a local MMMIDataWrapper for this
        call only — does not mutate ``self._data``.
    dims : dict[str, Any], optional
        Subset dimensions, e.g. ``{"geo": ["CA", "NY"]}``.
    figsize : tuple[float, float], optional
        Figure size injected into ``figure_kwargs``.
    backend : str, optional
        Rendering backend. Non-matplotlib backends require ``return_as_pc=True``.
    return_as_pc : bool, default False
        If True, return the PlotCollection instead of ``(Figure, NDArray[Axes])``.
    line_kwargs : dict, optional
        Forwarded to ``azp.visuals.line_xy`` for the median line.
    hdi_kwargs : dict, optional
        Forwarded to ``azp.visuals.fill_between_y`` for the HDI band.
    **pc_kwargs
        Forwarded to ``PlotCollection.wrap()``.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection

    Examples
    --------
    .. code-block:: python

        fig, axes = mmm.plot.diagnostics.posterior_predictive()
        fig, axes = mmm.plot.diagnostics.posterior_predictive(
            var_names=["y"], hdi_prob=0.50, dims={"geo": ["CA"]}
        )
    """
    data = (
        MMMIDataWrapper(idata, schema=self._data.schema)
        if idata is not None
        else self._data
    )

    pc_kwargs = _process_plot_params(
        figsize=figsize,
        backend=backend,
        return_as_pc=return_as_pc,
        **pc_kwargs,
    )

    if var_names is None:
        var_names = ["y"]

    pp_ds = _get_posterior_predictive(data)

    for vn in var_names:
        if vn not in pp_ds:
            raise ValueError(
                f"Variable '{vn}' not found in posterior_predictive. "
                f"Available: {list(pp_ds.data_vars)}"
            )

    # Use the first var to determine layout (extra dims for faceting)
    main_da = _select_dims(pp_ds[var_names[0]], dims)
    extra_dims = [d for d in main_da.dims if d not in ("chain", "draw", "date")]
    mean_da = main_da.mean(dim=("chain", "draw"))

    # Build PlotCollection faceting on extra dims only (not on date)
    layout_ds = mean_da.isel(date=0, drop=True).to_dataset(name="y")
    pc = PlotCollection.wrap(
        layout_ds,
        cols=extra_dims,
        backend=backend,
        **pc_kwargs,
    )

    dates = main_da.coords["date"].values

    for vn in var_names:
        var_da = _select_dims(pp_ds[vn], dims)
        median_da = var_da.mean(dim=("chain", "draw"))
        hdi_da = var_da.azstats.hdi(hdi_prob)

        pc.map(
            azp.visuals.line_xy,
            x=dates,
            y=median_da,
            **{"label": vn, **(line_kwargs or {})},
        )
        pc.map(
            azp.visuals.fill_between_y,
            x=dates,
            y_bottom=hdi_da.sel(ci_bound="lower"),
            y_top=hdi_da.sel(ci_bound="upper"),
            **{"alpha": 0.2, **(hdi_kwargs or {})},
        )

    pc.map(azp.visuals.labelled_x, text="Date", ignore_aes={"color"})
    pc.map(azp.visuals.labelled_y, text="Posterior Predictive", ignore_aes={"color"})
    pc.map(azp.visuals.labelled_title, subset_info=True, ignore_aes={"color"})

    return _extract_matplotlib_result(pc, return_as_pc)
```

- [ ] **Step 2.4: Run tests — expect all pass**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py -k "PostPred" -v --no-header
```

- [ ] **Step 2.5: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/diagnostics.py tests/mmm/plotting/test_diagnostics.py
```

- [ ] **Step 2.6: Commit**

```bash
git add pymc_marketing/mmm/plotting/diagnostics.py tests/mmm/plotting/test_diagnostics.py
git commit -m "feat(plotting): add DiagnosticsPlots.posterior_predictive()"
```

---

## Task 3: `prior_predictive()` method

Same structure as Task 2; uses `_get_prior_predictive` instead of `_get_posterior_predictive`.

**Files:**
- Modify: `pymc_marketing/mmm/plotting/diagnostics.py`
- Modify: `tests/mmm/plotting/test_diagnostics.py`

- [ ] **Step 3.1: Add failing tests**

Append to `tests/mmm/plotting/test_diagnostics.py`:

```python
# ============================================================================
# prior_predictive tests
# ============================================================================


class TestPriorPredictiveBasic:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.prior_predictive()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_single_panel_no_extra_dims(self, simple_plots):
        _, axes = simple_plots.prior_predictive()
        assert axes.size == 1

    def test_return_as_pc(self, simple_plots):
        result = simple_plots.prior_predictive(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_raises_on_missing_var(self, simple_plots):
        with pytest.raises(ValueError, match="nonexistent"):
            simple_plots.prior_predictive(var_names=["nonexistent"])

    def test_fix_iv1_error_messages_reference_prior(self, simple_plots):
        """IV.1: prior_predictive error messages must say 'prior', not 'posterior'."""
        with pytest.raises(ValueError, match="prior_predictive"):
            data = MMMIDataWrapper(az.InferenceData(), validate_on_init=False)
            DiagnosticsPlots(data).prior_predictive()


class TestPriorPredictiveDims:
    def test_panel_idata_creates_multiple_panels(self, panel_plots):
        _, axes = panel_plots.prior_predictive()
        assert axes.size >= 2

    def test_dims_filter_single_value(self, panel_plots):
        _, axes = panel_plots.prior_predictive(dims={"geo": ["CA"]})
        assert axes.size == 1


class TestPriorPredictiveCustomization:
    def test_figsize_accepted(self, simple_plots):
        fig, _ = simple_plots.prior_predictive(figsize=(10, 4))
        assert isinstance(fig, Figure)

    def test_line_kwargs_accepted(self, simple_plots):
        fig, _ = simple_plots.prior_predictive(line_kwargs={"color": "green"})
        assert isinstance(fig, Figure)


class TestPriorPredictiveIdataOverride:
    def test_idata_override_does_not_mutate_self(self, simple_plots, simple_idata, panel_idata):
        simple_plots.prior_predictive(idata=panel_idata)
        assert simple_plots._data.idata is simple_idata
```

- [ ] **Step 3.2: Run to verify all fail**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py -k "PriorPred" -v --no-header
```

- [ ] **Step 3.3: Implement `prior_predictive()`**

Add to `DiagnosticsPlots` (structure mirrors `posterior_predictive` exactly — only differences highlighted):

```python
def prior_predictive(
    self,
    var_names: list[str] | None = None,
    hdi_prob: float = 0.94,
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    line_kwargs: dict[str, Any] | None = None,
    hdi_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Plot time series from the prior predictive distribution.

    Mirrors ``posterior_predictive`` but draws from the prior_predictive
    group. Fix for issue IV.1: error messages correctly reference
    'prior_predictive' (the old code showed 'posterior' messages here).

    Parameters
    ----------
    var_names : list[str], optional
        Variable names from prior_predictive to plot. Default ``["y"]``.
    hdi_prob : float, default 0.94
        Probability mass of the HDI band.
    idata : az.InferenceData, optional
        Override instance data.
    dims : dict[str, Any], optional
        Subset dimensions.
    figsize : tuple[float, float], optional
    backend : str, optional
    return_as_pc : bool, default False
    line_kwargs : dict, optional
        Forwarded to ``azp.visuals.line_xy``.
    hdi_kwargs : dict, optional
        Forwarded to ``azp.visuals.fill_between_y``.
    **pc_kwargs
        Forwarded to ``PlotCollection.wrap()``.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection

    Examples
    --------
    .. code-block:: python

        fig, axes = mmm.plot.diagnostics.prior_predictive()
    """
    data = (
        MMMIDataWrapper(idata, schema=self._data.schema)
        if idata is not None
        else self._data
    )

    pc_kwargs = _process_plot_params(
        figsize=figsize,
        backend=backend,
        return_as_pc=return_as_pc,
        **pc_kwargs,
    )

    if var_names is None:
        var_names = ["y"]

    # Only difference from posterior_predictive: uses _get_prior_predictive
    pp_ds = _get_prior_predictive(data)

    for vn in var_names:
        if vn not in pp_ds:
            raise ValueError(
                f"Variable '{vn}' not found in prior_predictive. "
                f"Available: {list(pp_ds.data_vars)}"
            )

    main_da = _select_dims(pp_ds[var_names[0]], dims)
    extra_dims = [d for d in main_da.dims if d not in ("chain", "draw", "date")]
    mean_da = main_da.mean(dim=("chain", "draw"))

    layout_ds = mean_da.isel(date=0, drop=True).to_dataset(name="y")
    pc = PlotCollection.wrap(layout_ds, cols=extra_dims, backend=backend, **pc_kwargs)

    dates = main_da.coords["date"].values

    for vn in var_names:
        var_da = _select_dims(pp_ds[vn], dims)
        median_da = var_da.mean(dim=("chain", "draw"))
        hdi_da = var_da.azstats.hdi(hdi_prob)

        pc.map(
            azp.visuals.line_xy,
            x=dates,
            y=median_da,
            **{"label": vn, **(line_kwargs or {})},
        )
        pc.map(
            azp.visuals.fill_between_y,
            x=dates,
            y_bottom=hdi_da.sel(ci_bound="lower"),
            y_top=hdi_da.sel(ci_bound="upper"),
            **{"alpha": 0.2, **(hdi_kwargs or {})},
        )

    pc.map(azp.visuals.labelled_x, text="Date", ignore_aes={"color"})
    pc.map(azp.visuals.labelled_y, text="Prior Predictive", ignore_aes={"color"})
    pc.map(azp.visuals.labelled_title, subset_info=True, ignore_aes={"color"})

    return _extract_matplotlib_result(pc, return_as_pc)
```

- [ ] **Step 3.4: Run tests — expect all pass**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py -k "PriorPred" -v --no-header
```

- [ ] **Step 3.5: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/diagnostics.py tests/mmm/plotting/test_diagnostics.py
```

- [ ] **Step 3.6: Commit**

```bash
git add pymc_marketing/mmm/plotting/diagnostics.py tests/mmm/plotting/test_diagnostics.py
git commit -m "feat(plotting): add DiagnosticsPlots.prior_predictive()"
```

---

## Task 4: `residuals()` method

**Files:**
- Modify: `pymc_marketing/mmm/plotting/diagnostics.py`
- Modify: `tests/mmm/plotting/test_diagnostics.py`

- [ ] **Step 4.1: Add failing tests**

Append to `tests/mmm/plotting/test_diagnostics.py`:

```python
# ============================================================================
# residuals tests
# ============================================================================


class TestResidualsBasic:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.residuals()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_single_panel_no_extra_dims(self, simple_plots):
        _, axes = simple_plots.residuals()
        assert axes.size == 1

    def test_return_as_pc(self, simple_plots):
        result = simple_plots.residuals(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_custom_hdi_prob(self, simple_plots):
        fig, _ = simple_plots.residuals(hdi_prob=0.50)
        assert isinstance(fig, Figure)


class TestResidualsDims:
    def test_panel_idata_creates_multiple_panels(self, panel_plots):
        _, axes = panel_plots.residuals()
        assert axes.size >= 2

    def test_dims_filter(self, panel_plots):
        _, axes = panel_plots.residuals(dims={"geo": ["CA"]})
        assert axes.size == 1


class TestResidualsCustomization:
    def test_figsize_accepted(self, simple_plots):
        fig, _ = simple_plots.residuals(figsize=(10, 4))
        assert isinstance(fig, Figure)

    def test_hdi_kwargs_accepted(self, simple_plots):
        fig, _ = simple_plots.residuals(hdi_kwargs={"alpha": 0.1})
        assert isinstance(fig, Figure)

    def test_line_kwargs_accepted(self, simple_plots):
        fig, _ = simple_plots.residuals(line_kwargs={"color": "red"})
        assert isinstance(fig, Figure)


class TestResidualsIdataOverride:
    def test_idata_override_uses_different_data(self, simple_plots, panel_idata):
        _, axes = simple_plots.residuals(idata=panel_idata)
        assert axes.size >= 2

    def test_idata_override_does_not_mutate_self(self, simple_plots, simple_idata, panel_idata):
        simple_plots.residuals(idata=panel_idata)
        assert simple_plots._data.idata is simple_idata
```

- [ ] **Step 4.2: Run to verify all fail**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py -k "TestResiduals" -v --no-header
```

Expected: `AttributeError: 'DiagnosticsPlots' object has no attribute 'residuals'`

- [ ] **Step 4.3: Implement `residuals()`**

Add to `DiagnosticsPlots`:

```python
def residuals(
    self,
    hdi_prob: float = 0.94,
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    hdi_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Plot residuals (target − posterior predictions) over time.

    Computes residuals using ``y_original_scale`` from posterior_predictive
    and ``target_data`` from constant_data. One panel per extra-dimension
    combination. Each panel shows a mean residuals line, an HDI band,
    and a zero reference line.

    Parameters
    ----------
    hdi_prob : float, default 0.94
        HDI probability mass for the residual band.
    idata : az.InferenceData, optional
        Override instance data.
    dims : dict[str, Any], optional
        Subset dimensions.
    figsize : tuple[float, float], optional
    backend : str, optional
    return_as_pc : bool, default False
    hdi_kwargs : dict, optional
        Forwarded to ``azp.visuals.fill_between_y``.
    line_kwargs : dict, optional
        Forwarded to ``azp.visuals.line_xy`` for the mean residuals line.
    **pc_kwargs
        Forwarded to ``PlotCollection.wrap()``.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection

    Examples
    --------
    .. code-block:: python

        fig, axes = mmm.plot.diagnostics.residuals()
        fig, axes = mmm.plot.diagnostics.residuals(hdi_prob=0.50)
    """
    data = (
        MMMIDataWrapper(idata, schema=self._data.schema)
        if idata is not None
        else self._data
    )

    pc_kwargs = _process_plot_params(
        figsize=figsize,
        backend=backend,
        return_as_pc=return_as_pc,
        **pc_kwargs,
    )

    residuals_da = _compute_residuals(data)  # (chain, draw, date[, extra_dims])
    residuals_da = _select_dims(residuals_da, dims)

    extra_dims = [d for d in residuals_da.dims if d not in ("chain", "draw", "date")]
    mean_da = residuals_da.mean(dim=("chain", "draw"))      # (date[, extra_dims])
    hdi_da = residuals_da.azstats.hdi(hdi_prob)             # (date[, extra_dims], ci_bound)

    layout_ds = mean_da.isel(date=0, drop=True).to_dataset(name="residuals")
    pc = PlotCollection.wrap(layout_ds, cols=extra_dims, backend=backend, **pc_kwargs)

    dates = residuals_da.coords["date"].values

    pc.map(
        azp.visuals.fill_between_y,
        x=dates,
        y_bottom=hdi_da.sel(ci_bound="lower"),
        y_top=hdi_da.sel(ci_bound="upper"),
        **{"alpha": 0.3, "label": f"{100 * hdi_prob:.0f}% HDI", **(hdi_kwargs or {})},
    )
    pc.map(
        azp.visuals.line_xy,
        x=dates,
        y=mean_da,
        **{"label": "Mean residuals", **(line_kwargs or {})},
    )
    pc.map(_zero_hline, linestyle="--", color="black", label="zero")

    pc.map(azp.visuals.labelled_x, text="Date", ignore_aes={"color"})
    pc.map(azp.visuals.labelled_y, text="Target − Predictions", ignore_aes={"color"})
    pc.map(azp.visuals.labelled_title, subset_info=True, ignore_aes={"color"})

    return _extract_matplotlib_result(pc, return_as_pc)
```

**Implementation note on `_zero_hline`:** PlotCollection's `map()` calls the function
with the axes as the first positional argument. `_zero_hline(ax, **kwargs)` should
receive `ax` automatically. If PlotCollection passes it differently (e.g., as keyword
`ax=`), adjust the signature to `def _zero_hline(*, ax, **kwargs)`.

- [ ] **Step 4.4: Run tests — expect all pass**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py -k "TestResiduals" -v --no-header
```

- [ ] **Step 4.5: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/diagnostics.py tests/mmm/plotting/test_diagnostics.py
```

- [ ] **Step 4.6: Commit**

```bash
git add pymc_marketing/mmm/plotting/diagnostics.py tests/mmm/plotting/test_diagnostics.py
git commit -m "feat(plotting): add DiagnosticsPlots.residuals()"
```

---

## Task 5: `residuals_distribution()` method (matplotlib fallback)

`az.plot_dist` has no `PlotCollection` equivalent → matplotlib fallback.
`return_as_pc=True` raises `ValueError`. Non-matplotlib backends raise `ValueError`.

**Files:**
- Modify: `pymc_marketing/mmm/plotting/diagnostics.py`
- Modify: `tests/mmm/plotting/test_diagnostics.py`

- [ ] **Step 5.1: Add failing tests**

Append to `tests/mmm/plotting/test_diagnostics.py`:

```python
# ============================================================================
# residuals_distribution tests
# ============================================================================


class TestResidualsDistributionBasic:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.residuals_distribution()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_no_extra_dims_single_panel(self, simple_plots):
        _, axes = simple_plots.residuals_distribution()
        assert axes.size == 1

    def test_aggregation_mean_single_panel(self, simple_plots):
        _, axes = simple_plots.residuals_distribution(aggregation="mean")
        assert axes.size == 1

    def test_aggregation_sum_single_panel(self, simple_plots):
        _, axes = simple_plots.residuals_distribution(aggregation="sum")
        assert axes.size == 1

    def test_return_as_pc_raises(self, simple_plots):
        with pytest.raises(ValueError, match="return_as_pc"):
            simple_plots.residuals_distribution(return_as_pc=True)

    def test_invalid_aggregation_raises(self, simple_plots):
        with pytest.raises(ValueError, match="aggregation"):
            simple_plots.residuals_distribution(aggregation="invalid")

    def test_invalid_quantile_raises(self, simple_plots):
        with pytest.raises(ValueError, match="quantile"):
            simple_plots.residuals_distribution(quantiles=[0.5, 1.5])

    def test_non_matplotlib_backend_raises(self, simple_plots):
        with pytest.raises(ValueError, match="backend"):
            simple_plots.residuals_distribution(backend="plotly")


class TestResidualsDistributionDims:
    def test_panel_idata_multiple_panels(self, panel_plots):
        _, axes = panel_plots.residuals_distribution()
        assert axes.size >= 2

    def test_dims_filter(self, panel_plots):
        _, axes = panel_plots.residuals_distribution(dims={"geo": ["CA"]})
        assert axes.size == 1


class TestResidualsDistributionIdataOverride:
    def test_idata_override_does_not_mutate_self(self, simple_plots, simple_idata, panel_idata):
        simple_plots.residuals_distribution(idata=panel_idata)
        assert simple_plots._data.idata is simple_idata
```

- [ ] **Step 5.2: Run to verify all fail**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py -k "ResidualsDistribution" -v --no-header
```

- [ ] **Step 5.3: Implement `residuals_distribution()`**

Add to `DiagnosticsPlots`:

```python
def residuals_distribution(
    self,
    quantiles: list[float] | None = None,
    aggregation: str | None = None,
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    dist_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]]:
    """Plot the posterior distribution of residuals.

    **Matplotlib fallback** — uses ``az.plot_dist`` which has no
    PlotCollection equivalent. ``return_as_pc=True`` is not supported
    and raises ``ValueError``. Non-matplotlib backends raise ``ValueError``.

    When *aggregation* is ``None`` (default), creates one panel per extra-
    dimension combination. When *aggregation* is ``"mean"`` or ``"sum"``,
    reduces across all non-(chain, draw) dims first and renders a single panel.

    Parameters
    ----------
    quantiles : list[float], optional
        Quantiles to mark on the distribution. Default ``[0.25, 0.5, 0.75]``.
    aggregation : str, optional
        ``"mean"``, ``"sum"``, or None. Controls pre-plot reduction.
    idata : az.InferenceData, optional
        Override instance data.
    dims : dict[str, Any], optional
        Subset dimensions applied before plotting.
    figsize : tuple[float, float], optional
        Passed to ``plt.subplots``.
    backend : str, optional
        Only ``None`` or ``"matplotlib"`` accepted.
    return_as_pc : bool, default False
        Not supported; raises ``ValueError`` if True.
    dist_kwargs : dict, optional
        Extra kwargs forwarded to ``az.plot_dist``.
    **pc_kwargs
        Accepted for API consistency; ignored.

    Returns
    -------
    tuple[Figure, NDArray[Axes]]

    Examples
    --------
    .. code-block:: python

        fig, axes = mmm.plot.diagnostics.residuals_distribution()
        fig, axes = mmm.plot.diagnostics.residuals_distribution(
            quantiles=[0.05, 0.5, 0.95], aggregation="mean"
        )
    """
    if return_as_pc:
        raise ValueError(
            "residuals_distribution uses a matplotlib fallback and does not "
            "support return_as_pc=True."
        )
    if backend is not None and backend != "matplotlib":
        raise ValueError(
            f"backend='{backend}' is not supported for residuals_distribution "
            "(matplotlib fallback only)."
        )
    if aggregation not in (None, "mean", "sum"):
        raise ValueError(
            f"aggregation must be 'mean', 'sum', or None; got {aggregation!r}."
        )
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    for q in quantiles:
        if not 0.0 <= q <= 1.0:
            raise ValueError(
                f"Each quantile must be in [0, 1]; got {q}."
            )

    data = (
        MMMIDataWrapper(idata, schema=self._data.schema)
        if idata is not None
        else self._data
    )

    residuals_da = _compute_residuals(data)
    residuals_da = _select_dims(residuals_da, dims)

    plot_figsize = figsize or (8, 6)
    dist_kw: dict[str, Any] = {
        "color": "C3",
        "fill_kwargs": {"alpha": 0.7},
        **(dist_kwargs or {}),
    }

    if aggregation is not None:
        dims_to_agg = [d for d in residuals_da.dims if d not in ("chain", "draw")]
        res_agg = (
            residuals_da.mean(dim=dims_to_agg)
            if aggregation == "mean"
            else residuals_da.sum(dim=dims_to_agg)
        )
        fig, ax = plt.subplots(figsize=plot_figsize)
        az.plot_dist(res_agg, quantiles=quantiles, ax=ax, **dist_kw)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Residuals Distribution ({aggregation})")
        ax.set_xlabel("Residuals")
        return fig, np.array([[ax]])

    # No aggregation — one panel per extra dim combination
    extra_dims = [d for d in residuals_da.dims if d not in ("chain", "draw", "date")]
    if extra_dims:
        dim_values = [list(residuals_da.coords[d].values) for d in extra_dims]
        combos = list(itertools.product(*dim_values))
    else:
        combos = [()]

    n = len(combos)
    fig, axes_arr = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(plot_figsize[0], plot_figsize[1] * n),
        squeeze=False,
    )

    for row_idx, combo in enumerate(combos):
        ax = axes_arr[row_idx][0]
        indexers = dict(zip(extra_dims, combo, strict=False)) if extra_dims else {}
        subset = residuals_da.sel(**indexers) if indexers else residuals_da

        stack_dims: tuple[str, ...] = (
            ("chain", "draw", "date") if "date" in subset.dims else ("chain", "draw")
        )
        flat = subset.stack(all_samples=stack_dims)

        az.plot_dist(flat, quantiles=quantiles, ax=ax, **dist_kw)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)

        title = (
            ", ".join(f"{k}={v}" for k, v in indexers.items())
            if indexers
            else "Residuals Distribution"
        )
        ax.set_title(title)
        ax.set_xlabel("Residuals")

    return fig, axes_arr
```

- [ ] **Step 5.4: Run tests — expect all pass**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py -k "ResidualsDistribution" -v --no-header
```

- [ ] **Step 5.5: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/diagnostics.py tests/mmm/plotting/test_diagnostics.py
```

- [ ] **Step 5.6: Commit**

```bash
git add pymc_marketing/mmm/plotting/diagnostics.py tests/mmm/plotting/test_diagnostics.py
git commit -m "feat(plotting): add DiagnosticsPlots.residuals_distribution() (matplotlib fallback)"
```

---

## Task 6: Export + full test run

**Files:**
- Modify: `pymc_marketing/mmm/plotting/__init__.py`
- Modify: `tests/mmm/plotting/test_diagnostics.py`

- [ ] **Step 6.1: Add failing import test**

Append to `tests/mmm/plotting/test_diagnostics.py`:

```python
# ============================================================================
# Package-level import test
# ============================================================================


def test_diagnostics_plots_importable_from_package():
    """DiagnosticsPlots must be importable from pymc_marketing.mmm.plotting."""
    from pymc_marketing.mmm.plotting import DiagnosticsPlots as DP
    assert DP is DiagnosticsPlots
```

- [ ] **Step 6.2: Run to verify fail**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py::test_diagnostics_plots_importable_from_package -v --no-header
```

Expected: `ImportError`

- [ ] **Step 6.3: Update `__init__.py`**

```python
# pymc_marketing/mmm/plotting/__init__.py
#   Copyright 2022 - 2026 The PyMC Labs Developers
#   ... (keep existing Apache license header) ...
"""MMM plotting package — namespace-based plot suite."""

from pymc_marketing.mmm.plotting.diagnostics import DiagnosticsPlots
from pymc_marketing.mmm.plotting.transformations import TransformationPlots

__all__ = ["DiagnosticsPlots", "TransformationPlots"]
```

- [ ] **Step 6.4: Run full test suite for this module**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_diagnostics.py -v --no-header
```

Expected: All pass.

- [ ] **Step 6.5: Run existing tests to check for regressions**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/ -v --no-header
```

- [ ] **Step 6.6: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/__init__.py tests/mmm/plotting/test_diagnostics.py
```

- [ ] **Step 6.7: Commit**

```bash
git add pymc_marketing/mmm/plotting/__init__.py tests/mmm/plotting/test_diagnostics.py
git commit -m "feat(plotting): export DiagnosticsPlots from mmm.plotting package"
```

---

## Checklist: Design doc compliance

Before marking this PR complete, verify:

| Requirement | Source | Status |
|---|---|---|
| `posterior_predictive` renamed from old flat method | Design doc §PR2 | — |
| `prior_predictive` renamed and IV.1 error messages fixed | Design doc §IV.1 | — |
| `residuals` renamed from `residuals_over_time` | Design doc §PR2 | — |
| `residuals_distribution` renamed from `residuals_posterior_distribution` | Design doc §PR2 | — |
| `var` → `var_names` (list[str]) | Design doc §II.5 | — |
| `hdi_prob` default changed 0.85 → 0.94 | Design doc §Defaults | — |
| IV.18: `_compute_residuals` accepts `pp_var` param | Design doc §IV.18 | — |
| All methods accept `dims` parameter | Design doc §II.6 | — |
| All methods accept 5 standard customization params | Design doc §II.7 | — |
| All methods return `tuple[Figure, NDArray[Axes]]` by default | Design doc §II.1 | — |
| `return_as_pc=True` returns `PlotCollection` (methods 1–3) | Design doc §II.1 | — |
| No `plt.show()` calls | Design doc §II.3 | — |
| All data access via resolved local `data` (idata override pattern) | Design doc §I.3 | — |
| No nested functions — all helpers are module-level | Design doc §I.4 | — |
| `PlotCollection` used for methods 1–3 | Design doc §I.6 | — |
| `residuals_distribution` fallback acceptable (no PlotCollection equiv.) | Design doc §I.6 | — |
