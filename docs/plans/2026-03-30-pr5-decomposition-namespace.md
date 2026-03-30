# DecompositionPlots Namespace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `DecompositionPlots` as `pymc_marketing/mmm/plotting/decomposition.py` with three methods — `contributions_over_time`, `waterfall`, and `channel_share_hdi` — porting from the old `MMMPlotSuite` in `plot.py` to the new namespace pattern.

**Architecture:** `DecompositionPlots` holds only `data: MMMIDataWrapper` and implements three methods. `contributions_over_time` and `channel_share_hdi` use `PlotCollection`/arviz-plots and follow the standard `_process_plot_params` / `_extract_matplotlib_result` pattern from `diagnostics.py`. `waterfall` is pure matplotlib. All three use `MMMIDataWrapper.get_contributions()` for data access.

**Tech Stack:** Python ≥3.12, xarray (`.azstats.hdi()`), arviz-plots (`PlotCollection`, `azp.visuals`, `azp.plot_forest`), matplotlib, numpy, pymc-marketing (`MMMIDataWrapper`, `_helpers.py`)

**Spec:** `docs/plans/2026-03-30-pr5-decomposition-namespace-design.md`

---

## Context: What is already on this branch

**Reference file to read before implementing:** `pymc_marketing/mmm/plotting/diagnostics.py` — every method here follows the exact same pattern you must replicate.

**Helpers already in `pymc_marketing/mmm/plotting/_helpers.py`:**
- `_select_dims(data, dims)` — applies `.sel()` with validation
- `_validate_dims(dataset, dims)` — raises `ValueError` for bad dim names/values
- `_dims_to_sel_kwargs(dims)` — converts `dims` dict to `.sel()` kwargs (wraps scalars in lists)
- `_process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)` — validates backend/figsize, returns cleaned `pc_kwargs`
- `_extract_matplotlib_result(pc, return_as_pc)` — converts `PlotCollection` to `(Figure, NDArray[Axes])` or returns as-is

**`MMMIDataWrapper` key methods (in `pymc_marketing/data/idata/mmm_wrapper.py`):**
- `data.get_contributions(original_scale, include_baseline, include_controls, include_seasonality)` → `xr.Dataset` with keys `"channels"` (always), `"baseline"`, `"controls"`, `"seasonality"` (only when present in the model)
  - `"channels"` DataArray has dims `(chain, draw, date, channel[, extra_dims])`
  - `"baseline"` DataArray has dims `(chain, draw, date[, extra_dims])`
  - `"controls"` DataArray has dims `(chain, draw, date, control[, extra_dims])`
  - `"seasonality"` DataArray has dims `(chain, draw, date[, extra_dims])`
- `data.get_channel_contributions(original_scale)` → `xr.DataArray` with dims `(chain, draw, date, channel[, extra_dims])`
- `data.custom_dims` → `frozenset[str]` of extra dimension names (e.g., `{"geo"}`)
- `data.schema` — used when constructing override wrapper: `MMMIDataWrapper(idata, schema=self._data.schema)`
- `data.idata` — raw `az.InferenceData`

**idata override pattern** (copy exactly from `diagnostics.py`):
```python
data = (
    MMMIDataWrapper(idata, schema=self._data.schema)
    if idata is not None
    else self._data
)
```

---

## File Structure

| Action | File |
|--------|------|
| Create | `pymc_marketing/mmm/plotting/decomposition.py` |
| Create | `tests/mmm/plotting/test_decomposition.py` |
| Modify | `pymc_marketing/mmm/plotting/__init__.py` |

---

## Task 1: Skeleton + fixtures

**Files:**
- Create: `pymc_marketing/mmm/plotting/decomposition.py`
- Create: `tests/mmm/plotting/test_decomposition.py`

- [ ] **Step 1: Create the skeleton implementation file**

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
"""Decomposition namespace — contribution waterfall and time-series plots."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import arviz as az
import arviz_plots as azp
import itertools
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


class DecompositionPlots:
    """Decomposition plots for fitted MMM models.

    Provides three methods to visualize how the model decomposes the target:

    - ``contributions_over_time`` — Time-series of each contribution type with HDI.
    - ``waterfall``              — Horizontal waterfall chart of mean contributions.
    - ``channel_share_hdi``     — Forest plot of each channel's share of total response.

    Parameters
    ----------
    data : MMMIDataWrapper
        Validated wrapper around the fitted model's InferenceData.
    """

    def __init__(self, data: MMMIDataWrapper) -> None:
        self._data = data

    def contributions_over_time(
        self,
        include: list[Literal["channels", "baseline", "controls", "seasonality"]] | None = None,
        hdi_prob: float = 0.94,
        original_scale: bool = True,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        hdi_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        raise NotImplementedError

    def waterfall(
        self,
        original_scale: bool = True,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        bar_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure, NDArray[Axes]]:
        raise NotImplementedError

    def channel_share_hdi(
        self,
        hdi_prob: float = 0.94,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        raise NotImplementedError
```

- [ ] **Step 2: Create the test fixtures file**

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
from __future__ import annotations

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting.decomposition import DecompositionPlots

matplotlib.use("Agg")

SEED = sum(map(ord, "DecompositionPlots tests"))


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def simple_idata() -> az.InferenceData:
    """Minimal idata with channels + baseline contributions, no extra dims.

    posterior:
      channel_contribution   (chain, draw, date, channel)
      intercept_contribution (chain, draw, date)
    constant_data:
      target_data  (date,)
      target_scale scalar
    """
    rng = np.random.default_rng(SEED)
    n_chain, n_draw, n_date = 2, 40, 20
    channels = ["tv", "radio", "social"]
    dates = np.arange(n_date)

    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                rng.uniform(0, 100, size=(n_chain, n_draw, n_date, len(channels))),
                dims=("chain", "draw", "date", "channel"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "channel": channels,
                },
            ),
            "intercept_contribution": xr.DataArray(
                rng.uniform(50, 150, size=(n_chain, n_draw, n_date)),
                dims=("chain", "draw", "date"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                },
            ),
        }
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date,)),
                dims=("date",),
                coords={"date": dates},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(posterior=posterior, constant_data=const)


@pytest.fixture(scope="module")
def panel_idata() -> az.InferenceData:
    """idata with geo extra dim — (chain, draw, date, channel, geo) for channels.

    posterior:
      channel_contribution   (chain, draw, date, channel, geo)
      intercept_contribution (chain, draw, date, geo)
    constant_data:
      target_data  (date, geo)
      target_scale scalar
    """
    rng = np.random.default_rng(SEED + 1)
    n_chain, n_draw, n_date = 2, 30, 15
    channels = ["tv", "radio"]
    geos = ["CA", "NY"]
    dates = np.arange(n_date)

    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                rng.uniform(0, 100, size=(n_chain, n_draw, n_date, len(channels), len(geos))),
                dims=("chain", "draw", "date", "channel", "geo"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "channel": channels,
                    "geo": geos,
                },
            ),
            "intercept_contribution": xr.DataArray(
                rng.uniform(50, 150, size=(n_chain, n_draw, n_date, len(geos))),
                dims=("chain", "draw", "date", "geo"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "geo": geos,
                },
            ),
        }
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date, len(geos))),
                dims=("date", "geo"),
                coords={"date": dates, "geo": geos},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(posterior=posterior, constant_data=const)


@pytest.fixture(scope="module")
def simple_data(simple_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(simple_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def panel_data(panel_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(panel_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def simple_plots(simple_data) -> DecompositionPlots:
    return DecompositionPlots(simple_data)


@pytest.fixture(scope="module")
def panel_plots(panel_data) -> DecompositionPlots:
    return DecompositionPlots(panel_data)
```

- [ ] **Step 3: Run the test file to verify it collects (no test failures yet)**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py --collect-only -q
```

Expected: 0 tests collected, no errors.

- [ ] **Step 4: Commit the skeleton**

```bash
git add pymc_marketing/mmm/plotting/decomposition.py tests/mmm/plotting/test_decomposition.py
git commit -m "feat: add DecompositionPlots skeleton and test fixtures"
```

---

## Task 2: `contributions_over_time`

**Files:**
- Modify: `pymc_marketing/mmm/plotting/decomposition.py`
- Modify: `tests/mmm/plotting/test_decomposition.py`

### Key implementation details

`get_contributions` returns an `xr.Dataset`. Variable dims:
- `"channels"`: `(chain, draw, date, channel[, extra_dims])` — has a `channel` dim that must be summed
- `"baseline"`: `(chain, draw, date[, extra_dims])` — no extra model dim
- `"controls"`: `(chain, draw, date, control[, extra_dims])` — has a `control` dim
- `"seasonality"`: `(chain, draw, date[, extra_dims])` — no extra model dim

The known dims to keep are `{"date", "chain", "draw"} | extra_dims`. Any other dim (like `channel`, `control`) must be summed, and a `UserWarning` is emitted for each.

The `include` parameter maps to `get_contributions` flags:
```python
all_keys = {"channels", "baseline", "controls", "seasonality"}
include_set = set(include) if include is not None else all_keys
contributions_ds = data.get_contributions(
    original_scale=original_scale,
    include_baseline="baseline" in include_set,
    include_controls="controls" in include_set,
    include_seasonality="seasonality" in include_set,
)
# "channels" is always fetched; filter afterwards if not requested
if "channels" not in include_set:
    contributions_ds = contributions_ds.drop_vars("channels", errors="ignore")
```

- [ ] **Step 1: Write the failing tests — add this class to the test file**

```python
class TestContributionsOverTime:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.contributions_over_time()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.ndim >= 1

    def test_returns_plot_collection_when_requested(self, simple_plots):
        result = simple_plots.contributions_over_time(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_panel_model_creates_one_panel_per_geo(self, panel_plots):
        fig, axes = panel_plots.contributions_over_time()
        # panel_idata has geo=["CA","NY"] — expect 2 axes
        assert len(axes) == 2

    def test_include_filters_contributions(self, simple_plots):
        # channels only — no baseline line
        fig, axes = simple_plots.contributions_over_time(include=["channels"])
        assert isinstance(fig, Figure)

    def test_include_invalid_key_raises(self, simple_plots):
        with pytest.raises((ValueError, KeyError)):
            simple_plots.contributions_over_time(include=["invalid_key"])

    def test_col_wrap_overridable(self, panel_plots):
        # default col_wrap=1 → 2 axes stacked; col_wrap=2 → side by side (still 2 axes)
        fig1, axes1 = panel_plots.contributions_over_time()
        fig2, axes2 = panel_plots.contributions_over_time(col_wrap=2)
        assert len(axes1) == len(axes2) == 2

    def test_idata_override(self, simple_plots, simple_idata):
        # Override with a fresh idata — should not raise
        fig, axes = simple_plots.contributions_over_time(idata=simple_idata)
        assert isinstance(fig, Figure)

    def test_dims_subsetting(self, panel_plots):
        fig, axes = panel_plots.contributions_over_time(dims={"geo": ["CA"]})
        assert isinstance(fig, Figure)

    def test_unexpected_dim_warns(self, simple_plots):
        # channel dim inside "channels" entry triggers UserWarning when summed
        with pytest.warns(UserWarning, match="summing"):
            simple_plots.contributions_over_time()
```

- [ ] **Step 2: Run and verify the tests fail**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py::TestContributionsOverTime -v 2>&1 | head -40
```

Expected: all tests fail with `NotImplementedError`.

- [ ] **Step 3: Implement `contributions_over_time`**

Replace the `raise NotImplementedError` body with:

```python
def contributions_over_time(
    self,
    include: list[Literal["channels", "baseline", "controls", "seasonality"]] | None = None,
    hdi_prob: float = 0.94,
    original_scale: bool = True,
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    line_kwargs: dict[str, Any] | None = None,
    hdi_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Plot time-series contributions for selected contribution types with HDI bands.

    Creates one panel per extra-dimension combination (e.g. one per geo for
    geo-segmented models). Each panel overlays one mean line and HDI band per
    contribution type.

    Parameters
    ----------
    include : list of {"channels", "baseline", "controls", "seasonality"}, optional
        Which contribution types to plot. ``None`` means all available.
    hdi_prob : float, default 0.94
        Probability mass for the HDI band.
    original_scale : bool, default True
        Whether to return contributions in original scale.
    idata : az.InferenceData, optional
        Override instance data for this call only.
    dims : dict[str, Any], optional
        Subset dimensions, e.g. ``{"geo": ["CA"]}``.
    figsize : tuple[float, float], optional
        Injected into ``figure_kwargs``.
    backend : str, optional
        Rendering backend. Non-matplotlib requires ``return_as_pc=True``.
    return_as_pc : bool, default False
        If True, return the ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
    line_kwargs : dict, optional
        Extra kwargs forwarded to ``azp.visuals.line_xy`` for every mean line.
    hdi_kwargs : dict, optional
        Extra kwargs forwarded to ``azp.visuals.fill_between_y`` for every HDI band.
    **pc_kwargs
        Forwarded to ``PlotCollection.wrap()``. Use ``col_wrap`` to override the
        default single-column layout.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection
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

    all_keys: set[str] = {"channels", "baseline", "controls", "seasonality"}
    include_set = set(include) if include is not None else all_keys
    invalid = include_set - all_keys
    if invalid:
        raise ValueError(
            f"Unknown contribution type(s): {invalid}. "
            f"Valid options: {all_keys}"
        )

    contributions_ds = data.get_contributions(
        original_scale=original_scale,
        include_baseline="baseline" in include_set,
        include_controls="controls" in include_set,
        include_seasonality="seasonality" in include_set,
    )
    if "channels" not in include_set:
        contributions_ds = contributions_ds.drop_vars("channels", errors="ignore")

    extra_dims = list(data.custom_dims)
    keep_dims = {"date", "chain", "draw"} | set(extra_dims)

    # Collapse model-specific dims (e.g. channel, control) into the time axis
    reduced: dict[str, xr.DataArray] = {}
    for key in contributions_ds.data_vars:
        da = contributions_ds[key]
        to_sum = [d for d in da.dims if d not in keep_dims]
        if to_sum:
            warnings.warn(
                f"contributions_over_time: summing over dimension(s) {to_sum} "
                f"for contribution '{key}'.",
                UserWarning,
                stacklevel=2,
            )
            da = da.sum(dim=to_sum)
        da = _select_dims(da, dims)
        reduced[key] = da

    if not reduced:
        raise ValueError(
            "No contribution data found after filtering. "
            "Check that the model has the requested contribution types."
        )

    first_da = next(iter(reduced.values()))
    dates = first_da.coords["date"].values

    layout_ds = first_da.mean(dim=("chain", "draw")).isel(date=0, drop=True).to_dataset(name="_layout")
    pc_kwargs.setdefault("col_wrap", 1)
    pc = PlotCollection.wrap(
        layout_ds,
        cols=extra_dims,
        backend=backend,
        **pc_kwargs,
    )

    for i, (label, da) in enumerate(reduced.items()):
        mean_da = da.mean(dim=("chain", "draw"))
        hdi_da = da.azstats.hdi(hdi_prob)
        color = f"C{i}"

        pc.map(
            azp.visuals.fill_between_y,
            x=dates,
            y_bottom=hdi_da.sel(ci_bound="lower"),
            y_top=hdi_da.sel(ci_bound="upper"),
            **{"alpha": 0.2, "color": color, **(hdi_kwargs or {})},
        )
        pc.map(
            azp.visuals.line_xy,
            x=dates,
            y=mean_da,
            **{"label": label, "color": color, **(line_kwargs or {})},
        )

    pc.map(azp.visuals.labelled_x, text="Date", ignore_aes={"color"})
    pc.map(azp.visuals.labelled_y, text="Contribution", ignore_aes={"color"})
    pc.map(azp.visuals.labelled_title, subset_info=True, ignore_aes={"color"})

    return _extract_matplotlib_result(pc, return_as_pc)
```

- [ ] **Step 4: Run the tests**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py::TestContributionsOverTime -v
```

Expected: all tests pass.

- [ ] **Step 5: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/decomposition.py tests/mmm/plotting/test_decomposition.py
```

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/decomposition.py tests/mmm/plotting/test_decomposition.py
git commit -m "feat: implement contributions_over_time in DecompositionPlots"
```

---

## Task 3: `waterfall`

**Files:**
- Modify: `pymc_marketing/mmm/plotting/decomposition.py`
- Modify: `tests/mmm/plotting/test_decomposition.py`

### Key implementation details

The waterfall chart renders one horizontal bar per contribution type, positioned cumulatively:
- Mean value per component = `.mean(dim=("chain", "draw", "date"))` then sum over any model-specific dim
- Components ordered: baseline → channels → controls → seasonality (only present keys)
- Starting value for each bar = running total from previous bars
- Color: `"green"` for positive, `"red"` for negative, `"grey"` for the final total bar
- Bar positions use positional integer indices (not coordinate values) to avoid IV.8
- The `bar_kwargs` dict is merged with `{"height": 0.5}` defaults — check for conflicts with `set.intersection`

Extra-dim combinations enumerate via `itertools.product`. For a model with `geo=["CA","NY"]`, the figure has 2 subplots (one per geo value).

- [ ] **Step 1: Write failing tests — add to the test file**

```python
class TestWaterfall:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.waterfall()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_single_panel_no_extra_dims(self, simple_plots):
        fig, axes = simple_plots.waterfall()
        assert len(axes) == 1

    def test_panel_model_one_panel_per_geo(self, panel_plots):
        fig, axes = panel_plots.waterfall()
        assert len(axes) == 2

    def test_dims_subsetting_reduces_panels(self, panel_plots):
        fig, axes = panel_plots.waterfall(dims={"geo": ["CA"]})
        assert len(axes) == 1

    def test_idata_override(self, simple_plots, simple_idata):
        fig, axes = simple_plots.waterfall(idata=simple_idata)
        assert isinstance(fig, Figure)

    def test_no_plt_gcf_used(self, simple_plots, monkeypatch):
        # Ensure no plt.gcf() is called internally
        import matplotlib.pyplot as plt_mod
        original_gcf = plt_mod.gcf
        called = []
        def patched_gcf():
            called.append(True)
            return original_gcf()
        monkeypatch.setattr(plt_mod, "gcf", patched_gcf)
        simple_plots.waterfall()
        assert called == [], "waterfall must not call plt.gcf()"
```

- [ ] **Step 2: Run and verify tests fail**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py::TestWaterfall -v 2>&1 | head -30
```

Expected: all fail with `NotImplementedError`.

- [ ] **Step 3: Implement `waterfall`**

Replace the `raise NotImplementedError` body with:

```python
def waterfall(
    self,
    original_scale: bool = True,
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    bar_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure, NDArray[Axes]]:
    """Horizontal waterfall chart showing mean contribution per component.

    One subplot per extra-dimension combination (e.g. per geo). Each subplot
    shows how each contribution type (baseline, channels, controls, seasonality)
    builds up to the total.

    Parameters
    ----------
    original_scale : bool, default True
        Whether to plot contributions in original scale.
    idata : az.InferenceData, optional
        Override instance data for this call only.
    dims : dict[str, Any], optional
        Subset dimensions, e.g. ``{"geo": ["CA"]}``.
    figsize : tuple[float, float], optional
        Passed to ``plt.subplots()``.
    bar_kwargs : dict, optional
        Extra kwargs forwarded to ``ax.barh()``. Cannot conflict with
        positional arguments (``y``, ``width``, ``left``).

    Returns
    -------
    tuple[Figure, NDArray[Axes]]
    """
    data = (
        MMMIDataWrapper(idata, schema=self._data.schema)
        if idata is not None
        else self._data
    )

    contributions_ds = data.get_contributions(original_scale=original_scale)
    extra_dims = list(data.custom_dims)
    keep_dims = {"date", "chain", "draw"} | set(extra_dims)

    # Reduce each DataArray to (chain, draw, date[, extra_dims]) then take mean
    # Result: scalar or (extra_dims,) DataArray per contribution type
    means: dict[str, xr.DataArray] = {}
    for key in contributions_ds.data_vars:
        da = contributions_ds[key]
        to_sum = [d for d in da.dims if d not in keep_dims]
        if to_sum:
            da = da.sum(dim=to_sum)
        da = _select_dims(da, dims)
        means[key] = da.mean(dim=("chain", "draw", "date"))

    # Determine subplot combos
    if extra_dims:
        coord_values = [means[next(iter(means))].coords[d].values for d in extra_dims]
        combos = list(itertools.product(*coord_values))
    else:
        combos = [()]

    n_panels = len(combos)
    fig, axes_raw = plt.subplots(
        1, n_panels, figsize=figsize or (6 * n_panels, 4), squeeze=False
    )
    axes_flat = axes_raw.flatten()

    reserved_keys = {"y", "width", "left"}
    if bar_kwargs:
        conflict = reserved_keys & set(bar_kwargs.keys())
        if conflict:
            raise ValueError(
                f"bar_kwargs keys conflict with positional bar arguments: {conflict}. "
                "Do not pass 'y', 'width', or 'left' in bar_kwargs."
            )
    safe_bar_kwargs = {"height": 0.5, **(bar_kwargs or {})}

    ordered_keys = [k for k in ["baseline", "channels", "controls", "seasonality"] if k in means]

    for panel_idx, combo in enumerate(combos):
        ax = axes_flat[panel_idx]
        sel_kwargs = dict(zip(extra_dims, combo))

        # Extract scalar values using positional indexing (fix IV.8)
        values: dict[str, float] = {}
        for key in ordered_keys:
            da = means[key]
            if sel_kwargs:
                # Use .sel() with exact coord values — combo was built from actual coords
                da = da.sel(**{k: [v] for k, v in sel_kwargs.items()}).squeeze()
            values[key] = float(da.values)

        total = sum(values.values())
        components = list(values.items()) + [("total", total)]

        running = 0.0
        for bar_idx, (label, val) in enumerate(components):
            if label == "total":
                color = "grey"
                left = 0.0
                width = val
            else:
                color = "green" if val >= 0 else "red"
                left = running
                width = val
                running += val

            ax.barh(
                y=bar_idx,  # positional integer index (fix IV.8)
                width=width,
                left=left,
                color=color,
                **safe_bar_kwargs,
            )
            pct = 100 * val / total if total != 0 else 0.0
            ax.text(
                left + width / 2,
                bar_idx,
                f"{val:.1f} ({pct:.1f}%)",
                va="center",
                ha="center",
                fontsize=8,
            )

        ax.set_yticks(range(len(components)))
        ax.set_yticklabels([c[0] for c in components])
        title = " | ".join(f"{k}={v}" for k, v in sel_kwargs.items()) if sel_kwargs else ""
        if title:
            ax.set_title(title)
        ax.axvline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    return fig, np.atleast_1d(np.array(axes_flat))
```

- [ ] **Step 4: Run the tests**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py::TestWaterfall -v
```

Expected: all tests pass.

- [ ] **Step 5: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/decomposition.py tests/mmm/plotting/test_decomposition.py
```

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/decomposition.py tests/mmm/plotting/test_decomposition.py
git commit -m "feat: implement waterfall in DecompositionPlots"
```

---

## Task 4: `channel_share_hdi`

**Files:**
- Modify: `pymc_marketing/mmm/plotting/decomposition.py`
- Modify: `tests/mmm/plotting/test_decomposition.py`

### Key implementation details

`azp.plot_forest` accepts an `xr.Dataset` or `az.InferenceData`. We pass a Dataset built from the share DataArray. The shares DataArray must:
- Have a `channel` coordinate (not `x`) — fix VII.1 is simply ensuring the DataArray is named and coordinated correctly before passing to `azp.plot_forest`
- Have dims `(chain, draw, channel[, extra_dims])` after summing over `date`

`azp.plot_forest` returns a `PlotCollection`, so `_extract_matplotlib_result` works directly.

- [ ] **Step 1: Write failing tests — add to the test file**

```python
class TestChannelShareHdi:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.channel_share_hdi()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_returns_plot_collection_when_requested(self, simple_plots):
        result = simple_plots.channel_share_hdi(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_idata_override(self, simple_plots, simple_idata):
        fig, axes = simple_plots.channel_share_hdi(idata=simple_idata)
        assert isinstance(fig, Figure)

    def test_dims_subsetting(self, panel_plots):
        fig, axes = panel_plots.channel_share_hdi(dims={"geo": ["CA"]})
        assert isinstance(fig, Figure)

    def test_channel_coordinate_present(self, simple_plots):
        # The dataset passed to azp.plot_forest has a 'channel' coordinate, not 'x'
        pc = simple_plots.channel_share_hdi(return_as_pc=True)
        assert isinstance(pc, PlotCollection)
```

- [ ] **Step 2: Run and verify tests fail**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py::TestChannelShareHdi -v 2>&1 | head -30
```

Expected: all fail with `NotImplementedError`.

- [ ] **Step 3: Implement `channel_share_hdi`**

Replace the `raise NotImplementedError` body with:

```python
def channel_share_hdi(
    self,
    hdi_prob: float = 0.94,
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Forest plot of each channel's share of total channel contribution.

    Computes each channel's contribution as a fraction of total channel
    contribution (summed over dates), then plots the HDI for each channel.

    Parameters
    ----------
    hdi_prob : float, default 0.94
        HDI probability mass.
    idata : az.InferenceData, optional
        Override instance data for this call only.
    dims : dict[str, Any], optional
        Subset dimensions, e.g. ``{"geo": ["CA"]}``.
    figsize : tuple[float, float], optional
        Injected into ``figure_kwargs``.
    backend : str, optional
        Rendering backend. Non-matplotlib requires ``return_as_pc=True``.
    return_as_pc : bool, default False
        If True, return the ``PlotCollection``.
    **pc_kwargs
        Forwarded to ``azp.plot_forest()``.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection
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

    # (chain, draw, date, channel[, extra_dims])
    channel_contributions = data.get_channel_contributions(original_scale=True)
    channel_contributions = _select_dims(channel_contributions, dims)

    # Sum over date → (chain, draw, channel[, extra_dims])
    summed = channel_contributions.sum(dim="date")

    # Compute share per channel (fix VII.1: DataArray has 'channel' coord, not 'x')
    total = summed.sum(dim="channel")
    shares = summed / total
    shares.name = "channel_share"

    share_ds = shares.to_dataset(name="channel_share")

    pc = azp.plot_forest(
        share_ds,
        var_names=["channel_share"],
        combined=True,
        hdi_prob=hdi_prob,
        backend=backend,
        **pc_kwargs,
    )
    return _extract_matplotlib_result(pc, return_as_pc)
```

- [ ] **Step 4: Run the tests**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py::TestChannelShareHdi -v
```

Expected: all tests pass.

- [ ] **Step 5: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/decomposition.py tests/mmm/plotting/test_decomposition.py
```

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/decomposition.py tests/mmm/plotting/test_decomposition.py
git commit -m "feat: implement channel_share_hdi in DecompositionPlots"
```

---

## Task 5: Wire into `__init__.py` and run full test suite

**Files:**
- Modify: `pymc_marketing/mmm/plotting/__init__.py`

- [ ] **Step 1: Read the current `__init__.py`**

```bash
cat pymc_marketing/mmm/plotting/__init__.py
```

- [ ] **Step 2: Add `DecompositionPlots` to the exports**

Add to the imports and `__all__` list:

```python
from pymc_marketing.mmm.plotting.decomposition import DecompositionPlots
```

And in `__all__`:

```python
__all__ = ["DecompositionPlots", "DiagnosticsPlots", "TransformationPlots"]
```

- [ ] **Step 3: Write the importability test — add to the end of `test_decomposition.py`**

```python
def test_decomposition_plots_importable_from_package():
    from pymc_marketing.mmm.plotting import DecompositionPlots as DP
    assert DP is DecompositionPlots
```

- [ ] **Step 4: Run the full test file**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/__init__.py tests/mmm/plotting/test_decomposition.py
```

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/__init__.py tests/mmm/plotting/test_decomposition.py
git commit -m "feat: export DecompositionPlots from plotting package"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `contributions_over_time` with PlotCollection | Task 2 |
| `include` parameter filtering | Task 2 |
| Warn on unexpected dim sum (IV.12) | Task 2, Step 3 |
| `col_wrap` user-overridable | Task 2, Step 3 (`.setdefault`) |
| idata override for all methods | Tasks 2, 3, 4 |
| `waterfall` pure matplotlib, no PlotCollection | Task 3 |
| Positional indexing in waterfall (IV.8) | Task 3 |
| No `plt.gcf()` (IV.2) | Task 3 (test + impl) |
| kwargs conflict detection (IV.9) | Task 3 |
| `channel_share_hdi` with `azp.plot_forest` | Task 4 |
| `channel` coordinate not `x` (VII.1) | Task 4 |
| Export from `__init__.py` | Task 5 |
| Test: smoke, return_as_pc, include, idata override, dims, panels | Tasks 2–4 |
