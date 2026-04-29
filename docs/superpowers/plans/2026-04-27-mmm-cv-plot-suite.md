# MMMCVPlotSuite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `MMMCVPlotSuite` — a standalone, PlotCollection-native class for visualizing cross-validation results — replacing the raw-matplotlib CV methods in `MMMPlotSuite`.

**Architecture:** New file `pymc_marketing/mmm/plotting/cv.py` holds the class and all module-level helpers. The `TimeSliceCrossValidator.plot` property is updated to return `MMMCVPlotSuite(self.cv_idata)`. Three methods — `predictions`, `param_stability`, `crps` — rewrite the old monolithic methods using `PlotCollection` exclusively. No raw matplotlib calls, no `plt.show()`, no loops.

**Tech Stack:** arviz 0.23+, arviz_plots, xarray, pandas, numpy, pytest, matplotlib (Agg backend in tests)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pymc_marketing/mmm/plotting/cv.py` | **Create** | `MMMCVPlotSuite` + all module-level helpers |
| `tests/mmm/plotting/test_cv.py` | **Create** | Full test suite with fixture and 20 test cases |
| `pymc_marketing/mmm/plotting/__init__.py` | **Modify** | Add `MMMCVPlotSuite` import and `__all__` entry |
| `pymc_marketing/mmm/time_slice_cross_validation.py` | **Modify** | Replace `.plot` property at line 177 |

---

## Task 1: Scaffold — `cv.py` shell + `__init__` + validation + test fixtures

**Files:**
- Create: `pymc_marketing/mmm/plotting/cv.py`
- Create: `tests/mmm/plotting/test_cv.py`

- [ ] **Step 1: Write failing tests for `__init__` validation**

```python
# tests/mmm/plotting/test_cv.py
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import arviz as az
import xarray as xr
from matplotlib.figure import Figure
from arviz_plots import PlotCollection

SEED = 42


@pytest.fixture(scope="module")
def cv_results_idata():
    """Minimal az.InferenceData for MMMCVPlotSuite tests.

    Three folds over 30 daily dates:
      fold_0 — train 0-19, test 20-29
      fold_1 — train 0-24, test 25-29
      fold_2 — train 0-29, test [] (degenerate fold, no test rows)
    """
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    cv_labels = ["fold_0", "fold_1", "fold_2"]
    channels = ["tv", "radio"]
    n_chains, n_draws = 2, 50

    posterior_ds = xr.Dataset(
        {
            "beta_channel": xr.DataArray(
                rng.normal(size=(3, n_chains, n_draws, 2)),
                dims=["cv", "chain", "draw", "channel"],
                coords={
                    "cv": cv_labels,
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "channel": channels,
                },
            )
        }
    )

    pp_ds = xr.Dataset(
        {
            "y_original_scale": xr.DataArray(
                rng.normal(100, 10, size=(3, n_chains, n_draws, 30)),
                dims=["cv", "chain", "draw", "date"],
                coords={
                    "cv": cv_labels,
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "date": dates,
                },
            )
        }
    )

    fold_specs = [(20, 20), (25, 25), (30, 30)]
    meta_arr = np.empty(3, dtype=object)
    for i, (train_end, test_start) in enumerate(fold_specs):
        X_train = pd.DataFrame({"date": dates[:train_end]})
        y_train = pd.Series(rng.normal(100, 10, size=train_end), name="y")
        if test_start < 30:
            X_test = pd.DataFrame({"date": dates[test_start:]})
            y_test = pd.Series(rng.normal(100, 10, size=30 - test_start), name="y")
        else:
            X_test = pd.DataFrame({"date": pd.DatetimeIndex([])})
            y_test = pd.Series([], name="y", dtype=float)
        meta_arr[i] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    cv_metadata_ds = xr.Dataset(
        {
            "metadata": xr.DataArray(
                meta_arr,
                dims=["cv"],
                coords={"cv": cv_labels},
            )
        }
    )

    return az.InferenceData(
        posterior=posterior_ds,
        posterior_predictive=pp_ds,
        cv_metadata=cv_metadata_ds,
    )


@pytest.fixture(scope="module")
def cv_plot(cv_results_idata):
    from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite
    return MMMCVPlotSuite(cv_results_idata)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestInit:
    def test_stores_cv_data(self, cv_results_idata):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite
        suite = MMMCVPlotSuite(cv_results_idata)
        assert suite.cv_data is cv_results_idata

    def test_raises_type_error_for_non_idata(self):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite
        with pytest.raises(TypeError, match="az.InferenceData"):
            MMMCVPlotSuite({"not": "idata"})

    def test_raises_value_error_without_cv_metadata(self):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite
        bad = az.InferenceData(posterior=xr.Dataset())
        with pytest.raises(ValueError, match="cv_metadata"):
            MMMCVPlotSuite(bad)
```

- [ ] **Step 2: Run tests to confirm they fail (ImportError expected)**

```
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py::TestInit -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `cv.py` does not exist yet.

- [ ] **Step 3: Create `pymc_marketing/mmm/plotting/cv.py` with validation helpers and `__init__`**

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
"""CV plotting namespace — MMMCVPlotSuite for TimeSliceCrossValidator results."""

from __future__ import annotations

from typing import Any

import arviz as az
import arviz_plots as azp
import numpy as np
import pandas as pd
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.metrics import crps as _crps_score
from pymc_marketing.mmm.plotting._helpers import (
    _extract_matplotlib_result,
    _process_plot_params,
    _select_dims,
)


# ── Shared base validation ────────────────────────────────────────────────────


def _validate_cv_results(cv_data: az.InferenceData) -> None:
    """Raise if cv_data is not a valid CV InferenceData.

    Minimum required: correct type + cv_metadata group present.
    Method-specific checks (e.g. posterior_predictive contents) are
    performed inside each method.
    """
    if not isinstance(cv_data, az.InferenceData):
        raise TypeError(
            f"cv_data must be az.InferenceData, got {type(cv_data).__name__}."
        )
    if not hasattr(cv_data, "cv_metadata"):
        raise ValueError(
            "cv_data must have a 'cv_metadata' group. "
            "Ensure TimeSliceCrossValidator.run() has been called and the "
            "resulting InferenceData is passed here."
        )


def _extract_cv_labels(cv_data: az.InferenceData) -> list[str]:
    """Return the list of CV fold labels from cv_metadata coords."""
    return list(cv_data.cv_metadata.coords["cv"].values)


def _read_fold_meta(
    cv_data: az.InferenceData, cv_label: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_train, y_train, X_test, y_test) for a given fold label."""
    meta = cv_data.cv_metadata["metadata"].sel(cv=cv_label).values.item()
    return meta["X_train"], meta["y_train"], meta["X_test"], meta["y_test"]


# ── Main class ────────────────────────────────────────────────────────────────


class MMMCVPlotSuite:
    """PlotCollection-native plots for TimeSliceCrossValidator results.

    Parameters
    ----------
    cv_data : az.InferenceData
        Combined InferenceData produced by ``TimeSliceCrossValidator.run()``.
        Must contain a ``cv_metadata`` group with per-fold metadata.
    """

    def __init__(self, cv_data: az.InferenceData) -> None:
        _validate_cv_results(cv_data)
        self.cv_data = cv_data

    def predictions(self, *args, **kwargs):
        raise NotImplementedError

    def param_stability(self, *args, **kwargs):
        raise NotImplementedError

    def crps(self, *args, **kwargs):
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to confirm they pass**

```
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py::TestInit -v
```

Expected: `3 passed`.

- [ ] **Step 5: Run pre-commit on both files**

```
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
```

Fix any formatting issues, then re-run until clean.

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
git commit -m "feat(mmm): scaffold MMMCVPlotSuite with validation helpers and test fixtures"
```

---

## Task 2: `predictions()` — tests then implementation

**Files:**
- Modify: `pymc_marketing/mmm/plotting/cv.py` (replace `predictions` stub)
- Modify: `tests/mmm/plotting/test_cv.py` (add `TestPredictions` class)

- [ ] **Step 1: Add `TestPredictions` to the test file**

Append this class to `tests/mmm/plotting/test_cv.py`:

```python
class TestPredictions:
    def test_returns_tuple(self, cv_plot):
        result = cv_plot.predictions()
        assert isinstance(result, tuple) and len(result) == 2
        fig, axes = result
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_return_as_pc(self, cv_plot):
        result = cv_plot.predictions(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_n_axes_equals_n_folds(self, cv_plot):
        fig, axes = cv_plot.predictions()
        # 3 folds → at least 3 axes (one per fold panel)
        assert len(axes) >= 3

    def test_train_test_colors_differ(self, cv_plot):
        fig, axes = cv_plot.predictions()
        ax = axes[0]
        colors = set()
        for coll in ax.collections:
            fc = coll.get_facecolor()
            if fc is not None and len(fc) > 0:
                colors.add(tuple(np.round(fc[0][:3], 2)))
        assert len(colors) >= 2, "Expected at least two fill colors (train/test)"

    def test_missing_cv_metadata_raises(self, cv_plot, cv_results_idata):
        bad = az.InferenceData(posterior_predictive=cv_results_idata.posterior_predictive)
        # bad has no cv_metadata — _validate_cv_results raises ValueError
        with pytest.raises((TypeError, ValueError)):
            cv_plot.predictions(cv_data=bad)

    def test_missing_posterior_predictive_raises(self, cv_plot, cv_results_idata):
        bad = az.InferenceData(cv_metadata=cv_results_idata.cv_metadata)
        with pytest.raises(ValueError, match="posterior_predictive"):
            cv_plot.predictions(cv_data=bad)

    def test_dims_filtering(self, cv_plot):
        import pandas as pd
        # Filter to a single date — date dim becomes size-1, plot still renders
        single_date = pd.Timestamp("2024-01-05")
        fig, axes = cv_plot.predictions(dims={"date": [single_date]})
        assert isinstance(fig, Figure)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py::TestPredictions -v
```

Expected: all fail with `NotImplementedError`.

- [ ] **Step 3: Implement `predictions()` in `cv.py`**

Replace the `predictions` stub with:

```python
def predictions(
    self,
    cv_data: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    hdi_prob: float = 0.94,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    hdi_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Posterior predictive HDI bands per CV fold.

    For each fold: blue HDI band over train dates, orange HDI band over test
    dates, black observed line, and a green dashed vertical boundary at the
    train/test split.

    Parameters
    ----------
    cv_data : az.InferenceData or None
        Override the stored ``self.cv_data`` for this call only.
        ``_validate_cv_results`` is re-run on the override.
    dims : dict or None
        Filter coordinate values before rendering
        (e.g. ``{"geo": ["North"]}``).
    hdi_prob : float
        HDI probability mass (default 0.94).
    figsize : tuple or None
        Figure size in inches; injected into ``figure_kwargs``.
    backend : str or None
        PlotCollection backend (``"matplotlib"`` / ``"plotly"`` / ``"bokeh"``).
        Non-matplotlib requires ``return_as_pc=True``.
    return_as_pc : bool
        Return the raw ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
    hdi_kwargs : dict or None
        Extra kwargs forwarded to ``azp.visuals.fill_between_y``.
    **pc_kwargs
        Forwarded to ``PlotCollection.grid()``.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection
    """
    data = cv_data if cv_data is not None else self.cv_data
    if cv_data is not None:
        _validate_cv_results(data)

    if not hasattr(data, "cv_metadata") or "metadata" not in data.cv_metadata:
        raise ValueError(
            "cv_data must have a cv_metadata group containing a 'metadata' variable."
        )
    if (
        not hasattr(data, "posterior_predictive")
        or "y_original_scale" not in data.posterior_predictive
    ):
        raise ValueError(
            "cv_data must have posterior_predictive['y_original_scale']."
        )

    cv_labels = _extract_cv_labels(data)
    pp = data.posterior_predictive["y_original_scale"]
    full_dates = pp.coords["date"].values

    y_train_list: list[xr.DataArray] = []
    y_test_list: list[xr.DataArray] = []
    y_obs_list: list[xr.DataArray] = []
    train_end_list: list[Any] = []

    for lbl in cv_labels:
        X_train, y_train, X_test, y_test = _read_fold_meta(data, lbl)

        train_dates = pd.DatetimeIndex(X_train["date"].values)
        test_dates = (
            pd.DatetimeIndex(X_test["date"].values)
            if X_test is not None and len(X_test) > 0
            else pd.DatetimeIndex([])
        )

        train_mask = xr.DataArray(
            np.isin(full_dates, train_dates.values),
            dims=["date"],
            coords={"date": full_dates},
        )
        test_mask = xr.DataArray(
            np.isin(full_dates, test_dates.values),
            dims=["date"],
            coords={"date": full_dates},
        )

        pp_fold = pp.sel(cv=lbl)
        y_train_list.append(pp_fold.where(train_mask))
        y_test_list.append(pp_fold.where(test_mask))

        # Align observed actuals to the full date coordinate
        date_to_y: dict[Any, float] = {}
        for d, y in zip(X_train["date"].values, np.asarray(y_train)):
            date_to_y[d] = float(y)
        if X_test is not None and len(X_test) > 0:
            for d, y in zip(X_test["date"].values, np.asarray(y_test)):
                date_to_y[d] = float(y)
        y_obs_arr = np.array([date_to_y.get(d, np.nan) for d in full_dates])
        y_obs_list.append(
            xr.DataArray(y_obs_arr, dims=["date"], coords={"date": full_dates})
        )
        train_end_list.append(train_dates.max())

    cv_coord = xr.DataArray(cv_labels, dims=["cv"], name="cv")
    y_train_da = xr.concat(y_train_list, dim=cv_coord).assign_coords(cv=cv_labels)
    y_test_da = xr.concat(y_test_list, dim=cv_coord).assign_coords(cv=cv_labels)
    y_obs_da = xr.concat(y_obs_list, dim=cv_coord).assign_coords(cv=cv_labels)
    train_end_da = xr.DataArray(
        train_end_list, dims=["cv"], coords={"cv": cv_labels}
    )

    if dims:
        y_train_da = _select_dims(y_train_da, dims)
        y_test_da = _select_dims(y_test_da, dims)
        y_obs_da = _select_dims(y_obs_da, dims)

    standard_dims = {"cv", "chain", "draw", "date"}
    custom_dims = [d for d in y_train_da.dims if d not in standard_dims]

    split_ds = xr.Dataset({"train": y_train_da, "test": y_test_da})

    pc_kwargs = _process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)
    rows = pc_kwargs.pop("rows", [*custom_dims, "cv"])
    cols = pc_kwargs.pop("cols", [])

    pc = PlotCollection.grid(
        split_ds,
        rows=rows,
        cols=cols,
        aes={"color": ["__variable__"]},
        backend=backend,
        **pc_kwargs,
    )

    hdi_ds = split_ds.azstats.hdi(hdi_prob)
    date_da = split_ds["train"].coords["date"]

    pc.map(
        azp.visuals.fill_between_y,
        x=date_da,
        y_bottom=hdi_ds.sel(ci_bound="lower"),
        y_top=hdi_ds.sel(ci_bound="upper"),
        alpha=0.3,
        **(hdi_kwargs or {}),
    )

    pc.map(azp.visuals.line_xy, x=date_da, y=y_obs_da, color="black", linewidth=1.5)

    azp.add_lines(
        pc,
        train_end_da,
        orientation="vertical",
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.9,
    )

    pc.add_legend("__variable__")

    return _extract_matplotlib_result(pc, return_as_pc)
```

- [ ] **Step 4: Run tests to confirm they pass**

```
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py::TestPredictions -v
```

Expected: all 7 pass.

- [ ] **Step 5: Run pre-commit**

```
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
```

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
git commit -m "feat(mmm): implement MMMCVPlotSuite.predictions() with PlotCollection"
```

---

## Task 3: `param_stability()` — tests then implementation

**Files:**
- Modify: `pymc_marketing/mmm/plotting/cv.py` (replace `param_stability` stub)
- Modify: `tests/mmm/plotting/test_cv.py` (add `TestParamStability` class)

- [ ] **Step 1: Add `TestParamStability` to the test file**

Append this class to `tests/mmm/plotting/test_cv.py`:

```python
class TestParamStability:
    def test_returns_tuple(self, cv_plot):
        fig, axes = cv_plot.param_stability()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_return_as_pc(self, cv_plot):
        result = cv_plot.param_stability(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_var_names(self, cv_plot):
        # Should run without error when restricting to known variable
        fig, axes = cv_plot.param_stability(var_names=["beta_channel"])
        assert isinstance(fig, Figure)

    def test_dims_filtering(self, cv_plot):
        # Filter posterior to a single channel before plotting
        fig, axes = cv_plot.param_stability(dims={"channel": ["tv"]})
        assert isinstance(fig, Figure)

    def test_no_cv_coord_raises(self, cv_results_idata):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite
        # Strip cv coordinate from posterior
        posterior = cv_results_idata.posterior.isel(cv=0, drop=True)
        bad = az.InferenceData(
            posterior=posterior,
            cv_metadata=cv_results_idata.cv_metadata,
        )
        suite = MMMCVPlotSuite(bad)
        with pytest.raises(ValueError, match="cv"):
            suite.param_stability()

    def test_single_figure(self, cv_plot):
        import matplotlib.pyplot as plt
        plt.close("all")
        cv_plot.param_stability()
        assert len(plt.get_fignums()) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py::TestParamStability -v
```

Expected: all fail with `NotImplementedError`.

- [ ] **Step 3: Implement `param_stability()` in `cv.py`**

Replace the `param_stability` stub with:

```python
def param_stability(
    self,
    cv_data: az.InferenceData | None = None,
    var_names: list[str] | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Forest plot comparing parameter posteriors across all CV folds.

    Parameters
    ----------
    cv_data : az.InferenceData or None
        Override the stored ``self.cv_data`` for this call only.
    var_names : list[str] or None
        Variables to include (passed directly to ``azp.plot_forest``).
    dims : dict or None
        Filter coordinate values before plotting
        (e.g. ``{"channel": ["tv"]}``).
    figsize : tuple or None
        Figure size in inches; takes precedence over ``figure_kwargs["figsize"]``.
    figure_kwargs : dict or None
        Extra kwargs for the figure constructor; merged with defaults.
    backend : str or None
        PlotCollection backend.
    return_as_pc : bool
        Return the raw ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
    **pc_kwargs
        Forwarded to ``azp.plot_forest()``.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection
    """
    data = cv_data if cv_data is not None else self.cv_data
    if cv_data is not None:
        _validate_cv_results(data)

    if not hasattr(data, "posterior"):
        raise ValueError("cv_data has no 'posterior' group.")
    if "cv" not in data.posterior.coords:
        raise ValueError(
            "No 'cv' coordinate found in cv_data.posterior. "
            "Ensure the InferenceData was produced by TimeSliceCrossValidator.run()."
        )

    posterior = data.posterior
    if dims:
        posterior = _select_dims(posterior, dims)

    # Move labelled dims to the end so the forest plot reads naturally.
    # Guard: only include dims that actually exist after optional filtering.
    dims_to_end = [d for d in ("channel", "cv") if d in posterior.dims]
    if dims_to_end:
        posterior = posterior.transpose(..., *dims_to_end)

    idata_for_plot = az.InferenceData(posterior=posterior)

    fig_kw: dict[str, Any] = {
        "width_ratios": [1, 2],
        "layout": "none",
        **(figure_kwargs or {}),
    }
    if figsize is not None:
        fig_kw["figsize"] = figsize

    pc = azp.plot_forest(
        idata_for_plot.to_datatree(),
        var_names=var_names,
        aes={"color": ["cv"]},
        figure_kwargs=fig_kw,
        combined=True,
        shade_label="channel",
        backend=backend,
        **pc_kwargs,
    )
    return _extract_matplotlib_result(pc, return_as_pc)
```

- [ ] **Step 4: Run tests to confirm they pass**

```
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py::TestParamStability -v
```

Expected: all 6 pass.

- [ ] **Step 5: Run pre-commit**

```
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
```

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
git commit -m "feat(mmm): implement MMMCVPlotSuite.param_stability() via azp.plot_forest"
```

---

## Task 4: `crps()` — CRPS helpers + tests + implementation

**Files:**
- Modify: `pymc_marketing/mmm/plotting/cv.py` (add CRPS helpers + replace `crps` stub)
- Modify: `tests/mmm/plotting/test_cv.py` (add `TestCRPS` class)

- [ ] **Step 1: Add `TestCRPS` to the test file**

Append this class to `tests/mmm/plotting/test_cv.py`:

```python
class TestCRPS:
    def test_returns_tuple(self, cv_plot):
        fig, axes = cv_plot.crps()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_return_as_pc(self, cv_plot):
        result = cv_plot.crps(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_line_count(self, cv_plot):
        # Exactly 2 lines on the single panel (train + test)
        fig, axes = cv_plot.crps()
        ax = axes[0]
        assert len(ax.lines) == 2

    def test_train_test_colors_differ(self, cv_plot):
        fig, axes = cv_plot.crps()
        ax = axes[0]
        colors = {line.get_color() for line in ax.lines}
        assert len(colors) >= 2, "Expected train and test lines in distinct colors"

    def test_missing_cv_metadata_raises(self, cv_plot, cv_results_idata):
        bad = az.InferenceData(
            posterior_predictive=cv_results_idata.posterior_predictive
        )
        with pytest.raises((TypeError, ValueError)):
            cv_plot.crps(cv_data=bad)

    def test_nan_tolerant(self, cv_plot):
        # fold_2 has an empty test set → test CRPS is NaN; rendering must not crash
        fig, axes = cv_plot.crps()
        assert isinstance(fig, Figure)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py::TestCRPS -v
```

Expected: all fail with `NotImplementedError`.

- [ ] **Step 3: Add CRPS helper functions to `cv.py`**

Add these three module-level helpers before the `MMMCVPlotSuite` class definition (after `_read_fold_meta`):

```python
def _pred_matrix_for_rows(
    cv_data: az.InferenceData,
    cv_label: str,
    rows_df: pd.DataFrame,
) -> np.ndarray:
    """Build (n_samples, n_rows) prediction matrix for CRPS computation.

    Selects posterior_predictive['y_original_scale'] for the given CV fold,
    stacks (chain, draw) → sample, then iterates over rows in rows_df
    matching each row's 'date' value to the 'date' coordinate.

    Returns
    -------
    np.ndarray, shape (n_samples, n_rows)
    """
    da = cv_data.posterior_predictive["y_original_scale"].sel(cv=cv_label)
    da_s = da.stack(sample=("chain", "draw"))
    if da_s.dims[0] != "sample":
        da_s = da_s.transpose("sample", ...)

    n_samples = int(da_s.sizes["sample"])
    n_rows = len(rows_df)
    mat = np.empty((n_samples, n_rows))

    for j, (_, row) in enumerate(rows_df.iterrows()):
        date_val = row["date"]
        arr = np.squeeze(da_s.sel(date=date_val).values)
        if arr.ndim == 0:
            arr = arr.reshape(n_samples)
        mat[:, j] = arr[:n_samples]

    return mat


def _filter_rows_and_y(
    df: pd.DataFrame,
    y: pd.Series,
    indexers: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter DataFrame rows by column equality, return aligned y array.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (X_train or X_test).
    y : pd.Series
        Target Series aligned to df by position.
    indexers : dict
        Column-name → value filters to apply.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Filtered DataFrame and corresponding y values.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(), np.array([])
    mask = np.ones(len(df), dtype=bool)
    for col, val in indexers.items():
        if col in df.columns:
            mask &= df[col] == val
    return df[mask].reset_index(drop=True), np.asarray(y)[mask]


def _crps_for_split(
    cv_data: az.InferenceData,
    cv_label: str,
    X: pd.DataFrame,
    y: pd.Series,
    dim_indexers: dict[str, Any],
) -> float:
    """Compute mean CRPS for one fold/split. Returns np.nan on failure or empty set."""
    try:
        X_filtered, y_arr = _filter_rows_and_y(X, y, dim_indexers)
        if len(X_filtered) == 0:
            return float(np.nan)
        pred_mat = _pred_matrix_for_rows(cv_data, cv_label, X_filtered)
        return float(_crps_score(y_true=y_arr, y_pred=pred_mat))
    except Exception:
        return float(np.nan)
```

- [ ] **Step 4: Implement `crps()` in `cv.py`**

Replace the `crps` stub with:

```python
def crps(
    self,
    cv_data: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    line_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Line chart of mean CRPS per fold for train and test splits.

    Parameters
    ----------
    cv_data : az.InferenceData or None
        Override the stored ``self.cv_data`` for this call only.
    dims : dict or None
        Column-value filters applied when selecting rows for CRPS computation
        (e.g. ``{"channel": "tv"}``).
    figsize : tuple or None
        Figure size in inches.
    backend : str or None
        PlotCollection backend.
    return_as_pc : bool
        Return the raw ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
    line_kwargs : dict or None
        Extra kwargs forwarded to ``azp.visuals.line_xy``.
    **pc_kwargs
        Forwarded to ``PlotCollection.wrap()``.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection
    """
    data = cv_data if cv_data is not None else self.cv_data
    if cv_data is not None:
        _validate_cv_results(data)

    if not hasattr(data, "cv_metadata"):
        raise ValueError("cv_data must have a 'cv_metadata' group.")
    if (
        not hasattr(data, "posterior_predictive")
        or "y_original_scale" not in data.posterior_predictive
    ):
        raise ValueError(
            "cv_data must have posterior_predictive['y_original_scale']."
        )

    cv_labels = _extract_cv_labels(data)
    dim_indexers = dims or {}
    crps_train_list: list[float] = []
    crps_test_list: list[float] = []

    for lbl in cv_labels:
        X_train, y_train, X_test, y_test = _read_fold_meta(data, lbl)
        crps_train_list.append(
            _crps_for_split(data, lbl, X_train, y_train, dim_indexers)
        )
        crps_test_list.append(
            _crps_for_split(data, lbl, X_test, y_test, dim_indexers)
        )

    crps_da = xr.DataArray(
        np.stack([np.array(crps_train_list), np.array(crps_test_list)]),
        dims=["split", "cv"],
        coords={"split": ["train", "test"], "cv": cv_labels},
    )
    crps_ds = crps_da.to_dataset(name="crps")

    pc_kwargs = _process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)
    pc = PlotCollection.wrap(
        crps_ds, aes={"color": ["split"]}, backend=backend, **pc_kwargs
    )

    cv_x = xr.DataArray(
        np.arange(len(cv_labels)), dims=["cv"], coords={"cv": cv_labels}
    )
    pc.map(azp.visuals.line_xy, x=cv_x, y=crps_ds["crps"], **(line_kwargs or {}))
    pc.add_legend("split")

    return _extract_matplotlib_result(pc, return_as_pc)
```

- [ ] **Step 5: Run all tests to confirm they pass**

```
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py -v
```

Expected: all 22 tests pass (3 init + 7 predictions + 6 param_stability + 6 crps).

- [ ] **Step 6: Run pre-commit**

```
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
```

- [ ] **Step 7: Commit**

```bash
git add pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
git commit -m "feat(mmm): implement MMMCVPlotSuite.crps() and CRPS helper functions"
```

---

## Task 5: Wire up — exports and integration point

**Files:**
- Modify: `pymc_marketing/mmm/plotting/__init__.py` (add `MMMCVPlotSuite`)
- Modify: `pymc_marketing/mmm/time_slice_cross_validation.py` (update `.plot` property)

- [ ] **Step 1: Update `pymc_marketing/mmm/plotting/__init__.py`**

Current content of the file:
```python
from pymc_marketing.mmm.plotting.decomposition import DecompositionPlots
from pymc_marketing.mmm.plotting.diagnostics import DiagnosticsPlots
from pymc_marketing.mmm.plotting.sensitivity import SensitivityPlots
from pymc_marketing.mmm.plotting.transformations import TransformationPlots

__all__ = [
    "DecompositionPlots",
    "DiagnosticsPlots",
    "SensitivityPlots",
    "TransformationPlots",
]
```

Replace with:

```python
from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite
from pymc_marketing.mmm.plotting.decomposition import DecompositionPlots
from pymc_marketing.mmm.plotting.diagnostics import DiagnosticsPlots
from pymc_marketing.mmm.plotting.sensitivity import SensitivityPlots
from pymc_marketing.mmm.plotting.transformations import TransformationPlots

__all__ = [
    "DecompositionPlots",
    "DiagnosticsPlots",
    "MMMCVPlotSuite",
    "SensitivityPlots",
    "TransformationPlots",
]
```

- [ ] **Step 2: Verify the export is importable**

```
conda run -n pymc-marketing-dev python -c "from pymc_marketing.mmm.plotting import MMMCVPlotSuite; print(MMMCVPlotSuite)"
```

Expected: `<class 'pymc_marketing.mmm.plotting.cv.MMMCVPlotSuite'>`

- [ ] **Step 3: Update `TimeSliceCrossValidator.plot` in `time_slice_cross_validation.py`**

Find and update the import at the top of the file. Add:
```python
from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite
```

Then replace the `plot` property (currently at lines 177–182):

Current code:
```python
@property
def plot(self) -> MMMPlotSuite:
    """Use the MMMPlotSuite to plot the results."""
    self._validate_model_was_built()
    self._validate_idata_exists()
    return MMMPlotSuite(idata=self.idata)
```

New code:
```python
@property
def plot(self) -> MMMCVPlotSuite:
    """Plotting suite for cross-validation results."""
    self._validate_model_was_built()
    return MMMCVPlotSuite(self.cv_idata)
```

The `_validate_idata_exists()` call is intentionally dropped: `MMMCVPlotSuite.__init__` calls `_validate_cv_results` which performs the same type check, and `self.cv_idata` does not exist before `run()` completes (a `_validate_model_was_built` guard is sufficient).

- [ ] **Step 4: Run the full test suite for the affected modules**

```
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py tests/mmm/test_time_slice_cross_validation.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run pre-commit on modified files**

```
conda run -n pymc-marketing-dev pre-commit run --files \
  pymc_marketing/mmm/plotting/__init__.py \
  pymc_marketing/mmm/time_slice_cross_validation.py
```

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/__init__.py pymc_marketing/mmm/time_slice_cross_validation.py
git commit -m "feat(mmm): wire MMMCVPlotSuite into plotting __init__ and TimeSliceCrossValidator.plot"
```

---

## Self-Review Checklist

Spec requirements → plan coverage:

| Spec requirement | Task |
|---|---|
| `MMMCVPlotSuite.__init__` validates type + cv_metadata | Task 1 |
| `predictions()` — HDI bands, observed line, vline per fold | Task 2 |
| `predictions()` — per-call `cv_data` override + re-validation | Task 2 |
| `predictions()` — validation raises for missing pp + cv_metadata | Task 2 |
| `param_stability()` — single `azp.plot_forest` call, no loop | Task 3 |
| `param_stability()` — `aes={"color": ["cv"]}`, `shade_label="channel"` | Task 3 |
| `param_stability()` — raises for absent `cv` coord | Task 3 |
| `crps()` — `_pred_matrix_for_rows`, `_filter_rows_and_y`, `_crps_for_split` | Task 4 |
| `crps()` — NaN tolerant (empty test fold) | Task 4 |
| `crps()` — `PlotCollection.wrap` + `aes={"color": ["split"]}` | Task 4 |
| Export from `plotting/__init__.py` | Task 5 |
| `TimeSliceCrossValidator.plot` → `MMMCVPlotSuite(self.cv_idata)` | Task 5 |

All spec bug-fix requirements are addressed:
- No `plt.show()` calls anywhere (replaced by `PlotCollection`)
- `param_stability` always returns a single figure (single `azp.plot_forest` call)
- `_extract_matplotlib_result` always wraps axes in `NDArray` (no bare `Axes`)
- All HDI logic delegated to `.azstats.hdi()` + `azp.visuals.fill_between_y`
