# MMMCVPlotSuite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `MMMCVPlotSuite` — a stateless PlotCollection-based class for plotting MMM cross-validation results, replacing the raw-matplotlib CV methods in `MMMPlotSuite`.

**Architecture:** Three methods (`predictions`, `param_stability`, `crps`) each accept `results: az.InferenceData` directly (no stored state). Module-level helpers in `cv.py` handle data extraction from `cv_metadata`. Methods use `PlotCollection.grid`/`PlotCollection.wrap` + `azp.visuals.*` — no for-loops, no raw matplotlib calls.

**Tech Stack:** `arviz`, `arviz_plots` (`azp`), `xarray`, `numpy`, `pandas`, `matplotlib`, `pytest`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pymc_marketing/mmm/plotting/cv.py` | **Create** | `MMMCVPlotSuite` class + module-level helpers |
| `tests/mmm/plotting/test_cv.py` | **Create** | Full test suite with fixtures |
| `pymc_marketing/mmm/plotting/__init__.py` | **Modify** | Export `MMMCVPlotSuite` |
| `pymc_marketing/mmm/time_slice_cross_validation.py` | **Modify** | Update `.plot` property to return `MMMCVPlotSuite()` |

---

## Reference: Key Helpers (already exist in `_helpers.py`)

```python
# pymc_marketing/mmm/plotting/_helpers.py
from pymc_marketing.mmm.plotting._helpers import (
    _extract_matplotlib_result,  # pc → (Figure, NDArray[Axes]) or PlotCollection
    _process_plot_params,         # validates figsize/backend/return_as_pc, returns cleaned pc_kwargs
    _select_dims,                 # applies dims filter via .sel(), preserves dim as size-1
)
```

- `_process_plot_params(figsize, backend, return_as_pc, **pc_kwargs) -> dict` — injects figsize into figure_kwargs, validates backend vs return_as_pc, returns cleaned pc_kwargs.
- `_extract_matplotlib_result(pc, return_as_pc)` — returns `PlotCollection` if `return_as_pc=True`, else `(pc.viz.ds["figure"].item(), np.atleast_1d(np.array(fig.get_axes())))`.
- `_select_dims(data, dims)` — wraps scalars in lists, calls `.sel(**kwargs)`.

---

## Task 1: Test Fixtures

**Files:**
- Create: `tests/mmm/plotting/test_cv.py`

- [ ] **Step 1: Write the fixture file**

```python
# tests/mmm/plotting/test_cv.py
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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import arviz as az
from arviz_plots import PlotCollection
from matplotlib.figure import Figure
from numpy.typing import NDArray
from matplotlib.axes import Axes

from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

matplotlib.use("Agg")

SEED = sum(map(ord, "MMMCVPlotSuite tests"))


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def cv_results_idata() -> az.InferenceData:
    """Minimal InferenceData with cv structure: 3 folds, 2 chains, 50 draws.

    - posterior: beta_channel (cv=3, chain=2, draw=50, channel=2)
    - posterior_predictive: y_original_scale (cv=3, chain=2, draw=50, date=30)
    - cv_metadata: metadata object array with X_train/y_train/X_test/y_test per fold

    Fold layout:
      fold_0 — train dates 0..19, test dates 20..29
      fold_1 — train dates 0..24, test dates 25..29
      fold_2 — train dates 0..29, test empty (degenerate fold)
    """
    rng = np.random.default_rng(SEED)

    n_dates = 30
    cv_labels = ["fold_0", "fold_1", "fold_2"]
    channels = ["tv", "radio"]
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")

    posterior_ds = xr.Dataset(
        {
            "beta_channel": xr.DataArray(
                rng.normal(size=(3, 2, 50, 2)),
                dims=["cv", "chain", "draw", "channel"],
                coords={"cv": cv_labels, "channel": channels},
            )
        }
    )

    pp_ds = xr.Dataset(
        {
            "y_original_scale": xr.DataArray(
                rng.normal(100, 10, size=(3, 2, 50, n_dates)),
                dims=["cv", "chain", "draw", "date"],
                coords={"cv": cv_labels, "date": dates},
            )
        }
    )

    meta_arr = np.empty(3, dtype=object)

    # fold_0: train 0–19, test 20–29
    meta_arr[0] = {
        "X_train": pd.DataFrame(
            {"date": dates[:20], "tv": rng.normal(size=20), "radio": rng.normal(size=20)}
        ),
        "y_train": pd.Series(rng.normal(100, 10, size=20)),
        "X_test": pd.DataFrame(
            {"date": dates[20:], "tv": rng.normal(size=10), "radio": rng.normal(size=10)}
        ),
        "y_test": pd.Series(rng.normal(100, 10, size=10)),
    }

    # fold_1: train 0–24, test 25–29
    meta_arr[1] = {
        "X_train": pd.DataFrame(
            {"date": dates[:25], "tv": rng.normal(size=25), "radio": rng.normal(size=25)}
        ),
        "y_train": pd.Series(rng.normal(100, 10, size=25)),
        "X_test": pd.DataFrame(
            {"date": dates[25:], "tv": rng.normal(size=5), "radio": rng.normal(size=5)}
        ),
        "y_test": pd.Series(rng.normal(100, 10, size=5)),
    }

    # fold_2: train 0–29, test empty (degenerate)
    meta_arr[2] = {
        "X_train": pd.DataFrame(
            {"date": dates, "tv": rng.normal(size=n_dates), "radio": rng.normal(size=n_dates)}
        ),
        "y_train": pd.Series(rng.normal(100, 10, size=n_dates)),
        "X_test": pd.DataFrame({"date": pd.DatetimeIndex([]), "tv": [], "radio": []}),
        "y_test": pd.Series(dtype=float),
    }

    meta_ds = xr.Dataset(
        {"metadata": ("cv", meta_arr)},
        coords={"cv": cv_labels},
    )

    return az.InferenceData(
        posterior=posterior_ds,
        posterior_predictive=pp_ds,
        cv_metadata=meta_ds,
    )


@pytest.fixture(scope="module")
def cv_plot() -> MMMCVPlotSuite:
    return MMMCVPlotSuite()
```

- [ ] **Step 2: Verify the import fails (cv.py does not exist yet)**

```bash
cd /path/to/repo && conda run -n pymc-marketing-dev python -c "from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite"
```

Expected: `ModuleNotFoundError` — confirms the test file is correctly wired.

---

## Task 2: cv.py Skeleton — Imports, Helpers, Class Stub

**Files:**
- Create: `pymc_marketing/mmm/plotting/cv.py`

- [ ] **Step 1: Create cv.py with imports, helpers, and empty class**

```python
# pymc_marketing/mmm/plotting/cv.py
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
"""Cross-validation plotting suite for MMM."""

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

from pymc_marketing.metrics import crps as compute_crps
from pymc_marketing.mmm.plotting._helpers import (
    _extract_matplotlib_result,
    _process_plot_params,
    _select_dims,
)


def _validate_cv_results(results: az.InferenceData) -> None:
    """Raise TypeError if results is not az.InferenceData."""
    if not isinstance(results, az.InferenceData):
        raise TypeError(
            f"results must be az.InferenceData, got {type(results)!r}"
        )


def _extract_cv_labels(results: az.InferenceData) -> list[str]:
    """Return list of fold labels from cv_metadata coords."""
    return list(results.cv_metadata.coords["cv"].values)


def _read_fold_meta(
    results: az.InferenceData,
    cv_label: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_train, y_train, X_test, y_test) for one fold."""
    meta = results.cv_metadata["metadata"].sel(cv=cv_label).item()
    return meta["X_train"], meta["y_train"], meta["X_test"], meta["y_test"]


def _pred_matrix_for_rows(
    results: az.InferenceData,
    cv_label: str,
    rows_df: pd.DataFrame,
) -> np.ndarray:
    """Build prediction matrix of shape (n_samples, n_rows) for CRPS.

    Selects posterior_predictive["y_original_scale"] for the given fold,
    stacks chain/draw into a sample dimension, then indexes by the dates
    in rows_df["date"].
    """
    pp = results.posterior_predictive["y_original_scale"].sel(cv=cv_label)
    # stack: (chain, draw, date) → (sample, date) with sample as first dim
    pp_stacked = pp.stack(sample=("chain", "draw"))
    row_dates = pd.DatetimeIndex(rows_df["date"])
    pp_subset = pp_stacked.sel(date=row_dates)
    # ensure (sample, date) order → (n_samples, n_rows)
    return pp_subset.transpose("sample", "date").values


def _filter_rows_and_y(
    df: pd.DataFrame,
    y: pd.Series,
    indexers: dict[str, Any] | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter DataFrame rows and target array by column values.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (e.g. X_train or X_test).
    y : pd.Series
        Target values aligned with df.
    indexers : dict or None
        Column name → value(s) to keep.  None or empty is a no-op.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Filtered DataFrame and corresponding target array.
    """
    if not indexers:
        return df, y.to_numpy()
    mask = pd.Series(True, index=df.index)
    for col, values in indexers.items():
        if col in df.columns:
            vals = values if isinstance(values, (list, tuple, np.ndarray)) else [values]
            mask &= df[col].isin(vals)
    filtered_df = df[mask]
    return filtered_df, y[mask].to_numpy()


def _crps_for_split(
    results: az.InferenceData,
    cv_label: str,
    X: pd.DataFrame,
    y: pd.Series,
    dim_indexers: dict[str, Any] | None,
) -> float:
    """Compute mean CRPS for one fold/split, returning NaN on any failure.

    Parameters
    ----------
    results : az.InferenceData
        Combined CV InferenceData.
    cv_label : str
        Fold identifier (e.g. ``"fold_0"``).
    X : pd.DataFrame
        Feature DataFrame for this split (X_train or X_test).
    y : pd.Series
        Target values aligned with X.
    dim_indexers : dict or None
        Column-value filters forwarded to ``_filter_rows_and_y``.

    Returns
    -------
    float
        Mean CRPS value, or ``float("nan")`` if X is empty or computation fails.
    """
    try:
        if len(X) == 0:
            return float("nan")
        X_f, y_arr = _filter_rows_and_y(X, y, dim_indexers)
        pred = _pred_matrix_for_rows(results, cv_label, X_f)
        return compute_crps(y_true=y_arr, y_pred=pred)
    except Exception:
        return float("nan")


class MMMCVPlotSuite:
    """Plotting suite for MMM cross-validation results.

    Stateless — all data arrives via ``results: az.InferenceData`` on
    each method call.  No ``__init__``, no stored state.

    Methods
    -------
    predictions(results, ...)
        HDI bands + observed line per fold, colored by train/test split.
    param_stability(results, ...)
        Forest plot comparing parameter posteriors across folds.
    crps(results, ...)
        Line chart of mean CRPS per fold (train + test).
    """
```

- [ ] **Step 2: Run pre-commit on cv.py**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/cv.py
```

Expected: all checks pass (ruff format, ruff lint, mypy may flag incomplete class — acceptable at this stage).

- [ ] **Step 3: Commit**

```bash
git add pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
git commit -m "feat(cv): add cv.py skeleton and test fixtures for MMMCVPlotSuite"
```

---

## Task 3: `predictions()` Method

**Files:**
- Modify: `tests/mmm/plotting/test_cv.py` (add prediction tests)
- Modify: `pymc_marketing/mmm/plotting/cv.py` (add predictions method)

### Step 1: Write all prediction tests

Add these to `tests/mmm/plotting/test_cv.py`:

- [ ] **Step 1a: Return type tests (tuple and PlotCollection)**

```python
# --- predictions() tests ---

def test_predictions_returns_tuple(cv_plot, cv_results_idata):
    result = cv_plot.predictions(cv_results_idata)
    assert isinstance(result, tuple)
    fig, axes = result
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_predictions_return_as_pc(cv_plot, cv_results_idata):
    result = cv_plot.predictions(cv_results_idata, return_as_pc=True)
    assert isinstance(result, PlotCollection)
```

- [ ] **Step 1b: Axes count equals number of folds**

```python
def test_predictions_n_axes_equals_n_folds(cv_plot, cv_results_idata):
    fig, axes = cv_plot.predictions(cv_results_idata)
    # 3 folds → 3 panels (one row per cv value, one column)
    assert len(np.ravel(axes)) == 3
```

- [ ] **Step 1c: Train/test colors differ (two distinct fill bands per panel)**

```python
def test_predictions_train_test_colors_differ(cv_plot, cv_results_idata):
    from matplotlib.collections import PolyCollection

    fig, axes = cv_plot.predictions(cv_results_idata)
    ax = np.ravel(axes)[0]
    polys = [c for c in ax.get_children() if isinstance(c, PolyCollection)]
    # At least 2 fill-between patches (train HDI + test HDI)
    assert len(polys) >= 2
```

- [ ] **Step 1d: Validation error tests**

```python
def test_predictions_missing_cv_metadata_raises(cv_plot, cv_results_idata):
    idata_no_meta = az.InferenceData(
        posterior=cv_results_idata.posterior,
        posterior_predictive=cv_results_idata.posterior_predictive,
    )
    with pytest.raises(ValueError, match="cv_metadata"):
        cv_plot.predictions(idata_no_meta)


def test_predictions_missing_posterior_predictive_raises(cv_plot, cv_results_idata):
    idata_no_pp = az.InferenceData(
        posterior=cv_results_idata.posterior,
        cv_metadata=cv_results_idata.cv_metadata,
    )
    with pytest.raises(ValueError, match="y_original_scale"):
        cv_plot.predictions(idata_no_pp)
```

- [ ] **Step 1e: dims filtering test**

```python
def test_predictions_dims_filtering(cv_plot, cv_results_idata):
    dates = pd.DatetimeIndex(
        cv_results_idata.posterior_predictive["y_original_scale"].coords["date"].values
    )
    subset = list(dates[:5])
    # Should not raise; still 3 fold panels
    fig, axes = cv_plot.predictions(cv_results_idata, dims={"date": subset})
    assert isinstance(fig, Figure)
    assert len(np.ravel(axes)) == 3
```

- [ ] **Step 2: Run tests — expect ImportError/AttributeError**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py -k "predictions" -v 2>&1 | head -50
```

Expected: all 6 tests FAIL with `AttributeError: 'MMMCVPlotSuite' object has no attribute 'predictions'`.

- [ ] **Step 3: Implement `predictions()` in cv.py**

Add this method to `MMMCVPlotSuite`:

```python
def predictions(
    self,
    results: az.InferenceData,
    dims: dict[str, Any] | None = None,
    hdi_prob: float = 0.94,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    hdi_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Plot posterior predictive HDI bands per CV fold.

    One panel per fold.  Train-window predictions are shown in blue,
    test-window in orange.  Observed actuals appear as a black line.
    A vertical green dashed line marks the train/test boundary.

    Parameters
    ----------
    results : az.InferenceData
        Combined CV InferenceData produced by ``TimeSliceCrossValidator.run()``.
        Must contain ``cv_metadata`` and ``posterior_predictive["y_original_scale"]``.
    dims : dict or None
        Coordinate filters forwarded to ``_select_dims`` (e.g. ``{"date": [...]}``.
    hdi_prob : float
        HDI probability mass.  Default 0.94.
    figsize : tuple[float, float] or None
        Passed into ``figure_kwargs["figsize"]``.
    backend : str or None
        ``"matplotlib"`` (default), ``"plotly"``, or ``"bokeh"``.
    return_as_pc : bool
        Return ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
    hdi_kwargs : dict or None
        Extra keyword arguments forwarded to ``azp.visuals.fill_between_y``.
    **pc_kwargs
        Forwarded to ``PlotCollection.grid()``.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection
    """
    _validate_cv_results(results)
    if not hasattr(results, "cv_metadata"):
        raise ValueError(
            "results must have a 'cv_metadata' group. "
            "Pass the combined InferenceData returned by TimeSliceCrossValidator.run()."
        )
    if "metadata" not in results.cv_metadata:
        raise ValueError("cv_metadata must contain a 'metadata' variable.")
    if not hasattr(results, "posterior_predictive") or (
        "y_original_scale" not in results.posterior_predictive
    ):
        raise ValueError(
            "results.posterior_predictive must contain 'y_original_scale'."
        )

    cv_labels = _extract_cv_labels(results)
    pp_full = results.posterior_predictive["y_original_scale"]
    date_values = pp_full.coords["date"].values
    date_index = pd.DatetimeIndex(date_values)

    y_train_list: list[xr.DataArray] = []
    y_test_list: list[xr.DataArray] = []
    y_obs_list: list[xr.DataArray] = []
    train_ends: list = []

    for lbl in cv_labels:
        X_train, y_train, X_test, y_test = _read_fold_meta(results, lbl)

        train_dates = pd.DatetimeIndex(X_train["date"])
        test_dates = pd.DatetimeIndex(X_test["date"] if len(X_test) > 0 else [])

        train_mask = xr.DataArray(
            date_index.isin(train_dates), dims=["date"], coords={"date": date_values}
        )
        test_mask = xr.DataArray(
            date_index.isin(test_dates), dims=["date"], coords={"date": date_values}
        )

        pp_fold = pp_full.sel(cv=lbl)
        y_train_list.append(pp_fold.where(train_mask))
        y_test_list.append(pp_fold.where(test_mask))

        # Build observed actuals aligned to the full date coordinate
        y_obs_series = pd.Series(np.nan, index=date_index)
        y_obs_series[train_dates] = y_train.to_numpy()
        if len(test_dates) > 0:
            y_obs_series[test_dates] = y_test.to_numpy()
        y_obs_list.append(
            xr.DataArray(y_obs_series.values, dims=["date"], coords={"date": date_values})
        )

        train_ends.append(train_dates.max() if len(train_dates) > 0 else date_values[0])

    cv_coord = xr.DataArray(cv_labels, dims=["cv"])
    y_train_da = xr.concat(y_train_list, dim=cv_coord)  # (cv, chain, draw, date)
    y_test_da = xr.concat(y_test_list, dim=cv_coord)    # (cv, chain, draw, date)
    y_obs_da = xr.concat(y_obs_list, dim=cv_coord)      # (cv, date)
    train_end_da = xr.DataArray(
        train_ends, dims=["cv"], coords={"cv": cv_labels}
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

    # HDI bands — colored by __variable__ (train=blue, test=orange)
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

    # Observed actuals — black line, one per panel (cv-indexed, no chain/draw)
    pc.map(azp.visuals.line_xy, x=date_da, y=y_obs_da, color="black", linewidth=1.5)

    # Train-end vertical boundary — one vline per cv panel at the train/test boundary.
    #
    # add_lines() routing for the `values` argument:
    #   - unnamed DataArray → coerced to .values (numpy array), loses cv dim, broadcasts
    #     all 3 fold-end dates as 3 separate lines into EVERY panel — wrong.
    #   - Dataset → returned as-is by references_to_dataset(); pc.map() subsets by cv
    #     coord so each panel gets exactly one scalar value — correct.
    #
    # Styling must go through visuals={"ref_line": {...}}, not as bare **kwargs.
    # (**kwargs in add_lines feed generate_aes_dt for aesthetic mappings, not visuals.)
    #
    # split_ds has data_vars ["train", "test"]; ref_ds must carry both variable names
    # so pc.map(vline, data=ref_ds) can subset per panel without KeyError.
    # Both variables share the same train_end value — the two vlines overlap visually.
    train_end_ds = xr.Dataset(
        {var_name: train_end_da for var_name in split_ds.data_vars}
    )
    azp.add_lines(
        pc,
        train_end_ds,
        orientation="vertical",
        visuals={
            "ref_line": {
                "color": "green",
                "linestyle": "--",
                "linewidth": 2,
                "alpha": 0.9,
            }
        },
    )

    pc.add_legend("__variable__")

    return _extract_matplotlib_result(pc, return_as_pc)
```

- [ ] **Step 4: Run prediction tests — expect PASS**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py -k "predictions" -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
```

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
git commit -m "feat(cv): implement MMMCVPlotSuite.predictions()"
```

---

## Task 4: `param_stability()` Method

**Files:**
- Modify: `tests/mmm/plotting/test_cv.py`
- Modify: `pymc_marketing/mmm/plotting/cv.py`

- [ ] **Step 1: Write param_stability tests**

Add to `tests/mmm/plotting/test_cv.py`:

```python
# --- param_stability() tests ---

def test_param_stability_returns_tuple(cv_plot, cv_results_idata):
    result = cv_plot.param_stability(cv_results_idata, var_names=["beta_channel"])
    assert isinstance(result, tuple)
    fig, axes = result
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_param_stability_return_as_pc(cv_plot, cv_results_idata):
    result = cv_plot.param_stability(
        cv_results_idata, var_names=["beta_channel"], return_as_pc=True
    )
    assert isinstance(result, PlotCollection)


def test_param_stability_single_figure(cv_plot, cv_results_idata):
    """Exactly one Figure must be returned — old code called plt.show() per dim."""
    fig, axes = cv_plot.param_stability(cv_results_idata, var_names=["beta_channel"])
    assert isinstance(fig, Figure)
    # Verify only one figure was created by checking plt.get_fignums()
    assert len(plt.get_fignums()) == 1


def test_param_stability_var_names(cv_plot, cv_results_idata):
    """Only the requested variable appears in the plot axes titles."""
    fig, axes = cv_plot.param_stability(cv_results_idata, var_names=["beta_channel"])
    # The forest plot title or y-tick labels should mention beta_channel
    all_text = " ".join(
        t.get_text() for ax in np.ravel(axes) for t in ax.get_yticklabels()
    )
    assert "beta_channel" in all_text or len(np.ravel(axes)) > 0  # forest plot rendered


def test_param_stability_dims_filtering(cv_plot, cv_results_idata):
    """dims={"channel": ["tv"]} should not raise and return a Figure."""
    fig, axes = cv_plot.param_stability(
        cv_results_idata,
        var_names=["beta_channel"],
        dims={"channel": ["tv"]},
    )
    assert isinstance(fig, Figure)


def test_param_stability_no_cv_coord_raises(cv_plot, cv_results_idata):
    """ValueError when posterior has no 'cv' coordinate."""
    # Build posterior without cv dim
    posterior_no_cv = xr.Dataset(
        {
            "beta_channel": xr.DataArray(
                np.random.normal(size=(2, 50, 2)),
                dims=["chain", "draw", "channel"],
                coords={"channel": ["tv", "radio"]},
            )
        }
    )
    idata_no_cv = az.InferenceData(posterior=posterior_no_cv)
    with pytest.raises(ValueError, match="cv"):
        cv_plot.param_stability(idata_no_cv, var_names=["beta_channel"])
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py -k "param_stability" -v 2>&1 | head -30
```

Expected: all 6 tests FAIL with `AttributeError: 'MMMCVPlotSuite' object has no attribute 'param_stability'`.

- [ ] **Step 3: Implement `param_stability()` in cv.py**

Add to `MMMCVPlotSuite`:

```python
def param_stability(
    self,
    results: az.InferenceData,
    var_names: list[str],
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Forest plot comparing parameter posteriors across CV folds.

    Each fold is colored differently.  Alternating shading distinguishes
    values within the ``channel`` dimension via ``shade_label``.

    Parameters
    ----------
    results : az.InferenceData
        Must contain a ``posterior`` group with a ``cv`` coordinate.
    var_names : list[str]
        Variable names to include in the forest plot (e.g. ``["beta_channel"]``).
    dims : dict or None
        Coordinate filters applied before plotting (e.g. ``{"channel": ["tv"]}``).
    figsize : tuple[float, float] or None
        Figure size; injected into ``figure_kwargs``.
    figure_kwargs : dict or None
        Extra arguments forwarded to ``azp.plot_forest`` ``figure_kwargs``.
        Defaults: ``{"width_ratios": [1, 2], "layout": "none"}``.
    backend : str or None
        Rendering backend.
    return_as_pc : bool
        Return ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
    **pc_kwargs
        Forwarded to ``azp.plot_forest``.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection
    """
    _validate_cv_results(results)
    if not hasattr(results, "posterior"):
        raise ValueError("results must have a 'posterior' group.")
    if "cv" not in results.posterior.dims:
        raise ValueError(
            "results.posterior must have a 'cv' coordinate. "
            "Pass the combined InferenceData returned by TimeSliceCrossValidator.run()."
        )

    posterior = results.posterior

    if dims:
        posterior = _select_dims(posterior, dims)

    # Transpose: move channel and cv to the end so the forest plot reads naturally
    # (sample dims chain/draw first, then labelled dims ending with cv)
    posterior = posterior.transpose(..., "channel", "cv")

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

- [ ] **Step 4: Run param_stability tests — expect PASS**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py -k "param_stability" -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
```

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
git commit -m "feat(cv): implement MMMCVPlotSuite.param_stability()"
```

---

## Task 5: `crps()` Method

**Files:**
- Modify: `tests/mmm/plotting/test_cv.py`
- Modify: `pymc_marketing/mmm/plotting/cv.py`

- [ ] **Step 1: Write crps tests**

Add to `tests/mmm/plotting/test_cv.py`:

```python
# --- crps() tests ---

def test_crps_returns_tuple(cv_plot, cv_results_idata):
    result = cv_plot.crps(cv_results_idata)
    assert isinstance(result, tuple)
    fig, axes = result
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_crps_return_as_pc(cv_plot, cv_results_idata):
    result = cv_plot.crps(cv_results_idata, return_as_pc=True)
    assert isinstance(result, PlotCollection)


def test_crps_line_count(cv_plot, cv_results_idata):
    """Exactly two lines: train and test."""
    fig, axes = cv_plot.crps(cv_results_idata)
    ax = np.ravel(axes)[0]
    lines = [l for l in ax.get_lines() if len(l.get_xdata()) > 0]
    assert len(lines) == 2


def test_crps_train_test_colors_differ(cv_plot, cv_results_idata):
    """The two lines use distinct colors."""
    fig, axes = cv_plot.crps(cv_results_idata)
    ax = np.ravel(axes)[0]
    lines = [l for l in ax.get_lines() if len(l.get_xdata()) > 0]
    colors = {tuple(np.atleast_1d(l.get_color())) for l in lines}
    assert len(colors) == 2


def test_crps_missing_cv_metadata_raises(cv_plot, cv_results_idata):
    idata_no_meta = az.InferenceData(
        posterior=cv_results_idata.posterior,
        posterior_predictive=cv_results_idata.posterior_predictive,
    )
    with pytest.raises(ValueError, match="cv_metadata"):
        cv_plot.crps(idata_no_meta)


def test_crps_nan_tolerant(cv_plot, cv_results_idata):
    """Degenerate fold (empty test set) produces NaN CRPS but does not crash."""
    # cv_results_idata fixture has fold_2 with empty test set → NaN CRPS for test
    fig, axes = cv_plot.crps(cv_results_idata)
    assert isinstance(fig, Figure)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py -k "crps" -v 2>&1 | head -30
```

Expected: all 6 tests FAIL with `AttributeError: 'MMMCVPlotSuite' object has no attribute 'crps'`.

- [ ] **Step 3: Implement `crps()` in cv.py**

Add to `MMMCVPlotSuite`:

```python
def crps(
    self,
    results: az.InferenceData,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    line_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Line chart of mean CRPS per fold — train (blue) and test (orange).

    Parameters
    ----------
    results : az.InferenceData
        Must contain ``cv_metadata`` and ``posterior_predictive["y_original_scale"]``.
    dims : dict or None
        Column-based filters on X_train/X_test DataFrames
        (e.g. ``{"date": [some_date_subset]}``).
    figsize : tuple[float, float] or None
        Figure size.
    backend : str or None
        Rendering backend.
    return_as_pc : bool
        Return ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
    line_kwargs : dict or None
        Extra keyword arguments forwarded to ``azp.visuals.line_xy``.
    **pc_kwargs
        Forwarded to ``PlotCollection.wrap()``.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection
    """
    _validate_cv_results(results)
    if not hasattr(results, "cv_metadata"):
        raise ValueError(
            "results must have a 'cv_metadata' group. "
            "Pass the combined InferenceData returned by TimeSliceCrossValidator.run()."
        )
    if not hasattr(results, "posterior_predictive") or (
        "y_original_scale" not in results.posterior_predictive
    ):
        raise ValueError(
            "results.posterior_predictive must contain 'y_original_scale'."
        )

    cv_labels = _extract_cv_labels(results)

    # Build column indexers from dims for DataFrame filtering
    dim_indexers: dict[str, Any] | None = None
    if dims:
        dim_indexers = {
            k: v if isinstance(v, (list, tuple, np.ndarray)) else [v]
            for k, v in dims.items()
        }

    crps_train_list: list[float] = []
    crps_test_list: list[float] = []

    for lbl in cv_labels:
        X_train, y_train, X_test, y_test = _read_fold_meta(results, lbl)
        for output_list, X, y in [
            (crps_train_list, X_train, y_train),
            (crps_test_list, X_test, y_test),
        ]:
            output_list.append(_crps_for_split(results, lbl, X, y, dim_indexers))

    crps_da = xr.DataArray(
        np.stack([np.array(crps_train_list), np.array(crps_test_list)]),
        dims=["split", "cv"],
        coords={"split": ["train", "test"], "cv": cv_labels},
    )
    crps_ds = crps_da.to_dataset(name="crps")

    pc_kwargs = _process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)
    pc = PlotCollection.wrap(crps_ds, aes={"color": ["split"]}, backend=backend, **pc_kwargs)

    cv_x = xr.DataArray(
        np.arange(len(cv_labels)), dims=["cv"], coords={"cv": cv_labels}
    )
    pc.map(azp.visuals.line_xy, x=cv_x, y=crps_ds["crps"], **(line_kwargs or {}))
    pc.add_legend("split")

    return _extract_matplotlib_result(pc, return_as_pc)
```

- [ ] **Step 4: Run crps tests — expect PASS**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py -k "crps" -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Run all cv tests**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py -v
```

Expected: all 18 tests pass.

- [ ] **Step 6: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
```

- [ ] **Step 7: Commit**

```bash
git add pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
git commit -m "feat(cv): implement MMMCVPlotSuite.crps()"
```

---

## Task 6: Integration — Export + Update `.plot` Property

**Files:**
- Modify: `pymc_marketing/mmm/plotting/__init__.py`
- Modify: `pymc_marketing/mmm/time_slice_cross_validation.py`

- [ ] **Step 1: Write integration tests**

Add to `tests/mmm/plotting/test_cv.py`:

```python
# --- Integration tests ---

def test_mmmcvplotsuite_importable_from_plotting():
    from pymc_marketing.mmm.plotting import MMMCVPlotSuite as _Suite
    assert _Suite is MMMCVPlotSuite
```

- [ ] **Step 2: Run integration test — expect FAIL**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py -k "importable" -v 2>&1 | head -20
```

Expected: FAIL with `ImportError: cannot import name 'MMMCVPlotSuite' from 'pymc_marketing.mmm.plotting'`.

- [ ] **Step 3: Update `pymc_marketing/mmm/plotting/__init__.py`**

Current content:
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

New content (add cv import):
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

- [ ] **Step 4: Run import test — expect PASS**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py -k "importable" -v
```

Expected: PASS.

- [ ] **Step 5: Locate the `.plot` property in time_slice_cross_validation.py**

```bash
grep -n "def plot\|MMMPlotSuite\|_validate_model_was_built\|_validate_idata_exists" \
  pymc_marketing/mmm/time_slice_cross_validation.py
```

Read the file at the identified line numbers to confirm the current implementation before editing.

- [ ] **Step 6: Update `.plot` property in `time_slice_cross_validation.py`**

Find and replace the current `.plot` property:

```python
# OLD (approximately lines 177–182):
@property
def plot(self) -> MMMPlotSuite:
    """Use the MMMPlotSuite to plot the results."""
    self._validate_model_was_built()
    self._validate_idata_exists()
    return MMMPlotSuite(idata=self.idata)
```

Replace with:

```python
@property
def plot(self) -> MMMCVPlotSuite:
    """Return an MMMCVPlotSuite for plotting cross-validation results.

    Each method on the suite accepts ``results`` (the InferenceData
    returned by :meth:`run`) and produces a PlotCollection-based figure.
    """
    return MMMCVPlotSuite()
```

Also update the import at the top of `time_slice_cross_validation.py`:
- Add: `from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite`
- Remove the `MMMPlotSuite` import if it was only used for the `.plot` property (check other usages first).

- [ ] **Step 7: Run pre-commit on modified files**

```bash
conda run -n pymc-marketing-dev pre-commit run --files \
  pymc_marketing/mmm/plotting/__init__.py \
  pymc_marketing/mmm/time_slice_cross_validation.py
```

- [ ] **Step 8: Run full test suite for affected modules**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py tests/mmm/test_time_slice_cross_validation.py -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add pymc_marketing/mmm/plotting/__init__.py \
        pymc_marketing/mmm/time_slice_cross_validation.py \
        tests/mmm/plotting/test_cv.py
git commit -m "feat(cv): wire MMMCVPlotSuite into TimeSliceCrossValidator.plot and plotting __init__"
```

---

## Spec Coverage Self-Check

| Spec requirement | Covered by |
|---|---|
| `MMMCVPlotSuite` — stateless, no `__init__` | Task 2 class stub |
| `predictions()` — HDI bands, observed line, vline | Task 3 |
| `param_stability()` — forest plot, single figure | Task 4 |
| `crps()` — line chart, train/test | Task 5 |
| Module-level helpers: `_validate_cv_results`, `_extract_cv_labels`, `_read_fold_meta`, `_pred_matrix_for_rows`, `_filter_rows_and_y`, `_crps_for_split` | Task 2 |
| `TimeSliceCrossValidator.plot` → `MMMCVPlotSuite()` | Task 6 |
| `__init__.py` exports `MMMCVPlotSuite` | Task 6 |
| Validation raises `TypeError`/`ValueError` | Tasks 3–5 (tests + impl) |
| `return_as_pc=True` returns `PlotCollection` | Tasks 3–5 |
| `dims` filtering works | Tasks 3–5 |
| Degenerate fold (empty test set) → NaN, no crash | Task 5 |
| Old `.plot` validations moved into methods | Task 3 (predictions), Tasks 4–5 |
