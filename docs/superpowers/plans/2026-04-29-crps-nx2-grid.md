# CRPS n×2 Grid Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `MMMCVPlotSuite.crps()` from a single panel with two lines to an n×2 grid where left=train, right=test, rows=Cartesian combinations of extra dimensions (e.g. one row per geo).

**Architecture:** Detect extra dims from `y_original_scale`, compute per-(split, combo, fold) CRPS into a multi-dim DataArray, and render with `PlotCollection.grid(rows=[*extra_dims], cols=["split"])`. When no extra dims exist the result is a 1×2 grid; existing `_crps_for_split` and `_filter_rows_and_y` helpers are unchanged.

**Tech Stack:** Python, xarray, arviz-plots `PlotCollection.grid`, matplotlib, itertools, numpy

---

## Files

| File | Change |
|---|---|
| `pymc_marketing/mmm/plotting/cv.py` | Add `import itertools`; rewrite `crps()` body |
| `tests/mmm/plotting/test_cv.py` | Update 3 tests in `TestCRPS` |

---

## Task 1: Update the three affected tests to match the new n×2 layout

**Files:**
- Modify: `tests/mmm/plotting/test_cv.py:350-410`

- [ ] **Step 1: Replace `test_line_count`**

In `tests/mmm/plotting/test_cv.py`, find and replace the existing `test_line_count` method (currently asserts 2 lines on `axes[0]`):

```python
def test_line_count(self, cv_plot):
    # 1×2 grid: left panel = train, right panel = test; one line each
    _fig, axes = cv_plot.crps()
    assert len(axes) == 2
    assert len(axes[0].lines) == 1
    assert len(axes[1].lines) == 1
```

- [ ] **Step 2: Replace `test_train_test_colors_differ`**

Find and replace `test_train_test_colors_differ` in `TestCRPS`:

```python
def test_train_test_colors_differ(self, cv_plot):
    _fig, axes = cv_plot.crps()
    colors = {ax.lines[0].get_color() for ax in axes}
    assert len(colors) == 2, "Expected train and test panels in distinct colors"
```

- [ ] **Step 3: Replace `test_crps_multidim_geo_no_nan`**

Find and replace `test_crps_multidim_geo_no_nan`:

```python
def test_crps_multidim_geo_no_nan(self, cv_results_idata_geo):
    """crps() must produce finite values for multidimensional models.

    2 geos × 2 splits = 4 panels; each panel must have exactly one line
    and at least one finite CRPS value (fold_1 test is legitimately NaN
    because it has no test rows, but fold_0 test is finite).

    Regression: _pred_matrix_for_rows only selected by 'date', returning a
    2-D array (n_samples, n_geo) per observation. This caused every CRPS
    computation to fail silently, leaving all scores as NaN.
    """
    import warnings

    from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

    suite = MMMCVPlotSuite(cv_results_idata_geo)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _fig, axes = suite.crps()  # must not warn about failed CRPS
    assert len(axes) == 4, "Expected 2 geos × 2 splits = 4 panels"
    for ax in axes:
        assert len(ax.lines) == 1
        y_vals = ax.lines[0].get_ydata()
        assert np.any(np.isfinite(y_vals)), (
            "CRPS panel contains only NaN — CRPS computation failed"
        )
```

- [ ] **Step 4: Run the three updated tests to confirm they fail with the current implementation**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py::TestCRPS::test_line_count tests/mmm/plotting/test_cv.py::TestCRPS::test_train_test_colors_differ tests/mmm/plotting/test_cv.py::TestCRPS::test_crps_multidim_geo_no_nan -v
```

Expected: all 3 FAIL (current code produces 1 panel with 2 lines, not a 2- or 4-panel grid).

---

## Task 2: Implement the new `crps()` in `cv.py`

**Files:**
- Modify: `pymc_marketing/mmm/plotting/cv.py:16-20` (imports)
- Modify: `pymc_marketing/mmm/plotting/cv.py:464-544` (crps method body)

- [ ] **Step 1: Add `import itertools` to the module imports**

In `pymc_marketing/mmm/plotting/cv.py`, add `import itertools` after `from __future__ import annotations`:

```python
from __future__ import annotations

import itertools
import warnings
from typing import Any
```

- [ ] **Step 2: Replace the `crps()` method body**

Replace everything from the line `cv_labels = _extract_cv_labels(data)` (currently line ~512) through `return _extract_matplotlib_result(pc, return_as_pc)` (end of method) with the following. Keep the method signature and docstring unchanged; only the body from the first statement after the `posterior_predictive` validation check changes:

```python
    pp = data.posterior_predictive["y_original_scale"]
    cv_labels = _extract_cv_labels(data)

    base_dims = {"cv", "chain", "draw", "date"}
    extra_dims = [d for d in pp.dims if d not in base_dims]

    combo_coords: dict[str, list[Any]] = {
        d: (list(dims[d]) if dims and d in dims else list(pp.coords[d].values))
        for d in extra_dims
    }
    combos = list(itertools.product(*combo_coords.values()))
    combo_shape = [len(v) for v in combo_coords.values()]
    data_arr = np.full((2, *combo_shape, len(cv_labels)), np.nan)

    for flat_idx, combo in enumerate(combos):
        dim_indexers = dict(zip(extra_dims, combo))
        multi_idx = (
            tuple(np.unravel_index(flat_idx, combo_shape)) if combo_shape else ()
        )
        for fold_idx, lbl in enumerate(cv_labels):
            X_train, y_train, X_test, y_test = _read_fold_meta(data, lbl)
            data_arr[(0, *multi_idx, fold_idx)] = _crps_for_split(
                data, lbl, X_train, y_train, dim_indexers
            )
            data_arr[(1, *multi_idx, fold_idx)] = _crps_for_split(
                data, lbl, X_test, y_test, dim_indexers
            )

    coords: dict[str, Any] = {
        "split": ["train", "test"],
        **combo_coords,
        "cv": cv_labels,
    }
    crps_da = xr.DataArray(
        data_arr, dims=["split", *extra_dims, "cv"], coords=coords
    )
    crps_ds = crps_da.to_dataset(name="crps")

    pc_kwargs = _process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)
    pc = PlotCollection.grid(
        crps_ds,
        rows=[*extra_dims],
        cols=["split"],
        aes={"color": ["split"]},
        backend=backend,
        **pc_kwargs,
    )

    cv_x = xr.DataArray(
        np.arange(len(cv_labels)), dims=["cv"], coords={"cv": cv_labels}
    )
    pc.map(azp.visuals.line_xy, x=cv_x, y=crps_ds["crps"], **(line_kwargs or {}))
    pc.add_legend("split")

    return _extract_matplotlib_result(pc, return_as_pc)
```

The complete new `crps()` method (signature + docstring + new body) should look like:

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

    Renders an n×2 grid: left column = train CRPS, right column = test CRPS,
    one row per Cartesian combination of extra dimensions in
    ``y_original_scale`` (e.g. one row per geo). When no extra dimensions
    are present the result is a 1×2 grid.

    Parameters
    ----------
    cv_data : az.InferenceData or None
        Override the stored ``self.cv_data`` for this call only.
    dims : dict or None
        Filters which coordinate values of extra dimensions appear as rows
        (e.g. ``{"geo": ["geo_b"]}`` → only geo_b row).
        Non-extra-dim keys are silently ignored.
    figsize : tuple or None
        Figure size in inches.
    backend : str or None
        PlotCollection backend.
    return_as_pc : bool
        Return the raw ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
    line_kwargs : dict or None
        Extra kwargs forwarded to ``azp.visuals.line_xy``.
    **pc_kwargs
        Forwarded to ``PlotCollection.grid()``.

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

    pp = data.posterior_predictive["y_original_scale"]
    cv_labels = _extract_cv_labels(data)

    base_dims = {"cv", "chain", "draw", "date"}
    extra_dims = [d for d in pp.dims if d not in base_dims]

    combo_coords: dict[str, list[Any]] = {
        d: (list(dims[d]) if dims and d in dims else list(pp.coords[d].values))
        for d in extra_dims
    }
    combos = list(itertools.product(*combo_coords.values()))
    combo_shape = [len(v) for v in combo_coords.values()]
    data_arr = np.full((2, *combo_shape, len(cv_labels)), np.nan)

    for flat_idx, combo in enumerate(combos):
        dim_indexers = dict(zip(extra_dims, combo))
        multi_idx = (
            tuple(np.unravel_index(flat_idx, combo_shape)) if combo_shape else ()
        )
        for fold_idx, lbl in enumerate(cv_labels):
            X_train, y_train, X_test, y_test = _read_fold_meta(data, lbl)
            data_arr[(0, *multi_idx, fold_idx)] = _crps_for_split(
                data, lbl, X_train, y_train, dim_indexers
            )
            data_arr[(1, *multi_idx, fold_idx)] = _crps_for_split(
                data, lbl, X_test, y_test, dim_indexers
            )

    coords: dict[str, Any] = {
        "split": ["train", "test"],
        **combo_coords,
        "cv": cv_labels,
    }
    crps_da = xr.DataArray(
        data_arr, dims=["split", *extra_dims, "cv"], coords=coords
    )
    crps_ds = crps_da.to_dataset(name="crps")

    pc_kwargs = _process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)
    pc = PlotCollection.grid(
        crps_ds,
        rows=[*extra_dims],
        cols=["split"],
        aes={"color": ["split"]},
        backend=backend,
        **pc_kwargs,
    )

    cv_x = xr.DataArray(
        np.arange(len(cv_labels)), dims=["cv"], coords={"cv": cv_labels}
    )
    pc.map(azp.visuals.line_xy, x=cv_x, y=crps_ds["crps"], **(line_kwargs or {}))
    pc.add_legend("split")

    return _extract_matplotlib_result(pc, return_as_pc)
```

- [ ] **Step 3: Run pre-commit on both changed files**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
```

Expected: all hooks pass (ruff, mypy, trailing whitespace, etc.). Fix any issues before continuing.

- [ ] **Step 4: Run the full `TestCRPS` suite to confirm all tests pass**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py::TestCRPS -v
```

Expected output (all PASS):
```
PASSED tests/mmm/plotting/test_cv.py::TestCRPS::test_returns_tuple
PASSED tests/mmm/plotting/test_cv.py::TestCRPS::test_return_as_pc
PASSED tests/mmm/plotting/test_cv.py::TestCRPS::test_line_count
PASSED tests/mmm/plotting/test_cv.py::TestCRPS::test_train_test_colors_differ
PASSED tests/mmm/plotting/test_cv.py::TestCRPS::test_missing_cv_metadata_raises
PASSED tests/mmm/plotting/test_cv.py::TestCRPS::test_nan_tolerant
PASSED tests/mmm/plotting/test_cv.py::TestCRPS::test_crps_multidim_geo_no_nan
```

- [ ] **Step 5: Run the full test file to check for regressions**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_cv.py -v
```

Expected: all tests pass (including `TestInit`, `TestPredictions`, `TestParamStability`).

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/cv.py tests/mmm/plotting/test_cv.py
git commit -m "feat(mmm): change crps() to n×2 grid (left=train, right=test, rows=extra-dims)"
```
