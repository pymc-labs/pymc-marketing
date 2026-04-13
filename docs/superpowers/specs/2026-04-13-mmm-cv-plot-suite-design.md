# MMMCVPlotSuite — Design Spec

**PR:** 8 of the MMMPlotSuite v2 series
**Date:** 2026-04-13
**Branch:** `isofer/add-MMMCVPlotSuite`

---

## Overview

Introduces `MMMCVPlotSuite`: a stateless, standalone class that provides
PlotCollection-based plotting for cross-validation results produced by
`TimeSliceCrossValidator.run()`.

Three methods — `predictions`, `param_stability`, `crps` — are lifted from
the monolithic `MMMPlotSuite` and rewritten as full PlotCollection-native
implementations. All raw-matplotlib code in the old methods is replaced.

---

## Files Changed

| File | Change |
|------|--------|
| `pymc_marketing/mmm/plotting/cv.py` | **New** — `MMMCVPlotSuite` class |
| `pymc_marketing/mmm/time_slice_cross_validation.py` | Simplify `.plot` property to return `MMMCVPlotSuite()` |
| `pymc_marketing/mmm/plotting/__init__.py` | Export `MMMCVPlotSuite` |
| `tests/mmm/plotting/test_cv.py` | **New** — full test suite |

---

## Class Structure

```
MMMCVPlotSuite
  ├── predictions(results, ...)   — HDI bands, observed line, train-end vline
  ├── param_stability(results, ...)  — arviz_plots.plot_forest via DataTree dict
  └── crps(results, ...)          — scalar CRPS line chart per fold
```

No `__init__` — all data arrives via `results: az.InferenceData` on each method.
No `self._data`, no stored state.

---

## Integration Point

```python
# pymc_marketing/mmm/time_slice_cross_validation.py
class TimeSliceCrossValidator:
    @property
    def plot(self) -> MMMCVPlotSuite:
        return MMMCVPlotSuite()
```

The old property called `_validate_model_was_built()` and `_validate_idata_exists()`.
These validations move into each method (which validates `results` at call time).
The property itself becomes a zero-argument factory — no validation, no state.

---

## Method: `predictions()`

### What it renders

For each CV fold (one panel per fold): posterior predictive HDI in **blue** for
train dates and **orange** for test dates, with observed actuals in black and a
vertical green dashed line at the train/test boundary.

### API name change

Old: `MMMPlotSuite.cv_predictions(results, dims)`
New: `MMMCVPlotSuite.predictions(results, dims, hdi_prob, figsize, backend, return_as_pc, hdi_kwargs, **pc_kwargs)`

### Signature

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
```

### Input validation

Raise `TypeError` if `results` is not `az.InferenceData`.
Raise `ValueError` if:
- `cv_metadata` group absent or has no `"metadata"` variable
- `posterior_predictive["y_original_scale"]` absent

### Data preparation

1. Extract `cv_labels` from `results.cv_metadata.coords["cv"].values`.
2. Extract the full date coordinate from `results.posterior_predictive["y_original_scale"]`.
3. For each fold `lbl`:
   a. Read `X_train`, `y_train`, `X_test`, `y_test` from `cv_metadata["metadata"].sel(cv=lbl).item()`.
   b. Compute `train_dates` and `test_dates` as `pd.DatetimeIndex` from `X_train["date"]` / `X_test["date"]`.
   c. Build a boolean mask over the full date coordinate:
      - `train_mask`: True where date ∈ `train_dates`
      - `test_mask`: True where date ∈ `test_dates`
   d. From `pp = results.posterior_predictive["y_original_scale"].sel(cv=lbl)`:
      - `y_train_fold = pp.where(train_mask)` — NaN outside train window
      - `y_test_fold = pp.where(test_mask)` — NaN outside test window
4. Stack folds back along `cv`: `y_train_da` and `y_test_da` each have dims
   `(cv, chain, draw, date, ...)`. The `cv` coordinate must use `str(lbl)` for
   all labels so that `pc.viz.ds["plot"].sel(cv=str(lbl))` works in the
   post-processing step.
5. Apply `_select_dims` for any user-supplied `dims` (on both DataArrays).
6. Build `split_ds = xr.Dataset({"train": y_train_da, "test": y_test_da})`.

### PlotCollection rendering

```python
pc_kwargs = _process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)
cols = pc_kwargs.pop("cols", ["cv"])

pc = PlotCollection.grid(
    split_ds,
    cols=cols,
    aes={"color": ["__variable__"]},  # train → blue, test → orange
    backend=backend,
    **pc_kwargs,
)

# HDI bands (one per variable per panel, colored by __variable__)
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
pc.add_legend("__variable__")
```

### Post-processing (per-fold matplotlib)

After building the PC, iterate over folds to add observed actuals and the
train-end boundary. These require per-fold x-values and cannot be expressed
as a single scalar for `azp.add_lines`.

```python
for cv_label in cv_labels:
    ax = pc.viz.ds["plot"].sel(cv=str(cv_label)).item()
    # Observed actuals (black line, train then test)
    ax.plot(train_dates, y_train_vals, color="black", linewidth=1.5, label="observed")
    ax.plot(test_dates, y_test_vals, color="black", linewidth=1.5)
    # Train-end boundary
    ax.axvline(train_end_date, color="green", linestyle="--", linewidth=2, alpha=0.9, label="train end")
```

`y_train_vals` and `y_test_vals` are the observed series from `cv_metadata`,
aligned to the fold's dates.

### Returns

`tuple[Figure, NDArray[Axes]]` (default) or `PlotCollection` (`return_as_pc=True`).

---

## Method: `param_stability()`

### What it renders

A single forest plot comparing parameter posterior distributions across all CV
folds, using `arviz_plots.plot_forest` with one DataTree per fold.

### API change

Old: `MMMPlotSuite.param_stability(results, parameter, dims)`
New: `MMMCVPlotSuite.param_stability(results, var_names, dims, combined, figsize, backend, return_as_pc, **pc_kwargs)`

Changes:
- `parameter: list[str]` → `var_names: list[str]` (consistent with arviz naming)
- Single figure — old code looped per dim value calling `plt.show()`, producing
  multiple figures. New code produces one figure with all information.
- Returns `tuple[Figure, NDArray[Axes]]` — old code sometimes returned bare `Axes`.

### Signature

```python
def param_stability(
    self,
    results: az.InferenceData,
    var_names: list[str],
    dims: dict[str, Any] | None = None,
    combined: bool = True,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
```

### Input validation

Raise `TypeError` if `results` is not `az.InferenceData`.
Raise `ValueError` if:
- `results` has no `posterior` group (`not hasattr(results, "posterior")`)
- No `"cv"` coordinate found in `results.posterior`

### Implementation

```python
pc_kwargs = _process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)
cv_labels = list(results.posterior.coords["cv"].values)

dt_dict: dict[str, DataTree] = {}
for lbl in cv_labels:
    fold_ds = results.posterior.sel(cv=lbl, drop=True)
    if dims:
        fold_ds = _select_dims(fold_ds, dims)
    dt_dict[str(lbl)] = az.InferenceData(posterior=fold_ds).to_datatree()

pc = azp.plot_forest(
    dt_dict,
    var_names=var_names,
    combined=combined,
    backend=backend,
    **pc_kwargs,
)
return _extract_matplotlib_result(pc, return_as_pc)
```

`_select_dims` filters each fold's Dataset before it becomes a DataTree, so
`dims` narrows which coordinate values are shown in the forest plot.

### Returns

`tuple[Figure, NDArray[Axes]]` or `PlotCollection`.

---

## Method: `crps()`

### What it renders

A line chart of mean CRPS per fold, with train (blue) and test (orange) as two
colored lines on a single panel. X-axis = fold index, Y-axis = CRPS score.

### API name change

Old: `MMMPlotSuite.cv_crps(results, dims)`
New: `MMMCVPlotSuite.crps(results, dims, figsize, backend, return_as_pc, line_kwargs, **pc_kwargs)`

### Signature

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
```

### Input validation

Raise `TypeError` if `results` is not `az.InferenceData`.
Raise `ValueError` if:
- `cv_metadata` group absent
- `posterior_predictive["y_original_scale"]` absent

### CRPS computation

The computation is identical to the old `cv_crps` logic:

1. Extract `cv_labels`.
2. Define `_pred_matrix_for_rows(results, cv_label, rows_df) → np.ndarray` (shape `(n_samples, n_rows)`) — builds a prediction matrix by selecting `posterior_predictive["y_original_scale"].sel(cv=cv_label)` stacked over `(chain, draw)` and indexing by date.
3. Define `_filter_rows_and_y(df, y, indexers) → (filtered_df, y_arr)` — filters `X_train`/`X_test` rows matching the `dims` coordinate filters.
4. For each fold:
   - Read `X_train`, `y_train`, `X_test`, `y_test` from `cv_metadata`.
   - Apply `_filter_rows_and_y` using `dims` indexers.
   - Compute `crps(y_true=y_train_arr, y_pred=y_pred_train)` from `pymc_marketing.metrics`.
   - Append to `crps_train_list` / `crps_test_list` (NaN on failure).

Both helpers are module-level functions in `cv.py`.

### PlotCollection rendering

```python
crps_da = xr.DataArray(
    np.stack([np.array(crps_train_list), np.array(crps_test_list)]),
    dims=["split", "cv"],
    coords={"split": ["train", "test"], "cv": cv_labels},
)
crps_ds = crps_da.to_dataset(name="crps")

pc_kwargs = _process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)
pc = PlotCollection.wrap(crps_ds, aes={"color": ["split"]}, backend=backend, **pc_kwargs)

cv_x = xr.DataArray(np.arange(len(cv_labels)), dims=["cv"], coords={"cv": cv_labels})
pc.map(azp.visuals.line_xy, x=cv_x, y=crps_ds["crps"], **(line_kwargs or {}))
pc.add_legend("split")
```

`PlotCollection.wrap` creates a single-panel collection; `aes={"color": ["split"]}`
ensures `pc.map` loops over train/test with different colors.

### Returns

`tuple[Figure, NDArray[Axes]]` or `PlotCollection`.

---

## Module-level Helpers (in `cv.py`)

| Helper | Purpose |
|--------|---------|
| `_validate_cv_results(results)` | Type check + group presence validation; raises `TypeError`/`ValueError` |
| `_extract_cv_labels(results)` | Returns `list[str]` of fold labels from `cv_metadata.coords["cv"]` |
| `_read_fold_meta(results, cv_label)` | Returns `(X_train, y_train, X_test, y_test)` from `cv_metadata` |
| `_pred_matrix_for_rows(results, cv_label, rows_df)` | Builds `(n_samples, n_rows)` prediction matrix for CRPS |
| `_filter_rows_and_y(df, y, indexers)` | Filters DataFrame rows by column values; used in `crps()` |

---

## Standard Parameters

All three methods accept the standard suite:

| Parameter | Purpose |
|-----------|---------|
| `dims` | Filter to specific coordinate values (applied via `_select_dims`) |
| `figsize` | Injected into `figure_kwargs` via `_process_plot_params` |
| `backend` | `"matplotlib"` (default), `"plotly"`, `"bokeh"` |
| `return_as_pc` | Return `PlotCollection` instead of `(Figure, NDArray[Axes])` |
| `**pc_kwargs` | Forwarded to `PlotCollection.grid/wrap()` for layout control |

**No `idata` override parameter** — unlike other namespace classes, `MMMCVPlotSuite`
is stateless and receives all data via `results`. There is nothing to override.

---

## Bug Fixes Resolved

| Old bug | Resolution |
|---------|-----------|
| `param_stability` looped per dim, calling `plt.show()` each iteration — produced multiple figures and stray display calls | New implementation: single call to `azp.plot_forest`; one figure always |
| `param_stability` returned bare `Axes` (not `NDArray[Axes]`) in the no-dims path | `_extract_matplotlib_result` always wraps in `NDArray` |
| `cv_predictions` embedded 350+ lines of raw matplotlib HDI logic | Replaced by `split_ds.azstats.hdi()` + `azp.visuals.fill_between_y` |
| `cv_crps` embedded raw matplotlib scatter without PlotCollection | Replaced by `PlotCollection.wrap` + `azp.visuals.line_xy` |

---

## Tests (`tests/mmm/plotting/test_cv.py`)

### Fixtures

- `cv_results_idata` — minimal `az.InferenceData` with:
  - `posterior`: `(cv=3, chain=2, draw=50, channel=2)` for `beta_channel`
  - `posterior_predictive["y_original_scale"]`: `(cv=3, chain=2, draw=50, date=30)`
  - `cv_metadata["metadata"]`: per-fold `{X_train, y_train, X_test, y_test}` dicts
    - Fold 0: train dates 0–19, test dates 20–29
    - Fold 1: train dates 0–24, test dates 25–29  (step=5)
    - Fold 2: train dates 0–29, test dates — (degenerate, should not crash)
  - `cv_metadata.coords["cv"]`: `["fold_0", "fold_1", "fold_2"]`
- `cv_plot` — `MMMCVPlotSuite()` instance

### Test cases

| Test | What it validates |
|------|-----------------|
| `test_predictions_returns_tuple` | Return type is `(Figure, NDArray[Axes])` |
| `test_predictions_return_as_pc` | `return_as_pc=True` yields `PlotCollection` |
| `test_predictions_n_axes_equals_n_folds` | Number of axes == number of CV folds |
| `test_predictions_train_test_colors_differ` | Fill patches in each panel have two distinct colors (blue + orange) |
| `test_predictions_missing_cv_metadata_raises` | `ValueError` when `cv_metadata` absent |
| `test_predictions_missing_posterior_predictive_raises` | `ValueError` when `posterior_predictive` absent |
| `test_predictions_dims_filtering` | `dims={"date": [...]}` — only a subset of dates present in result |
| `test_param_stability_returns_tuple` | Return type is `(Figure, NDArray[Axes])` |
| `test_param_stability_return_as_pc` | `return_as_pc=True` yields `PlotCollection` |
| `test_param_stability_var_names` | Only the requested `var_names` appear in the plot |
| `test_param_stability_dims_filtering` | `dims={"channel": ["tv"]}` — filtered channel only |
| `test_param_stability_no_cv_coord_raises` | `ValueError` when `cv` coord absent in posterior |
| `test_param_stability_single_figure` | Exactly one `Figure` object returned (not multiple) |
| `test_crps_returns_tuple` | Return type is `(Figure, NDArray[Axes])` |
| `test_crps_return_as_pc` | `return_as_pc=True` yields `PlotCollection` |
| `test_crps_train_test_colors_differ` | Two lines with distinct colors present |
| `test_crps_line_count` | Exactly two lines on the axes (train + test) |
| `test_crps_missing_cv_metadata_raises` | `ValueError` when `cv_metadata` absent |
| `test_crps_nan_tolerant` | NaN CRPS values (from failed folds) don't crash rendering |
