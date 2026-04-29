# MMMCVPlotSuite — Design Spec

**PR:** 8 of the MMMPlotSuite v2 series
**Date:** 2026-04-13
**Branch:** `isofer/add-MMMCVPlotSuite`

---

## Overview

Introduces `MMMCVPlotSuite`: a standalone class that provides
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
| `pymc_marketing/mmm/time_slice_cross_validation.py` | Update `.plot` property to return `MMMCVPlotSuite(self.cv_idata)` |
| `pymc_marketing/mmm/plotting/__init__.py` | Export `MMMCVPlotSuite` |
| `tests/mmm/plotting/test_cv.py` | **New** — full test suite |

---

## Class Structure

```
MMMCVPlotSuite
  ├── __init__(cv_data)              — validates and stores self.cv_data
  ├── predictions(cv_data=None, ...)   — HDI bands, observed line, train-end vline
  ├── param_stability(cv_data=None, ...)  — arviz_plots.plot_forest via DataTree dict
  └── crps(cv_data=None, ...)          — scalar CRPS line chart per fold
```

Follows the same pattern as `DecompositionPlots`, `DiagnosticsPlots`, etc.:
- `__init__` accepts `cv_data`, runs `_validate_cv_results` (type check + `cv_metadata` presence), and stores it as `self.cv_data`.
- Each method accepts `cv_data: az.InferenceData | None = None`.  When `None`, the method uses `self.cv_data`.  When provided, it overrides for that call only and `_validate_cv_results` is re-run on the override.
- Additional method-specific validation (e.g. checking `posterior_predictive["y_original_scale"]` or the `cv` coordinate) happens inside each method, after `_validate_cv_results`.

---

## Integration Point

```python
# pymc_marketing/mmm/time_slice_cross_validation.py
class TimeSliceCrossValidator:
    @property
    def plot(self) -> MMMCVPlotSuite:
        self._validate_model_was_built()
        return MMMCVPlotSuite(self.cv_idata)
```

The old property also called `_validate_idata_exists()`.  That check is subsumed by
`MMMCVPlotSuite.__init__`, which runs `_validate_cv_results` on the passed InferenceData.
`_validate_model_was_built()` is kept in the property because `self.cv_idata` does not exist
before `run()` has been called.

---

## Class Initialization

```python
def __init__(self, cv_data: az.InferenceData) -> None:
```

Calls `_validate_cv_results(cv_data)` and stores `self.cv_data = cv_data`.

---

## Method: `predictions()`

### What it renders

For each CV fold (one panel per fold): posterior predictive HDI in **blue** for
train dates and **orange** for test dates, with observed actuals in black and a
vertical green dashed line at the train/test boundary.

### API name change

Old: `MMMPlotSuite.cv_predictions(results, dims)`
New: `MMMCVPlotSuite.predictions(cv_data, dims, hdi_prob, figsize, backend, return_as_pc, hdi_kwargs, **pc_kwargs)`

### Signature

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
```

`cv_data=None` uses `self.cv_data`.  If an override is provided, `_validate_cv_results`
is called on it before use.

### Input validation

Raise `TypeError` if the resolved `cv_data` is not `az.InferenceData`.
Raise `ValueError` if:
- `cv_metadata` group absent or has no `"metadata"` variable
- `posterior_predictive["y_original_scale"]` absent

### Data preparation

1. Extract `cv_labels` from `cv_data.cv_metadata.coords["cv"].values`.
2. Extract the full date coordinate from `cv_data.posterior_predictive["y_original_scale"]`.
3. For each fold `lbl`:
   a. Read `X_train`, `y_train`, `X_test`, `y_test` from `cv_metadata["metadata"].sel(cv=lbl).item()`.
   b. Compute `train_dates` and `test_dates` as `pd.DatetimeIndex` from `X_train["date"]` / `X_test["date"]`.
   c. Build a boolean mask over the full date coordinate:
      - `train_mask`: True where date ∈ `train_dates`
      - `test_mask`: True where date ∈ `test_dates`
   d. From `pp = cv_data.posterior_predictive["y_original_scale"].sel(cv=lbl)`:
      - `y_train_fold = pp.where(train_mask)` — NaN outside train window
      - `y_test_fold = pp.where(test_mask)` — NaN outside test window
   e. Build `y_obs_fold` — observed actuals (no chain/draw) for this fold by
      combining `y_train` and `y_test` from metadata, aligned to the full date
      coordinate. Shape: `(date, ...)`.
   f. Record `train_end = train_dates.max()` for this fold.
4. Stack all folds along `cv`:
   - `y_train_da`, `y_test_da`: dims `(cv, chain, draw, date, ...)`
   - `y_obs_da`: dims `(cv, date, ...)`
   - `train_end_da`: `xr.DataArray` with dims `(cv,)` — one train-end date per fold.
5. Apply `_select_dims` for any user-supplied `dims` (on `y_train_da`, `y_test_da`,
   and `y_obs_da`).
6. Determine `custom_dims` — extra dims beyond `(cv, chain, draw, date)` from the
   posterior predictive.
7. Build `split_ds = xr.Dataset({"train": y_train_da, "test": y_test_da})`.

### PlotCollection rendering

```python
pc_kwargs = _process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)
rows = pc_kwargs.pop("rows", [*custom_dims, "cv"])  # one row per (custom_dim..., fold)
cols = pc_kwargs.pop("cols", [])

pc = PlotCollection.grid(
    split_ds,
    rows=rows,
    cols=cols,
    aes={"color": ["__variable__"]},  # train → blue, test → orange
    backend=backend,
    **pc_kwargs,
)

# HDI bands — colored by __variable__
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

# Observed actuals — single black line per panel; y_obs_da has (cv, date, ...)
# which PlotCollection subsets correctly per panel without needing chain/draw
pc.map(azp.visuals.line_xy, x=date_da, y=y_obs_da, color="black", linewidth=1.5)

# Train-end vertical boundary — per-fold via DataArray; add_lines accepts
# a DataArray with a 'cv' dimension and places the correct value in each panel
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
```

No for loops, no raw matplotlib calls.

### Returns

`tuple[Figure, NDArray[Axes]]` (default) or `PlotCollection` (`return_as_pc=True`).

---

## Method: `param_stability()`

### What it renders

A single forest plot comparing parameter posterior distributions across all CV
folds. Folds are colored differently via `aes={"color": ["cv"]}`. Alternating
shading distinguishes values within other dimensions (e.g. channel) via
`shade_label`.

### API change

Old: `MMMPlotSuite.param_stability(results, parameter, dims)`
New: `MMMCVPlotSuite.param_stability(cv_data, var_names, dims, figsize, figure_kwargs, backend, return_as_pc, **pc_kwargs)`

Changes:
- `parameter: list[str]` → `var_names: list[str]` (consistent with arviz naming)
- Single figure always — old code looped per dim value calling `plt.show()`.
- Returns `tuple[Figure, NDArray[Axes]]` — old code sometimes returned bare `Axes`.

### Signature

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
```

`cv_data=None` uses `self.cv_data`.  If an override is provided, `_validate_cv_results`
is called on it before use.

### Input validation

Raise `TypeError` if the resolved `cv_data` is not `az.InferenceData`.
Raise `ValueError` if:
- `cv_data` has no `posterior` group (`not hasattr(cv_data, "posterior")`)
- No `"cv"` coordinate found in `cv_data.posterior`

### Implementation

```python
# 1. Transpose posterior so dimension order reads well in the forest plot
#    (sample dims first, then the labelled dims ending with "cv")
posterior = cv_data.posterior
if dims:
    posterior = _select_dims(posterior, dims)
# Move "channel" and "cv" to the end so it appears as the innermost loop in the plot;
posterior = posterior.transpose(..., "channel", "cv")
# Rebuild a minimal InferenceData for plot_forest
idata_for_plot = az.InferenceData(posterior=posterior)

# 2. Merge figure_kwargs: defaults + user overrides + optional figsize
fig_kw: dict[str, Any] = {
    "width_ratios": [1, 2],
    "layout": "none",
    **(figure_kwargs or {}),
}
if figsize is not None:
    fig_kw["figsize"] = figsize

# 3. Call plot_forest
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

No manual iteration over cv_labels. The `cv` dimension is treated as a regular
coordinate; `plot_forest` handles faceting and coloring via `aes={"color": ["cv"]}`.

`dims` narrows which coordinate values appear (e.g. `{"channel": ["tv"]}`)
before the transpose step.

### Returns

`tuple[Figure, NDArray[Axes]]` or `PlotCollection`.

---

## Method: `crps()`

### What it renders

A line chart of mean CRPS per fold, with train (blue) and test (orange) as two
colored lines on a single panel. X-axis = fold index, Y-axis = CRPS score.

### API name change

Old: `MMMPlotSuite.cv_crps(results, dims)`
New: `MMMCVPlotSuite.crps(cv_data, dims, figsize, backend, return_as_pc, line_kwargs, **pc_kwargs)`

### Signature

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
```

`cv_data=None` uses `self.cv_data`.  If an override is provided, `_validate_cv_results`
is called on it before use.

### Input validation

Raise `TypeError` if the resolved `cv_data` is not `az.InferenceData`.
Raise `ValueError` if:
- `cv_metadata` group absent
- `posterior_predictive["y_original_scale"]` absent

### CRPS computation

The computation is identical to the old `cv_crps` logic:

1. Extract `cv_labels`.
2. Define `_pred_matrix_for_rows(cv_data, cv_label, rows_df) → np.ndarray` (shape `(n_samples, n_rows)`) — builds a prediction matrix by selecting `posterior_predictive["y_original_scale"].sel(cv=cv_label)` stacked over `(chain, draw)` and indexing by date.
3. Define `_filter_rows_and_y(df, y, indexers) → (filtered_df, y_arr)` — filters `X_train`/`X_test` rows matching the `dims` coordinate filters.
4. For each fold:
   - Read `X_train`, `y_train`, `X_test`, `y_test` from `cv_metadata`.
   - Use `_crps_for_split` for both train and test splits (NaN on failure or empty set).
   - Append to `crps_train_list` / `crps_test_list`.

All helpers are module-level functions in `cv.py`.

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
| `_validate_cv_results(cv_data)` | Shared base validation called at `__init__` and on any per-call override. Raises `TypeError` if `cv_data` is not `az.InferenceData`. Raises `ValueError` if the `cv_metadata` group is absent — this is the minimum required for any of the three methods to function. Method-specific checks (e.g. `posterior_predictive["y_original_scale"]` for `predictions`/`crps`, `posterior` + `cv` coord for `param_stability`) are performed inside each method, not here. |
| `_extract_cv_labels(cv_data)` | Returns `list[str]` of fold labels from `cv_metadata.coords["cv"]` |
| `_read_fold_meta(cv_data, cv_label)` | Returns `(X_train, y_train, X_test, y_test)` from `cv_metadata` |
| `_pred_matrix_for_rows(cv_data, cv_label, rows_df)` | Builds `(n_samples, n_rows)` prediction matrix for CRPS |
| `_filter_rows_and_y(df, y, indexers)` | Filters DataFrame rows by column values; used in `crps()` |
| `_crps_for_split(cv_data, cv_label, X, y, dim_indexers)` | Computes mean CRPS for one fold/split; returns NaN on failure or empty set |

---

## Standard Parameters

All three methods share this parameter set:

| Parameter | Purpose |
|-----------|---------|
| `cv_data` | `az.InferenceData \| None` — override stored `self.cv_data` for this call only |
| `dims` | Filter to specific coordinate values (applied via `_select_dims`) |
| `figsize` | Injected into `figure_kwargs` via `_process_plot_params` |
| `backend` | `"matplotlib"` (default), `"plotly"`, `"bokeh"` |
| `return_as_pc` | Return `PlotCollection` instead of `(Figure, NDArray[Axes])` |
| `**pc_kwargs` | Forwarded to `PlotCollection.grid/wrap()` for layout control |

When `cv_data` is not `None`, `_validate_cv_results` is called on it before proceeding.

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
- `cv_plot` — `MMMCVPlotSuite(cv_results_idata)` instance (initialized with the fixture)

All method calls use `cv_plot.<method>()` with no `cv_data` argument (uses `self.cv_data`).
Override tests call `cv_plot.<method>(cv_data=some_other_idata)` to test the override path.

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
