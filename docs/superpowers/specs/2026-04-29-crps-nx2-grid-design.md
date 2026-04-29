# CRPS Plot: n×2 Grid Layout

**Date:** 2026-04-29
**Branch:** `isofer/add-MMMCVPlotSuite`
**File:** `pymc_marketing/mmm/plotting/cv.py` — `MMMCVPlotSuite.crps()`

## Problem

The current `crps()` method renders a single panel with two lines (train and test). When `y_original_scale` has extra dimensions (e.g. `geo`), CRPS is aggregated over them and the per-dimension breakdown is invisible.

## Goal

Render an **n×2 grid** where:
- **Left column** = train CRPS across folds
- **Right column** = test CRPS across folds
- **Rows** = one per Cartesian combination of extra-dimension coordinates (e.g. one row per geo)
- When there are no extra dims: **1×2 grid** (left=train, right=test)

## Design

### 1. Extra-dim detection

At the start of `crps()`, detect non-standard dims from `y_original_scale`:

```python
pp = data.posterior_predictive["y_original_scale"]
base_dims = {"cv", "chain", "draw", "date"}
extra_dims = [d for d in pp.dims if d not in base_dims]
```

Build filtered coordinate lists (respecting the `dims` parameter):

```python
combo_coords = {
    d: (dims[d] if dims and d in dims else list(pp.coords[d].values))
    for d in extra_dims
}
combos = list(itertools.product(*combo_coords.values()))  # [()] when no extra dims
```

### 2. CRPS DataArray shape

```
# no extra dims:       (split=2, cv=n_folds)
# with geo:            (split=2, geo=n_geo, cv=n_folds)
# with geo + segment:  (split=2, geo=n_geo, segment=n_seg, cv=n_folds)
```

Allocation:

```python
shape = (2, *[len(v) for v in combo_coords.values()], len(cv_labels))
data_arr = np.full(shape, np.nan)
```

### 3. Computation loop

Outer loop over combinations; inner loop over folds. `_crps_for_split` is unchanged.

```python
for flat_idx, combo in enumerate(combos):
    dim_indexers = dict(zip(extra_dims, combo))
    multi_idx = np.unravel_index(flat_idx, [len(v) for v in combo_coords.values()])
    for fold_idx, lbl in enumerate(cv_labels):
        X_train, y_train, X_test, y_test = _read_fold_meta(data, lbl)
        data_arr[(0, *multi_idx, fold_idx)] = _crps_for_split(
            data, lbl, X_train, y_train, dim_indexers
        )
        data_arr[(1, *multi_idx, fold_idx)] = _crps_for_split(
            data, lbl, X_test, y_test, dim_indexers
        )
```

When `extra_dims=[]`, `combos=[()]`, `dim_indexers={}`, `multi_idx=()` — identical to current behaviour.

Build the DataArray:

```python
coords = {"split": ["train", "test"], **combo_coords, "cv": cv_labels}
crps_da = xr.DataArray(data_arr, dims=["split", *extra_dims, "cv"], coords=coords)
```

### 4. Layout

Replace `PlotCollection.wrap()` with `PlotCollection.grid()`:

```python
crps_ds = crps_da.to_dataset(name="crps")

pc = PlotCollection.grid(
    crps_ds,
    rows=[*extra_dims],        # [] → 1 row; ["geo"] → n_geo rows
    cols=["split"],             # always 2 columns: train | test
    aes={"color": ["split"]},  # consistent color per split across all rows
    backend=backend,
    **pc_kwargs,
)

cv_x = xr.DataArray(np.arange(len(cv_labels)), dims=["cv"], coords={"cv": cv_labels})
pc.map(azp.visuals.line_xy, x=cv_x, y=crps_ds["crps"], **(line_kwargs or {}))
pc.add_legend("split")
```

Column titles ("train" / "test") and row titles (coord values) are generated automatically by PlotCollection.

### 5. `dims` parameter

`dims` controls which coordinate values of extra dims appear as rows. Non-extra-dim keys are silently ignored (consistent with the rest of the suite). Example: `dims={"geo": ["geo_a"]}` → 1×2 grid showing only geo_a.

### 6. `itertools` import

Add `import itertools` to the module imports.

## Test Changes

| Test | Before | After |
|---|---|---|
| `test_line_count` | `axes[0]` has 2 lines | `axes[0]` has 1 line; `axes[1]` has 1 line |
| `test_train_test_colors_differ` | 2 colors on `axes[0]` | color of line in `axes[0]` ≠ color of line in `axes[1]` |
| `test_returns_tuple` | unchanged | unchanged |
| `test_return_as_pc` | unchanged | unchanged |
| `test_crps_multidim_geo_no_nan` | checks `axes[0]`, 2 lines each with finite values | checks 4 axes (2 geo × 2 split), each with 1 finite-valued line |

No new fixtures needed.

## API Compatibility

- `crps()` signature unchanged
- Return type unchanged: `tuple[Figure, NDArray[Axes]] | PlotCollection`
- `return_as_pc=True` works as before
- Only breaking change: `axes` array shape changes (more panels), which callers that index `axes[0]` to get both lines will need to update
