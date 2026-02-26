# Cast channel_data to float at construction time

**Issue:** [#2340](https://github.com/pymc-labs/pymc-marketing/issues/2340)

## Problem

The budget optimizer fails when `channel_data` has `dtype=int` because PyTensor
cannot compute gradients through `pt.cast(int_var, float)` (zero gradients).
A previous hack that mutated the PyTensor node type in-place was reverted in
commit `fbc28310` ("Remove type manipulation hack"), and replaced with a
`ValueError` at optimization time.

The team agreed to fix this at the source: always cast `channel_data` to float
at model construction time.

## Design

### Core change

In `MMM._generate_and_preprocess_model_data()`, immediately after building
`self.xarray_dataset`, cast the `_channel` variable to float if it isn't
already:

```python
if not np.issubdtype(self.xarray_dataset["_channel"].dtype, np.floating):
    self.xarray_dataset["_channel"] = self.xarray_dataset["_channel"].astype(float)
```

This is a single-line change at approximately line 1418 of
`pymc_marketing/mmm/multidimensional.py`.

### Why cast early at the xarray level

- All downstream consumers — `_compute_scales()`, `pm.Data("channel_data")`,
  `_replace_data()`, budget optimizer, incrementality, sensitivity analysis —
  automatically see float data with zero additional changes.
- The scalers computed by `_compute_scales()` are guaranteed to be float,
  avoiding potential integer-division issues.
- The `_replace_data()` dtype-matching code naturally works since the original
  dtype is already float.

### Alternatives considered

1. **Cast only at `pm.Data` creation** — minimal change but leaves
   `xarray_dataset._channel` as int, risking subtle int-related bugs in other
   code paths (e.g., scaler computation).
2. **Cast at xarray level + warning** — same as chosen approach but with a
   `warnings.warn()`. Rejected because there's no actionable step for the user
   and it would be noisy.

### Downstream impact (none)

- `BudgetOptimizer._replace_channel_data_by_optimization_variable()` has a
  defensive `ValueError` check for non-float dtype (added in `fbc28310`). It
  will never trigger but is left in place as a safety guard.
- `sensitivity_analysis.py` has a float64 cast for sweep evaluation. This
  remains correct and harmless.

## Tests

- Add a test that constructs an MMM with integer channel data and verifies:
  1. `self.xarray_dataset["_channel"].dtype` is float after `build_model()`
  2. The model's `channel_data` variable has a float dtype
  3. The model builds without error
