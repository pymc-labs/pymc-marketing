# PyMC 6 / ArviZ 1.1 / PyTensor 3 Migration Guide

This PR migrates pymc-marketing from `pymc<6`, `arviz<1`, `pytensor<3` to
`pymc>=6.0.0`, `arviz>=1.1.0`, `pytensor>=3.0.0`. The core breaking change is
that **`arviz.InferenceData` is replaced by `xr.DataTree`** as the data container.

Reference: https://python.arviz.org/en/stable/user_guide/migration_guide.html#datatree

---

## A — Source-code changes

### A1. Type system: `InferenceData` → `DataTree`

Every type annotation, return type, and cast changes:

```python
# OLD
import arviz as az
idata: az.InferenceData
return az.InferenceData(...)
idata = cast(az.InferenceData, self.idata)

# NEW
import xarray as xr
idata: xr.DataTree
return xr.DataTree.from_dict({"/posterior": posterior})
idata = cast(xr.DataTree, self.idata)
```

### A2. Group access — slash-prefixed paths

Groups are DataTree nodes accessed with `/`-prefix:

```python
# OLD                                    # NEW
self.idata["posterior"]                   self.idata["/posterior"]
self.idata.posterior                      self.idata["/posterior"]
self.idata.posterior[var]                 self.idata["/posterior"].dataset[var]
self.idata.posterior.coords[dim]          self.idata["/posterior"].dataset.coords[dim]
self.idata.posterior.data_vars            self.idata["/posterior"].dataset.data_vars
```

The leading `/` is optional — `dt["posterior"]` and `dt["/posterior"]` refer to the
same node. The codebase uses the leading-`/` convention consistently for clarity.

### A3. `.dataset` (view) vs `.to_dataset()` (copy)

Accessing a leaf group always returns a `DataTree`. To get the underlying `Dataset`:

- **`.dataset`** — returns a **view** (no-copy, read-only). Prefer for internal reads.
- **`.to_dataset()`** — returns a **shallow copy** (mutable). Use only when you need to edit.

```python
# Read-only — use .dataset
self.idata["/posterior"].dataset[var]
self.idata.posterior.dataset[var]

# Mutable copy — use .to_dataset()
self.idata["/fit_data"].to_dataset().to_dataframe()
```

### A4. Group existence checks

```python
# OLD                                    # NEW (preferred)
hasattr(self.idata, "posterior")          "/posterior" in self.idata.groups
"posterior" in self.idata                 "posterior" in self.idata.children
```

For single-level nesting (our standard case), `.children` is cleaner because it
avoids the `/` prefix and the root-group issue:

```python
"fit_data" in self.idata.children    # preferred
"/fit_data" in self.idata.groups     # also valid
```

### A5. Adding / removing groups

```python
# OLD                                    # NEW
self.idata.add_groups(fit_data=fit_data)  self.idata["/fit_data"] = fit_data
del self.idata.fit_data                    self.idata = self.idata.drop_nodes("fit_data")
del self.idata.posterior_predictive        del self.idata["posterior_predictive"]
```

### A6. Extending / updating

```python
# OLD                                    # NEW
self.idata.extend(prior_pred, join="right")  self.idata.update(prior_pred)
```

`DataTree.update()` behaves like `dict.update()` — it overwrites matching keys.
To replicate the old default `.extend(how="left")` behaviour, swap the order:

```python
# old default: idata.extend(new, join="right")  →  idata kept on conflict
# equivalent:
new_idata.update(idata)
```

### A7. Reading `fit_data`

```python
# OLD                                    # NEW
idata.fit_data.to_dataframe()             idata.fit_data.dataset.to_dataframe()
```

### A8. Saving / loading

```python
# OLD                                    # NEW
az.from_netcdf(str(filepath))             xr.open_datatree(str(filepath))
self.idata.to_netcdf(str(file))           idata_to_save.to_netcdf(str(file))
```

Using `az.from_netcdf()` still works (it's a thin wrapper around `open_datatree`),
but the codebase now calls `xr.open_datatree()` directly.

When saving a subset of groups with `zarr`, prefer `DataTree.filter()`:

```python
# Instead of manual dict construction:
idata = idata.filter(lambda g: g.name in groups)
idata.to_zarr(store)
```

### A9. `pm.sample_prior_predictive` → returns `DataTree`

```python
# OLD
return pm.sample_prior_predictive(...).prior

# NEW
prior_pred = pm.sample_prior_predictive(...)
return prior_pred["/prior"].to_dataset()
```

### A10. `pm.compute_deterministics`

```python
# OLD
pm.compute_deterministics(idata.posterior, ...)

# NEW
pm.compute_deterministics(idata["/posterior"], ...)
```

### A11. `az.hdi` — keyword and coordinate rename

`hdi_prob` → `prob`, `input_core_dims` → `dim`:

```python
# OLD
az.hdi(data, hdi_prob=0.94, input_core_dims=[["sample"]])
hdi[var].sel(hdi="lower"), hdi[var].sel(hdi="higher")

# NEW
az.hdi(data, prob=0.94, dim="sample")
hdi.sel(ci_bound="lower"), hdi.sel(ci_bound="upper")
```

**Performance tip (from Oriol):** compute HDI once outside loops and subset inside:

```python
# Avoid recomputing HDI per category:
all_hdi = az.hdi(data, prob=0.94, ...)
for i, product in enumerate(products):
    hdi_for_product = all_hdi.isel(existing_product=i)
```

### A12. `az.summary` — `hdi_prob` → `ci_prob`

```python
# OLD:  az.summary(data, hdi_prob=0.94, kind="stats")
# NEW:  az.summary(data, ci_prob=0.94, kind="stats")
```

### A13. `az.extract` — optional `keep_dataset`

```python
# Sometimes needed to force Dataset return:
az.extract(post_pred, group, combined=combined, keep_dataset=True)
```

### A14. `az.r2_score` removed — use inline function

```python
def _bayesian_r2(y_true, y_pred):
    var_resid = np.var(y_true - y_pred)
    var_pred = np.var(y_pred)
    return var_pred / (var_pred + var_resid)
```

### A15. `az.plot_hdi` removed → `az.hdi` + `fill_between`

```python
# OLD                                      # NEW
az.plot_hdi(dates, data, ax=ax)            hdi = az.hdi(data, prob=0.94, dim="sample")
                                           ax.fill_between(dates,
                                               hdi.sel(ci_bound="lower"),
                                               hdi.sel(ci_bound="upper"),
                                               alpha=0.2)
```

### A16. `az.plot_*` → `azp.plot_*` (arviz\_plots)

ArviZ 1.x moved most plotting functions to `arviz_plots`. Use the `azp`
alias:

```python
# OLD                                      # NEW
import arviz as az                          import arviz as az
az.plot_forest(idata, ...)                  import arviz_plots as azp
                                            azp.plot_forest(idata, ...)

# All affected functions:                  azp.plot_dist(...)
az.plot_forest(idata, ...)                  azp.plot_trace(idata, ...)
az.plot_posterior(idata, ...)               azp.plot_hdi(...)  # (if applicable)
                                            # (all take DataTree directly)
```

### A17. `az.plot_posterior` → `az.plot_dist`

```python
# OLD:  az.plot_posterior(idata.posterior, var_names=[...], ref_val=...)
# NEW:  az.plot_dist(idata.posterior, var_names=[...])
```
`ref_val` support is dropped. Use manual `ax.axvline()` if needed.

### A18. `az.plot_ppc` → `az.plot_ppc_dist`

```python
# OLD:  az.plot_ppc(idata, var_names=["y"])
# NEW:  az.plot_ppc_dist(idata, var_names=["y"])
```

### A19. `az.style.use("arviz-darkgrid")` → `"arviz-vibrant"`

```python
# OLD:  az.style.use("arviz-darkgrid")
# NEW:  az.style.use("arviz-vibrant")
```

### A20. `azp` (arviz\_plots) `.azstats.hdi` API

```python
# OLD:  ds.azstats.hdi(hdi_prob)
# NEW:  ds.azstats.hdi(prob=hdi_prob)
```

### A21. PyTensor imports restructured

```python
# OLD                                    # NEW
from pytensor import Variable             from pytensor.graph.basic import Variable
from pytensor import graph_replace        from pytensor.graph.replace import graph_replace
from pytensor import as_symbolic          from pymc.pytensorf import StringConstant
```

### A22. FourierBase `samples` → `draws`

```python
# OLD:  self.prior.sample_prior(..., samples=500)
# NEW:  self.prior.sample_prior(..., draws=500)
```

### A23. `pm.sample_posterior_predictive` with `extend_inferencedata`

```python
# Manual extend                                 # Using kwarg
pm.sample_posterior_predictive(idata,            pm.sample_posterior_predictive(
    extend_inferencedata=True,                   idata,
)                                                extend_inferencedata=True,
                                                 )
```

### A24. Return type: `sample_posterior_predictive` → `DataArray`

`pm.sample_posterior_predictive` returns a `Dataset`, but when passed to
`az.extract(..., var_names="str")` the result is a `DataArray`. Annotate
accordingly:

```python
# OLD (wrong)
def sample_posterior_predictive(...) -> xr.Dataset:

# NEW (correct)
def sample_posterior_predictive(...) -> xr.DataArray:
```

### A25. Building `DataTree` from dict

```python
# OLD
az.InferenceData(posterior=posterior)

# NEW
xr.DataTree.from_dict({"/posterior": posterior})
```

### A26. `pymc_extras` warns about implicit array → DataArray conversion

When passing array-like parameters (e.g. `sigma`) to prior distributions in
channel-specific dims, `pymc_extras` emits a warning:

```
UserWarning: Implicit conversion of array-like parameter sigma to DataArray
with dims ('channel',). Use DataArray with explicit dims to avoid this warning
```

**Fix:** wrap the parameter in a `DataArray` with explicit dimension names:

```python
import xarray as xr

# OLD — implicit, triggers warning
sigma=np.array([1.0, 2.0, 3.0]),

# NEW — explicit, no warning
sigma=xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["channel"]),
```

### A27. Dependency changes in `pyproject.toml`

```text
arviz>=0.13.0,<1.0.0  →  arviz>=1.1.0
pymc>=5.28.5,<6.0.0   →  pymc>=6.0.0
pytensor>=2.38.2,<3.0.0 →  pytensor>=3.0.0
pymc-extras>=0.9.2     →  pymc-extras>=0.9.3
```

---

## B — Notebook changes (`docs/source/notebooks/`)

For each notebook, apply these patterns:

### B1. Style

`az.style.use("arviz-darkgrid")` → `az.style.use("arviz-vibrant")`

### B2. `az.plot_posterior` → `az.plot_dist`

Replace all `az.plot_posterior(...)` calls with `az.plot_dist(...)`.
Drop `ref_val=` arguments (or replace with manual `ax.axvline()`).

### B3. `az.plot_ppc` → `az.plot_ppc_dist`

### B4. `az.plot_hdi` → `az.hdi` + `fill_between`

### B5. `idata.extend(...)` → `idata.update(...)`

### B6. References to `InferenceData` in markdown cells → `DataTree` (or `xr.DataTree`)

### B7. `az.plot_*` → `azp.plot_*` using `import arviz_plots as azp`

### B8. No changes needed for `az.plot_trace`, `az.plot_energy`, `az.ess` — these still accept `DataTree`

### B9. When notebooks call raw `pm.sample_posterior_predictive`:

```python
# OLD
idata = pm.sample_posterior_predictive(idata, ...)

# NEW — extend_inferencedata=True is the idiomatic replacement
pm.sample_posterior_predictive(idata, ..., extend_inferencedata=True)
```

### B10. `plot_dist` / `plot_forest` / `plot_trace` CANNOT accept bare DataArrays or ndarrays

These functions require a `DataTree` (or sometimes `Dataset`). Passing a raw `DataArray`
or numpy `ndarray` causes `KeyError: 'posterior'` (on DataArray) or
`IndexError: only integers...` (on ndarray).

```python
# ❌ BROKEN
azp.plot_dist(my_dataarray)
azp.plot_dist(my_numpy_array)
azp.plot_forest(some_dataarray, ...)

# ✅ FIXED
azp.plot_dist(my_dataarray.to_dataset(name="my_var"))
azp.plot_dist(xr.Dataset({"x": xr.DataArray(my_numpy_array, dims=["sample"])}))
azp.plot_forest(xr.Dataset({"var": my_dataarray}), ...)
```

**Rule of thumb**: if the first argument to an `azp.plot_*` function is NOT wrapped in
`xr.Dataset()` or `xr.DataTree()`, wrap it.

### B11. `figsize` → `figure_kwargs` (plot_forest, plot_dist, plot_trace)

arviz_plots 2.x moved `figsize`/`backend_kwargs` into a nested dict:

```python
# ❌ BROKEN
azp.plot_forest(idata, figsize=(8, 8))
azp.plot_trace(idata, backend_kwargs={"figsize": (12, 10)})
azp.plot_dist(idata, figsize=(10, 6))

# ✅ FIXED
azp.plot_forest(idata, figure_kwargs={"figsize": (8, 8)})
azp.plot_trace(idata, figure_kwargs={"figsize": (12, 10)})
azp.plot_dist(idata, figure_kwargs={"figsize": (10, 6)})
```

### B12. `rug=True` removed from `plot_dist`

```python
# ❌ BROKEN
azp.plot_dist(idata, rug=True)

# ✅ FIXED — just remove rug=True
azp.plot_dist(idata)
```

### B13. `sample_dims` must match the data dimensions

`validate_sample_dims()` auto-detects `["chain", "draw"]`, but some data uses
pre-stacked `"sample"` dim. **Check what dims your data actually has.**

```python
# Data with chain/draw dims → auto-detection works, do NOT specify sample_dims
azp.plot_dist(xr.Dataset({"roas": roas}))  # roas has (chain, draw, channel)

# Data with pre-stacked sample dim → must specify explicitly
azp.plot_dist(
    xr.Dataset({"x": stacked_data}),  # stacked_data has dim (sample,)
    sample_dims=["sample"],
)
```

### B14. List-based `plot_forest` no longer supported

Old API accepted `[idata1, idata2]` with `model_names`. New API expects a single DataTree:

```python
# ❌ BROKEN
azp.plot_forest([idata1, idata2], model_names=["A", "B"], ...)

# ✅ FIXED — combine into single DataTree with model dim
combined = xr.concat([idata1["/posterior"].dataset, idata2["/posterior"].dataset],
                     dim="model").assign_coords(model=["A", "B"])
combined_dt = xr.DataTree.from_dict({"/posterior": xr.DataTree(combined)})
azp.plot_forest(combined_dt, ...)
```

### B15. `data=` kwarg → first positional arg in `plot_forest`

```python
# ❌ BROKEN
azp.plot_forest(data=some_data, combined=True, figsize=(8, 7))

# ✅ FIXED
azp.plot_forest(some_data, combined=True, figure_kwargs={"figsize": (8, 7)})
```

### B16. Return type: `ax, *_` → `PlotCollection.viz[...]`

arviz_plots returns a `PlotCollection`, not `(axes, ...)`:

```python
# ❌ BROKEN
ax, *_ = azp.plot_forest(idata, ...)
axes = azp.plot_trace(idata, ...)

# ✅ FIXED
pc = azp.plot_forest(idata, ...)
ax = pc.viz["/"]["figure"].values.item().axes[0]
```

For `plot_trace` — the returned PlotCollection renders automatically; don't assign axes.
Use `plt.suptitle()` / `plt.tight_layout()` after the call.

### B17. `compact=True` removed from `plot_trace`

```python
# ❌ BROKEN
azp.plot_trace(idata, compact=True)

# ✅ FIXED — just remove compact=True
azp.plot_trace(idata)
```

### B18. `group` parameter for `pm.sample_posterior_predictive` results

`pm.sample_posterior_predictive()` returns a DataTree with `posterior_predictive` group,
NOT `posterior`. The default `group="posterior"` fails:

```python
# ❌ BROKEN
idata_pp = pm.sample_posterior_predictive(idata)
azp.plot_dist(idata_pp)  # KeyError: 'posterior'

# ✅ FIXED
azp.plot_dist(idata_pp, group="posterior_predictive")
```

### B19. `.to_dataset()` on unnamed DataArrays needs `name=`

```python
# ❌ BROKEN — ValueError: unable to convert unnamed DataArray
da.isel(customer_id=slice(0, 10)).to_dataset()

# ✅ FIXED
da.isel(customer_id=slice(0, 10)).to_dataset(name="expected_spend")
```

### B20. Missing `import xarray as xr` in notebooks

Any notebook that uses `xr.Dataset(...)`, `xr.DataArray(...)`, `xr.DataTree(...)`,
or `xr.concat(...)` must have:

```python
import xarray as xr
```

Check the imports cell of every notebook touched.

### B21. `.posterior.to_dataframe()` on DataTree needs `.ds` accessor

DataTree child nodes don't have `.to_dataframe()`. Use `.ds.to_dataframe()`:

```python
# ❌ BROKEN
model.fit(method="map").posterior.to_dataframe()

# ✅ FIXED (if display needed)
model.fit(method="map").posterior.ds.to_dataframe()

# ✅ FIXED (if DataTree/Dataset access is sufficient)
model.fit(method="map").posterior
```

### B22. `hdi_prob` → `prob` in `az.hdi()` and `ci_bounds` → `ci_bound`

```python
# ❌ BROKEN
az.hdi(data, hdi_prob=0.94)
hdi.sel(ci_bounds="lower"), hdi.sel(ci_bounds="higher")

# ✅ FIXED
az.hdi(data, prob=0.94)
hdi.sel(ci_bound="lower"), hdi.sel(ci_bound="upper")
```

Note: the dimension value changed from `"higher"` → `"upper"` AND from `ci_bounds` → `ci_bound`.

---

## C — Systematic notebook workflow

For each notebook:

1. Run it — failures reveal which patterns need updating
2. Fix `az.style.use` at the top
3. Fix `az.plot_posterior(...)` → `az.plot_dist(...)` (drop `ref_val`)
4. Fix `az.plot_ppc(...)` → `az.plot_ppc_dist(...)`
5. Fix `az.plot_hdi(...)` → `az.hdi()` + `fill_between`
6. Fix `idata.extend(...)` → `idata.update(...)`
7. Fix markdown references to `InferenceData` → `DataTree`
8. **Check EVERY `azp.plot_*` call** for the patterns in B10–B22 above:
   - First argument wrapped in Dataset/DataTree? (B10)
   - `figsize` → `figure_kwargs`? (B11)
   - `rug=True` removed? (B12)
   - `sample_dims` matches data? (B13)
   - Lists passed to `plot_forest`? (B14)
   - `data=` kwarg → positional? (B15)
   - Return value unpacking fixed? (B16)
   - `compact=True` removed? (B17)
   - `group` correct for pp samples? (B18)
   - `.to_dataset()` has `name=`? (B19)
   - `import xarray as xr` present? (B20)
   - `.to_dataframe()` uses `.ds`? (B21)
   - `hdi_prob`/`ci_bounds` updated? (B22)
9. Re-run and verify
