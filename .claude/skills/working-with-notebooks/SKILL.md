---
name: working-with-notebooks
description: Fixing and validating Jupyter notebooks in pymc-marketing. Use when notebooks fail CI, need arviz_plots 2.x migration fixes, or when source code changes require notebook output updates.
disable-model-invocation: false
---

# Working with Notebooks

How to efficiently debug, fix, and validate notebooks in pymc-marketing.

---

## 1. The Mock Runner (fast local testing)

The mock runner injects mocked PyMC sampling into notebooks, letting you check for runtime errors without real MCMC. See `scripts/run_notebooks/README.md`.

### Run a single notebook locally

```bash
uv run python scripts/run_notebooks/runner.py --notebooks docs/source/notebooks/mmm/mmm_example.ipynb
```

### Run multiple notebooks

```bash
uv run python scripts/run_notebooks/runner.py --notebooks docs/source/notebooks/clv/sbg.ipynb docs/source/notebooks/mmm/mmm_example.ipynb
```

### Run by category (CI-like)

```bash
uv run python scripts/run_notebooks/runner.py --exclude-dirs mmm    # non-MMM only
uv run python scripts/run_notebooks/runner.py --exclude-dirs clv bass customer_choice general  # MMM only
```

---

## 2. Real Notebook Execution (for functional code changes)

When source code changes affect model outputs, notebooks must be re-executed with real sampling to generate updated output cells.

### Via jupyter nbconvert

```bash
jupyter nbconvert --to notebook --execute --inplace docs/source/notebooks/mmm/mmm_example.ipynb
```

### Via the runner's internal API (if `--no-mock` flag is added)

```bash
uv run python -c "
from scripts.run_notebooks.runner import run_notebook
from pathlib import Path
run_notebook(Path('docs/source/notebooks/mmm/mmm_example.ipynb'), mock=False)
"
```

### When to execute vs mock

| Scenario | Tool |
|---|---|
| Notebook API migration (plot_dist, DataTree, kwarg changes) | Mock runner |
| Source code changes (model fixes, new return types) | Real execution |
| Before pushing ANY notebook change | Mock runner (minimum) |
| Before pushing functional model changes | Both: real execution + mock runner |

---

## 3. Core Workflow (Iteration Loop)

**CRITICAL: Never push until all targeted notebooks pass locally.**

```
1. Identify failing notebooks (from CI or initial mock run)
2. FOR EACH notebook (one at a time):
   a. Run mock runner: python scripts/run_notebooks/runner.py --notebooks <path>
   b. Read the error (Cell In[X], exception type, traceback)
   c. Identify the pattern (see §4 below)
   d. Fix the notebook cell
   e. Repeat a-d until mock runner passes
3. Run ruff on changed files:
   ruff check --fix <files>
   ruff format <files>
4. Run pre-commit (or at minimum: ruff check + ruff format + mypy)
5. Push ONCE when all notebooks pass
```

---

## 4. Common Notebook Failure Patterns

Reference: `PYMC6_MIGRATION_GUIDE.md` sections B10-B26 for full details.

### B10: Bare DataArray/ndarray passed to plot_dist/plot_forest

```python
# ❌ BROKEN — KeyError: 'posterior' or IndexError
azp.plot_dist(my_dataarray)
azp.plot_dist(my_numpy_array)

# ✅ FIXED — wrap in Dataset
azp.plot_dist(my_dataarray.to_dataset(name="var_name"))
azp.plot_dist(xr.Dataset({"x": xr.DataArray(data, dims=["sample"])}))
```

### B11: figsize → figure_kwargs

```python
# ❌ — ValueError: extra keywords
azp.plot_forest(idata, figsize=(8, 8))

# ✅
azp.plot_forest(idata, figure_kwargs={"figsize": (8, 8)})
```

### B12: rug=True removed

```python
# ❌ — ValueError: extra keywords
azp.plot_dist(idata, rug=True)
# ✅ — just remove rug=True
azp.plot_dist(idata)
```

### B13: sample_dims mismatch

```python
# ❌ — ValueError: sample_dims not found
azp.plot_dist(xr.Dataset({"x": data}), sample_dims=["sample"])  # data has chain/draw

# ✅ — remove sample_dims (auto-detect) or match data dims
azp.plot_dist(xr.Dataset({"x": data}))  # auto-detect for chain/draw data
azp.plot_dist(xr.Dataset({"x": data}), sample_dims=["sample"])  # for pre-stacked data
```

### B14: List-based plot_forest with model_names

```python
# ❌ — TypeError: list indices must be integers
azp.plot_forest([idata1, idata2], model_names=["A", "B"], ...)

# ✅ — combine into single DataTree with model dim
combined = xr.concat([ds1, ds2], dim="model").assign_coords(model=["A", "B"])
dt = xr.DataTree.from_dict({"/posterior": xr.DataTree(combined)})
azp.plot_forest(dt, ...)
```

### B15/B16: Return type — PlotCollection, not axes

```python
# ❌ — TypeError: 'PlotCollection' object is not subscriptable
ax = azp.plot_forest(idata, ...); ax[0].set_title("...")

# ✅ — extract axes from PlotCollection
pc = azp.plot_forest(idata, ...)
ax = pc.viz["/"]["figure"].values.item().axes[0]
ax.set_title("...")
```

### B17: compact=True removed

```python
# ❌ — ValueError: extra keywords
azp.plot_trace(idata, compact=True)
# ✅ — remove
azp.plot_trace(idata)
```

### B18: group parameter for posterior_predictive

```python
# ❌ — KeyError: 'posterior'
azp.plot_dist(pp_idata)  # pm.sample_posterior_predictive result
# ✅
azp.plot_dist(pp_idata, group="posterior_predictive")
```

### B20: Missing xarray import

```python
# ❌ — NameError: name 'xr' is not defined
# ✅ — add to imports cell:
import xarray as xr
```

### B22: hdi_prob → prob, ci_bounds → ci_bound

```python
# ❌
az.hdi(data, hdi_prob=0.94)
hdi.sel(ci_bounds="lower"), hdi.sel(ci_bounds="higher")
# ✅
az.hdi(data, prob=0.94)
hdi.sel(ci_bound="lower"), hdi.sel(ci_bound="upper")
```

### B23: HDI integer indexing [0]/[1] → .sel(ci_bound=...)

```python
# ❌ — IndexError
hdi_result[0], hdi_result[1]
# ✅
hdi_result.sel(ci_bound="lower"), hdi_result.sel(ci_bound="upper")
```

### B24: az.plot_density removed

```python
# ❌
az.plot_density(data, ...)
# ✅
azp.plot_dist(data)
```

### B25: prob= in plot_dist/plot_forest (not same as ci_prob=)

```python
# ❌ — ValueError: extra keywords
azp.plot_dist(idata, prob=1)
# ✅ — prob= is NOT a valid kwarg for plot_dist/plot_forest
azp.plot_dist(idata)  # just remove prob=
```

---

## 5. Identifying Failing Notebooks from CI

The CI runs notebooks in batches on the `Test Notebooks` workflow:

1. Go to the PR checks tab on GitHub
2. Find entries like `notebooks (cvm, --notebooks clv)` with ❌
3. Click through to the job log
4. Search for `Error running notebook:` — this gives the notebook name
5. Search for `Exception encountered at "In [X]":` — this gives the cell number

Alternatively, find the run ID with `gh pr checks` or `gh run list`:

```bash
# List recent notebook CI runs
gh run list --branch chore/pymc6-migrate --workflow "Test Notebooks" --limit 5

# Get error details from a specific run
gh run view <run_id> --log --job <job_id> | grep -A10 'Exception encountered'
```

---

## 6. Pre-Push Checklist

Before pushing notebook changes:

1. ☐ Mock runner passes for ALL changed notebooks
2. ☐ If source code changed: real execution passes for affected notebooks
3. ☐ `ruff check --fix <files> && ruff format <files>` on changed files
4. ☐ `ruff check docs/source/notebooks/ --exclude 'docs/source/notebooks/*/dev/*'` passes
5. ☐ `ruff format --check docs/source/notebooks/` shows no reformats needed
6. ☐ `pre-commit run --all-files` passes locally
7. ☐ Push ONCE
