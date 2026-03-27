# PR3 — Distribution Diagnostics Design

> Design for adding `posterior` and `prior_vs_posterior` methods to `DiagnosticsPlots`.
> Prepared 2026-03-27.

---

## Context

The attack plan ([2026-03-11-mmmplotsuite-v2-attack-plan-design.md][attack-plan]) originally
assigned these methods to a separate `DistributionPlots` namespace (PR3).
Two decisions supersede that plan:

1. **No `DistributionPlots` class** — the methods are diagnostic in nature and belong
   directly in `DiagnosticsPlots`.
2. **`channel_parameter` dropped** — its functionality is covered by
   `posterior(var_names=[...])`.

PR3 therefore extends the existing `DiagnosticsPlots` class in `diagnostics.py`
(implemented in `isofer/time-series-plotting`) with two new methods, rebasing onto
that branch before starting work.

[attack-plan]: ../../plans/2026-03-11-mmmplotsuite-v2-attack-plan-design.md

---

## Migration impact

| Old API (v0.18) | New API (v0.19) |
|---|---|
| `mmm.plot.posterior_distribution(var)` | `mmm.plot.diagnostics.posterior(var_names=[var])` |
| `mmm.plot.channel_parameter(param_name)` | `mmm.plot.diagnostics.posterior(var_names=[param_name])` |
| `mmm.plot.prior_vs_posterior(var)` | `mmm.plot.diagnostics.prior_vs_posterior(var_names=[var])` |

---

## File changes

| File | Action |
|---|---|
| `pymc_marketing/mmm/plotting/diagnostics.py` | Add `posterior()` and `prior_vs_posterior()` methods to `DiagnosticsPlots` |
| `pymc_marketing/mmm/plotting/_helpers.py` | Update `_extract_matplotlib_result` to handle `azp.plot_dist`-style PCs |
| `tests/mmm/plotting/test_diagnostics.py` | Add `posterior_idata` fixture and `TestPosterior` / `TestPriorVsPosterior` classes |
| `tests/mmm/plotting/test_helpers.py` | Add fallback extraction test to `TestExtractMatplotlibResult` |

No new files are created. `distributions.py` is not created.

---

## `_helpers.py` — `_extract_matplotlib_result` update

`azp.plot_dist` and `azp.plot_prior_posterior` produce `PlotCollection` objects
whose `viz.ds` does not contain a `"plot"` key (unlike `PlotCollection.wrap()`).
The current helper raises a `KeyError` on these. Updated logic:

```python
def _extract_matplotlib_result(pc, return_as_pc):
    if return_as_pc:
        return pc
    fig = pc.viz.ds["figure"].item()
    axes = np.atleast_1d(np.array(fig.get_axes()))
    return fig, axes
```

`fig.get_axes()` works for both `PlotCollection.wrap()`-based and
`azp.plot_dist`-based PCs — no conditional needed. `residuals_distribution`
(PR2) currently does this inline; PR3 updates it to use the unified helper.

---

## Method: `posterior()`

### Purpose

Plot 1-D marginal KDE distributions for one or more posterior variables.
Thin wrapper around `azp.plot_dist`.

### Signature

```python
def posterior(
    self,
    var_names: list[str] | str | None = None,
    kind: str = "kde",
    group: str = "posterior",
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    visuals: dict[str, Any] | None = None,
    aes: dict[str, Any] | None = None,
    aes_by_visuals: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
```

### Behaviour

1. Resolve `data` from `idata` override or `self._data`.
2. Call `_process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)`.
3. Build `coords = _dims_to_sel_kwargs(dims)`.
4. Call:
   ```python
   pc = azp.plot_dist(
       data.idata,
       kind=kind,
       var_names=var_names,
       group=group,
       coords=coords,
       visuals=visuals,
       aes_by_visuals=aes_by_visuals,
       backend=backend,
       **({"aes": aes} if aes is not None else {}),
       **pc_kwargs,
   )
   ```
5. Return `_extract_matplotlib_result(pc, return_as_pc)`.

### Notes

- `var_names=None` plots all variables in the group (ArviZ default).
- `group="posterior"` by default; can be overridden to `"prior"` for a quick
  prior check without calling `prior_vs_posterior`.
- Validate that the requested group exists in `data.idata` before calling ArviZ;
  raise `ValueError` with an actionable message if missing (e.g.
  `"No posterior group found in idata. Fit the model first."`). This keeps
  error messages consistent with the rest of `DiagnosticsPlots`.

---

## Method: `prior_vs_posterior()`

### Purpose

Overlay prior and posterior 1-D marginal KDE distributions for one or more
variables. Thin wrapper around `azp.plot_prior_posterior`, which handles
the prior/posterior color legend automatically.

### Signature

```python
def prior_vs_posterior(
    self,
    var_names: list[str] | str | None = None,
    kind: str = "kde",
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    visuals: dict[str, Any] | None = None,
    aes: dict[str, Any] | None = None,
    aes_by_visuals: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
```

### Behaviour

1. Resolve `data`.
2. Validate that `data.idata` has both a `prior` and a `posterior` group;
   raise `ValueError` with actionable message if either is missing.
3. Call `_process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)`.
4. Build `coords = _dims_to_sel_kwargs(dims)`.
5. Call:
   ```python
   pc = azp.plot_prior_posterior(
       data.idata,
       kind=kind,
       var_names=var_names,
       coords=coords,
       visuals=visuals,
       aes_by_visuals=aes_by_visuals,
       backend=backend,
       **({"aes": aes} if aes is not None else {}),
       **pc_kwargs,
   )
   ```
6. Return `_extract_matplotlib_result(pc, return_as_pc)`.

### Validation messages

```
"No prior group found in idata. Run MMM.sample_prior_predictive() first."
"No posterior group found in idata. Fit the model first."
```

---

## Testing

### New fixture: `posterior_idata`

Module-scoped fixture added to `test_diagnostics.py`. Contains:
- `posterior` group with variables `alpha` (dims: `chain, draw, channel`) and
  `sigma` (dims: `chain, draw`)
- `prior` group with the same variables
- `constant_data` with `channel` coordinate

Channels: `["tv", "radio", "social"]`. This fixture is reused by both test
classes.

### `TestPosterior`

| Test | Assertion |
|---|---|
| `test_returns_figure_and_axes` | `isinstance(fig, Figure)` and `isinstance(axes, np.ndarray)` |
| `test_return_as_pc` | `isinstance(result, PlotCollection)` |
| `test_var_names_filters_variables` | `var_names=["alpha"]` → fewer axes than `var_names=None` |
| `test_group_prior` | `group="prior"` completes without error |
| `test_dims_filters_coords` | `dims={"channel": ["tv"]}` produces fewer panels than no filter |
| `test_raises_when_posterior_missing` | `ValueError` with message matching `"posterior"` |

### `TestPriorVsPosterior`

| Test | Assertion |
|---|---|
| `test_returns_figure_and_axes` | `isinstance(fig, Figure)` and `isinstance(axes, np.ndarray)` |
| `test_return_as_pc` | `isinstance(result, PlotCollection)` |
| `test_var_names_filters_variables` | `var_names=["alpha"]` → fewer axes than `var_names=None` |
| `test_dims_filters_coords` | `dims={"channel": ["tv"]}` produces fewer panels |
| `test_raises_when_prior_missing` | `ValueError` matching `"prior"` |
| `test_raises_when_posterior_missing` | `ValueError` matching `"posterior"` |

### `test_helpers.py` addition

One test in `TestExtractMatplotlibResult`: construct a mock PC where `viz.ds`
has `"figure"` but no `"plot"` key, verify the fallback returns axes from
`fig.get_axes()`.

---

## Constraints

- No method calls `plt.show()`.
- No method mutates `self._data`.
- `dims` → `coords` translation via `_dims_to_sel_kwargs` (already in `_helpers.py`).
- `aes` is passed as an explicit keyword merged into the arviz call, not via `**pc_kwargs`,
  to avoid silent omission.
- PR3 rebases onto `isofer/time-series-plotting` before any code is written.
