# MMMPlotSuite v2 — Attack Plan Design

> Design document for the major-release rewrite of `MMMPlotSuite`.
> Addresses 29 of the 45 issues identified in the
> [comprehensive audit](./2026-03-10-mmmplotsuite-comprehensive-issues.md).
> The remaining 16 issues ship in follow-up minor releases.
>
> Prepared 2026-03-11.

---

## Table of Contents

- [Decisions](#decisions)
- [Target Architecture](#target-architecture)
- [Standardized API Contract](#standardized-api-contract)
- [Helper Function Removal Plan](#helper-function-removal-plan)
- [Scope Split: Major Release vs Follow-up](#scope-split-major-release-vs-follow-up)
- [PR Sequence](#pr-sequence)
- [Cross-Cutting Concerns](#cross-cutting-concerns)
- [Migration Guide Outline](#migration-guide-outline)
- [Issue Traceability](#issue-traceability)

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Decomposition approach | **D: Namespace Sub-Objects** | Best discoverability (`mmm.plot.sensitivity.analysis()`); aligns with 6 plotting families; each namespace is small and focused |
| Suite separation | **Two separate suites: `MMMPlotSuite` + `MMMCVPlotSuite`** | 5 of 20 methods don't need `idata` at all (3 CV + 2 budget). CV methods share zero meaningful code with idata methods — only 4 subplot-plumbing helpers, all of which are replaced by arviz-plots. `mmm.plot` should not expose CV methods; `cv.plot` should not expose model-fit methods. Clean separation of concerns. |
| arviz-plots adoption | **Included** in major release | Avoids a second breaking change for figure customization; II.7 is solved by exposing arviz-plots' native customization model. See [figure customization design](./2026-03-11-figure-customization-design.md) |
| Release strategy | **Major release** for breaking changes + decomposition; **follow-up minor releases** for missing features, performance, tests, docs | Keeps the major release focused and reviewable |
| Backward compatibility | **Hard break** — old method names stop working | Migration guide provides the full old→new mapping; no deprecation shims |
| PR strategy | **Family-by-family clean rooms** — one PR per namespace, each family is "born clean" | Focused, reviewable PRs (~300–700 lines each); parallelizable across contributors |
| Namespace constructor arg | **`data: MMMIDataWrapper` only** — no separate `idata` arg | `MMMIDataWrapper` already holds `idata` as a public attribute; passing both creates a redundant access path that undermines I.3 (all access through wrapper). `MMM.plot` already constructs `MMMPlotSuite(data=data)` without a separate `idata`. Applies only to `MMMPlotSuite` namespaces; `MMMCVPlotSuite` takes no constructor data. |

---

## Target Architecture

### File Layout

```
mmm/plotting/
    __init__.py              # re-exports MMMPlotSuite, MMMCVPlotSuite
    _helpers.py              # shared: _process_plot_params, _extract_matplotlib_result,
                             #   _validate_dims, channel_color_map
    suite.py                 # MMMPlotSuite — namespace wrapper with 6 sub-objects
    cv_suite.py              # MMMCVPlotSuite — 3 flat CV methods, no data dependency
    diagnostics.py           # DiagnosticsPlots namespace (4 methods)
    distributions.py         # DistributionPlots namespace (3 methods)
    saturation.py            # SaturationPlots namespace (2 methods)
    budget.py                # BudgetPlots namespace (2 methods)
    sensitivity.py           # SensitivityPlots namespace (3 methods)
    decomposition.py         # DecompositionPlots namespace (3 methods)
```

### User Experience

```python
# MMMPlotSuite — model-fit plots (requires fitted model with idata)
mmm.plot.diagnostics.posterior_predictive(var_names=["y"])
mmm.plot.diagnostics.prior_predictive(var_names=["y"])
mmm.plot.diagnostics.residuals()
mmm.plot.diagnostics.residuals_distribution()

mmm.plot.distributions.posterior()
mmm.plot.distributions.channel_parameter()
mmm.plot.distributions.prior_vs_posterior()

mmm.plot.saturation.scatterplot()
mmm.plot.saturation.curves()

mmm.plot.budget.allocation(samples)
mmm.plot.budget.contribution_over_time(samples)

mmm.plot.sensitivity.analysis(channels=["tv", "radio"])
mmm.plot.sensitivity.uplift(channel="tv")
mmm.plot.sensitivity.marginal(channel="tv")

mmm.plot.decomposition.waterfall()
mmm.plot.decomposition.contributions_over_time()
mmm.plot.decomposition.channel_share_hdi()

# MMMCVPlotSuite — cross-validation plots (no model data needed)
cv.plot.predictions(results)
cv.plot.param_stability(results)
cv.plot.crps(results)
```

### Suite Wrappers

```python
# mmm/plotting/suite.py
class MMMPlotSuite:
    """Model-fit plotting. Requires valid MMMIDataWrapper."""
    def __init__(self, data: MMMIDataWrapper):
        self.diagnostics = DiagnosticsPlots(data)
        self.distributions = DistributionPlots(data)
        self.saturation = SaturationPlots(data)
        self.budget = BudgetPlots(data)
        self.sensitivity = SensitivityPlots(data)
        self.decomposition = DecompositionPlots(data)
```

```python
# mmm/plotting/cv_suite.py
class MMMCVPlotSuite:
    """Cross-validation plotting. No model data dependency.

    All data comes from the `results` argument on each method —
    the combined InferenceData returned by TimeSliceCrossValidator.run().
    """
    def predictions(self, results: az.InferenceData, ...) -> ...: ...
    def param_stability(self, results: az.InferenceData, ...) -> ...: ...
    def crps(self, results: az.InferenceData, ...) -> ...: ...
```

> **Design note — MMMPlotSuite:** Namespace classes receive only
> `data: MMMIDataWrapper`, not a separate `idata` argument. The wrapper
> already holds `idata` as a public attribute (`data.idata`), so passing
> it separately would create a redundant access path and undermine the
> rule that all data access goes through the wrapper (I.3).
>
> **Design note — MMMCVPlotSuite:** This suite has no constructor data
> dependency. The 3 CV methods receive all their plotting data via the
> `results` argument (the combined `InferenceData` from
> `TimeSliceCrossValidator.run()`). Analysis showed that these methods
> share zero domain-specific code with the idata-dependent methods — the
> only overlap was 4 subplot-plumbing helpers, all of which are replaced
> by arviz-plots' `PlotCollection`.
>
> **Design note — Budget methods on MMMPlotSuite:** `budget.allocation()`
> and `budget.contribution_over_time()` receive external data via their
> `samples` argument, but budget optimization always occurs in the
> context of a fitted model, so placing them on `MMMPlotSuite` is
> contextually correct.

### Caller Integration

```python
# multidimensional.py — returns MMMPlotSuite (requires fitted model)
class MMM:
    @property
    def plot(self) -> MMMPlotSuite:
        self._validate_model_was_built()
        self._validate_idata_exists()
        return MMMPlotSuite(data=self.data)

# time_slice_cross_validation.py — returns MMMCVPlotSuite (no data needed)
class TimeSliceCrossValidator:
    @property
    def plot(self) -> MMMCVPlotSuite:
        return MMMCVPlotSuite()
```

### Namespace Class Pattern

Each namespace follows the same structure (see [figure customization design](./2026-03-11-figure-customization-design.md)):

```python
# mmm/plotting/sensitivity.py
class SensitivityPlots:
    def __init__(self, data: MMMIDataWrapper):
        self._data = data

    def analysis(
        self,
        channels: list[str] | None = None,
        hdi_prob: float = 0.94,
        original_scale: bool = True,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        plot_collection: PlotCollection | None = None,
        backend: str | None = None,
        visuals: dict[str, Any] | None = None,
        aes_by_visuals: dict[str, list[str]] | None = None,
        return_as_pc: bool = False,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        ...
```

---

## Standardized API Contract

Every public method in every namespace must follow these rules. These resolve
issues II.1, II.2, II.5, II.6, and II.7.

### Return Type

`tuple[Figure, NDArray[Axes]]` by default. When `return_as_pc=True`, returns
`PlotCollection` for full arviz-plots composability. Even single-axes methods
wrap the axes in a 1-element ndarray when returning the tuple form.

### Required Parameters

Every method accepts at minimum (see [figure customization design](./2026-03-11-figure-customization-design.md) for full details):

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `dims` | `dict[str, Any] \| None` | `None` | Subset dimensions (e.g., `{"geo": ["CA", "NY"]}`) |
| `figsize` | `tuple[float, float] \| None` | `None` | Convenience for `figure_kwargs`; ignored with `plot_collection` |
| `plot_collection` | `PlotCollection \| None` | `None` | Plot onto existing arviz-plots collection |
| `backend` | `str \| None` | `None` | `"matplotlib"`, `"plotly"`, `"bokeh"` |
| `visuals` | `dict[str, Any] \| None` | `None` | Element-level customization (method-specific keys) |
| `aes_by_visuals` | `dict[str, list[str]] \| None` | `None` | Aesthetic mapping per visual element |
| `return_as_pc` | `bool` | `False` | Opt-in to `PlotCollection` return |
| `**pc_kwargs` | | | Forwarded to `PlotCollection.wrap()` or `.grid()` |

### Naming Conventions

| Old name | New name | Reason |
|----------|----------|--------|
| `var` (str or list) | `var_names` (list[str]) | ArviZ convention (II.5, #1751) |
| `hdi_probs` (plural, list) | `hdi_prob` (singular, float) | Consistency; single HDI level per call |
| `figsize: tuple[int, int]` | `figsize: tuple[float, float]` | Consistency across all methods |

### Defaults

| Parameter | Standard default |
|-----------|-----------------|
| `original_scale` | `True` everywhere (II.2) |
| `hdi_prob` | `0.94` |

### Behavioral Rules

- No method calls `plt.show()` (II.3)
- No method monkey-patches `self._data.idata` (II.4)
- No method bypasses `self._data` to access raw `self._data.idata.posterior` etc. (I.3)
- All data access goes through `MMMIDataWrapper` methods
- All methods use `PlotCollection` internally for rendering (I.6)
- Bar-plot methods that cannot use arviz-plots fall back to matplotlib with the same parameter signature (see [figure customization design](./2026-03-11-figure-customization-design.md#arviz-plots-coverage-gaps))

### MMMCVPlotSuite Contract

`MMMCVPlotSuite` methods follow the same standard customization parameters
as `MMMPlotSuite` methods (figsize, plot_collection, backend, visuals,
aes_by_visuals, return_as_pc, **pc_kwargs). Differences:

- No `self._data` — all plotting data comes from the `results` argument
- `dims` validation runs against the `results` data, not a model's `idata`
- `_validate_dims` is called as a standalone function from `_helpers.py`,
  passing the relevant dataset from `results` (not `self.idata.posterior`)

---

## Helper Function Removal Plan

The current `MMMPlotSuite` has ~15 private helper methods. With the move
to arviz-plots (`PlotCollection`) and the two-suite separation, most
subplot-scaffolding helpers become redundant. This section maps each
helper to its fate.

### Helpers replaced by arviz-plots

These helpers exist because the current code manually manages matplotlib
subplot grids, title generation, and dimension iteration. `PlotCollection`
handles all of this natively.

| Helper | Current purpose | Replaced by | Removed in PR |
|--------|----------------|-------------|---------------|
| `_init_subplots` | `fig, axes = plt.subplots(nrows, ncols, figsize)` with layout math | `PlotCollection.wrap()` / `.grid()` handles subplot creation, layout, and sizing | 2–8 (each family PR drops usage) |
| `_build_subplot_title` | Builds `"geo=CA, region=West"` from dim names + combo tuple | `PlotCollection.grid(cols=[...], rows=[...])` auto-labels panels from xarray coords | 2–8 |
| `_dim_list_handler` | Extracts dim keys/values, generates `itertools.product` combinations | `PlotCollection` handles dimension-aware layout natively via xarray coords | 2–8 |
| `_get_additional_dim_combinations` | Identifies non-standard dims and their cartesian product | Same as `_dim_list_handler` — subsumed by `PlotCollection` dimension handling | 2–8 |
| `_add_median_and_hdi` | Plots median line + HDI fill_between on an `Axes` | `pc.map("line", ...)` + `pc.map("hdi_band", fill_between_y, ...)` | 2 (diagnostics) |

### Helpers refactored to standalone functions in `_helpers.py`

These helpers perform validation or data manipulation that is independent
of the plotting library. They move from instance methods on `MMMPlotSuite`
to standalone functions.

| Helper | Current issue | New form in `_helpers.py` | PR |
|--------|--------------|--------------------------|-----|
| `_validate_dims` | Hardcoded to `self.idata.posterior.coords` (IV.3) | `_validate_dims(dataset: xr.Dataset, dims: dict)` — accepts target dataset as parameter. Usable by both suites. | 1 |
| `_filter_df_by_indexer` | Instance method only used by `cv_predictions` | Module-level function in `cv_suite.py` (CV-specific) | 8 |

### Helpers that stay (data access / computation)

These helpers perform data extraction or computation, not subplot
scaffolding. They move into their respective namespace classes or into
`MMMIDataWrapper`.

| Helper | Current purpose | New location | PR |
|--------|----------------|-------------|-----|
| `_get_posterior_predictive_data` | Retrieves `posterior_predictive` group | `DiagnosticsPlots` private method or inline (only 2 callers) | 2 |
| `_get_prior_predictive_data` | Retrieves `prior_predictive` group | `DiagnosticsPlots` private method or inline (only 2 callers) | 2 |
| `_compute_residuals` | Computes `y_pred - y_obs` (IV.18: hardcoded var) | `DiagnosticsPlots` private method; fix IV.18 | 2 |
| `_reduce_and_stack` | Sums leftover dims, stacks chain+draw (IV.12: silent sum) | Module-level in `decomposition.py` (only decomposition methods use it); fix IV.12 | 5 |
| `_spend_or_data_label` | Returns `"spend"` or `"data"` label | Inline in saturation namespace (trivial, 2 callers) | 4 |
| `_select_channel_x_for_indexers` | Picks spend vs raw channel data | Saturation namespace private method | 4 |
| `_plot_budget_allocation_bars` | Renders budget bar chart | `BudgetPlots` private method | 6 |
| `_prepare_allocated_contribution_data` | Extracts allocation data from samples | `BudgetPlots` private method | 6 |
| `_plot_single_allocated_contribution` | Renders single allocation panel | `BudgetPlots` private method | 6 |

### Summary

| Category | Count | Action |
|----------|-------|--------|
| Replaced by arviz-plots | 5 | Delete — `PlotCollection` handles subplot creation, layout, titles, dimension iteration |
| Refactored to standalone | 2 | Move to `_helpers.py` or `cv_suite.py` as module-level functions |
| Stay (data/computation) | 9 | Move into respective namespace classes |
| **Total** | **16** | |

---

## Scope Split: Major Release vs Follow-up

### Major Release (29 issues)

**Breaking API changes:**

| ID | Issue | Change |
|----|-------|--------|
| I.1 | God class decomposition | 6 namespace sub-objects on `MMMPlotSuite` + separate `MMMCVPlotSuite` |
| I.2 | Constructor accepts None | `MMMPlotSuite` requires valid `data` (wraps `idata`), fail-fast. `MMMCVPlotSuite` takes no data — methods receive `results` as argument. |
| II.1 | Return type roulette | Always `tuple[Figure, NDArray[Axes]]` |
| II.2 | Inconsistent `original_scale` default | `True` everywhere |
| II.3 | `plt.show()` in methods | Remove; `param_stability` uses multi-panel figure instead of per-dim loop |
| II.5 | Parameter naming inconsistencies | `var_names`, `hdi_prob` singular, `figsize: tuple[float, float]` |
| VII.1 | Coord name `x` → `channel` | Rename coordinate in `channel_share_hdi` |
| — | Remove `saturation_curves_scatter` | Already deprecated; hard removal with no shim (callers get `AttributeError`). Migration guide points to `saturation.scatterplot()` |

**Non-breaking, included during decomposition:**

| ID | Issue | Change |
|----|-------|--------|
| I.3 | MMMIDataWrapper bypassed | All methods use wrapper |
| I.4 | Deep nested functions | Extract to module-level or namespace methods |
| I.5 | No shared color palette | Shared channel→color mapping in `_helpers.py` |
| I.6 | Duplicated subplot logic / arviz-plots | All methods use `PlotCollection`; bar-plot methods fall back to matplotlib |
| II.4 | Monkey-patching idata | Pass data as parameter to shared helper |
| II.6 | No dims filtering on predictive/media | Add `dims` parameter |
| II.7 | Inconsistent figure customization | 6 standard params via arviz-plots (see [figure customization design](./2026-03-11-figure-customization-design.md)) |
| IV.1 | Copy-paste bug in prior_predictive | Fix error messages and docstrings |
| IV.2 | `plt.gcf()` fragility in `channel_share_hdi` | Replace with explicit figure reference |
| IV.3 | `_validate_dims` hardcoded to posterior | Validate against correct dataset |
| IV.4 | color_cycle iteration bug | Reset cycle per panel |
| IV.5 | title parameter shadowing | Rename local variable |
| IV.8 | Index-as-coordinate bug in waterfall | Use positional index |
| IV.9 | `**kwargs` + `**subplot_kwargs` conflict | Merge with conflict detection |
| IV.11 | Dead `_cache` in MMMIDataWrapper | Remove |
| IV.12 | `_reduce_and_stack` silent sum | Add warning for unknown dimensions |
| IV.13 | Redundant local imports | Remove |
| IV.14 | `ax` unpacking fragility | Handle multi-axes return |
| IV.15 | Seaborn dependency for 2 methods | Convert to lazy import |
| IV.16 | Error message formatting | Fix raw `\n` in f-strings |
| IV.17 | Dead `agg` parameter path | Remove dead code |
| IV.18 | `_compute_residuals` hardcoded var | Add optional parameter |

### Follow-up Minor Releases (16 issues)

| ID | Issue | Why deferred |
|----|-------|-------------|
| III.1 | Four legacy methods not in suite | Additive feature |
| III.2 | Gradient/HDI bands for posterior predictive | Additive enhancement (may already be done) |
| III.3 | Aggregated channel contributions | Additive option |
| III.4 | Out-of-sample plotting | New method |
| III.5 | Parametric fit overlay on saturation scatter | Additive feature |
| IV.6 | Per-date HDI computation loop | Performance optimization |
| IV.7 | O(n×m) loop in cv_crps | Performance optimization |
| IV.10 | Broad exception catching | Behavior change; needs careful rollout |
| V.1 | Methods with zero tests | Test-only |
| V.2 | Weak test assertions | Test-only |
| V.3 | No edge case tests | Test-only |
| V.4 | Legacy test migration | Test-only |
| V.5 | Thread-safety tests | Test-only |
| VI.1 | Plotting gallery | Documentation |
| VII.2 | Use `xarray.to_dataframe` more | Internal refactoring |
| VII.3 | Time-varying media visualization | New feature |

---

## PR Sequence

### PR 1 — Foundation

**Content:** `_helpers.py` with shared infrastructure.

**Includes:**
- `_process_plot_params()` — validates and normalizes the 6 standard customization params
- `_extract_matplotlib_result()` — converts `PlotCollection` to `tuple[Figure, NDArray[Axes]]`
- Shared channel→color mapping (I.5)
- `_validate_dims(dataset, dims)` as standalone function accepting target dataset (IV.3)
- Contribution variable resolution helper (consolidates 5 strategies)
- arviz-plots imports and version compatibility checks

**Does NOT include** (see [Helper Function Removal Plan](#helper-function-removal-plan)):
Old subplot-scaffolding helpers (`_init_subplots`, `_build_subplot_title`,
`_dim_list_handler`, `_get_additional_dim_combinations`, `_add_median_and_hdi`)
are not ported — `PlotCollection` replaces all of them. Each family PR
rewrites its methods using `PlotCollection` directly.

**Resolves:** I.5, I.6 (partially — infrastructure created; each family PR adopts it), IV.3 (partially)

**LOE:** L

---

### PR 2 — Time-Series Namespace (`diagnostics`)

**Methods:**
- `posterior_predictive` → `mmm.plot.diagnostics.posterior_predictive()`
- `prior_predictive` → `mmm.plot.diagnostics.prior_predictive()`
- `residuals_over_time` → `mmm.plot.diagnostics.residuals()`
- `residuals_posterior_distribution` → `mmm.plot.diagnostics.residuals_distribution()`

**Fixes included:**
- IV.1 — Copy-paste bug (wrong error messages in prior_predictive)
- IV.18 — Add optional parameter to `_compute_residuals`
- II.5 — `var` → `var_names` (list[str] for both methods)
- II.6 — Add `dims` parameter
- II.7 — Add 6 standard customization params (arviz-plots)

**LOE:** L (4 methods, establishes the pattern for all subsequent PRs)

---

### PR 3 — Distributions Namespace

**Methods:**
- `posterior_distribution` → `mmm.plot.distributions.posterior()`
- `channel_parameter` → `mmm.plot.distributions.channel_parameter()`
- `prior_vs_posterior` → `mmm.plot.distributions.prior_vs_posterior()`

**Fixes included:**
- II.1 — `channel_parameter` currently returns bare `Figure`; fix to standard return
- IV.15 — Lazy seaborn import (only `posterior_distribution` and `prior_vs_posterior`)
- II.7 — Add 6 standard customization params (arviz-plots)

**LOE:** M

---

### PR 4 — Saturation Namespace

**Methods:**
- `saturation_scatterplot` → `mmm.plot.saturation.scatterplot()`
- `saturation_curves` → `mmm.plot.saturation.curves()`
- Remove `saturation_curves_scatter` entirely (already deprecated; not carried into new namespace)

**Fixes included:**
- II.2 — `original_scale` default → `True`
- II.5 — `hdi_probs` → `hdi_prob` (singular, single float)
- IV.16 — Fix error message formatting (raw `\n`)
- II.6 — Add channel subsetting via `dims`

**LOE:** M

---

### PR 5 — Decomposition Namespace

**Methods:**
- `waterfall_components_decomposition` → `mmm.plot.decomposition.waterfall()`
- `contributions_over_time` → `mmm.plot.decomposition.contributions_over_time()`
- `channel_contribution_share_hdi` → `mmm.plot.decomposition.channel_share_hdi()`

**Fixes included:**
- IV.2 — Replace `plt.gcf()` with explicit figure reference
- IV.8 — Fix index-as-coordinate bug in waterfall
- IV.9 — Merge kwargs with conflict detection
- IV.12 — Add warning when `_reduce_and_stack` sums unknown dims
- IV.14 — Handle multi-axes return from `az.plot_forest`
- IV.17 — Remove dead `agg` parameter path
- II.5 — `figsize: tuple[int, int]` → `tuple[float, float]` in waterfall
- VII.1 — Rename coord `x` → `channel` in `channel_share_hdi`

**LOE:** L

---

### PR 6 — Budget Namespace

**Methods:**
- `budget_allocation` → `mmm.plot.budget.allocation()`
- `allocated_contribution_by_channel_over_time` → `mmm.plot.budget.contribution_over_time()`

**Fixes included:**
- II.1 — Standardize return type
- IV.9 — kwargs conflict detection
- II.7 — Add 6 standard customization params (arviz-plots)

**LOE:** M

---

### PR 7 — Sensitivity Namespace

**Methods:**
- `sensitivity_analysis` → `mmm.plot.sensitivity.analysis()`
- `uplift_curve` → `mmm.plot.sensitivity.uplift()`
- `marginal_curve` → `mmm.plot.sensitivity.marginal()`

**Fixes included:**
- II.4 — Replace monkey-patching with shared `_sensitivity_plot()` helper that accepts data as parameter
- IV.4 — Reset color cycle per panel
- IV.5 — Rename local `title` variable to avoid shadowing parameter
- IV.13 — Remove redundant `import warnings`
- II.1 — Return `tuple[Figure, NDArray[Axes]]` always (not bare `Axes`)

**LOE:** L

---

### PR 8 — MMMCVPlotSuite

**Content:** `cv_suite.py` — the separate cross-validation plot suite.

**Methods (flat on `MMMCVPlotSuite`, no namespace nesting):**
- `cv_predictions` → `cv.plot.predictions(results)`
- `param_stability` → `cv.plot.param_stability(results)`
- `cv_crps` → `cv.plot.crps(results)`

**Key design points:**
- `MMMCVPlotSuite` has no constructor data dependency — all plotting data
  comes from the `results: az.InferenceData` argument on each method
- `_validate_dims` called as standalone function from `_helpers.py`,
  passing the relevant dataset from `results` (not `self.idata.posterior`)
- `_filter_df_by_indexer` moves to `cv_suite.py` as a module-level function

**Fixes included:**
- II.3 — Remove all `plt.show()` calls
- II.3 — `param_stability` combines all dimension values into a single multi-panel figure (one panel per dim value) instead of creating separate figures in a loop. Returns standard `tuple[Figure, NDArray[Axes]]`. Users can subset via `dims` to control figure size.
- II.1 — Fix `cv_predictions` wrapping axes in list instead of ndarray
- II.1 — Fix `param_stability` returning single `Axes` instead of `NDArray[Axes]`
- I.4 — Extract nested functions (`_align_y_to_df`, `_plot_hdi_from_sel`, `_pred_matrix_for_rows`, `_filter_rows_and_y`, `_plot_line`) to module-level functions in `cv_suite.py`

**LOE:** L

---

### PR 9 — Suite Wrappers + Cleanup

**Content:**
- `MMMPlotSuite` class with 6 namespace sub-objects (no CV namespace)
- `MMMCVPlotSuite` re-exported from `__init__.py`
- Constructor validation: `MMMPlotSuite(data)` requires valid `MMMIDataWrapper` (I.2)
- Remove dead `_cache` from `MMMIDataWrapper` (IV.11)
- Delete old `mmm/plot.py`
- Update `multidimensional.py` `.plot` property → returns `MMMPlotSuite(data=self.data)`
- Update `time_slice_cross_validation.py` `.plot` property → returns `MMMCVPlotSuite()`
- Update all imports across the codebase

**LOE:** M

---

### PR 10 — Migration Guide

**Content:**
- Full old→new method name mapping table
- Breaking changes summary (return types, defaults, parameter renames)
- Code migration examples (before/after)
- Removed methods (`saturation_curves_scatter`)

**LOE:** S

---

### PR 11 — Test Overhaul

**Content:**
- Tests for each namespace class
- Richer assertions (axis labels, titles, legend entries, data values)
- Edge cases (single channel, single geo, missing data)
- Remove legacy `test_plotting.py` tests that duplicate new coverage

**LOE:** L

---

## Cross-Cutting Concerns

These are resolved in PR 1 (Foundation) and enforced by every subsequent PR:

| Concern | Resolution |
|---------|-----------|
| Suite separation | `MMMPlotSuite` for model-fit plots (requires `data`); `MMMCVPlotSuite` for CV plots (stateless). `mmm.plot` returns the former; `cv.plot` returns the latter. No cross-contamination. |
| Helper removal | 5 subplot-scaffolding helpers (`_init_subplots`, `_build_subplot_title`, `_dim_list_handler`, `_get_additional_dim_combinations`, `_add_median_and_hdi`) are **not ported** — `PlotCollection` replaces them. See [Helper Function Removal Plan](#helper-function-removal-plan). |
| I.3 — All methods use `MMMIDataWrapper` | `MMMPlotSuite` namespace classes receive only `data: MMMIDataWrapper` (no separate `idata` arg); no raw `self._data.idata.posterior` access. `MMMCVPlotSuite` has no `self._data` at all. |
| I.4 — No nested functions | Extract to module-level private functions or namespace private methods |
| I.5 — Shared color palette | `_helpers.channel_color_map(channels)` returns consistent channel→color dict |
| I.6 — arviz-plots adoption | All methods use `PlotCollection` internally; bar-plot methods fall back to matplotlib |
| II.1 — Standard return type | `tuple[Figure, NDArray[Axes]]` by default; `PlotCollection` opt-in via `return_as_pc=True` |
| II.2 — `original_scale=True` | Default on every method that exposes the parameter |
| II.5 — Consistent param names | `var_names`, `hdi_prob`, `figsize: tuple[float, float]` |
| II.6 — `dims` on all methods | Every method accepts `dims: dict[str, Any] | None` |
| II.7 — Figure customization | 6 standard params: `figsize`, `plot_collection`, `backend`, `visuals`, `aes_by_visuals`, `**pc_kwargs` (see [design](./2026-03-11-figure-customization-design.md)) |

---

## Migration Guide Outline

```markdown
# MMMPlotSuite Migration Guide (v0.18 → v0.19)

## Overview
MMMPlotSuite has been reorganized from a flat API into grouped namespaces
for better discoverability. Model-fit methods are accessed through family
sub-objects on `mmm.plot`. Cross-validation methods have moved to a
separate `MMMCVPlotSuite`, accessed via `cv.plot`.

## Method Name Mapping

| Old (v0.18) | New (v0.19) |
|-------------|-------------|
| `mmm.plot.posterior_predictive()` | `mmm.plot.diagnostics.posterior_predictive()` |
| `mmm.plot.prior_predictive()` | `mmm.plot.diagnostics.prior_predictive()` |
| `mmm.plot.residuals_over_time()` | `mmm.plot.diagnostics.residuals()` |
| `mmm.plot.residuals_posterior_distribution()` | `mmm.plot.diagnostics.residuals_distribution()` |
| `mmm.plot.posterior_distribution()` | `mmm.plot.distributions.posterior()` |
| `mmm.plot.channel_parameter()` | `mmm.plot.distributions.channel_parameter()` |
| `mmm.plot.prior_vs_posterior()` | `mmm.plot.distributions.prior_vs_posterior()` |
| `mmm.plot.saturation_scatterplot()` | `mmm.plot.saturation.scatterplot()` |
| `mmm.plot.saturation_curves()` | `mmm.plot.saturation.curves()` |
| `mmm.plot.saturation_curves_scatter()` | **Removed** (use `saturation.scatterplot`) |
| `mmm.plot.waterfall_components_decomposition()` | `mmm.plot.decomposition.waterfall()` |
| `mmm.plot.contributions_over_time()` | `mmm.plot.decomposition.contributions_over_time()` |
| `mmm.plot.channel_contribution_share_hdi()` | `mmm.plot.decomposition.channel_share_hdi()` |
| `mmm.plot.budget_allocation()` | `mmm.plot.budget.allocation()` |
| `mmm.plot.allocated_contribution_by_channel_over_time()` | `mmm.plot.budget.contribution_over_time()` |
| `mmm.plot.sensitivity_analysis()` | `mmm.plot.sensitivity.analysis()` |
| `mmm.plot.uplift_curve()` | `mmm.plot.sensitivity.uplift()` |
| `mmm.plot.marginal_curve()` | `mmm.plot.sensitivity.marginal()` |
| `mmm.plot.cv_predictions()` | `cv.plot.predictions()` (moved to `MMMCVPlotSuite`) |
| `mmm.plot.param_stability()` | `cv.plot.param_stability()` (moved to `MMMCVPlotSuite`) |
| `mmm.plot.cv_crps()` | `cv.plot.crps()` (moved to `MMMCVPlotSuite`) |

## Parameter Changes

| Old | New | Affected methods |
|-----|-----|-----------------|
| `var: str` or `var: list[str]` | `var_names: list[str]` | `posterior_predictive`, `prior_predictive` |
| `hdi_probs: list[float]` | `hdi_prob: float` | `saturation.curves` |
| `figsize: tuple[int, int]` | `figsize: tuple[float, float]` | `decomposition.waterfall` |

## Default Changes

| Parameter | Old default | New default | Affected methods |
|-----------|------------|-------------|-----------------|
| `original_scale` | `False` | `True` | `saturation.scatterplot`, `saturation.curves` |

## Return Type Changes

All methods now return `tuple[Figure, NDArray[Axes]]` by default.
Pass `return_as_pc=True` to get a `PlotCollection` for full arviz-plots
composability.

Previously inconsistent methods:
- `channel_parameter` returned bare `Figure`
- `sensitivity_analysis`, `uplift_curve`, `marginal_curve` could return bare `Axes`
- `budget_allocation`, `waterfall_components_decomposition` could return single `Axes`
- `param_stability` returned single `Axes` despite declaring `NDArray[Axes]`

## Figure Customization Changes

All methods now accept a consistent set of customization parameters
(see [figure customization design](./2026-03-11-figure-customization-design.md)):

| Old | New | Notes |
|-----|-----|-------|
| `ax: plt.Axes` (4 methods) | `plot_collection: PlotCollection` | arviz-plots composability |
| `**kwargs` (varies) | `visuals: dict` + `**pc_kwargs` | Element-level control via `visuals`; collection-level via `pc_kwargs` |
| `rc_params` (1 method) | `**pc_kwargs` | Pass as `figure_kwargs` in `pc_kwargs` |
| `subplot_kwargs` (1 method) | `**pc_kwargs` | Forwarded to `PlotCollection.wrap/grid` |
| No customization (7 methods) | Full standard params | All methods now have `figsize`, `backend`, `visuals`, etc. |

## Behavioral Changes

- No method calls `plt.show()` — you control when to display
- `param_stability` combines all dimension values into a single multi-panel figure (use `dims` to subset)
- `MMMPlotSuite` constructor requires valid `data: MMMIDataWrapper` (no more `MMMPlotSuite(idata=None)`)
- CV methods moved to separate `MMMCVPlotSuite` — access via `cv.plot` instead of `mmm.plot`
- `cv.plot` no longer exposes model-fit methods; `mmm.plot` no longer exposes CV methods
- All methods use arviz-plots `PlotCollection` internally
- Multi-backend support: pass `backend="plotly"` with `return_as_pc=True`

## Removed

- `saturation_curves_scatter` — use `mmm.plot.saturation.scatterplot()` instead
- `ax` parameter — use `plot_collection` for composing onto existing figures
- `MMMPlotSuite(idata=None)` pattern — CV methods no longer need it (they live on `MMMCVPlotSuite`)
```

---

## Issue Traceability

Every issue from the comprehensive audit mapped to its resolution:

| ID | Issue | Resolution | PR |
|----|-------|-----------|-----|
| I.1 | God class (5,150 lines) | Decompose into 6 namespace sub-objects (`MMMPlotSuite`) + separate `MMMCVPlotSuite` (3 methods) | 2–9 |
| I.2 | Constructor allows invalid state | `MMMPlotSuite` requires valid `data`; `MMMCVPlotSuite` is stateless (no data needed) | 8, 9 |
| I.3 | MMMIDataWrapper bypassed | All methods use wrapper | 1–8 |
| I.4 | Deep nested functions | Extract to module-level | 2–8 |
| I.5 | No shared color palette | `_helpers.channel_color_map()` | 1 |
| I.6 | Duplicated subplot logic / arviz-plots | All methods use `PlotCollection`; bar-plot methods fall back to matplotlib. 5 subplot-scaffolding helpers removed (see [Helper Function Removal Plan](#helper-function-removal-plan)). | 1–8 |
| II.1 | Return type roulette | `tuple[Figure, NDArray[Axes]]` always | 2–8 |
| II.2 | Inconsistent `original_scale` default | `True` everywhere | 4 |
| II.3 | `plt.show()` + figure discarding | Remove; return all figures | 8 (MMMCVPlotSuite) |
| II.4 | Monkey-patching idata | Shared helper with data parameter | 7 |
| II.5 | Parameter naming inconsistencies | `var_names`, `hdi_prob`, `figsize` types | 2–8 |
| II.6 | No dims filtering | Add `dims` on all methods | 2–8 |
| II.7 | Inconsistent figure customization | 6 standard params via arviz-plots (see [figure customization design](./2026-03-11-figure-customization-design.md)) | 2–8 |
| III.1–III.5 | Missing methods | **Deferred** — follow-up release | — |
| IV.1 | Copy-paste bug in prior_predictive | Fix messages and docstrings | 2 |
| IV.2 | `plt.gcf()` fragility | Explicit figure reference | 5 |
| IV.3 | `_validate_dims` hardcoded to posterior | Accept target dataset | 1 |
| IV.4 | color_cycle iteration bug | Reset per panel | 7 |
| IV.5 | title parameter shadowing | Rename local variable | 7 |
| IV.6 | Per-date HDI loop | **Deferred** — follow-up release | — |
| IV.7 | O(n×m) loop in cv_crps | **Deferred** — follow-up release | — |
| IV.8 | Index-as-coordinate bug | Use positional index | 5 |
| IV.9 | kwargs + subplot_kwargs conflict | Merge with conflict detection | 5, 6 |
| IV.10 | Broad exception catching | **Deferred** — follow-up release | — |
| IV.11 | Dead `_cache` in MMMIDataWrapper | Remove | 9 |
| IV.12 | `_reduce_and_stack` silent sum | Add warning | 5 |
| IV.13 | Redundant local imports | Remove | 7 |
| IV.14 | `ax` unpacking fragility | Handle multi-axes | 5 |
| IV.15 | Seaborn dependency for 2 methods | Lazy import | 3 |
| IV.16 | Error message formatting | Fix raw `\n` | 4 |
| IV.17 | Dead `agg` parameter | Remove dead code | 5 |
| IV.18 | `_compute_residuals` hardcoded var | Add optional parameter | 2 |
| V.1–V.5 | Test coverage gaps | **Deferred** (partially addressed in PR 11) | 11 |
| VI.1 | Plotting gallery | **Deferred** — follow-up release | — |
| VII.1 | Coord `x` → `channel` | Rename coordinate in `channel_share_hdi` | 5 |
| VII.2 | Use `xarray.to_dataframe` | **Deferred** — follow-up release | — |
| VII.3 | Time-varying media visualization | **Deferred** — follow-up release | — |
