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
- [Scope Split: Major Release vs Follow-up](#scope-split-major-release-vs-follow-up)
- [PR Sequence](#pr-sequence)
- [Cross-Cutting Concerns](#cross-cutting-concerns)
- [Migration Guide Outline](#migration-guide-outline)
- [Issue Traceability](#issue-traceability)

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Decomposition approach | **D: Namespace Sub-Objects** | Best discoverability (`mmm.plot.sensitivity.analysis()`); aligns with 7 plotting families; each namespace is small and focused |
| arviz-plots adoption | **Included** in major release | Avoids a second breaking change for figure customization; II.7 is solved by exposing arviz-plots' native customization model. See [figure customization design](./2026-03-11-figure-customization-design.md) |
| Release strategy | **Major release** for breaking changes + decomposition; **follow-up minor releases** for missing features, performance, tests, docs | Keeps the major release focused and reviewable |
| Backward compatibility | **Hard break** — old method names stop working | Migration guide provides the full old→new mapping; no deprecation shims |
| PR strategy | **Family-by-family clean rooms** — one PR per namespace, each family is "born clean" | Focused, reviewable PRs (~300–700 lines each); parallelizable across contributors |
| Namespace constructor arg | **`data: MMMIDataWrapper` only** — no separate `idata` arg | `MMMIDataWrapper` already holds `idata` as a public attribute; passing both creates a redundant access path that undermines I.3 (all access through wrapper). `MMM.plot` already constructs `MMMPlotSuite(data=data)` without a separate `idata`. |

---

## Target Architecture

### File Layout

```
mmm/plotting/
    __init__.py              # re-exports MMMPlotSuite
    _helpers.py              # shared: subplot creation, color mapping, HDI bands, type defs
    suite.py                 # MMMPlotSuite — namespace wrapper with 7 sub-objects
    diagnostics.py           # DiagnosticsPlots namespace (4 methods)
    distributions.py         # DistributionPlots namespace (3 methods)
    saturation.py            # SaturationPlots namespace (2 methods)
    budget.py                # BudgetPlots namespace (2 methods)
    sensitivity.py           # SensitivityPlots namespace (3 methods)
    decomposition.py         # DecompositionPlots namespace (3 methods)
    cross_validation.py      # CrossValidationPlots namespace (3 methods)
```

### User Experience

```python
# Grouped discovery via tab-completion
mmm.plot.sensitivity.analysis(channels=["tv", "radio"])
mmm.plot.sensitivity.uplift(channel="tv")
mmm.plot.sensitivity.marginal(channel="tv")

mmm.plot.cv.predictions(results)
mmm.plot.cv.param_stability(results)
mmm.plot.cv.crps(results)

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

mmm.plot.decomposition.waterfall()
mmm.plot.decomposition.contributions_over_time()
mmm.plot.decomposition.channel_share_hdi()
```

### Suite Wrapper

```python
# mmm/plotting/suite.py
class MMMPlotSuite:
    def __init__(self, data: MMMIDataWrapper):

        self.diagnostics = DiagnosticsPlots(data)
        self.distributions = DistributionPlots(data)
        self.saturation = SaturationPlots(data)
        self.budget = BudgetPlots(data)
        self.sensitivity = SensitivityPlots(data)
        self.decomposition = DecompositionPlots(data)
        self.cv = CrossValidationPlots(data)
```

> **Design note:** Namespace classes receive only `data: MMMIDataWrapper`, not
> a separate `idata` argument. The wrapper already holds `idata` as a public
> attribute (`data.idata`), so passing it separately would create a redundant
> access path and undermine the rule that all data access goes through the
> wrapper (I.3). The real-world caller (`MMM.plot`) already constructs
> `MMMPlotSuite(data=data)` without a separate `idata`.

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

---

## Scope Split: Major Release vs Follow-up

### Major Release (29 issues)

**Breaking API changes:**

| ID | Issue | Change |
|----|-------|--------|
| I.1 | God class decomposition | 7 namespace sub-objects |
| I.2 | Constructor accepts None | Require valid `data` (wraps `idata`), fail-fast |
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
- `_validate_dims` that accepts the target dataset (IV.3)
- Contribution variable resolution helper (consolidates 5 strategies)
- arviz-plots imports and version compatibility checks

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

### PR 8 — Cross-Validation Namespace

**Methods:**
- `cv_predictions` → `mmm.plot.cv.predictions()`
- `param_stability` → `mmm.plot.cv.param_stability()`
- `cv_crps` → `mmm.plot.cv.crps()`

**Fixes included:**
- II.3 — Remove all `plt.show()` calls
- II.3 — `param_stability` combines all dimension values into a single multi-panel figure (one panel per dim value) instead of creating separate figures in a loop. Returns standard `tuple[Figure, NDArray[Axes]]`. Users can subset via `dims` to control figure size.
- II.1 — Fix `cv_predictions` wrapping axes in list instead of ndarray
- II.1 — Fix `param_stability` returning single `Axes` instead of `NDArray[Axes]`
- I.4 — Extract nested functions (`_align_y_to_df`, `_plot_hdi_from_sel`, `_pred_matrix_for_rows`, `_filter_rows_and_y`, `_plot_line`)

**LOE:** L

---

### PR 9 — Suite Wrapper + Cleanup

**Content:**
- `MMMPlotSuite` class with 7 namespace sub-objects
- Constructor validation: `data` required, must wrap valid `idata` (I.2)
- Remove dead `_cache` from `MMMIDataWrapper` (IV.11)
- Delete old `mmm/plot.py`
- Update `multidimensional.py` `.plot` property
- Update `time_slice_cross_validation.py` `.plot` property
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
| I.3 — All methods use `MMMIDataWrapper` | Namespace classes receive only `data: MMMIDataWrapper` (no separate `idata` arg); no raw `self._data.idata.posterior` access |
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
for better discoverability. All methods are now accessed through family
sub-objects on `mmm.plot`.

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
| `mmm.plot.cv_predictions()` | `mmm.plot.cv.predictions()` |
| `mmm.plot.param_stability()` | `mmm.plot.cv.param_stability()` |
| `mmm.plot.cv_crps()` | `mmm.plot.cv.crps()` |

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
- Constructor requires valid `data: MMMIDataWrapper` (no more `MMMPlotSuite(idata=None)`)
- All methods use arviz-plots `PlotCollection` internally
- Multi-backend support: pass `backend="plotly"` with `return_as_pc=True`

## Removed

- `saturation_curves_scatter` — use `mmm.plot.saturation.scatterplot()` instead
- `ax` parameter — use `plot_collection` for composing onto existing figures
```

---

## Issue Traceability

Every issue from the comprehensive audit mapped to its resolution:

| ID | Issue | Resolution | PR |
|----|-------|-----------|-----|
| I.1 | God class (5,150 lines) | Decompose into 7 namespace sub-objects | 2–9 |
| I.2 | Constructor allows invalid state | Require valid `data` (wraps `idata`) | 9 |
| I.3 | MMMIDataWrapper bypassed | All methods use wrapper | 1–8 |
| I.4 | Deep nested functions | Extract to module-level | 2–8 |
| I.5 | No shared color palette | `_helpers.channel_color_map()` | 1 |
| I.6 | Duplicated subplot logic / arviz-plots | All methods use `PlotCollection`; bar-plot methods fall back to matplotlib | 1–8 |
| II.1 | Return type roulette | `tuple[Figure, NDArray[Axes]]` always | 2–8 |
| II.2 | Inconsistent `original_scale` default | `True` everywhere | 4 |
| II.3 | `plt.show()` + figure discarding | Remove; return all figures | 8 |
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
