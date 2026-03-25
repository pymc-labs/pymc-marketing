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

1. **Decompose the god class into namespace sub-objects.**
   The current `MMMPlotSuite` is a 5,150-line monolith with 20 public methods spanning
   6 unrelated plotting families. The rewrite decomposes it into 6 small namespace classes
   (`DiagnosticsPlots`, `DistributionPlots`, `TransformationPlots`, `BudgetPlots`,
   `SensitivityPlots`, `DecompositionPlots`), each mounted as a sub-object on `MMMPlotSuite`.
   Users access methods via `mmm.plot.<family>.<method>()` — e.g.,
   `mmm.plot.sensitivity.analysis()`. This gives IDE discoverability (autocomplete shows
   6 families, then only the methods in that family), keeps each namespace small and focused,
   and aligns the code layout with the 6 natural plotting families.
   See [Target Architecture](#target-architecture) for the file layout, user experience,
   and suite wrapper details.
   Analysis of all four decomposition approaches:
   [Approach D: Namespace Sub-Objects](./2026-03-10-mmmplotsuite-decomposition-approaches.md#approach-d-namespace-sub-objects-structured-facade).
   Discussion: [pymc-devs Discord thread](https://discord.com/channels/745261709622771773/948587000464834580/1481289904192356396).

2. **Separate cross-validation plots into their own suite (`MMMCVPlotSuite`).**
   Of the 20 current methods, 3 are cross-validation methods (`cv_predictions`,
   `param_stability`, `cv_crps`) that share zero domain-specific code with the other 17.
   They don't need `idata` or `MMMIDataWrapper` at all — they receive all data via a
   `results: az.InferenceData` argument (the output of `TimeSliceCrossValidator.run()`).
   The only overlap with the idata methods was 4 subplot-plumbing helpers, all of which
   are replaced by arviz-plots' `PlotCollection`. A separate `MMMCVPlotSuite` ensures
   `mmm.plot` never exposes CV methods and `cv.plot` never exposes model-fit methods.
   `MMMCVPlotSuite` is stateless — no constructor data, no `self._data`.
   See [Suite Wrappers](#suite-wrappers) for the class design,
   [MMMCVPlotSuite Contract](#mmmcvplotsuite-contract) for the API rules,
   [Caller Integration](#caller-integration) for how `cv.plot` is wired up,
   and [PR 8 — MMMCVPlotSuite](#pr-8--mmmcvplotsuite) for implementation details.
   Original issue: [I.2 Constructor allows invalid state](./2026-03-10-mmmplotsuite-comprehensive-issues.md#i2-constructor-allows-invalid-state).

3. **Adopt arviz-plots (`PlotCollection`) in the major release, not later.**
   All 17 `MMMPlotSuite` methods and 3 `MMMCVPlotSuite` methods will use `PlotCollection`
   internally for subplot creation, layout, dimension iteration, and rendering. This
   replaces 5 hand-rolled subplot-scaffolding helpers (`_init_subplots`,
   `_build_subplot_title`, `_dim_list_handler`, `_get_additional_dim_combinations`,
   `_add_median_and_hdi`). Bundling this with the major release avoids imposing a second
   breaking change later for figure customization. It also solves inconsistent figure
   customization (issue II.7) by exposing a consistent customization model: 5
   standard parameters (`figsize`, `backend`, per-element `*_kwargs`, `return_as_pc`,
   `**pc_kwargs`) on every method. Each visual element gets its own explicit keyword
   argument (e.g., `scatter_kwargs`, `hdi_kwargs`, `line_kwargs`) rather than a
   single opaque `visuals` dict — this makes available kwargs discoverable via IDE
   autocomplete. Bar-plot methods that cannot use arviz-plots fall back to matplotlib
   with the same parameter signature.
   See [Helper Function Removal Plan](#helper-function-removal-plan) for what `PlotCollection` replaces.
   Full customization API: [figure customization design](./2026-03-11-figure-customization-design.md).
   Original issues:
   [I.6 Duplicated subplot logic](./2026-03-10-mmmplotsuite-comprehensive-issues.md#i6-duplicated-subplot-creation-and-population-logic--delegate-to-arviz-plots),
   [II.7 Inconsistent figure customization](./2026-03-10-mmmplotsuite-comprehensive-issues.md#ii7-inconsistent-figure-customization-surface-github-822-2378).
   Prior art: [Stale Branch Assessment](./2026-03-10-mmmplotsuite-comprehensive-issues.md#viii-stale-branch-assessment-featuremmmplotsuite-arviz).

4. **Pass only `data: MMMIDataWrapper` to namespace constructors — no separate `idata` arg.**
   Each `MMMPlotSuite` namespace class (`DiagnosticsPlots`, etc.) receives a single
   `data: MMMIDataWrapper` in its constructor. `MMMIDataWrapper` is a typed wrapper
   around `az.InferenceData` that provides validated, domain-aware access to model
   results. Instead of manually navigating `idata.posterior.channel_contribution`,
   `idata.constant_data.channel_data`, etc. (which requires knowing the exact variable
   names and group locations), the wrapper exposes semantic methods like
   `data.get_contributions(original_scale=True)`, `data.get_channel_spend()`,
   `data.get_target()`, and `data.get_channel_scale()`. It also handles scale
   conversions, contribution variable resolution (which has 5 different strategies in
   the current code), and coordinate validation. The wrapper already holds `idata` as a
   public attribute (`data.idata`) for cases where raw access is needed, so accepting a
   separate `idata` would create a redundant access path and undermine the rule that all
   data access goes through the wrapper. The `MMM.plot` property already constructs
   `MMMPlotSuite(data=self.data)` without a separate `idata`. This decision applies only
   to `MMMPlotSuite` namespaces; `MMMCVPlotSuite` takes no constructor data at all
   (see decision 2).
   See [Suite Wrappers](#suite-wrappers) design notes,
   [Namespace Class Pattern](#namespace-class-pattern) for the concrete `__init__` signature,
   and [Caller Integration](#caller-integration) for how `MMM.plot` constructs the suite.
   Original issue: [I.3 MMMIDataWrapper largely bypassed](./2026-03-10-mmmplotsuite-comprehensive-issues.md#i3-mmmdatawrapper-largely-bypassed).

5. **Every data-dependent method accepts an `idata` override parameter.**
   All 17 `MMMPlotSuite` methods that access `self._data` accept
   `idata: az.InferenceData | None = None`. When provided, the method constructs
   `MMMIDataWrapper(idata, schema=self._data.schema)` and uses that local wrapper for
   all subsequent access — both `data.idata` and wrapper helpers like
   `data.get_channel_spend()`. Schema propagation is required so the override wrapper
   knows the model's variable naming conventions (e.g., contribution variable names,
   dimension structure). This makes individual methods fully reusable with different
   fitted models (e.g., comparing two model fits side-by-side) without constructing a
   new suite instance. The parameter type is `az.InferenceData` (not `MMMIDataWrapper`)
   because that is the object users have in hand — the wrapper construction is an
   internal detail. Currently only `posterior_predictive` and `prior_predictive` have
   an ad-hoc `idata` override; this decision standardizes the pattern across all 17
   data-dependent methods. Resolution at the top of every method:
   `data = MMMIDataWrapper(idata, schema=self._data.schema) if idata is not None else self._data`.
   See [Standardized API Contract](#standardized-api-contract) for the full parameter table,
   [Behavioral Rules](#behavioral-rules) for the resolution pattern,
   and [Namespace Class Pattern](#namespace-class-pattern) for a concrete method signature example.

6. **Hard break — old method names stop working, no deprecation shims.**
   The old flat API (`mmm.plot.posterior_predictive()`, etc.) is removed entirely.
   Calling removed names raises `AttributeError`. The old import path
   (`from pymc_marketing.mmm.plot import MMMPlotSuite`) is replaced with a stub that
   raises `ImportError` with an actionable migration message. A full migration guide
   provides the complete old-to-new method mapping, parameter renames, default changes,
   and before/after code examples. No deprecation warnings or shims — this is a major
   version bump.
   See [Migration Guide Outline](#migration-guide-outline) for the full old-to-new mapping.

7. **Major release for breaking changes; follow-up minors for additive work.**
   All breaking API changes (decomposition, return-type standardization, parameter renames,
   default changes, method removals) ship together in one major release. Every PR in the
   major release ships with its own tests — no test debt at ship time. Additive features
   (missing methods, performance optimizations, edge-case tests, plotting gallery, docs)
   are deferred to follow-up minor releases to keep the major release focused and reviewable.
   See [Scope Split: Major Release vs Follow-up](#scope-split-major-release-vs-follow-up)
   for the full 29-issue / 16-issue breakdown.

8. **Family-by-family clean-room PRs — one PR per namespace, each born clean with tests.**
   Each of the 6 namespace families gets its own PR (PRs 2–7), plus one for
   `MMMCVPlotSuite` (PR 8). Every family PR writes its namespace class from scratch
   (no incremental refactoring of the old monolith), ships code and tests together,
   and is independently reviewable. PR 4 (Transformations) ships first as the template —
   it demonstrates every structural pattern (namespace class, standard params,
   `PlotCollection`, `idata` override, tests) in a concrete, copy-able form.
   Remaining family PRs follow the template in parallel.
   See [PR Sequence](#pr-sequence) for the dependency graph, LOE estimates, and
   per-PR scope.

9. **`MMMPlotlyFactory` is out of scope for this release.**
   The new `backend` parameter on every method provides a migration path to Plotly via
   arviz-plots' native multi-backend support. Deprecation of the existing
   `MMMPlotlyFactory` class can be evaluated once arviz-plots' Plotly backend stabilizes.
   No changes to `MMMPlotlyFactory` in this release.

---

## Target Architecture

### File Layout

```
mmm/plotting/
    __init__.py              # re-exports MMMPlotSuite, MMMCVPlotSuite
    _helpers.py              # shared: _process_plot_params, _extract_matplotlib_result,
                             #   _validate_dims, _dims_to_sel_kwargs, _select_dims,
                             #   channel_color_map
    suite.py                 # MMMPlotSuite — namespace wrapper with 6 sub-objects
    cv_suite.py              # MMMCVPlotSuite — 3 flat CV methods, no data dependency
    diagnostics.py           # DiagnosticsPlots namespace (4 methods)
    distributions.py         # DistributionPlots namespace (3 methods)
    transformations.py       # TransformationPlots namespace (2 saturation methods; future home for adstock plots)
    budget.py                # BudgetPlots namespace (2 methods)
    sensitivity.py           # SensitivityPlots namespace (3 methods)
    decomposition.py         # DecompositionPlots namespace (3 methods)
```

### User Experience

**Basic — namespace access (all methods, default settings):**

```python
# MMMPlotSuite — model-fit plots (requires fitted model with idata)
mmm.plot.diagnostics.posterior_predictive(target_var="y")
mmm.plot.diagnostics.prior_predictive(target_var="y")
mmm.plot.diagnostics.residuals()
mmm.plot.diagnostics.residuals_distribution()

mmm.plot.distributions.posterior()
mmm.plot.distributions.channel_parameter()
mmm.plot.distributions.prior_vs_posterior()

mmm.plot.transformations.saturation_scatterplot()
mmm.plot.transformations.saturation_curves()

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
        self.transformations = TransformationPlots(data)
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
> **Design note — method-level `idata` override:** Every method that
> accesses `self._data` accepts `idata: az.InferenceData | None = None`
> and resolves `data = MMMIDataWrapper(idata, schema=self._data.schema) if idata is not None else self._data`
> at the top. Schema propagation is required so the override wrapper knows
> the model's variable naming conventions. All subsequent access in the
> method goes through the resolved local `data` — both `data.idata` and
> wrapper helpers like `data.get_channel_spend()`. This makes individual
> methods fully reusable with different fitted models (e.g., comparing two
> model fits side-by-side) without constructing a new suite instance.
> Currently only `posterior_predictive` and `prior_predictive` have an
> ad-hoc `idata` override; this decision standardizes the pattern across
> all 17 data-dependent methods. The parameter is `az.InferenceData` (not
> `MMMIDataWrapper`) because that is the object users have in hand —
> the wrapper construction is an internal detail.
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
class SomethingPlots:
    def __init__(self, data: MMMIDataWrapper):
        self._data = data


    def method(
        self,
        # 1. Method-specific data params (varies per method)
        idata: az.InferenceData | None = None,
        channels=None,
        hdi_prob=0.94,
        original_scale=True,
        # 2. Dimension subsetting (standard)
        dims=None,
        # 3. Figure customization (standard — identical across all methods)
        figsize=None,
        backend=None,
        # 4. Return control (standard)
        return_as_pc=False,
        # 5. Per-element visual kwargs (method-specific)
        scatter_kwargs=None,
        hdi_kwargs=None,
        # ... other *_kwargs matching the method's visual elements
        # 6. PlotCollection kwargs catch-all (standard)
        **pc_kwargs,
    ):
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
| `idata` | `az.InferenceData \| None` | `None` | Override instance data; method resolves `data = MMMIDataWrapper(idata, schema=self._data.schema) if idata is not None else self._data`. Only on `MMMPlotSuite` methods that access `self._data`. |
| `dims` | `dict[str, Any] \| None` | `None` | Subset dimensions (e.g., `{"geo": ["CA", "NY"]}`) |
| `figsize` | `tuple[float, float] \| None` | `None` | Passed to `PlotCollection` figure creation |
| `backend` | `str \| None` | `None` | `"matplotlib"`, `"plotly"`, `"bokeh"` |
| `<element>_kwargs` | `dict[str, Any] \| None` | `None` | Per-element visual kwargs — one param per visual element (method-specific names, e.g., `scatter_kwargs`, `hdi_kwargs`, `line_kwargs`). Forwarded directly to the corresponding `azp.visuals.*` call. |
| `return_as_pc` | `bool` | `False` | Opt-in to `PlotCollection` return |
| `**pc_kwargs` | | | Forwarded to `PlotCollection.wrap()` or `.grid()` |

### Naming Conventions

| Old name | New name | Reason |
|----------|----------|--------|
| `var` (predictive methods) | `target_var: str = "y"` | Identifies target variable; not a filter (II.5) |
| `var` (single posterior selector) | `var_name: str` | Singular; method accepts exactly one variable (II.5) |
| `var` / `parameter` / `param_name` (multi posterior filter) | `var_names: list[str]` | ArviZ convention; multi-select filter (II.5) |
| `hdi_probs` (plural, list) | `hdi_prob` (singular, float) | Consistency; single HDI level per call |
| `figsize: tuple[int, int]` | `figsize: tuple[float, float]` | Consistency across all methods |

### Defaults

| Parameter | Standard default |
|-----------|-----------------|
| `original_scale` | `True` everywhere (II.2) |
| `hdi_prob` | `0.94` |

### Examples

**Figsize, domain params, and dimension subsetting:**

`figsize` and `dims` are standard parameters on every method (see
[Required Parameters](#required-parameters)).

```python
# figsize replaces the old plt.rcParams["figure.figsize"] global pattern
fig, axes = mmm.plot.diagnostics.posterior_predictive(figsize=(14, 6))

# Geo-level model: subset to specific markets with dims
fig, axes = mmm.plot.decomposition.waterfall(
    dims={"geo": ["CA", "NY"]}, figsize=(12, 8)
)
```

**Visual element customization:**

```python
# Style individual visual elements with per-element kwargs
# Each *_kwargs dict is forwarded directly to the corresponding azp.visuals.* call
fig, axes = mmm.plot.diagnostics.posterior_predictive(
    line_kwargs={"color": "darkblue", "linewidth": 2},
    hdi_kwargs={"alpha": 0.15},
    observed_kwargs={"marker": "o", "s": 12, "color": "black"},
)
```

**idata override — plot with a different fitted model's data:**

```python
other_idata: az.InferenceData = other_mmm.idata
mmm.plot.diagnostics.posterior_predictive(idata=other_idata)
mmm.plot.decomposition.waterfall(idata=other_idata)
```

**PlotCollection — get back for post-processing:**

```python
# Get PlotCollection back for further manipulation via return_as_pc=True
pc = mmm.plot.sensitivity.analysis(
    channels=["tv", "radio"], return_as_pc=True
)
pc.map(azp.visuals.line_xy, x=reference_x, y=reference_y, color="red", linestyle="--")
```

**Backend and layout kwargs:**

```python
# Render with Plotly instead of matplotlib (requires return_as_pc=True)
pc = mmm.plot.sensitivity.analysis(
    channels=["tv", "radio"],
    backend="plotly",
    return_as_pc=True,
)

# pc_kwargs flow through to PlotCollection for layout control
fig, axes = mmm.plot.sensitivity.analysis(
    channels=["tv", "radio", "social", "search"],
    cols=["channel"],
    col_wrap=2, # <- goes through pc_kwargs
    figsize=(14, 10),
)
```

### Behavioral Rules

- No method calls `plt.show()` (II.3)
- No method monkey-patches `self._data.idata` (II.4)
- No method bypasses `self._data` to access raw `self._data.idata.posterior` etc. (I.3)
- All data access goes through `MMMIDataWrapper` methods
- All methods use `PlotCollection` internally for rendering (I.6)
- Bar-plot methods that cannot use arviz-plots fall back to matplotlib with the same parameter signature (see [figure customization design](./2026-03-11-figure-customization-design.md#arviz-plots-coverage-gaps))
- Every method that accesses `self._data` resolves `data = MMMIDataWrapper(idata, schema=self._data.schema) if idata is not None else self._data` at the top of the method body. All subsequent access in the method uses the resolved local `data` (both `data.idata` and wrapper helpers like `data.get_channel_spend()`), never `self._data` directly.

### MMMCVPlotSuite Contract

`MMMCVPlotSuite` methods follow the same standard customization parameters
as `MMMPlotSuite` methods (`figsize`, `backend`, per-element `*_kwargs`,
`return_as_pc`, `**pc_kwargs`). Differences:

- No `self._data` — all plotting data comes from the `results` argument
- **No `idata` override parameter** — since there is no `self._data`, the `idata` parameter does not apply. CV methods already receive all their data via the `results` argument.
- `dims` validation runs against the `results` data, not a model's `idata`
- `_validate_dims` is called as a standalone function from `_helpers.py`,
  passing the relevant dataset from `results` (not `self.idata.posterior`)

---

## Helper Function Removal Plan

The current `MMMPlotSuite` has 16 private helper methods. With the move
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
| `_dims_to_sel_kwargs` | Not present | New: converts validated `dims` dict to `.sel()` kwargs. Can be used independently when only the conversion step is needed. | 1 |
| `_select_dims` | Not present | New: combines `_validate_dims` + `_dims_to_sel_kwargs` + `.sel()` into a single call. Accepts both `xr.Dataset` and `xr.DataArray`. Use `_select_dims(data, dims)` instead of the three-step pattern. | 1 |
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
| `_spend_or_data_label` | Returns `"spend"` or `"data"` label | Inline in transformations namespace (trivial, 2 callers) | 4 |
| `_select_channel_x_for_indexers` | Picks spend vs raw channel data | Transformations namespace private method | 4 |
| `_plot_budget_allocation_bars` | Renders budget bar chart | `BudgetPlots` private method | 6 |
| `_prepare_allocated_contribution_data` | Extracts allocation data from samples | `BudgetPlots` private method | 6 |
| `_plot_single_allocated_contribution` | Renders single allocation panel | `BudgetPlots` private method | 6 |

### Summary

| Category | Count | Action |
|----------|-------|--------|
| Replaced by arviz-plots | 5 | Delete — `PlotCollection` handles subplot creation, layout, titles, dimension iteration |
| Refactored to standalone | 2 | Move to `_helpers.py` or `cv_suite.py` as module-level functions |
| Stay (data/computation) | 9 | Move into respective namespace classes |
| **Total (existing helpers)** | **16** | |
| New standalone functions (PR 1) | 2 | `_dims_to_sel_kwargs`, `_select_dims` — new functions added to `_helpers.py` |

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
| II.5 | Parameter naming inconsistencies | `target_var` (predictive), `var_name` (single posterior), `var_names` (multi posterior); `hdi_prob` singular; `figsize: tuple[float, float]` |
| VII.1 | Coord name `x` → `channel` | Rename coordinate in `channel_share_hdi` |
| — | Remove `saturation_curves_scatter` | Already deprecated; hard removal with no shim (callers get `AttributeError`). Migration guide points to `transformations.saturation_scatterplot()` |

**Non-breaking, included during decomposition:**

| ID | Issue | Change |
|----|-------|--------|
| I.3 | MMMIDataWrapper bypassed | All methods use wrapper |
| I.4 | Deep nested functions | Extract to module-level or namespace methods |
| I.5 | No shared color palette | Shared channel→color mapping in `_helpers.py` |
| I.6 | Duplicated subplot logic / arviz-plots | All methods use `PlotCollection`; bar-plot methods fall back to matplotlib |
| II.4 | Monkey-patching idata | Pass data as parameter to shared helper |
| II.6 | No dims filtering on predictive/media | Add `dims` parameter |
| II.7 | Inconsistent figure customization | 5 standard params on every method: `figsize`, `backend`, per-element `*_kwargs`, `return_as_pc`, `**pc_kwargs` (see [figure customization design](./2026-03-11-figure-customization-design.md)) |
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
| V.1 | Methods with zero tests | Partially addressed in PRs 2–8 (each ships with tests); follow-up for comprehensive coverage |
| V.2 | Weak test assertions | Partially addressed in PRs 2–8; follow-up for exhaustive checks |
| V.3 | No edge case tests | Follow-up: single channel, single geo, missing data, etc. |
| V.4 | Legacy test migration | Addressed in PR 9 (legacy `test_plotting.py` removal) |
| V.5 | Thread-safety tests | Follow-up: verify `PlotCollection`-based methods are thread-safe |
| VI.1 | Plotting gallery | Documentation |
| VII.2 | Use `xarray.to_dataframe` more | Internal refactoring |
| VII.3 | Time-varying media visualization | New feature |

---

## PR Sequence

> **LOE scale:** S (1–4 hours) · M (1–3 days) · L (1–2 weeks).
> See [comprehensive audit](./2026-03-10-mmmplotsuite-comprehensive-issues.md) for full scale.

**Dependency graph:**

```
PR 1 (Foundation)
 ├→ PR 4 (Transformations — template PR, ships first)
 │   └→ PRs 2, 3, 5, 6, 7 (parallel, follow PR 4's established pattern)
 ├→ PR 8 (MMMCVPlotSuite — parallel with PR 4; different architecture, no shared template)
 │
 All family PRs (2–8) → PR 9 (Suite Wrappers + Cleanup)
                          └→ PR 10 (Migration Guide)
```

**Why PR 4 is the template:** One family PR must ship first to prove
the structural pattern (namespace class, standard params, `PlotCollection`,
tests) in production code. PR 4 is the best candidate: M LOE (2 methods,
~471 lines) ships faster than PR 2's L LOE (4 methods, ~512 lines);
covers both core plot types (line + HDI band, scatter); and has minimal
domain-specific complications — the work is almost entirely the
structural pattern itself.

PR 8 (`MMMCVPlotSuite`) is independently parallel because it has a
fundamentally different architecture — no `self._data`, all plotting
data arrives via the `results` argument, and it's a separate suite
class. It doesn't benefit from the same template as PRs 2–7.
**Testing strategy:** Every family PR (2–8) ships with its own tests —
return-type checks, axis-count assertions, parameter variations, and
basic edge cases. PR 9 removes legacy `test_plotting.py` tests and
adds parametrized cross-cutting tests that verify the standard API contract
(return type, standard params, `dims` filtering) across all namespaces.

### PR 1 — Foundation

**Content:** `_helpers.py` with shared infrastructure.

**Includes:**
- `_process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)` — validates and normalizes the 5 standard customization params
- `_extract_matplotlib_result()` — converts `PlotCollection` to `tuple[Figure, NDArray[Axes]]`
- Shared channel→color mapping (I.5)
- `_validate_dims(dataset, dims)` as standalone function accepting target dataset (IV.3)
- `_dims_to_sel_kwargs(dataset, dims)` — converts validated `dims` dict to `.sel()` kwargs
- `_select_dims(data, dims)` — combines validate + convert + `.sel()` in one call; accepts `xr.Dataset` or `xr.DataArray`
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
- II.5 — `var` → `target_var: str = "y"` for `posterior_predictive` and `prior_predictive`;
         tighten `posterior_predictive` from `list[str] | None` to `str`
- II.6 — Add `dims` parameter
- II.7 — Add 5 standard customization params (`figsize`, `backend`, per-element `*_kwargs`, `return_as_pc`, `**pc_kwargs`)

**LOE:** L (4 methods; follows the pattern established by PR 4)

---

### PR 3 — Distributions Namespace

**Methods:**
- `posterior_distribution` → `mmm.plot.distributions.posterior()`
- `channel_parameter` → `mmm.plot.distributions.channel_parameter()`
- `prior_vs_posterior` → `mmm.plot.distributions.prior_vs_posterior()`

**Fixes included:**
- II.1 — `channel_parameter` currently returns bare `Figure`; fix to standard return
- IV.15 — Lazy seaborn import (only `posterior_distribution` and `prior_vs_posterior`)
- II.7 — Add 5 standard customization params (`figsize`, `backend`, per-element `*_kwargs`, `return_as_pc`, `**pc_kwargs`)
- II.5 — `var: str` → `var_name: str` in `posterior_distribution` and `prior_vs_posterior`;
         `param_name: str` → `var_name: str` in `channel_parameter`

**LOE:** M

---

### PR 4 — Transformations Namespace ⟵ template PR

> **Template role:** This is the first family PR to land. It establishes
> the concrete reference implementation that PRs 2, 3, 5, 6, 7 follow.
> The PR must demonstrate every structural pattern in a reviewable,
> copy-able form:
>
> 1. Namespace class structure (`class TransformationPlots`, `__init__(self, data)`, `self._data`)
> 2. Full standard signature: `idata`, `dims`, `figsize`, `backend`, `return_as_pc`, per-element `*_kwargs` (method-specific), `**pc_kwargs`
> 3. `data = MMMIDataWrapper(idata, schema=self._data.schema) if idata is not None else self._data` resolution at the top of each method
> 4. `_process_plot_params()` call to validate/normalize params
> 5. `_select_dims(data, dims)` call — combined validate + filter in one step
> 6. `PlotCollection.grid()` or `.wrap()` creation from xarray data — method chooses best fit
> 7. Native `azp.visuals.*` functions for rendering (`scatter_xy`, `line_xy`, `fill_between_y`, `labelled_*`); custom module-level callbacks only when no native visual exists
> 8. Per-element `*_kwargs` forwarded directly to the corresponding `azp.visuals.*` call
> 9. DRY composition via `return_as_pc=True`: `saturation_curves` calls `saturation_scatterplot(return_as_pc=True)` to reuse the scatter layer instead of duplicating setup code
> 10. `_extract_matplotlib_result()` conversion to `tuple[Figure, NDArray[Axes]]`
> 11. All access exclusively through resolved local `data` (no `self._data` after resolution)
> 12. Test file with return-type checks, axis-count assertions, `dims` filtering, `idata` override, `return_as_pc=True`

**Methods:**
- `saturation_scatterplot` → `mmm.plot.transformations.saturation_scatterplot()`
- `saturation_curves` → `mmm.plot.transformations.saturation_curves()`
- Remove `saturation_curves_scatter` entirely (already deprecated; not carried into new namespace)

**Fixes included:**
- II.2 — `original_scale` default → `True`
- II.5 — `hdi_probs` → `hdi_prob` (singular, single float); `curve` (singular) → `curves` (plural, reflects that the DataArray contains multiple posterior samples)
- IV.16 — Fix error message formatting (raw `\n`)
- II.6 — Add channel subsetting via `dims`

**Additional patterns introduced in this PR (available to subsequent family PRs):**
- `_ensure_chain_draw_dims` — module-level helper in `transformations.py` that normalises curve dimension format: `(chain, draw, ...)` returned as-is; `sample` MultiIndex over `(chain, draw)` unstacked; plain `sample` integer index expanded to `chain=0, draw=0..N-1`. Bridges the gap between `mmm.sample_saturation_curve()` output and the `(chain, draw)` format expected by arviz-plots HDI computation.
- Mean curve as a dedicated visual layer — `mean_curve_kwargs` parameter renders the posterior mean as a visually prominent solid line, separate from individual sample lines and the HDI band.
- Scale mismatch warning — heuristic `UserWarning` when `curves.max()` magnitude appears inconsistent with the `original_scale` flag; available as a template for any method accepting externally-scaled data.

**LOE:** M (2 methods; ships first to unblock parallel work on PRs 2, 3, 5, 6, 7)

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
- II.5 — `var: list[str] | None` → `var_names: list[str] | None` in `waterfall_components_decomposition`;
         `var: list[str]` → `var_names: list[str]` in `contributions_over_time`
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
- II.7 — Add 5 standard customization params (`figsize`, `backend`, per-element `*_kwargs`, `return_as_pc`, `**pc_kwargs`)

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
- II.5 — `parameter: list[str]` → `var_names: list[str]` in `param_stability`

**LOE:** L

---

### PR 9 — Suite Wrappers + Cleanup + Cross-Cutting Tests

**Content:**
- `MMMPlotSuite` class with 6 namespace sub-objects (no CV namespace)
- `MMMCVPlotSuite` re-exported from `__init__.py`
- Constructor validation: `MMMPlotSuite(data)` requires valid `MMMIDataWrapper` (I.2)
- Remove dead `_cache` from `MMMIDataWrapper` (IV.11)
- Delete old `mmm/plot.py` and replace with a stub that raises `ImportError` with a message:
  `"MMMPlotSuite has moved to pymc_marketing.mmm.plotting. See the migration guide."`
  This keeps the hard break but gives users an actionable error instead of a bare `ModuleNotFoundError`.
- Update `multidimensional.py` `.plot` property → returns `MMMPlotSuite(data=self.data)`
- Update `time_slice_cross_validation.py` `.plot` property → returns `MMMCVPlotSuite()`
- Update all imports across the codebase (including notebooks)

**Tests included:**
- Remove legacy `test_plotting.py` tests that duplicate new coverage
- Add cross-cutting parametrized tests that verify the standard API contract
  across all namespaces (unify repeated assertions from PRs 2–8):
  - `@pytest.mark.parametrize` over all public methods: return type is
    `tuple[Figure, NDArray[Axes]]`
  - `@pytest.mark.parametrize` over all public methods: `return_as_pc=True`
    returns `PlotCollection`
  - `@pytest.mark.parametrize` over all public methods: `dims` parameter
    accepted and filters correctly
  - `@pytest.mark.parametrize` over all data-dependent methods: `idata`
    override accepted and used instead of `self._data`
  - Constructor validation: `MMMPlotSuite(data=None)` raises (I.2)
  - Old import path raises `ImportError` with migration message

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

## Cross-Cutting Concerns

These are resolved in PR 1 (Foundation) and enforced by every subsequent PR:

| Concern | Resolution |
|---------|-----------|
| Suite separation | `MMMPlotSuite` for model-fit plots (requires `data`); `MMMCVPlotSuite` for CV plots (stateless). `mmm.plot` returns the former; `cv.plot` returns the latter. No cross-contamination. |
| Helper removal | 5 subplot-scaffolding helpers (`_init_subplots`, `_build_subplot_title`, `_dim_list_handler`, `_get_additional_dim_combinations`, `_add_median_and_hdi`) are **not ported** — `PlotCollection` replaces them. See [Helper Function Removal Plan](#helper-function-removal-plan). |
| I.3 — All methods use `MMMIDataWrapper` | `MMMPlotSuite` namespace classes receive only `data: MMMIDataWrapper` (no separate `idata` arg); no raw `self._data.idata.posterior` access. `MMMCVPlotSuite` has no `self._data` at all. Every data-dependent method accepts `idata: az.InferenceData | None = None` and resolves `data = MMMIDataWrapper(idata, schema=self._data.schema) if idata is not None else self._data` — all access goes through the resolved local `data`. |
| I.4 — No nested functions | Extract to module-level private functions or namespace private methods |
| I.5 — Shared color palette | `_helpers.channel_color_map(channels)` returns consistent channel→color dict |
| I.6 — arviz-plots adoption | All methods use `PlotCollection` internally; bar-plot methods fall back to matplotlib |
| II.1 — Standard return type | `tuple[Figure, NDArray[Axes]]` by default; `PlotCollection` opt-in via `return_as_pc=True` |
| II.2 — `original_scale=True` | Default on every method that exposes the parameter |
| II.5 — Consistent param names | `target_var` (predictive group methods), `var_name` (single posterior), `var_names` (multi posterior); `hdi_prob`; `figsize: tuple[float, float]` |
| II.6 — `dims` on all methods | Every method accepts `dims: dict[str, Any] | None` |
| II.7 — Figure customization | 5 standard params: `figsize`, `backend`, per-element `*_kwargs`, `return_as_pc`, `**pc_kwargs` (see [design](./2026-03-11-figure-customization-design.md)) |

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
| `mmm.plot.saturation_scatterplot()` | `mmm.plot.transformations.saturation_scatterplot()` |
| `mmm.plot.saturation_curves()` | `mmm.plot.transformations.saturation_curves()` |
| `mmm.plot.saturation_curves_scatter()` | **Removed** (use `transformations.saturation_scatterplot`) |
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
| `var: list[str] \| None` | `target_var: str = "y"` | `posterior_predictive` |
| `var: str \| None` | `target_var: str = "y"` | `prior_predictive` |
| `var: str` (required) | `var_name: str` | `posterior_distribution`, `prior_vs_posterior` |
| `param_name: str` | `var_name: str` | `channel_parameter` |
| `var: list[str]` (required) | `var_names: list[str]` | `contributions_over_time` |
| `var: list[str] \| None` | `var_names: list[str] \| None` | `waterfall_components_decomposition` |
| `parameter: list[str]` | `var_names: list[str]` | `param_stability` |
| `hdi_probs: list[float]` | `hdi_prob: float` | `transformations.saturation_curves` |
| `figsize: tuple[int, int]` | `figsize: tuple[float, float]` | `decomposition.waterfall` |
| `idata: xr.Dataset \| None` (ad-hoc, 2 methods) | `idata: az.InferenceData \| None` (consistent, all data-dependent methods) | All `MMMPlotSuite` methods that access `self._data` |

## Default Changes

| Parameter | Old default | New default | Affected methods |
|-----------|------------|-------------|-----------------|
| `original_scale` | `False` | `True` | `transformations.saturation_scatterplot`, `transformations.saturation_curves` |

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
| `ax: plt.Axes` (4 methods) | `return_as_pc=True` | Get `PlotCollection` back for post-processing; or use `**pc_kwargs` for layout |
| `**kwargs` (varies) | per-element `*_kwargs` + `**pc_kwargs` | Element-level control via `scatter_kwargs`, `hdi_kwargs`, etc.; collection-level via `pc_kwargs` |
| `rc_params` (1 method) | `**pc_kwargs` | Pass as `figure_kwargs` in `pc_kwargs` |
| `subplot_kwargs` (1 method) | `**pc_kwargs` | Forwarded to `PlotCollection.wrap/grid` |
| No customization (7 methods) | Full standard params | All methods now have `figsize`, `backend`, per-element `*_kwargs`, etc. |

## Behavioral Changes

- No method calls `plt.show()` — you control when to display
- `param_stability` combines all dimension values into a single multi-panel figure (use `dims` to subset)
- `MMMPlotSuite` constructor requires valid `data: MMMIDataWrapper` (no more `MMMPlotSuite(idata=None)`)
- All data-dependent methods accept `idata: az.InferenceData | None = None` to override instance data (e.g., plot with a different fitted model's InferenceData without constructing a new suite — the wrapper is constructed internally)
- CV methods moved to separate `MMMCVPlotSuite` — access via `cv.plot` instead of `mmm.plot`
- `cv.plot` no longer exposes model-fit methods; `mmm.plot` no longer exposes CV methods
- All methods use arviz-plots `PlotCollection` internally
- Multi-backend support: pass `backend="plotly"` with `return_as_pc=True`

## Removed

- **Import path changed:** `from pymc_marketing.mmm.plot import MMMPlotSuite` → `from pymc_marketing.mmm.plotting import MMMPlotSuite`. The old `mmm/plot.py` module is replaced with a stub that raises `ImportError` with migration guidance.
- `saturation_curves_scatter` — use `mmm.plot.transformations.saturation_scatterplot()` instead
- `ax` parameter — use `return_as_pc=True` to get a `PlotCollection` back for post-processing
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
| II.7 | Inconsistent figure customization | 5 standard params on every method: `figsize`, `backend`, per-element `*_kwargs`, `return_as_pc`, `**pc_kwargs` (see [figure customization design](./2026-03-11-figure-customization-design.md)) | 2–8 |
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
| V.1–V.5 | Test coverage gaps | Partially addressed: each family PR (2–8) ships with tests; PR 9 removes legacy tests and adds cross-cutting parametrized tests. Follow-up for edge cases (V.3) and thread-safety (V.5). | 2–9 |
| VI.1 | Plotting gallery | **Deferred** — follow-up release | — |
| VII.1 | Coord `x` → `channel` | Rename coordinate in `channel_share_hdi` | 5 |
| VII.2 | Use `xarray.to_dataframe` | **Deferred** — follow-up release | — |
| VII.3 | Time-varying media visualization | **Deferred** — follow-up release | — |
