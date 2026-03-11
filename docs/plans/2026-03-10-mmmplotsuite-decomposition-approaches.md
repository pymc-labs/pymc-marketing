# MMMPlotSuite Decomposition — Approach Analysis

> Exploration of four architectural approaches for breaking up the `MMMPlotSuite`
> god class (5,150 lines, 21 public methods, 19 private helpers).
>
> Prepared 2026-03-10.

---

## Table of Contents

- [Context](#context)
  - [The Problem](#the-problem)
  - [The Seven Plotting Families](#the-seven-plotting-families)
  - [Constraints](#constraints)
  - [Relevant Codebase Patterns](#relevant-codebase-patterns)
- [Approach A: Standalone Functions in a `plotting/` Package](#approach-a-standalone-functions-in-a-plotting-package)
- [Approach B: Mixin-Based Decomposition](#approach-b-mixin-based-decomposition)
- [Approach C: Domain-Aligned Placement](#approach-c-domain-aligned-placement)
- [Approach D: Namespace Sub-Objects (Structured Facade)](#approach-d-namespace-sub-objects-structured-facade)
- [Comparison Matrix](#comparison-matrix)

---

## Context

### The Problem

`MMMPlotSuite` in `pymc_marketing/mmm/plot.py` is a single class that handles seven
distinct plotting families. At 5,150 lines it violates the Single Responsibility
Principle and is painful to navigate, test, and extend. A comprehensive audit
identified 63 issues across structural, API consistency, code quality, and test
coverage dimensions (see `2026-03-10-mmmplotsuite-comprehensive-issues.md`).

### The Seven Plotting Families

| Family | Methods |
|--------|---------|
| Time-series diagnostics | `posterior_predictive`, `prior_predictive`, `residuals_over_time`, `residuals_posterior_distribution` |
| Distribution diagnostics | `posterior_distribution`, `channel_parameter`, `prior_vs_posterior` |
| Saturation / response curves | `saturation_scatterplot`, `saturation_curves`, `saturation_curves_scatter` (deprecated) |
| Budget allocation | `budget_allocation`, `allocated_contribution_by_channel_over_time` |
| Sensitivity / uplift / marginal | `sensitivity_analysis`, `uplift_curve`, `marginal_curve` |
| Decomposition | `waterfall_components_decomposition`, `contributions_over_time`, `channel_contribution_share_hdi` |
| Cross-validation | `cv_predictions`, `param_stability`, `cv_crps` |

### Constraints

- **Breaking changes are acceptable** — this is a major refactor; a migration guide
  will be provided but backward compatibility is not required.
- **Scope is MMMPlotSuite (matplotlib) only** — the Plotly factory
  (`MMMPlotlyFactory`) is a separate concern.
- **Goals are weighted equally:** maintainability, extensibility, testability.

### Relevant Codebase Patterns

| Pattern | Location | Notes |
|---------|----------|-------|
| Shared plot helpers | `pymc_marketing/plot.py` | `plot_curve`, `plot_hdi`, `plot_samples` — used by components |
| Validation mixins | `mmm/validating.py` | `ValidateTargetColumn`, `ValidateDateColumn`, etc. composed into `MMM` |
| `ModelIO` mixin | `model_builder.py` | Handles save/load via mixin |
| `MMMIDataWrapper` | `data/idata/mmm_wrapper.py` | Type-safe access to idata; largely bypassed by current plot methods |
| `.plot` property | `mmm/multidimensional.py:1145` | Returns `MMMPlotSuite(data=self.data)` |
| `.plot` property | `mmm/time_slice_cross_validation.py:166` | Returns `MMMPlotSuite(idata=self.idata)` |

---

## Approach A: Standalone Functions in a `plotting/` Package

### Concept

Convert each plotting family into a module of standalone functions inside a new
`pymc_marketing/mmm/plotting/` package. Shared helpers (subplot creation, color
mapping, HDI bands) go into `_helpers.py`. A thin `MMMPlotSuite` facade delegates
to these functions, preserving the `mmm.plot.*` access pattern that users rely on.

### File Layout

```
mmm/plotting/
    __init__.py            # re-exports all public functions + MMMPlotSuite facade
    _helpers.py            # _init_subplots, color mapping, _add_median_and_hdi, etc.
    suite.py               # MMMPlotSuite facade — delegates to standalone functions
    time_series.py         # posterior_predictive, prior_predictive, residuals_*
    distributions.py       # posterior_distribution, channel_parameter, prior_vs_posterior
    saturation.py          # saturation_scatterplot, saturation_curves
    budget.py              # budget_allocation, allocated_contribution_by_channel_over_time
    sensitivity.py         # sensitivity_analysis, uplift_curve, marginal_curve
    decomposition.py       # waterfall, contributions_over_time, channel_contribution_share_hdi
    cross_validation.py    # cv_predictions, param_stability, cv_crps
```

### Example Signature

```python
# mmm/plotting/time_series.py
def posterior_predictive(
    idata: az.InferenceData,
    var: list[str] | None = None,
    hdi_prob: float = 0.94,
    figsize: tuple[float, float] = (15, 5),
    original_scale: bool = True,
    dims: dict[str, Any] | None = None,
    **subplot_kwargs,
) -> tuple[Figure, NDArray[Axes]]:
    ...
```

### Facade (preserves `mmm.plot.*`)

The `MMMPlotSuite` facade is retained so that `mmm.plot.posterior_predictive()`
continues to work. Each method is a thin delegation that passes `self.idata` /
`self.data` into the corresponding standalone function:

```python
# mmm/plotting/suite.py
from pymc_marketing.mmm.plotting import time_series, sensitivity, ...

class MMMPlotSuite:
    def __init__(self, idata=None, data=None):
        self.idata = idata
        self.data = data

    def posterior_predictive(self, **kwargs):
        return time_series.posterior_predictive(self.idata, **kwargs)

    def sensitivity_analysis(self, **kwargs):
        return sensitivity.sensitivity_analysis(self.idata, self.data, **kwargs)

    # ... one-liner delegation for each of the 21 methods
```

The `multidimensional.MMM.plot` property and
`TimeSliceCrossValidator.plot` property continue to return a
`MMMPlotSuite` instance, so user code like `mmm.plot.posterior_predictive()`
is unchanged.

### Pros

1. **Independently testable** — each function can be tested with a mock `idata`
   without constructing a `MMMPlotSuite` instance.
2. **Explicit dependencies** — every function declares what it needs in its
   signature; no hidden `self.idata` contracts.
3. **Maximum composability** — advanced users call functions directly; casual
   users use the facade.
4. **Natural file split** — 7 families map to 7 files averaging ~700 lines each.
5. **Low migration cost to domain modules** — moving a function from
   `plotting/sensitivity.py` to `sensitivity_analysis.py` later is trivial.
6. **Shared helpers are clean** — `_helpers.py` replaces the current 3 different
   subplot-creation patterns with one.

### Cons

1. **Explicit `idata`/`data` parameter on every standalone function** — more
   boilerplate compared to `self.idata` (mitigated by the facade for end users).
2. **Facade is delegation boilerplate** — ~21 one-liner pass-throughs in
   `suite.py`, though each is trivial.
3. **Shared helpers become an internal API** — changes to `_helpers.py` can
   ripple across all 7 modules.
4. **Two entry points to maintain** — both `mmm.plot.*` (facade) and direct
   function imports are public; signatures must stay in sync, and docstrings need to be maintained in two places.

### Summary

Strongest on testability. Retains the `mmm.plot.*` access pattern via the facade.
Serves as a natural foundation for Approach C or D later without requiring a
second refactor.

---

## Approach B: Mixin-Based Decomposition

### Concept

Each plotting family becomes a mixin class. `MMMPlotSuite` inherits from all seven
mixins, gaining their methods. The file splits into 7 mixin files plus a suite file,
but the public API stays as a single class.

### File Layout

```
mmm/plotting/
    __init__.py
    _helpers.py
    time_series_mixin.py
    distributions_mixin.py
    saturation_mixin.py
    budget_mixin.py
    sensitivity_mixin.py
    decomposition_mixin.py
    cross_validation_mixin.py
    suite.py               # MMMPlotSuite inherits from all mixins
```

### Example Structure

```python
# mmm/plotting/sensitivity_mixin.py
class SensitivityPlotsMixin:
    idata: az.InferenceData      # type annotation for IDE support
    data: MMMIDataWrapper

    def sensitivity_analysis(self, ...): ...
    def uplift_curve(self, ...): ...
    def marginal_curve(self, ...): ...

# mmm/plotting/suite.py
class MMMPlotSuite(
    TimeSeriesPlotsMixin,
    DistributionPlotsMixin,
    SaturationPlotsMixin,
    BudgetPlotsMixin,
    SensitivityPlotsMixin,
    DecompositionPlotsMixin,
    CrossValidationPlotsMixin,
):
    def __init__(self, idata=None, data=None): ...
```

### Pros

1. **Smallest API change** — `mmm.plot.sensitivity_analysis()` continues to work
   with exactly the same call site.
2. **Natural shared state** — `self.idata` and `self.data` are available
   implicitly; no parameter threading.
3. **Solves the file-size problem** — each mixin is its own file.
4. **Good extensibility** — new family = new mixin + add to inheritance list.
5. **Precedent exists** — validation mixins (`ValidateTargetColumn`, etc.) and
   `ModelIO` already use this pattern in the codebase.

### Cons

1. **Tight coupling** — all mixins depend on the same implicit `self` contract
   (`self.idata`, `self.data`, shared helpers). This contract is not enforced by
   the type system.
2. **Testing still requires instantiation** — you need a `MMMPlotSuite` (or
   carefully mock `self`) to test any mixin method.
3. **IDE support is weaker** — harder to trace which mixin a method comes from;
   "Go to definition" on `mmm.plot.sensitivity_analysis()` may land in the
   wrong place.
4. **Diamond inheritance risk** — if two mixins share a helper method name,
   MRO resolution becomes subtle.
5. **Doesn't address domain coupling** — CV plots still live with saturation
   plots in the same class, just in different files.
6. **Mixins are considered an anti-pattern** by a portion of the Python
   community, and can be confusing for contributors unfamiliar with the pattern.

### Summary

Lowest migration cost of all approaches. Preserves the existing API surface
exactly. Weakest on testability since methods still require a class instance.
Well-suited when API stability is the dominant concern.

---

## Approach C: Domain-Aligned Placement

### Concept

Move each plotting family to the module that owns its data domain. Sensitivity
plots go to `sensitivity_analysis.py`, CV plots go to
`time_slice_cross_validation.py`, budget plots go to `budget_optimizer.py`.
Methods that don't have a clear domain home (time-series, distributions,
decomposition) stay in a smaller general plotting module.

### File Layout

```
mmm/
    sensitivity_analysis.py        # existing + plot functions added
    time_slice_cross_validation.py # existing + plot functions added
    budget_optimizer.py            # existing + plot functions added
    plotting/
        __init__.py
        _helpers.py
        time_series.py             # posterior_predictive, prior_predictive, residuals_*
        distributions.py           # posterior_distribution, channel_parameter, prior_vs_posterior
        saturation.py              # saturation_scatterplot, saturation_curves
        decomposition.py           # waterfall, contributions_over_time, channel_contribution_share_hdi
```

### Example

```python
# mmm/sensitivity_analysis.py
class SensitivityAnalysis:
    def run_sweep(self, ...): ...

    def plot(self, figsize=..., ax=None, ...):
        """Plot sensitivity analysis results."""
        ...

    def plot_uplift(self, figsize=..., ax=None, ...):
        """Plot uplift curve."""
        ...

    def plot_marginal(self, figsize=..., ax=None, ...):
        """Plot marginal curve."""
        ...
```

### Pros

1. **Most natural grouping** — sensitivity plots live right next to sensitivity
   computation code.
2. **Self-contained domain modules** — each module handles both computation and
   visualization.
3. **Reduced cross-module imports** — `sensitivity_analysis.py` doesn't need to
   know about CV or budget allocation.

### Cons

1. **Breaks the unified access pattern entirely** — there is no single
   `mmm.plot.*` namespace. Users must know which module owns each plot.
2. **~60% of methods have no clear domain home** — time-series diagnostics,
   distribution diagnostics, decomposition, and saturation plots don't belong
   to a specific compute module. You still need a "general plots" module.
3. **Domain modules grow significantly** — `sensitivity_analysis.py` goes from
   526 to ~800+ lines; `time_slice_cross_validation.py` from 670 to ~1,100+
   lines.
4. **Cross-cutting concerns remain** — shared color palettes, subplot creation,
   HDI bands still need a common location.
5. **Discoverability suffers** — users can't tab-complete `mmm.plot.<TAB>` to
   see all available plots.
6. **Testing is fragmented** — plot tests are spread across domain test files
   rather than focused plotting test files.

### Summary

Strongest cohesion for the 3 families with clear domain homes (sensitivity,
CV, budget). The remaining 4 families still need a general plotting module.
Breaks the unified `mmm.plot.*` access pattern. Can also serve as a follow-up
step after Approach A or B rather than a standalone strategy.

---

## Approach D: Namespace Sub-Objects (Structured Facade)

### Concept

Instead of a flat `mmm.plot.sensitivity_analysis()`, group related plots under
sub-namespace objects: `mmm.plot.sensitivity.analysis()`. Each namespace is a
lightweight class holding a reference to `idata`/`data` and exposing only its
family's methods. The plotting logic lives directly in the namespace class
methods.

### File Layout

```
mmm/plotting/
    __init__.py              # MMMPlotSuite with namespace sub-objects
    _helpers.py              # shared helpers
    time_series.py           # TimeSeriesPlots namespace class
    distributions.py         # DistributionPlots namespace class
    saturation.py            # SaturationPlots namespace class
    budget.py                # BudgetPlots namespace class
    sensitivity.py           # SensitivityPlots namespace class
    decomposition.py         # DecompositionPlots namespace class
    cross_validation.py      # CrossValidationPlots namespace class
```

### Example Structure

```python
# mmm/plotting/sensitivity.py
class SensitivityPlots:
    def __init__(self, idata, data):
        self._idata = idata
        self._data = data

    def analysis(self, channels=None, figsize=..., ax=None, ...):
        """Plot sensitivity analysis results."""
        ...

    def uplift(self, figsize=..., ax=None, ...):
        """Plot uplift curve."""
        ...

    def marginal(self, figsize=..., ax=None, ...):
        """Plot marginal curve."""
        ...


# mmm/plotting/__init__.py
class MMMPlotSuite:
    def __init__(self, idata=None, data=None):
        self.sensitivity = SensitivityPlots(idata, data)
        self.cv = CrossValidationPlots(idata, data)
        self.diagnostics = DiagnosticsPlots(idata, data)
        self.distributions = DistributionPlots(idata, data)
        self.saturation = SaturationPlots(idata, data)
        self.budget = BudgetPlots(idata, data)
        self.decomposition = DecompositionPlots(idata, data)
```

### User Experience

```python
mmm.plot.sensitivity.analysis(channels=["tv", "radio"])
mmm.plot.cv.predictions(results)
mmm.plot.diagnostics.posterior_predictive(var=["y"])
```

### Pros

1. **Excellent discoverability** — `mmm.plot.<TAB>` shows 7 families, then
   `mmm.plot.sensitivity.<TAB>` shows the 3 methods in that family.
2. **Logical grouping** — related plots are always together; users don't need
   to scan 21 flat methods to find what they need.
3. **Small, focused namespace classes** — each has 2–4 methods, easy to
   understand.
4. **Guides users naturally** — the namespace structure teaches users about
   the plotting families without requiring documentation.
5. **Single public API** — only the namespace access pattern is exposed,
   keeping the surface area manageable.

### Cons

1. **Deeper call chain** — `mmm.plot.sensitivity.analysis()` is more verbose
   than `mmm.plot.sensitivity_analysis()`.
2. **More boilerplate** — 7 namespace classes + the suite class.
3. **Grouping disputes** — some plots may not fit neatly into one family
   (e.g., does `contributions_over_time` belong in "decomposition" or
   "diagnostics"?).
4. **Method renaming required** — `sensitivity_analysis` becomes
   `mmm.plot.sensitivity.analysis` which is a different name than today.
5. **Testing requires instantiation** — methods live on classes, so tests
   need to construct namespace objects (though each is small and focused).

### Variant: D + A

Approach D can be combined with Approach A by backing each namespace method
with a standalone function. This adds a direct function access layer
(e.g., `from pymc_marketing.mmm.plotting.sensitivity import sensitivity_analysis`)
for users who prefer working with functions and for easier unit testing. The
trade-off is two public API surfaces to maintain.

### Summary

Most polished user experience with grouped discoverability. Requires method
renaming (e.g., `sensitivity_analysis` → `sensitivity.analysis`). Each
namespace class is small and focused but testing still requires class
instantiation unless combined with Approach A.

---

## Comparison Matrix

| Criterion | A: Standalone Fns | B: Mixins | C: Domain-Aligned | D: Namespace Sub-Objects |
|-----------|-------------------|-----------|-------------------|--------------------------|
| **Maintainability** | High — 7 focused files | Medium — files split but coupled via `self` | Medium — 4 families still need general module | High — 7 focused files + thin namespaces |
| **Extensibility** | High — add a function | High — add a mixin | Medium — must decide which module | High — add a function + update namespace |
| **Testability** | High — no class needed | Low — requires `self` | High — functions in domain modules | Medium — small classes to instantiate |
| **API change** | Moderate — flat functions or facade | Minimal — same `mmm.plot.*` | Large — no unified access | Moderate — nested namespaces |
| **Discoverability** | Medium — flat list or imports | Medium — flat list on class | Low — spread across modules | High — grouped namespaces |
| **Complexity** | Low | Medium (MRO, implicit contracts) | Medium (fragmented) | Medium (two API layers) |
| **Migration effort** | Medium | Low | High | Medium |
| **Future-proof** | Good — easy to move fns later | Locked into class shape | Already domain-aligned | Good — functions are portable |
