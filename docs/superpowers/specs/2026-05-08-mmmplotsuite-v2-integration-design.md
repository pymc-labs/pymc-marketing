# MMMPlotSuite v2 Integration Design

**Issue:** [#2527](https://github.com/pymc-labs/pymc-marketing/issues/2527)
**Date:** 2026-05-08
**Status:** Approved

## Overview

PRs 1–8 of the MMMPlotSuite v2 migration are complete. The new namespace-based plotting classes (`DecompositionPlots`, `DiagnosticsPlots`, `SensitivityPlots`, `TransformationPlots`, `BudgetPlots`, `MMMCVPlotSuite`) exist in `pymc_marketing/mmm/plotting/` but are not yet wired into `MMM`, `BudgetOptimizerWrapper`, or `TimeSliceCrossValidator`.

This spec covers the integration work (PRs 9–10 scope), adapted to support **both** the legacy `MMMPlotSuite` and the new namespace suite simultaneously via a `plot_suite` argument.

## Goals

- Add `plot_suite: Literal["legacy", "new"] = "legacy"` as an instance attribute on `MMM` and `TimeSliceCrossValidator` (set post-construction, not in `__init__`).
- Default to legacy — zero breaking changes for existing users.
- Emit a one-time `FutureWarning` on first `.plot` access in legacy mode, pointing to the new migration guide.
- Provide a clean opt-in path to the new namespace API via `mmm.plot_suite = "new"`.
- Write a new `mmm_plot_suite_migration_guide.ipynb` notebook.
- Update two notebooks that already use new-mode CV plot methods.

## Non-Goals

- Removing or modifying the legacy `MMMPlotSuite` class.
- Changing any plot method implementations.
- Updating notebooks that currently work correctly in legacy mode.

---

## Section 1: `MMM` changes

### `plot_suite` property

`MMM` gains a `plot_suite` property backed by `_plot_suite`. In `__init__`:

```python
self._plot_suite: Literal["legacy", "new"] = "legacy"
```

Property definition:

```python
@property
def plot_suite(self) -> Literal["legacy", "new"]:
    return self._plot_suite

@plot_suite.setter
def plot_suite(self, value: Literal["legacy", "new"]) -> None:
    if value not in ("legacy", "new"):
        raise ValueError(f"plot_suite must be 'legacy' or 'new', got {value!r}")
    self._plot_suite = value
```

Set on an instance to switch modes:

```python
mmm = MMM(...)
mmm.plot_suite = "new"
mmm.plot.decomposition.contributions_over_time(...)
```

No `__init__` signature change. No serialization needed (it's an opt-in runtime flag, not a model configuration value).

### `.plot` property

```python
@property
def plot(self) -> MMMPlotSuite | MMMPlotSuiteFacade:
    self._validate_model_was_built()
    self._validate_idata_exists()
    if self.plot_suite == "legacy":
        if not getattr(self, "_plot_suite_warned", False):
            warnings.warn(
                "The legacy MMMPlotSuite will be removed in pymc-marketing 2.0.0. "
                "Set mmm.plot_suite = 'new' to opt in to the new namespace-based API. "
                "See the migration guide: "
                "docs/source/notebooks/mmm/mmm_plot_suite_migration_guide.ipynb",
                FutureWarning,
                stacklevel=2,
            )
            self._plot_suite_warned = True
        return MMMPlotSuite(data=self.data)
    return MMMPlotSuiteFacade(data=self.data)
```

The `_plot_suite_warned` flag suppresses repeated warnings on the same instance.

---

## Section 2: New `MMMPlotSuiteFacade` container

**File:** `pymc_marketing/mmm/plotting/suite.py`

```python
class MMMPlotSuiteFacade:
    """Namespace container for the new MMMPlotSuite v2 API.

    Access via ``mmm.plot`` when ``plot_suite='new'``.

    Attributes
    ----------
    decomposition : DecompositionPlots
    diagnostics : DiagnosticsPlots
    sensitivity : SensitivityPlots
    transformation : TransformationPlots
    """

    def __init__(self, data: MMMIDataWrapper) -> None:
        self.decomposition = DecompositionPlots(data)
        self.diagnostics = DiagnosticsPlots(data)
        self.sensitivity = SensitivityPlots(data)
        self.transformation = TransformationPlots(data)
```

`BudgetPlots` and `MMMCVPlotSuite` are intentionally excluded — they live on `BudgetOptimizerWrapper.plot` and `TimeSliceCrossValidator.plot` respectively.

**Exports:** Add `MMMPlotSuiteFacade` to `pymc_marketing/mmm/plotting/__init__.py` and to `pymc_marketing/mmm/__init__.py`.

---

## Section 3: `TimeSliceCrossValidator` changes

### `plot_suite` property

`TimeSliceCrossValidator` gains the same `plot_suite` property pattern. In `__init__`:

```python
self._plot_suite: Literal["legacy", "new"] = "legacy"
```

Property definition (identical to `MMM`):

```python
@property
def plot_suite(self) -> Literal["legacy", "new"]:
    return self._plot_suite

@plot_suite.setter
def plot_suite(self, value: Literal["legacy", "new"]) -> None:
    if value not in ("legacy", "new"):
        raise ValueError(f"plot_suite must be 'legacy' or 'new', got {value!r}")
    self._plot_suite = value
```

`TimeSliceCrossValidator` does not hold an `MMM` reference, so this must be set independently:

```python
cv = TimeSliceCrossValidator(...)
cv.plot_suite = "new"
cv.plot.predictions(...)
```

### `.plot` property

Revert to the pre-PR-#2530 legacy behavior by default:

```python
@property
def plot(self) -> MMMPlotSuite | MMMCVPlotSuite:
    self._validate_model_was_built()  # sets self.idata from last fold
    if self.plot_suite == "legacy":
        if not hasattr(self, "idata") or self.idata is None:
            raise ValueError(
                "idata is not available. Ensure TimeSliceCrossValidator.run() "
                "completed successfully."
            )
        return MMMPlotSuite(idata=self.idata)
    if not hasattr(self, "cv_idata"):
        raise ValueError(
            "cv_idata is not available. Ensure TimeSliceCrossValidator.run() "
            "completed successfully."
        )
    return MMMCVPlotSuite(self.cv_idata)
```

Legacy mode returns the last fold's `idata` wrapped in the monolithic `MMMPlotSuite` — identical to the original behavior before PR #2530.

### Notebook updates

Two notebooks already use new-mode CV methods and will break when `.plot` reverts to legacy default. Set `cv.plot_suite = "new"` after constructing `TimeSliceCrossValidator`:

- `docs/source/notebooks/mmm/mmm_time_slice_cross_validation.ipynb`
- `docs/source/notebooks/mmm/mmm_roas.ipynb`

---

## Section 4: `BudgetOptimizerWrapper` changes

Add an explicit `.plot` property (overrides `__getattr__` delegation):

```python
@property
def plot(self) -> BudgetPlots | MMMPlotSuite:
    if self.model_class.plot_suite == "new":
        return BudgetPlots()
    return self.model_class.plot  # legacy MMMPlotSuite with budget methods
```

In legacy mode this returns `self.model_class.plot` which triggers the one-time `FutureWarning` on the `MMM` instance.
In new mode this returns a stateless `BudgetPlots()` instance (all data passed per-call via `samples=`).

No notebook changes needed — `mmm_multidimensional_example.ipynb` uses legacy method names (`budget_allocation`, `allocated_contribution_by_channel_over_time`) which remain available on the legacy `MMMPlotSuite`.

---

## Section 5: Migration guide notebook

**File:** `docs/source/notebooks/mmm/mmm_plot_suite_migration_guide.ipynb`

A new notebook (no code execution required — prose + code snippets only) covering:

1. **Why the new API exists** — namespace-based separation, cleaner boundaries
2. **How to opt in** — `mmm.plot_suite = "new"`, `cv.plot_suite = "new"`
3. **`mmm.plot.*` namespace map** — side-by-side old vs new for each method group:
   - Diagnostics: `mmm.plot.posterior_predictive()` → `mmm.plot.diagnostics.posterior_predictive()`
   - Decomposition: `mmm.plot.contributions_over_time()` → `mmm.plot.decomposition.contributions_over_time()`
   - Sensitivity: `mmm.plot.sensitivity_analysis()` → `mmm.plot.sensitivity.sensitivity_analysis()`
   - Transformation: `mmm.plot.saturation_curves()` → `mmm.plot.transformation.saturation_curves()`
4. **Budget plots moved** — `mmm.plot.budget_allocation()` → `optimizer.plot.allocation_roas(samples=...)`
5. **CV plots moved** — complete mapping of old `MMMPlotSuite` CV methods to new `MMMCVPlotSuite` methods
6. **Removal timeline** — legacy `MMMPlotSuite` will be removed in pymc-marketing 2.0.0

---

## File change summary

| File | Change |
|------|--------|
| `pymc_marketing/mmm/mmm.py` | Add `plot_suite` instance attr in `MMM.__init__` body; update `.plot` property; add `.plot` property to `BudgetOptimizerWrapper` |
| `pymc_marketing/mmm/plotting/suite.py` | New file: `MMMPlotSuiteFacade` |
| `pymc_marketing/mmm/plotting/__init__.py` | Export `MMMPlotSuiteFacade` |
| `pymc_marketing/mmm/__init__.py` | Export `MMMPlotSuiteFacade` |
| `pymc_marketing/mmm/time_slice_cross_validation.py` | Add `plot_suite` instance attr; revert `.plot` to legacy default |
| `docs/source/notebooks/mmm/mmm_plot_suite_migration_guide.ipynb` | New notebook |
| `docs/source/notebooks/mmm/mmm_time_slice_cross_validation.ipynb` | Set `cv.plot_suite = "new"` after CV construction |
| `docs/source/notebooks/mmm/mmm_roas.ipynb` | Set `cv.plot_suite = "new"` after CV construction |

## Out of scope

- Removing `pymc_marketing/mmm/plot.py` (legacy suite stays intact)
- Updating any other notebooks (they work unchanged in legacy mode)
- Changes to plot method implementations
