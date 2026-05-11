# MMMPlotSuite v2 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing v2 namespace-based plotting classes into `MMM`, `BudgetOptimizerWrapper`, and `TimeSliceCrossValidator` behind a `plot_suite` argument, defaulting to the legacy suite so nothing breaks.

**Architecture:** A new `MMMPlotSuiteFacade` container class wraps the four v2 namespace classes (`DecompositionPlots`, `DiagnosticsPlots`, `SensitivityPlots`, `TransformationPlots`). `MMM` gains a `plot_suite` constructor arg; its `.plot` property returns either the legacy `MMMPlotSuite` or the new `MMMPlotSuiteFacade` depending on that arg. `TimeSliceCrossValidator` gets its own `plot_suite` arg (it holds no `MMM` reference). `BudgetOptimizerWrapper` adds an explicit `.plot` property that reads `model_class._plot_suite`.

**Tech Stack:** Python, pymc-marketing, pydantic `Field`, `FutureWarning`, nbformat for notebook edits.

**Activate conda env before running any command:**
```bash
conda run -n pymc-dev-2527 <command>
```

**Run pre-commit after every file you create or modify:**
```bash
conda run -n pymc-dev-2527 pre-commit run --files <file>
```

---

## File Map

| Action | File |
|--------|------|
| Create | `pymc_marketing/mmm/plotting/suite.py` |
| Modify | `pymc_marketing/mmm/plotting/__init__.py` |
| Modify | `pymc_marketing/mmm/__init__.py` |
| Modify | `pymc_marketing/mmm/mmm.py` |
| Modify | `pymc_marketing/mmm/time_slice_cross_validation.py` |
| Modify | `docs/source/notebooks/mmm/mmm_time_slice_cross_validation.ipynb` |
| Modify | `docs/source/notebooks/mmm/mmm_roas.ipynb` |
| Create | `docs/source/notebooks/mmm/mmm_plot_suite_migration_guide.ipynb` |
| Modify | `tests/mmm/test_time_slice_cross_validator.py` |

---

## Task 1: Create `MMMPlotSuiteFacade`

**Files:**
- Create: `pymc_marketing/mmm/plotting/suite.py`
- Test: `tests/mmm/plotting/test_suite.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/mmm/plotting/test_suite.py`:

```python
import pytest
from unittest.mock import MagicMock

from pymc_marketing.mmm.plotting.suite import MMMPlotSuiteFacade
from pymc_marketing.mmm.plotting.decomposition import DecompositionPlots
from pymc_marketing.mmm.plotting.diagnostics import DiagnosticsPlots
from pymc_marketing.mmm.plotting.sensitivity import SensitivityPlots
from pymc_marketing.mmm.plotting.transformations import TransformationPlots


def test_facade_creates_namespace_attributes():
    data = MagicMock()
    facade = MMMPlotSuiteFacade(data=data)
    assert isinstance(facade.decomposition, DecompositionPlots)
    assert isinstance(facade.diagnostics, DiagnosticsPlots)
    assert isinstance(facade.sensitivity, SensitivityPlots)
    assert isinstance(facade.transformation, TransformationPlots)


def test_facade_passes_data_to_each_namespace():
    data = MagicMock()
    facade = MMMPlotSuiteFacade(data=data)
    assert facade.decomposition.data is data
    assert facade.diagnostics.data is data
    assert facade.sensitivity.data is data
    assert facade.transformation.data is data
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/plotting/test_suite.py -v
```

Expected: `ModuleNotFoundError` — `suite.py` does not exist yet.

- [ ] **Step 3: Create `pymc_marketing/mmm/plotting/suite.py`**

```python
#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""MMMPlotSuiteFacade — namespace container for v2 plotting API."""

from __future__ import annotations

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.mmm.plotting.decomposition import DecompositionPlots
from pymc_marketing.mmm.plotting.diagnostics import DiagnosticsPlots
from pymc_marketing.mmm.plotting.sensitivity import SensitivityPlots
from pymc_marketing.mmm.plotting.transformations import TransformationPlots

__all__ = ["MMMPlotSuiteFacade"]


class MMMPlotSuiteFacade:
    """Namespace container for the v2 MMMPlotSuite API.

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

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/plotting/test_suite.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Run pre-commit**

```bash
conda run -n pymc-dev-2527 pre-commit run --files pymc_marketing/mmm/plotting/suite.py tests/mmm/plotting/test_suite.py
```

- [ ] **Step 6: Commit**

```bash
git add pymc_marketing/mmm/plotting/suite.py tests/mmm/plotting/test_suite.py
git commit -m "feat(mmm): add MMMPlotSuiteFacade namespace container"
```

---

## Task 2: Export `MMMPlotSuiteFacade`

**Files:**
- Modify: `pymc_marketing/mmm/plotting/__init__.py`
- Modify: `pymc_marketing/mmm/__init__.py`

- [ ] **Step 1: Write the failing import test**

Add to `tests/mmm/plotting/test_suite.py`:

```python
def test_facade_importable_from_plotting_package():
    from pymc_marketing.mmm.plotting import MMMPlotSuiteFacade as F
    assert F is MMMPlotSuiteFacade


def test_facade_importable_from_mmm_package():
    from pymc_marketing.mmm import MMMPlotSuiteFacade as F
    assert F is MMMPlotSuiteFacade
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/plotting/test_suite.py::test_facade_importable_from_plotting_package tests/mmm/plotting/test_suite.py::test_facade_importable_from_mmm_package -v
```

Expected: `ImportError`.

- [ ] **Step 3: Update `pymc_marketing/mmm/plotting/__init__.py`**

Add the import and update `__all__`:

```python
from pymc_marketing.mmm.plotting.suite import MMMPlotSuiteFacade
```

Append `"MMMPlotSuiteFacade"` to the `__all__` list. The file becomes:

```python
"""MMM plotting package — namespace-based plot suite."""

from pymc_marketing.mmm.plotting.budget import BudgetPlots
from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite
from pymc_marketing.mmm.plotting.decomposition import DecompositionPlots
from pymc_marketing.mmm.plotting.diagnostics import DiagnosticsPlots
from pymc_marketing.mmm.plotting.sensitivity import SensitivityPlots
from pymc_marketing.mmm.plotting.suite import MMMPlotSuiteFacade
from pymc_marketing.mmm.plotting.transformations import TransformationPlots

__all__ = [
    "BudgetPlots",
    "DecompositionPlots",
    "DiagnosticsPlots",
    "MMMCVPlotSuite",
    "MMMPlotSuiteFacade",
    "SensitivityPlots",
    "TransformationPlots",
]
```

- [ ] **Step 4: Update `pymc_marketing/mmm/__init__.py`**

Add after the existing `from pymc_marketing.mmm.mmm import (...)` block (around line 60):

```python
from pymc_marketing.mmm.plotting import MMMPlotSuiteFacade
```

And add `"MMMPlotSuiteFacade"` to `__all__` (keep list alphabetically sorted, insert between `"MMMBuilder"` and `"MediaConfig"`):

```python
"MMMPlotSuiteFacade",
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/plotting/test_suite.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 6: Run pre-commit**

```bash
conda run -n pymc-dev-2527 pre-commit run --files pymc_marketing/mmm/plotting/__init__.py pymc_marketing/mmm/__init__.py
```

- [ ] **Step 7: Commit**

```bash
git add pymc_marketing/mmm/plotting/__init__.py pymc_marketing/mmm/__init__.py tests/mmm/plotting/test_suite.py
git commit -m "feat(mmm): export MMMPlotSuiteFacade from plotting and mmm packages"
```

---

## Task 3: `MMM.plot_suite` constructor arg and updated `.plot` property

**Files:**
- Modify: `pymc_marketing/mmm/mmm.py`

### 3a: Add `plot_suite` parameter

- [ ] **Step 1: Write the failing test**

In `tests/mmm/test_mmm.py`, add a new test function. The existing fitted-MMM fixture is `fit_mmm`:

```python
def test_mmm_plot_suite_defaults_to_legacy(fit_mmm):
    """MMM.plot_suite defaults to 'legacy'."""
    assert fit_mmm._plot_suite == "legacy"


def test_mmm_plot_suite_new_mode(fit_mmm):
    """Setting _plot_suite to 'new' stores the setting."""
    import copy
    mmm = copy.copy(fit_mmm)
    mmm._plot_suite = "new"
    assert mmm._plot_suite == "new"
```

- [ ] **Step 2: Run to see current state**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_mmm.py::test_mmm_plot_suite_defaults_to_legacy -v
```

Expected: FAIL with `AttributeError: '_plot_suite' not found` (attribute doesn't exist yet).

- [ ] **Step 3: Add `Literal` to imports in `mmm.py`**

Find line 185:
```python
from typing import Annotated, Any, Self, cast
```
Change to:
```python
from typing import Annotated, Any, Literal, Self, cast
```

- [ ] **Step 4: Add `MMMPlotSuiteFacade` import in `mmm.py`**

Find line 224:
```python
from pymc_marketing.mmm.plot import MMMPlotSuite
```
Add after it:
```python
from pymc_marketing.mmm.plotting import MMMPlotSuiteFacade
```

- [ ] **Step 5: Add `plot_suite` parameter to `MMM.__init__`**

In `mmm.py`, find the last parameter before `-> None:` in `MMM.__init__` (currently `cost_per_unit`, ending around line 450). Add `plot_suite` as a new keyword-only parameter after `cost_per_unit`:

```python
        plot_suite: Literal["legacy", "new"] = Field(
            "legacy",
            description="Which plot suite to use. 'legacy' for the monolithic MMMPlotSuite, 'new' for the namespace-based MMMPlotSuiteFacade.",
        ),
```

- [ ] **Step 6: Store `plot_suite` in `__init__` body**

In the `__init__` body, find the line:
```python
        self._cost_per_unit_input = cost_per_unit
```
Add after it:
```python
        self._plot_suite = plot_suite
        self._plot_suite_warned = False
```

- [ ] **Step 7: Run tests**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_mmm.py::test_mmm_plot_suite_defaults_to_legacy -v
```

Expected: PASS.

### 3b: Update `MMM.plot` property

- [ ] **Step 8: Write the failing test for `MMM.plot` behavior**

Add to `tests/mmm/test_mmm.py`:

```python
import warnings
from pymc_marketing.mmm.plot import MMMPlotSuite
from pymc_marketing.mmm.plotting import MMMPlotSuiteFacade


def test_mmm_plot_legacy_returns_legacy_suite(fit_mmm):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = fit_mmm.plot
    assert isinstance(result, MMMPlotSuite)


def test_mmm_plot_legacy_emits_future_warning(fit_mmm):
    fit_mmm._plot_suite_warned = False  # reset flag
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit_mmm.plot
    assert any(issubclass(warning.category, FutureWarning) for warning in w)
    assert any("2.0.0" in str(warning.message) for warning in w)


def test_mmm_plot_legacy_warns_only_once(fit_mmm):
    fit_mmm._plot_suite_warned = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit_mmm.plot
        fit_mmm.plot
    future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
    assert len(future_warnings) == 1


def test_mmm_plot_new_returns_facade(fit_mmm):
    fit_mmm._plot_suite = "new"
    result = fit_mmm.plot
    assert isinstance(result, MMMPlotSuiteFacade)
```

> **Note:** If the file already has tests that call `mmm.plot` without suppressing `FutureWarning`, they'll start emitting warnings. You may need to add `warnings.catch_warnings()` context managers around those too, or add `@pytest.mark.filterwarnings("ignore::FutureWarning")` decorator.

- [ ] **Step 9: Run to verify failure**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_mmm.py::test_mmm_plot_legacy_returns_legacy_suite -v
```

Expected: FAIL — `_plot_suite_warned` doesn't exist yet on the fitted fixture instance.

- [ ] **Step 10: Replace `MMM.plot` property in `mmm.py`**

Find and replace the current property (lines 1117–1125):

```python
    @property
    def plot(self) -> MMMPlotSuite:
        """Use the MMMPlotSuite to plot the results."""
        self._validate_model_was_built()
        self._validate_idata_exists()
        data = self.data
        # TODO: We would like to validate the data here for the plot suite using data.validate_or_raise()
        # However the schema is not very flexiable and the plot suite is (too) flexiable.
        return MMMPlotSuite(data=data)
```

Replace with:

```python
    @property
    def plot(self) -> MMMPlotSuite | MMMPlotSuiteFacade:
        """Access the plot suite for visualizing MMM results."""
        self._validate_model_was_built()
        self._validate_idata_exists()
        if self._plot_suite == "legacy":
            if not getattr(self, "_plot_suite_warned", False):
                warnings.warn(
                    "The legacy MMMPlotSuite will be removed in pymc-marketing 2.0.0. "
                    "Pass plot_suite='new' to opt in to the new namespace-based API. "
                    "See the migration guide: "
                    "docs/source/notebooks/mmm/mmm_plot_suite_migration_guide.ipynb",
                    FutureWarning,
                    stacklevel=2,
                )
                self._plot_suite_warned = True
            # TODO: We would like to validate the data here for the plot suite using data.validate_or_raise()
            # However the schema is not very flexible and the plot suite is (too) flexible.
            return MMMPlotSuite(data=self.data)
        return MMMPlotSuiteFacade(data=self.data)
```

- [ ] **Step 11: Run tests**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_mmm.py::test_mmm_plot_legacy_returns_legacy_suite tests/mmm/test_mmm.py::test_mmm_plot_legacy_emits_future_warning tests/mmm/test_mmm.py::test_mmm_plot_legacy_warns_only_once tests/mmm/test_mmm.py::test_mmm_plot_new_returns_facade -v
```

Expected: PASS.

### 3c: Serialization round-trip

- [ ] **Step 12: Write failing serialization test**

Add to `tests/mmm/test_mmm.py`:

```python
def test_mmm_plot_suite_roundtrips_through_save_load(fit_mmm, tmp_path):
    """plot_suite='new' survives save/load."""
    import copy
    mmm = copy.deepcopy(fit_mmm)
    mmm._plot_suite = "new"
    fpath = str(tmp_path / "mmm_test.nc")
    mmm.save(fpath)
    loaded = MMM.load(fpath)
    assert loaded._plot_suite == "new"
```

- [ ] **Step 13: Run to verify failure**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_mmm.py::test_mmm_plot_suite_roundtrips_through_save_load -v
```

Expected: FAIL — `_plot_suite` not in attrs yet, so loaded instance defaults to `"legacy"`.

- [ ] **Step 14: Add `plot_suite` to `create_idata_attrs` in `mmm.py`**

Find `create_idata_attrs` (around line 967). After the line:
```python
        attrs["cost_per_unit"] = ...
```
(i.e., after the `if self._cost_per_unit_input is not None` block, at the end of the method), add:

```python
        attrs["plot_suite"] = json.dumps(self._plot_suite)
```

- [ ] **Step 15: Add `plot_suite` to `attrs_to_init_kwargs` in `mmm.py`**

Find `attrs_to_init_kwargs` (around line 1068). Add `"plot_suite"` to the returned dict after `"cost_per_unit"`:

```python
            "plot_suite": json.loads(attrs.get("plot_suite", '"legacy"')),
```

- [ ] **Step 16: Run serialization test**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_mmm.py::test_mmm_plot_suite_roundtrips_through_save_load -v
```

Expected: PASS.

- [ ] **Step 17: Run pre-commit**

```bash
conda run -n pymc-dev-2527 pre-commit run --files pymc_marketing/mmm/mmm.py
```

- [ ] **Step 18: Run all MMM tests to check for regressions**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_mmm.py -v --tb=short -x 2>&1 | tail -30
```

Suppress the new `FutureWarning` in any existing test that calls `.plot` without expecting it:

For any failing test that calls `mmm.plot` and gets unexpected warning output, add `@pytest.mark.filterwarnings("ignore::FutureWarning")` or wrap in `warnings.catch_warnings`.

- [ ] **Step 19: Commit**

```bash
git add pymc_marketing/mmm/mmm.py tests/mmm/test_mmm.py
git commit -m "feat(mmm): add plot_suite arg to MMM with legacy/new switching and FutureWarning"
```

---

## Task 4: `BudgetOptimizerWrapper.plot` property

**Files:**
- Modify: `pymc_marketing/mmm/mmm.py`
- Modify: `tests/mmm/test_budget_optimizer_mmm.py`

- [ ] **Step 1: Write the failing test**

In `tests/mmm/test_budget_optimizer_mmm.py`, add. The existing fitted-MMM and data fixtures are `fitted_mmm` and `dummy_df` (both `scope="module"`). `BudgetOptimizerWrapper` objects are constructed inline — do the same:

```python
import warnings
import pandas as pd
from pymc_marketing.mmm.plot import MMMPlotSuite
from pymc_marketing.mmm.plotting.budget import BudgetPlots


def test_budget_optimizer_plot_legacy_returns_mmm_plot_suite(fitted_mmm, dummy_df):
    """In legacy mode, BudgetOptimizerWrapper.plot returns the legacy MMMPlotSuite."""
    _df_kwargs, X_dummy, _y_dummy = dummy_df
    optimizable_model = BudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=4),
    )
    optimizable_model.model_class._plot_suite = "legacy"
    optimizable_model.model_class._plot_suite_warned = False
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = optimizable_model.plot
    assert isinstance(result, MMMPlotSuite)


def test_budget_optimizer_plot_new_returns_budget_plots(fitted_mmm, dummy_df):
    """In new mode, BudgetOptimizerWrapper.plot returns BudgetPlots."""
    _df_kwargs, X_dummy, _y_dummy = dummy_df
    optimizable_model = BudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=4),
    )
    optimizable_model.model_class._plot_suite = "new"
    result = optimizable_model.plot
    assert isinstance(result, BudgetPlots)
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_budget_optimizer_mmm.py::test_budget_optimizer_plot_legacy_returns_mmm_plot_suite -v
```

Expected: FAIL — `BudgetOptimizerWrapper` has no explicit `.plot` property; the `__getattr__` delegation path does not produce the right branch behavior in this test.

- [ ] **Step 3: Add `BudgetPlots` import to `mmm.py`**

Find the `from pymc_marketing.mmm.plot import MMMPlotSuite` line (224) and add nearby:

```python
from pymc_marketing.mmm.plotting.budget import BudgetPlots
```

- [ ] **Step 4: Add `.plot` property to `BudgetOptimizerWrapper`**

Find `BudgetOptimizerWrapper.__getattr__` (around line 3582). Add the `plot` property **before** `__getattr__`:

```python
    @property
    def plot(self) -> BudgetPlots | MMMPlotSuite:
        """Access budget plotting functionality."""
        if self.model_class._plot_suite == "new":
            return BudgetPlots()
        return self.model_class.plot
```

- [ ] **Step 5: Run tests**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_budget_optimizer_mmm.py::test_budget_optimizer_plot_legacy_returns_mmm_plot_suite tests/mmm/test_budget_optimizer_mmm.py::test_budget_optimizer_plot_new_returns_budget_plots -v
```

Expected: PASS.

- [ ] **Step 6: Run pre-commit and commit**

```bash
conda run -n pymc-dev-2527 pre-commit run --files pymc_marketing/mmm/mmm.py tests/mmm/test_budget_optimizer_mmm.py
git add pymc_marketing/mmm/mmm.py tests/mmm/test_budget_optimizer_mmm.py
git commit -m "feat(mmm): add explicit plot property to BudgetOptimizerWrapper"
```

---

## Task 5: `TimeSliceCrossValidator` — revert `.plot` to legacy default

**Files:**
- Modify: `pymc_marketing/mmm/time_slice_cross_validation.py`
- Modify: `tests/mmm/test_time_slice_cross_validator.py`

- [ ] **Step 1: Write the failing tests**

In `tests/mmm/test_time_slice_cross_validator.py`, replace the existing `test_plot_property_returns_mmm_cv_plot_suite` with two tests (keeping the same helper infrastructure):

```python
def test_plot_property_legacy_default_returns_mmm_plot_suite():
    """TimeSliceCrossValidator.plot returns MMMPlotSuite by default (legacy mode)."""
    dates1 = pd.to_datetime(["2025-01-01", "2025-01-08"])
    dates2 = pd.to_datetime(["2025-01-15", "2025-01-22"])
    df_train = pd.DataFrame({"date": dates1})
    df_test = pd.DataFrame({"date": dates2})

    r1 = TimeSliceCrossValidationResult(
        X_train=df_train,
        y_train=pd.Series([1, 2]),
        X_test=df_test,
        y_test=pd.Series([3, 4]),
        idata=_build_simple_idata(dates1),
    )
    r2 = TimeSliceCrossValidationResult(
        X_train=df_train,
        y_train=pd.Series([5, 6]),
        X_test=df_test,
        y_test=pd.Series([7, 8]),
        idata=_build_simple_idata(dates2),
    )

    cv = TimeSliceCrossValidator.__new__(TimeSliceCrossValidator)
    cv._cv_results = [r1, r2]
    cv._plot_suite = "legacy"
    cv._combine_idata([r1, r2], ["fold_0", "fold_1"])  # sets cv.idata from last fold

    result = cv.plot
    assert isinstance(result, MMMPlotSuite)


def test_plot_property_new_mode_returns_mmm_cv_plot_suite():
    """TimeSliceCrossValidator.plot returns MMMCVPlotSuite when plot_suite='new'."""
    dates1 = pd.to_datetime(["2025-01-01", "2025-01-08"])
    dates2 = pd.to_datetime(["2025-01-15", "2025-01-22"])
    df_train = pd.DataFrame({"date": dates1})
    df_test = pd.DataFrame({"date": dates2})

    r1 = TimeSliceCrossValidationResult(
        X_train=df_train,
        y_train=pd.Series([1, 2]),
        X_test=df_test,
        y_test=pd.Series([3, 4]),
        idata=_build_simple_idata(dates1),
    )
    r2 = TimeSliceCrossValidationResult(
        X_train=df_train,
        y_train=pd.Series([5, 6]),
        X_test=df_test,
        y_test=pd.Series([7, 8]),
        idata=_build_simple_idata(dates2),
    )

    cv = TimeSliceCrossValidator.__new__(TimeSliceCrossValidator)
    cv._cv_results = [r1, r2]
    cv._plot_suite = "new"
    cv_idata = cv._combine_idata([r1, r2], ["fold_0", "fold_1"])
    cv.cv_idata = cv_idata

    result = cv.plot
    assert isinstance(result, MMMCVPlotSuite)
```

- [ ] **Step 2: Run to verify state**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_time_slice_cross_validator.py::test_plot_property_legacy_default_returns_mmm_plot_suite -v
```

Expected: FAIL — `_plot_suite` attr not present, and `.plot` currently returns `MMMCVPlotSuite` unconditionally.

- [ ] **Step 3: Add `MMMPlotSuite` import in `time_slice_cross_validation.py`**

Find line 34:
```python
from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite
```
Add after it:
```python
from pymc_marketing.mmm.plot import MMMPlotSuite
```

- [ ] **Step 4: Add `plot_suite` parameter to `TimeSliceCrossValidator.__init__`**

Find the `__init__` signature (line 155). Add `plot_suite` as the last parameter before `-> None`:

```python
        plot_suite: Literal["legacy", "new"] = "legacy",
```

(`Literal` is already imported in this file at line 25.)

In the `__init__` body, find:
```python
        self.sampler_config = sampler_config
```
Add after it:
```python
        self._plot_suite = plot_suite
```

- [ ] **Step 5: Replace the `.plot` property**

Find the current property (lines 177–185):
```python
    @property
    def plot(self) -> MMMCVPlotSuite:
        """Plotting suite for cross-validation results."""
        self._validate_model_was_built()
        if not hasattr(self, "cv_idata"):
            raise ValueError(
                "cv_idata is not available. Ensure TimeSliceCrossValidator.run() completed successfully."
            )
        return MMMCVPlotSuite(self.cv_idata)
```

Replace with:

```python
    @property
    def plot(self) -> MMMPlotSuite | MMMCVPlotSuite:
        """Plotting suite for cross-validation results."""
        self._validate_model_was_built()
        if self._plot_suite == "legacy":
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

- [ ] **Step 6: Run tests**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_time_slice_cross_validator.py::test_plot_property_legacy_default_returns_mmm_plot_suite tests/mmm/test_time_slice_cross_validator.py::test_plot_property_new_mode_returns_mmm_cv_plot_suite -v
```

Expected: PASS.

- [ ] **Step 7: Remove or rename the old test**

Delete the old test `test_plot_property_returns_mmm_cv_plot_suite` from `test_time_slice_cross_validator.py` since it is superseded by the two new tests above.

- [ ] **Step 8: Run the full CV test suite to check for regressions**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_time_slice_cross_validator.py -v --tb=short 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 9: Run pre-commit and commit**

```bash
conda run -n pymc-dev-2527 pre-commit run --files pymc_marketing/mmm/time_slice_cross_validation.py tests/mmm/test_time_slice_cross_validator.py
git add pymc_marketing/mmm/time_slice_cross_validation.py tests/mmm/test_time_slice_cross_validator.py
git commit -m "feat(mmm): add plot_suite arg to TimeSliceCrossValidator, revert .plot to legacy default"
```

---

## Task 6: Update notebooks that use new-mode CV plots

**Files:**
- Modify: `docs/source/notebooks/mmm/mmm_time_slice_cross_validation.ipynb`
- Modify: `docs/source/notebooks/mmm/mmm_roas.ipynb`

Both notebooks call `cv.plot.param_stability(...)`, `cv.plot.predictions(...)`, `cv.plot.crps(...)` — these are `MMMCVPlotSuite` methods. They'll break now that `.plot` defaults to legacy. Fix: add `plot_suite="new"` to the `TimeSliceCrossValidator(...)` constructor call in each notebook.

- [ ] **Step 1: Update `mmm_time_slice_cross_validation.ipynb`**

Use Python to update the notebook programmatically:

```python
import json

path = "docs/source/notebooks/mmm/mmm_time_slice_cross_validation.ipynb"
with open(path) as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "TimeSliceCrossValidator(" in src and "import" not in src:
        # Find the closing paren line and add plot_suite before it
        new_lines = []
        for line in cell["source"]:
            if line.strip() == ")":
                new_lines.append("    plot_suite=\"new\",\n")
            new_lines.append(line)
        cell["source"] = new_lines
        break  # only the first constructor call cell

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
```

After running, open the notebook and verify the cell now reads:
```python
cv = TimeSliceCrossValidator(
    n_init=163,
    forecast_horizon=12,
    date_column="date",
    step_size=1,
    plot_suite="new",
)
```

- [ ] **Step 2: Update `mmm_roas.ipynb`**

```python
import json

path = "docs/source/notebooks/mmm/mmm_roas.ipynb"
with open(path) as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "TimeSliceCrossValidator(" in src and "import" not in src:
        new_lines = []
        for line in cell["source"]:
            if line.strip() == ")":
                new_lines.append("    plot_suite=\"new\",\n")
            new_lines.append(line)
        cell["source"] = new_lines
        break

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
```

After running, verify the cell reads:
```python
cv = TimeSliceCrossValidator(
    n_init=115,
    forecast_horizon=12,
    date_column="date",
    step_size=1,
    plot_suite="new",
)
```

> **Note:** The `mmm_roas.ipynb` notebook has multiple `TimeSliceCrossValidator` usages. Verify that all constructor call cells were updated (run the grep again to confirm). If the loop only caught the first, run it in a `for cell in nb["cells"]` loop without `break`.

- [ ] **Step 3: Run pre-commit on notebooks**

```bash
conda run -n pymc-dev-2527 pre-commit run --files docs/source/notebooks/mmm/mmm_time_slice_cross_validation.ipynb docs/source/notebooks/mmm/mmm_roas.ipynb
```

- [ ] **Step 4: Commit**

```bash
git add docs/source/notebooks/mmm/mmm_time_slice_cross_validation.ipynb docs/source/notebooks/mmm/mmm_roas.ipynb
git commit -m "docs(mmm): add plot_suite='new' to CV notebooks that use MMMCVPlotSuite API"
```

---

## Task 7: Create `mmm_plot_suite_migration_guide.ipynb`

**Files:**
- Create: `docs/source/notebooks/mmm/mmm_plot_suite_migration_guide.ipynb`

This notebook uses markdown and code-snippet cells only (no execution output). It serves as the target of the `FutureWarning` link.

- [ ] **Step 1: Create the notebook**

```python
import json

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": [
        {
            "cell_type": "markdown",
            "id": "intro",
            "metadata": {},
            "source": [
                "# MMMPlotSuite v2 Migration Guide\n",
                "\n",
                "PyMC-Marketing is moving from the monolithic `MMMPlotSuite` to a namespace-based plotting API.\n",
                "The legacy suite will be removed in **pymc-marketing 2.0.0**.\n",
                "\n",
                "This guide covers how to opt in to the new API and migrate your existing code.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "opting-in",
            "metadata": {},
            "source": [
                "## Opting In\n",
                "\n",
                "Pass `plot_suite='new'` when constructing `MMM` or `TimeSliceCrossValidator`:\n",
                "\n",
                "```python\n",
                "from pymc_marketing.mmm import MMM, TimeSliceCrossValidator\n",
                "\n",
                "# MMM with new plot suite\n",
                "mmm = MMM(\n",
                "    ...,\n",
                "    plot_suite='new',\n",
                ")\n",
                "\n",
                "# Cross-validator with new plot suite\n",
                "cv = TimeSliceCrossValidator(\n",
                "    n_init=100,\n",
                "    forecast_horizon=12,\n",
                "    date_column='date',\n",
                "    plot_suite='new',\n",
                ")\n",
                "```\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "mmm-plot-namespace",
            "metadata": {},
            "source": [
                "## `mmm.plot` namespace map\n",
                "\n",
                "In the new API, `mmm.plot` returns a `MMMPlotSuiteFacade` with four sub-namespaces.\n",
                "\n",
                "### Diagnostics\n",
                "\n",
                "| Legacy | New |\n",
                "|--------|-----|\n",
                "| `mmm.plot.posterior_predictive(...)` | `mmm.plot.diagnostics.posterior_predictive(...)` |\n",
                "| `mmm.plot.prior_predictive(...)` | `mmm.plot.diagnostics.prior_predictive(...)` |\n",
                "| `mmm.plot.residuals(...)` | `mmm.plot.diagnostics.residuals(...)` |\n",
                "\n",
                "### Decomposition\n",
                "\n",
                "| Legacy | New |\n",
                "|--------|-----|\n",
                "| `mmm.plot.contributions_over_time(...)` | `mmm.plot.decomposition.contributions_over_time(...)` |\n",
                "| `mmm.plot.waterfall_components_decomposition(...)` | `mmm.plot.decomposition.waterfall_components_decomposition(...)` |\n",
                "| `mmm.plot.channel_contribution_share_hdi(...)` | `mmm.plot.decomposition.channel_contribution_share_hdi(...)` |\n",
                "\n",
                "### Sensitivity\n",
                "\n",
                "| Legacy | New |\n",
                "|--------|-----|\n",
                "| `mmm.plot.sensitivity_analysis(...)` | `mmm.plot.sensitivity.sensitivity_analysis(...)` |\n",
                "| `mmm.plot.saturation_scatterplot(...)` | `mmm.plot.sensitivity.saturation_scatterplot(...)` |\n",
                "\n",
                "### Transformation\n",
                "\n",
                "| Legacy | New |\n",
                "|--------|-----|\n",
                "| `mmm.plot.saturation_curves(...)` | `mmm.plot.transformation.saturation_curves(...)` |\n",
                "| `mmm.plot.adstock_curves(...)` | `mmm.plot.transformation.adstock_curves(...)` |\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "budget-plots",
            "metadata": {},
            "source": [
                "## Budget plots\n",
                "\n",
                "Budget plots have moved from `mmm.plot` to `optimizer.plot`.\n",
                "\n",
                "```python\n",
                "from pymc_marketing.mmm import BudgetOptimizerWrapper, MMM\n",
                "\n",
                "mmm = MMM(..., plot_suite='new')\n",
                "optimizer = BudgetOptimizerWrapper(mmm, start_date='2024-01-01', end_date='2024-12-31')\n",
                "\n",
                "# Run optimization first to get samples\n",
                "samples = optimizer.allocate_budget(...)\n",
                "\n",
                "# Legacy\n",
                "# optimizer.plot.budget_allocation(samples=samples)\n",
                "\n",
                "# New\n",
                "fig, axes = optimizer.plot.allocation_roas(samples=samples)\n",
                "fig, axes = optimizer.plot.contribution_over_time(samples=samples)\n",
                "```\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "cv-plots",
            "metadata": {},
            "source": [
                "## Cross-validation plots\n",
                "\n",
                "CV plots use `MMMCVPlotSuite` when `plot_suite='new'`.\n",
                "\n",
                "```python\n",
                "cv = TimeSliceCrossValidator(\n",
                "    n_init=100, forecast_horizon=12, date_column='date', plot_suite='new'\n",
                ")\n",
                "cv_idata = cv.run(X, y, mmm=mmm)\n",
                "\n",
                "# Parameter stability across folds\n",
                "cv.plot.param_stability(cv_idata)\n",
                "\n",
                "# Predictions vs actuals\n",
                "cv.plot.predictions()\n",
                "\n",
                "# CRPS score\n",
                "cv.plot.crps()\n",
                "```\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "removal-timeline",
            "metadata": {},
            "source": [
                "## Removal timeline\n",
                "\n",
                "The legacy `MMMPlotSuite` (accessed via `mmm.plot` with the default `plot_suite='legacy'`) will be **removed in pymc-marketing 2.0.0**.\n",
                "\n",
                "Until then, using the legacy suite emits a `FutureWarning` on first access. To suppress it, opt in to the new API with `plot_suite='new'`.\n"
            ]
        }
    ]
}

path = "docs/source/notebooks/mmm/mmm_plot_suite_migration_guide.ipynb"
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
```

> **Note on method names in the tables:** Before finalising, verify each method name in the tables above against the actual method names in `pymc_marketing/mmm/plotting/decomposition.py`, `diagnostics.py`, `sensitivity.py`, and `transformations.py`. Grep for `def ` in each file and update any incorrect names.

- [ ] **Step 2: Verify method names are correct**

```bash
grep -n "def " pymc_marketing/mmm/plotting/decomposition.py pymc_marketing/mmm/plotting/diagnostics.py pymc_marketing/mmm/plotting/sensitivity.py pymc_marketing/mmm/plotting/transformations.py | grep -v "__"
```

Update the notebook tables to match the actual method names.

- [ ] **Step 3: Run pre-commit on notebook**

```bash
conda run -n pymc-dev-2527 pre-commit run --files docs/source/notebooks/mmm/mmm_plot_suite_migration_guide.ipynb
```

- [ ] **Step 4: Commit**

```bash
git add docs/source/notebooks/mmm/mmm_plot_suite_migration_guide.ipynb
git commit -m "docs(mmm): add MMMPlotSuite v2 migration guide notebook"
```

---

## Task 8: Final regression check

- [ ] **Step 1: Run the full MMM and plotting test suite**

```bash
conda run -n pymc-dev-2527 pytest tests/mmm/test_mmm.py tests/mmm/test_time_slice_cross_validator.py tests/mmm/test_budget_optimizer_mmm.py tests/mmm/plotting/ -v --tb=short 2>&1 | tail -40
```

Expected: all pass, no unexpected failures.

- [ ] **Step 2: Commit any fixes, then final commit**

```bash
git add -p  # stage only relevant fixes
git commit -m "fix(mmm): address test regressions from plot_suite integration"
```

---

## Self-Review Notes

- **Spec §1 `MMM`:** Covered in Tasks 3a–3c (constructor arg, `.plot` property, serialization).
- **Spec §2 `MMMPlotSuiteFacade`:** Covered in Task 1 + Task 2 (file, exports).
- **Spec §3 `TimeSliceCrossValidator`:** Covered in Task 5 (revert, legacy path, new path, notebook updates).
- **Spec §4 `BudgetOptimizerWrapper`:** Covered in Task 4.
- **Spec §5 Migration guide:** Covered in Task 7.
- **FutureWarning pointing to migration guide:** Included in Task 3b step 10 — message references `mmm_plot_suite_migration_guide.ipynb`.
- **Notebook updates:** Covered in Task 6 for both `mmm_time_slice_cross_validation.ipynb` and `mmm_roas.ipynb`.
