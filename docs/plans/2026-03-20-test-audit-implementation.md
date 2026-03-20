# Test Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ~14 targeted tests to `test_transformations.py` filling
coverage gaps identified by the [test audit design](./2026-03-20-test-audit-design.md).

**Architecture:** All changes are in a single test file. No production
code changes. Tests are written against existing code so they should
pass immediately. Group related tests into existing or new test classes.

**Tech Stack:** pytest, xarray, numpy, matplotlib, arviz_plots.

**Python:** `/Users/imrisofer/miniconda3/envs/pymc-dev-pr1-foundation/bin/python`

**pytest:** `/Users/imrisofer/miniconda3/envs/pymc-dev-pr1-foundation/bin/pytest`

---

### Task 1: Add `_x_axis_label` and `_get_channel_x_data` unit tests

**Files:**
- Modify: `tests/mmm/plotting/test_transformations.py`

**Step 1: Add imports**

At the top of the file, update the import from `transformations` to also
import the two module-level helpers:

```python
from pymc_marketing.mmm.plotting.transformations import (
    _SCALED_SPACE_MAX_THRESHOLD,
    TransformationPlots,
    _ensure_chain_draw_dims,
    _get_channel_x_data,
    _x_axis_label,
)
```

**Step 2: Add `TestXAxisLabel` class**

Place it right after the fixtures section (before `TestSaturationScatterplotBasic`):

```python
class TestXAxisLabel:
    def test_returns_spend_when_cost_per_unit_available(self, panel_data):
        assert _x_axis_label(panel_data, apply_cost_per_unit=True) == "Spend"

    def test_returns_channel_data_when_no_cost_per_unit(self, simple_data):
        assert _x_axis_label(simple_data, apply_cost_per_unit=True) == "Channel Data"

    def test_returns_channel_data_when_flag_false(self, panel_data):
        assert _x_axis_label(panel_data, apply_cost_per_unit=False) == "Channel Data"
```

Note: `simple_data` has no `channel_spend` so `cost_per_unit` is `None`;
`panel_data` has `channel_spend`.

**Step 3: Add `TestGetChannelXData` class**

```python
class TestGetChannelXData:
    def test_returns_spend_when_flag_true(self, panel_data):
        result = _get_channel_x_data(panel_data, apply_cost_per_unit=True)
        expected = panel_data.get_channel_spend()
        xr.testing.assert_identical(result, expected)

    def test_returns_raw_data_when_flag_false(self, panel_data):
        result = _get_channel_x_data(panel_data, apply_cost_per_unit=False)
        expected = panel_data.get_channel_data()
        xr.testing.assert_identical(result, expected)
```

**Step 4: Run tests**

```bash
/Users/imrisofer/miniconda3/envs/pymc-dev-pr1-foundation/bin/pytest \
  tests/mmm/plotting/test_transformations.py::TestXAxisLabel \
  tests/mmm/plotting/test_transformations.py::TestGetChannelXData -v
```

Expected: 5 PASSED.

**Step 5: Run pre-commit**

```bash
pre-commit run --files tests/mmm/plotting/test_transformations.py
```

**Step 6: Commit**

```bash
git add tests/mmm/plotting/test_transformations.py
git commit -m "test: add unit tests for _x_axis_label and _get_channel_x_data"
```

---

### Task 2: Add `_ensure_chain_draw_dims` MultiIndex test

**Files:**
- Modify: `tests/mmm/plotting/test_transformations.py`

**Step 1: Add test to `TestEnsureChainDrawDims`**

```python
def test_multiindex_sample_unstacked(self, simple_curve):
    stacked = simple_curve.stack(sample=("chain", "draw"))
    result = _ensure_chain_draw_dims(stacked)
    assert "chain" in result.dims
    assert "draw" in result.dims
    assert "sample" not in result.dims
    assert result.sizes["chain"] == simple_curve.sizes["chain"]
    assert result.sizes["draw"] == simple_curve.sizes["draw"]
```

This tests the MultiIndex path: `simple_curve` has `(chain, draw)`,
`.stack()` creates a MultiIndex `sample`, and the function should
unstack it back.

**Step 2: Run test**

```bash
/Users/imrisofer/miniconda3/envs/pymc-dev-pr1-foundation/bin/pytest \
  tests/mmm/plotting/test_transformations.py::TestEnsureChainDrawDims::test_multiindex_sample_unstacked -v
```

Expected: PASSED.

**Step 3: Run pre-commit and commit**

```bash
pre-commit run --files tests/mmm/plotting/test_transformations.py
git add tests/mmm/plotting/test_transformations.py
git commit -m "test: add MultiIndex sample path test for _ensure_chain_draw_dims"
```

---

### Task 3: Add `hdi_prob=None` and mean-curve-with-samples tests

**Files:**
- Modify: `tests/mmm/plotting/test_transformations.py`

**Step 1: Add test to `TestSaturationCurvesHDI`**

```python
def test_hdi_prob_none_disables_band(self, simple_plots, simple_curve):
    _, axes = simple_plots.saturation_curves(
        curves=simple_curve, hdi_prob=None, n_samples=0
    )
    for ax in axes.flat:
        polys = [c for c in ax.collections if "Poly" in type(c).__name__]
        assert len(polys) == 0, "No HDI band when hdi_prob=None"
```

**Step 2: Add test to `TestSaturationCurvesSamples`**

```python
def test_mean_curve_present_with_samples(self, simple_plots, simple_curve):
    n = 5
    _, axes = simple_plots.saturation_curves(curves=simple_curve, n_samples=n)
    for ax in axes.flat:
        lines = ax.get_lines()
        assert len(lines) >= n + 1, (
            f"Expected at least {n + 1} lines (mean + {n} samples), "
            f"got {len(lines)}"
        )
```

**Step 3: Run tests**

```bash
/Users/imrisofer/miniconda3/envs/pymc-dev-pr1-foundation/bin/pytest \
  tests/mmm/plotting/test_transformations.py::TestSaturationCurvesHDI::test_hdi_prob_none_disables_band \
  tests/mmm/plotting/test_transformations.py::TestSaturationCurvesSamples::test_mean_curve_present_with_samples -v
```

Expected: 2 PASSED.

**Step 4: Run pre-commit and commit**

```bash
pre-commit run --files tests/mmm/plotting/test_transformations.py
git add tests/mmm/plotting/test_transformations.py
git commit -m "test: add hdi_prob=None and mean-curve-with-samples tests"
```

---

### Task 4: Add `*_kwargs` forwarding tests

**Files:**
- Modify: `tests/mmm/plotting/test_transformations.py`

**Step 1: Add test to `TestSaturationScatterplotCustomization`**

```python
def test_scatter_kwargs_forwarded(self, simple_plots):
    _, axes = simple_plots.saturation_scatterplot(
        scatter_kwargs={"alpha": 0.5}
    )
    for ax in axes.flat:
        for coll in ax.collections:
            alphas = coll.get_alpha()
            if alphas is not None:
                assert np.all(np.isclose(alphas, 0.5)), (
                    f"Expected alpha=0.5 from scatter_kwargs, got {alphas}"
                )
```

**Step 2: Add new class `TestSaturationCurvesKwargsForwarding`**

Place after `TestSaturationCurvesScaleWarning`:

```python
class TestSaturationCurvesKwargsForwarding:
    """Verify that per-element *_kwargs reach the matplotlib artists."""

    def test_scatter_kwargs_forwarded(self, simple_plots, simple_curve):
        _, axes = simple_plots.saturation_curves(
            curves=simple_curve,
            scatter_kwargs={"alpha": 0.5},
            n_samples=0,
            hdi_prob=None,
        )
        for ax in axes.flat:
            for coll in ax.collections:
                alphas = coll.get_alpha()
                if alphas is not None:
                    assert np.all(np.isclose(alphas, 0.5))

    def test_hdi_kwargs_forwarded(self, simple_plots, simple_curve):
        _, axes = simple_plots.saturation_curves(
            curves=simple_curve,
            hdi_kwargs={"alpha": 0.5},
            n_samples=0,
        )
        for ax in axes.flat:
            polys = [c for c in ax.collections if "Poly" in type(c).__name__]
            for poly in polys:
                assert np.isclose(poly.get_alpha(), 0.5)

    def test_mean_curve_kwargs_forwarded(self, simple_plots, simple_curve):
        _, axes = simple_plots.saturation_curves(
            curves=simple_curve,
            mean_curve_kwargs={"linewidth": 5},
            n_samples=0,
            hdi_prob=None,
        )
        for ax in axes.flat:
            for line in ax.get_lines():
                assert line.get_linewidth() == pytest.approx(5)

    def test_sample_curves_kwargs_forwarded(self, simple_plots, simple_curve):
        _, axes = simple_plots.saturation_curves(
            curves=simple_curve,
            sample_curves_kwargs={"alpha": 0.1},
            n_samples=3,
            hdi_prob=None,
        )
        for ax in axes.flat:
            lines = ax.get_lines()
            sample_lines = lines[1:]  # first line is mean curve
            for line in sample_lines:
                assert np.isclose(line.get_alpha(), 0.1), (
                    f"Expected alpha=0.1, got {line.get_alpha()}"
                )
```

**Step 3: Run tests**

```bash
/Users/imrisofer/miniconda3/envs/pymc-dev-pr1-foundation/bin/pytest \
  tests/mmm/plotting/test_transformations.py::TestSaturationScatterplotCustomization::test_scatter_kwargs_forwarded \
  tests/mmm/plotting/test_transformations.py::TestSaturationCurvesKwargsForwarding -v
```

Expected: 5 PASSED.

**Step 4: Run pre-commit and commit**

```bash
pre-commit run --files tests/mmm/plotting/test_transformations.py
git add tests/mmm/plotting/test_transformations.py
git commit -m "test: add *_kwargs forwarding tests for scatter, hdi, mean, samples"
```

---

### Task 5: Add `**pc_kwargs` forwarding smoke test

**Files:**
- Modify: `tests/mmm/plotting/test_transformations.py`

**Step 1: Add test to `TestSaturationScatterplotCustomization`**

```python
def test_pc_kwargs_forwarded(self, simple_plots):
    fig, axes = simple_plots.saturation_scatterplot(
        plot_kwargs={"zorder": 10}
    )
    assert isinstance(fig, Figure)
```

This is a smoke test — we just verify the extra kwarg doesn't raise.
Deep inspection of `PlotCollection` internals isn't practical.

**Step 2: Run test**

```bash
/Users/imrisofer/miniconda3/envs/pymc-dev-pr1-foundation/bin/pytest \
  tests/mmm/plotting/test_transformations.py::TestSaturationScatterplotCustomization::test_pc_kwargs_forwarded -v
```

Expected: PASSED.

**Step 3: Run pre-commit and commit**

```bash
pre-commit run --files tests/mmm/plotting/test_transformations.py
git add tests/mmm/plotting/test_transformations.py
git commit -m "test: add pc_kwargs forwarding smoke test"
```

---

### Task 6: Run full test suite and verify

**Step 1: Run all tests in both files**

```bash
/Users/imrisofer/miniconda3/envs/pymc-dev-pr1-foundation/bin/pytest \
  tests/mmm/plotting/test_helpers.py \
  tests/mmm/plotting/test_transformations.py -v --tb=short
```

Expected: All tests PASSED (existing + new).

**Step 2: Run pre-commit on both files**

```bash
pre-commit run --files tests/mmm/plotting/test_helpers.py tests/mmm/plotting/test_transformations.py
```

Expected: All hooks passed.
