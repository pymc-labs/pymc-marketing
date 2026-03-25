# Test Improvements for `test_transformations.py` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill coverage gaps, add behavioral verification via matplotlib artist inspection, and strengthen weak assertions in `tests/mmm/plotting/test_transformations.py`.

**Architecture:** All changes are confined to a single test file. No new fixtures are needed — all tests use the existing module-scoped fixtures. Tests are grouped into six independent tasks that can be committed separately. Each task follows TDD: write the test, verify it passes (the behavior already exists in the implementation), commit.

**Tech Stack:** pytest, numpy, matplotlib (Agg backend), xarray, arviz

---

## File Map

| Action | File |
|--------|------|
| Modify | `tests/mmm/plotting/test_transformations.py` |

Read these before starting:
- `pymc_marketing/mmm/plotting/transformations.py` — the implementation under test
- `pymc_marketing/mmm/plotting/_helpers.py` — `_select_dims`, `_process_plot_params`
- `docs/superpowers/specs/2026-03-25-test-transformations-improvements-design.md` — the full spec

---

## How to run tests

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_transformations.py -v
```

To run a single class:
```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_transformations.py::ClassName -v
```

---

## Task 1: Coverage gap — dim validation

**What:** Add missing invalid-dim-key test to `TestSaturationScatterplotDims` and create `TestSaturationCurvesDimsInvalid` with two tests (invalid key, invalid value).

**Why this is a gap:** `_select_dims` raises `ValueError` for both unknown keys and unknown values, but only the value-error path was tested for `saturation_scatterplot`, and *neither* path was tested for `saturation_curves`.

**Files:** Modify `tests/mmm/plotting/test_transformations.py`

- [ ] **Step 1: Add `test_invalid_dim_key_raises` to `TestSaturationScatterplotDims`**

  In `TestSaturationScatterplotDims` (after the existing `test_invalid_dim_value_raises` test, around line 233):

  ```python
  def test_invalid_dim_key_raises(self, panel_plots):
      with pytest.raises(ValueError, match="Dimension 'bad_dim' not found"):
          panel_plots.saturation_scatterplot(dims={"bad_dim": "US"})
  ```

- [ ] **Step 2: Add `TestSaturationCurvesDimsInvalid` after `TestSaturationCurvesDims`**

  Insert after `TestSaturationCurvesDims` (around line 468):

  ```python
  class TestSaturationCurvesDimsInvalid:
      def test_invalid_dim_key_raises(self, panel_plots, panel_curve):
          with pytest.raises(ValueError, match="Dimension 'bad_dim' not found"):
              panel_plots.saturation_curves(curves=panel_curve, dims={"bad_dim": "US"})

      def test_invalid_dim_value_raises(self, panel_plots, panel_curve):
          with pytest.raises(ValueError, match="Value 'FR' not found"):
              panel_plots.saturation_curves(curves=panel_curve, dims={"country": "FR"})
  ```

- [ ] **Step 3: Run the new tests**

  ```bash
  conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_transformations.py::TestSaturationScatterplotDims::test_invalid_dim_key_raises tests/mmm/plotting/test_transformations.py::TestSaturationCurvesDimsInvalid -v
  ```

  Expected: All 3 PASS.

- [ ] **Step 4: Commit**

  ```bash
  git add tests/mmm/plotting/test_transformations.py
  git commit -m "test: add invalid dim key/value tests for scatterplot and curves"
  ```

---

## Task 2: Coverage gap — edge cases (`hdi_prob=None` and `n_samples` clamping)

**What:** Create `TestSaturationCurvesEdgeCases` with two tests.

**Why these are gaps:**
- `hdi_prob=None` disables the HDI band (`if hdi_prob is not None:` branch) — never tested.
- When `n_samples` exceeds available draws, `min(n_samples, stacked.sizes["sample"])` clamps it — never tested.

**How the matplotlib check works:**
- HDI band: `azp.visuals.fill_between_y` creates a `PolyCollection`. Filter by `"Poly" in type(c).__name__`. When `hdi_prob=None`, zero such collections should exist.
- Line count: `ax.get_lines()` returns `Line2D` objects only (scatter uses `PathCollection`, so it's not included). With `n_samples=10_000` clamped to 100, expect `100 + 1 = 101` lines.

**Files:** Modify `tests/mmm/plotting/test_transformations.py`

- [ ] **Step 1: Add `TestSaturationCurvesEdgeCases` after `TestSaturationCurvesDimsInvalid`**

  ```python
  class TestSaturationCurvesEdgeCases:
      def test_hdi_prob_none_disables_band(self, simple_plots, simple_curve):
          _, axes = simple_plots.saturation_curves(
              curves=simple_curve, hdi_prob=None, n_samples=0
          )
          for ax in axes.flat:
              polys = [c for c in ax.collections if "Poly" in type(c).__name__]
              assert len(polys) == 0, "No HDI fill when hdi_prob=None"

      def test_n_samples_clamped_to_available(self, simple_plots, simple_curve):
          # simple_curve: chain=2, draw=50 → stacked.sizes["sample"] = 100
          n_available = 2 * 50
          _, axes = simple_plots.saturation_curves(
              curves=simple_curve, n_samples=10_000, hdi_prob=None
          )
          for ax in axes.flat:
              lines = ax.get_lines()
              # 1 mean line + n_available sample lines (clamped from 10_000)
              assert len(lines) == n_available + 1
  ```

- [ ] **Step 2: Run the new tests**

  ```bash
  conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_transformations.py::TestSaturationCurvesEdgeCases -v
  ```

  Expected: Both PASS.

  Note: `test_n_samples_clamped_to_available` may be slow (~10 seconds) because it renders 100 sample lines. This is expected.

- [ ] **Step 3: Commit**

  ```bash
  git add tests/mmm/plotting/test_transformations.py
  git commit -m "test: add edge case tests for hdi_prob=None and n_samples clamping"
  ```

---

## Task 3: Coverage gap — customization and idata mutation guard

**What:** Add `test_figsize_overrides_figure_kwargs_warns` to `TestSaturationScatterplotCustomization`, and add `test_idata_override_does_not_mutate_self_data` to `TestSaturationCurvesIdataOverride`.

**Why these are gaps:**
- `_process_plot_params` in `_helpers.py` emits a `UserWarning` when `figsize` and `figure_kwargs["figsize"]` are both set — untested.
- The idata-override mutation guard was tested for `saturation_scatterplot` but not for `saturation_curves`.

**Files:** Modify `tests/mmm/plotting/test_transformations.py`

- [ ] **Step 1: Add `test_figsize_overrides_figure_kwargs_warns` to `TestSaturationScatterplotCustomization`**

  In `TestSaturationScatterplotCustomization` (after `test_non_matplotlib_backend_without_return_as_pc_raises`):

  ```python
  def test_figsize_overrides_figure_kwargs_warns(self, simple_plots):
      with pytest.warns(UserWarning, match="figsize parameter overrides"):
          simple_plots.saturation_scatterplot(
              figsize=(10, 5),
              figure_kwargs={"figsize": (8, 4)},
          )
  ```

- [ ] **Step 2: Add `test_idata_override_does_not_mutate_self_data` to `TestSaturationCurvesIdataOverride`**

  In `TestSaturationCurvesIdataOverride` (after `test_idata_override`):

  ```python
  def test_idata_override_does_not_mutate_self_data(
      self, simple_plots, simple_data, panel_idata, panel_curve
  ):
      simple_plots.saturation_curves(
          curves=panel_curve, idata=panel_idata, n_samples=2
      )
      assert simple_plots._data is simple_data
  ```

- [ ] **Step 3: Run the new tests**

  ```bash
  conda run -n pymc-marketing-dev pytest \
    "tests/mmm/plotting/test_transformations.py::TestSaturationScatterplotCustomization::test_figsize_overrides_figure_kwargs_warns" \
    "tests/mmm/plotting/test_transformations.py::TestSaturationCurvesIdataOverride::test_idata_override_does_not_mutate_self_data" \
    -v
  ```

  Expected: Both PASS.

- [ ] **Step 4: Commit**

  ```bash
  git add tests/mmm/plotting/test_transformations.py
  git commit -m "test: add figsize conflict warning and curves idata mutation guard tests"
  ```

---

## Task 4: Behavioral verification — scatter data values

**What:** Add `TestSaturationScatterplotDataValues` with three tests that extract rendered data from matplotlib `PathCollection` artists and compare to values computed directly from fixture data.

**How `PathCollection.get_offsets()` works:**
`ax.collections[0]` is the scatter's `PathCollection`. `.get_offsets()` returns a masked array of shape `(n_dates, 2)` where column 0 is x-data and column 1 is y-data. Column ordering matches the order `azp.visuals.scatter_xy` was called.

**What the implementation plots:**
- y: `contributions.mean(dim=["chain", "draw"])` where `contributions = data.get_channel_contributions(original_scale=...)`
- x (when `apply_cost_per_unit=False`): `data.get_channel_data()`

The axes in the returned array iterate over channels in order (tv, radio, social for `simple_plots`).

**Files:** Modify `tests/mmm/plotting/test_transformations.py`

- [ ] **Step 1: Add `TestSaturationScatterplotDataValues` after `TestSaturationScatterplotLabels`**

  ```python
  class TestSaturationScatterplotDataValues:
      def test_scatter_y_matches_mean_contributions(self, simple_plots, simple_data):
          _, axes = simple_plots.saturation_scatterplot(original_scale=True)
          channels = simple_data.channels
          contributions = simple_data.get_channel_contributions(original_scale=True)
          expected_y = contributions.mean(dim=["chain", "draw"])

          for ax, ch in zip(axes.flat, channels):
              offsets = ax.collections[0].get_offsets()
              np.testing.assert_allclose(
                  offsets[:, 1], expected_y.sel(channel=ch).values, rtol=1e-5
              )

      def test_scatter_x_matches_channel_data(self, simple_plots, simple_data):
          _, axes = simple_plots.saturation_scatterplot(apply_cost_per_unit=False)
          channels = simple_data.channels
          x_data = simple_data.get_channel_data()

          for ax, ch in zip(axes.flat, channels):
              offsets = ax.collections[0].get_offsets()
              np.testing.assert_allclose(
                  offsets[:, 0], x_data.sel(channel=ch).values, rtol=1e-5
              )

      def test_original_scale_changes_scatter_y(self, simple_plots):
          _, axes_true = simple_plots.saturation_scatterplot(original_scale=True)
          _, axes_false = simple_plots.saturation_scatterplot(original_scale=False)
          offsets_true = axes_true.flat[0].collections[0].get_offsets()
          offsets_false = axes_false.flat[0].collections[0].get_offsets()
          assert not np.allclose(offsets_true[:, 1], offsets_false[:, 1])
  ```

  > **Why the last test works:** `simple_idata` has both `channel_contribution` and `channel_contribution_original_scale`. The latter is `channel_contribution * 100` (see fixture). So `original_scale=True` and `False` produce y-values that differ by a factor of ~100 — `np.allclose` will not hold.

- [ ] **Step 2: Run the new tests**

  ```bash
  conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_transformations.py::TestSaturationScatterplotDataValues -v
  ```

  Expected: All 3 PASS.

- [ ] **Step 3: Commit**

  ```bash
  git add tests/mmm/plotting/test_transformations.py
  git commit -m "test: add scatter data value verification via PathCollection inspection"
  ```

---

## Task 5: Behavioral verification — curves data values

**What:** Add `TestSaturationCurvesDataValues` with three tests: mean curve y-values, HDI band bounds containing mean, and exact line count.

**How mean curve extraction works:**
With `n_samples=0, hdi_prob=None`, `saturation_curves` draws only the mean line. `ax.get_lines()` returns `Line2D` objects. `lines[0].get_ydata()` is the y-data of the mean curve.

The implementation computes: `mean_curve = curves.mean(dim=["chain", "draw"])` (after `_ensure_chain_draw_dims`). Since `simple_curve` already has chain/draw dims, `_ensure_chain_draw_dims` returns an identical copy, so `expected_mean = simple_curve.mean(dim=["chain", "draw"])` is the correct oracle. Note: `saturation_curves` also replaces `curves["x"]` with `range(n)` before computing the mean — this changes the coordinate but not the data values, so `expected_mean.values` remains correct.

**How HDI polygon extraction works:**
`matplotlib.fill_between` for `n` x-points produces a `PolyCollection` whose path has shape `(2*n+3, 2)`:
- index 0: closing/start point
- indices `1..n` (inclusive): lower bound, left-to-right
- index `n+1`: join point
- indices `n+2..2*n+1` (inclusive): upper bound, right-to-left
- index `2*n+2`: closing point

Extract lower with `verts[1:n+1, 1]`, upper with `verts[n+2:2*n+2, 1][::-1]`.

**Files:** Modify `tests/mmm/plotting/test_transformations.py`

- [ ] **Step 1: Add `TestSaturationCurvesDataValues` after `TestSaturationCurvesLabels`**

  ```python
  class TestSaturationCurvesDataValues:
      def test_mean_curve_y_matches_curves_mean(self, simple_plots, simple_curve):
          # simple_curve already has (chain, draw, channel, x) dims
          # saturation_curves replaces curves["x"] internally but that only changes
          # the coordinate, not the data values — expected_mean.values is unaffected
          expected_mean = simple_curve.mean(dim=["chain", "draw"])
          channels = ["tv", "radio", "social"]

          _, axes = simple_plots.saturation_curves(
              curves=simple_curve, n_samples=0, hdi_prob=None
          )
          for ax, ch in zip(axes.flat, channels):
              line = ax.get_lines()[0]
              np.testing.assert_allclose(
                  line.get_ydata(), expected_mean.sel(channel=ch).values, rtol=1e-5
              )

      def test_hdi_band_bounds_contain_mean(self, simple_plots, simple_curve):
          # simple_curve already has (chain, draw, channel, x) dims
          expected_mean = simple_curve.mean(dim=["chain", "draw"])
          channels = ["tv", "radio", "social"]

          _, axes = simple_plots.saturation_curves(
              curves=simple_curve, n_samples=0, hdi_prob=0.94
          )
          for ax, ch in zip(axes.flat, channels):
              polys = [c for c in ax.collections if "Poly" in type(c).__name__]
              assert len(polys) > 0
              verts = polys[0].get_paths()[0].vertices  # shape (2*n+3, 2)
              n = len(expected_mean.sel(channel=ch))
              y_lower = verts[1 : n + 1, 1]                    # indices 1..n
              y_upper = verts[n + 2 : 2 * n + 2, 1][::-1]     # indices n+2..2n+1, reversed
              mean_vals = expected_mean.sel(channel=ch).values
              assert np.all(mean_vals >= y_lower - 1e-6)
              assert np.all(mean_vals <= y_upper + 1e-6)

      def test_exact_line_count_with_samples(self, simple_plots, simple_curve):
          n = 5
          _, axes = simple_plots.saturation_curves(
              curves=simple_curve, n_samples=n, hdi_prob=None
          )
          for ax in axes.flat:
              assert len(ax.get_lines()) == n + 1
  ```

- [ ] **Step 2: Run the new tests**

  ```bash
  conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_transformations.py::TestSaturationCurvesDataValues -v
  ```

  Expected: All 3 PASS.

  If `test_hdi_band_bounds_contain_mean` fails with an index error, print `verts.shape` to verify the `(2*n+3, 2)` layout. The shape depends on the version of matplotlib installed — if it differs, adjust the slices accordingly and update the spec comment.

- [ ] **Step 3: Commit**

  ```bash
  git add tests/mmm/plotting/test_transformations.py
  git commit -m "test: add curves data value verification (mean line, HDI bounds, line count)"
  ```

---

## Task 6: Strengthen weak tests + remove redundant tests

**What:** Four in-place changes to existing tests.

1. **`test_original_scale_true_is_default`** — verify it actually reflects a different scale from `original_scale=False`, not just that a Figure is returned.
2. **`test_sample_curves_drawn`** in `TestSaturationCurvesSamples` — `>= 5` → `== 6`.
3. **`test_sample_curves_drawn`** in `TestSaturationCurvesWithSampleDim` — `>= 5` → `== 6`.
4. **`test_original_scale_false`** in `TestSaturationCurvesWithSampleDim` — Figure assertion → no-warning assertion.
5. **Remove** `TestSaturationScatterplotCostPerUnit` — both its tests only assert `isinstance(fig, Figure)`, which is already covered by `TestSaturationScatterplotLabels` which also checks the label content.

**Files:** Modify `tests/mmm/plotting/test_transformations.py`

- [ ] **Step 1: Upgrade `test_original_scale_true_is_default` in `TestSaturationScatterplotBasic`**

  Find (around line 197):
  ```python
  def test_original_scale_true_is_default(self, simple_plots):
      fig, _axes = simple_plots.saturation_scatterplot()
      assert isinstance(fig, Figure)
  ```

  Replace with:
  ```python
  def test_original_scale_true_is_default(self, simple_plots):
      # Default (original_scale=True) should differ from original_scale=False
      _, axes_default = simple_plots.saturation_scatterplot()
      _, axes_false = simple_plots.saturation_scatterplot(original_scale=False)
      offsets_default = axes_default.flat[0].collections[0].get_offsets()
      offsets_false = axes_false.flat[0].collections[0].get_offsets()
      assert not np.allclose(offsets_default[:, 1], offsets_false[:, 1])
  ```

- [ ] **Step 2: Upgrade `test_sample_curves_drawn` in `TestSaturationCurvesSamples`**

  Find (around line 428):
  ```python
  def test_sample_curves_drawn(self, simple_plots, simple_curve):
      """Sample curves should add Line2D objects to axes."""
      _, axes = simple_plots.saturation_curves(curves=simple_curve, n_samples=5)
      for ax in axes.flat:
          lines = ax.get_lines()
          assert len(lines) >= 5, "At least n_samples lines per panel"
  ```

  Replace with:
  ```python
  def test_sample_curves_drawn(self, simple_plots, simple_curve):
      """Sample curves add exactly n_samples + 1 Line2D objects (samples + mean)."""
      _, axes = simple_plots.saturation_curves(curves=simple_curve, n_samples=5)
      for ax in axes.flat:
          assert len(ax.get_lines()) == 6  # 5 samples + 1 mean
  ```

- [ ] **Step 3: Upgrade `test_sample_curves_drawn` in `TestSaturationCurvesWithSampleDim`**

  Find (around line 706):
  ```python
  def test_sample_curves_drawn(self, simple_plots, simple_sample_curve):
      _, axes = simple_plots.saturation_curves(
          curves=simple_sample_curve, n_samples=5
      )
      for ax in axes.flat:
          lines = ax.get_lines()
          assert len(lines) >= 5
  ```

  Replace with:
  ```python
  def test_sample_curves_drawn(self, simple_plots, simple_sample_curve):
      _, axes = simple_plots.saturation_curves(
          curves=simple_sample_curve, n_samples=5
      )
      for ax in axes.flat:
          assert len(ax.get_lines()) == 6  # 5 samples + 1 mean
  ```

- [ ] **Step 4: Upgrade `test_original_scale_false` in `TestSaturationCurvesWithSampleDim`**

  Find (around line 734):
  ```python
  def test_original_scale_false(self, simple_plots, simple_sample_curve_scaled):
      fig, _axes = simple_plots.saturation_curves(
          curves=simple_sample_curve_scaled, original_scale=False
      )
      assert isinstance(fig, Figure)
  ```

  Replace with:
  ```python
  def test_original_scale_false(self, simple_plots, simple_sample_curve_scaled):
      # Scaled curves with original_scale=False should produce no UserWarning
      with warnings.catch_warnings():
          warnings.simplefilter("error")
          simple_plots.saturation_curves(
              curves=simple_sample_curve_scaled, original_scale=False
          )
  ```

- [ ] **Step 5: Remove `TestSaturationScatterplotCostPerUnit`**

  Delete the entire class (around lines 282-289):
  ```python
  class TestSaturationScatterplotCostPerUnit:
      def test_apply_cost_per_unit_true(self, panel_plots):
          fig, _axes = panel_plots.saturation_scatterplot(apply_cost_per_unit=True)
          assert isinstance(fig, Figure)

      def test_apply_cost_per_unit_false(self, panel_plots):
          fig, _axes = panel_plots.saturation_scatterplot(apply_cost_per_unit=False)
          assert isinstance(fig, Figure)
  ```

  > `TestSaturationScatterplotLabels` already covers both cases with richer assertions (`"Spend" in ax.get_xlabel()` and `"Channel Data" in ax.get_xlabel()`).

- [ ] **Step 6: Run all tests to verify no regressions**

  ```bash
  conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_transformations.py -v
  ```

  Expected: All tests PASS. Total count should be higher by ~15 net (new tests added minus 2 removed).

- [ ] **Step 7: Commit**

  ```bash
  git add tests/mmm/plotting/test_transformations.py
  git commit -m "test: strengthen weak assertions and remove redundant cost-per-unit tests"
  ```

---

## Final verification

- [ ] **Run the full test file one more time**

  ```bash
  conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_transformations.py -v --tb=short
  ```

  Expected: All tests PASS, no warnings about unraisable exceptions.

- [ ] **Run pre-commit**

  ```bash
  conda run -n pymc-marketing-dev pre-commit run --files tests/mmm/plotting/test_transformations.py
  ```

  Expected: All hooks pass (ruff format, ruff check, mypy if applicable).
