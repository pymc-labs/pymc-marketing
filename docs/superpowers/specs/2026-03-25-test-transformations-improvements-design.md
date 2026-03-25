# Test Improvements for `test_transformations.py`

**Date:** 2026-03-25
**File under review:** `tests/mmm/plotting/test_transformations.py`
**Implementation file:** `pymc_marketing/mmm/plotting/transformations.py`

---

## Context

`TransformationPlots` exposes two public methods:

- `saturation_scatterplot()` — scatter plot of channel spend/data vs mean contributions
- `saturation_curves()` — overlays saturation curves (HDI band + sample lines + mean) on the scatter

The existing test suite covers return types, axis counts, dim filtering, scale-mismatch warnings, and `_ensure_chain_draw_dims`. The goal of this spec is to (a) fill coverage gaps, (b) verify behavioral correctness via matplotlib artist inspection, and (c) strengthen weak assertions.

**Chosen approach:** Matplotlib artist inspection (Approach B) — extract rendered data from `PathCollection`, `Line2D`, and `PolyCollection` artists and compare against values computed directly from fixture data. No mocking of internals, no snapshot images.

---

## Section 1: Coverage Gaps

Seven behaviors are currently untested. Each maps to a specific test location.

### 1.1 Invalid dim key raises `ValueError` for `saturation_scatterplot`

**Where:** Add to `TestSaturationScatterplotDims`

```python
def test_invalid_dim_key_raises(self, panel_plots):
    with pytest.raises(ValueError, match="Dimension 'bad_dim' not found"):
        panel_plots.saturation_scatterplot(dims={"bad_dim": "US"})
```

### 1.2 Invalid dim key raises `ValueError` for `saturation_curves`

**Where:** New class `TestSaturationCurvesDimsInvalid`

```python
def test_invalid_dim_key_raises(self, panel_plots, panel_curve):
    with pytest.raises(ValueError, match="Dimension 'bad_dim' not found"):
        panel_plots.saturation_curves(curves=panel_curve, dims={"bad_dim": "US"})
```

### 1.3 Invalid dim value raises `ValueError` for `saturation_curves`

**Where:** `TestSaturationCurvesDimsInvalid`

```python
def test_invalid_dim_value_raises(self, panel_plots, panel_curve):
    with pytest.raises(ValueError, match="Value 'FR' not found"):
        panel_plots.saturation_curves(curves=panel_curve, dims={"country": "FR"})
```

### 1.4 `hdi_prob=None` disables HDI band

**Where:** New class `TestSaturationCurvesEdgeCases`

```python
def test_hdi_prob_none_disables_band(self, simple_plots, simple_curve):
    _, axes = simple_plots.saturation_curves(
        curves=simple_curve, hdi_prob=None, n_samples=0
    )
    for ax in axes.flat:
        polys = [c for c in ax.collections if "Poly" in type(c).__name__]
        assert len(polys) == 0, "No HDI fill when hdi_prob=None"
```

### 1.5 `n_samples` clamped when larger than available draws

**Where:** `TestSaturationCurvesEdgeCases`

The fixture `simple_curve` has 2 chains × 50 draws = 100 stacked samples. Requesting 10 000 samples should clamp to 100.

```python
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

### 1.6 `figsize` + `figure_kwargs['figsize']` conflict emits `UserWarning`

**Where:** `TestSaturationScatterplotCustomization`

```python
def test_figsize_overrides_figure_kwargs_warns(self, simple_plots):
    with pytest.warns(UserWarning, match="figsize parameter overrides"):
        simple_plots.saturation_scatterplot(
            figsize=(10, 5),
            figure_kwargs={"figsize": (8, 4)},
        )
```

### 1.7 `saturation_curves` idata override does not mutate `self._data`

**Where:** `TestSaturationCurvesIdataOverride`

```python
def test_idata_override_does_not_mutate_self_data(
    self, simple_plots, simple_data, panel_idata, panel_curve
):
    simple_plots.saturation_curves(
        curves=panel_curve, idata=panel_idata, n_samples=2
    )
    assert simple_plots._data is simple_data
```

---

## Section 2: Behavioral Verification via Matplotlib Artist Inspection

### 2a. Scatter data correctness — new class `TestSaturationScatterplotDataValues`

#### `test_scatter_y_matches_mean_contributions`

Extract y-offsets from each panel's `PathCollection` and compare to the mean of `channel_contribution_original_scale` across (chain, draw).

```python
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
```

#### `test_scatter_x_matches_channel_data`

```python
def test_scatter_x_matches_channel_data(self, simple_plots, simple_data):
    _, axes = simple_plots.saturation_scatterplot(apply_cost_per_unit=False)
    channels = simple_data.channels
    x_data = simple_data.get_channel_data()

    for ax, ch in zip(axes.flat, channels):
        offsets = ax.collections[0].get_offsets()
        np.testing.assert_allclose(
            offsets[:, 0], x_data.sel(channel=ch).values, rtol=1e-5
        )
```

#### `test_original_scale_changes_scatter_y`

Verify `original_scale=True` and `original_scale=False` produce different y-offsets (the fixture has `channel_contribution` and `channel_contribution_original_scale` with different scales).

```python
def test_original_scale_changes_scatter_y(self, simple_plots):
    _, axes_true = simple_plots.saturation_scatterplot(original_scale=True)
    _, axes_false = simple_plots.saturation_scatterplot(original_scale=False)
    offsets_true = axes_true.flat[0].collections[0].get_offsets()
    offsets_false = axes_false.flat[0].collections[0].get_offsets()
    assert not np.allclose(offsets_true[:, 1], offsets_false[:, 1])
```

### 2b. Saturation curves — mean line and HDI — new class `TestSaturationCurvesDataValues`

#### `test_mean_curve_y_matches_curves_mean`

With `n_samples=0, hdi_prob=None`, the single `Line2D` per panel should have y-data matching `curves.mean(["chain","draw"])`.

`simple_curve` already has `(chain, draw, channel, x)` dims so no conversion is needed.
Note: `saturation_curves` replaces `curves["x"]` with `range(n)` internally, but this only
changes the coordinate, not the underlying data values, so `expected_mean.values` is unaffected.

```python
def test_mean_curve_y_matches_curves_mean(self, simple_plots, simple_curve):
    # simple_curve already has (chain, draw, channel, x) dims
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
```

#### `test_hdi_band_bounds_contain_mean`

For each panel, extract the y-bounds of the HDI polygon and verify the mean curve falls within them.

`matplotlib.axes.Axes.fill_between` produces a `PolyCollection` whose path has shape `(2*n+3, 2)`:
- index 0: closing/start point (= last upper value)
- indices `1..n`: lower bound left-to-right
- index `n+1`: join point (last lower repeated)
- indices `n+2..2n+1`: upper bound right-to-left
- index `2n+2`: closing point

```python
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
        y_lower = verts[1 : n + 1, 1]           # indices 1..n
        y_upper = verts[n + 2 : 2 * n + 2, 1][::-1]  # indices n+2..2n+1, reversed
        mean_vals = expected_mean.sel(channel=ch).values
        assert np.all(mean_vals >= y_lower - 1e-6)
        assert np.all(mean_vals <= y_upper + 1e-6)
```

#### `test_exact_line_count_with_samples`

n_samples + 1 lines (samples + mean), not just ≥ n_samples.

```python
def test_exact_line_count_with_samples(self, simple_plots, simple_curve):
    n = 5
    _, axes = simple_plots.saturation_curves(
        curves=simple_curve, n_samples=n, hdi_prob=None
    )
    for ax in axes.flat:
        assert len(ax.get_lines()) == n + 1
```

---

## Section 3: Weak Test Upgrades

### 3.1 `test_original_scale_true_is_default` (scatterplot)

**Current:** `assert isinstance(fig, Figure)`
**Upgraded:** verify y-offsets differ from `original_scale=False` — delegate to `test_original_scale_changes_scatter_y` (Section 2a). The `_is_default` test keeps the Figure assertion and adds a call to both scales.

### 3.2 `TestSaturationScatterplotCostPerUnit` — label assertions already covered; remove Figure-only assertions

`TestSaturationScatterplotLabels` already tests x-label content for both `apply_cost_per_unit=True` and `False`.
The two `TestSaturationScatterplotCostPerUnit` tests that only assert `isinstance(fig, Figure)` should be
removed — they add no coverage on top of the label tests.

### 3.3 `test_sample_curves_drawn` in both `TestSaturationCurvesSamples` and `TestSaturationCurvesWithSampleDim`

**Current:** `assert len(lines) >= 5`
**Upgraded:** `assert len(lines) == 6` (5 samples + 1 mean). Same fix in the sample-dim variant.

### 3.4 `test_original_scale_false` in `TestSaturationCurvesWithSampleDim`

**Current:** `assert isinstance(fig, Figure)`
**Upgraded:** assert no `UserWarning` is raised — confirming the no-warning path works for sample-dim curves in scaled space. Use the same pattern as `TestSaturationCurvesScaleWarning.test_no_warning_when_scales_match_scaled`:

```python
def test_original_scale_false(self, simple_plots, simple_sample_curve_scaled):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        simple_plots.saturation_curves(
            curves=simple_sample_curve_scaled, original_scale=False
        )
```

---

## Summary of Changes

| Category | Count |
|----------|-------|
| New test classes | 4 (`TestSaturationScatterplotDataValues`, `TestSaturationCurvesDimsInvalid`, `TestSaturationCurvesEdgeCases`, `TestSaturationCurvesDataValues`) |
| New tests added to existing classes | 4 (invalid dim key for scatter, figsize conflict warning, curves idata mutation, curves dims invalid class) |
| Weak tests upgraded / removed | 4 (original_scale default, sample line count ×2, test_original_scale_false; cost-per-unit Figure-only tests removed) |
| **Total new/changed tests** | ~20 |

No new fixtures required. All tests use existing `simple_plots`, `panel_plots`, `simple_curve`, `simple_curve_scaled`, `simple_sample_curve`, `panel_curve`, `panel_sample_curve`.
