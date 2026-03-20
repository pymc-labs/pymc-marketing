# Test Audit Design — `test_helpers.py` & `test_transformations.py`

> Incremental audit of existing tests against the current production code
> and the [13 implementation amendments](./2026-03-20-pr1-pr4-implementation-amendments.md).
>
> Approach: keep all passing tests, add targeted tests for identified gaps.
>
> Prepared 2026-03-20.

---

## `test_helpers.py` — No Changes

All six helper functions have thorough coverage. The amendments that
affected helpers (#1 removing `plot_collection`, #14 simplified
`_process_plot_params` signature) are already reflected in the tests.

---

## `test_transformations.py` — Gap Analysis & Additions

### Existing coverage retained (no changes)

| Test Class | # Tests | Covers |
|---|---|---|
| `TestSaturationScatterplotBasic` | 5 | Return types, default scale, axes count, scatter data |
| `TestSaturationScatterplotDims` | 4 | Single/list/channel dims, invalid value |
| `TestSaturationScatterplotCustomization` | 3 | `return_as_pc`, `figsize`, non-mpl backend error |
| `TestSaturationScatterplotIdataOverride` | 2 | Override uses different data, doesn't mutate self |
| `TestSaturationScatterplotLabels` | 3 | ylabel, xlabel with/without `cost_per_unit` |
| `TestSaturationScatterplotCostPerUnit` | 2 | True/false smoke tests |
| `TestSaturationCurvesBasic` | 4 | Return types, default scale, axes count |
| `TestSaturationCurvesHDI` | 2 | HDI band present, custom prob |
| `TestSaturationCurvesSamples` | 3 | Lines drawn, `n_samples=0`, reproducibility |
| `TestSaturationCurvesDims` | 2 | Single/list dim value |
| `TestSaturationCurvesCustomization` | 2 | `return_as_pc`, `figsize` |
| `TestSaturationCurvesIdataOverride` | 1 | Override axes count |
| `TestSaturationCurvesLabels` | 2 | ylabel, xlabel |
| `TestSaturationCurvesScaleWarning` | 4 | Warn/no-warn for both scale combos |
| `TestEnsureChainDrawDims` | 4 | chain/draw, sample, values preserved, unknown dims |
| `TestSaturationCurvesWithSampleDim` | 8 | Full sample-dim path coverage |

### Gap 1: `*_kwargs` forwarding (Amendment #2)

Per-element kwargs are the core API change from the amendments but have
zero test coverage. Add to existing customization classes.

**`TestSaturationScatterplotCustomization`** — add:

- `test_scatter_kwargs_forwarded`: pass `scatter_kwargs={"alpha": 0.5}`,
  verify scatter collection alpha is 0.5 (not default 0.8).

**New class `TestSaturationCurvesKwargsForwarding`** — add:

- `test_scatter_kwargs_forwarded`: pass `scatter_kwargs={"alpha": 0.5}`,
  verify scatter alpha.
- `test_hdi_kwargs_forwarded`: pass `hdi_kwargs={"alpha": 0.5}`, verify
  fill collection alpha.
- `test_mean_curve_kwargs_forwarded`: pass
  `mean_curve_kwargs={"linewidth": 3}`, verify line width on the mean
  curve (the first solid-alpha line).
- `test_sample_curves_kwargs_forwarded`: pass
  `sample_curves_kwargs={"alpha": 0.1}`, verify translucent lines have
  alpha 0.1 (not default 0.3).

### Gap 2: `hdi_prob=None` disables HDI band

Code has `if hdi_prob is not None:` but no test for the `None` case.

**`TestSaturationCurvesHDI`** — add:

- `test_hdi_prob_none_disables_band`: pass `hdi_prob=None, n_samples=0`,
  verify no PolyCollection in axes.

### Gap 3: Mean curve always present (Amendment #13)

Only `test_n_samples_zero_draws_only_mean_line` covers mean presence
implicitly. No test that the mean exists *alongside* samples.

**`TestSaturationCurvesSamples`** — add:

- `test_mean_curve_present_with_samples`: with `n_samples=5`, verify
  `len(lines) >= n_samples + 1` (mean + samples).

### Gap 4: `_ensure_chain_draw_dims` — MultiIndex sample path (Amendment #8)

Three input formats exist; only two are tested. The MultiIndex path
(chain/draw as non-dim coords on `sample`) is uncovered.

**`TestEnsureChainDrawDims`** — add:

- `test_multiindex_sample_unstacked`: create a `(chain, draw, channel, x)`
  DataArray, `.stack(sample=("chain", "draw"))` it, pass to
  `_ensure_chain_draw_dims`, verify it unstacks back to original shape
  with `chain` and `draw` dims.

### Gap 5: `_x_axis_label` and `_get_channel_x_data` unit tests

Tested implicitly via label assertions but no direct unit tests.

**New class `TestXAxisLabel`** — add:

- `test_returns_spend_when_cost_per_unit_available`: mock or use
  `panel_data` (has `channel_spend`), verify returns `"Spend"`.
- `test_returns_channel_data_when_no_cost_per_unit`: use `simple_data`
  (no `channel_spend`), verify returns `"Channel Data"`.
- `test_returns_channel_data_when_flag_false`: even with spend available,
  `apply_cost_per_unit=False` returns `"Channel Data"`.

**New class `TestGetChannelXData`** — add:

- `test_returns_spend_when_flag_true`: with `panel_data`, verify returns
  same as `data.get_channel_spend()`.
- `test_returns_raw_data_when_flag_false`: verify returns same as
  `data.get_channel_data()`.

### Gap 6: `**pc_kwargs` forwarding

No test that extra kwargs reach `PlotCollection`.

**`TestSaturationScatterplotCustomization`** — add:

- `test_pc_kwargs_forwarded`: pass an extra kwarg (e.g.,
  `plot_kwargs={"some_key": True}`) and verify it doesn't raise. This is
  a smoke test — deep PlotCollection internals are hard to assert on.

---

## Summary of changes

| File | Action | Net new tests |
|---|---|---|
| `test_helpers.py` | None | 0 |
| `test_transformations.py` | Add tests to existing classes + 2 new classes | ~14 |
