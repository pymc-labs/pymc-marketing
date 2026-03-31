# DecompositionPlots Fix — Design Spec

**Date:** 2026-03-31
**Branch:** `isofer/decomposition-namespace`
**Files affected:**
- `pymc_marketing/mmm/plotting/decomposition.py`
- `tests/mmm/plotting/test_decomposition.py`

---

## Problem Summary

Three bugs in `DecompositionPlots`:

1. **`intercept_contribution` has no `date` dim** but current code assumes it does (it tries to sum over `date` and passes it to the time-series rendering loop raw). Fixtures incorrectly give it a `date` dim.
2. **`contributions_over_time` sums over `channel`** before plotting, collapsing all channels into one line. Each channel should appear as its own line.
3. **`waterfall` sums over `channel` and `control`** before plotting, showing only aggregated bars. Each channel and each control should be its own bar.

---

## Fixture Changes

### `simple_idata` (no extra dims)

| Variable | Dims |
|---|---|
| `channel_contribution` | `(chain, draw, date, channel)` |
| `intercept_contribution` | `(chain, draw)` — **no `date`** |

No controls or seasonality in the simple fixture.

### `panel_idata` (geo extra dim)

| Variable | Dims |
|---|---|
| `channel_contribution` | `(chain, draw, date, geo, channel)` |
| `intercept_contribution` | `(chain, draw, geo)` — **no `date`** |
| `control_contribution` | `(chain, draw, date, geo, control)` |
| `yearly_seasonality_contribution` | `(chain, draw, date, geo)` |

---

## `contributions_over_time` — Approach

Replace the `reduced` dict + generic `.sum(to_sum)` loop with a flat `entries: list[tuple[str, xr.DataArray]]` pre-processing block. Each entry's DataArray has dims `(chain, draw, date[, extra_dims])` so the existing rendering loop is completely unchanged.

### Expansion rules

| Component | Expansion |
|---|---|
| `channels` | `[(ch, da.sel(channel=ch)) for ch in da.coords["channel"].values]` |
| `baseline` | `[("baseline", da.expand_dims({"date": data.dates}))]` — broadcasts constant over dates |
| `controls` | `[("controls", da.sum(dim="control"))]` — controls are aggregated |
| `seasonality` | `[("seasonality", da)]` — unchanged |

`dates` is taken from `data.dates` (not derived from `contributions_ds`).

After `_select_dims`, the existing rendering loop iterates over `entries` with color `C{i}` and `label` = entry name. The baseline produces a flat line and constant HDI band (correct — it is time-invariant).

---

## `waterfall` — Approach

### Pre-processing: `entries` list

Replace `ordered_keys` / `means` dict with `entries: list[tuple[str, xr.DataArray]]` where each DataArray is already reduced to a scalar (or `(extra_dims,)` array for panel models).

| Component | Reduction | Order |
|---|---|---|
| `baseline` | `da.mean(dim=("chain", "draw"))` — **no `"date"`** | first |
| `channels` | `for ch → da.sel(channel=ch).mean(dim=("chain","draw","date"))` | second, one per channel |
| `controls` | `for ctrl → da.sel(control=ctrl).mean(dim=("chain","draw","date"))` | third, one per control |
| `seasonality` | `da.mean(dim=("chain","draw","date"))` | last |

### Extract `_plot_waterfall_panel`

Move the single-panel rendering into a private helper:

```python
def _plot_waterfall_panel(
    ax: Axes,
    entries: list[tuple[str, float]],   # (label, value) already scalar-selected
    bar_kwargs: dict,
) -> None:
```

The outer `waterfall` method handles: data preparation, `entries` construction, `_select_dims`, subplot creation, panel iteration (extracts scalar float per entry for each combo), and calls `_plot_waterfall_panel`.

---

## Test Changes

### Fixture updates

- `simple_idata`: remove `date` from `intercept_contribution` dims; remove controls and seasonality.
- `panel_idata`: remove `date` from `intercept_contribution` dims; add `control_contribution` and `yearly_seasonality_contribution` with correct dims.

### New / updated tests

**`TestContributionsOverTime`:**
- `test_each_channel_has_own_line` — assert number of lines with channel-name labels equals `n_channels`
- `test_baseline_is_horizontal` — assert all y-values of the baseline line are equal (constant across dates)
- Update `test_no_summing_warning` comment: channels are no longer summed; still no warning emitted

**`TestWaterfall`:**
- `test_bars_include_all_channels_and_controls` (panel fixture) — assert all channel names and control names appear in ytick labels
- `test_baseline_bar_present` — assert "baseline" appears in ytick labels

---

## Out of Scope

- `channel_share_hdi` — not affected by these bugs; no changes.
- Seasonality handling in `contributions_over_time` — already correct (single DataArray, no dim to expand over).
- Adding `yearly_seasonality_contribution` to `MMMIDataWrapper.get_contributions` — already handled; `seasonality` key is populated if the variable is present.
