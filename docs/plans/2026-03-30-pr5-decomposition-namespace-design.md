# DecompositionPlots Namespace — Design Spec (PR 5)

**Date:** 2026-03-30
**Branch:** `isofer/decomposition-namespace`
**Context:** Part of the MMMPlotSuite v2 rewrite. PRs 1–4 are already merged on this branch.

---

## Overview

PR 5 adds `pymc_marketing/mmm/plotting/decomposition.py`, exposing a `DecompositionPlots` class
mounted as `mmm.plot.decomposition`. Three methods are ported and modernised from the old
`MMMPlotSuite` in `plot.py`:

| New method | Old method |
|---|---|
| `contributions_over_time` | `contributions_over_time` |
| `waterfall` | `waterfall_components_decomposition` |
| `channel_share_hdi` | `channel_contribution_share_hdi` |

Critical fixes addressed in this PR:

| Issue | Fix |
|---|---|
| IV.2 | No `plt.gcf()` — all figure references come from PlotCollection or `az.plot_forest` return value |
| IV.8 | Waterfall uses positional indexing, not coordinate-as-index |
| IV.9 | kwargs conflict detection when merging dicts |
| IV.12 | Warn when `_reduce_and_stack`-equivalent sums over unexpected dimensions |
| IV.14 | Multi-axes return from `az.plot_forest` collected into `NDArray[Axes]` |
| II.5 | `var_names` instead of `var`; updated parameter types |
| VII.1 | Rename coordinate from `x` to `channel` in `channel_share_hdi` |

---

## File Structure

```
pymc_marketing/mmm/plotting/
└── decomposition.py   ← new file (this PR)
```

No other files are added. `suite.py` wires `DecompositionPlots` onto `MMMPlotSuite` (already
stubbed from PR 1).

---

## Class Skeleton

```python
class DecompositionPlots:
    def __init__(self, data: MMMIDataWrapper) -> None: ...
    def contributions_over_time(...) -> tuple[Figure, NDArray[Axes]] | PlotCollection: ...
    def waterfall(...) -> tuple[Figure, NDArray[Axes]]: ...
    def channel_share_hdi(...) -> tuple[Figure, NDArray[Axes]] | PlotCollection: ...
```

No shared private helpers beyond what is already in `_helpers.py`. Each method is
self-contained. Follows the same structure as `DiagnosticsPlots` (PR 2 template).

---

## Method 1: `contributions_over_time`

### Purpose

Plot time-series contributions for selected contribution types, with HDI uncertainty bands.
One panel per extra-dimension combination (e.g. one per geo). Multiple colored lines per panel
— one per contribution type.

### Signature

```python
def contributions_over_time(
    self,
    include: list[Literal["channels", "baseline", "controls", "seasonality"]] | None = None,
    hdi_prob: float = 0.94,
    original_scale: bool = True,
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    line_kwargs: dict[str, Any] | None = None,
    hdi_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection
```

`include=None` means all available contribution types.

`col_wrap` defaults to `1` but is user-overridable via `**pc_kwargs`
(`pc_kwargs.setdefault("col_wrap", 1)`).

### Data Flow

1. Resolve `data`:
   ```python
   data = MMMIDataWrapper(idata, schema=self._data.schema) if idata is not None else self._data
   ```
2. Call `data.get_contributions(original_scale=original_scale)` — returns
   `xr.Dataset` with variables from `{"channels", "baseline", "controls", "seasonality"}`
   (only keys present in the model are included).
   Filter to the keys in `include` (if provided).
3. For each DataArray (`(chain, draw, date[, channel/control/...][, extra_dims])`):
   - Identify any dimension that is not `date`, `chain`, `draw`, or a known extra-dim
     (e.g., `channel` inside the `"channels"` entry).
   - Sum over those dimensions to collapse them to `(chain, draw, date[, extra_dims])`.
   - Emit `UserWarning` when summing over an unexpected dimension (fix IV.12).
4. Apply `_select_dims` to each DataArray.
5. Compute `mean_da` and `hdi_da` (via `.azstats.hdi(hdi_prob)`) for each.
6. Build layout dataset from the first DataArray at `isel(date=0, drop=True)`.
7. `pc_kwargs.setdefault("col_wrap", 1)`
8. `pc = PlotCollection.wrap(layout_ds, cols=extra_dims, backend=backend, **pc_kwargs_processed)`
9. For each `(i, label, mean_da, hdi_da)`:
   - `pc.map(azp.visuals.fill_between_y, ..., color=f"C{i}", alpha=0.2, **(hdi_kwargs or {}))`
   - `pc.map(azp.visuals.line_xy, ..., color=f"C{i}", label=label, **(line_kwargs or {}))`
10. Add axis labels + `labelled_title(subset_info=True)`.
11. Return via `_extract_matplotlib_result(pc, return_as_pc)`.

---

## Method 2: `waterfall`

### Purpose

Horizontal waterfall chart showing mean contribution of each component (baseline, channels,
controls, seasonality) to the total target. Pure matplotlib — no PlotCollection, no `backend`,
no `return_as_pc`.

### Signature

```python
def waterfall(
    self,
    original_scale: bool = True,
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    bar_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure, NDArray[Axes]]
```

### Data Flow

1. Resolve `data` via idata override.
2. Call `data.get_contributions(original_scale=original_scale)` → `xr.Dataset`.
3. For each DataArray: `.mean(dim=("chain", "draw", "date"))` → scalar per extra-dim combination.
   Apply `_select_dims`.
4. Determine subplot layout: one subplot per extra-dim combination, enumerated via
   `itertools.product` over extra-dim coordinate values (single axes if no extra dims).
   Use `plt.subplots(nrows, ncols, figsize=figsize)` — no `plt.gcf()` (fix IV.2).
5. For each subplot:
   - Compute cumulative running total to position bars correctly.
   - Use positional indexing, not coordinate-as-index (fix IV.8).
   - Colors: green for positive contributions, red for negative, grey for total bar.
   - Labels: value + percentage of total, rendered inside/beside each bar.
6. Merge `bar_kwargs` safely (conflict detection per IV.9).
7. Return `(fig, np.atleast_1d(axes))`.

---

## Method 3: `channel_share_hdi`

### Purpose

Forest plot showing each channel's share of total channel contribution (as a percentage HDI).
Follows the full standard pattern — supports `backend` and `return_as_pc` via `azp.plot_forest`.

### Signature

```python
def channel_share_hdi(
    self,
    hdi_prob: float = 0.94,
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection
```

### Data Flow

1. Resolve `data` via idata override.
2. `channel_contributions = data.get_channel_contributions(original_scale=True)`
   → dims `(chain, draw, date, channel[, extra_dims])`.
3. Apply `_select_dims`.
4. Sum over `date` → `(chain, draw, channel[, extra_dims])`.
5. Divide by total (sum over `channel`) to compute shares. Rename coordinate from `x` to
   `channel` (fix VII.1).
6. Process standard params: `_process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)`.
7. Call `azp.plot_forest(share_ds, hdi_prob=hdi_prob, backend=backend, **pc_kwargs_processed)`.
8. Return via `_extract_matplotlib_result(pc, return_as_pc)`.

---

## Testing

Follows the same test structure as `tests/mmm/plotting/test_diagnostics.py`:

- One test class per method: `TestContributionsOverTime`, `TestWaterfall`, `TestChannelShareHdi`.
- Fixtures: minimal `MMMIDataWrapper` with fake `idata` (posterior samples, constant_data).
- Smoke test: default call returns `(Figure, NDArray[Axes])`.
- `return_as_pc=True` returns `PlotCollection` (for `contributions_over_time` and `channel_share_hdi`).
- `include` filtering works correctly.
- `idata` override resolves to the override, not `self._data`.
- `dims` subsetting reduces panels.
- `waterfall`: single panel when no extra dims; multiple panels for geo-segmented model.
- `channel_share_hdi`: coordinate is named `channel` not `x`.
