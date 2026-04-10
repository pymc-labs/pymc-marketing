# Sensitivity Namespace — Design Spec

**PR:** 7 of the MMMPlotSuite v2 series
**Date:** 2026-04-10
**Branch:** `isofer/sensitivity-namespace`

---

## Overview

Implements `SensitivityPlots`, a namespace class that consolidates the three
sensitivity-analysis plotting methods currently in the monolithic `MMMPlotSuite`
(`sensitivity_analysis`, `uplift_curve`, `marginal_curve`) into the new
namespace architecture introduced in previous PRs.

Follows the same structural template as `DiagnosticsPlots`,
`TransformationPlots`, and `DecompositionPlots`.

---

## Files Changed

| File | Change |
|------|--------|
| `pymc_marketing/mmm/plotting/sensitivity.py` | **New** — `SensitivityPlots` class |
| `pymc_marketing/mmm/plotting/_helpers.py` | Promote `_ensure_chain_draw_dims` from `transformations.py` |
| `pymc_marketing/mmm/plotting/transformations.py` | Import `_ensure_chain_draw_dims` from `_helpers` instead of defining it locally |
| `pymc_marketing/mmm/plotting/__init__.py` | Export `SensitivityPlots` |
| `tests/mmm/plotting/test_sensitivity.py` | **New** — full test suite |

---

## Class Structure

```
SensitivityPlots
  ├── __init__(data: MMMIDataWrapper)
  ├── analysis(...)      resolves idata.sensitivity_analysis["x"]
  ├── uplift(...)        resolves idata.sensitivity_analysis["uplift_curve"]
  ├── marginal(...)      resolves idata.sensitivity_analysis["marginal_effects"]
  └── _sensitivity_plot(sa_da, data, ylabel, ...)   shared rendering helper
```

`analysis`, `uplift`, and `marginal` each:
1. Resolve `data` from `idata` override or `self._data`
2. Check `sensitivity_analysis` group exists on `data.idata`; raise `ValueError` if not
3. Extract the relevant `xr.DataArray` by key; raise `ValueError` if key absent
4. Delegate everything else to `_sensitivity_plot(sa_da, data, ...)`

`_sensitivity_plot` receives an already-resolved `sa_da` and `data` — it never
touches `idata` directly.

---

## Method Signature

All three public methods share the same signature:

```python
def analysis(
    self,
    idata: az.InferenceData | None = None,
    dims: dict[str, Any] | None = None,
    aggregation: dict[str, str | list[str]] | None = None,
    x_sweep_axis: Literal["relative", "absolute"] = "relative",
    apply_cost_per_unit: bool = True,
    hdi_prob: float = 0.94,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    line_kwargs: dict[str, Any] | None = None,
    hdi_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,   # rows, cols, col_wrap, figure_kwargs, etc.
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
```

`rows` and `cols` are passed through `**pc_kwargs` and extracted inside
`_sensitivity_plot`. This keeps the public signature lean while allowing
full PlotCollection layout control.

Per-method differences:

| Method | `idata` key | Default `ylabel` |
|--------|-------------|-----------------|
| `analysis` | `"x"` | `"Effect"` |
| `uplift` | `"uplift_curve"` | `"Uplift"` |
| `marginal` | `"marginal_effects"` | `"Marginal Effect"` |

---

## Rendering Flow (`_sensitivity_plot`)

`_sensitivity_plot` receives `sa_da: xr.DataArray` (already extracted by the
caller) and `data: MMMIDataWrapper` (already resolved). It is responsible only
for aggregation, filtering, reshaping, and rendering.

1. **Apply aggregation**
   Normalize values: `dims_list = [v] if isinstance(v, str) else list(v)`.
   Support `"sum"` and `"mean"` operations.
   Applied before any other transformation.

2. **Apply dimension filtering**
   `sa_da = _select_dims(sa_da, dims)`

3. **Reshape sample → (chain, draw)**
   Use `_ensure_chain_draw_dims(sa_da)` (promoted to `_helpers.py`).
   This handles three input formats: `(chain, draw, ...)`, MultiIndex sample,
   and plain integer sample → `(chain=1, draw=N)`.

4. **Determine faceting and hue**
   ```python
   cols = pc_kwargs.pop("cols", list(data.custom_dims))
   rows = pc_kwargs.pop("rows", [])
   hue_dims = [
       d for d in sa_da.dims
       if d not in {"chain", "draw", "sweep", *cols, *rows}
   ]
   ```
   Default: `custom_dims` → cols, `channel` → hue (multiple colored lines per panel).
   Override: pass `cols=` or `rows=` in `**pc_kwargs`.

5. **Compute sweep x-values**
   - `"relative"`: `sweep_x = sa_da.coords["sweep"]` (multipliers as-is)
   - `"absolute"`: `sweep_x = sweep_coords * channel_scale` where
     `channel_scale` is total spend (`apply_cost_per_unit=True`) or total
     raw channel data (`apply_cost_per_unit=False`), summed over `"date"`.
     Result has dims `(sweep, channel, *custom_dims)` so PlotCollection
     selects the right slice per panel/hue.

6. **Build PlotCollection**
   ```python
   sa_ds = sa_da.to_dataset(name="sensitivity")
   pc = PlotCollection.grid(
       sa_ds,
       rows=rows,
       cols=cols,
       aes={"color": hue_dims} if hue_dims else {},
       backend=backend,
       **pc_kwargs,
   )
   ```

7. **HDI band**
   ```python
   hdi_da = sa_ds.azstats.hdi(hdi_prob)
   pc.map(
       azp.visuals.fill_between_y,
       x=sweep_x,
       y_bottom=hdi_da.sel(ci_bound="lower"),
       y_top=hdi_da.sel(ci_bound="upper"),
       **{"alpha": 0.2, **(hdi_kwargs or {})},
   )
   ```

8. **Mean line**
   ```python
   mean_da = sa_ds.mean(dim=["chain", "draw"])
   pc.map(azp.visuals.line_xy, x=sweep_x, y=mean_da, **(line_kwargs or {}))
   ```

9. **Axis labels and title**
   ```python
   pc.map(azp.visuals.labelled_x, text=xlabel, ignore_aes={"color"})
   pc.map(azp.visuals.labelled_y, text=ylabel, ignore_aes={"color"})
   pc.map(azp.visuals.labelled_title, subset_info=True, ignore_aes={"color"})
   ```

10. **Legend** (only when hue dims are present)
    ```python
    if hue_dims:
        pc.add_legend(hue_dims[0])
    ```

11. **Return**
    ```python
    return _extract_matplotlib_result(pc, return_as_pc)
    ```

---

## Bug Fixes

| ID | Description | Resolution |
|----|-------------|------------|
| **II.4** | `uplift_curve` / `marginal_curve` mutate `self.idata` during execution | `_sensitivity_plot` accepts `sa_da: xr.DataArray` directly — no instance state mutation |
| **II.1** | `sensitivity_analysis` returns bare `plt.Axes` for single-panel case | Always returns `tuple[Figure, NDArray[Axes]]` or `PlotCollection` via `_extract_matplotlib_result` |
| **IV.4** | Color cycle not reset between panel iterations | PlotCollection `aes={"color": hue_dims}` manages color assignment — no manual cycling |
| **IV.5** | Local variable `title` shadows the `title` method parameter | Fresh implementation; no name conflicts |
| **IV.13** | Redundant `import warnings` inside method body | New file has clean imports at module level |

---

## Refactor: `_ensure_chain_draw_dims` Promotion

Currently defined locally in `transformations.py`. Moving it to `_helpers.py`
so both `transformations.py` and `sensitivity.py` can import it from one place.

`transformations.py` is updated with a single import line change — no logic
changes.

---

## Tests (`tests/mmm/plotting/test_sensitivity.py`)

**Fixtures:**

- `simple_sa_idata` — minimal `az.InferenceData` with a `sensitivity_analysis`
  group containing `"x"`, `"uplift_curve"`, and `"marginal_effects"` variables.
  Dims: `(sample=50, sweep=11, channel=3)`. Sweep coords: `np.linspace(0.5, 1.5, 11)`.
- `sensitivity_plots` — `SensitivityPlots` constructed from the fixture.

**Test cases:**

| Test | What it validates |
|------|------------------|
| `test_analysis_returns_tuple` | Return type is `tuple[Figure, NDArray[Axes]]` |
| `test_analysis_return_as_pc` | `return_as_pc=True` yields `PlotCollection` |
| `test_uplift_reads_correct_key` | Reads `"uplift_curve"`; raises `ValueError` when key absent |
| `test_marginal_reads_correct_key` | Reads `"marginal_effects"`; raises `ValueError` when key absent |
| `test_missing_sa_group_raises` | Clear `ValueError` when `sensitivity_analysis` group absent |
| `test_dims_filtering` | `dims={"channel": ["tv"]}` → fewer lines than unfiltered plot |
| `test_aggregation_str` | `aggregation={"sum": "channel"}` (str) → single line per panel |
| `test_aggregation_list` | `aggregation={"sum": ["channel"]}` (list) → same result as str form |
| `test_idata_override` | `idata=` builds new wrapper; `self._data` unchanged |
| `test_missing_key_raises` | `ValueError` for absent `"uplift_curve"` / `"marginal_effects"` |
| `test_x_sweep_axis_absolute` | Plotted x-values equal `sweep_coords × channel_spend` (within tolerance) |
| `test_custom_rows_cols_in_pc_kwargs` | `cols=["channel"]` → one panel per channel, not per custom dim |
| `test_mean_line_values` | Plotted y-data on mean line matches `sa_da.mean("sample")` at each sweep point |
| `test_hdi_band_values` | Fill-between y-bottom / y-top match `az.hdi(sa_da)` lower/upper bounds |
| `test_n_lines_equals_hue_cardinality` | Lines per panel equals cardinality of hue dim (3 channels → 3 lines) |
| `test_sweep_x_values_relative` | Plotted x-data matches sweep coordinate values exactly |
