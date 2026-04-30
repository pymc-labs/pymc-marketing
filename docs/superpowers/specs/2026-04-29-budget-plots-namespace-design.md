# Design: PR 6 — BudgetPlots Namespace

**Date:** 2026-04-29
**Branch:** `isofer/2525-migrate-budget-allocation-plots`
**Context:** Part of the MMMPlotSuite v2 migration. Extracts budget-related plots from the monolithic `MMMPlotSuite` into a focused `BudgetPlots` namespace class, following the same pattern as `SensitivityPlots`, `DecompositionPlots`, etc.

---

## Goal

Add a `BudgetPlots` namespace class accessible as `mmm.plot.budget`, with two methods:

- `mmm.plot.budget.allocation_roas(samples)` — forest plot of per-channel ROAS distributions, with a vertical reference line at 1 (break-even).
- `mmm.plot.budget.contribution_over_time(samples)` — time series of channel contributions from an optimized budget allocation.

The existing `budget_allocation_roas` and `allocated_contribution_by_channel_over_time` methods on `MMMPlotSuite` serve as the starting point for logic, but both are replaced/upgraded.

---

## Files Changed

| File | Action |
|---|---|
| `pymc_marketing/mmm/plotting/budget.py` | **New** — `BudgetPlots` class |
| `pymc_marketing/mmm/plotting/_helpers.py` | Add `_plot_timeseries_channel` shared helper |
| `pymc_marketing/mmm/plotting/decomposition.py` | Refactor `contributions_over_time` to use shared helper |
| `pymc_marketing/mmm/plotting/__init__.py` | Add `BudgetPlots` import and export |
| `pymc_marketing/mmm/plot.py` | Add `budget` property to `MMMPlotSuite` |
| `tests/mmm/plotting/test_budget.py` | **New** — unit tests |

---

## `BudgetPlots` Class

### Constructor

```python
class BudgetPlots:
    # No __init__ — stateless, all data supplied via the samples argument per method
```

`mmm.plot.budget` is a `@property` on `MMMPlotSuite` that simply returns `BudgetPlots()`.

### `allocation_roas`

**Signature:**
```python
def allocation_roas(
    self,
    samples: xr.Dataset,
    dims: dict[str, Any] | None = None,
    hdi_prob: float = 0.94,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
```

**Logic:**
1. Validate: `channel_contribution_original_scale` and `allocation` in `samples`; `channel` dimension present.
2. Compute ROAS:
   ```python
   roas_da = samples["channel_contribution_original_scale"].sum("date") / samples["allocation"]
   roas_da = roas_da.rename("roas")
   ```
3. Apply optional dim filtering: `_select_dims(roas_da, dims)`.
4. Normalize sample dims: `_ensure_chain_draw_dims(roas_da)` → `(chain, draw, channel, ...)`.
5. Plot: `azp.plot_forest(roas_da.to_dataset(), ci_kind="hdi", ci_probs=(0.5, hdi_prob), backend=backend, **pc_kwargs)`.
6. Add break-even reference: `azp.add_lines(pc, 1.0, orientation="vertical")`.
7. Return: `_extract_matplotlib_result(pc, return_as_pc)`.

**What the plot shows:**
One row per channel; x-axis is ROAS; thick bar = 50% HDI, thin bar = `hdi_prob` HDI; point = median; vertical line at x=1 marks break-even (ROAS < 1 means money-losing channel at this allocation).

### `contribution_over_time`

**Signature:**
```python
def contribution_over_time(
    self,
    samples: xr.Dataset,
    dims: dict[str, Any] | None = None,
    hdi_prob: float = 0.85,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    return_as_pc: bool = False,
    line_kwargs: dict[str, Any] | None = None,
    hdi_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
```

**Logic** (migrated from `MMMPlotSuite.allocated_contribution_by_channel_over_time`, plus standard params):
1. Validate: `channel`, `date`, `sample` dimensions in `samples`; at least one variable containing `"channel_contribution"` present.
2. Find contribution variable: first var whose name contains `"channel_contribution"`.
3. Apply optional dim filtering via `_select_dims`.
4. Compute extra dims (all except `channel`, `date`, `sample`) for panel faceting.
5. Delegate rendering to shared helper `_plot_timeseries_channel` (see below).
6. Return: `_extract_matplotlib_result(pc, return_as_pc)`.

---

## Shared Helper: `_plot_timeseries_channel` (in `_helpers.py`)

Extracts the identical rendering loop used by both `BudgetPlots.contribution_over_time`
and `DecompositionPlots.contributions_over_time`.

```python
def _plot_timeseries_channel(
    ds: xr.Dataset,
    sample_dims: list[str],
    color_dim: str,
    extra_dims: list[str],
    hdi_prob: float,
    backend: str | None,
    line_kwargs: dict | None,
    hdi_kwargs: dict | None,
    **pc_kwargs,
) -> PlotCollection:
    """Render a time-series Dataset as one line+HDI band per `color_dim` value.

    Parameters
    ----------
    ds : xr.Dataset
        Data with dims (sample_dims..., date, color_dim[, extra_dims...]).
    sample_dims : list[str]
        Dimensions to reduce for HDI/mean (e.g. ["chain", "draw"] or ["sample"]).
    color_dim : str
        Dimension mapped to colour aesthetic (e.g. "channel" or "component").
    extra_dims : list[str]
        Additional dimensions used to create facet panels (e.g. ["geo"]).
    hdi_prob : float
        HDI probability mass.
    backend : str or None
        Rendering backend.
    line_kwargs, hdi_kwargs : dict or None
        Extra kwargs forwarded to line/HDI visuals.
    **pc_kwargs
        Forwarded to PlotCollection.wrap().
    """
```

**Callers:**

| Caller | `sample_dims` | `color_dim` |
|---|---|---|
| `DecompositionPlots.contributions_over_time` | `["chain", "draw"]` | `"component"` |
| `BudgetPlots.contribution_over_time` | `["sample"]` | `"channel"` |

`DecompositionPlots.contributions_over_time` is refactored to call this helper instead of
inlining the rendering loop. No change in its public API or behaviour.

---

## Standard Parameters (both methods)

Consistent with other namespace classes (`SensitivityPlots`, etc.):

| Param | Type | Default | Meaning |
|---|---|---|---|
| `dims` | `dict \| None` | `None` | Dimension filters via `_select_dims` |
| `hdi_prob` | `float` | 0.94 / 0.85 | Credible interval probability |
| `figsize` | `tuple \| None` | `None` | Injected into `figure_kwargs` via `_process_plot_params` |
| `backend` | `str \| None` | `None` | Rendering backend |
| `return_as_pc` | `bool` | `False` | Return `PlotCollection` instead of `(Figure, Axes)` |
| `**pc_kwargs` | | | Forwarded to `PlotCollection` constructor |

`_process_plot_params` enforces that non-matplotlib backends require `return_as_pc=True`.

---

## Wiring: `MMMPlotSuite.budget` property

```python
@property
def budget(self) -> BudgetPlots:
    """Access budget allocation plots."""
    from pymc_marketing.mmm.plotting.budget import BudgetPlots
    return BudgetPlots()
```

---

## `samples` Input Format

Both methods expect an `xr.Dataset` as returned by `mmm.allocate_budget_to_maximize_response(...)` or equivalent:

- `allocation_roas` requires: `channel_contribution_original_scale` (dims: `sample`|`chain/draw`, `date`, `channel`, ...) and `allocation` (dims: `channel`, ...).
- `contribution_over_time` requires: a variable containing `"channel_contribution"` (dims: `sample`, `date`, `channel`, ...).

---

## Out of Scope

- `dims_to_group_by` — not included (removed from the original `budget_allocation_roas` design).
- `mmm.plot.budget.allocation` — not implemented (replaced entirely by `allocation_roas`).
- Tests for `MMMPlotSuite.budget` property — covered by the unit tests for `BudgetPlots` directly.
