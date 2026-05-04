# Figure Customization Surface Design (Issue II.7 + arviz-plots Integration)

> Resolves issue II.7 (inconsistent figure customization) from the
> [comprehensive audit](./2026-03-10-mmmplotsuite-comprehensive-issues.md)
> by adopting arviz-plots as the rendering engine and exposing its
> customization model directly.
>
> **Key change to the attack plan:** Issue I.6 (arviz-plots adoption) is
> moved from "deferred follow-up" into the major release. II.7 is solved
> organically through arviz-plots' customization model rather than by
> adding matplotlib-specific `figsize + ax` parameters.
>
> Prepared 2026-03-11.

---

## Table of Contents

- [Context](#context)
- [Decisions](#decisions)
- [Standard Customization Parameters](#standard-customization-parameters)
- [Parameter Interaction Rules](#parameter-interaction-rules)
- [Concrete Method Signatures](#concrete-method-signatures)
- [Shared Infrastructure](#shared-infrastructure)
- [arviz-plots Coverage Gaps](#arviz-plots-coverage-gaps)
- [Impact on Attack Plan](#impact-on-attack-plan)

---

## Context

### The Problem (II.7)

The current `MMMPlotSuite` has wildly inconsistent figure customization:

| Customization   | Coverage                                           |
|-----------------|----------------------------------------------------|
| `figsize`       | 14/21 methods                                      |
| `ax` parameter  | 4/21 methods                                       |
| `**kwargs`      | Varies — sometimes to `plt.subplots()`, sometimes to plot calls |
| `rc_params`     | 1/21 methods (`saturation_curves` only)            |
| `title` control | Sensitivity family only                            |
| No customization at all | 7 methods                                 |

### Why arviz-plots Changes the Answer

The original attack plan proposed adding `figsize + ax` to all methods as a
minimal consistent surface. However, the planned adoption of arviz-plots
introduces its own customization model (`PlotCollection`, `visuals`,
`aes_by_visuals`, `backend`, `**pc_kwargs`). These two models are
incompatible — `ax` doesn't exist in arviz-plots, and `figsize` is a nested
key inside `figure_kwargs`.

Rather than solving II.7 with matplotlib-native args now and breaking the API
again when arviz-plots arrives, this design **moves arviz-plots adoption into
the major release** and solves both issues in one break.

### Approaches Considered

1. **Flat — all arviz-plots params explicit (chosen):** Every method lists
   `figsize`, `plot_collection`, `backend`, `visuals`, `aes_by_visuals`, and
   `**pc_kwargs` as named keyword arguments. Matches arviz-plots' own
   function signature convention.

2. **Layered — common params explicit, advanced via pc_kwargs:** Only
   `figsize` and `backend` are named; everything else flows through
   `**pc_kwargs`. Cleaner signatures but advanced params are invisible to
   IDE autocomplete.

3. **Config object — PlotConfig dataclass:** All rendering params bundled
   into a `PlotConfig` object. Clean separation but introduces a new type
   that doesn't exist in the arviz ecosystem.

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| arviz-plots timing | **Included in major release** (un-defer I.6) | Avoids a second breaking change; one clean break |
| Customization approach | **Flat — explicit params** (Approach 1) | Matches arviz-plots conventions; maximum IDE discoverability |
| `ax` parameter | **Removed** | Doesn't exist in arviz-plots; replaced by `plot_collection` |
| `figsize` | **Kept as top-level convenience** | Most commonly used param (14/21 today); translated to `figure_kwargs` internally |
| Return type | **`tuple[Figure, NDArray[Axes]]` by default**; `PlotCollection` opt-in via `return_as_pc=True` | Backward-friendly default; power users get full arviz-plots composability |
| Bar-plot methods | **Replace with arviz-plots-compatible visuals where possible**; matplotlib fallback for `waterfall` | Minimizes methods stuck on matplotlib-only |

---

## Standard Customization Parameters

Every public method in every namespace exposes these six standard
customization parameters, always in this order, always after the
method-specific data parameters:

| # | Parameter | Type | Default | Purpose |
|---|-----------|------|---------|---------|
| 1 | `figsize` | `tuple[float, float] \| None` | `None` | Convenience for `figure_kwargs={"figsize": ...}`. Ignored when `plot_collection` is provided. |
| 2 | `plot_collection` | `PlotCollection \| None` | `None` | Plot onto an existing collection instead of creating a new one. |
| 3 | `backend` | `str \| None` | `None` | Rendering backend: `"matplotlib"`, `"plotly"`, `"bokeh"`. |
| 4 | `visuals` | `dict[str, Any] \| None` | `None` | Element-level customization. Keys are visual element names (method-specific), values are backend kwargs or `False` to disable. |
| 5 | `aes_by_visuals` | `dict[str, list[str]] \| None` | `None` | Controls which aesthetics apply to which visual elements. |
| 6 | `**pc_kwargs` | | | Forwarded to `PlotCollection.wrap()` or `.grid()`. |

Plus one control parameter:

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `return_as_pc` | `bool` | `False` | When `True`, return `PlotCollection`. When `False`, return `tuple[Figure, NDArray[Axes]]`. |

### Parameter Ordering Convention

```python
def method(
    self,
    # 1. Method-specific data params (varies per method)
    channels=None, hdi_prob=0.94, original_scale=True,
    # 2. Dimension subsetting (standard)
    dims=None,
    # 3. Figure customization (standard — identical across all methods)
    figsize=None,
    plot_collection=None,
    backend=None,
    visuals=None,
    aes_by_visuals=None,
    # 4. Return control (standard)
    return_as_pc=False,
    # 5. PlotCollection kwargs catch-all (standard)
    **pc_kwargs,
):
```

---

## Parameter Interaction Rules

1. **`plot_collection` overrides `figsize` and layout kwargs:** When
   `plot_collection` is provided, `figsize`, `backend`, and layout-related
   `pc_kwargs` are ignored (the collection already has a figure/grid).
   A warning is emitted if `figsize` is also set.

2. **`figsize` is injected into `figure_kwargs`:** When `figsize` is
   provided without `plot_collection`, it's set as
   `pc_kwargs["figure_kwargs"]["figsize"]`. If `figure_kwargs` is also in
   `pc_kwargs` with its own `figsize`, the top-level `figsize` parameter
   takes precedence (with a warning).

3. **`return_as_pc=False` requires matplotlib backend:** When
   `return_as_pc=False` and `backend` is not `"matplotlib"` (or `None`),
   raise `ValueError` — non-matplotlib backends can't produce
   `(Figure, NDArray[Axes])`.

4. **`visuals` disabling:** Setting a visual key to `False` disables that
   visual element (e.g., `visuals={"hdi_band": False}` removes the HDI
   band). Setting it to a dict passes those kwargs to the backend plotting
   function for that element.

---

## Concrete Method Signatures

### Time-Series — `diagnostics.posterior_predictive`

```python
def posterior_predictive(
    self,
    var_names: list[str] | None = None,
    hdi_prob: float = 0.94,
    original_scale: bool = True,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    plot_collection: PlotCollection | None = None,
    backend: str | None = None,
    visuals: dict[str, Any] | None = None,
    aes_by_visuals: dict[str, list[str]] | None = None,
    return_as_pc: bool = False,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
```

Visual element keys: `"line"` (median), `"hdi_band"` (HDI fill),
`"observed"` (scatter of actuals), `"title"`.

### Saturation — `saturation.curves`

```python
def curves(
    self,
    curve: xr.DataArray,
    original_scale: bool = True,
    n_samples: int = 10,
    hdi_prob: float = 0.94,
    random_seed: np.random.Generator | None = None,
    colors: Iterable[str] | None = None,
    apply_cost_per_unit: bool = True,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    plot_collection: PlotCollection | None = None,
    backend: str | None = None,
    visuals: dict[str, Any] | None = None,
    aes_by_visuals: dict[str, list[str]] | None = None,
    return_as_pc: bool = False,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
```

Visual element keys: `"samples"` (posterior curves), `"hdi_band"`,
`"median"`, `"title"`.

### Sensitivity — `sensitivity.analysis`

```python
def analysis(
    self,
    channels: list[str] | None = None,
    hdi_prob: float = 0.94,
    original_scale: bool = True,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    plot_collection: PlotCollection | None = None,
    backend: str | None = None,
    visuals: dict[str, Any] | None = None,
    aes_by_visuals: dict[str, list[str]] | None = None,
    return_as_pc: bool = False,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
```

Visual element keys: `"line"` (sensitivity curve), `"hdi_band"`,
`"reference"` (zero/baseline line), `"title"`.

### Decomposition — `decomposition.waterfall`

```python
def waterfall(
    self,
    original_scale: bool = True,
    dims: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    plot_collection: PlotCollection | None = None,
    backend: str | None = None,
    visuals: dict[str, Any] | None = None,
    aes_by_visuals: dict[str, list[str]] | None = None,
    return_as_pc: bool = False,
    **pc_kwargs,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
```

Visual element keys: `"bar"`, `"connector"` (lines between bars),
`"label"` (value text), `"title"`.

---

## Shared Infrastructure

### `_process_plot_params` — Validates and normalizes standard params

```python
# mmm/plotting/_helpers.py

def _process_plot_params(
    figsize: tuple[float, float] | None,
    plot_collection: PlotCollection | None,
    backend: str | None,
    return_as_pc: bool,
    **pc_kwargs,
) -> dict:
    if plot_collection is not None and figsize is not None:
        warnings.warn("figsize is ignored when plot_collection is provided")

    if not return_as_pc and backend is not None and backend != "matplotlib":
        raise ValueError(
            f"backend='{backend}' requires return_as_pc=True. "
            "Non-matplotlib backends cannot return (Figure, NDArray[Axes])."
        )

    if figsize is not None and plot_collection is None:
        fig_kwargs = pc_kwargs.pop("figure_kwargs", {})
        if "figsize" in fig_kwargs:
            warnings.warn(
                "figsize parameter overrides figure_kwargs['figsize']"
            )
        fig_kwargs["figsize"] = figsize
        pc_kwargs["figure_kwargs"] = fig_kwargs

    return pc_kwargs
```

### `_extract_matplotlib_result` — Converts PlotCollection to tuple

```python
def _extract_matplotlib_result(
    pc: PlotCollection,
    return_as_pc: bool,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    if return_as_pc:
        return pc

    fig = pc.viz["plot"].item().figure
    axes = np.array([ax.item() for ax in pc.viz["plot"].values()])
    return fig, axes
```

### Usage pattern in each method

```python
def analysis(self, channels=None, ..., figsize=None, plot_collection=None,
             backend=None, visuals=None, aes_by_visuals=None,
             return_as_pc=False, **pc_kwargs):

    pc_kwargs = _process_plot_params(
        figsize, plot_collection, backend, return_as_pc, **pc_kwargs
    )

    data = self._data.get_sensitivity_data(channels=channels, dims=dims)

    if plot_collection is None:
        pc = PlotCollection.wrap(data, backend=backend, **pc_kwargs)
    else:
        pc = plot_collection

    pc.map("line", line_xy, ...)
    pc.map("hdi_band", fill_between_y, ...)

    return _extract_matplotlib_result(pc, return_as_pc)
```

---

## arviz-plots Coverage Gaps

### Methods requiring bar-type visuals

arviz-plots does not currently support bar plots. Four methods are affected:

| Method | Current visual | Strategy |
|--------|---------------|----------|
| `decomposition.waterfall` | Horizontal bar chart | **Matplotlib fallback** — inherently a bar chart; no good alternative |
| `budget.allocation` | Bar chart | **Replace with dot/lollipop plot** — cleaner for comparing values with uncertainty |
| `decomposition.channel_share_hdi` | `az.plot_forest` | **Replace with arviz-plots `plot_forest`** — arviz-plots has native forest plot support |
| `decomposition.contributions_over_time` | Stacked area/bar | **Replace with stacked `fill_between_y`** — stacked area is better suited for time series |

### Fallback rules

For methods that must fall back to matplotlib:

- **Same API signature** — all 6 standard customization params are present
- `backend` other than `"matplotlib"` raises `NotImplementedError` with a
  message explaining bar plot support is pending in arviz-plots
- `plot_collection` is still accepted (for matplotlib composability)
- `visuals` keys map to matplotlib kwargs for these methods
- When arviz-plots adds bar support, the implementation switches to
  `PlotCollection` with no API change

---

## Impact on Attack Plan

### Changes to scope

| Aspect | Original plan | Updated plan |
|--------|--------------|--------------|
| I.6 (arviz-plots) | Deferred to follow-up release | **Included in major release** |
| II.7 (figure customization) | `figsize + ax` on all methods | 6 standard params via arviz-plots |
| `ax` parameter | Added to all methods | **Removed** — replaced by `plot_collection` |
| Return type | Always `tuple[Figure, NDArray[Axes]]` | Default `tuple[Figure, NDArray[Axes]]`, opt-in `PlotCollection` via `return_as_pc=True` |
| PR 1 (Foundation) | Subplot creation, color mapping, HDI helpers | + `_process_plot_params`, `_extract_matplotlib_result`, arviz-plots imports |
| Family PRs (2–8) | Port methods, add figsize+ax | Port methods, **rewrite with PlotCollection** |
| LOE per family PR | M–L | L–XL (~50% more work, but significantly less code output) |

### Changes to the standardized API contract

Replace the "Required Parameters" table in the attack plan:

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `figsize` | `tuple[float, float] \| None` | `None` | Convenience for `figure_kwargs`; ignored with `plot_collection` |
| `plot_collection` | `PlotCollection \| None` | `None` | Plot onto existing collection |
| `backend` | `str \| None` | `None` | `"matplotlib"`, `"plotly"`, `"bokeh"` |
| `visuals` | `dict[str, Any] \| None` | `None` | Element-level customization |
| `aes_by_visuals` | `dict[str, list[str]] \| None` | `None` | Aesthetic mapping per visual |
| `dims` | `dict[str, Any] \| None` | `None` | Subset dimensions |
| `return_as_pc` | `bool` | `False` | Opt-in to `PlotCollection` return |
| `**pc_kwargs` | | | Forwarded to `PlotCollection.wrap/grid` |

### Removed from contract

| Old parameter | Reason |
|---------------|--------|
| `ax: plt.Axes \| None` | Doesn't exist in arviz-plots; replaced by `plot_collection` |
| `**kwargs` to primary plot call | Replaced by `visuals` dict |
