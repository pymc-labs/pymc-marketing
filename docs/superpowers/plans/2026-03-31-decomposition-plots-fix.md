# DecompositionPlots Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three bugs in `DecompositionPlots`: baseline has no date dim, channels must be plotted individually not summed, and waterfall must show one bar per channel/control.

**Architecture:** Pre-process contribution datasets into a flat `entries: list[tuple[str, xr.DataArray]]` list before rendering loops, so the rendering code is unchanged. Baseline is broadcast over dates via `expand_dims`. Extract `_plot_waterfall_panel` helper to keep `waterfall` focused.

**Tech Stack:** Python, xarray, arviz-plots, matplotlib, pytest

---

## File Map

| File | Change |
|---|---|
| `tests/mmm/plotting/test_decomposition.py` | Fix fixtures + add new tests |
| `pymc_marketing/mmm/plotting/decomposition.py` | Fix `contributions_over_time`, fix `waterfall`, add `_plot_waterfall_panel` |

---

### Task 1: Fix `simple_idata` fixture

**Files:**
- Modify: `tests/mmm/plotting/test_decomposition.py:43-91`

- [ ] **Step 1: Update the `simple_idata` fixture**

Replace the entire `simple_idata` fixture so `intercept_contribution` has no `date` dim, and remove controls and seasonality variables entirely.

```python
@pytest.fixture(scope="module")
def simple_idata() -> az.InferenceData:
    """Minimal idata with channels + baseline contributions, no extra dims.

    posterior:
      channel_contribution   (chain, draw, date, channel)
      intercept_contribution (chain, draw)              -- no date dim
    constant_data:
      target_data  (date,)
      target_scale scalar
    """
    rng = np.random.default_rng(SEED)
    n_chain, n_draw, n_date = 2, 40, 20
    channels = ["tv", "radio", "social"]
    dates = np.arange(n_date)

    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                rng.uniform(0, 100, size=(n_chain, n_draw, n_date, len(channels))),
                dims=("chain", "draw", "date", "channel"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "channel": channels,
                },
            ),
            "intercept_contribution": xr.DataArray(
                rng.uniform(50, 150, size=(n_chain, n_draw)),
                dims=("chain", "draw"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                },
            ),
        }
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date,)),
                dims=("date",),
                coords={"date": dates},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(posterior=posterior, constant_data=const)
```

- [ ] **Step 2: Run existing tests to see which ones break**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py -x -q 2>&1 | head -60
```

Expected: Some tests fail (that's fine — we're fixing the implementation next).

---

### Task 2: Fix `panel_idata` fixture

**Files:**
- Modify: `tests/mmm/plotting/test_decomposition.py:94-148`

- [ ] **Step 1: Update the `panel_idata` fixture**

Replace the entire `panel_idata` fixture so `intercept_contribution` has no `date` dim, and add `control_contribution` and `yearly_seasonality_contribution`.

```python
@pytest.fixture(scope="module")
def panel_idata() -> az.InferenceData:
    """idata with geo extra dim.

    posterior:
      channel_contribution          (chain, draw, date, geo, channel)
      intercept_contribution        (chain, draw, geo)              -- no date dim
      control_contribution          (chain, draw, date, geo, control)
      yearly_seasonality_contribution (chain, draw, date, geo)
    constant_data:
      target_data  (date, geo)
      target_scale scalar
    """
    rng = np.random.default_rng(SEED + 1)
    n_chain, n_draw, n_date = 2, 30, 15
    channels = ["tv", "radio"]
    controls = ["price", "trend"]
    geos = ["CA", "NY"]
    dates = np.arange(n_date)

    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                rng.uniform(
                    0, 100, size=(n_chain, n_draw, n_date, len(geos), len(channels))
                ),
                dims=("chain", "draw", "date", "geo", "channel"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "geo": geos,
                    "channel": channels,
                },
            ),
            "intercept_contribution": xr.DataArray(
                rng.uniform(50, 150, size=(n_chain, n_draw, len(geos))),
                dims=("chain", "draw", "geo"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "geo": geos,
                },
            ),
            "control_contribution": xr.DataArray(
                rng.uniform(
                    -20, 20, size=(n_chain, n_draw, n_date, len(geos), len(controls))
                ),
                dims=("chain", "draw", "date", "geo", "control"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "geo": geos,
                    "control": controls,
                },
            ),
            "yearly_seasonality_contribution": xr.DataArray(
                rng.uniform(
                    -10, 10, size=(n_chain, n_draw, n_date, len(geos))
                ),
                dims=("chain", "draw", "date", "geo"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "geo": geos,
                },
            ),
        }
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date, len(geos))),
                dims=("date", "geo"),
                coords={"date": dates, "geo": geos},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(posterior=posterior, constant_data=const)
```

- [ ] **Step 2: Verify fixtures parse without error**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py -x -q --collect-only 2>&1 | head -20
```

Expected: collection succeeds (no fixture errors).

---

### Task 3: Fix `contributions_over_time`

**Files:**
- Modify: `pymc_marketing/mmm/plotting/decomposition.py:129-196`

- [ ] **Step 1: Replace the `reduced` pre-processing block**

Replace these lines in `contributions_over_time` (the block from `extra_dims = list(data.custom_dims)` down to `if not reduced:`) with the `entries` expansion:

```python
        extra_dims = list(data.custom_dims)
        dates = data.dates

        # Build flat entries list: each entry has dims (chain, draw, date[, extra_dims])
        # so the rendering loop below is unchanged.
        entries: list[tuple[str, xr.DataArray]] = []

        if "channels" in contributions_ds:
            ch_da = contributions_ds["channels"]
            for ch in ch_da.coords["channel"].values:
                entries.append((str(ch), _select_dims(ch_da.sel(channel=ch), dims)))

        if "baseline" in contributions_ds:
            bl_da = contributions_ds["baseline"]
            # baseline has no date dim — broadcast it over the date axis
            bl_broadcast = bl_da.expand_dims({"date": dates})
            entries.append(("baseline", _select_dims(bl_broadcast, dims)))

        if "controls" in contributions_ds:
            ctrl_da = contributions_ds["controls"]
            # sum over the control dim → single time-series
            entries.append(("controls", _select_dims(ctrl_da.sum(dim="control"), dims)))

        if "seasonality" in contributions_ds:
            seas_da = contributions_ds["seasonality"]
            entries.append(("seasonality", _select_dims(seas_da, dims)))

        if not entries:
            raise ValueError(
                "No contribution data found after filtering. "
                "Check that the model has the requested contribution types."
            )

        first_da = entries[0][1]
```

Also update the layout dataset and PlotCollection creation to use `first_da` (it's already used correctly after this block) and update the rendering loop variable from `reduced.items()` to `entries`:

```python
        for i, (label, da) in enumerate(entries):
```

The full updated method body (replacing from `extra_dims = list(data.custom_dims)` to end of method):

```python
        extra_dims = list(data.custom_dims)
        dates = data.dates

        # Build flat entries: each entry has dims (chain, draw, date[, extra_dims])
        entries: list[tuple[str, xr.DataArray]] = []

        if "channels" in contributions_ds:
            ch_da = contributions_ds["channels"]
            for ch in ch_da.coords["channel"].values:
                entries.append((str(ch), _select_dims(ch_da.sel(channel=ch), dims)))

        if "baseline" in contributions_ds:
            bl_da = contributions_ds["baseline"]
            bl_broadcast = bl_da.expand_dims({"date": dates})
            entries.append(("baseline", _select_dims(bl_broadcast, dims)))

        if "controls" in contributions_ds:
            ctrl_da = contributions_ds["controls"]
            entries.append(("controls", _select_dims(ctrl_da.sum(dim="control"), dims)))

        if "seasonality" in contributions_ds:
            seas_da = contributions_ds["seasonality"]
            entries.append(("seasonality", _select_dims(seas_da, dims)))

        if not entries:
            raise ValueError(
                "No contribution data found after filtering. "
                "Check that the model has the requested contribution types."
            )

        first_da = entries[0][1]
        layout_ds = (
            first_da.mean(dim=("chain", "draw"))
            .isel(date=0, drop=True)
            .to_dataset(name="_layout")
        )
        pc_kwargs.setdefault("col_wrap", 1)
        pc = PlotCollection.wrap(
            layout_ds,
            cols=extra_dims,
            backend=backend,
            **pc_kwargs,
        )

        for i, (label, da) in enumerate(entries):
            mean_da = da.mean(dim=("chain", "draw"))
            hdi_da = da.azstats.hdi(hdi_prob)
            color = f"C{i}"

            pc.map(
                azp.visuals.fill_between_y,
                x=dates,
                y_bottom=hdi_da.sel(ci_bound="lower"),
                y_top=hdi_da.sel(ci_bound="upper"),
                **{"alpha": 0.2, "color": color, **(hdi_kwargs or {})},
            )
            pc.map(
                azp.visuals.line_xy,
                x=dates,
                y=mean_da,
                **{"label": label, "color": color, **(line_kwargs or {})},
            )

        pc.map(azp.visuals.labelled_x, text="Date", ignore_aes={"color"})
        pc.map(azp.visuals.labelled_y, text="Contribution", ignore_aes={"color"})
        pc.map(azp.visuals.labelled_title, subset_info=True, ignore_aes={"color"})

        return _extract_matplotlib_result(pc, return_as_pc)
```

- [ ] **Step 2: Run existing `TestContributionsOverTime` tests**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py::TestContributionsOverTime -x -q 2>&1 | head -60
```

Expected: most pass; some may fail on line-count assertions that reference the old summed structure.

---

### Task 4: Add `contributions_over_time` tests

**Files:**
- Modify: `tests/mmm/plotting/test_decomposition.py` — `TestContributionsOverTime` class

- [ ] **Step 1: Add `test_each_channel_has_own_line`**

Add inside `TestContributionsOverTime` after `test_x_axis_is_dates_y_axis_is_contributions`:

```python
    def test_each_channel_has_own_line(self, simple_plots):
        """Each channel must produce its own labeled line."""
        channels = ["tv", "radio", "social"]
        _fig, axes = simple_plots.contributions_over_time(include=["channels"])
        ax = axes[0]
        line_labels = [ln.get_label() for ln in ax.get_lines() if len(ln.get_xdata()) > 1]
        for ch in channels:
            assert ch in line_labels, (
                f"Expected a line labeled '{ch}' but found: {line_labels}"
            )

    def test_baseline_is_horizontal(self, simple_plots):
        """Baseline line must be constant across all dates (time-invariant intercept)."""
        _fig, axes = simple_plots.contributions_over_time(include=["baseline"])
        ax = axes[0]
        lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 1]
        baseline_lines = [ln for ln in lines if ln.get_label() == "baseline"]
        assert baseline_lines, "No line labeled 'baseline' found"
        ydata = baseline_lines[0].get_ydata()
        assert np.allclose(ydata, ydata[0]), (
            f"Baseline line should be horizontal (constant y), got: {ydata[:5]}…"
        )
```

- [ ] **Step 2: Update comment in `test_no_summing_warning`**

Replace the comment on the line before `simple_plots.contributions_over_time()`:

Old:
```python
        # Multi-dim contributions (e.g. channel) are silently summed — no UserWarning
```

New:
```python
        # Channels are plotted individually (not summed) — no UserWarning should be emitted
```

- [ ] **Step 3: Run the new tests**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py::TestContributionsOverTime -x -q 2>&1 | head -60
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/mmm/plotting/test_decomposition.py pymc_marketing/mmm/plotting/decomposition.py
git commit -m "fix: contributions_over_time plots each channel individually; baseline broadcast over dates"
```

---

### Task 5: Extract `_plot_waterfall_panel` helper

**Files:**
- Modify: `pymc_marketing/mmm/plotting/decomposition.py`

- [ ] **Step 1: Add `_plot_waterfall_panel` as a module-level private function**

Add this function just before the `DecompositionPlots` class definition (after imports, before the class):

```python
def _plot_waterfall_panel(
    ax: Axes,
    entries: list[tuple[str, float]],
    bar_kwargs: dict,
) -> None:
    """Draw a single waterfall panel onto *ax*.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    entries : list of (label, value)
        Ordered contribution components. A "total" bar is appended automatically.
    bar_kwargs : dict
        Extra kwargs forwarded to ``ax.barh()``.
    """
    total = sum(v for _, v in entries)
    components = [*entries, ("total", total)]

    running = 0.0
    for bar_idx, (label, val) in enumerate(components):
        if label == "total":
            color = "grey"
            left = 0.0
            width = val
        else:
            color = "green" if val >= 0 else "red"
            left = running
            width = val
            running += val

        ax.barh(
            y=bar_idx,
            width=width,
            left=left,
            color=color,
            **bar_kwargs,
        )
        pct = 100 * val / total if total != 0 else 0.0
        ax.text(
            left + width / 2,
            bar_idx,
            f"{val:.1f} ({pct:.1f}%)",
            va="center",
            ha="center",
            fontsize=8,
        )

    ax.set_yticks(range(len(components)))
    ax.set_yticklabels([c[0] for c in components])
    ax.axvline(0, color="black", linewidth=0.8)
```

- [ ] **Step 2: Run tests to verify no regressions**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py::TestWaterfall -x -q 2>&1 | head -40
```

Expected: existing waterfall tests still pass (the helper isn't wired in yet).

---

### Task 6: Fix `waterfall` to use `entries` + `_plot_waterfall_panel`

**Files:**
- Modify: `pymc_marketing/mmm/plotting/decomposition.py:236-336`

- [ ] **Step 1: Replace the `waterfall` method body**

Replace everything from `contributions_ds = data.get_contributions(...)` to `return fig, np.atleast_1d(np.array(axes_flat))` with:

```python
        contributions_ds = data.get_contributions(original_scale=original_scale)
        extra_dims = list(data.custom_dims)

        # Build entries: (label, xr.DataArray) where DataArray has dims (extra_dims,) or scalar
        entries: list[tuple[str, xr.DataArray]] = []

        if "baseline" in contributions_ds:
            bl_da = contributions_ds["baseline"]
            # baseline has no date dim — mean over chain, draw only
            entries.append(("baseline", bl_da.mean(dim=("chain", "draw"))))

        if "channels" in contributions_ds:
            ch_da = contributions_ds["channels"]
            ch_da = _select_dims(ch_da, dims)
            for ch in ch_da.coords["channel"].values:
                entries.append(
                    (str(ch), ch_da.sel(channel=ch).mean(dim=("chain", "draw", "date")))
                )

        if "controls" in contributions_ds:
            ctrl_da = contributions_ds["controls"]
            ctrl_da = _select_dims(ctrl_da, dims)
            for ctrl in ctrl_da.coords["control"].values:
                entries.append(
                    (
                        str(ctrl),
                        ctrl_da.sel(control=ctrl).mean(dim=("chain", "draw", "date")),
                    )
                )

        if "seasonality" in contributions_ds:
            seas_da = contributions_ds["seasonality"]
            seas_da = _select_dims(seas_da, dims)
            entries.append(("seasonality", seas_da.mean(dim=("chain", "draw", "date"))))

        # Apply dims filter to baseline separately (it may have extra_dims but no date/channel)
        if entries and "baseline" in dict(entries):
            bl_idx = next(i for i, (k, _) in enumerate(entries) if k == "baseline")
            entries[bl_idx] = ("baseline", _select_dims(entries[bl_idx][1], dims))

        # Determine panel combos from extra dims
        if extra_dims:
            ref_da = entries[0][1]
            coord_values = [ref_da.coords[d].values for d in extra_dims]
            combos = list(itertools.product(*coord_values))
        else:
            combos = [()]

        n_panels = len(combos)
        fig, axes_raw = plt.subplots(
            1, n_panels, figsize=figsize or (6 * n_panels, 4), squeeze=False
        )
        axes_flat = axes_raw.flatten()

        reserved_keys = {"y", "width", "left"}
        if bar_kwargs:
            conflict = reserved_keys & set(bar_kwargs.keys())
            if conflict:
                raise ValueError(
                    f"bar_kwargs keys conflict with positional bar arguments: {conflict}. "
                    "Do not pass 'y', 'width', or 'left' in bar_kwargs."
                )
        safe_bar_kwargs = {"height": 0.5, **(bar_kwargs or {})}

        for panel_idx, combo in enumerate(combos):
            ax = axes_flat[panel_idx]
            sel_kwargs = dict(zip(extra_dims, combo, strict=True))

            # Extract scalar (label, float) for this panel
            panel_entries: list[tuple[str, float]] = []
            for label, da in entries:
                if sel_kwargs:
                    da = da.sel(**{k: [v] for k, v in sel_kwargs.items()}).squeeze()
                panel_entries.append((label, float(da.values)))

            title = (
                " | ".join(f"{k}={v}" for k, v in sel_kwargs.items())
                if sel_kwargs
                else ""
            )
            if title:
                ax.set_title(title)

            _plot_waterfall_panel(ax, panel_entries, safe_bar_kwargs)

        fig.tight_layout()
        return fig, np.atleast_1d(np.array(axes_flat))
```

- [ ] **Step 2: Run waterfall tests**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py::TestWaterfall -x -q 2>&1 | head -60
```

Expected: all existing waterfall tests pass.

---

### Task 7: Add `waterfall` tests

**Files:**
- Modify: `tests/mmm/plotting/test_decomposition.py` — `TestWaterfall` class

- [ ] **Step 1: Add `test_baseline_bar_present`**

Add inside `TestWaterfall`:

```python
    def test_baseline_bar_present(self, simple_plots):
        """Waterfall must include a 'baseline' bar."""
        _fig, axes = simple_plots.waterfall()
        ax = axes[0]
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "baseline" in ytick_labels, (
            f"Expected 'baseline' in ytick labels, got: {ytick_labels}"
        )
```

- [ ] **Step 2: Add `test_bars_include_all_channels_and_controls`**

Add inside `TestWaterfall`:

```python
    def test_bars_include_all_channels_and_controls(self, panel_plots):
        """Each channel and each control must appear as its own bar."""
        channels = ["tv", "radio"]
        controls = ["price", "trend"]
        _fig, axes = panel_plots.waterfall()
        # one panel per geo — check the first one
        ax = axes[0]
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        for ch in channels:
            assert ch in ytick_labels, (
                f"Expected channel '{ch}' in ytick labels, got: {ytick_labels}"
            )
        for ctrl in controls:
            assert ctrl in ytick_labels, (
                f"Expected control '{ctrl}' in ytick labels, got: {ytick_labels}"
            )
```

- [ ] **Step 3: Run all waterfall tests**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py::TestWaterfall -x -q 2>&1 | head -60
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add pymc_marketing/mmm/plotting/decomposition.py tests/mmm/plotting/test_decomposition.py
git commit -m "fix: waterfall shows one bar per channel and control; extract _plot_waterfall_panel helper"
```

---

### Task 8: Full test run + pre-commit

**Files:** none

- [ ] **Step 1: Run the full decomposition test suite**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_decomposition.py -v 2>&1 | tail -30
```

Expected: all tests pass, no warnings about unexpected summing.

- [ ] **Step 2: Run pre-commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/decomposition.py tests/mmm/plotting/test_decomposition.py
```

Expected: all hooks pass.

- [ ] **Step 3: Final commit if pre-commit made any auto-fixes**

```bash
git add pymc_marketing/mmm/plotting/decomposition.py tests/mmm/plotting/test_decomposition.py
git commit -m "style: pre-commit auto-fixes for decomposition plots"
```

(Skip this step if pre-commit made no changes.)
