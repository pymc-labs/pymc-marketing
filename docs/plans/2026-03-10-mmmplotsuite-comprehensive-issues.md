# MMMPlotSuite — Comprehensive Issue Analysis

> Critical code review of `pymc_marketing/mmm/plot.py` (5,150 lines, 21 public methods).
> Covers 20 open GitHub issues plus additional problems discovered via code audit.
> Prepared 2026-03-10.
>
> **LOE Scale:** XS (< 1 hour) · S (1–4 hours) · M (1–3 days) · L (1–2 weeks) · XL (2+ weeks)

---

## Table of Contents

- [I. Structural / Architectural Issues](#i-structural--architectural-issues)
- [II. API Consistency Issues (GitHub #822, #2369, #2371, #2373–#2376, #2378)](#ii-api-consistency-issues)
- [III. Missing Functionality (GitHub #2052–#2242, #2153)](#iii-missing-functionality)
- [IV. Code Quality & Bugs (Not on GitHub)](#iv-code-quality--bugs-not-on-github)
- [V. Test Coverage Gaps](#v-test-coverage-gaps)
- [VI. Documentation (GitHub #820)](#vi-documentation)
- [VII. Older / Miscellaneous (GitHub #76, #765, #1183)](#vii-older--miscellaneous)
- [VIII. Stale Branch Assessment (`feature/mmmplotsuite-arviz`)](#viii-stale-branch-assessment)
- [IX. Priority Matrix](#ix-priority-matrix)

---

## I. Structural / Architectural Issues

These are systemic problems not captured by any single GitHub issue.

### I.1 God Class — 5,150 lines, 21+ public methods

`MMMPlotSuite` violates the Single Responsibility Principle. It handles seven distinct
plotting families:


| Family                      | Methods                                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| Time-series diagnostics     | `posterior_predictive`, `prior_predictive`, `residuals_over_time`, `residuals_posterior_distribution` |
| Distribution diagnostics    | `posterior_distribution`, `channel_parameter`, `prior_vs_posterior`                                   |
| Saturation/response curves  | `saturation_scatterplot`, `saturation_curves`, `saturation_curves_scatter` (deprecated)               |
| Budget allocation           | `budget_allocation`, `allocated_contribution_by_channel_over_time`                                    |
| Sensitivity/uplift/marginal | `sensitivity_analysis`, `uplift_curve`, `marginal_curve`                                              |
| Decomposition               | `waterfall_components_decomposition`, `contributions_over_time`, `channel_contribution_share_hdi`     |
| Cross-validation            | `cv_predictions`, `param_stability`, `cv_crps`                                                        |


Each of these families could be a separate module or mixin. The monolithic class makes
the file extremely difficult to navigate, test, and extend.

> **Backward Compatible:** Yes (refactor into modules/mixins preserving public API) — **LOE:** XL

### I.2 Constructor allows invalid state

```python
MMMPlotSuite(idata=None, data=None)
```

When both are `None` (line 232–233), the constructor `cast`s `None` to
`az.InferenceData` and `MMMIDataWrapper`. This creates an object where every method
will crash with an `AttributeError` on first use, instead of failing fast at
construction time. (GitHub #2388)

> **Backward Compatible:** Yes (no real usage relies on None-None construction; both paths crash on first use anyway) — **LOE:** XS

### I.3 MMMIDataWrapper largely bypassed

The wrapper (`self.data`) provides validated, type-safe access to idata groups. However,
~90% of methods bypass it and reach directly into `self.idata.posterior`,
`self.idata.constant_data`, etc. Only `saturation_curves`, `sensitivity_analysis`, and
a few helpers use `self.data`.

This means the validation, scaling, and filtering capabilities of the wrapper are
wasted. If the wrapper's contract changes, 17+ methods won't benefit.

**Key example — contribution variable name resolution (GitHub #2370):** Because methods
bypass the wrapper, five different strategies exist for choosing between
`"channel_contribution"` and `"channel_contribution_original_scale"`: ternary string
selection, `next()` generator scan (throws `StopIteration` instead of `ValueError`),
prefer-original-then-fallback, hardcoded `_original_scale` only, and manual
original-scale suffix iteration. All five would collapse into a single call to
`self.data.get_channel_contributions(original_scale)`, which already handles name
resolution, fallback computation, and scaling in one place. The optimizer/budget methods
that operate on an external `samples` dataset would need a small adapter.

> **Backward Compatible:** Yes (internal refactoring only) — **LOE:** L

### I.4 Deep nested functions instead of methods

Several methods contain complex nested functions (50–90 lines) that are hard to test,
debug, and reuse:


| Method                 | Nested function         | Lines     |
| ---------------------- | ----------------------- | --------- |
| `cv_predictions`       | `_align_y_to_df`        | ~15 lines |
| `cv_predictions`       | `_plot_hdi_from_sel`    | ~50 lines |
| `cv_crps`              | `_pred_matrix_for_rows` | ~90 lines |
| `cv_crps`              | `_filter_rows_and_y`    | ~15 lines |
| `sensitivity_analysis` | `_plot_line`            | ~57 lines |


These should be class methods, private module functions, or factored into helper
classes.

> **Backward Compatible:** Yes (internal refactoring only) — **LOE:** M

### I.5 No shared color palette or mapping

Each method picks colors independently (`"C0"`, `"C3"`, channel-indexed colors from the
default matplotlib cycle, etc.). There is no shared palette, color mapping, or
channel→color assignment. This means the same channel can appear as different colors
across different plots, making it harder for users to visually correlate information.

> **Backward Compatible:** Yes (additive; visual output changes but no API breakage) — **LOE:** M

### I.6 Duplicated subplot creation and population logic — delegate to arviz-plots

Multiple methods duplicate boilerplate for subplot grid creation, axes normalization,
HDI band drawing, legend assembly, and time-series line/fill layering.

Currently three different subplot-creation patterns coexist:


| Pattern                                          | Used by                                                                                                                                          |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `self._init_subplots()` (standardized helper)    | Most methods                                                                                                                                     |
| `plt.subplots()` directly                        | `saturation_scatterplot`, `saturation_curves`, `allocated_contribution`, `cv_predictions`, `param_stability`, `residuals_posterior_distribution` |
| Manual axes normalization after `plt.subplots()` | `saturation_curves` (2277–2282), `allocated_contribution` (3033–3040), `waterfall` (4094–4102)                                                   |


The duplicated axes-normalization code (flatten + wrap in ndarray) appears in at least
three methods, each with slightly different implementations.

All of this is already handled by `arviz_plots`' `PlotCollection` API
(`PlotCollection.wrap()` for dimension-based layouts, `PlotCollection.grid()` for
row×col grids, `pc.map()` for layering visual elements, `pc.add_legend()` for legends,
and built-in visuals like `fill_between_y`, `line_xy`, `scatter_xy`).

The stale `feature/mmmplotsuite-arviz` branch demonstrated this approach for 9 methods,
eliminating hundreds of lines of manual subplot management. Adopting this across all 21
methods — without pursuing multi-backend support — would significantly reduce code
volume and consolidate the three subplot-creation patterns into one.

> **Backward Compatible:** Yes (internal refactoring; public API and return types unchanged) — **LOE:** XL

---

## II. API Consistency Issues

These correspond to GitHub issues #2369–#2378 and #2388, plus additional findings.

### II.1 Return type roulette (GitHub #822, #2369)

Methods return at least **five different shapes**:


| Return type                                                     | Methods                                                                                                  |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `tuple[Figure, NDArray[Axes]]`                                  | Most methods                                                                                             |
| `Figure` (bare)                                                 | `channel_parameter`                                                                                      |
| `plt.Axes` or `tuple[Figure, NDArray[Axes]]` (union)            | `sensitivity_analysis`, `uplift_curve`, `marginal_curve`                                                 |
| `tuple[Figure, Axes]` or `tuple[Figure, NDArray[Axes]]` (union) | `budget_allocation`, `waterfall_components_decomposition`, `allocated_contribution_by_channel_over_time` |
| `tuple[Figure, Axes]` (single)                                  | `param_stability` (declared as `NDArray[Axes]`, actually returns single `Axes`)                          |


**Additional finding:** `cv_predictions` declares return `tuple[Figure, NDArray[Axes]]`
but when `n_axes == 1`, it wraps axes in a Python `list`, not an `NDArray`. The type
annotation is incorrect.

**Additional finding:** `param_stability` declares `tuple[Figure, NDArray[Axes]]` but
returns `tuple[Figure, Axes]` (single Axes, not array) in most branches.

> **Backward Compatible:** No (changing return types breaks callers; requires deprecation cycle) — **LOE:** L

### II.2 Inconsistent `original_scale` default (GitHub #2371)


| Default     | Methods                                                                                                        |
| ----------- | -------------------------------------------------------------------------------------------------------------- |
| `True`      | `contributions_over_time`, `waterfall_components_decomposition`, `allocated_contribution_by_channel_over_time` |
| `False`     | `saturation_scatterplot`, `saturation_curves`                                                                  |
| Not exposed | `channel_contribution_share_hdi`                                                                               |


> **Backward Compatible:** No (changing defaults alters behavior for existing callers; requires deprecation cycle) — **LOE:** M

### II.3 `plt.show()` and figure discarding in `param_stability` / `cv_predictions` (GitHub #2373)


| Method            | `plt.show()` calls             |
| ----------------- | ------------------------------ |
| `cv_predictions`  | 1 (line 4641)                  |
| `param_stability` | **3** (lines 4762, 4794, 4813) |
| All other methods | 0                              |


`param_stability` is especially problematic — when `dims` is provided, it creates a
*separate figure per dimension value* in a for-loop, calling `plt.show()` on each.
It returns only the **last** `(fig, ax)` pair, silently discarding all previous figures.
This makes all figures except the last unreachable programmatically — a silent data loss
bug. Fixing this requires both removing the `plt.show()` calls and changing the return
type to include all figures (the latter is a breaking change requiring a deprecation
cycle).

> **Backward Compatible:** No (removing `plt.show()` is BC, but returning all figures changes the API contract) — **LOE:** M

### II.4 Monkey-patching `self.idata.sensitivity_analysis` (GitHub #2374)

`uplift_curve` (line 3544) and `marginal_curve` (line 3642) temporarily replace
`self.idata.sensitivity_analysis` with synthetic data, then restore in a `finally`
block. Thread-unsafe, mutates shared state.

> **Backward Compatible:** Yes (internal refactoring; public API unchanged) — **LOE:** M

### II.5 Parameter naming and type inconsistencies (GitHub #1751, #2375)


| Parameter                 | Inconsistency                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `var` vs `var_names`      | `posterior_predictive` takes `list[str]`, `prior_predictive` takes `str`; neither uses ArviZ's `var_names` convention (GitHub #1751)              |
| `hdi_prob` vs `hdi_probs` | Singular everywhere except `saturation_curves` (plural + accepts `list[float]`)                                                                   |
| `figsize`                 | `tuple[float, float]` in most places; `tuple[int, int]` in `waterfall_components_decomposition`; bare `tuple` in `channel_contribution_share_hdi` |
| `subplot_kwargs`          | `saturation_curves` uses a less-precise annotation                                                                                                |
| `dims` type               | `param_stability` uses a more restrictive type than other methods                                                                                 |
| `rc_params`               | Only available on `saturation_curves`                                                                                                             |


> **Backward Compatible:** No (renaming parameters breaks keyword callers; requires deprecation aliases) — **LOE:** M

### II.6 No subsetting on predictive/residual or media methods (GitHub #822, #2376)

`posterior_predictive`, `prior_predictive`, `residuals_over_time`, and
`residuals_posterior_distribution` auto-detect extra dimensions and create subplot grids,
but offer **no** `dims` parameter. A 50-geo model generates 50+ subplots with no way to
select a subset.

Additionally, media transformation plots (`saturation_scatterplot`, `saturation_curves`)
have no option to subset by channel — all channels are always plotted.

> **Backward Compatible:** Yes (additive — new optional parameter) — **LOE:** S

### II.7 Inconsistent figure customization surface (GitHub #822, #2378)


| Customization   | Coverage                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------- |
| `figsize`       | 14/21 methods                                                                                |
| `ax` parameter  | 4/21 methods (`budget_allocation`, `sensitivity_analysis`, `uplift_curve`, `marginal_curve`) |
| `**kwargs`      | Varies — sometimes to `plt.subplots()`, sometimes to plot calls                              |
| `rc_params`     | 1/21 methods (`saturation_curves` only)                                                      |
| `title` control | Sensitivity family only                                                                      |


7 methods offer **zero** figure customization parameters.

> **Backward Compatible:** Yes (additive — new optional parameters) — **LOE:** L

---

## III. Missing Functionality

### III.1 Four legacy methods not in the suite (GitHub #2128)


| Legacy method                                   | File           | Suite equivalent                                  |
| ----------------------------------------------- | -------------- | ------------------------------------------------- |
| `plot_grouped_contribution_breakdown_over_time` | `base.py:1068` | **None** (GitHub #2052)                           |
| `plot_prior_vs_posterior`                       | `base.py:1225` | Exists in suite but legacy has different features |
| `plot_channel_contribution_grid`                | `mmm.py:1805`  | **None** (GitHub #2053)                           |
| `plot_new_spend_contributions`                  | `mmm.py:2007`  | **None** (GitHub #2241)                           |


> **Backward Compatible:** Yes (additive — new methods) — **LOE:** L

### III.2 Gradient/HDI bands for posterior predictive (GitHub #2054)

Status: `implementation-done`, `plan-done`. May already be merged.

> **Backward Compatible:** Yes (enhancement) — **LOE:** S (possibly already done)

### III.3 Missing: aggregated channel contributions (GitHub #2242)

`contributions_over_time` shows individual channels. No option to aggregate channels
into a single stacked view.

> **Backward Compatible:** Yes (additive — new aggregation option on existing method) — **LOE:** S

### III.4 Missing: out-of-sample plotting methods (GitHub #2153)

Users must manually write train/test split visualization code. No built-in
out-of-sample assessment plots.

> **Backward Compatible:** Yes (new method) — **LOE:** M

### III.5 Missing: parametric fit overlay on saturation scatter (Not on GitHub)

`MMM.plot_direct_contribution_curves()` (mmm.py:2292) draws scatter plus a parametric
saturation fit curve via `_plot_response_curve_fit()`. The suite's
`saturation_scatterplot()` only covers the scatter part — the fit overlay has no
equivalent.

> **Backward Compatible:** Yes (additive feature on existing method) — **LOE:** S

---

## IV. Code Quality & Bugs (Not on GitHub)

These are issues found during code review that have no corresponding GitHub issue.

### IV.1 Copy-paste bugs in `prior_predictive` and its helper

Line 370–377: The error message in `_get_prior_predictive_data` says
"posterior_predictive" but should say "prior_predictive". The code comment also says
"check if self.idata has posterior_predictive" when it should say "prior_predictive".
Copy-pasted from `_get_posterior_predictive_data` without updating.

**Additional locations of the same copy-paste problem:**

- Line 587–614: The `prior_predictive` method's own docstring says "Plot time series
from the **posterior** predictive distribution" and references
`self.idata.posterior_predictive`.
- Line 662: The fallback title string reads "Posterior Predictive Time Series" when it
should say "Prior Predictive Time Series".

**Testing gap:** There is a test for `_get_posterior_predictive_data` error handling but
not for the prior equivalent `_get_prior_predictive_data`. A regression test should be
added alongside the fix.

> **Backward Compatible:** Yes (bug fix — correcting wrong error messages and strings) — **LOE:** XS

### IV.2 `plt.gcf()` fragility in `channel_contribution_share_hdi`

Line 4265: After `az.plot_forest` creates a figure, the code grabs the current figure
via `plt.gcf()`. If another thread or callback creates a figure between the
`az.plot_forest` call and `plt.gcf()`, the wrong figure is returned.

> **Backward Compatible:** Yes (internal fix) — **LOE:** S

### IV.3 `_validate_dims` always validates against `self.idata.posterior`

Line 427–450: This helper always validates dimension names against `posterior.coords`,
even when called by methods that operate on non-posterior data (e.g., `budget_allocation`
validates against `samples.dims`).

> **Backward Compatible:** Yes (bug fix) — **LOE:** S

### IV.4 `color_cycle` iteration bug in `sensitivity_analysis`

Lines 3319+3398: `color_cycle = itertools.cycle(hue_colors)` is created once outside the
panel loop but consumed inside nested loops. For multi-panel plots, the cycle continues
from where the previous panel left off, meaning different panels get different color
assignments for the same hue values.

This can be fixed as part of I.5 (shared color palette/mapping) or as a standalone XS
bug fix by resetting the cycle at the start of each panel iteration.

> **Backward Compatible:** Yes (bug fix — visual output correction) — **LOE:** XS

### IV.5 `title` parameter shadowing in `sensitivity_analysis`

Line 3435: The local `title` variable from `_build_subplot_title` overwrites the `title`
parameter. Then at line 3449, `if add_figure_title and title is not None` checks the
*last subplot's* title, not the original parameter.

> **Backward Compatible:** Yes (bug fix) — **LOE:** XS

### IV.6 Per-date HDI computation loop

`_plot_single_allocated_contribution` (lines 2863–2867): Computes HDI per-date in a
Python for-loop instead of vectorized. For 365 dates, this is 365 separate `az.hdi()`
calls. Extremely slow for large datasets.

> **Backward Compatible:** Yes (performance improvement; same output) — **LOE:** M

### IV.7 O(n×m) inner loop in `cv_crps`

`_pred_matrix_for_rows` (line 4914): Iterates over every row of `rows_df` and does
`.sel()` per row. For large datasets this is very slow.

> **Backward Compatible:** Yes (performance improvement; same output) — **LOE:** M

### IV.8 Index-as-coordinate bug in `_plot_single_waterfall`

Line 3897: `index` is the pandas DataFrame index value (not necessarily a sequential
integer position), but `ax.text()` uses it as a y-coordinate. If the DataFrame has been
filtered, the index may not be sequential, causing misplaced text labels.

> **Backward Compatible:** Yes (bug fix) — **LOE:** XS

### IV.9 `**kwargs` + `**subplot_kwargs` duplicate-key risk

Lines 3029, 4058: Both `allocated_contribution_by_channel_over_time` and
`waterfall_components_decomposition` unpack `**subplot_kwargs` and `**kwargs` into
`plt.subplots()`. If a user passes `figsize` in both, a `TypeError` occurs.

> **Backward Compatible:** Yes (adding key-conflict validation) — **LOE:** XS

### IV.10 Broad exception catching masks real bugs

Multiple locations catch `except Exception` or `except (KeyError, ValueError, TypeError)`
and emit warnings instead of raising:

- `sensitivity_analysis` line 3189: catches all exceptions during float casting
- `cv_predictions` lines 4430, 4467, 4485, 4534: catch broad exception tuples
- `cv_crps` multiple locations: silently warn on data issues

This hides real data problems from users.

> **Backward Compatible:** Partially (previously silent errors will now surface; correctness fix but behavior change) — **LOE:** M

### IV.11 Dead code: `MMMIDataWrapper._cache`

Line 69 in `mmm_wrapper.py`: `self._cache: dict[str, Any] = {}` is initialized but never
read or written to anywhere in the class.

> **Backward Compatible:** Yes (removing dead code) — **LOE:** XS

### IV.12 `_reduce_and_stack` silently sums over unknown dimensions

Lines 325–340: When a variable has dimensions beyond `date`, `chain`, `draw`, and
`channel`, this helper sums over them silently. This could mask data bugs — if a user
adds an unexpected dimension, contributions will be silently aggregated without warning.

> **Backward Compatible:** Partially (adding a warning is BC; raising an error would break code relying on implicit aggregation) — **LOE:** XS

### IV.13 Redundant local imports

`sensitivity_analysis` line 3191: `import warnings` is redundant — already imported at
module level (line 187). A second instance exists in `saturation_curves_scatter` at
line 2424 — same redundant `import warnings`.

> **Backward Compatible:** Yes (cleanup) — **LOE:** XS

### IV.14 `ax` unpacking fragility in `channel_contribution_share_hdi`

Line 4253: `ax, *_ = az.plot_forest(...)` — the unpacking silently discards extra axes.
If ArviZ's `plot_forest` returns multiple axes (e.g., for multi-group data), the extra
information is lost.

> **Backward Compatible:** Yes (internal fix) — **LOE:** XS

### IV.15 Seaborn dependency for only 2 methods

The entire `seaborn` import (line 195) is used by only `posterior_distribution` (violin
plots) and `prior_vs_posterior` (KDE). The rest of the module is pure matplotlib. This
is a heavy dependency for limited use.

> **Backward Compatible:** Yes (convert to lazy import; no API change) — **LOE:** S

### IV.16 Error message formatting in `saturation_scatterplot` / `saturation_curves`

Lines 2058–2067, 2210–2218: Error messages use raw `f"""...\n..."""` with literal `\n`
characters in the string, producing poorly formatted multi-line messages with extra
whitespace and misaligned indentation when displayed to the user.

> **Backward Compatible:** Yes (cosmetic fix) — **LOE:** XS

### IV.17 No validation of `agg` parameter in `waterfall_components_decomposition`

Line 3921: The method accepts `agg` indirectly via `_prepare_waterfall_data` →
`build_contributions`, but there is no validation that `agg` is one of `"mean"` or
`"median"`. The method signature doesn't even expose `agg` — it is hardcoded to
`"mean"` at line 4034, making the parameter pathway dead code.

> **Backward Compatible:** Yes (dead code cleanup) — **LOE:** XS

### IV.18 `_compute_residuals` hardcodes `y_original_scale`

Lines 670–711: The residuals computation hardcodes `y_original_scale` as the prediction
variable and `target_data` as the observed variable. There is no way to compute
residuals from different variables (e.g., in scaled space), limiting the usefulness of
`residuals_over_time` and `residuals_posterior_distribution`.

> **Backward Compatible:** Yes (additive — new optional parameter with backward-compatible default) — **LOE:** S

---

## V. Test Coverage Gaps

Based on analysis of `tests/mmm/test_plot.py` (~5,336 lines).

### V.1 Methods with zero or near-zero dedicated tests


| Method                      | Test count         | Quality                                                           |
| --------------------------- | ------------------ | ----------------------------------------------------------------- |
| `prior_predictive`          | **0** unit tests   | Only tested indirectly via legacy `test_plotting.py`              |
| `posterior_predictive`      | 1 integration test | No mocked unit tests; missing hdi_prob, original_scale, multi-dim |
| `budget_allocation`         | 2 tests            | Type-check only; missing figsize, error paths, bar labels         |
| `uplift_curve`              | 1 test             | "Doesn't crash" quality                                           |
| `marginal_curve`            | 1 test             | "Doesn't crash" quality                                           |
| `saturation_curves_scatter` | 1 test             | Only checks deprecation warning                                   |


> **Backward Compatible:** N/A (test-only) — **LOE:** L

### V.2 Assertion quality is weak in many tests

Many tests only assert `isinstance(fig, Figure)` or `fig is not None`. They don't
check:

- Axis labels, titles, legend entries
- Number of axes matches expected dimensions
- Data plotted matches expected values
- Correct number of lines/bars/patches

> **Backward Compatible:** N/A (test-only) — **LOE:** M

### V.3 No edge case tests

No tests cover:

- Single channel scenario (1 media channel)
- Single geo scenario (1 dimension value)
- NaN/missing values in contributions or posterior data
- Empty/zero contribution values
- Very large number of geos (performance)
- Custom date frequencies (quarterly, daily)
- `original_scale=True` with missing `*_original_scale` variables

> **Backward Compatible:** N/A (test-only) — **LOE:** L

### V.4 Legacy test migration incomplete (GitHub #86)

`test_plotting.py` (~373 lines) tests old `BaseMMM`/`MMM` plotting methods (not
MMMPlotSuite) with weak assertions (`isinstance(func(**kwargs), plt.Figure)`). These
should be migrated to test `MMMPlotSuite` directly with richer assertions.

> **Backward Compatible:** N/A (test-only) — **LOE:** M

### V.5 Thread-safety of monkey-patching not tested

`uplift_curve` and `marginal_curve` monkey-patching is not tested for concurrent access.

> **Backward Compatible:** N/A (test-only) — **LOE:** S

---

## VI. Documentation

### VI.1 Plotting gallery (GitHub #820)

No comprehensive gallery of all available plots. Users must read docstrings to discover
what's available.

> **Backward Compatible:** N/A (documentation) — **LOE:** L

---

## VII. Older / Miscellaneous

### VII.1 Coord name `x` → `channel` (GitHub #1183)

`plot_channel_contribution_share_hdi` uses `x` as coordinate name instead of `channel`.

> **Backward Compatible:** No (changing coordinate names may break downstream code; requires deprecation) — **LOE:** S

### VII.2 Rely more on `xarray.DataArray.to_dataframe` (GitHub #76)

Legacy issue. Multiple methods manually extract data from xarray instead of using
`to_dataframe()`.

**Concrete example — `_process_decomposition_components`:** Lines 3692–3703 assume
specific column ordering after `stack()`. If xarray changes its stacking behavior or the
data has unexpected dimensions, the rename will break silently. Using `to_dataframe()`
with named column access would eliminate this fragility.

> **Backward Compatible:** Yes (internal refactoring) — **LOE:** M

### VII.3 Time-varying media visualization (GitHub #765)

Enhancement to visualize how media effects change over time (time-varying coefficients).

> **Backward Compatible:** Yes (new method) — **LOE:** L

---

## VIII. Stale Branch Assessment (`feature/mmmplotsuite-arviz`)

### What was attempted

The branch introduces a multi-backend architecture powered by `arviz_plots`:

- **Config system:** `MMMPlotConfig` singleton (modeled after ArviZ `rcParams`) with
`plot.backend`, `plot.use_v2`, `plot.show_warnings` settings
- **Legacy extraction:** Original `MMMPlotSuite` renamed to `LegacyMMMPlotSuite` and
moved to `legacy_plot.py`
- **Version dispatch:** `MMM.plot` property returns v1 or v2 suite based on config flag
- **Deprecation timeline:** v0.18.0 (introduce), v0.19.0 (default), v0.20.0 (remove legacy)

### Migration progress: ~60–65% complete

**Converted to arviz-plots (9 methods):**

- `posterior_predictive`, `contributions_over_time`
- `saturation_scatterplot`, `saturation_curves`
- `budget_allocation_roas` (new method, no legacy equivalent)
- `allocated_contribution_by_channel_over_time`
- `sensitivity_analysis`, `uplift_curve`, `marginal_curve`

**Still matplotlib-only stubs (4 methods):**

- `prior_predictive`, `residuals_over_time`, `residuals_posterior_distribution`,
`posterior_distribution`

**Missing entirely (7+ methods from current main):**

- `channel_parameter`, `prior_vs_posterior`, `waterfall_components_decomposition`,
`channel_contribution_share_hdi`, `cv_predictions`, `param_stability`, `cv_crps`

### Key arviz-plots APIs used


| API                              | Purpose                                  |
| -------------------------------- | ---------------------------------------- |
| `PlotCollection.wrap()`          | Time-series layouts (dim-based subplots) |
| `PlotCollection.grid()`          | Grid layouts (rows × cols)               |
| `pc.map()`                       | Layering visual elements                 |
| `pc.add_legend()`                | Channel legends                          |
| `visuals.fill_between_y`         | HDI bands                                |
| `visuals.line_xy`                | Median/mean lines                        |
| `visuals.scatter_xy`             | Scatter plots                            |
| `visuals.multiple_lines`         | Sampled posterior curves                 |
| `xr.DataArray.azstats.hdi()`     | HDI computation on xarray                |
| `arviz_base.convert_to_datatree` | DataArray → DataTree for `plot_dist`     |


### Design decisions worth preserving

1. **Every method accepts `plot_collection` and `backend` parameters** — composable
  plots and per-call backend override
2. **Shared `_sensitivity_analysis_plot` helper** — eliminated the monkey-patching
  anti-pattern from `uplift_curve`/`marginal_curve`
3. **Data injection pattern** — methods accept explicit data parameters with fallback
  to `self.idata`, enabling testability

### Why the branch is stale

31 commits behind `main`. The branch was created when `MMMPlotSuite` was ~1,937 lines.
It is now 5,150 lines (main has added 8+ new methods since the branch was created).
Merging would require re-implementing the conversion for all new methods and resolving
conflicts in the existing ones.

---

## IX. Priority Matrix

### Critical (blocks other work or causes user-facing bugs)


| ID   | Issue                                                                    | Reason                                   | BC  | LOE |
| ---- | ------------------------------------------------------------------------ | ---------------------------------------- | --- | --- |
| IV.1 | Copy-paste bug in `_get_prior_predictive_data`                           | Wrong error message misleads users       | Yes | XS  |
| II.3 | `plt.show()` + figure discarding in `param_stability` / `cv_predictions` | Silent data loss; prevents customization | No  | M   |
| II.4 | Monkey-patching is thread-unsafe                                         | Corrupts state under concurrency         | Yes | M   |
| I.2  | Constructor accepts `None, None`                                         | Deferred crash instead of fail-fast      | Yes | XS  |


### High (API inconsistencies that confuse users)


| ID   | Issue                                                       | Reason                                                          | BC  | LOE |
| ---- | ----------------------------------------------------------- | --------------------------------------------------------------- | --- | --- |
| I.1  | 5,150-line god class                                        | Unmaintainable, hard to extend                                  | Yes | XL  |
| I.3  | MMMIDataWrapper largely bypassed (incl. #2370)              | Wrapper bypassed; 5 contrib-var strategies, `StopIteration` bug | Yes | L   |
| II.1 | Return type roulette (5 shapes)                             | Callers can't write generic code                                | No  | L   |
| II.2 | Inconsistent `original_scale` default                       | Surprising behavior differences                                 | No  | M   |
| II.5 | Parameter naming inconsistencies (incl. `var_names`, #1751) | Confusing API surface                                           | No  | M   |
| II.6 | No subsetting on predictive/media methods                   | 50-geo or many-channel models produce unusable output           | Yes | S   |
| II.7 | Inconsistent figure customization                           | 7 methods have zero customization                               | Yes | L   |
| I.6  | Duplicated subplot logic — delegate to arviz-plots          | Major code reduction opportunity                                | Yes | XL  |


### Medium (missing functionality, performance, testing)


| ID          | Issue                                             | Reason                           | BC      | LOE |
| ----------- | ------------------------------------------------- | -------------------------------- | ------- | --- |
| III.1–III.5 | Missing methods (#2128, #2054, #2242, #2153)      | Feature gaps                     | Yes     | S–L |
| IV.6        | Per-date HDI computation loop                     | Performance bottleneck           | Yes     | M   |
| IV.7        | O(n×m) loop in `cv_crps`                          | Performance bottleneck           | Yes     | M   |
| IV.10       | Broad exception catching                          | Masks real bugs                  | Partial | M   |
| V.1         | `prior_predictive` has 0 tests                    | Risk of regressions              | N/A     | L   |
| V.2         | Weak test assertions                              | False confidence in test suite   | N/A     | M   |
| I.5         | No shared color palette / mapping                 | Inconsistent colors across plots | Yes     | M   |
| IV.18       | `_compute_residuals` hardcodes `y_original_scale` | Limits residual analysis         | Yes     | S   |


### Lower (cleanup, nice-to-have)


| ID          | Issue                                               | Reason                                         | BC      | LOE |
| ----------- | --------------------------------------------------- | ---------------------------------------------- | ------- | --- |
| IV.2        | `plt.gcf()` fragility                               | Unlikely in practice but architecturally wrong | Yes     | S   |
| IV.9        | `**kwargs` + `**subplot_kwargs` conflict            | Edge case                                      | Yes     | XS  |
| IV.11       | Dead `_cache` in MMMIDataWrapper                    | Dead code                                      | Yes     | XS  |
| IV.12       | `_reduce_and_stack` silently sums                   | Subtle data bug risk                           | Partial | XS  |
| IV.13       | Redundant local imports (2 locations)               | Noise                                          | Yes     | XS  |
| IV.15       | Seaborn dependency for 2 methods                    | Optional optimization                          | Yes     | S   |
| IV.16       | Error message formatting (raw `\n` in f-strings)    | Cosmetic                                       | Yes     | XS  |
| IV.17       | No validation of `agg` parameter (hardcoded anyway) | Dead code path                                 | Yes     | XS  |
| VI.1        | Plotting gallery (#820)                             | Documentation                                  | N/A     | L   |
| VII.1–VII.3 | Older issues (#76, #765, #1183)                     | Low urgency                                    | Mixed   | S–L |


---

## Appendix: Issue Count Summary


| Category                   | GitHub Issues                                     | Code-Review Findings | Total  |
| -------------------------- | ------------------------------------------------- | -------------------- | ------ |
| Structural / Architectural | —                                                 | 6                    | 6      |
| API Consistency            | 9 (#822, #1751, #2369, #2371, #2373–#2376, #2378) | 0                    | 7      |
| Missing Functionality      | 7 (#2052–#2242, #2153)                            | 1                    | 5      |
| Code Quality & Bugs        | —                                                 | 18                   | 18     |
| Test Coverage Gaps         | —                                                 | 5                    | 5      |
| Documentation              | 1 (#820)                                          | 0                    | 1      |
| Older / Miscellaneous      | 3 (#76, #765, #1183)                              | 0                    | 3      |
| **Total**                  | **20**                                            | **30**               | **45** |
