(experimental-term-optimizer)=
# Experimental term-first optimization

`pymc_marketing.experimental.terms.Problem` defines an optimization over deterministic terms from a fitted Structural Causal Model (`SCM`).
Construct a problem once, use `problem.evaluate(plan)` to score a concrete plan, then call `problem.solve(x0=plan)` to run the optimizer.
The fitted model supplies parameter identities and joint posterior draws; the caller supplies the objective, bounds, constraints and scoring policy.
This API is experimental; the stable `MMM` budget optimizer is unchanged.

Graph-first modeling and optimization live in `pymc_marketing.experimental.terms`; stable primitives such as `Parameter`, `Named`, and `Transform` remain in `pymc_marketing.terms`, while adstock, saturation, and Fourier components remain in `pymc_marketing.mmm`.

The lifecycle follows the [optimizer API decisions in PR #2981](https://github.com/pymc-labs/pymc-marketing/pull/2981#issuecomment-6102274917).
It replaces the earlier synchronous `optimize(...)` interface; the stable `MMM` optimizer is unchanged.

## Model and decisions in original units

Keep inputs, expected responses, bounds, constraints, and objective values in their business units.
Training-derived normalization constants belong inside the model graph and remain fixed during optimization.
If raw observations use a scaled Normal mean, its standard deviation must be scaled as well.

The following builds two responses sharing a spend input; `train` is a labeled `xarray.Dataset` with `spend(date, channel)`, `sales(date)`, and `signups(date)` in original units.

```python
from pymc_extras.prior import Prior
from pymc_marketing.experimental.terms import SCM, Data, Equation, MediaTransform, Problem
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation

spend = Data("spend")
spend_scale = train["spend"].max("date")
sales_scale = float(train["sales"].max())
signups_scale = float(train["signups"].max())

adstocked = MediaTransform(
    spend * (1.0 / spend_scale),
    GeometricAdstock(l_max=3, priors={"alpha": 0.4}),
)
sales_media = (
    (adstocked >> LogisticSaturation(prefix="sales")) * sales_scale
).named("sales_media")
signups_media = (
    (adstocked >> LogisticSaturation(prefix="signups")) * signups_scale
).named("signups_media")

scm = SCM(
    Equation(
        observed="sales",
        mu=sales_media.sum("channel"),
        likelihood=Prior("Normal", sigma=0.05 * sales_scale),
    ),
    Equation(
        observed="signups",
        mu=signups_media.sum("channel"),
        likelihood=Prior("Normal", sigma=0.05 * signups_scale),
    ),
)
scm.fit(train, random_seed=42)
```

`MediaTransform` accepts composite expressions such as normalized spend.
The `>>` shorthand starts directly from `Data` or another experimental graph term.
Reuse the fitted term objects when constructing optimization expressions; a new `Parameter` with the same name is not the fitted parameter.
After loading a saved `SCM`, select terms from the loaded equations rather than retaining objects from the original model.
New deterministic compositions and reductions of fitted terms are supported, including named references to fitted deterministics.
Decision values are continuous floating-point values, including when the corresponding training data were integers or Boolean.
Experimental numeric inputs are registered as unscaled `float64` before the fitted graph is constructed, including declared custom shared-term inputs, so integer training columns cannot insert implicit casts that truncate fractional allocations.
Custom shared terms declare their original Dataset inputs through `ModelTerm.data_vars`; when the registration lifecycle uniquely associates a new shared container with one declared input, that ownership is retained even if the container uses a different model name.
Keep original-unit values in those containers and express normalization in the graph; ambiguous undeclared ownership is not inferred from training values.
Raw observations and the stored training dataset retain their values and dtypes; explicit caller-authored casts and fixed training-derived normalization remain part of the fitted graph.
Integer inputs outside the exact `float64` range $[-2^{53}, 2^{53}]$ are rejected before conversion in training, scenarios, measured history, and numeric evaluation, rather than silently losing their original values; complex-valued inputs are also rejected.
An older in-memory graph with selected integer or `float32` shared inputs must be rebuilt and refitted; a saved reload is usable only when its rebuilt graph passes the standard joint-logp check and has canonical inputs.

## Build once, evaluate, then solve

`initial` is a labeled Dataset containing an initial spend array; `channel_caps` is a labeled channel DataArray, and `budget` and `weekly_cap` are in the same monetary units as spend.
The required `dates` index describes only periods whose inputs can be decided, not historical rows or a carryover tail.

```python
def profit(evaluate, u):
    out = evaluate(u)
    value = 0.3 * out["sales"] + 40.0 * out["signups"]
    return value.sum("date").sum("channel").mean("sample")


def late_sales_floor(evaluate, u):
    late = evaluate(u)["sales"].isel(date=slice(-2, None))
    return (
        late.sum("date").sum("channel").mean("sample")
        - evaluate.parameter("floor")
    )


problem = Problem(
    model=scm,
    terms={"sales": sales_media, "signups": signups_media},
    dates=initial.indexes["date"],
    objective=profit,
    bounds={"spend": (0.0, channel_caps)},
    parameters={"budget": budget, "weekly_cap": weekly_cap, "floor": floor},
    constraints=[
        {
            "type": "eq",
            "fun": lambda evaluate, u: (
                u["spend"].sum("date").sum("channel")
                - evaluate.parameter("budget")
            ),
        },
        {
            "type": "ineq",
            "fun": lambda evaluate, u: (
                evaluate.parameter("weekly_cap") - u["spend"].sum("channel")
            ),
        },
        {"type": "ineq", "fun": late_sales_floor},
    ],
)

problem.decisions  # NaN template with inferred input names, dimensions and labels.
baseline_draws = problem.evaluate(initial)  # Scores a plan; no optimizer runs.
result = problem.solve(x0=initial)
optimized_draws = problem.evaluate(result.allocation)

# Reuse the same graphs and callbacks with a different budget and start.
smaller = problem.solve(
    x0=result.allocation,
    parameters={"budget": 0.8 * budget},
    options={"maxiter": 1000},
)
```

Construction binds the fitted terms and compiles the numerical graphs, but does not select a starting plan or run a solver.
Each solve requires `x0`; it can use different initial values and declared parameter overrides without recompilation.
Automatic numerical scales are computed from that solve's initial point and remain fixed throughout that solve.
An override does not change the problem's default parameters or earlier results.
Changing the objective, constraints, dates, bounds, terms, history or carryover requires a new `Problem`.

- Decision inputs are inferred from the selected terms' data dependencies.
  A shared input is optimized once; unrelated equations and their inputs are not required.
  Selecting a full mean that also reads price requires bounds and plan values for both spend and price.
- Every decision requires an explicit bounds entry; use `(-float("inf"), float("inf"))` for an unbounded input.
  Unknown or missing entries are errors.
  Bounds accept scalars or labeled DataArrays over a subset of the input dimensions; equal bounds hold individual entries fixed.
  `x0` must be finite and lie inside the bounds.
- `problem.decisions` is a labeled NaN template, not an automatically chosen initial point.
  Fill every inferred input with finite values before evaluating or solving.
  A plan can be a Dataset or a mapping of names to DataArrays; variables, dimension sets, dates and labels must match exactly, with transposition and label reordering allowed.
  Plans are not implicitly broadcast.
  `evaluate` does not enforce bounds or constraints: it scores the supplied plan, not its feasibility.
- `u` contains original-unit named PyTensor arrays over decided periods, not NumPy arrays or sampled decisions.
  Callbacks run during problem construction, not on every solve or SciPy iteration.
- `evaluate(u)` returns each selected term separately over the decided periods plus any carryover tail.
  Posterior-dependent outputs retain the same joint `sample` dimension; parameter-free outputs do not gain a sample dimension.
  The caller writes every reduction, broadcast, risk adjustment and unit conversion explicitly.
  A separate output that explicitly declares `sample` alongside posterior-dependent outputs must use the fitted joint posterior sample labels in their paired order; arbitrary labels are never relabeled to draws.
- One expression may be supplied directly as `terms=expression`; its evaluator key is `"term"`.
- An equality residual must be zero; an inequality residual must be nonnegative.
  Vector residuals retain their coordinate lengths; reduce a sliced window explicitly before returning it.
  Provably constant, exactly satisfied equality rows are kept in the report but omitted from SLSQP to avoid singular redundant rows.
- Bounds and aligned decision or constraint scales reject complex values and inexact integer-to-`float64` conversions; exactly representable large integer endpoints remain supported.
  A fitted input's dimension set cannot change implicitly.

### Labeled constants and changing parameters

Use `evaluate.constant(...)` for a fixed unit conversion, and `evaluate.parameter(...)` for a declared value that may change between solves:

```python
def dollars(evaluate, u):
    return u["spend"] / evaluate.constant(spend_unit)


def budget_remaining(evaluate, u):
    return evaluate.parameter("budget") - dollars(evaluate, u).sum("date").sum("channel")
```

`spend_unit` is a DataArray with exactly the scenario channel labels, in any order.
This avoids positional conversions when channels use different units.
Parameters are finite scalars or labeled DataArrays declared in the constructor; overrides retain each parameter's dimensions and labels, with label reordering allowed.
Unknown parameter names and incompatible overrides are errors.
Nondate labels must match exactly for fitted model and new wrapper constants, bounds, explicit scales, callback constants and parameters.
Every named dimension of a selected deterministic DataArray constant must have one-dimensional coordinate labels; unlabeled positional factors are rejected.
Date labels in these arrays use the same normalization as scenario data, including string dates.
Callback constants and parameters can use the exact decided-date axis or the exact full scored-date axis; use the axis appropriate to the expression.

A stochastic `Equation` is not an optimization target, including when wrapped in a deterministic expression.
For a non-Normal likelihood, `mu` may not be the expected outcome; explicitly construct the deterministic expectation or other quantity you want to optimize.
Date-indexed posterior parameters may be used only on covered fitted dates, including labeled in-sample subsets; the optimizer does not simulate new latent trajectories.
Date-indexed deterministic DataArray constants retain their labeled fitted values rather than being reused by position.
Their labels must cover every requested decided and carryover date and, when supplied, every prepended history date; covered subsets are selected and reordered by label, while uncovered dates are rejected.
New deterministic wrappers may supply constants covering the full combined history, decision and carryover axis, in any label order.
Only constants that contribute to the selected deterministic graph are required; constants inside frozen parameter priors do not need scenario-date coverage.
Likewise, fitted random-variable draws do not require Data used only by their original prior recipe; those training inputs are not decision inputs unless another selected value graph reads them.
Admission checks likewise do not reopen a cached fitted coefficient's prior-only recipe; an actual selected `Equation` output or foreign random variable remains invalid.
Provably value-independent shape metadata is frozen before posterior conditioning, so a channel count or Fourier date count cannot turn into a posterior sample count.

## History, scoring windows, and carryover

History is optional, fixed and explicit; carryover is a separate scoring policy:

```python
problem = Problem(
    model=scm,
    terms={"sales": sales_media},
    dates=initial.indexes["date"],
    history=train[["spend"]].isel(date=slice(-2, None)),
    carryover=2,
    objective=lambda evaluate, u: (
        evaluate(u)["sales"].sum("date").sum("channel").mean("sample")
    ),
    bounds={"spend": (0.0, channel_caps)},
)
result = problem.solve(x0=initial)
scored = problem.evaluate(result.allocation)
```

Historical rows are prepended only to date-indexed inputs and excluded from both allocation and evaluated outputs.
Choose sufficient measured history for the selected transformations; omitting it starts the scenario independently.
For temporal terms, decided dates must follow the fitted cadence even when no history is supplied.
Any supplied history must immediately precede the decided dates; any tail must immediately follow them.

Integer `carryover=N` appends `N` fixed zero rows to every dated input at the fitted cadence.
Those rows contribute to evaluated outputs and the objective, but never appear in `u`, `problem.decisions` or `result.allocation`.
Static decision inputs are not padded.
No tail is added by default.

Zero is not appropriate for every input: a price inside `log(price)` must remain positive.
For a selected graph with both spend and price, pass an explicit tail Dataset as `carryover` instead:

```python
tail_dates = pd.date_range(initial.indexes["date"][-1], periods=3, freq="W-MON")[1:]
fixed_tail = xr.Dataset(
    {
        "spend": xr.zeros_like(initial["spend"].isel(date=-1, drop=True)).expand_dims(
            date=tail_dates
        ),
        "price": xr.DataArray(
            [10.0, 10.0], dims="date", coords={"date": tail_dates}
        ),
    }
)
# Pass carryover=fixed_tail when constructing that model's Problem.
```

The Dataset must supply exactly the dated decision inputs, their fitted nondate labels and the consecutive tail dates.
These values are fixed snapshots, not extra decisions.
The optimizer does not infer a special media or price policy from an input's name.

Use `evaluate.decided(output)` before a week-by-week calculation combining model outputs with `u`:

```python
def weekly_sales_cap(evaluate, u):
    sales = evaluate.decided(evaluate(u)["sales"])
    return evaluate.parameter("weekly_sales_cap") - sales.sum("channel").mean("sample")
```

Without this selection, decided and full scored dates are different axes and cannot be multiplied or subtracted directly, even when a singleton decision could otherwise broadcast.
Call `evaluate.decided` on the full ordered output before selecting or permuting a smaller scoring window.
For a full-horizon objective, keep the full output and explicitly reduce its dates, as in the examples above.

Selected dated outputs must retain a provable full date axis in its original order, even without history; evaluated values are never silently relabeled after a permutation or subset.
Without history, scalar date reductions and date-reduced scalar factors are supported.
With explicit history, targets that reduce dates before history can be removed are rejected; select a date-indexed term and write date selection and reduction inside the objective or constraints instead.
A date-reduced scalar factor cannot bypass the history rule by broadcasting back over another dated input.
A full Boolean date mask must be all true; integer identity indices may use equivalent negative positions.
Subsets and permutations are rejected even when their values resemble a positional identity after a dtype conversion.
Ordinary nondate channel indexing, scalar channel selection, and reductions remain supported, including channel means before temporal transformations.
Static decision inputs and date-reduced intermediates derived from dated decisions must not feed history-dependent transformations: changing those factors would rewrite measured history.
This guard also applies to explicit nonempty history for a static-only temporal input graph; history cannot silently become a cold start merely because no raw input has a date dimension.
Shape-only factors such as the channel-count denominator of a mean do not depend on decision values and cannot rewrite history; they are permitted.
Data-dependent cardinalities, such as the number of nonzero decision values, are not fixed shape factors and remain subject to the history value-dependency guard.
Fixed posterior factors and static decisions downstream of temporal transformations remain supported; reduce dates inside callbacks after the evaluator removes history.

## Fixed numerical scaling

SciPy SLSQP solves movable entries in fixed affine coordinates:

$$u = c + S z.$$

With finite, nondegenerate bounds, `c` is the lower bound and `S` is the bound range.
Otherwise `S` uses the initial magnitude.
Equal-bound entries are frozen before differentiation, so their units and gradients cannot destabilize movable decisions.
Fixed entries are projected to their original bound values, independently of their scaled coordinate representation.
Automatic objective and constraint divisors use initial derivatives with respect to movable scaled decisions, falling back to nonzero initial value magnitudes.
The transforms stay fixed throughout each solve; they do not normalize against the current decisions.

Supply explicit positive, finite scales for unbounded zero starts or connected expressions whose initial value and derivatives both vanish.
The chosen scales must also produce representable finite movable coordinates, preserve movable bound intervals, and keep normalized objective, derivative, and constraint values representable; overflow or underflow that erases a nonzero value raises `ValueError` rather than clipping values or substituting a different scale.
For example:

```python
problem = Problem(
    model=scm,
    terms=sales_media,
    dates=initial.indexes["date"],
    bounds={"spend": (0.0, float("inf"))},
    parameters={"budget": budget},
    objective=lambda evaluate, u: (
        evaluate(u)["term"].sum("date").sum("channel").mean("sample")
    ),
    scaling={"decisions": {"spend": spend_scale}, "objective": sales_scale},
    constraints=[
        {
            "type": "eq",
            "fun": lambda evaluate, u: (
                u["spend"].sum("date").sum("channel")
                - evaluate.parameter("budget")
            ),
            "scale": budget,
        }
    ],
)
result = problem.solve(x0=initial)
```

`scaling=None` disables automatic scaling; explicit per-constraint scales still apply.
Changing units preserves the mathematical problem under fixed invertible transforms, but does not guarantee global optimality or identical convergence for every flat or nonconvex objective.
Only SLSQP is supported; `solve(..., options=...)` forwards its solver options, defaulting to `maxiter=500` and `ftol=1e-9` in scaled coordinates.

## Reading the result

`OptimizationResult` provides:

| Field | Meaning |
|---|---|
| `allocation` | Labeled decided-period Dataset in original units, without history or tail |
| `objective` | Maximized objective in original units |
| `constraints` | Original-unit labeled residuals, in declaration order |
| `decision_centers`, `decision_scales` | Fixed affine transformation, labeled by input |
| `objective_scale`, `constraint_scales` | Reported positive objective and row divisors |
| `scipy` | Untouched SciPy result in scaled coordinates |
| `feasible` | Independent decoded feasibility check |
| `max_bound_violation`, `max_constraint_violation` | Dimensionless maximum violations after division by reported scales |

Final objective values and constraint residuals are evaluated from the captured original-unit callback expressions at the actual decoded allocation, independently of the solver's scaled-coordinate values.
Callback operation order is preserved during this replay; callbacks are not called again.
Feasibility uses these reported residuals.

Check both `result.scipy.success` and `result.feasible`.
`feasibility_tol=1e-6` is separate from SLSQP's stopping tolerance and applies to residuals divided by their reported scales.
SciPy success without decoded feasibility emits a warning; the optimizer never rewrites SciPy's success flag.
Solver failures remain available for diagnosis.
A problem with all entries fixed is evaluated without running SLSQP and reports whether the fixed allocation is feasible.

## Experimental API change record

- Renamed the experimental support model from `GAM` to `SCM` (Structural Causal Model), with no compatibility alias. The serialized format is now `pymc_marketing.experimental.SCM/1`; older GAM stores must be recreated with `SCM`, refitted, and saved again. Causal interpretation depends on the specified assumptions; prediction does not automatically identify causal effects or perform interventions or counterfactual inference.
- Moved graph-first modeling and term-first optimization to `pymc_marketing.experimental.terms`, outside the MMM namespace. Stores using former MMM graph-node identifiers must likewise be recreated, refitted, and saved again; `check=False` does not migrate them.
- Replaced the synchronous `optimize` function with public `Problem`, `evaluate` and `solve`; retained `OptimizationResult` and original-unit reporting.
- Infer decisions from selected value graphs, require explicit bounds, separate construction from the required initial plan, and reuse compiled graphs across parameter updates and starting points.
- Separate decided dates, fixed history and scored carryover; support explicit nonzero tail data and the `evaluate.decided` helper.
- Validate temporal cadence without requiring history and preserve exact labeled plan and parameter alignment.
- Replaced prototype posterior-name matching with actual fitted stochastic identities and labeled joint draws; fitted state is never assigned or mutated.
- Added behavioral and derivative checks, exact shared-posterior unit sweeps, and true unit-refit comparisons against sampling variability.
- Reject history targets with unproven date order and decision-value reductions upstream of temporal terms, while supporting safe shape-only paths and identity indices.
- Preserve labeled deterministic constant provenance, require date coverage, and select covered date subsets in scenario order without rebuilding fitted stochastic terms.
- Normalize string-date bounds, scales, and callback constants consistently and retain exact nondate label alignment.
- Keep integer residual feasibility arithmetic in floating point, preserve mixed-dtype increment dependencies, and validate explicit sample labels against the paired joint posterior.
- Preserve atomic tuple-valued fitted coordinates during scenario alignment instead of implicitly promoting them to MultiIndexes.
- Keep authoritative fitted and fresh input identities when raw data names resemble internal shared-variable names.
- Reject pre-reduced dated posterior and constant outputs with explicit history, including mixed outputs that obtain their date axis from another selected term.
- Construct numeric input graphs with canonical floating data while preserving cached fitted mathematics, stochastic identities, raw observations, and explicit casts.
- Reserve raw dataset names when allocating internal containers, preserving custom shared-term aliases and equation-order-independent input ownership.
- Trace dated values through scalar and nondate operand branches, retaining a shared conservative shape-only proof instead of accepting rebroadcast history reductions.
- Preserve original axes in posterior shape metadata, retain constant-index shape proofs, and exclude frozen prior-only Data from decision closure.
- Validate canonical numeric input integrity before scenario, history, and numeric-evaluator conversion, retain declared custom container ownership, and preserve Fourier shape boundaries through dated-constant rebinding.
- Preserve the original feasible set by rejecting inexact integer bounds, including Python-sized integers in object arrays, validate all retained shared-container state, and keep fitted hierarchical prior recipes outside selected-value admission.
