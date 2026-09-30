# Budget optimizer causal and decision audit

This folder contains deliberately small, reproducible counterexamples. They
distinguish four questions: whether the media effect is causal, whether future
inputs describe the decision period, whether the response variable contains all
spend-driven paths, and whether the decision space includes the desired plan.
The examples are *stress tests*, not estimates of how often a problem occurs in
real MMMs.

## Reproduce

From the repository root with its Python dependencies installed:

```bash
python scripts/budget_optimizer_causal_audit/run.py
```

The script checks each expected qualitative result and writes `results.json`.
The recorded run used Python 3.12.12, PyMC 6.3.2, SciPy 1.15.3, and repository
commit `814ad07467c6a983a2ab44deae3014da49ba68b8` (before the audit files).

The proposed price map is checked separately against PR #3045 commit
`0faa14cc2c672517e9619eefaa049c3a9cd502ac`:

```bash
git fetch origin pull/3045/head:refs/remotes/origin/pr-3045
git worktree add --detach /tmp/pymc-marketing-pr3045 origin/pr-3045
PYTHONPATH=/tmp/pymc-marketing-pr3045 python \
  scripts/budget_optimizer_causal_audit/run_pr3045.py
```

This writes `pr3045_results.json`. The script opts into
`assume_delivery_units=True` because its custom known model consumes delivery
units, and pins the price reference at two monetary units. The result only
establishes behavior at the listed PR commit.

## Method

For model and optimizer behavior, `run.py` constructs a two-channel PyMC graph
and gives the real `BudgetOptimizer` a one-draw posterior fixed at known
coefficients. It compares the allocation with an independent analytic or dense
grid optimum. The grid step is 0.01. The budget variable is a **per-period**
level of 10, so the two-period cases spend 20 over the window. Regret is the
true response at the feasible oracle plan minus that at the returned plan.

The confounding case first generates 1,000 historical observations and fits an
ordinary linear regression with the observed season covariate. It passes the
fitted coefficients into the same optimizer graph. This establishes the causal
identification failure; it is **not** a fitted `MMM` end-to-end test. The
standard `MMM.create_optimization_model` is exercised directly in the final
case to inspect its future control inputs.

| Case | Optimizer A/B | Oracle A/B | Regret | Reading |
| --- | ---: | ---: | ---: | --- |
| Correct additive seasonality | 6.385 / 3.615 | 6.38 / 3.62 | 0 | Works as intended |
| Future control under log link | 8.724 / 1.276 | 2.34 / 7.66 | 2.668 | Wrong future scenario; supplying the control recovers the oracle |
| A-driven mediator excluded from objective | 2.529 / 7.471 | 6.64 / 3.36 | 0.825 | `total_response_original_scale` recovers the oracle |
| Uniform spending across weeks | 7 / 3 each week | See schedule below | 2.183 | Restricted decision space, not a wrong solution within that space |
| Unobserved demand drives A spend | 10 / 0 | 0 / 10 | 6 | Causal identification; season adjustment and high fit do not suffice |
| Price rises with A spend | 7 / 3 | 4.73 / 5.27 | 0.147 | Addressed by PR #3045 in this example |

### Causal structures and mechanisms

**Additive seasonality (works).** The true response is
`baseline[t] + 0.8 log(1+A[t]) + 0.5 log(1+B[t])`. The baseline is 2 in week 1
and 20 in week 2. Since it is additive and unaffected by spend, it cancels
from the allocation comparison.

**Log-link future control.** A channel A spends only in week 1 and B only in
week 2 through a fixed `budget_distribution_over_period`. The true response is
`(1+2A)^0.6 + 4(1+2B)^0.4`; the factor 4 is an upstream week-2 control. The
zero-control scenario chooses 8.724 / 1.276. Supplying the true control
chooses 2.339 / 7.661, matching the oracle. A separate direct `MMM` API check
shows `create_optimization_model` keeps the observed carry-in control at 0.9
and sets four future/carry-over values to zero. Zero can be a valid scenario;
the risk is treating it as an implicit forecast.

**Media-driven mediator.** Channel A has direct coefficient 0.25 and an
additional path `A -> mediator -> outcome` with coefficient 0.8; B has direct
coefficient 0.6. The default channel-only response sees `0.25 log(1+A)` and
prefers B. A full-response objective sees `1.05 log(1+A)` and matches the
oracle. This custom graph tests direct `BudgetOptimizer` construction.
`MMM.budget_optimizer` has a warning when it detects a media-dependent
`MuEffect`, so the standard MMM entry point has a partial safeguard.

**Time allocation.** The true date-by-channel response weights are
`[[1, 0.5], [5, 2.5]]` for weeks 1 and 2. The optimizer chooses 7 / 3 in each
week because the default decision repeats a channel level across dates. The
analytic water-filling oracle for the *larger* week-by-channel decision space
spends `[[1.667, 0.333], [12.333, 5.667]]`. These are different feasible sets.
A caller can supply a fixed weekly pattern, but this API does not choose the
pattern.

**Confounding despite seasonal adjustment.** The simulation has
`season -> A spend`, `season -> sales`, `demand shock -> A spend`, and
`demand shock -> sales`. True media coefficients are A=0.25 and B=0.55. A
regression adjusted for season estimates A=1.213 and B=0.548 with R²=0.969;
the optimizer chooses A. The hidden demand shock is the unblocked backdoor
path. No objective or solver change can make that fitted association causal.

**Spend-dependent price.** The known model consumes delivery units. At the
planned spend, A delivery is `sqrt(2*spend)` and B delivery is `spend`.
The fixed-price path feeds money as delivery and chooses 7 / 3. The oracle
chooses 4.73 / 5.27. `PowerPriceResponse(elasticity={"A": 0.5, "B": 0},
reference_spend=2)` in PR #3045 chooses 4.731 / 5.269 at the pinned commit.

## Interpretation

- **Handled with current configuration:** a correctly specified additive
  baseline; a media-driven effect included in a suitable response graph and
  full-response objective; a correctly supplied future exogenous scenario; a
  specified feasible weekly spending pattern.
- **Potential API work:** make future-driver assumptions visible and easy to
  supply as scenarios; optionally make week-by-channel spend a decision rather
  than a fixed pattern; clarify the chosen objective and its causal scope.
- **Outside the optimizer:** identifying intervention effects when spending
  follows hidden demand, and establishing the behavior of downstream mediators
  or effect modifiers. These require design assumptions and often experiments.

This audit does not claim that every use of the current optimizer is wrong, or
that its numerical solver failed any case. Each example isolates a condition
under which a recommendation can differ from the desired causal optimum.
Date-level decisions are already tracked in issue #2799, spend-dependent
prices in issue #3036 and PR #3045, and driver composition in issue #3088.
