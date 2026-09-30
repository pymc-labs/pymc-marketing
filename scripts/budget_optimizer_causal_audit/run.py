"""Reproducible, small decision experiments for the MMM budget optimizer.

Run with the repository's Python environment::

    python scripts/budget_optimizer_causal_audit/run.py

The analytic functions below are the data-generating mechanisms. They provide
an independent oracle; the PyMC graphs express the model given to the optimizer.
The examples are intentionally small enough to grid-search the decision space.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor.math as ptxm
import xarray as xr

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

BUDGET = 10.0  # Per-period budget level, hence 20 across a two-period window.
GRID = np.linspace(0, BUDGET, 1001)


def make_optimizer(
    beta: tuple[float, float],
    *,
    n_dates: int = 2,
    baseline: tuple[float, ...] | None = None,
    date_weights: np.ndarray | None = None,
    mediator_gain: float = 0.0,
    log_link: bool = False,
    linear_media: bool = False,
    time_pattern: np.ndarray | None = None,
    price_response=None,
    response_variable: str = "total_media_contribution_original_scale",
) -> BudgetOptimizer:
    """Give BudgetOptimizer a known, deterministic one-draw posterior."""
    baseline_array = np.zeros(n_dates) if baseline is None else np.asarray(baseline)
    weight_array = (
        np.ones((n_dates, 2))
        if date_weights is None
        else np.asarray(date_weights, dtype=float)
    )
    coords = {"date": np.arange(n_dates), "channel": ["A", "B"]}
    with pm.Model(coords=coords) as model:
        channel_data = pmd.Data(
            "channel_data", np.zeros((n_dates, 2)), dims=("date", "channel")
        )
        beta_rv = pmd.Normal("beta", 0, 1, dims="channel")
        weights = pmd.Data(
            "date_channel_weight", weight_array, dims=("date", "channel")
        )
        baseline_data = pmd.Data("baseline_data", baseline_array, dims="date")
        mediator_selector = pmd.Data(
            "mediator_selector", np.array([1.0, 0.0]), dims="channel"
        )
        media_input = channel_data if linear_media else ptxm.log1p(channel_data)
        media = pmd.Deterministic(
            "channel_contribution",
            media_input * beta_rv * weights,
            dims=("date", "channel"),
        ).sum(dim="channel")
        mediator = mediator_gain * (ptxm.log1p(channel_data) * mediator_selector).sum(
            dim="channel"
        )
        if log_link:
            full = ptxm.exp(baseline_data + media + mediator)
            media_only = ptxm.exp(baseline_data + media) - ptxm.exp(baseline_data)
        else:
            full = baseline_data + media + mediator
            media_only = media
        pmd.Deterministic(
            "total_media_contribution_original_scale", media_only.sum(), dims=()
        )
        pmd.Deterministic("total_response_original_scale", full.sum(), dims=())

    posterior = xr.Dataset(
        {
            "beta": xr.DataArray(
                np.asarray(beta, dtype=float)[None, None, :],
                dims=("chain", "draw", "channel"),
                coords={"chain": [0], "draw": [0], "channel": coords["channel"]},
            )
        }
    )
    pattern = None
    if time_pattern is not None:
        pattern = xr.DataArray(
            np.asarray(time_pattern, dtype=float),
            dims=("date", "channel"),
            coords=coords,
        )
    optimizer_kwargs = (
        {"price_response": price_response} if price_response is not None else {}
    )
    return BudgetOptimizer(
        model=model,
        idata=xr.DataTree.from_dict({"/posterior": posterior}),
        num_periods=n_dates,
        response_variable=response_variable,
        budget_distribution_over_period=pattern,
        budgets_to_optimize=xr.DataArray(
            [True, True], dims="channel", coords={"channel": coords["channel"]}
        ),
        **optimizer_kwargs,
    )


def solve(optimizer: BudgetOptimizer) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No budget bounds provided")
        result = optimizer.allocate_budget(total_budget=BUDGET)
    return tuple(float(v) for v in result.budgets.values)


def optimum_on_grid(value) -> tuple[tuple[float, float], float]:
    values = np.array([value(x, BUDGET - x) for x in GRID])
    i = int(np.argmax(values))
    return (float(GRID[i]), float(BUDGET - GRID[i])), float(values[i])


def case_additive_seasonality() -> dict:
    """A correctly specified additive seasonal baseline should cancel."""
    beta = (0.8, 0.5)
    baseline = (2.0, 20.0)
    optimizer = make_optimizer(beta, baseline=baseline)
    chosen = solve(optimizer)

    def truth(a, b):
        return sum(baseline) + 2 * (beta[0] * np.log1p(a) + beta[1] * np.log1p(b))

    oracle, best = optimum_on_grid(truth)
    return result_row("additive_seasonality", chosen, oracle, truth, best)


def case_log_link_future_control() -> dict:
    """An incorrect zero forecast of a control changes the best channel."""
    beta = (0.6, 0.4)
    actual_baseline = (0.0, float(np.log(4)))
    pattern = np.array([[1.0, 0.0], [0.0, 1.0]])
    assumed_zero = make_optimizer(beta, log_link=True, time_pattern=pattern)
    supplied_future = make_optimizer(
        beta, baseline=actual_baseline, log_link=True, time_pattern=pattern
    )
    chosen = solve(assumed_zero)
    corrected = solve(supplied_future)

    def truth(a, b):
        return (1 + 2 * a) ** beta[0] + 4 * (1 + 2 * b) ** beta[1]

    oracle, best = optimum_on_grid(truth)
    row = result_row("log_link_future_control", chosen, oracle, truth, best)
    row["corrected_allocation"] = [round(v, 4) for v in corrected]
    row["corrected_regret"] = round(max(0.0, best - truth(*corrected)), 6)
    return row


def case_mediator() -> dict:
    """Default contribution misses an A-driven mediator; full response includes it."""
    beta = (0.25, 0.6)
    mediator_gain = 0.8
    direct = make_optimizer(beta, mediator_gain=mediator_gain)
    full = make_optimizer(
        beta,
        mediator_gain=mediator_gain,
        response_variable="total_response_original_scale",
    )
    chosen = solve(direct)
    corrected = solve(full)

    def truth(a, b):
        return 2 * ((beta[0] + mediator_gain) * np.log1p(a) + beta[1] * np.log1p(b))

    oracle, best = optimum_on_grid(truth)
    row = result_row("media_driven_mediator", chosen, oracle, truth, best)
    row["corrected_allocation"] = [round(v, 4) for v in corrected]
    row["corrected_regret"] = round(max(0.0, best - truth(*corrected)), 6)
    return row


def case_fixed_timing() -> dict:
    """A uniform schedule excludes the best week-level plan even with a true model."""
    beta = (1.0, 0.5)
    weights = np.array([[1.0, 1.0], [5.0, 5.0]])
    optimizer = make_optimizer(beta, date_weights=weights)
    chosen = solve(optimizer)

    def truth_by_schedule(spend):
        return float(np.sum(weights * np.asarray(beta) * np.log1p(spend)))

    uniform = np.tile(np.asarray(chosen), (2, 1))
    # Independent analytic oracle: the KKT condition for w*log(1+s) gives
    # s=max(w/lambda-1, 0). Bisect lambda to spend exactly the 20-unit budget.
    marginal_weights = weights * np.asarray(beta)
    lo, hi = 1e-12, float(marginal_weights.max())
    for _ in range(100):
        middle = (lo + hi) / 2
        if np.maximum(marginal_weights / middle - 1, 0).sum() > 2 * BUDGET:
            lo = middle
        else:
            hi = middle
    flexible = np.maximum(marginal_weights / hi - 1, 0)
    return {
        "case": "fixed_timing",
        "optimizer_allocation_per_period": list(chosen),
        "optimizer_schedule": uniform.round(4).tolist(),
        "flexible_oracle_schedule": flexible.round(4).tolist(),
        "optimizer_truth": round(truth_by_schedule(uniform), 6),
        "oracle_truth": round(truth_by_schedule(flexible), 6),
        "regret": round(truth_by_schedule(flexible) - truth_by_schedule(uniform), 6),
        "comparison": "Broader time-by-channel decision set; not an optimizer bug",
    }


def case_unobserved_demand() -> dict:
    """Season-adjusted observational coefficients can still be noncausal."""
    rng = np.random.default_rng(734)
    n = 1000
    season = np.sin(2 * np.pi * np.arange(n) / 52)
    demand_shock = rng.normal(size=n)
    spend_a = np.maximum(
        0.01, 5 + 2 * season + 2 * demand_shock + rng.normal(0, 0.4, n)
    )
    spend_b = np.maximum(0.01, 5 + rng.normal(0, 2, n))
    true_beta = (0.25, 0.55)
    sales = (
        20
        + true_beta[0] * spend_a
        + true_beta[1] * spend_b
        + 3 * season
        + 2 * demand_shock
        + rng.normal(0, 0.5, n)
    )
    design = np.column_stack([np.ones(n), spend_a, spend_b, season])
    coefficients = np.linalg.lstsq(design, sales, rcond=None)[0]
    fitted_beta = tuple(float(v) for v in coefficients[1:3])
    fitted_sales = design @ coefficients
    r_squared = 1 - np.sum((sales - fitted_sales) ** 2) / np.sum(
        (sales - sales.mean()) ** 2
    )
    chosen = solve(make_optimizer(fitted_beta, linear_media=True))

    def truth(a, b):
        return 2 * (true_beta[0] * a + true_beta[1] * b)

    oracle, best = optimum_on_grid(truth)
    row = result_row("unobserved_demand", chosen, oracle, truth, best)
    row["true_beta"] = list(true_beta)
    row["season_adjusted_fitted_beta"] = [round(v, 6) for v in fitted_beta]
    row["season_adjusted_fit_r_squared"] = round(float(r_squared), 6)
    row["fit_note"] = (
        "OLS establishes the causal identification failure; the fitted coefficients "
        "are then passed as posterior draws to the real BudgetOptimizer."
    )
    return row


def case_spend_dependent_price() -> dict:
    """The fixed unit price ranks allocations using the wrong delivery curve."""
    beta = (2.0, 1.0)
    chosen = solve(make_optimizer(beta, n_dates=1))

    def truth(a, b):
        # Channel A price is p(s)=sqrt(s/2), so delivery is sqrt(2s).
        # Channel B costs one unit of money per delivery unit.
        return beta[0] * np.log1p(np.sqrt(2 * a)) + beta[1] * np.log1p(b)

    oracle, best = optimum_on_grid(truth)
    row = result_row("spend_dependent_price", chosen, oracle, truth, best)
    row["price_map"] = "A: delivery=sqrt(2*spend); B: delivery=spend"
    row["status"] = "Fixed-price path on this checkout; PR #3045 proposes a power map"
    return row


def case_standard_mmm_future_control_default() -> dict:
    """Inspect the future data in MMM.create_optimization_model directly."""
    n = 20
    X = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="W-MON"),
            "A": np.linspace(1, 3, n),
            "B": np.linspace(2, 4, n),
            "promo": np.linspace(0.1, 0.9, n),
        }
    )
    y = pd.Series(10 + X["A"] + X["B"] + X["promo"], name="y")
    mmm = MMM(
        date_column="date",
        channel_columns=["A", "B"],
        control_columns=["promo"],
        adstock=GeometricAdstock(l_max=1),
        saturation=LogisticSaturation(),
    )
    mmm.build_model(X, y)
    future_model = mmm.create_optimization_model("2024-05-20", "2024-06-03")
    controls = future_model["control_data"].get_value().ravel()
    return {
        "case": "standard_mmm_future_control_default",
        "historical_carry_in_control": float(controls[0]),
        "future_and_tail_controls": controls[1:].tolist(),
        "status": "API inspection; no optimization or causal regret claimed",
    }


def result_row(name, chosen, oracle, truth, best) -> dict:
    return {
        "case": name,
        "optimizer_allocation": [round(x, 4) for x in chosen],
        "oracle_allocation": [round(x, 4) for x in oracle],
        "optimizer_truth": round(truth(*chosen), 6),
        "oracle_truth": round(best, 6),
        "regret": round(max(0.0, best - truth(*chosen)), 6),
    }


def main() -> None:
    rows = [
        case_additive_seasonality(),
        case_log_link_future_control(),
        case_mediator(),
        case_fixed_timing(),
        case_unobserved_demand(),
        case_spend_dependent_price(),
        case_standard_mmm_future_control_default(),
    ]
    by_case = {row["case"]: row for row in rows}
    checks = {
        "additive seasonality": by_case["additive_seasonality"]["regret"] < 0.001,
        "wrong future control": by_case["log_link_future_control"]["regret"] > 1,
        "supplied future control": by_case["log_link_future_control"][
            "corrected_regret"
        ]
        < 0.001,
        "omitted mediator": by_case["media_driven_mediator"]["regret"] > 0.5,
        "full response": by_case["media_driven_mediator"]["corrected_regret"] < 0.001,
        "fixed schedule": by_case["fixed_timing"]["regret"] > 1,
        "unobserved demand": by_case["unobserved_demand"]["regret"] > 5,
        "fixed price": by_case["spend_dependent_price"]["regret"] > 0.1,
        "future controls zero": by_case["standard_mmm_future_control_default"][
            "future_and_tail_controls"
        ]
        == [0.0] * 4,
    }
    for name, passed in checks.items():
        if not passed:
            raise RuntimeError(f"Experiment check failed: {name}")
    destination = Path(__file__).with_name("results.json")
    destination.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
