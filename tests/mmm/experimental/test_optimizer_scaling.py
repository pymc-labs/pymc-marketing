#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Shared-posterior unit invariance and actual refits against MCMC control noise."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pymc.dims as pmd
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.experimental import (
    GAM,
    Data,
    Equation,
    OptimizationResult,
    optimize,
)
from pymc_marketing.terms import Parameter, Transform

CHANNELS = ["tv", "search"]
SAMPLE_KWARGS = {
    "draws": 60,
    "tune": 80,
    "chains": 1,
    "cores": 1,
    "progressbar": False,
    "compute_convergence_checks": False,
}


@dataclass(frozen=True)
class Units:
    name: str
    inputs: tuple[float, float]
    sales: float
    leads: float

    def spend(self) -> xr.DataArray:
        return xr.DataArray(
            np.asarray(self.inputs), dims="channel", coords={"channel": CHANNELS}
        )


REFERENCE = Units("reference", (1.0, 1.0), 1.0, 1.0)
UNIT_CASES = [
    REFERENCE,
    Units("inputs-1e3-outputs-1e-3", (1e3, 1e3), 1e-3, 1e-3),
    Units("inputs-1e6-outputs-1e-6", (1e6, 1e6), 1e-6, 1e-6),
    Units("inputs-1e9-outputs-1e-6", (1e9, 1e9), 1e-6, 1e-6),
    Units("inputs-1e-6-outputs-1e6", (1e-6, 1e-6), 1e6, 1e6),
    Units("inputs-1e-6-outputs-1e9", (1e-6, 1e-6), 1e9, 1e9),
    Units("cents-thousands-and-leads-thousands", (1e2, 1e-3), 1.0, 1e-3),
    Units("mixed-channel-and-outcome-units", (1e6, 1e-6), 1e-3, 1e3),
]
REFIT_CASES = [
    Units(
        "refit-inputs-2pow20-outputs-2powminus20",
        (2.0**20, 2.0**20),
        2.0**-20,
        2.0**-20,
    ),
    Units(
        "refit-inputs-2powminus20-outputs-2pow20",
        (2.0**-20, 2.0**-20),
        2.0**20,
        2.0**20,
    ),
]


@dataclass
class FittedUnits:
    model: GAM
    spend: Data
    sales_beta: Parameter
    leads_beta: Parameter


@dataclass
class ReferenceProblem:
    train: xr.Dataset
    initial: xr.DataArray
    budget: float
    weekly_cap: float
    caps: xr.DataArray
    sales_floor: float


@dataclass
class Setup:
    fitted: FittedUnits
    reference: ReferenceProblem
    baseline: OptimizationResult


def _training() -> xr.Dataset:
    rng = np.random.default_rng(103)
    dates = pd.date_range("2025-01-06", periods=20, freq="W-MON")
    spend = rng.uniform(0.2, 1.5, (len(dates), len(CHANNELS))) * np.array([120.0, 80.0])
    normalized = spend / spend.max(axis=0)
    sales = 1000 * (
        (np.log1p(normalized) * np.array([1.8, 0.9])).sum(axis=1)
        + rng.normal(0, 0.02, len(dates))
    )
    leads = 20 * (
        (np.tanh(normalized) * np.array([0.6, 1.4])).sum(axis=1)
        + rng.normal(0, 0.02, len(dates))
    )
    return xr.Dataset(
        {
            "spend": (("date", "channel"), spend),
            "sales": ("date", sales),
            "leads": ("date", leads),
        },
        coords={"date": dates, "channel": CHANNELS},
    )


def _in_units(data: xr.Dataset, units: Units) -> xr.Dataset:
    return data.assign(
        spend=data["spend"] * units.spend(),
        sales=data["sales"] * units.sales,
        leads=data["leads"] * units.leads,
    )


def _terms(
    spend: Data,
    sales_beta: Parameter,
    leads_beta: Parameter,
    train: xr.Dataset,
) -> dict[str, Any]:
    # These wrappers keep the actual fitted Parameter identities and joint posterior.
    # Only fixed, training-derived normalization and explicit output units change.
    normalized = spend * (1.0 / train["spend"].max("date"))
    return {
        "sales": Transform(normalized, pmd.math.log1p)
        * sales_beta
        * float(train["sales"].max()),
        "leads": Transform(normalized, pmd.math.tanh)
        * leads_beta
        * float(train["leads"].max()),
    }


def _fit(train: xr.Dataset, seed: int) -> FittedUnits:
    spend = Data("spend")
    sales_beta = Parameter(
        "sales_beta", Prior("LogNormal", mu=0, sigma=0.7, dims="channel")
    )
    leads_beta = Parameter(
        "leads_beta", Prior("LogNormal", mu=0, sigma=0.7, dims="channel")
    )
    terms = _terms(spend, sales_beta, leads_beta, train)
    model = GAM(
        Equation(
            observed="sales",
            mu=terms["sales"].sum("channel").named("sales_mean"),
            likelihood=Prior("Normal", sigma=0.025 * float(train["sales"].max())),
        ),
        Equation(
            observed="leads",
            mu=terms["leads"].sum("channel").named("leads_mean"),
            likelihood=Prior("Normal", sigma=0.025 * float(train["leads"].max())),
        ),
    )
    model.fit(train, random_seed=seed, **SAMPLE_KWARGS)
    return FittedUnits(model, spend, sales_beta, leads_beta)


def _posterior(fitted: FittedUnits) -> xr.Dataset:
    assert fitted.model.idata is not None
    return fitted.model.idata["posterior"].to_dataset()


def _numpy_outputs(
    fitted: FittedUnits, train: xr.Dataset, allocation: xr.DataArray
) -> tuple[np.ndarray, np.ndarray]:
    posterior = _posterior(fitted)
    normalized = (
        allocation.transpose("date", "channel").values
        / train["spend"].max("date").values
    )
    sales_beta = posterior["sales_beta"].transpose("chain", "draw", "channel").values
    leads_beta = posterior["leads_beta"].transpose("chain", "draw", "channel").values
    sales = (
        sales_beta[:, :, None, :]
        * np.log1p(normalized)[None, None, :, :]
        * float(train["sales"].max())
    )
    leads = (
        leads_beta[:, :, None, :]
        * np.tanh(normalized)[None, None, :, :]
        * float(train["leads"].max())
    )
    return sales, leads


def _solve(
    fitted: FittedUnits, reference: ReferenceProblem, units: Units
) -> OptimizationResult:
    terms = _terms(
        fitted.spend,
        fitted.sales_beta,
        fitted.leads_beta,
        _in_units(reference.train, units),
    )

    def objective(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        out = evaluate(u)
        value = 0.3 * out["sales"] + (20.0 * units.sales / units.leads) * out["leads"]
        return value.sum("date").sum("channel").mean("sample")

    def dollars(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        return u["spend"] / evaluate.constant(units.spend())

    return optimize(
        model=fitted.model,
        terms=terms,
        inputs=[fitted.spend],
        data=xr.Dataset({"spend": reference.initial * units.spend()}),
        objective=objective,
        bounds={"spend": (0.0, reference.caps * units.spend())},
        constraints=[
            {
                "type": "eq",
                "fun": lambda e, u: (
                    dollars(e, u).sum("date").sum("channel") - reference.budget
                ),
            },
            {
                "type": "ineq",
                "fun": lambda e, u: reference.weekly_cap - dollars(e, u).sum("channel"),
            },
            {
                "type": "ineq",
                "fun": lambda e, u: (
                    e(u)["sales"]
                    .isel(date=slice(-2, None))
                    .sum("date")
                    .sum("channel")
                    .mean("sample")
                    - reference.sales_floor * units.sales
                ),
            },
        ],
        options={"ftol": 1e-12, "maxiter": 200},
        feasibility_tol=1e-7,
    )


def _check_result(
    fitted: FittedUnits,
    reference: ReferenceProblem,
    units: Units,
    result: OptimizationResult,
) -> None:
    allocation = result.allocation["spend"]
    dollars = allocation / units.spend()
    sales, leads = _numpy_outputs(fitted, _in_units(reference.train, units), allocation)
    expected_objective = (
        (0.3 * sales + 20.0 * units.sales / units.leads * leads).sum(axis=(2, 3)).mean()
    )
    expected_residuals = [
        xr.DataArray(float(dollars.sum()) - reference.budget),
        reference.weekly_cap - dollars.sum("channel"),
        xr.DataArray(
            sales[:, :, -2:].sum(axis=(2, 3)).mean()
            - reference.sales_floor * units.sales
        ),
    ]
    assert result.scipy.success
    assert result.feasible
    assert np.isfinite(allocation.values).all()
    assert np.isfinite(result.objective)
    assert result.objective_scale > 0 and np.isfinite(result.objective_scale)
    assert np.isfinite(result.decision_scales["spend"].values).all()
    assert (result.decision_scales["spend"] > 0).all()
    np.testing.assert_allclose(result.objective, expected_objective, rtol=1e-10)
    np.testing.assert_allclose(
        result.scipy.fun, -result.objective / result.objective_scale, rtol=1e-10
    )
    residual_units = [
        reference.budget,
        reference.weekly_cap,
        reference.sales_floor * units.sales,
    ]
    for actual, expected, scale, unit in zip(
        result.constraints,
        expected_residuals,
        result.constraint_scales,
        residual_units,
        strict=True,
    ):
        assert np.isfinite(actual.values).all()
        assert np.isfinite(scale.values).all() and (scale > 0).all()
        xr.testing.assert_allclose(
            actual / unit, expected / unit, rtol=1e-10, atol=1e-12
        )
    # Feasibility is recomputed from the allocation, not trusted from report flags.
    assert float((-dollars / reference.caps).clip(min=0).max()) <= 1e-8
    assert (
        float(((dollars - reference.caps) / reference.caps).clip(min=0).max()) <= 1e-8
    )
    assert abs(float(expected_residuals[0])) / reference.budget <= 1e-8
    assert float(expected_residuals[1].min()) / reference.weekly_cap >= -1e-8
    assert float(expected_residuals[2]) / (reference.sales_floor * units.sales) >= -1e-8
    violation = max(
        float(abs(expected_residuals[0] / result.constraint_scales[0]).max()),
        *[
            float((-residual / scale).clip(min=0).max())
            for residual, scale in zip(
                expected_residuals[1:], result.constraint_scales[1:], strict=True
            )
        ],
    )
    assert result.max_constraint_violation == pytest.approx(violation, abs=1e-10)
    lower_error = (-allocation).clip(min=0)
    upper_error = (allocation - reference.caps * units.spend()).clip(min=0)
    bound_violation = float(
        (
            xr.where(lower_error > upper_error, lower_error, upper_error)
            / result.decision_scales["spend"]
        ).max()
    )
    assert result.max_bound_violation == pytest.approx(bound_violation, abs=1e-12)


def _gaps(
    reference: ReferenceProblem,
    baseline: OptimizationResult,
    result: OptimizationResult,
    units: Units,
) -> tuple[float, float]:
    allocation_gap = (
        float(
            abs(
                result.allocation["spend"] / units.spend()
                - baseline.allocation["spend"]
            ).max()
        )
        / reference.budget
    )
    objective_gap = abs(result.objective / units.sales - baseline.objective) / abs(
        baseline.objective
    )
    assert np.isfinite(allocation_gap) and np.isfinite(objective_gap)
    return allocation_gap, objective_gap


@pytest.fixture(scope="module")
def setup() -> Setup:
    train = _training()
    fitted = _fit(train, seed=19)
    spend_scale = train["spend"].max("date")
    future_dates = pd.date_range(
        train["date"].values[-1] + pd.Timedelta(weeks=1), periods=4, freq="W-MON"
    )
    initial = (
        (0.65 * spend_scale).expand_dims(date=future_dates).transpose("date", "channel")
    )
    sales, _ = _numpy_outputs(fitted, train, initial)
    budget = float(initial.sum())
    reference = ReferenceProblem(
        train=train,
        initial=initial,
        budget=budget,
        weekly_cap=1.2 * budget / len(future_dates),
        caps=1.2 * spend_scale,
        sales_floor=0.9 * float(sales[:, :, -2:].sum(axis=(2, 3)).mean()),
    )
    baseline = _solve(fitted, reference, REFERENCE)
    _check_result(fitted, reference, REFERENCE, baseline)
    return Setup(fitted, reference, baseline)


@pytest.mark.parametrize("units", UNIT_CASES[1:], ids=lambda units: units.name)
def test_same_joint_posterior_gives_same_allocation_and_objective_in_eight_unit_cases(
    setup: Setup, units: Units
) -> None:
    result = _solve(setup.fitted, setup.reference, units)
    _check_result(setup.fitted, setup.reference, units, result)
    allocation_gap, objective_gap = _gaps(
        setup.reference, setup.baseline, result, units
    )
    assert allocation_gap <= 1e-6
    assert objective_gap <= 1e-9


@pytest.fixture(scope="module")
def refit_control(setup: Setup) -> tuple[float, float]:
    fitted = _fit(setup.reference.train, seed=27)
    result = _solve(fitted, setup.reference, REFERENCE)
    _check_result(fitted, setup.reference, REFERENCE, result)
    gaps = _gaps(setup.reference, setup.baseline, result, REFERENCE)
    # An unstable control must not make the refit tolerance arbitrarily permissive.
    assert gaps[0] < 0.05
    assert gaps[1] < 0.05
    return gaps


@pytest.mark.parametrize("units", REFIT_CASES, ids=lambda units: units.name)
def test_actual_unit_refits_stay_within_different_seed_control_noise(
    setup: Setup, refit_control: tuple[float, float], units: Units
) -> None:
    fitted = _fit(_in_units(setup.reference.train, units), seed=27)
    result = _solve(fitted, setup.reference, units)
    _check_result(fitted, setup.reference, units, result)
    allocation_gap, objective_gap = _gaps(
        setup.reference, setup.baseline, result, units
    )
    # The 1e-3 floor avoids treating a coincidentally tiny control gap as exact MCMC equivalence.
    assert allocation_gap <= max(5 * refit_control[0], 1e-3)
    assert objective_gap <= max(5 * refit_control[1], 1e-3)
