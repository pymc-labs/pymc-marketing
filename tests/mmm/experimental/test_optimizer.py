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
"""Original-unit, labeled, posterior-conditioned optimization contracts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import pytest
import xarray as xr
from pymc_extras.prior import CUSTOM_TRANSFORMS, Prior
from pytensor.tensor.shape import Shape_i
from pytensor.xtensor.type import XTensorVariable, as_xtensor

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.experimental import (
    GAM,
    Data,
    Equation,
    MediaTransform,
    optimize,
)
from pymc_marketing.mmm.experimental._graph import walk
from pymc_marketing.mmm.experimental._optimizer import _build_problem
from pymc_marketing.terms import (
    ModelTerm,
    Named,
    Parameter,
    Ref,
    Transform,
    build_param,
)

CHANNELS = ["tv", "radio"]
ALPHA = 0.4
LAM = 1.2
L_MAX = 3
LEFT = xr.DataArray([0.35, 0.8], dims="channel", coords={"channel": CHANNELS})
RIGHT = xr.DataArray([1.35, 0.2], dims="channel", coords={"channel": CHANNELS})
SAMPLE_KWARGS = {
    "draws": 25,
    "tune": 40,
    "chains": 2,
    "cores": 1,
    "random_seed": 892,
    "progressbar": False,
    "compute_convergence_checks": False,
}


@dataclass
class FittedProblem:
    model: GAM
    train: xr.Dataset
    future: xr.Dataset
    spend: Data
    price: Data
    level: Data
    weights: Data
    dated: Parameter
    media: Any
    left: Any
    right: Any
    weighted: Any
    mean: Any


@dataclass
class DatedConstantProblem:
    model: GAM
    train: xr.Dataset
    level: Data
    constant: xr.DataArray
    term: Any


@dataclass
class UnlabeledConstantProblem:
    model: GAM
    train: xr.Dataset
    spend: Data
    labeled: Any
    unlabeled: Any


def _adstock(spend: np.ndarray) -> np.ndarray:
    weights = ALPHA ** np.arange(L_MAX)
    weights /= weights.sum()
    result = np.zeros_like(spend, dtype=float)
    for lag, weight in enumerate(weights):
        result[lag:] += weight * spend[: len(spend) - lag]
    return result


def _media(posterior: xr.Dataset, spend: np.ndarray) -> np.ndarray:
    beta = posterior["saturation_beta"].transpose("chain", "draw", "channel")
    response = np.tanh(LAM * _adstock(spend) / 2)
    return beta.values[:, :, None, :] * response[None, None, :, :]


@pytest.fixture(scope="module")
def fitted() -> FittedProblem:
    rng = np.random.default_rng(15)
    dates = pd.date_range("2025-01-06", periods=12, freq="W-MON")
    spend_values = rng.uniform(0.2, 1.8, size=(len(dates), len(CHANNELS)))
    price_values = rng.uniform(0.5, 1.5, size=len(dates))
    preference = np.array([1.4, 0.9])
    media_values = np.array([1.7, 0.9]) * np.tanh(LAM * _adstock(spend_values) / 2)
    weights_values = np.array([1.1, 0.7])
    unrelated = rng.normal(size=len(dates))
    train = xr.Dataset(
        {
            "spend": (("date", "channel"), spend_values),
            "price": ("date", price_values),
            "level": 3.0,
            "channel_weight": ("channel", weights_values),
            "sales": (
                "date",
                3
                + (media_values * weights_values).sum(axis=1)
                - 0.3 * price_values
                + rng.normal(0, 0.05, len(dates)),
            ),
            "left_obs": (
                ("date", "channel"),
                -preference * (spend_values - LEFT.values) ** 2
                + rng.normal(0, 0.05, spend_values.shape),
            ),
            "right_obs": (
                ("date", "channel"),
                -preference * (spend_values - RIGHT.values) ** 2
                + rng.normal(0, 0.05, spend_values.shape),
            ),
            "unrelated": ("date", unrelated),
            "auxiliary": (
                "date",
                0.6 * unrelated + 0.2 + rng.normal(0, 0.1, len(dates)),
            ),
        },
        coords={"date": dates, "channel": CHANNELS},
    )
    spend, price = Data("spend"), Data("price")
    level, weights = Data("level"), Data("channel_weight")
    media = (
        spend
        >> GeometricAdstock(l_max=L_MAX, priors={"alpha": ALPHA})
        >> LogisticSaturation(priors={"lam": LAM, "beta": Prior("HalfNormal", sigma=2)})
    ).named("media")
    preference_term = Parameter(
        "preference", Prior("HalfNormal", sigma=2, dims="channel")
    )
    left = (-preference_term * Transform(spend - LEFT, pmd.math.square)).named(
        "left_value"
    )
    right = (-preference_term * Transform(spend - RIGHT, pmd.math.square)).named(
        "right_value"
    )
    weighted = media * weights
    mean = (
        weighted.sum("channel")
        + price * Parameter("price_beta", Prior("Normal", mu=-0.3, sigma=0.4))
        + level
    ).named("sales_mean")
    dated = Parameter("dated_beta", Prior("Normal", mu=0.2, sigma=0.05, dims="date"))
    model = GAM(
        Equation(observed="sales", mu=mean, likelihood=Prior("Normal", sigma=0.05)),
        Equation(observed="left_obs", mu=left, likelihood=Prior("Normal", sigma=0.05)),
        Equation(
            observed="right_obs", mu=right, likelihood=Prior("Normal", sigma=0.05)
        ),
        Equation(
            observed="auxiliary",
            mu=Data("unrelated") * Parameter("aux_beta", Prior("Normal")) + dated,
            likelihood=Prior("Normal", sigma=0.1),
        ),
    )
    model.fit(train, **SAMPLE_KWARGS)
    future = xr.Dataset(
        {
            "spend": (("date", "channel"), [[0.7, 1.2], [0.9, 0.8], [1.1, 0.6]]),
            "price": ("date", [0.8, 1.1, 0.7]),
        },
        coords={
            "date": pd.date_range(
                dates[-1] + pd.Timedelta(weeks=1), periods=3, freq="W-MON"
            ),
            "channel": CHANNELS,
        },
    )
    return FittedProblem(
        model,
        train,
        future,
        spend,
        price,
        level,
        weights,
        dated,
        media,
        left,
        right,
        weighted,
        mean,
    )


@pytest.fixture(scope="module")
def dated_constant_fitted() -> DatedConstantProblem:
    dates = pd.date_range("2025-01-06", periods=3, freq="W-MON")
    constant = xr.DataArray(
        [1.0, 2.0, 100.0], dims="date", coords={"date": dates}
    ).isel(date=[2, 0, 1])
    train = xr.Dataset(
        {"level": 2.0, "sales": ("date", [2.3, 3.9, 200.2])},
        coords={"date": dates},
    )
    level = Data("level")
    term = (level * constant).named("dated_level")
    model = GAM(
        Equation(
            observed="sales",
            mu=term + Parameter("offset", Prior("Normal", sigma=1.0)),
            likelihood=Prior("Normal", sigma=0.2),
        )
    )
    model.fit(train, **{**SAMPLE_KWARGS, "chains": 1, "random_seed": 1995})
    return DatedConstantProblem(model, train, level, constant, term)


@pytest.fixture(scope="module")
def unlabeled_constant_fitted(fitted: FittedProblem) -> UnlabeledConstantProblem:
    train = fitted.train[["spend"]]
    spend = Data("spend")
    preference = Parameter(
        "constant_preference",
        Prior(
            "HalfNormal",
            sigma=xr.DataArray([1.0, 2.0], dims="channel"),
            dims="channel",
        ),
    )
    labeled = (-preference * Transform(spend - LEFT, pmd.math.square)).named(
        "labeled_value"
    )
    factor = xr.DataArray([0.5, 2.0], dims="channel")
    unlabeled = (labeled * factor).named("unlabeled_value")
    observations = (
        -np.array([0.8, 1.3])
        * (train["spend"].values - LEFT.values) ** 2
        * factor.values
    )
    train = train.assign(constant_obs=(("date", "channel"), observations))
    model = GAM(
        Equation(
            observed="constant_obs",
            mu=unlabeled,
            likelihood=Prior("Normal", sigma=0.1),
        )
    )
    model.fit(train, **{**SAMPLE_KWARGS, "chains": 1, "random_seed": 2117})
    return UnlabeledConstantProblem(model, train, spend, labeled, unlabeled)


def _posterior(fitted: FittedProblem) -> xr.Dataset:
    assert fitted.model.idata is not None
    return fitted.model.idata["posterior"].to_dataset()


def _media_objective(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
    return evaluate(u)["term"].sum("date").sum("channel").mean("sample")


def _arguments(fitted: FittedProblem, **changes: Any) -> dict[str, Any]:
    return {
        "model": fitted.model,
        "terms": fitted.media,
        "inputs": [fitted.spend],
        "data": fitted.future[["spend"]],
        "objective": _media_objective,
        "bounds": {"spend": (0.0, 2.5)},
        **changes,
    }


@pytest.mark.parametrize("weight", [0.25, 0.75])
def test_shared_input_terms_obey_user_tradeoff_and_known_optimum(
    fitted: FittedProblem, weight: float
) -> None:
    def objective(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        out = evaluate(u)
        assert isinstance(u["spend"], XTensorVariable)
        assert isinstance(out["left"], XTensorVariable)
        return (
            (weight * out["left"] + (1 - weight) * out["right"])
            .sum("date")
            .sum("channel")
            .mean("sample")
        )

    result = optimize(
        **_arguments(
            fitted,
            terms={"left": fitted.left, "right": fitted.right},
            objective=objective,
            options={"ftol": 1e-12},
        )
    )
    target = weight * LEFT + (1 - weight) * RIGHT
    expected = target.broadcast_like(fitted.future["spend"])
    preference = _posterior(fitted)["preference"].mean(("chain", "draw"))
    expected_objective = (
        -float((preference * weight * (1 - weight) * (LEFT - RIGHT) ** 2).sum())
        * fitted.future.sizes["date"]
    )

    assert result.scipy.success
    assert result.feasible
    assert set(result.allocation.data_vars) == {"spend"}
    xr.testing.assert_allclose(
        result.allocation["spend"], expected, atol=1e-7, rtol=1e-7
    )
    np.testing.assert_allclose(result.objective, expected_objective, atol=1e-10)


@pytest.mark.parametrize(
    "declarations",
    ["missing", "extra", "duplicate-object", "duplicate-name"],
)
def test_inputs_are_exactly_the_selected_terms_dependencies(
    fitted: FittedProblem, declarations: str
) -> None:
    inputs = {
        "missing": [fitted.spend],
        "extra": [
            fitted.spend,
            fitted.price,
            fitted.weights,
            fitted.level,
            Data("unrelated"),
        ],
        "duplicate-object": [
            fitted.spend,
            fitted.price,
            fitted.weights,
            fitted.level,
            fitted.spend,
        ],
        "duplicate-name": [
            fitted.spend,
            fitted.price,
            fitted.weights,
            fitted.level,
            Data("spend"),
        ],
    }[declarations]
    with pytest.raises(ValueError):
        optimize(
            **_arguments(fitted, terms=fitted.mean, inputs=inputs, data=fitted.train)
        )


def test_missing_scenario_input_is_not_held_at_training_values(
    fitted: FittedProblem,
) -> None:
    with pytest.raises(ValueError):
        optimize(**_arguments(fitted, data=fitted.future[["price"]]))


def test_mixed_outputs_preserve_paired_joint_posterior_and_implicit_dimensions(
    fitted: FittedProblem,
) -> None:
    def objective(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        out = evaluate(u)
        return (
            ((out["left"] - out["right"]) ** 2)
            .sum("date")
            .sum("channel")
            .mean("sample")
        )

    problem = _build_problem(
        **_arguments(
            fitted,
            terms={
                "media": fitted.media,
                "left": fitted.left,
                "right": fitted.right,
                "raw": fitted.spend,
            },
            data=fitted.train[["spend"]],
            objective=objective,
        )
    )
    evaluated = problem.evaluator.evaluate(fitted.train[["spend"]])
    posterior = _posterior(fitted)
    for name, stored in [
        ("media", "media"),
        ("left", "left_value"),
        ("right", "right_value"),
    ]:
        expected = (
            posterior[stored]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "date", "channel")
        )
        assert set(evaluated[name].dims) == {"sample", "date", "channel"}
        np.testing.assert_allclose(
            evaluated[name].transpose("sample", "date", "channel").values,
            expected.values,
            atol=1e-10,
        )
    assert evaluated["raw"].dims == ("date", "channel")
    xr.testing.assert_allclose(evaluated["raw"], fitted.train["spend"])
    expected_objective = float(
        ((posterior["left_value"] - posterior["right_value"]) ** 2)
        .sum(("date", "channel"))
        .mean(("chain", "draw"))
    )
    np.testing.assert_allclose(
        problem.raw(problem.to_z(problem.initial))[0], expected_objective
    )

    reduced = _build_problem(
        **_arguments(
            fitted,
            terms=fitted.media.sum("channel"),
            data=fitted.train[["spend"]],
            objective=lambda e, u: e(u)["term"].sum("date").mean("sample"),
        )
    ).evaluator.evaluate(fitted.train[["spend"]])
    np.testing.assert_allclose(
        reduced["term"].transpose("sample", "date").values,
        posterior["media"]
        .sum("channel")
        .stack(sample=("chain", "draw"))
        .transpose("sample", "date")
        .values,
        atol=1e-10,
    )


def test_label_and_dimension_permutations_preserve_original_unit_solution(
    fitted: FittedProblem,
) -> None:
    spend_target = xr.DataArray(
        [[0.5, 1.0], [1.1, 0.4], [0.8, 0.7]],
        dims=("date", "channel"),
        coords={"date": fitted.future["date"], "channel": CHANNELS},
    )
    price_target = xr.DataArray(
        [0.6, 1.2, 0.9], dims="date", coords={"date": fitted.future["date"]}
    )
    caps = xr.DataArray([1.6, 1.3], dims="channel", coords={"channel": CHANNELS})
    decision_scale = xr.DataArray(
        [0.3, 2.0], dims="channel", coords={"channel": CHANNELS}
    )
    row_scale = xr.DataArray(
        [0.2, 1.3, 0.7], dims="date", coords={"date": fitted.future["date"]}
    )

    def problem(reverse: bool) -> Any:
        spend_constant = (
            spend_target.isel(channel=slice(None, None, -1)).transpose(
                "channel", "date"
            )
            if reverse
            else spend_target
        )
        price_constant = (
            price_target.isel(date=slice(None, None, -1)) if reverse else price_target
        )

        def objective(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
            out = evaluate(u)
            return -((out["spend"] - evaluate.constant(spend_constant)) ** 2).sum(
                "date"
            ).sum("channel") - (
                (out["price"] - evaluate.constant(price_constant)) ** 2
            ).sum("date")

        data = (
            fitted.future.isel(channel=slice(None, None, -1)).transpose(
                "channel", "date"
            )
            if reverse
            else fitted.future
        )
        return _build_problem(
            **_arguments(
                fitted,
                terms={"spend": fitted.spend, "price": fitted.price},
                inputs=[fitted.spend, fitted.price]
                if reverse
                else [fitted.price, fitted.spend],
                data=data,
                objective=objective,
                bounds={
                    "spend": (
                        0.0,
                        caps.isel(channel=slice(None, None, -1)) if reverse else caps,
                    ),
                    "price": (0.0, 2.0),
                },
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda e, u: 2.5 - u["spend"].sum("channel"),
                        "scale": row_scale.isel(date=slice(None, None, -1))
                        if reverse
                        else row_scale,
                    }
                ],
                scaling={
                    "decisions": {
                        "spend": decision_scale.isel(channel=slice(None, None, -1))
                        if reverse
                        else decision_scale,
                        "price": 0.8,
                    },
                    "objective": 1.7,
                },
            )
        )

    baseline, reordered = problem(False), problem(True)
    np.testing.assert_allclose(
        baseline.to_z(baseline.initial), reordered.to_z(reordered.initial)
    )
    original = baseline.solve(options={"ftol": 1e-12})
    permuted = reordered.solve(options={"ftol": 1e-12})
    for result in (original, permuted):
        assert result.scipy.success
        assert result.feasible
        assert result.allocation["spend"].dims == ("date", "channel")
        np.testing.assert_array_equal(result.allocation["channel"], CHANNELS)
        xr.testing.assert_allclose(
            result.allocation["spend"], spend_target, atol=1e-6, rtol=1e-6
        )
        xr.testing.assert_allclose(
            result.allocation["price"], price_target, atol=1e-6, rtol=1e-6
        )
        xr.testing.assert_allclose(
            result.constraints[0], 2.5 - result.allocation["spend"].sum("channel")
        )
        xr.testing.assert_allclose(result.constraint_scales[0], row_scale)
        xr.testing.assert_allclose(
            result.decision_scales["spend"], decision_scale.broadcast_like(spend_target)
        )
        assert result.objective_scale == 1.7
        np.testing.assert_allclose(result.scipy.fun, -result.objective / 1.7)
    xr.testing.assert_allclose(original.allocation, permuted.allocation, atol=1e-7)


def test_objective_gradient_and_every_constraint_jacobian_match_finite_differences(
    fitted: FittedProblem,
) -> None:
    data = fitted.future[["spend"]].assign(level=1.0)
    lower = xr.DataArray([0.2, 0.1], dims="channel", coords={"channel": CHANNELS})
    upper = xr.DataArray([3.0, 1.8], dims="channel", coords={"channel": CHANNELS})

    def objective(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        out = evaluate(u)
        return (
            (out["media"] ** 2 + 0.3 * out["left"])
            .sum("date")
            .sum("channel")
            .mean("sample")
        )

    problem = _build_problem(
        **_arguments(
            fitted,
            terms={"media": fitted.media, "left": fitted.left, "unused": fitted.level},
            inputs=[fitted.spend, fitted.level],
            data=data,
            objective=objective,
            bounds={"spend": (lower, upper), "level": (-2.0, 4.0)},
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda e, u: u["spend"].sum("date").sum("channel") - 5.0,
                },
                {"type": "ineq", "fun": lambda e, u: 2.4 - u["spend"].sum("channel")},
                {
                    "type": "ineq",
                    "fun": lambda e, u: (
                        (e(u)["media"].isel(date=slice(1, None)) ** 2)
                        .sum("date")
                        .sum("channel")
                        .mean("sample")
                        - 0.2
                    ),
                },
                {
                    "type": "ineq",
                    "fun": lambda e, u: as_xtensor(
                        np.ones(data.sizes["date"]), dims=("date",)
                    ),
                },
            ],
        )
    )
    point_data = data.assign(spend=data["spend"] + 0.07)
    z = problem.to_z({name: point_data[name] for name in problem.initial})
    current = problem.raw(z)
    scales_before = (
        problem.objective_scale,
        [scale.copy(deep=True) for scale in problem.constraint_scales],
    )
    step = 1e-6
    plus = [problem.raw(z + step * direction) for direction in np.eye(z.size)]
    minus = [problem.raw(z - step * direction) for direction in np.eye(z.size)]
    for value_index in range(0, len(current), 2):
        finite = np.stack(
            [
                (np.asarray(a[value_index]) - np.asarray(b[value_index])) / (2 * step)
                for a, b in zip(plus, minus, strict=True)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(
            current[value_index + 1], finite, rtol=2e-6, atol=2e-8
        )
        np.testing.assert_array_equal(np.asarray(current[value_index + 1])[..., 0], 0.0)
    np.testing.assert_array_equal(current[-1], 0.0)
    decoded = problem.to_u(z)
    xr.testing.assert_allclose(xr.Dataset(decoded), point_data)
    assert problem.objective_scale == scales_before[0]
    for actual, expected in zip(
        problem.constraint_scales, scales_before[1], strict=True
    ):
        xr.testing.assert_identical(actual, expected)
    outside = {
        "level": xr.DataArray(-2.6),
        "spend": (upper + 0.1 * (upper - lower)).broadcast_like(data["spend"]),
    }
    residuals = problem.labeled_constraints(problem.raw(problem.to_z(outside)))
    bound_violation, _ = problem._violations(outside, residuals)
    assert bound_violation == pytest.approx(0.1)


def test_equality_vector_and_model_constraints_are_reported_in_original_units(
    fitted: FittedProblem,
) -> None:
    budget = float(fitted.future["spend"].sum())
    floor = 0.7 * float(
        _media(_posterior(fitted), fitted.future["spend"].values)[:, :, 1:]
        .sum(axis=(2, 3))
        .mean()
    )
    result = optimize(
        **_arguments(
            fitted,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda e, u: u["spend"].sum("date").sum("channel") - budget,
                },
                {"type": "ineq", "fun": lambda e, u: 2.2 - u["spend"].sum("channel")},
                {
                    "type": "ineq",
                    "fun": lambda e, u: (
                        e(u)["term"]
                        .isel(date=slice(1, None))
                        .sum("date")
                        .sum("channel")
                        .mean("sample")
                        - floor
                    ),
                },
            ],
            options={"ftol": 1e-11},
        )
    )
    allocation = result.allocation["spend"]
    response = _media(_posterior(fitted), allocation.values)
    expected_residuals = [
        xr.DataArray(float(allocation.sum()) - budget),
        2.2 - allocation.sum("channel"),
        xr.DataArray(response[:, :, 1:].sum(axis=(2, 3)).mean() - floor),
    ]
    assert result.scipy.success
    assert result.feasible
    np.testing.assert_allclose(
        result.objective, response.sum(axis=(2, 3)).mean(), atol=1e-10
    )
    for actual, expected in zip(result.constraints, expected_residuals, strict=True):
        xr.testing.assert_allclose(actual, expected, atol=1e-10)
    violations = [
        float(abs(result.constraints[0] / result.constraint_scales[0]).max()),
        *[
            float((-residual / scale).clip(min=0).max())
            for residual, scale in zip(
                result.constraints[1:], result.constraint_scales[1:], strict=True
            )
        ],
    ]
    bound_violation = float(
        (
            xr.where(
                allocation < 0,
                -allocation,
                xr.where(allocation > 2.5, allocation - 2.5, 0),
            )
            / result.decision_scales["spend"]
        ).max()
    )
    assert result.max_constraint_violation == pytest.approx(max(violations))
    assert result.max_bound_violation == pytest.approx(bound_violation)
    assert abs(float(result.constraints[0])) < 1e-7
    assert float(result.constraints[1].min()) >= -1e-7
    assert float(result.constraints[2]) >= -1e-7


def test_posterior_model_inequality_changes_the_analytic_optimum(
    fitted: FittedProblem,
) -> None:
    preference = _posterior(fitted)["preference"].mean(("chain", "draw"))
    distance = float(
        fitted.future.sizes["date"] * (preference * (LEFT - RIGHT) ** 2).sum()
    )
    floor = -distance / 4
    result = optimize(
        **_arguments(
            fitted,
            terms={"left": fitted.left, "right": fitted.right},
            objective=lambda e, u: (
                e(u)["left"].mean("sample").sum("date").sum("channel")
            ),
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda e, u: (
                        e(u)["right"].mean("sample").sum("date").sum("channel") - floor
                    ),
                }
            ],
            options={"ftol": 1e-12},
        )
    )
    expected = ((LEFT + RIGHT) / 2).broadcast_like(result.allocation["spend"])
    assert result.scipy.success and result.feasible
    xr.testing.assert_allclose(result.allocation["spend"], expected, atol=1e-6)
    np.testing.assert_allclose(result.objective, floor, atol=1e-7)
    independent_residual = (
        float(-(preference * (result.allocation["spend"] - RIGHT) ** 2).sum()) - floor
    )
    np.testing.assert_allclose(result.constraints[0], independent_residual, atol=1e-10)
    assert abs(independent_residual) < 1e-7
    assert -distance - floor < 0  # The unconstrained LEFT optimum is infeasible.


def test_history_is_fixed_and_only_scenario_dates_and_explicit_tail_are_scored(
    fitted: FittedProblem,
) -> None:
    history = fitted.train[["spend"]].isel(date=slice(-2, None))
    history_before = history.copy(deep=True)
    dates = pd.date_range(fitted.future["date"].values[0], periods=5, freq="W-MON")
    spend = xr.DataArray(
        np.concatenate([fitted.future["spend"].values, np.zeros((2, 2))]),
        dims=("date", "channel"),
        coords={"date": dates, "channel": CHANNELS},
    )
    data = xr.Dataset(
        {"spend": spend, "channel_weight": fitted.train["channel_weight"]}
    )
    upper = xr.full_like(spend, 2.5)
    upper.loc[{"date": dates[-2:]}] = 0.0
    budget = float(spend.sum())
    kwargs = _arguments(
        fitted,
        terms=fitted.weighted,
        inputs=[fitted.weights, fitted.spend],
        data=data,
        history=history,
        objective=lambda e, u: (
            e(u)["term"]
            .isel(date=slice(1, None))
            .sum("date")
            .sum("channel")
            .mean("sample")
        ),
        bounds={
            "spend": (0.0, upper),
            "channel_weight": (data["channel_weight"], data["channel_weight"]),
        },
        constraints=[
            {
                "type": "eq",
                "fun": lambda e, u: (
                    u["spend"].isel(date=slice(0, 3)).sum("date").sum("channel")
                    - budget
                ),
            }
        ],
    )
    problem = _build_problem(**kwargs)
    for scenario in (data, data.assign(spend=data["spend"] * 0.7)):
        evaluated = problem.evaluator.evaluate(scenario)["term"]
        combined = np.concatenate([history["spend"].values, scenario["spend"].values])
        expected = (
            _media(_posterior(fitted), combined)[:, :, 2:]
            * data["channel_weight"].values
        )
        np.testing.assert_allclose(
            evaluated.transpose("sample", "date", "channel").values,
            expected.reshape((-1, len(dates), 2)),
            atol=1e-10,
        )
        np.testing.assert_array_equal(evaluated["date"].values, dates.values)
    no_history = _build_problem(**{**kwargs, "history": None}).evaluator.evaluate(data)[
        "term"
    ]
    no_history_expected = (
        _media(_posterior(fitted), spend.values) * data["channel_weight"].values
    )
    np.testing.assert_allclose(
        no_history.transpose("sample", "date", "channel").values,
        no_history_expected.reshape((-1, len(dates), 2)),
        atol=1e-10,
    )
    assert not np.allclose(
        no_history_expected[:, :, :2],
        _media(
            _posterior(fitted), np.concatenate([history["spend"].values, spend.values])
        )[:, :, 2:4]
        * data["channel_weight"].values,
    )
    result = problem.solve(options={"ftol": 1e-11})
    final_response = (
        _media(
            _posterior(fitted),
            np.concatenate(
                [history["spend"].values, result.allocation["spend"].values]
            ),
        )[:, :, 2:]
        * data["channel_weight"].values
    )
    assert result.scipy.success
    assert result.feasible
    assert result.allocation["channel_weight"].dims == ("channel",)
    xr.testing.assert_allclose(
        result.allocation["channel_weight"], data["channel_weight"]
    )
    np.testing.assert_array_equal(result.allocation["date"].values, dates.values)
    np.testing.assert_array_equal(
        result.allocation["spend"].isel(date=slice(-2, None)), 0.0
    )
    assert np.all(final_response[:, :, -2:].sum(axis=-1) > 0)
    np.testing.assert_allclose(
        result.objective, final_response[:, :, 1:].sum(axis=(2, 3)).mean(), atol=1e-10
    )
    xr.testing.assert_identical(history, history_before)
    with pytest.raises(ValueError):
        _build_problem(**{**kwargs, "terms": fitted.weighted.sum("date")})


def test_optimization_and_rejection_do_not_mutate_fitted_state(
    fitted: FittedProblem,
) -> None:
    model = fitted.model.model
    assert model is not None
    assert fitted.model._context is not None
    names = tuple(fitted.model._context.data_variables.values())
    containers = {name: model[name].eval().copy() for name in names}
    coords = dict(model.coords)
    posterior = _posterior(fitted).copy(deep=True)
    training = fitted.train.copy(deep=True)
    assert fitted.model._training_data is not None
    stored_training = fitted.model._training_data.copy(deep=True)
    saturation = fitted.media.expr.transformation
    prior = saturation.function_priors["beta"].to_dict()
    optimize(**_arguments(fitted))
    xr.testing.assert_identical(fitted.model._training_data, stored_training)
    with pytest.raises(ValueError):
        optimize(**_arguments(fitted, inputs=[]))
    assert fitted.model.model is model
    assert model.coords == coords
    xr.testing.assert_identical(_posterior(fitted), posterior)
    xr.testing.assert_identical(fitted.train, training)
    xr.testing.assert_identical(fitted.model._training_data, stored_training)
    assert saturation.function_priors["beta"].to_dict() == prior
    for name, before in containers.items():
        np.testing.assert_array_equal(model[name].eval(), before)


@pytest.mark.parametrize("source", ["scenario", "history", "numeric-evaluator"])
@pytest.mark.parametrize(
    "value, message",
    [
        pytest.param(
            np.int64(2**53 + 1), "exact float64 conversion", id="positive-integer"
        ),
        pytest.param(
            np.int64(-(2**53 + 1)), "exact float64 conversion", id="negative-integer"
        ),
        pytest.param(2.0 + 1.0j, "finite real numeric", id="complex"),
    ],
)
def test_optimizer_inputs_reject_inexact_or_nonreal_values_without_mutation(
    fitted: FittedProblem, source: str, value: Any, message: str
) -> None:
    model = fitted.model.model
    assert model is not None
    assert fitted.model._context is not None
    assert fitted.model._training_data is not None
    containers = {
        name: model[name].get_value().copy()
        for name in fitted.model._context.data_variables.values()
    }
    coords = dict(model.coords)
    posterior = _posterior(fitted).copy(deep=True)
    training = fitted.train.copy(deep=True)
    stored_training = fitted.model._training_data.copy(deep=True)
    history: xr.Dataset | None = None
    if source == "history":
        history = xr.Dataset(
            {"price": ("date", np.asarray([value]))},
            coords={"date": fitted.train["date"].isel(date=slice(-1, None))},
        )
        data = fitted.future[["price"]].isel(date=slice(0, 1))
        kwargs = _arguments(
            fitted,
            terms=fitted.price,
            inputs=[fitted.price],
            data=data,
            history=history,
            objective=lambda e, u: e(u)["term"].sum("date"),
            bounds={"price": (data["price"], data["price"])},
        )
    else:
        data = xr.Dataset({"level": value})
        kwargs = _arguments(
            fitted,
            terms=fitted.level,
            inputs=[fitted.level],
            data=data,
            objective=lambda e, u: e(u)["term"],
            bounds={"level": (value, value)},
        )
    data_before = data.copy(deep=True)
    history_before = None if history is None else history.copy(deep=True)
    problem = None
    if source == "numeric-evaluator":
        problem = _build_problem(
            **{
                **kwargs,
                "data": xr.Dataset({"level": 1.0}),
                "bounds": {"level": (1.0, 1.0)},
            }
        )
    with pytest.raises(ValueError, match=message):
        if problem is None:
            optimize(**kwargs)
        else:
            problem.evaluator.evaluate(data)
    if problem is not None:
        accepted = problem.evaluator.evaluate(xr.Dataset({"level": 1.0}))
        assert float(accepted["term"]) == 1.0
    xr.testing.assert_identical(data, data_before)
    for name in data.data_vars:
        assert data[name].dtype == data_before[name].dtype
    if history is not None:
        xr.testing.assert_identical(history, history_before)
        assert history["price"].dtype == history_before["price"].dtype
    assert fitted.model.model is model
    assert model.coords == coords
    xr.testing.assert_identical(_posterior(fitted), posterior)
    xr.testing.assert_identical(fitted.train, training)
    xr.testing.assert_identical(fitted.model._training_data, stored_training)
    for name, before in containers.items():
        np.testing.assert_array_equal(model[name].get_value(), before)


@pytest.mark.parametrize("value", [2**53, -(2**53)])
def test_optimizer_integer_precision_endpoints_preserve_original_units(
    fitted: FittedProblem, value: int
) -> None:
    data = xr.Dataset({"level": np.int64(value)})
    before = data.copy(deep=True)
    kwargs = _arguments(
        fitted,
        terms=fitted.level,
        inputs=[fitted.level],
        data=data,
        objective=lambda e, u: e(u)["term"],
        bounds={"level": (value, value)},
    )
    evaluated = _build_problem(**kwargs).evaluator.evaluate(data)["term"]
    assert float(evaluated) == value
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    assert float(result.allocation["level"]) == value
    assert result.objective == value
    xr.testing.assert_identical(data, before)
    assert data["level"].dtype == np.dtype("int64")


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_history_precision_endpoints_are_owned_and_preserve_carryover(
    fitted: FittedProblem, dtype: str
) -> None:
    history = xr.Dataset(
        {"price": ("date", np.asarray([-(2**53), 2**53], dtype=dtype))},
        coords={"date": fitted.train["date"].isel(date=slice(-2, None))},
    )
    before = history.copy(deep=True)
    data = (
        fitted.future[["price"]]
        .isel(date=slice(0, 2))
        .assign(price=("date", np.zeros(2, dtype="int64")))
    )
    term = MediaTransform(
        fitted.price,
        GeometricAdstock(
            l_max=3,
            normalize=False,
            priors={"alpha": 0.5},
            prefix="exact_history",
        ),
    )
    kwargs = _arguments(
        fitted,
        terms=term,
        inputs=[fitted.price],
        data=data,
        history=history,
        objective=lambda e, u: e(u)["term"].sum("date"),
        bounds={"price": (0.0, 0.0)},
    )
    # With lag weights [1, 1/2, 1/4], both signed endpoints contribute to the scored carryover.
    expected = data["price"].copy(data=np.asarray([2**51, 2**51], dtype="float64"))
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    assert result.objective == 2**52
    np.testing.assert_array_equal(result.allocation["price"], 0.0)
    problem = _build_problem(**kwargs)
    xr.testing.assert_identical(history, before)
    assert history["price"].dtype == np.dtype(dtype)
    history["price"].values[:] = 0
    evaluated = problem.evaluator.evaluate(data)["term"]
    xr.testing.assert_allclose(evaluated, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("nested", [False, True])
def test_stochastic_equations_cannot_be_targets_or_dependencies(
    fitted: FittedProblem, nested: bool
) -> None:
    equation = fitted.model.equations[0]
    term = Transform(equation, pmd.math.square) if nested else equation
    with pytest.raises(ValueError):
        optimize(
            **_arguments(
                fitted,
                terms=term,
                inputs=[fitted.spend, fitted.price, fitted.weights, fitted.level],
                data=fitted.train,
            )
        )


def test_same_name_foreign_parameter_does_not_borrow_a_fitted_posterior(
    fitted: FittedProblem,
) -> None:
    foreign = Parameter("preference", Prior("HalfNormal", sigma=2, dims="channel"))
    with pytest.raises(ValueError):
        optimize(**_arguments(fitted, terms=foreign * fitted.spend))


def test_fitted_reference_exposes_complete_data_dependencies(
    fitted: FittedProblem,
) -> None:
    term = Ref("media") * fitted.level
    with pytest.raises(ValueError):
        _build_problem(
            **_arguments(
                fitted,
                terms=term,
                inputs=[fitted.level],
                data=fitted.future.assign(level=2.0),
            )
        )
    problem = _build_problem(
        **_arguments(
            fitted,
            terms=term,
            inputs=[fitted.level, fitted.spend],
            data=fitted.future[["spend"]].assign(level=2.0),
            bounds={"spend": (0.0, 2.5), "level": (1.0, 3.0)},
        )
    )
    expected = 2 * _media(_posterior(fitted), fitted.future["spend"].values)
    np.testing.assert_allclose(
        problem.evaluator.evaluate(fitted.future[["spend"]].assign(level=2.0))["term"]
        .transpose("sample", "date", "channel")
        .values,
        expected.reshape((-1, fitted.future.sizes["date"], 2)),
        atol=1e-10,
    )


@pytest.mark.parametrize("remaining", ["date", "channel", "sample"])
def test_objective_requires_every_reduction_to_be_explicit(
    fitted: FittedProblem, remaining: str
) -> None:
    def objective(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        value = evaluate(u)["term"]
        for dim in ("date", "channel", "sample"):
            if dim != remaining:
                value = value.sum(dim)
        return value

    with pytest.raises(ValueError):
        optimize(**_arguments(fitted, objective=objective))


@pytest.mark.parametrize("kind", ["unknown-dimension", "sliced-date"])
def test_constraints_must_have_complete_labeled_dimensions(
    fitted: FittedProblem, kind: str
) -> None:
    def constraint(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        if kind == "unknown-dimension":
            total = u["spend"].sum("date").sum("channel").values
            return as_xtensor(pt.stack([total, -total]), dims=("unlabeled",))
        return (
            evaluate(u)["term"].isel(date=slice(1, None)).sum("channel").mean("sample")
        )

    with pytest.raises(ValueError):
        optimize(
            **_arguments(fitted, constraints=[{"type": "ineq", "fun": constraint}])
        )


@pytest.mark.parametrize("kind", ["scenario", "bound", "unlabeled-bound", "constant"])
def test_mismatched_or_unlabeled_coordinate_values_are_rejected(
    fitted: FittedProblem, kind: str
) -> None:
    changes: dict[str, Any] = {}
    if kind == "scenario":
        changes["data"] = fitted.future[["spend"]].assign_coords(
            channel=["other", "radio"]
        )
    elif kind in {"bound", "unlabeled-bound"}:
        cap = xr.DataArray(
            [1.0, 2.0],
            dims="channel",
            coords={"channel": ["other", "radio"]} if kind == "bound" else None,
        )
        changes["bounds"] = {"spend": (0.0, cap)}
    else:
        constant = LEFT.isel(channel=slice(0, 1))
        changes["objective"] = lambda e, u: (
            (u["spend"] * e.constant(constant)).sum("date").sum("channel")
        )
    with pytest.raises(ValueError):
        optimize(**_arguments(fitted, **changes))


@pytest.mark.parametrize("required", [0.0, 1.0])
def test_all_fixed_scalar_bounds_report_feasible_and_infeasible_problems(
    fitted: FittedProblem, required: float
) -> None:
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.level,
            inputs=[fitted.level],
            data=xr.Dataset({"level": 0.0}),
            objective=lambda e, u: -((e(u)["term"] - 3.0) ** 2),
            bounds={"level": (0.0, 0.0)},
            constraints=[
                {"type": "eq", "fun": lambda e, u: u["level"] - required, "scale": 2.0}
            ],
        )
    )
    assert result.allocation["level"].dims == ()
    assert float(result.allocation["level"]) == 0.0
    assert result.objective == -9.0
    assert float(result.constraints[0]) == -required
    assert result.max_bound_violation == 0.0
    assert result.max_constraint_violation == required / 2.0
    assert result.feasible is (required == 0.0)
    assert bool(result.scipy.success) is result.feasible
    assert result.scipy.nit == 0


def test_unbounded_zero_start_needs_an_explicit_scale(fitted: FittedProblem) -> None:
    kwargs = _arguments(
        fitted,
        terms=fitted.level,
        inputs=[fitted.level],
        data=xr.Dataset({"level": 0.0}),
        bounds=None,
        objective=lambda e, u: -((e(u)["term"] - 7.0) ** 2),
    )
    with pytest.raises(ValueError):
        optimize(**kwargs)
    result = optimize(
        **{
            **kwargs,
            "scaling": {"decisions": {"level": 2.0}, "objective": 10.0},
            "options": {"ftol": 1e-12},
        }
    )
    assert result.scipy.success
    assert result.feasible
    np.testing.assert_allclose(float(result.allocation["level"]), 7.0, atol=1e-9)
    np.testing.assert_allclose(result.objective, 0.0, atol=1e-15)
    assert float(result.decision_scales["level"]) == 2.0
    assert result.objective_scale == 10.0


@pytest.mark.parametrize("category", ["decision", "objective", "constraint"])
@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_every_explicit_scale_is_positive_and_finite(
    fitted: FittedProblem, category: str, invalid: float
) -> None:
    scaling = {
        "decisions": {"level": invalid if category == "decision" else 1.0},
        "objective": invalid if category == "objective" else 1.0,
    }
    constraint_scale = invalid if category == "constraint" else 1.0
    with pytest.raises(ValueError):
        optimize(
            **_arguments(
                fitted,
                terms=fitted.level,
                inputs=[fitted.level],
                data=xr.Dataset({"level": 1.0}),
                objective=lambda e, u: -((u["level"] - 0.5) ** 2),
                bounds={"level": (0.0, 2.0)},
                scaling=scaling,
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda e, u: u["level"] - 1.0,
                        "scale": constraint_scale,
                    }
                ],
            )
        )


def test_initial_values_outside_bounds_are_not_silently_clipped(
    fitted: FittedProblem,
) -> None:
    with pytest.raises(ValueError):
        optimize(**_arguments(fitted, bounds={"spend": (0.0, 0.5)}))


@pytest.mark.parametrize("tolerance", [0.001, 0.01])
def test_feasibility_uses_scaled_residuals_not_absolute_original_units(
    fitted: FittedProblem, tolerance: float
) -> None:
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.level,
            inputs=[fitted.level],
            data=xr.Dataset({"level": 0.0}),
            objective=lambda e, u: u["level"],
            bounds={"level": (0.0, 0.0)},
            constraints=[
                {"type": "eq", "fun": lambda e, u: 0.01 + 0 * u["level"], "scale": 2.0},
                {
                    "type": "ineq",
                    "fun": lambda e, u: -0.02 + 0 * u["level"],
                    "scale": 10.0,
                },
            ],
            feasibility_tol=tolerance,
        )
    )
    np.testing.assert_allclose(
        [float(value) for value in result.constraints], [0.01, -0.02]
    )
    assert result.max_constraint_violation == pytest.approx(0.005)
    assert result.feasible is (0.005 <= tolerance)


@pytest.mark.parametrize(
    "invalid",
    [0.0, -1.0, np.nan, np.inf, np.array([1e-6, 1e-6])],
    ids=["zero", "negative", "nan", "infinite", "nonscalar"],
)
def test_feasibility_tolerance_requires_one_positive_finite_scalar(
    fitted: FittedProblem, invalid: Any
) -> None:
    with pytest.raises(ValueError, match="feasibility_tol"):
        optimize(**_arguments(fitted, feasibility_tol=invalid))


def test_solver_success_does_not_override_decoded_infeasibility(
    fitted: FittedProblem,
) -> None:
    required = 1e-8
    tolerance = 1e-12
    with pytest.warns(RuntimeWarning):
        result = optimize(
            **_arguments(
                fitted,
                terms=fitted.level,
                inputs=[fitted.level],
                data=xr.Dataset({"level": 0.0}),
                objective=lambda e, u: -(u["level"] ** 2),
                bounds={"level": (-1.0, 1.0)},
                scaling={"objective": 1.0},
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda e, u: u["level"] - required,
                        "scale": 1.0,
                    }
                ],
                options={"ftol": 1e-3},
                feasibility_tol=tolerance,
            )
        )
    residual = float(result.allocation["level"]) - required
    assert result.scipy.success
    assert result.feasible is False
    assert abs(residual) > tolerance
    np.testing.assert_allclose(result.constraints[0], residual, atol=1e-15)
    assert result.max_constraint_violation == pytest.approx(abs(residual))
    assert result.max_bound_violation == 0.0


def test_solver_iteration_limit_does_not_mean_allocation_is_infeasible(
    fitted: FittedProblem,
) -> None:
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.level,
            inputs=[fitted.level],
            data=xr.Dataset({"level": 0.5}),
            objective=lambda e, u: -((u["level"] - 1.2) ** 2),
            bounds={"level": (0.0, 2.0)},
            options={"maxiter": 1},
        )
    )
    assert not result.scipy.success
    assert result.feasible
    assert 0 <= float(result.allocation["level"]) <= 2
    assert result.max_bound_violation == 0.0
    assert result.max_constraint_violation == 0.0


def test_fixed_variables_cannot_dilute_movable_objective_or_constraint_scales(
    fitted: FittedProblem,
) -> None:
    literal_kwargs = _arguments(
        fitted,
        terms=fitted.level,
        inputs=[fitted.level],
        data=xr.Dataset({"level": 0.5}),
        objective=lambda e, u: e(u)["term"],
        bounds={"level": (0.0, 1.0)},
        constraints=[{"type": "eq", "fun": lambda e, u: e(u)["term"] - 1.0}],
    )
    literal = optimize(**literal_kwargs)
    assert literal.scipy.success and literal.feasible
    for coefficient in (1.0, 1e9):
        term = fitted.level + coefficient * fitted.weights.sum("channel")
        problem = _build_problem(
            **{
                **literal_kwargs,
                "terms": term,
                "inputs": [fitted.weights, fitted.level],
                "data": xr.Dataset(
                    {
                        "level": 0.5,
                        "channel_weight": xr.zeros_like(fitted.train["channel_weight"]),
                    }
                ),
                "bounds": {"level": (0.0, 1.0), "channel_weight": (0.0, 0.0)},
            }
        )
        values = problem.raw(problem.to_z(problem.initial))
        np.testing.assert_array_equal(values[1], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(values[3], [[0.0, 0.0, 1.0]])
        result = problem.solve()
        assert result.scipy.success and result.feasible
        assert result.objective_scale == literal.objective_scale == 1.0
        assert (
            float(result.constraint_scales[0])
            == float(literal.constraint_scales[0])
            == 1.0
        )
        np.testing.assert_allclose(
            float(result.allocation["level"]), float(literal.allocation["level"])
        )
        np.testing.assert_allclose(result.objective, 1.0)
        np.testing.assert_allclose(float(result.constraints[0]), 0.0, atol=1e-12)
        np.testing.assert_array_equal(result.allocation["channel_weight"], 0.0)


def test_public_fit_save_load_and_optimize_preserves_posterior_behavior(
    fitted: FittedProblem, tmp_path: Path
) -> None:
    path = tmp_path / "optimizer.zarr"
    fitted.model.save(path)
    loaded = GAM.load(path)
    media = next(
        node
        for node in walk(loaded.equations)
        if isinstance(node, Named) and node.name == "media"
    )
    kwargs = _arguments(
        fitted,
        constraints=[
            {
                "type": "eq",
                "fun": lambda e, u: (
                    u["spend"].sum("date").sum("channel")
                    - float(fitted.future["spend"].sum())
                ),
            }
        ],
        options={"ftol": 1e-11},
    )
    original = optimize(**kwargs)
    restored = optimize(
        **{**kwargs, "model": loaded, "terms": media, "inputs": [Data("spend")]}
    )
    assert original.scipy.success and restored.scipy.success
    assert original.feasible and restored.feasible
    xr.testing.assert_allclose(
        original.allocation, restored.allocation, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(original.objective, restored.objective, atol=1e-10)


@pytest.mark.parametrize("kind", ["date-only", "renamed-dimension"])
def test_decision_dimension_sets_cannot_change_after_fitting(
    fitted: FittedProblem, kind: str
) -> None:
    data = (
        fitted.future[["spend"]].assign(spend=fitted.future["spend"].sum("channel"))
        if kind == "date-only"
        else fitted.future[["spend"]].rename(channel="route")
    )
    with pytest.raises(ValueError):
        optimize(**_arguments(fitted, data=data))


def test_dated_posterior_parameters_are_selected_by_date_labels(
    fitted: FittedProblem,
) -> None:
    data = fitted.train[["price"]].isel(date=[2, 4, 7])
    kwargs = _arguments(
        fitted,
        terms=fitted.dated + fitted.price,
        inputs=[fitted.price],
        data=data,
        objective=lambda e, u: e(u)["term"].sum("date").mean("sample"),
        bounds={"price": (0.0, 2.0)},
    )
    problem = _build_problem(**kwargs)
    expected = (
        (_posterior(fitted)["dated_beta"].isel(date=[2, 4, 7]) + data["price"])
        .stack(sample=("chain", "draw"))
        .transpose("sample", "date")
    )
    evaluated = problem.evaluator.evaluate(data)["term"].transpose("sample", "date")
    np.testing.assert_array_equal(evaluated["date"].values, data["date"].values)
    np.testing.assert_allclose(evaluated.values, expected.values, atol=1e-10)
    wrong_dates = data.assign_coords(date=data["date"] + pd.Timedelta(days=1))
    with pytest.raises(ValueError):
        _build_problem(**{**kwargs, "data": wrong_dates})


def test_static_decisions_upstream_of_temporal_history_are_rejected(
    fitted: FittedProblem,
) -> None:
    term = MediaTransform(
        fitted.spend * fitted.weights,
        GeometricAdstock(l_max=L_MAX, priors={"alpha": ALPHA}, prefix="upstream"),
    )
    with pytest.raises(ValueError):
        _build_problem(
            **_arguments(
                fitted,
                terms=term,
                inputs=[fitted.spend, fitted.weights],
                data=fitted.future[["spend"]].assign(
                    channel_weight=fitted.train["channel_weight"]
                ),
                history=fitted.train[["spend"]].isel(date=slice(-2, None)),
                objective=lambda e, u: e(u)["term"].sum("date").sum("channel"),
                bounds={"spend": (0.0, 2.5), "channel_weight": (0.1, 2.0)},
            )
        )


def test_temporal_history_cannot_skip_a_measurement_period(
    fitted: FittedProblem,
) -> None:
    history = fitted.train[["spend"]].isel(date=slice(-2, None))
    history = history.assign_coords(date=history["date"] - pd.Timedelta(weeks=1))
    with pytest.raises(ValueError):
        _build_problem(**_arguments(fitted, history=history))


def test_satisfied_constant_equalities_do_not_block_a_nonconstant_objective(
    fitted: FittedProblem,
) -> None:
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.level,
            inputs=[fitted.level],
            data=xr.Dataset({"level": 0.3}),
            objective=lambda e, u: -((u["level"] - 1.3) ** 2),
            bounds={"level": (0.0, 2.0)},
            constraints=[{"type": "eq", "fun": lambda e, u: 0.0 * u["level"]}],
            options={"ftol": 1e-12},
        )
    )
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(float(result.allocation["level"]), 1.3, atol=1e-9)
    np.testing.assert_allclose(result.objective, 0.0, atol=1e-15)
    assert float(result.constraints[0]) == 0.0


def test_mixed_constant_and_dependent_equality_rows_preserve_labels_and_scales(
    fitted: FittedProblem,
) -> None:
    target = xr.DataArray([1.3, 1.4], dims="channel", coords={"channel": CHANNELS})
    mask = target.copy(data=[0.0, 1.0])
    required = target.copy(data=[0.0, 0.8])
    scales = target.copy(data=[17.0, 0.03])
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.weights,
            inputs=[fitted.weights],
            data=xr.Dataset({"channel_weight": target.copy(data=[0.2, 0.5])}),
            objective=lambda e, u: (
                -((u["channel_weight"] - e.constant(target)) ** 2).sum("channel")
            ),
            bounds={"channel_weight": (0.0, 2.0)},
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda e, u: (
                        e.constant(mask) * (u["channel_weight"] - e.constant(required))
                    ),
                    "scale": scales,
                }
            ],
            options={"ftol": 1e-12},
        )
    )
    assert result.scipy.success and result.feasible
    expected = target.copy(data=[1.3, 0.8])
    xr.testing.assert_allclose(result.allocation["channel_weight"], expected, atol=1e-9)
    np.testing.assert_allclose(result.objective, -0.36, atol=1e-12)
    xr.testing.assert_allclose(
        result.constraints[0], target.copy(data=[0.0, 0.0]), atol=1e-12
    )
    xr.testing.assert_identical(result.constraint_scales[0], scales)


@pytest.mark.parametrize("category", ["objective", "constraint"])
def test_connected_stationary_zero_expressions_require_explicit_scales(
    fitted: FittedProblem, category: str
) -> None:
    kwargs = _arguments(
        fitted,
        terms=fitted.level,
        inputs=[fitted.level],
        data=xr.Dataset({"level": 2.0}),
        bounds={"level": (0.0, 4.0)},
        objective=(
            (lambda e, u: -((u["level"] - 2.0) ** 2))
            if category == "objective"
            else (lambda e, u: -(u["level"] ** 2))
        ),
        constraints=(
            []
            if category == "objective"
            else [{"type": "eq", "fun": lambda e, u: (u["level"] - 2.0) ** 2}]
        ),
    )
    with pytest.raises(ValueError):
        _build_problem(**kwargs)
    if category == "objective":
        kwargs["scaling"] = {"objective": 4.0}
    else:
        kwargs["constraints"][0]["scale"] = 4.0
    problem = _build_problem(**kwargs)
    values = problem.raw(problem.to_z({"level": xr.DataArray(3.0)}))
    if category == "objective":
        np.testing.assert_allclose(values[0], -1.0)
        np.testing.assert_allclose(values[1], [-8.0])
        assert problem.objective_scale == 4.0
        result = problem.solve()
        assert result.scipy.success and result.feasible
        assert float(result.allocation["level"]) == 2.0
    else:
        np.testing.assert_allclose(values[2], [1.0])
        np.testing.assert_allclose(values[3], [[8.0]])
        assert float(problem.constraint_scales[0]) == 4.0


def test_fixed_zero_equality_rows_do_not_block_movable_coordinates(
    fitted: FittedProblem,
) -> None:
    required = xr.DataArray([0.0, 0.8], dims="channel", coords={"channel": CHANNELS})
    unconstrained = required.copy(data=[1.3, 1.4])
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.weights,
            inputs=[fitted.weights],
            data=xr.Dataset({"channel_weight": required.copy(data=[0.0, 0.5])}),
            objective=lambda e, u: (
                -((e(u)["term"] - e.constant(unconstrained)) ** 2).sum("channel")
            ),
            bounds={"channel_weight": (0.0, required.copy(data=[0.0, 2.0]))},
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda e, u: u["channel_weight"] - e.constant(required),
                }
            ],
            options={"ftol": 1e-12},
        )
    )
    assert result.scipy.success and result.feasible
    xr.testing.assert_allclose(result.allocation["channel_weight"], required, atol=1e-9)
    np.testing.assert_allclose(
        result.objective, -float(((required - unconstrained) ** 2).sum()), atol=1e-12
    )
    xr.testing.assert_allclose(
        result.constraints[0], result.allocation["channel_weight"] - required
    )
    xr.testing.assert_allclose(
        result.constraints[0], required.copy(data=[0.0, 0.0]), atol=1e-12
    )
    xr.testing.assert_allclose(
        result.constraint_scales[0], required.copy(data=[1.0, 2.0])
    )
    assert result.max_bound_violation == 0.0
    assert result.max_constraint_violation < 1e-10


def test_decoded_rounding_is_reported_in_original_objective_and_residual_units(
    fitted: FittedProblem,
) -> None:
    origin = 1e9
    offset = 0.3
    tolerance = 1e-9
    with pytest.warns(RuntimeWarning):
        result = optimize(
            **_arguments(
                fitted,
                terms=fitted.level,
                inputs=[fitted.level],
                data=xr.Dataset({"level": origin}),
                objective=lambda e, u: -((e(u)["term"] - origin) ** 2),
                bounds={"level": (origin, origin + 1.0)},
                scaling={"objective": 1.0},
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda e, u: u["level"] - origin - offset,
                        "scale": 1.0,
                    }
                ],
                options={"ftol": 1e-12},
                feasibility_tol=tolerance,
            )
        )
    allocation = float(result.allocation["level"])
    residual = np.float64(allocation) - np.float64(origin) - offset
    np.testing.assert_allclose(residual, -4.768371580921027e-08, rtol=0.0, atol=1e-16)
    assert result.scipy.success
    assert result.feasible is False
    assert abs(residual) > tolerance
    np.testing.assert_allclose(result.constraints[0], residual, rtol=0.0, atol=1e-16)
    np.testing.assert_allclose(
        result.objective, -((allocation - origin) ** 2), rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        result.max_constraint_violation, abs(residual), rtol=0.0, atol=1e-16
    )
    assert float(result.constraint_scales[0]) == 1.0
    assert result.max_bound_violation == 0.0


@pytest.mark.parametrize(
    "symbolic_shape", [False, True], ids=["known-axis-lengths", "runtime-axis-lengths"]
)
def test_constraint_axis_lengths_cannot_be_reshaped_to_fit_equal_total_size(
    fitted: FittedProblem, symbolic_shape: bool
) -> None:
    def constraint(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        values = pt.as_tensor_variable(np.arange(1.0, 7.0).reshape((2, 3)))
        if symbolic_shape:
            stop = pt.cast(u["spend"].sum("date").sum("channel").values, "int64")
            values = values[:stop, :stop]
        return as_xtensor(values, dims=("date", "channel"))

    data = fitted.future[["spend"]].assign(spend=xr.ones_like(fitted.future["spend"]))
    with pytest.raises(ValueError):
        optimize(
            **_arguments(
                fitted,
                data=data,
                constraints=[{"type": "ineq", "fun": constraint}],
            )
        )


@pytest.mark.parametrize(
    "indexer",
    [slice(2, None), slice(None, None, -1)],
    ids=["suffix", "permuted-full-axis"],
)
def test_history_targets_cannot_slice_or_permute_dates_before_callback_scoring(
    fitted: FittedProblem, indexer: slice
) -> None:
    history = (
        fitted.train[["spend"]]
        .isel(date=slice(-2, None))
        .assign(spend=(("date", "channel"), [[10.0, 20.0], [30.0, 40.0]]))
    )
    before = history.copy(deep=True)
    term = Transform(fitted.media, lambda value: value.isel(date=indexer))
    with pytest.raises(ValueError):
        optimize(**_arguments(fitted, terms=term, history=history))
    xr.testing.assert_identical(history, before)


def test_date_reduced_decision_factors_cannot_rewrite_adstock_history(
    fitted: FittedProblem,
) -> None:
    term = MediaTransform(
        fitted.spend * fitted.spend.sum("date"),
        GeometricAdstock(
            l_max=L_MAX, priors={"alpha": ALPHA}, prefix="reduced_upstream"
        ),
    )
    with pytest.raises(ValueError):
        optimize(
            **_arguments(
                fitted,
                terms=term,
                history=fitted.train[["spend"]].isel(date=slice(-2, None)),
                objective=lambda e, u: e(u)["term"].sum("date").sum("channel"),
            )
        )


def test_new_wrappers_align_channel_factors_to_fitted_posterior_identities(
    fitted: FittedProblem,
) -> None:
    factors = xr.DataArray([0.5, 2.0], dims="channel", coords={"channel": CHANNELS})
    term = (fitted.left * factors.isel(channel=[1, 0]) + fitted.right).named(
        "weighted_preference"
    )
    problem = _build_problem(**_arguments(fitted, terms=term))
    evaluated = problem.evaluator.evaluate(fitted.future[["spend"]])["term"].transpose(
        "sample", "date", "channel"
    )
    posterior = _posterior(fitted)
    expected = (
        (
            -posterior["preference"]
            * (
                factors * (fitted.future["spend"] - LEFT) ** 2
                + (fitted.future["spend"] - RIGHT) ** 2
            )
        )
        .stack(sample=("chain", "draw"))
        .transpose("sample", "date", "channel")
    )
    xr.testing.assert_allclose(evaluated, expected, atol=1e-10)

    result = problem.solve(options={"ftol": 1e-12})
    target = ((factors * LEFT + RIGHT) / (factors + 1.0)).broadcast_like(
        fitted.future["spend"]
    )
    assert result.scipy.success and result.feasible
    xr.testing.assert_allclose(result.allocation["spend"], target, atol=1e-6)
    independent = -float(
        (
            posterior["preference"].mean(("chain", "draw"))
            * (
                factors * (result.allocation["spend"] - LEFT) ** 2
                + (result.allocation["spend"] - RIGHT) ** 2
            )
        ).sum()
    )
    np.testing.assert_allclose(result.objective, independent, atol=1e-10)


def test_new_wrapper_factors_cannot_introduce_mismatched_channel_labels(
    fitted: FittedProblem,
) -> None:
    factors = xr.DataArray(
        [0.5, 2.0], dims="channel", coords={"channel": ["other", "radio"]}
    )
    with pytest.raises(ValueError):
        optimize(**_arguments(fitted, terms=fitted.left * factors + fitted.right))


def test_scalar_decisions_preserve_independent_dates_for_paired_posterior_terms(
    fitted: FittedProblem,
) -> None:
    dated = _posterior(fitted)["dated_beta"].isel(date=[2, 4, 7])
    data = xr.Dataset({"level": 1.0}, coords={"date": dated["date"]})
    total = 6.0
    problem = _build_problem(
        **_arguments(
            fitted,
            terms=fitted.dated + fitted.level,
            inputs=[fitted.level],
            data=data,
            objective=lambda e, u: (
                -((e(u)["term"].sum("date") - total) ** 2).mean("sample")
            ),
            bounds={"level": (0.0, 3.0)},
        )
    )
    expected = (
        (dated + data["level"])
        .stack(sample=("chain", "draw"))
        .transpose("sample", "date")
    )
    evaluated = problem.evaluator.evaluate(data)["term"].transpose("sample", "date")
    xr.testing.assert_allclose(evaluated, expected, atol=1e-10)

    result = problem.solve(options={"ftol": 1e-12})
    target = (total - float(dated.mean(("chain", "draw")).sum("date"))) / len(
        dated["date"]
    )
    assert result.scipy.success and result.feasible
    assert result.allocation["level"].dims == ()
    np.testing.assert_allclose(float(result.allocation["level"]), target, atol=1e-8)
    independent = ((dated + result.allocation["level"]).sum("date") - total) ** 2
    np.testing.assert_allclose(
        result.objective, -float(independent.mean(("chain", "draw"))), atol=1e-10
    )


def test_string_dates_align_caller_bounds_scales_and_callback_constants(
    fitted: FittedProblem,
) -> None:
    canonical = fitted.future[["price"]].assign(price=("date", [0.8, 0.7, 0.7]))
    strings = canonical.assign_coords(
        date=canonical["date"].dt.strftime("%Y-%m-%d").values
    )

    def arguments(data: xr.Dataset) -> dict[str, Any]:
        target = data["price"].copy(data=[0.6, 1.2, 0.9])
        return _arguments(
            fitted,
            terms=fitted.price,
            inputs=[fitted.price],
            data=data,
            objective=lambda e, u: (
                -((e(u)["term"] - e.constant(target)) ** 2).sum("date")
            ),
            bounds={
                "price": (
                    data["price"].copy(data=[0.2, 0.2, 0.4]),
                    data["price"].copy(data=[1.4, 1.0, 1.6]),
                )
            },
            scaling={"decisions": {"price": data["price"].copy(data=[0.3, 2.0, 0.7])}},
            options={"ftol": 1e-12},
        )

    original = optimize(**arguments(canonical))
    normalized = optimize(**arguments(strings))
    expected = canonical["price"].copy(data=[0.6, 1.0, 0.9])
    for result in (original, normalized):
        assert result.scipy.success and result.feasible
        xr.testing.assert_allclose(result.allocation["price"], expected, atol=1e-7)
        xr.testing.assert_allclose(
            result.decision_scales["price"],
            canonical["price"].copy(data=[0.3, 2.0, 0.7]),
        )
        np.testing.assert_allclose(result.objective, -0.04, atol=1e-10)
        assert result.max_bound_violation == 0.0
    xr.testing.assert_allclose(original.allocation, normalized.allocation, atol=1e-8)

    mismatched = (
        strings["price"]
        .copy(data=[0.6, 1.2, 0.9])
        .assign_coords(
            date=(canonical["date"] + pd.Timedelta(days=1))
            .dt.strftime("%Y-%m-%d")
            .values
        )
    )
    kwargs = arguments(strings)
    kwargs["objective"] = lambda e, u: (
        -((e(u)["term"] - e.constant(mismatched)) ** 2).sum("date")
    )
    with pytest.raises(ValueError):
        optimize(**kwargs)


@pytest.mark.parametrize(
    "explicit_rows", [False, True], ids=["auto", "heterogeneous-row-scales"]
)
def test_matrix_fixed_zero_tail_equalities_preserve_every_reported_row(
    fitted: FittedProblem, explicit_rows: bool
) -> None:
    required = xr.DataArray(
        [[0.8, 0.8], [0.8, 0.8], [0.0, 0.0]],
        dims=("date", "channel"),
        coords=fitted.future["spend"].coords,
    )
    upper = required.copy(data=[[2.0, 2.0], [2.0, 2.0], [0.0, 0.0]])
    initial = required.copy(data=[[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]])
    row_scales = required.copy(data=[[0.1, 7.0], [3.0, 0.02], [19.0, 0.4]])
    kwargs = _arguments(
        fitted,
        terms=fitted.spend,
        data=xr.Dataset({"spend": initial}),
        objective=lambda e, u: -((e(u)["term"] - 1.0) ** 2).sum("date").sum("channel"),
        bounds={"spend": (0.0, upper)},
        constraints=[
            {
                "type": "eq",
                "fun": lambda e, u: u["spend"] - e.constant(required),
            }
        ],
        options={"ftol": 1e-12},
    )
    if explicit_rows:
        kwargs["constraints"][0]["scale"] = row_scales.transpose(
            "channel", "date"
        ).isel(channel=[1, 0], date=[2, 0, 1])
    result = optimize(**kwargs)
    allocation = result.allocation["spend"]
    assert result.scipy.success and result.feasible
    xr.testing.assert_allclose(allocation, required, atol=1e-9)
    np.testing.assert_array_equal(allocation.isel(date=-1), 0.0)
    np.testing.assert_allclose(
        result.objective, -np.sum((allocation.values - 1.0) ** 2), atol=1e-12
    )
    xr.testing.assert_allclose(result.constraints[0], allocation - required, atol=1e-12)
    xr.testing.assert_allclose(
        result.constraints[0], xr.zeros_like(required), atol=1e-12
    )
    if explicit_rows:
        xr.testing.assert_identical(result.constraint_scales[0], row_scales)
    assert result.max_bound_violation == 0.0
    assert result.max_constraint_violation == pytest.approx(
        float(abs((allocation - required) / result.constraint_scales[0]).max())
    )


def test_all_fixed_objective_rejects_overflowing_solver_normalization(
    fitted: FittedProblem,
) -> None:
    kwargs = _arguments(
        fitted,
        terms=fitted.level,
        inputs=[fitted.level],
        data=xr.Dataset({"level": 0.0}),
        objective=lambda e, u: 1.0 + e(u)["term"],
        bounds={"level": (0.0, 0.0)},
        scaling={"objective": 1e-300},
    )
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    assert float(result.allocation["level"]) == 0.0
    assert result.objective == 1.0
    assert result.objective_scale == 1e-300
    assert np.isfinite(result.scipy.fun)
    np.testing.assert_allclose(result.scipy.fun, -1e300)
    with pytest.raises(ValueError):
        optimize(**{**kwargs, "scaling": {"objective": 1e-320}})


def test_all_fixed_constraint_rejects_overflowing_normalized_residual(
    fitted: FittedProblem,
) -> None:
    kwargs = _arguments(
        fitted,
        terms=fitted.level,
        inputs=[fitted.level],
        data=xr.Dataset({"level": 0.0}),
        objective=lambda e, u: 1.0 + e(u)["term"],
        bounds={"level": (0.0, 0.0)},
        constraints=[
            {
                "type": "eq",
                "fun": lambda e, u: u["level"] - 1.0,
                "scale": 1e-300,
            }
        ],
    )
    result = optimize(**kwargs)
    assert not result.scipy.success and not result.feasible
    assert result.objective == 1.0
    assert float(result.constraints[0]) == -1.0
    assert float(result.constraint_scales[0]) == 1e-300
    assert result.max_constraint_violation == pytest.approx(1e300)
    kwargs["constraints"][0]["scale"] = 1e-320
    with pytest.raises(ValueError):
        optimize(**kwargs)


@pytest.mark.parametrize("category", ["objective-gradient", "constraint-jacobian"])
def test_solver_normalized_derivatives_must_be_finite_when_values_are_zero(
    fitted: FittedProblem, category: str
) -> None:
    kwargs = _arguments(
        fitted,
        terms=fitted.level,
        inputs=[fitted.level],
        data=xr.Dataset({"level": 0.0}),
        objective=(
            (lambda e, u: e(u)["term"])
            if category == "objective-gradient"
            else (lambda e, u: -((e(u)["term"] - 0.5) ** 2))
        ),
        bounds={"level": (0.0, 1.0)},
        scaling={"objective": 1.0},
        constraints=(
            []
            if category == "objective-gradient"
            else [{"type": "eq", "fun": lambda e, u: u["level"], "scale": 1.0}]
        ),
        options={"ftol": 1e-12},
    )
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    assert np.isfinite(result.scipy.jac).all()
    np.testing.assert_allclose(
        float(result.allocation["level"]),
        1.0 if category == "objective-gradient" else 0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.objective,
        1.0 if category == "objective-gradient" else -0.25,
        atol=1e-10,
    )
    if category == "objective-gradient":
        kwargs["scaling"] = {"objective": 1e-320}
    else:
        np.testing.assert_allclose(result.constraints[0], 0.0, atol=1e-12)
        kwargs["constraints"][0]["scale"] = 1e-320
    with pytest.raises(ValueError):
        optimize(**kwargs)


@pytest.mark.parametrize("reduction", ["channel-sum", "scalar-channel-selection"])
def test_channel_reductions_preserve_fixed_history_and_scenario_dates(
    fitted: FittedProblem, reduction: str
) -> None:
    history = fitted.train[["spend"]].isel(date=slice(-2, None))
    before = history.copy(deep=True)
    data = fitted.future[["spend"]]
    term = (
        fitted.media.sum("channel")
        if reduction == "channel-sum"
        else Transform(fitted.media, lambda value: value.isel(channel=0))
    )
    problem = _build_problem(
        **_arguments(
            fitted,
            terms=term,
            history=history,
            objective=lambda e, u: e(u)["term"].sum("date").mean("sample"),
        )
    )
    posterior = _posterior(fitted)
    for scenario in (data, data.assign(spend=0.7 * data["spend"])):
        combined = np.concatenate([history["spend"].values, scenario["spend"].values])
        response = _media(posterior, combined)[:, :, history.sizes["date"] :]
        expected = (
            response.sum(axis=-1) if reduction == "channel-sum" else response[..., 0]
        )
        evaluated = problem.evaluator.evaluate(scenario)["term"].transpose(
            "sample", "date"
        )
        np.testing.assert_array_equal(evaluated["date"], scenario["date"])
        np.testing.assert_allclose(
            evaluated.values, expected.reshape((-1, data.sizes["date"])), atol=1e-10
        )
    result = problem.solve(options={"ftol": 1e-11})
    combined = np.concatenate(
        [history["spend"].values, result.allocation["spend"].values]
    )
    response = _media(posterior, combined)[:, :, history.sizes["date"] :]
    expected = response.sum(axis=-1) if reduction == "channel-sum" else response[..., 0]
    assert result.scipy.success and result.feasible
    np.testing.assert_array_equal(result.allocation["date"], data["date"])
    optimized = (
        result.allocation["spend"]
        if reduction == "channel-sum"
        else result.allocation["spend"].sel(channel=CHANNELS[0])
    )
    np.testing.assert_allclose(optimized, 2.5, atol=1e-7)
    np.testing.assert_allclose(
        result.objective, expected.sum(axis=-1).mean(), atol=1e-10
    )
    xr.testing.assert_identical(history, before)


def test_actual_fitted_posterior_factors_can_precede_adstock_with_history(
    fitted: FittedProblem,
) -> None:
    preference = next(
        node
        for node in walk(fitted.left)
        if isinstance(node, Parameter) and node.name == "preference"
    )
    term = MediaTransform(
        fitted.spend * preference,
        GeometricAdstock(
            l_max=L_MAX, priors={"alpha": ALPHA}, prefix="posterior_upstream"
        ),
    )
    history = fitted.train[["spend"]].isel(date=slice(-2, None))
    before = history.copy(deep=True)
    data = fitted.future[["spend"]]
    problem = _build_problem(**_arguments(fitted, terms=term, history=history))
    factors = _posterior(fitted)["preference"].transpose("chain", "draw", "channel")
    for scenario in (data, data.assign(spend=0.7 * data["spend"])):
        combined = np.concatenate([history["spend"].values, scenario["spend"].values])
        expected = (
            factors.values[:, :, None, :]
            * _adstock(combined)[None, None, history.sizes["date"] :, :]
        )
        evaluated = problem.evaluator.evaluate(scenario)["term"].transpose(
            "sample", "date", "channel"
        )
        np.testing.assert_array_equal(evaluated["date"], scenario["date"])
        np.testing.assert_allclose(
            evaluated.values,
            expected.reshape((-1, data.sizes["date"], len(CHANNELS))),
            atol=1e-10,
        )
    result = problem.solve(options={"ftol": 1e-11})
    combined = np.concatenate(
        [history["spend"].values, result.allocation["spend"].values]
    )
    expected = (
        factors.values[:, :, None, :]
        * _adstock(combined)[None, None, history.sizes["date"] :, :]
    )
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(result.allocation["spend"], 2.5, atol=1e-7)
    np.testing.assert_allclose(
        result.objective, expected.sum(axis=(2, 3)).mean(), atol=1e-10
    )
    xr.testing.assert_identical(history, before)


@pytest.mark.parametrize("factor_kind", ["date-decision", "movable-static-decision"])
def test_decision_factors_after_adstock_do_not_rewrite_measured_history(
    fitted: FittedProblem, factor_kind: str
) -> None:
    history = fitted.train[["spend"]].isel(date=slice(-2, None))
    before = history.copy(deep=True)
    data = fitted.future[["spend"]]
    inputs = [fitted.spend]
    bounds: dict[str, Any] = {"spend": (0.0, 2.5)}
    if factor_kind == "date-decision":
        term = fitted.media * fitted.spend.sum("channel")
    else:
        term = fitted.media * fitted.weights
        data = data.assign(channel_weight=fitted.train["channel_weight"])
        inputs.append(fitted.weights)
        bounds["channel_weight"] = (0.1, 2.0)
    problem = _build_problem(
        **_arguments(
            fitted,
            terms=term,
            inputs=inputs,
            data=data,
            bounds=bounds,
            history=history,
        )
    )
    changed = data.assign(spend=0.7 * data["spend"])
    if factor_kind == "movable-static-decision":
        changed = changed.assign(channel_weight=0.8 * data["channel_weight"])
    posterior = _posterior(fitted)
    for scenario in (data, changed):
        combined = np.concatenate([history["spend"].values, scenario["spend"].values])
        response = _media(posterior, combined)[:, :, history.sizes["date"] :]
        expected = response * (
            scenario["spend"].sum("channel").values[None, None, :, None]
            if factor_kind == "date-decision"
            else scenario["channel_weight"].values
        )
        evaluated = problem.evaluator.evaluate(scenario)["term"].transpose(
            "sample", "date", "channel"
        )
        np.testing.assert_array_equal(evaluated["date"], scenario["date"])
        np.testing.assert_allclose(
            evaluated.values,
            expected.reshape((-1, data.sizes["date"], len(CHANNELS))),
            atol=1e-10,
        )
    result = problem.solve(options={"ftol": 1e-11})
    allocation = result.allocation
    combined = np.concatenate([history["spend"].values, allocation["spend"].values])
    response = _media(posterior, combined)[:, :, history.sizes["date"] :]
    expected = response * (
        allocation["spend"].sum("channel").values[None, None, :, None]
        if factor_kind == "date-decision"
        else allocation["channel_weight"].values
    )
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(allocation["spend"], 2.5, atol=1e-7)
    if factor_kind == "movable-static-decision":
        np.testing.assert_allclose(allocation["channel_weight"], 2.0, atol=1e-7)
    np.testing.assert_allclose(
        result.objective, expected.sum(axis=(2, 3)).mean(), atol=1e-10
    )
    xr.testing.assert_identical(history, before)


def test_channel_mean_before_adstock_uses_shape_not_historical_decision_values(
    fitted: FittedProblem,
) -> None:
    term = MediaTransform(
        Transform(fitted.spend, lambda value: value.mean("channel")),
        GeometricAdstock(l_max=L_MAX, priors={"alpha": ALPHA}, prefix="channel_mean"),
    )
    history = fitted.train[["spend"]].isel(date=slice(-2, None))
    before = history.copy(deep=True)
    data = fitted.future[["spend"]]
    problem = _build_problem(
        **_arguments(
            fitted,
            terms=term,
            history=history,
            objective=lambda e, u: e(u)["term"].sum("date"),
        )
    )
    weights = ALPHA ** np.arange(L_MAX)
    weights /= weights.sum()
    for scenario in (data, data.assign(spend=0.7 * data["spend"])):
        combined = np.concatenate([history["spend"].values, scenario["spend"].values])
        expected = np.convolve(combined.mean(axis=1), weights)[: len(combined)][
            history.sizes["date"] :
        ]
        evaluated = problem.evaluator.evaluate(scenario)["term"]
        assert evaluated.dims == ("date",)
        np.testing.assert_array_equal(evaluated["date"], scenario["date"])
        np.testing.assert_allclose(evaluated.values, expected, atol=1e-10)
    result = problem.solve(options={"ftol": 1e-11})
    combined = np.concatenate(
        [history["spend"].values, result.allocation["spend"].values]
    )
    expected = np.convolve(combined.mean(axis=1), weights)[: len(combined)][
        history.sizes["date"] :
    ]
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(result.allocation["spend"], 2.5, atol=1e-7)
    np.testing.assert_allclose(result.objective, expected.sum(), atol=1e-10)
    xr.testing.assert_identical(history, before)


def test_named_boolean_date_mask_cannot_discard_history_before_scenario_cropping(
    fitted: FittedProblem,
) -> None:
    history = fitted.train[["spend"]].isel(date=slice(-1, None))
    before = history.copy(deep=True)
    data = fitted.future[["spend"]].isel(date=slice(0, 1))
    term = Transform(
        fitted.media, lambda value: value.isel(date=("date", np.array([False, True])))
    )
    with pytest.raises(ValueError):
        optimize(**_arguments(fitted, terms=term, history=history, data=data))
    xr.testing.assert_identical(history, before)


@pytest.mark.parametrize("kind", ["all-true-boolean", "negative-integer-identity"])
def test_complete_named_date_identity_indices_preserve_history_outputs(
    fitted: FittedProblem, kind: str
) -> None:
    history = fitted.train[["spend"]].isel(date=slice(-1, None))
    before = history.copy(deep=True)
    data = fitted.future[["spend"]].isel(date=slice(0, 1))
    length = history.sizes["date"] + data.sizes["date"]
    index = (
        np.ones(length, dtype=bool)
        if kind == "all-true-boolean"
        else np.arange(-length, 0)
    )
    term = Transform(fitted.media, lambda value: value.isel(date=("date", index)))
    problem = _build_problem(
        **_arguments(fitted, terms=term, history=history, data=data)
    )
    posterior = _posterior(fitted)
    for scenario in (data, data.assign(spend=0.7 * data["spend"])):
        combined = np.concatenate([history["spend"].values, scenario["spend"].values])
        expected = _media(posterior, combined)[:, :, history.sizes["date"] :]
        evaluated = problem.evaluator.evaluate(scenario)["term"].transpose(
            "sample", "date", "channel"
        )
        np.testing.assert_array_equal(evaluated["date"], scenario["date"])
        np.testing.assert_allclose(
            evaluated.values,
            expected.reshape((-1, data.sizes["date"], len(CHANNELS))),
            atol=1e-10,
        )
    result = problem.solve(options={"ftol": 1e-11})
    combined = np.concatenate(
        [history["spend"].values, result.allocation["spend"].values]
    )
    expected = _media(posterior, combined)[:, :, history.sizes["date"] :]
    assert result.scipy.success and result.feasible
    np.testing.assert_array_equal(result.allocation["date"], data["date"])
    np.testing.assert_allclose(result.allocation["spend"], 2.5, atol=1e-7)
    np.testing.assert_allclose(
        result.objective, expected.sum(axis=(2, 3)).mean(), atol=1e-10
    )
    xr.testing.assert_identical(history, before)


@pytest.mark.parametrize("loaded", [False, True], ids=["fitted", "save-load"])
def test_fitted_date_constants_select_training_labels_without_positional_relabeling(
    dated_constant_fitted: DatedConstantProblem, loaded: bool, tmp_path: Path
) -> None:
    fitted = dated_constant_fitted
    if loaded:
        path = tmp_path / "dated-constant-optimizer.zarr"
        fitted.model.save(path)
        model = GAM.load(path)
        term = next(
            node
            for node in walk(model.equations)
            if isinstance(node, Named) and node.name == "dated_level"
        )
        level = Data("level")
    else:
        model, term, level = fitted.model, fitted.term, fitted.level
    assert model.idata is not None
    posterior_before = model.idata["posterior"].to_dataset().copy(deep=True)
    fitted_model = model.model
    data = xr.Dataset(
        {"level": 1.0}, coords={"date": fitted.train["date"].isel(date=slice(0, 2))}
    )
    kwargs = {
        "model": model,
        "terms": term,
        "inputs": [level],
        "data": data,
        "objective": lambda e, u: -((e(u)["term"].sum("date") - 30.0) ** 2),
        "bounds": {"level": (0.0, 20.0)},
    }
    problem = _build_problem(**kwargs)
    selected = fitted.constant.sel(date=data["date"])
    for value in (1.0, 4.0, 10.0):
        evaluated = problem.evaluator.evaluate(data.assign(level=value))["term"]
        assert evaluated.dims == ("date",)
        np.testing.assert_array_equal(evaluated["date"], data["date"])
        xr.testing.assert_allclose(evaluated, value * selected, atol=1e-12)
    result = problem.solve(options={"ftol": 1e-12})
    allocation = float(result.allocation["level"])
    assert result.scipy.success and result.feasible
    assert result.allocation["level"].dims == ()
    np.testing.assert_allclose(allocation, 10.0, atol=1e-8)
    np.testing.assert_allclose(
        result.objective,
        -((allocation * float(selected.sum()) - 30.0) ** 2),
        atol=1e-12,
    )
    wrong_dates = data.assign_coords(date=data["date"] + pd.Timedelta(days=1))
    with pytest.raises(ValueError):
        optimize(**{**kwargs, "data": wrong_dates})
    assert model.model is fitted_model
    xr.testing.assert_identical(model.idata["posterior"].to_dataset(), posterior_before)


def test_new_date_constant_wrappers_align_the_full_history_and_scenario_axis(
    fitted: FittedProblem,
) -> None:
    history = fitted.train[["spend"]].isel(date=slice(-2, None))
    before = history.copy(deep=True)
    data = fitted.future[["spend"]]
    constant = xr.DataArray(
        [1.0, 2.0, 4.0, 7.0, 100.0],
        dims="date",
        coords={"date": np.concatenate([history["date"].values, data["date"].values])},
    ).isel(date=[4, 2, 0, 3, 1])
    term = (fitted.media * constant).named("dated_media")
    kwargs = _arguments(fitted, terms=term, history=history)
    problem = _build_problem(**kwargs)
    posterior = _posterior(fitted)
    factors = constant.sel(date=data["date"]).values
    for scenario in (data, data.assign(spend=0.7 * data["spend"])):
        combined = np.concatenate([history["spend"].values, scenario["spend"].values])
        expected = (
            _media(posterior, combined)[:, :, history.sizes["date"] :]
            * factors[None, None, :, None]
        )
        evaluated = problem.evaluator.evaluate(scenario)["term"].transpose(
            "sample", "date", "channel"
        )
        np.testing.assert_array_equal(evaluated["date"], scenario["date"])
        np.testing.assert_allclose(
            evaluated.values,
            expected.reshape((-1, data.sizes["date"], len(CHANNELS))),
            atol=1e-10,
        )
    result = problem.solve(options={"ftol": 1e-11})
    combined = np.concatenate(
        [history["spend"].values, result.allocation["spend"].values]
    )
    expected = (
        _media(posterior, combined)[:, :, history.sizes["date"] :]
        * factors[None, None, :, None]
    )
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(result.allocation["spend"], 2.5, atol=1e-7)
    np.testing.assert_allclose(
        result.objective, expected.sum(axis=(2, 3)).mean(), atol=1e-10
    )
    wrong_constant = constant.assign_coords(
        date=constant["date"] + pd.Timedelta(days=1)
    )
    with pytest.raises(ValueError):
        optimize(**{**kwargs, "terms": fitted.media * wrong_constant})
    xr.testing.assert_identical(history, before)


@pytest.mark.parametrize("kind", ["fitted", "new-wrapper"])
def test_selected_deterministic_constant_dimensions_require_labels(
    unlabeled_constant_fitted: UnlabeledConstantProblem, kind: str
) -> None:
    fitted = unlabeled_constant_fitted
    term = (
        fitted.unlabeled
        if kind == "fitted"
        else fitted.labeled * xr.DataArray([0.5, 2.0], dims="channel")
    )
    with pytest.raises(ValueError, match="constant dimension 'channel' must label"):
        optimize(
            model=fitted.model,
            terms=term,
            inputs=[fitted.spend],
            data=fitted.train[["spend"]].isel(date=slice(-3, None)),
            objective=_media_objective,
            bounds={"spend": (0.0, 2.5)},
        )


def test_unlabeled_prior_only_constants_do_not_restrict_fitted_parameter_outputs(
    unlabeled_constant_fitted: UnlabeledConstantProblem,
) -> None:
    fitted = unlabeled_constant_fitted
    data = fitted.train[["spend"]].isel(date=slice(-3, None))
    kwargs = {
        "model": fitted.model,
        "terms": fitted.labeled,
        "inputs": [fitted.spend],
        "data": data,
        "objective": _media_objective,
        "bounds": {"spend": (0.0, 2.5)},
    }
    problem = _build_problem(**kwargs)
    assert fitted.model.idata is not None
    posterior = fitted.model.idata["posterior"].to_dataset()
    expected = (
        (-posterior["constant_preference"] * (data["spend"] - LEFT) ** 2)
        .stack(sample=("chain", "draw"))
        .transpose("sample", "date", "channel")
    )
    evaluated = problem.evaluator.evaluate(data)["term"].transpose(
        "sample", "date", "channel"
    )
    xr.testing.assert_allclose(evaluated, expected, atol=1e-10)
    result = optimize(**kwargs, options={"ftol": 1e-12})
    assert result.scipy.success and result.feasible
    xr.testing.assert_allclose(
        result.allocation["spend"], LEFT.broadcast_like(data["spend"]), atol=1e-7
    )
    np.testing.assert_allclose(result.objective, 0.0, atol=1e-10)


def test_new_dated_wrapper_selects_permuted_training_constants_for_paired_draws(
    fitted: FittedProblem,
) -> None:
    data = fitted.train[["price"]].isel(date=[2, 4, 7])
    constant = xr.DataArray(
        np.arange(1.0, fitted.train.sizes["date"] + 1.0),
        dims="date",
        coords={"date": fitted.train["date"]},
    ).isel(date=slice(None, None, -1))
    before = constant.copy(deep=True)
    kwargs = _arguments(
        fitted,
        terms=(fitted.dated + fitted.price) * constant,
        inputs=[fitted.price],
        data=data,
        objective=lambda e, u: e(u)["term"].sum("date").mean("sample"),
        bounds={"price": (0.0, 2.0)},
    )
    problem = _build_problem(**kwargs)
    posterior = _posterior(fitted)["dated_beta"].sel(date=data["date"])
    selected = constant.sel(date=data["date"])
    for scenario in (data, data.assign(price=0.7 * data["price"])):
        expected = (
            ((posterior + scenario["price"]) * selected)
            .stack(sample=("chain", "draw"))
            .transpose("sample", "date")
        )
        evaluated = problem.evaluator.evaluate(scenario)["term"].transpose(
            "sample", "date"
        )
        xr.testing.assert_allclose(evaluated, expected, atol=1e-10)
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(result.allocation["price"], 2.0, atol=1e-7)
    independent = float(
        ((posterior + result.allocation["price"]) * selected)
        .sum("date")
        .mean(("chain", "draw"))
    )
    np.testing.assert_allclose(result.objective, independent, atol=1e-10)
    wrong_constant = constant.assign_coords(
        date=constant["date"] + pd.Timedelta(days=1)
    )
    with pytest.raises(ValueError, match="does not cover the requested 'date' labels"):
        optimize(**{**kwargs, "terms": (fitted.dated + fitted.price) * wrong_constant})
    xr.testing.assert_identical(constant, before)


@pytest.mark.parametrize("kind", ["unlabeled", "mismatched", "subset"])
def test_dated_wrapper_selection_keeps_nondate_constant_labels_strict(
    fitted: FittedProblem, kind: str
) -> None:
    labels = {
        "unlabeled": None,
        "mismatched": ["other", "radio"],
        "subset": ["tv"],
    }[kind]
    coords: dict[str, Any] = {"date": fitted.train["date"]}
    if labels is not None:
        coords["channel"] = labels
    constant = xr.DataArray(
        np.ones((fitted.train.sizes["date"], len(labels) if labels else len(CHANNELS))),
        dims=("date", "channel"),
        coords=coords,
    ).isel(date=slice(None, None, -1))
    with pytest.raises(ValueError):
        optimize(
            **_arguments(
                fitted,
                terms=(fitted.dated + fitted.price) * constant,
                inputs=[fitted.price],
                data=fitted.train[["price"]]
                .isel(date=[2, 4, 7])
                .assign_coords(channel=CHANNELS),
                objective=lambda e, u: (
                    e(u)["term"].sum("date").sum("channel").mean("sample")
                ),
                bounds={"price": (0.0, 2.0)},
            )
        )


def test_static_shape_factors_before_adstock_do_not_rewrite_history_values(
    fitted: FittedProblem,
) -> None:
    count = Transform(fitted.weights, lambda weights: as_xtensor(weights.shape[0]))
    term = MediaTransform(
        fitted.spend * count,
        GeometricAdstock(l_max=L_MAX, priors={"alpha": ALPHA}, prefix="static_shape"),
    )
    history = fitted.train[["spend"]].isel(date=slice(-2, None))
    before = history.copy(deep=True)
    data = fitted.future[["spend"]].assign(
        channel_weight=fitted.train["channel_weight"]
    )
    kwargs = _arguments(
        fitted,
        terms=term,
        inputs=[fitted.spend, fitted.weights],
        data=data,
        objective=lambda e, u: e(u)["term"].sum("date").sum("channel"),
        bounds={"spend": (0.0, 2.5), "channel_weight": (0.1, 2.0)},
        history=history,
    )
    problem = _build_problem(**kwargs)
    changed_weights = data.assign(channel_weight=("channel", [0.2, 1.9]))
    original = problem.evaluator.evaluate(data)["term"]
    xr.testing.assert_allclose(
        problem.evaluator.evaluate(changed_weights)["term"], original, atol=1e-12
    )
    for scenario in (
        data,
        changed_weights,
        changed_weights.assign(spend=0.7 * data["spend"]),
    ):
        combined = np.concatenate([history["spend"].values, scenario["spend"].values])
        expected = _adstock(combined)[history.sizes["date"] :] * len(CHANNELS)
        evaluated = problem.evaluator.evaluate(scenario)["term"].transpose(
            "date", "channel"
        )
        np.testing.assert_array_equal(evaluated["date"], scenario["date"])
        np.testing.assert_allclose(evaluated.values, expected, atol=1e-10)
    result = optimize(**kwargs, options={"ftol": 1e-11})
    combined = np.concatenate(
        [history["spend"].values, result.allocation["spend"].values]
    )
    independent = _adstock(combined)[history.sizes["date"] :] * len(CHANNELS)
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(result.allocation["spend"], 2.5, atol=1e-7)
    np.testing.assert_allclose(result.objective, independent.sum(), atol=1e-10)
    xr.testing.assert_identical(history, before)


@pytest.mark.parametrize(
    "bounds",
    [(1e-300, 2e-300), (0.0, np.inf), (-np.inf, np.inf)],
    ids=[
        "finite-endpoint-underflow",
        "one-sided-initial-underflow",
        "unbounded-initial-underflow",
    ],
)
def test_decision_scales_reject_underflowing_bounds_or_initial_values(
    fitted: FittedProblem, bounds: tuple[float, float]
) -> None:
    with pytest.raises(ValueError, match=r"underflow.*zero"):
        optimize(
            **_arguments(
                fitted,
                terms=fitted.level,
                inputs=[fitted.level],
                data=xr.Dataset({"level": 1.5e-300}),
                objective=lambda e, u: u["level"],
                bounds={"level": bounds},
                scaling={"decisions": {"level": 1e300}, "objective": 1e300},
            )
        )


def test_decision_scales_reject_nonzero_encoded_bound_collisions(
    fitted: FittedProblem,
) -> None:
    lower = float(np.nextafter(1.5, np.inf))
    upper = float(np.nextafter(lower, np.inf))
    with pytest.raises(ValueError, match="collapses a nondegenerate bound interval"):
        optimize(
            **_arguments(
                fitted,
                terms=fitted.level,
                inputs=[fitted.level],
                data=xr.Dataset({"level": lower}),
                objective=lambda e, u: u["level"],
                bounds={"level": (lower, upper)},
                scaling={"decisions": {"level": 3.0}},
            )
        )


@pytest.mark.parametrize("unit", [1e-300, 1e-320], ids=["tiny-normal", "subnormal"])
@pytest.mark.parametrize("explicit_scale", [False, True], ids=["auto", "explicit"])
def test_representable_tiny_decision_intervals_reach_the_physical_upper_bound(
    fitted: FittedProblem, unit: float, explicit_scale: bool
) -> None:
    upper = 2.0 * unit
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.level,
            inputs=[fitted.level],
            data=xr.Dataset({"level": 1.5 * unit}),
            objective=lambda e, u: u["level"],
            bounds={"level": (unit, upper)},
            scaling={"decisions": {"level": unit}} if explicit_scale else "auto",
        )
    )
    assert result.scipy.success and result.feasible
    # Absolute allclose tolerances cannot distinguish these values from an invalid zero allocation.
    assert float(result.allocation["level"]) == upper
    assert result.objective == upper
    assert float(result.decision_scales["level"]) == unit
    assert result.max_bound_violation == 0.0
    np.testing.assert_allclose(result.scipy.fun, -2.0, atol=1e-12)


def test_physically_equal_tiny_bounds_remain_fixed_with_large_decision_scales(
    fitted: FittedProblem,
) -> None:
    fixed = 1.5e-300
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.level,
            inputs=[fitted.level],
            data=xr.Dataset({"level": fixed}),
            objective=lambda e, u: u["level"],
            bounds={"level": (fixed, fixed)},
            scaling={"decisions": {"level": 1e300}},
        )
    )
    assert result.scipy.success and result.feasible
    assert result.scipy.nit == 0
    assert float(result.allocation["level"]) == fixed
    assert result.objective == fixed
    assert float(result.decision_scales["level"]) == 1e300
    assert result.max_bound_violation == 0.0
    np.testing.assert_allclose(result.scipy.fun, -1.0, atol=1e-12)


@pytest.mark.parametrize("explicit_rows", [False, True], ids=["auto", "explicit"])
def test_identity_matrix_fixed_zero_equalities_preserve_every_reported_row(
    fitted: FittedProblem, explicit_rows: bool
) -> None:
    required = xr.DataArray([0.0, 0.8], dims="channel", coords={"channel": CHANNELS})
    row_scales = required.copy(data=[17.0, 0.03])
    constraint = {
        "type": "eq",
        "fun": lambda e, u: as_xtensor(
            pt.dot(np.eye(2), u["channel_weight"].values) - required.values,
            dims=("channel",),
        ),
    }
    if explicit_rows:
        constraint["scale"] = row_scales.isel(channel=[1, 0])
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.weights,
            inputs=[fitted.weights],
            data=xr.Dataset({"channel_weight": required.copy(data=[0.0, 0.5])}),
            objective=lambda e, u: -((u["channel_weight"] - 0.8) ** 2).sum("channel"),
            bounds={"channel_weight": (0.0, required.copy(data=[0.0, 2.0]))},
            constraints=[constraint],
            options={"ftol": 1e-12},
        )
    )
    assert result.scipy.success and result.feasible
    xr.testing.assert_allclose(result.allocation["channel_weight"], required, atol=1e-9)
    np.testing.assert_allclose(result.objective, -0.64, atol=1e-12)
    xr.testing.assert_allclose(
        result.constraints[0], required.copy(data=[0.0, 0.0]), atol=1e-12
    )
    xr.testing.assert_identical(
        result.constraint_scales[0],
        row_scales if explicit_rows else required.copy(data=[1.0, 2.0]),
    )
    assert result.max_bound_violation == 0.0
    assert result.max_constraint_violation < 1e-10


def test_nonzero_constant_matrix_equalities_are_not_omitted_from_the_solver(
    fitted: FittedProblem,
) -> None:
    required = xr.DataArray([0.25, 0.8], dims="channel", coords={"channel": CHANNELS})
    row_scales = required.copy(data=[17.0, 0.03])
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.weights,
            inputs=[fitted.weights],
            data=xr.Dataset({"channel_weight": required.copy(data=[0.0, 0.5])}),
            objective=lambda e, u: -((u["channel_weight"] - 0.8) ** 2).sum("channel"),
            bounds={"channel_weight": (0.0, required.copy(data=[0.0, 2.0]))},
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda e, u: as_xtensor(
                        pt.dot(np.eye(2), u["channel_weight"].values) - required.values,
                        dims=("channel",),
                    ),
                    "scale": row_scales,
                }
            ],
        )
    )
    # Dropping a nonzero constant row would falsely permit a successful solve.
    assert not result.scipy.success
    assert not result.feasible
    assert float(result.constraints[0].sel(channel="tv")) == -0.25
    xr.testing.assert_allclose(
        result.constraints[0], result.allocation["channel_weight"] - required
    )
    xr.testing.assert_identical(result.constraint_scales[0], row_scales)
    assert result.max_constraint_violation >= 0.25 / 17.0


def test_discrete_matrix_constraint_remains_connected_despite_an_identically_zero_jacobian(
    fitted: FittedProblem,
) -> None:
    initial = xr.DataArray([0.0, 0.5], dims="channel", coords={"channel": CHANNELS})
    kwargs = _arguments(
        fitted,
        terms=fitted.weights,
        inputs=[fitted.weights],
        data=xr.Dataset({"channel_weight": initial}),
        objective=lambda e, u: -((u["channel_weight"] - 0.8) ** 2).sum("channel"),
        bounds={"channel_weight": (0.0, initial.copy(data=[0.0, 2.0]))},
        constraints=[
            {
                "type": "eq",
                "fun": lambda e, u: as_xtensor(
                    pt.dot(
                        np.eye(2),
                        pt.cast(u["channel_weight"].values > 0.75, "float64"),
                    ),
                    dims=("channel",),
                ),
            }
        ],
    )
    with pytest.raises(ValueError, match="connected constraint"):
        _build_problem(**kwargs)
    kwargs["constraints"][0]["scale"] = 1.0
    problem = _build_problem(**kwargs)
    at_start = problem.raw(problem.to_z({"channel_weight": initial}))
    changed = initial.copy(data=[0.0, 1.0])
    after_threshold = problem.raw(problem.to_z({"channel_weight": changed}))
    np.testing.assert_array_equal(at_start[2], [0.0, 0.0])
    np.testing.assert_array_equal(after_threshold[2], [0.0, 1.0])
    np.testing.assert_array_equal(at_start[3], np.zeros((2, 2)))
    np.testing.assert_array_equal(after_threshold[3], np.zeros((2, 2)))
    np.testing.assert_array_equal(problem.constraints[0].solver_rows, [1])


@pytest.mark.parametrize(
    "quantity",
    ["objective", "objective-gradient", "constraint", "constraint-jacobian"],
)
def test_solver_normalization_rejects_nonzero_underflow(
    fitted: FittedProblem, quantity: str
) -> None:
    kwargs = _arguments(
        fitted,
        terms=fitted.level,
        inputs=[fitted.level],
        data=xr.Dataset({"level": 0.5}),
        objective=lambda e, u: -((u["level"] - 1.0) ** 2),
        bounds={"level": (0.0, 1.0)},
        scaling={"decisions": {"level": 1.0}, "objective": 1.0},
    )
    if quantity.startswith("objective"):
        kwargs["objective"] = (
            (lambda e, u: 1e-300 + 0.0 * u["level"])
            if quantity == "objective"
            else (lambda e, u: 1.0 + 1e-300 * u["level"])
        )
        kwargs["scaling"]["objective"] = 1e300
    else:
        kwargs["constraints"] = [
            {
                "type": "ineq",
                "fun": (
                    (lambda e, u: 1e-300 + 0.0 * u["level"])
                    if quantity == "constraint"
                    else (lambda e, u: 1.0 + 1e-300 * u["level"])
                ),
                "scale": 1e300,
            }
        ]
    with pytest.raises(ValueError, match=r"underflow.*zero"):
        optimize(**kwargs)


@pytest.mark.parametrize("explicit_scale", [False, True], ids=["auto", "explicit"])
@pytest.mark.parametrize(
    "kind,value,feasible",
    [
        ("eq", np.int64(np.iinfo(np.int64).min), False),
        ("ineq", np.int64(np.iinfo(np.int64).min), False),
        ("ineq", np.uint64(1), True),
    ],
    ids=[
        "signed-min-equality",
        "signed-min-inequality",
        "unsigned-positive-inequality",
    ],
)
def test_integer_constraint_residuals_do_not_overflow_feasibility(
    fitted: FittedProblem, explicit_scale: bool, kind: str, value: Any, feasible: bool
) -> None:
    constraint = {"type": kind, "fun": lambda e, u: as_xtensor(value)}
    if explicit_scale:
        constraint["scale"] = 1.0
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.level,
            inputs=[fitted.level],
            data=xr.Dataset({"level": 0.0}),
            objective=lambda e, u: 0.0 * u["level"],
            bounds={"level": (0.0, 0.0)},
            constraints=[constraint],
        )
    )
    assert result.scipy.success == feasible
    assert result.feasible == feasible
    assert result.constraints[0].item() == value
    expected = (
        0.0 if feasible else abs(float(value)) / float(result.constraint_scales[0])
    )
    assert result.max_constraint_violation == expected
    assert result.max_constraint_violation >= 0.0


def test_mixed_dtype_increment_constraints_are_not_proven_constant(
    fitted: FittedProblem,
) -> None:
    initial = xr.DataArray([0.5, 0.5], dims="channel", coords={"channel": CHANNELS})
    problem = _build_problem(
        **_arguments(
            fitted,
            terms=fitted.weights,
            inputs=[fitted.weights],
            data=xr.Dataset({"channel_weight": initial}),
            objective=lambda e, u: -((u["channel_weight"] - 0.8) ** 2).sum("channel"),
            bounds={"channel_weight": (0.0, 1.0)},
            constraints=[
                {
                    "type": "eq",
                    "scale": 1.0,
                    "fun": lambda e, u: as_xtensor(
                        pt.inc_subtensor(
                            (-pt.cast(u["channel_weight"].values, "float32"))[
                                np.arange(2)
                            ],
                            u["channel_weight"].values,
                        )
                        * 1e9,
                        dims=("channel",),
                    ),
                }
            ],
        )
    )
    changed = initial.copy(data=[0.8, 0.8])
    np.testing.assert_array_equal(
        problem.raw(problem.to_z({"channel_weight": initial}))[2], [0.0, 0.0]
    )
    expected = (
        -changed.values.astype(np.float32).astype(float) + changed.values
    ).astype(np.float32) * np.float32(1e9)
    np.testing.assert_allclose(
        problem.raw(problem.to_z({"channel_weight": changed}))[2],
        expected,
        atol=1e-6,
    )
    np.testing.assert_array_equal(problem.constraints[0].solver_rows, [0, 1])
    result = problem.solve()
    # These quantized equalities have zero Jacobians; they cannot be silently dropped.
    assert not result.scipy.success
    assert result.feasible
    xr.testing.assert_allclose(result.allocation["channel_weight"], initial, atol=1e-12)


@pytest.mark.parametrize("source", ["static", "dated"])
def test_data_dependent_cardinality_cannot_change_measured_history(
    fitted: FittedProblem, source: str
) -> None:
    parent = fitted.weights if source == "static" else fitted.spend
    count = Transform(
        parent,
        lambda value: as_xtensor(pt.nonzero(value.values > 0)[0].shape[0]),
    )
    term = MediaTransform(
        fitted.spend * count,
        GeometricAdstock(l_max=L_MAX, priors={"alpha": ALPHA}, prefix="cardinality"),
    )
    data = fitted.future[["spend"]]
    inputs = [fitted.spend]
    bounds = {"spend": (0.0, 2.5)}
    if source == "static":
        data = data.assign(channel_weight=fitted.train["channel_weight"])
        inputs.append(fitted.weights)
        bounds["channel_weight"] = (-2.0, 2.0)
    with pytest.raises(ValueError, match=r"Static inputs|reduces date"):
        optimize(
            **_arguments(
                fitted,
                terms=term,
                inputs=inputs,
                data=data,
                objective=lambda e, u: e(u)["term"].sum("date").sum("channel"),
                history=fitted.train[["spend"]].isel(date=slice(-2, None)),
                bounds=bounds,
            )
        )


@pytest.mark.parametrize("paired_labels", [False, True], ids=["mismatched", "paired"])
def test_separate_output_sample_constants_preserve_joint_posterior_labels(
    fitted: FittedProblem, paired_labels: bool
) -> None:
    posterior = _posterior(fitted).stack(sample=("chain", "draw"))
    samples = posterior.get_index("sample")
    weights = xr.DataArray(
        np.linspace(0.5, 1.5, len(samples)),
        dims="sample",
        coords={"sample": samples if paired_labels else np.arange(len(samples))},
    )
    data = fitted.future[["spend"]].assign(level=1.0)
    kwargs = _arguments(
        fitted,
        terms={"left": fitted.left, "weights": fitted.level * weights},
        inputs=[fitted.spend, fitted.level],
        data=data,
        bounds={"spend": (0.0, 2.5), "level": (1.0, 1.0)},
        objective=lambda e, u: (
            (e(u)["left"] * e(u)["weights"]).sum("date").sum("channel").mean("sample")
        ),
    )
    if not paired_labels:
        with pytest.raises(ValueError, match="sample labels"):
            optimize(**kwargs)
        return
    problem = _build_problem(**kwargs)
    evaluated = problem.evaluator.evaluate(data)
    xr.testing.assert_allclose(evaluated["weights"], weights, atol=0.0)
    independent = (
        (-posterior["preference"] * (data["spend"] - LEFT) ** 2 * weights)
        .sum(("date", "channel"))
        .mean("sample")
    )
    objective, _ = problem.report(problem.initial)
    np.testing.assert_allclose(objective, float(independent), atol=1e-10)


def test_atomic_tuple_labels_preserve_fitted_order_and_decision_values() -> None:
    labels = pd.Index(
        [("north", "tv"), ("south", "radio")], name="geo", tupleize_cols=False
    )
    train = xr.Dataset(
        {"x": ("geo", [0.2, 0.8]), "y": ("geo", [0.25, 0.85])}, coords={"geo": labels}
    )
    x = Data("x")
    beta = Parameter("beta", Prior("Normal", mu=1.0, sigma=0.2))
    model = GAM(
        Equation(observed="y", mu=x * beta, likelihood=Prior("Normal", sigma=0.1))
    )
    # Native PyMC honors explicit atomic tuple coordinates during inference export.
    # Nutpie rejects tuple coordinates before the optimizer is reached.
    model.fit(
        train,
        draws=20,
        tune=20,
        chains=1,
        cores=1,
        random_seed=42,
        progressbar=False,
        compute_convergence_checks=False,
        nuts_sampler="pymc",
        idata_kwargs={"coords": {"geo": labels}},
    )
    scenario = train[["x"]].isel(geo=[1, 0])
    result = optimize(
        model=model,
        terms=x,
        inputs=[x],
        data=scenario,
        objective=lambda e, u: u["x"].sum("geo"),
        bounds={"x": (0.0, 1.0)},
    )
    assert result.scipy.success and result.feasible
    assert result.allocation.get_index("geo").equals(labels)
    np.testing.assert_allclose(result.allocation["x"], [1.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(result.objective, 2.0, atol=1e-12)
    xr.testing.assert_identical(scenario, train[["x"]].isel(geo=[1, 0]))


@pytest.mark.parametrize("reference", [False, True], ids=["direct-data", "fitted-ref"])
def test_unused_training_columns_cannot_override_fitted_input_identity(
    reference: bool,
) -> None:
    spend = Data("spend")
    signal = (spend * Parameter("beta", Prior("Normal", mu=1.0, sigma=0.2))).named(
        "signal"
    )
    train = xr.Dataset(
        {
            "spend": ("date", [0.2, 0.8, 1.0]),
            "sales": ("date", [0.25, 0.85, 1.05]),
            "_experimental_data_0": ("date", [100.0, 200.0, 300.0]),
        },
        coords={"date": pd.date_range("2025-01-01", periods=3)},
    )
    model = GAM(
        Equation(observed="sales", mu=signal, likelihood=Prior("Normal", sigma=0.1))
    )
    model.fit(
        train,
        **{**SAMPLE_KWARGS, "chains": 1, "random_seed": 42},
    )
    assert model.idata is not None
    posterior = model.idata["posterior"].to_dataset().copy(deep=True)
    result = optimize(
        model=model,
        terms=Ref("signal") if reference else spend,
        inputs=[spend],
        data=train[["spend"]],
        objective=lambda e, u: (
            e(u)["term"].sum("date").mean("sample")
            if reference
            else e(u)["term"].sum("date")
        ),
        bounds={"spend": (0.0, 2.0)},
    )
    assert result.scipy.success and result.feasible
    assert set(result.allocation.data_vars) == {"spend"}
    np.testing.assert_allclose(result.allocation["spend"], 2.0, atol=1e-8)
    coefficient = float(posterior["beta"].mean()) if reference else 1.0
    np.testing.assert_allclose(result.objective, 6.0 * coefficient, atol=1e-10)
    xr.testing.assert_identical(model.idata["posterior"].to_dataset(), posterior)


def test_fresh_input_aliases_cannot_replace_another_declared_decision(
    fitted: FittedProblem,
) -> None:
    spend, other = Data("spend"), Data("_experimental_data_1")
    data = xr.Dataset(
        {
            "spend": ("date", [1.0, 2.0, 3.0]),
            "_experimental_data_1": ("date", [10.0, 20.0, 30.0]),
        },
        coords={"date": fitted.future["date"]},
    )
    # Match the fitted spend dimension set while preserving distinct input values.
    data["spend"] = data["spend"].expand_dims(channel=CHANNELS, axis=-1)
    before = data.copy(deep=True)
    result = optimize(
        model=fitted.model,
        terms=spend.sum("channel") + other,
        inputs=[spend, other],
        data=data,
        objective=lambda e, u: e(u)["term"].sum("date"),
        bounds={name: (array, array) for name, array in data.data_vars.items()},
        scaling=None,
    )
    assert result.scipy.success and result.feasible
    independent = float((data["spend"].sum("channel") + data[other.var_name]).sum())
    np.testing.assert_allclose(result.objective, independent, atol=0.0)
    xr.testing.assert_allclose(result.allocation, data, atol=0.0)
    xr.testing.assert_identical(data, before)


def test_reduced_dated_posterior_does_not_absorb_other_output_history(
    fitted: FittedProblem,
) -> None:
    history = fitted.train[["spend"]].isel(date=slice(2, 3))
    data = fitted.train[["spend"]].isel(date=slice(3, 5)).assign(level=1.0)
    kwargs = _arguments(
        fitted,
        terms={
            "raw": fitted.spend,
            "posterior": fitted.level * fitted.dated.sum("date"),
        },
        inputs=[fitted.spend, fitted.level],
        data=data,
        history=history,
        objective=lambda e, u: e(u)["posterior"].mean("sample"),
        bounds={"spend": (data["spend"], data["spend"]), "level": (0.0, 2.0)},
    )
    with pytest.raises(ValueError, match="reduces date before history"):
        optimize(**kwargs)
    kwargs["terms"]["posterior"] = fitted.level * fitted.dated
    kwargs["objective"] = lambda e, u: e(u)["posterior"].sum("date").mean("sample")
    problem = _build_problem(**kwargs)
    expected = (
        (_posterior(fitted)["dated_beta"].isel(date=slice(3, 5)) * data["level"])
        .stack(sample=("chain", "draw"))
        .transpose("sample", "date")
    )
    evaluated = problem.evaluator.evaluate(data)["posterior"].transpose(
        "sample", "date"
    )
    xr.testing.assert_allclose(evaluated, expected, atol=1e-12)
    result = problem.solve(options={"ftol": 1e-12})
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(float(result.allocation["level"]), 2.0, atol=1e-8)
    np.testing.assert_allclose(
        result.objective, float((2.0 * expected).sum("date").mean("sample")), atol=1e-10
    )


def test_custom_shared_term_alias_keeps_its_own_decision_values(
    fitted: FittedProblem,
) -> None:
    class RawAlias(ModelTerm):
        def __init__(self, name: str) -> None:
            self.name = name

        @property
        def data_vars(self) -> tuple[str, ...]:
            return (self.name,)

        def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
            return {dim: ds[self.name].coords[dim].values for dim in ds[self.name].dims}

        def register_data(self, ds: xr.Dataset) -> None:
            model = pm.modelcontext(None)
            if self.name not in model:
                pmd.Data(self.name, ds[self.name])

        def create_variable(self) -> XTensorVariable:
            return pm.modelcontext(None)[self.name]

    price, other = Data("price"), RawAlias("_experimental_data_1")
    data = xr.Dataset(
        {
            "price": ("date", [1.0, 2.0, 3.0]),
            other.name: ("date", [10.0, 20.0, 30.0]),
        },
        coords={"date": fitted.future["date"]},
    )
    result = optimize(
        model=fitted.model,
        terms=price + other,
        inputs=[price, Data(other.name)],
        data=data,
        objective=lambda e, u: e(u)["term"].sum("date"),
        bounds={name: (array, array) for name, array in data.data_vars.items()},
        scaling=None,
    )
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(result.objective, 66.0, atol=0.0)
    xr.testing.assert_allclose(result.allocation, data, atol=0.0)


@pytest.mark.parametrize("source", ["posterior", "raw", "recorded-constant"])
def test_history_rejects_date_reduced_factors_rebroadcast_over_dates(
    fitted: FittedProblem, source: str
) -> None:
    history = fitted.train[["price"]].isel(date=slice(2, 3))
    before = history.copy(deep=True)
    data = fitted.train[["price"]].isel(date=slice(3, 5)).assign(level=1.0)
    if source == "posterior":
        factor = fitted.dated.sum("date")
    elif source == "raw":
        factor = fitted.price.sum("date")
    else:
        constant = xr.DataArray(
            [100.0, 1.0, 2.0],
            dims="date",
            coords={"date": np.concatenate([history["date"], data["date"]])},
        )
        factor = Transform(constant, lambda value: value.sum("date"))

    def score(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        value = evaluate(u)["term"].sum("date")
        return value.mean("sample") if "sample" in value.dims else value

    with pytest.raises(ValueError, match="reduces date before history"):
        optimize(
            **_arguments(
                fitted,
                terms=fitted.price * fitted.level * factor,
                inputs=[fitted.price, fitted.level],
                data=data,
                history=history,
                objective=score,
                bounds={"price": (data["price"], data["price"]), "level": (0.0, 2.0)},
            )
        )
    xr.testing.assert_identical(history, before)


def test_history_preserves_dated_factors_for_callback_only_reduction(
    fitted: FittedProblem,
) -> None:
    history = fitted.train[["price"]].isel(date=slice(2, 3))
    data = fitted.train[["price"]].isel(date=slice(3, 5)).assign(level=1.0)
    posterior = _posterior(fitted).copy(deep=True)
    result = optimize(
        **_arguments(
            fitted,
            terms=fitted.price * fitted.level * fitted.dated,
            inputs=[fitted.price, fitted.level],
            data=data,
            history=history,
            objective=lambda e, u: e(u)["term"].sum("date").mean("sample"),
            bounds={"price": (data["price"], data["price"]), "level": (0.0, 2.0)},
            options={"ftol": 1e-12},
        )
    )
    coefficient = float(
        (posterior["dated_beta"].sel(date=data["date"]) * data["price"])
        .sum("date")
        .mean(("chain", "draw"))
    )
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(float(result.allocation["level"]), 2.0, atol=1e-8)
    np.testing.assert_allclose(result.objective, 2.0 * coefficient, atol=1e-10)
    xr.testing.assert_identical(_posterior(fitted), posterior)


@pytest.mark.parametrize("output", ["factor", "selected-scalar"])
def test_dated_shape_metadata_remains_value_independent(
    fitted: FittedProblem, output: str
) -> None:
    history = fitted.train[["price"]].isel(date=slice(2, 3))
    data = fitted.train[["price"]].isel(date=slice(3, 5)).assign(level=1.0)
    count = Transform(fitted.price, lambda value: as_xtensor(value.shape[0]))
    term = (
        fitted.price * fitted.level * count
        if output == "factor"
        else fitted.level * count
    )

    def score(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        value = evaluate(u)["term"]
        return value.sum("date") if "date" in value.dims else value

    result = optimize(
        **_arguments(
            fitted,
            terms=term,
            inputs=[fitted.price, fitted.level],
            data=data,
            history=history,
            objective=score,
            bounds={"price": (data["price"], data["price"]), "level": (0.0, 2.0)},
        )
    )
    count_value = history.sizes["date"] + data.sizes["date"]
    expected = (
        2.0 * count_value * (float(data["price"].sum()) if output == "factor" else 1.0)
    )
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(float(result.allocation["level"]), 2.0, atol=1e-8)
    np.testing.assert_allclose(result.objective, expected, atol=1e-10)


def test_dynamic_nonzero_count_cannot_hide_in_a_rebroadcast_factor(
    fitted: FittedProblem,
) -> None:
    history = fitted.train[["price"]].isel(date=slice(2, 3))
    data = fitted.train[["price"]].isel(date=slice(3, 5)).assign(level=1.0)
    count = Transform(
        fitted.price,
        lambda value: as_xtensor(pt.nonzero(value.values > 0)[0].shape[0]),
    )
    with pytest.raises(ValueError, match="reduces date before history"):
        optimize(
            **_arguments(
                fitted,
                terms=fitted.price * fitted.level * count,
                inputs=[fitted.price, fitted.level],
                data=data,
                history=history,
                objective=lambda e, u: e(u)["term"].sum("date"),
                bounds={"price": (0.0, 2.0), "level": (0.0, 2.0)},
            )
        )


@dataclass
class DiscreteReductionProblem:
    model: GAM
    train: xr.Dataset
    x: Data
    total: Any
    beta: Parameter
    gamma: Parameter
    beta_total: Any
    explicit: Any
    factory: Any


@pytest.fixture(scope="module", params=["int32", "uint32", "bool"])
def discrete_reduction_fitted(
    request: pytest.FixtureRequest,
) -> DiscreteReductionProblem:
    values = np.asarray(
        [0, 1] if request.param == "bool" else [1, 3], dtype=request.param
    )
    total_value = float(values.sum())
    x = Data("x")
    total = x.sum("geo").named("discrete_total")
    beta = Parameter("discrete_beta", Prior("Normal"))
    gamma = Parameter("discrete_gamma", Prior("Normal"))
    beta_total = (beta * total).named("discrete_beta_total")
    explicit = (
        Transform(x, lambda value: value.astype("int32"))
        .sum("geo")
        .named("explicit_total")
    )
    factory = (x * Prior("Normal", mu=0.2, sigma=0.05)).named("factory_value")
    train = xr.Dataset(
        {
            "x": ("geo", values),
            "y": (
                "geo",
                0.7 * total_value
                + 1.3 * values.astype(float)
                + 0.01 * total_value
                + np.asarray([-0.02, 0.02]),
            ),
        },
        coords={"geo": ["north", "south"]},
    )
    model = GAM(
        Equation(
            observed="y",
            mu=beta_total + gamma * x + 0.01 * explicit + factory,
            likelihood=Prior("Normal", sigma=0.1),
        )
    )
    model.fit(train, **{**SAMPLE_KWARGS, "random_seed": 7132})
    return DiscreteReductionProblem(
        model, train, x, total, beta, gamma, beta_total, explicit, factory
    )


@pytest.mark.parametrize("reference", [False, True], ids=["direct", "ref"])
def test_discrete_fitted_terms_rebind_fractional_inputs_without_new_draws(
    discrete_reduction_fitted: DiscreteReductionProblem, reference: bool
) -> None:
    fitted = discrete_reduction_fitted
    assert fitted.model.idata is not None
    assert fitted.model._training_data is not None
    assert fitted.model._context is not None
    posterior = fitted.model.idata["posterior"].to_dataset().copy(deep=True)
    stored_training = fitted.model._training_data.copy(deep=True)
    containers = {
        name: fitted.model._context.model[name].get_value().copy()
        for name in fitted.model._context.data_variables.values()
    }
    data = xr.Dataset({"x": ("geo", [0.25, 0.75])}, coords={"geo": fitted.train["geo"]})
    total = Ref("discrete_total") if reference else fitted.total
    beta_total = Ref("discrete_beta_total") if reference else fitted.beta_total
    explicit = Ref("explicit_total") if reference else fitted.explicit
    factory = Ref("factory_value") if reference else fitted.factory
    kwargs = {
        "model": fitted.model,
        "terms": {
            "total": total,
            "beta_total": beta_total,
            "paired": fitted.beta * fitted.gamma * total,
            "new": total + 0.125,
            "explicit": explicit,
            "factory": factory,
        },
        "inputs": [fitted.x],
        "data": data,
        "objective": lambda e, u: e(u)["paired"].mean("sample"),
        "bounds": {"x": (data["x"], data["x"])},
    }
    evaluated = _build_problem(**kwargs).evaluator.evaluate(data)
    paired = (posterior["discrete_beta"] * posterior["discrete_gamma"]).stack(
        sample=("chain", "draw")
    )
    xr.testing.assert_allclose(
        evaluated["beta_total"],
        posterior["discrete_beta"].stack(sample=("chain", "draw")),
        atol=1e-12,
    )
    xr.testing.assert_allclose(evaluated["paired"], paired, atol=1e-12)
    assert evaluated.get_index("sample").equals(paired.get_index("sample"))
    np.testing.assert_allclose(evaluated["total"], float(data["x"].sum()), atol=0.0)
    np.testing.assert_allclose(evaluated["new"], 1.125, atol=0.0)
    np.testing.assert_allclose(
        evaluated["explicit"], float(data["x"].values.astype("int32").sum()), atol=0.0
    )
    factory_name = next(
        rv.name
        for rv in fitted.model._context.model.free_RVs
        if rv.name not in ("discrete_beta", "discrete_gamma")
    )
    expected_factory = (
        (posterior[factory_name] * data["x"])
        .stack(sample=("chain", "draw"))
        .transpose("sample", "geo")
    )
    xr.testing.assert_allclose(
        evaluated["factory"].transpose("sample", "geo"), expected_factory, atol=1e-12
    )
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(
        result.objective, float(paired.mean("sample")), atol=1e-12
    )
    xr.testing.assert_allclose(result.allocation, data, atol=0.0)
    moving = optimize(
        model=fitted.model,
        terms=total,
        inputs=[fitted.x],
        data=data,
        objective=lambda e, u: e(u)["term"],
        bounds={"x": (0.0, 0.9)},
    )
    assert moving.scipy.success and moving.feasible
    np.testing.assert_allclose(moving.allocation["x"], 0.9, atol=1e-8)
    np.testing.assert_allclose(moving.objective, 1.8, atol=1e-8)
    xr.testing.assert_identical(fitted.model.idata["posterior"].to_dataset(), posterior)
    xr.testing.assert_identical(fitted.model._training_data, stored_training)
    for name, before in containers.items():
        np.testing.assert_array_equal(
            fitted.model._context.model[name].get_value(), before
        )


@pytest.fixture(scope="module", params=["data", "shared-alias"])
def custom_continuous_fitted(request: pytest.FixtureRequest) -> dict[str, Any]:
    @dataclass
    class RawAlias(ModelTerm):
        name: str

        @property
        def data_vars(self) -> tuple[str, ...]:
            return (self.name,)

        def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
            return {dim: ds[self.name].coords[dim].values for dim in ds[self.name].dims}

        def register_data(self, ds: xr.Dataset) -> None:
            model = pm.modelcontext(None)
            if self.name not in model:
                pmd.Data(self.name, ds[self.name])

        def create_variable(self) -> XTensorVariable:
            return pm.modelcontext(None)[self.name]

    @dataclass
    class HiddenSum(ModelTerm):
        inner: Any

        def create_variable(self) -> XTensorVariable:
            return build_param(self.inner).sum("geo")

    @dataclass
    class HiddenRVSum(ModelTerm):
        inner: Any

        def create_variable(self) -> XTensorVariable:
            return build_param(self.inner).sum("geo") * pmd.Normal(
                "hidden_beta", mu=1.0, sigma=0.1
            )

    @dataclass
    class DataPrior(ModelTerm):
        inner: Any

        def create_variable(self) -> XTensorVariable:
            return pmd.Normal(
                "data_prior_beta", mu=build_param(self.inner).sum("geo"), sigma=0.2
            )

    @dataclass
    class TrainingNormalized(ModelTerm):
        inner: Any
        scale: float = 1.0

        def register_data(self, ds: xr.Dataset) -> None:
            self.scale = float(ds["x"].max())

        def create_variable(self) -> XTensorVariable:
            return (build_param(self.inner) / self.scale).sum("geo")

    @dataclass
    class HiddenCastSum(ModelTerm):
        inner: Any

        def create_variable(self) -> XTensorVariable:
            return build_param(self.inner).astype("int32").sum("geo")

    x = Data("x") if request.param == "data" else RawAlias("x")
    prior = DataPrior(x)
    terms = {
        "pure": HiddenSum(x).named("custom_sum"),
        "owned": HiddenRVSum(x).named("custom_owned"),
        "prior": (prior * x).sum("geo").named("custom_prior"),
        "normalized": TrainingNormalized(x).named("custom_normalized"),
        "explicit": HiddenCastSum(x).named("custom_explicit"),
    }
    train = xr.Dataset(
        {"x": ("geo", np.asarray([1, 3], dtype="int32")), "y": 4.0},
        coords={"geo": ["a", "b"]},
    )
    model = GAM(
        Equation(
            observed="y",
            mu=0.01 * terms["pure"]
            + terms["owned"]
            + 0.01 * terms["prior"]
            + 0.01 * terms["normalized"]
            + 0.01 * terms["explicit"],
            likelihood=Prior("Normal", sigma=0.1),
        )
    )
    model.fit(train, **{**SAMPLE_KWARGS, "random_seed": 8021})
    return {"model": model, "train": train, "terms": terms, "prior": prior}


@pytest.mark.parametrize("reference", [False, True], ids=["direct", "ref"])
def test_custom_integer_training_inputs_remain_continuous_and_fitted(
    custom_continuous_fitted: dict[str, Any], reference: bool
) -> None:
    fitted = custom_continuous_fitted
    model, train = fitted["model"], fitted["train"]
    terms = {
        name: Ref(term.name) if reference else term
        for name, term in fitted["terms"].items()
    }
    before = model.idata["posterior"].to_dataset().copy(deep=True)
    data = xr.Dataset({"x": ("geo", [0.25, 0.75])}, coords={"geo": train["geo"]})
    kwargs = {
        "model": model,
        "terms": terms,
        "inputs": [Data("x")],
        "data": data,
        "objective": lambda e, u: e(u)["owned"].mean("sample"),
        "bounds": {"x": (data["x"], data["x"])},
    }
    evaluated = _build_problem(**kwargs).evaluator.evaluate(data)
    np.testing.assert_allclose(evaluated["pure"], 1.0, atol=0.0)
    np.testing.assert_allclose(evaluated["normalized"], 1.0 / float(train["x"].max()))
    np.testing.assert_allclose(evaluated["explicit"], 0.0, atol=0.0)
    xr.testing.assert_allclose(
        evaluated["owned"], before["hidden_beta"].stack(sample=("chain", "draw"))
    )
    xr.testing.assert_allclose(
        evaluated["prior"], before["data_prior_beta"].stack(sample=("chain", "draw"))
    )
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(result.objective, float(before["hidden_beta"].mean()))
    xr.testing.assert_allclose(result.allocation, data, atol=0.0)
    moving = optimize(
        model=model,
        terms=terms["pure"],
        inputs=[Data("x")],
        data=data,
        objective=lambda e, u: e(u)["term"],
        bounds={"x": (0.0, 0.9)},
    )
    assert moving.scipy.success and moving.feasible
    np.testing.assert_allclose(moving.objective, 1.8, atol=1e-8)
    np.testing.assert_allclose(moving.allocation["x"], 0.9, atol=1e-8)
    xr.testing.assert_identical(model.idata["posterior"].to_dataset(), before)
    xr.testing.assert_identical(model._training_data, train)


@pytest.fixture(scope="module")
def retained_input_fitted() -> dict[str, Any]:
    class RegisteredSum(ModelTerm):
        @property
        def data_vars(self) -> tuple[str, ...]:
            return ("x",)

        def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
            return {"geo": ds["geo"].values}

        def register_data(self, ds: xr.Dataset) -> None:
            self.value = pmd.Data("x_shared", ds["x"])

        def create_variable(self) -> XTensorVariable:
            return self.value.sum("geo")

    total = RegisteredSum().named("registered_sum")
    beta = Parameter("registered_beta", Prior("Normal", sigma=0.2))
    train = xr.Dataset(
        {
            "x": ("geo", np.asarray([1, 3], dtype="int32")),
            "y": 4.0,
            "canonical_y": 4.0,
        },
        coords={"geo": ["a", "b"]},
    )
    model = GAM(
        Equation(
            observed="canonical_y",
            mu=Data("x").sum("geo"),
            likelihood=Prior("Normal", sigma=0.1),
        ),
        Equation(observed="y", mu=total + beta, likelihood=Prior("Normal", sigma=0.1)),
    )
    model.fit(train, **{**SAMPLE_KWARGS, "random_seed": 4821})
    return {"model": model, "train": train, "total": total, "beta": beta}


@pytest.mark.parametrize("reference", [False, True], ids=["direct", "ref"])
def test_retained_registered_inputs_remain_declared_and_continuous(
    retained_input_fitted: dict[str, Any], reference: bool
) -> None:
    fitted = retained_input_fitted
    model, train = fitted["model"], fitted["train"]
    posterior = model.idata["posterior"].to_dataset().copy(deep=True)
    containers = {
        variable.name: variable.get_value().copy()
        for variable in model._context.model.data_vars
    }
    training = train.copy(deep=True)

    def assert_data_unchanged() -> None:
        for name, previous in containers.items():
            np.testing.assert_array_equal(
                model._context.model[name].get_value(), previous
            )

    total = Ref("registered_sum") if reference else fitted["total"]
    data = xr.Dataset({"x": ("geo", [0.25, 0.75])}, coords={"geo": train["geo"]})
    before = data.copy(deep=True)
    kwargs = {
        "model": model,
        "terms": {"raw": total, "posterior": total * fitted["beta"]},
        "inputs": [Data("x")],
        "data": data,
        "objective": lambda e, u: e(u)["raw"],
        "bounds": {"x": (data["x"], data["x"])},
    }
    evaluated = _build_problem(**kwargs).evaluator.evaluate(data)
    assert_data_unchanged()
    assert float(evaluated["raw"]) == 1.0
    xr.testing.assert_allclose(
        evaluated["posterior"],
        posterior["registered_beta"].stack(sample=("chain", "draw")),
        atol=1e-12,
    )
    fixed = optimize(**kwargs)
    assert_data_unchanged()
    assert fixed.scipy.success and fixed.feasible
    assert fixed.objective == 1.0
    xr.testing.assert_allclose(fixed.allocation, data, atol=0.0, rtol=0.0)
    moving = optimize(**{**kwargs, "bounds": {"x": (0.0, 0.9)}})
    assert_data_unchanged()
    assert moving.scipy.success and moving.feasible
    np.testing.assert_allclose(moving.allocation["x"], 0.9, atol=1e-8)
    np.testing.assert_allclose(moving.objective, 1.8, atol=1e-8)
    combined_kwargs = {
        **kwargs,
        "terms": {**kwargs["terms"], "combined": total + Data("x").sum("geo")},
        "objective": lambda e, u: e(u)["combined"],
        "bounds": {"x": (0.0, 0.9)},
    }
    evaluated_combined = _build_problem(**combined_kwargs).evaluator.evaluate(data)
    assert_data_unchanged()
    assert float(evaluated_combined["combined"]) == 2.0
    combined = optimize(**combined_kwargs)
    assert_data_unchanged()
    assert combined.scipy.success and combined.feasible
    np.testing.assert_allclose(combined.allocation["x"], 0.9, atol=1e-8)
    np.testing.assert_allclose(combined.objective, 3.6, atol=1e-8)
    xr.testing.assert_identical(data, before)
    xr.testing.assert_identical(model.idata["posterior"].to_dataset(), posterior)
    xr.testing.assert_identical(train, training)
    xr.testing.assert_identical(model._training_data, training)
    assert_data_unchanged()


@pytest.fixture(scope="module")
def fitted_fourier_history() -> dict[str, Any]:
    from pymc_marketing.mmm import YearlyFourier

    dates = pd.date_range("2025-01-06", periods=4, freq="W-MON")
    train = xr.Dataset(
        {
            "level": 1.0,
            "spend": ("date", [1.0, 2.0, 3.0, 4.0]),
            "y": ("date", [2.0, 3.0, 4.0, 5.0]),
        },
        coords={"date": dates},
    )
    level, spend = Data("level"), Data("spend")
    fourier = YearlyFourier(n_order=1)
    seasonal = (level + fourier).named("seasonal_level")
    model = GAM(
        Equation(
            observed="y", mu=seasonal + spend, likelihood=Prior("Normal", sigma=1.0)
        )
    )
    model.fit(train, **{**SAMPLE_KWARGS, "random_seed": 871})
    return {
        "model": model,
        "train": train,
        "level": level,
        "spend": spend,
        "fourier": fourier,
        "seasonal": seasonal,
    }


@pytest.mark.parametrize("reference", [False, True], ids=["direct", "ref"])
def test_fitted_fourier_history_retains_scenario_values_and_rejects_reductions(
    fitted_fourier_history: dict[str, Any], reference: bool
) -> None:
    fitted = fitted_fourier_history
    model, train = fitted["model"], fitted["train"]
    history = train[["spend"]].isel(date=slice(0, 1))
    data = train[["spend"]].isel(date=slice(1, 3)).assign(level=1.0)
    seasonal = Ref("seasonal_level") if reference else fitted["seasonal"]
    posterior = model.idata["posterior"].to_dataset().copy(deep=True)
    modes = np.column_stack(
        [
            np.sin(2 * np.pi * data["date"].dt.dayofyear / 365.25),
            np.cos(2 * np.pi * data["date"].dt.dayofyear / 365.25),
        ]
    )
    beta = posterior[str(fitted["fourier"].variable_name)].values.reshape((-1, 2))
    expected = 1.0 + beta @ modes.T

    def score(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        value = evaluate(u)["seasonal"]
        return (
            value.sum("date").mean("sample")
            if "date" in value.dims
            else value.mean("sample")
        )

    kwargs = {
        "model": model,
        "terms": {"raw": fitted["spend"], "seasonal": seasonal},
        "inputs": [fitted["level"], fitted["spend"]],
        "data": data,
        "history": history,
        "objective": score,
        "bounds": {name: (array, array) for name, array in data.data_vars.items()},
    }
    values = _build_problem(**kwargs).evaluator.evaluate(data)["seasonal"]
    np.testing.assert_allclose(values.transpose("sample", "date"), expected, atol=1e-12)
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(
        result.objective, expected.sum(axis=-1).mean(), atol=1e-12
    )
    with pytest.raises(ValueError):
        optimize(
            **{
                **kwargs,
                "terms": {"raw": fitted["spend"], "seasonal": seasonal.sum("date")},
            }
        )
    xr.testing.assert_identical(model.idata["posterior"].to_dataset(), posterior)


@pytest.mark.parametrize("shape", ["xtensor-shape", "tensor-shape-i"])
@pytest.mark.parametrize(
    "dated_constant", [False, True], ids=["plain", "dated-constant"]
)
def test_fourier_date_count_is_fixed_metadata_not_historical_values(
    fitted_fourier_history: dict[str, Any], shape: str, dated_constant: bool
) -> None:
    fitted = fitted_fourier_history
    model = fitted["model"]
    history = fitted["train"][["spend"]].isel(date=slice(0, 1))
    data = fitted["train"][["spend"]].isel(date=slice(1, 3))
    posterior = model.idata["posterior"].to_dataset().copy(deep=True)

    def shape_count(value: XTensorVariable) -> XTensorVariable:
        size = value.shape[0] if shape == "xtensor-shape" else Shape_i(0)(value.values)
        return as_xtensor(size)

    count = Transform(fitted["fourier"], shape_count)
    terms = {
        "count": fitted["spend"] * count,
        "seasonal": fitted["spend"] * fitted["fourier"],
    }
    if dated_constant:
        combined = xr.concat([history["spend"], data["spend"]], dim="date")
        terms["extra"] = fitted["spend"] * xr.ones_like(combined)

    def score(evaluate: Any, u: dict[str, XTensorVariable]) -> XTensorVariable:
        value = evaluate(u)["count"].sum("date")
        return value.mean("sample") if "sample" in value.dims else value

    kwargs = {
        "model": model,
        "terms": terms,
        "inputs": [fitted["spend"]],
        "data": data,
        "history": history,
        "objective": score,
        "bounds": {"spend": (data["spend"], data["spend"])},
    }
    count_value = len(history["date"]) + len(data["date"])
    evaluated = _build_problem(**kwargs).evaluator.evaluate(data)
    xr.testing.assert_allclose(
        evaluated["count"], data["spend"] * count_value, atol=0.0, rtol=0.0
    )
    modes = np.column_stack(
        [
            np.sin(2 * np.pi * data["date"].dt.dayofyear / 365.25),
            np.cos(2 * np.pi * data["date"].dt.dayofyear / 365.25),
        ]
    )
    beta = posterior[str(fitted["fourier"].variable_name)].values.reshape((-1, 2))
    expected_seasonal = (beta @ modes.T) * data["spend"].values
    np.testing.assert_allclose(
        evaluated["seasonal"].transpose("sample", "date"),
        expected_seasonal,
        atol=1e-12,
    )
    if dated_constant:
        xr.testing.assert_allclose(
            evaluated["extra"], data["spend"], atol=0.0, rtol=0.0
        )
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(
        result.objective, float(data["spend"].sum()) * count_value, atol=1e-12
    )
    xr.testing.assert_identical(model.idata["posterior"].to_dataset(), posterior)


@pytest.mark.parametrize("selection", ["slice", "constant-vector"])
def test_indexed_channel_means_keep_measured_history_fixed(
    fitted: FittedProblem, selection: str
) -> None:
    index = slice(0, 1) if selection == "slice" else ("channel", np.array([0]))
    mean = Transform(
        fitted.spend, lambda value: value.isel(channel=index).mean("channel")
    )
    term = MediaTransform(
        mean,
        GeometricAdstock(l_max=L_MAX, priors={"alpha": ALPHA}, prefix="indexed_mean"),
    )
    history = fitted.train[["spend"]].isel(date=slice(-2, None))
    before = history.copy(deep=True)
    data = fitted.future[["spend"]]
    result = optimize(
        model=fitted.model,
        terms=term,
        inputs=[fitted.spend],
        data=data,
        history=history,
        objective=lambda e, u: e(u)["term"].sum("date"),
        bounds={"spend": (0.0, 2.5)},
    )
    values = np.concatenate(
        [
            history["spend"].isel(channel=0).values,
            result.allocation["spend"].isel(channel=0).values,
        ]
    )
    weights = ALPHA ** np.arange(L_MAX)
    weights /= weights.sum()
    expected = np.convolve(values, weights)[: len(values)][history.sizes["date"] :]
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(
        result.allocation["spend"].isel(channel=0), 2.5, atol=1e-8
    )
    np.testing.assert_allclose(result.objective, expected.sum(), atol=1e-10)
    xr.testing.assert_identical(history, before)


@pytest.mark.parametrize("reference", [False, True], ids=["direct", "ref"])
def test_frozen_custom_prior_does_not_require_its_training_inputs(
    custom_continuous_fitted: dict[str, Any], reference: bool
) -> None:
    fitted = custom_continuous_fitted
    model = fitted["model"]
    posterior = model.idata["posterior"].to_dataset().copy(deep=True)
    term = Ref("data_prior_beta") if reference else fitted["prior"]
    level = Data("level")
    data = xr.Dataset({"level": 1.0}, coords={"geo": fitted["train"]["geo"]})
    kwargs = {
        "model": model,
        "terms": term * level,
        "inputs": [level],
        "data": data,
        "objective": lambda e, u: e(u)["term"].mean("sample"),
        "bounds": {"level": (0.0, 2.0)},
    }
    values = _build_problem(**kwargs).evaluator.evaluate(data)["term"]
    expected = posterior["data_prior_beta"].stack(sample=("chain", "draw"))
    xr.testing.assert_allclose(values, expected, atol=1e-12)
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(
        result.objective, 2.0 * float(expected.mean("sample")), atol=1e-12
    )
    np.testing.assert_allclose(float(result.allocation["level"]), 2.0, atol=1e-8)
    xr.testing.assert_identical(model.idata["posterior"].to_dataset(), posterior)


def test_posterior_channel_mean_uses_channel_count_not_sample_count(
    fitted: FittedProblem,
) -> None:
    level = Data("level")
    data = xr.Dataset({"level": 1.0}, coords={"channel": fitted.train["channel"]})
    term = (
        Transform(Ref("saturation_beta"), lambda value: value.mean("channel")) * level
    )
    kwargs = _arguments(
        fitted,
        terms=term,
        inputs=[level],
        data=data,
        objective=lambda e, u: e(u)["term"].mean("sample"),
        bounds={"level": (1.0, 1.0)},
    )
    expected = (
        _posterior(fitted)["saturation_beta"]
        .mean("channel")
        .stack(sample=("chain", "draw"))
    )
    evaluated = _build_problem(**kwargs).evaluator.evaluate(data)["term"]
    xr.testing.assert_allclose(evaluated, expected, atol=1e-12)
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(
        result.objective, float(expected.mean("sample")), atol=1e-12
    )


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_noncanonical_fitted_inputs_reject_only_reachable_value_paths(
    dtype: str,
) -> None:
    class TypedShared(ModelTerm):
        @property
        def data_vars(self) -> tuple[str, ...]:
            return ("x",)

        def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
            return {"geo": ds["geo"].values}

        def register_data(self, ds: xr.Dataset) -> None:
            if "x" not in pm.modelcontext(None):
                pmd.Data("x", ds["x"].astype(dtype))

        def create_variable(self) -> XTensorVariable:
            value = pm.modelcontext(None)["x"]
            beta = pmd.Normal("typed_beta", mu=value.sum("geo"), sigma=0.2)
            return value.sum("geo") * beta

    term = TypedShared().named("typed_signal")
    train = xr.Dataset({"x": ("geo", [1.0, 3.0]), "y": 4.0}, coords={"geo": ["a", "b"]})
    model = GAM(Equation(observed="y", mu=term, likelihood=Prior("Normal", sigma=0.1)))
    model.fit(train, **{**SAMPLE_KWARGS, "random_seed": 9201})
    posterior = model.idata["posterior"].to_dataset().copy(deep=True)
    data = xr.Dataset({"x": ("geo", [0.25, 0.75])}, coords={"geo": train["geo"]})
    for selected in (term, Ref("typed_signal")):
        with pytest.raises(ValueError):
            optimize(
                model=model,
                terms=selected,
                inputs=[Data("x")],
                data=data,
                objective=lambda e, u: e(u)["term"].mean("sample"),
                bounds={"x": (data["x"], data["x"])},
            )
    fresh = Data("x")
    result = optimize(
        model=model,
        terms=fresh.sum("geo"),
        inputs=[fresh],
        data=data,
        objective=lambda e, u: e(u)["term"],
        bounds={"x": (data["x"], data["x"])},
    )
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(result.objective, 1.0, atol=0.0)
    # Its prior reads typed x, but selected fitted draws do not read or update x.
    level = Data("level")
    result = optimize(
        model=model,
        terms=Ref("typed_beta") * level,
        inputs=[level],
        data=xr.Dataset({"level": 1.0}, coords={"geo": train["geo"]}),
        objective=lambda e, u: e(u)["term"].mean("sample"),
        bounds={"level": (1.0, 1.0)},
    )
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(result.objective, float(posterior["typed_beta"].mean()))
    xr.testing.assert_identical(model.idata["posterior"].to_dataset(), posterior)


def test_canonical_integer_input_graph_survives_verified_save_load(
    discrete_reduction_fitted: DiscreteReductionProblem,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted = discrete_reduction_fitted
    monkeypatch.setitem(
        CUSTOM_TRANSFORMS, "optimizer_cast_int32", fitted.explicit.expr.inner.func
    )
    filename = tmp_path / "canonical_inputs.zarr"
    fitted.model.save(filename)
    loaded = GAM.load(filename)
    xr.testing.assert_identical(
        loaded.idata["posterior"].to_dataset(),
        fitted.model.idata["posterior"].to_dataset(),
    )
    xr.testing.assert_identical(loaded._training_data, fitted.train)
    data = xr.Dataset({"x": ("geo", [0.25, 0.75])}, coords={"geo": fitted.train["geo"]})
    result = optimize(
        model=loaded,
        terms=Ref("discrete_total"),
        inputs=[Data("x")],
        data=data,
        objective=lambda e, u: e(u)["term"],
        bounds={"x": (0.0, 0.9)},
    )
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(result.objective, 1.8, atol=1e-8)
    np.testing.assert_allclose(result.allocation["x"], 0.9, atol=1e-8)


def test_dynamic_channel_index_mean_cannot_rewrite_measured_history(
    fitted: FittedProblem,
) -> None:
    def dynamic_mean(value: XTensorVariable) -> XTensorVariable:
        positions = pt.nonzero(value.sum("date").values > 0)[0]
        return value.isel(channel=("channel", positions)).mean("channel")

    term = MediaTransform(
        Transform(fitted.spend, dynamic_mean),
        GeometricAdstock(l_max=L_MAX, priors={"alpha": ALPHA}, prefix="dynamic_mean"),
    )
    with pytest.raises(ValueError):
        optimize(
            model=fitted.model,
            terms=term,
            inputs=[fitted.spend],
            data=fitted.future[["spend"]],
            history=fitted.train[["spend"]].isel(date=slice(-2, None)),
            objective=lambda e, u: e(u)["term"].sum("date"),
            bounds={"spend": (0.0, 2.5)},
        )


@pytest.mark.parametrize("power", [53, 64], ids=["numpy-integer", "python-integer"])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("labeled", [False, True], ids=["scalar", "labeled"])
def test_integer_bounds_cannot_round_the_original_feasible_set(
    fitted: FittedProblem, sign: int, labeled: bool, power: int
) -> None:
    level = Data("level")
    initial = np.int64(sign * 2**power) if power == 53 else float(sign * 2**power)
    raw = np.int64(sign * (2**power + 1)) if power == 53 else sign * (2**power + 1)
    bound = xr.DataArray(raw) if labeled else raw
    data = xr.Dataset({"level": initial})
    before = data.copy(deep=True)
    with pytest.raises(ValueError, match="exactly representable as float64"):
        optimize(
            model=fitted.model,
            terms=level,
            inputs=[level],
            data=data,
            objective=lambda e, u: e(u)["term"],
            bounds={"level": (bound, bound)},
        )
    xr.testing.assert_identical(data, before)


@pytest.mark.parametrize("power", [54, 65], ids=["numpy-integer", "python-integer"])
@pytest.mark.parametrize("labeled", [False, True], ids=["scalar", "labeled"])
def test_exact_large_integer_bounds_preserve_the_original_value(
    fitted: FittedProblem, labeled: bool, power: int
) -> None:
    level = Data("level")
    raw = np.int64(2**power) if power == 54 else 2**power
    bound = xr.DataArray(raw) if labeled else raw
    result = optimize(
        model=fitted.model,
        terms=level,
        inputs=[level],
        data=xr.Dataset({"level": float(raw)}),
        objective=lambda e, u: e(u)["term"],
        bounds={"level": (bound, bound)},
    )
    assert result.scipy.success and result.feasible
    assert int(result.allocation["level"]) == int(raw)
    assert result.objective == float(raw)


@pytest.mark.parametrize("object_dtype", [False, True], ids=["complex", "object"])
def test_complex_labeled_bounds_cannot_discard_their_imaginary_component(
    fitted: FittedProblem, object_dtype: bool
) -> None:
    level = Data("level")
    bound = xr.DataArray(
        np.array(np.complex128(1.0 + 1.0j), dtype=object if object_dtype else complex)
    )
    with pytest.raises(ValueError, match="real numeric values"):
        optimize(
            model=fitted.model,
            terms=level,
            inputs=[level],
            data=xr.Dataset({"level": 1.0}),
            objective=lambda e, u: e(u)["term"],
            bounds={"level": (bound, bound)},
        )


@pytest.fixture(scope="module")
def hierarchical_coefficient_fitted() -> dict[str, Any]:
    @dataclass
    class HierarchicalCoefficient(ModelTerm):
        prior_mu: Any

        def create_variable(self) -> XTensorVariable:
            return pmd.Normal("hier_beta", mu=build_param(self.prior_mu), sigma=1.0)

    prior_mu = Equation(name="hier_mu", mu=0.0, likelihood=Prior("Normal", sigma=1.0))
    coefficient = HierarchicalCoefficient(prior_mu)
    level = Data("level")
    train = xr.Dataset({"level": 1.0, "y": 1.0})
    model = GAM(
        Equation(
            observed="y",
            mu=coefficient * level,
            likelihood=Prior("Normal", sigma=0.1),
        )
    )
    model.fit(train, **{**SAMPLE_KWARGS, "random_seed": 881})
    return {
        "model": model,
        "coefficient": coefficient,
        "prior_mu": prior_mu,
        "level": level,
        "train": train,
    }


@pytest.mark.parametrize("reference", [False, True], ids=["direct", "ref"])
def test_frozen_hierarchical_coefficients_do_not_read_prior_only_equations(
    hierarchical_coefficient_fitted: dict[str, Any], reference: bool
) -> None:
    fitted = hierarchical_coefficient_fitted
    model, level = fitted["model"], fitted["level"]
    posterior = model.idata["posterior"].to_dataset().copy(deep=True)
    coefficient = Ref("hier_beta") if reference else fitted["coefficient"]
    data = fitted["train"][["level"]]
    kwargs = {
        "model": model,
        "terms": coefficient * level,
        "inputs": [level],
        "data": data,
        "objective": lambda e, u: e(u)["term"].mean("sample"),
        "bounds": {"level": (1.0, 1.0)},
    }
    evaluated = _build_problem(**kwargs).evaluator.evaluate(data)
    xr.testing.assert_allclose(
        evaluated["term"],
        posterior["hier_beta"].stack(sample=("chain", "draw")),
        atol=1e-12,
    )
    result = optimize(**kwargs)
    assert result.scipy.success and result.feasible
    np.testing.assert_allclose(
        result.objective, float(posterior["hier_beta"].mean()), atol=1e-12
    )
    prior_mu = Ref("hier_mu") if reference else fitted["prior_mu"]
    with pytest.raises(ValueError, match="stochastic Equation"):
        optimize(**{**kwargs, "terms": prior_mu * level})
    xr.testing.assert_identical(model.idata["posterior"].to_dataset(), posterior)
    xr.testing.assert_identical(model._training_data, fitted["train"])
