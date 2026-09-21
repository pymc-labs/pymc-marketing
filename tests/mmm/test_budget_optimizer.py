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
import ast
import inspect
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytest
import xarray as xr
from pydantic import ValidationError
from pytensor.compile import UnusedInputError
from pytensor.graph.traversal import ancestors
from scipy.optimize import OptimizeResult
from xarray import DataArray

import pymc_marketing.mmm.budget_optimizer as budget_optimizer_module
from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.additive_effect import MuEffect
from pymc_marketing.mmm.budget_optimizer import (
    BudgetOptimizationResult,
    BudgetOptimizer,
    CustomModelWrapper,
    MinimizeException,
    optimizer_xarray_builder,
)
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.constraints import Constraint, build_default_sum_constraint
from pymc_marketing.mmm.optimization_variables import FLAT_DIM
from pymc_marketing.mmm.utility import (
    _check_samples_dimensionality,
    diversification_ratio,
)


@pytest.fixture(scope="module")
def dummy_df():
    n = 10
    df = pd.DataFrame(
        data={
            "date_week": pd.date_range(start=pd.Timestamp.today(), periods=n, freq="W"),
            "channel_1": np.linspace(0, 1, num=n),
            "channel_2": np.linspace(0, 1, num=n),
            "event_1": np.concatenate([np.zeros(n - 1), [1]]),
            "event_2": np.concatenate([[1], np.zeros(n - 1)]),
            "t": range(n),
        }
    )

    y = pd.Series(np.ones(n), name="y")

    df_kwargs = {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "control_columns": ["event_1", "event_2", "t"],
    }

    return df_kwargs, df, y


@pytest.fixture(scope="module")
def dummy_idata(dummy_df) -> xr.DataTree:
    df_kwargs, _df, _y = dummy_df

    channels = df_kwargs["channel_columns"]
    chain_coord = [0, 1]
    draw_coord = [0, 1]
    date_coord = [0, 1]

    return xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset(
                {
                    "saturation_lam": xr.DataArray(
                        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
                        dims=["chain", "draw", "channel"],
                        coords={
                            "chain": chain_coord,
                            "draw": draw_coord,
                            "channel": channels,
                        },
                    ),
                    "saturation_beta": xr.DataArray(
                        [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]],
                        dims=["chain", "draw", "channel"],
                        coords={
                            "chain": chain_coord,
                            "draw": draw_coord,
                            "channel": channels,
                        },
                    ),
                    "adstock_alpha": xr.DataArray(
                        [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]],
                        dims=["chain", "draw", "channel"],
                        coords={
                            "chain": chain_coord,
                            "draw": draw_coord,
                            "channel": channels,
                        },
                    ),
                    "channel_contribution": xr.DataArray(
                        np.array(
                            [
                                [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                                [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                            ]
                        ),
                        dims=["chain", "draw", "channel", "date"],
                        coords={
                            "chain": chain_coord,
                            "draw": draw_coord,
                            "channel": channels,
                            "date": date_coord,
                        },
                    ),
                }
            ),
        }
    )


@pytest.fixture(scope="module")
def mmm_wrapper(dummy_df, dummy_idata) -> CustomModelWrapper:
    """Build an MMM, then wrap it for the BudgetOptimizer protocol."""
    df_kwargs, X_dummy, y_dummy = dummy_df
    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )
    mmm.build_model(X=X_dummy, y=y_dummy)
    return CustomModelWrapper(
        base_model=mmm.model,
        idata=dummy_idata,
        channels=df_kwargs["channel_columns"],
    )


@pytest.mark.parametrize(
    argnames="total_budget, budget_bounds, x0, parameters, minimize_kwargs, expected_optimal, expected_response",
    argvalues=[
        (
            100,
            None,
            None,
            {
                "saturation_params": {
                    "lam": np.array(
                        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
                    ),  # dims: chain, draw, channel
                    "beta": np.array(
                        [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]]
                    ),  # dims: chain, draw, channel
                },
                "adstock_params": {
                    "alpha": np.array(
                        [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]]
                    )  # dims: chain, draw, channel
                },
                "channel_contribution": np.array(
                    [
                        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                    ]
                ),  # dims: chain, draw, channel, date
            },
            None,
            {"channel_1": 58.97600120944057, "channel_2": 41.02399879055943},
            44.94,
        ),
        # set x0 manually
        (
            100,
            None,
            np.array([50, 50]),
            {
                "saturation_params": {
                    "lam": np.array(
                        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
                    ),  # dims: chain, draw, channel
                    "beta": np.array(
                        [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]]
                    ),  # dims: chain, draw, channel
                },
                "adstock_params": {
                    "alpha": np.array(
                        [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]]
                    )  # dims: chain, draw, channel
                },
                "channel_contribution": np.array(
                    [
                        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                    ]
                ),  # dims: chain, draw, channel, date
            },
            None,
            {"channel_1": 58.97600120944057, "channel_2": 41.02399879055943},
            44.94,
        ),
        # custom minimize kwargs
        (
            100,
            optimizer_xarray_builder(
                np.array([[0, 50], [0, 50]]),
                channel=["channel_1", "channel_2"],
                bound=["lower", "upper"],
            ),
            None,
            {
                "saturation_params": {
                    "lam": np.array(
                        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
                    ),  # dims: chain, draw, channel
                    "beta": np.array(
                        [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]]
                    ),  # dims: chain, draw, channel
                },
                "adstock_params": {
                    "alpha": np.array(
                        [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]]
                    )  # dims: chain, draw, channel
                },
                "channel_contribution": np.array(
                    [
                        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                    ]
                ),  # dims: chain, draw, channel, date
            },
            {
                "method": "SLSQP",
                "options": {"ftol": 1e-8, "maxiter": 1_002},
            },
            {"channel_1": 50.0, "channel_2": 50.0},
            44.92,
        ),
        # Zero budget case
        (
            0,
            optimizer_xarray_builder(
                np.array([[0, 50], [0, 50]]),
                channel=["channel_1", "channel_2"],
                bound=["lower", "upper"],
            ),
            None,
            {
                "saturation_params": {
                    "lam": np.array(
                        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
                    ),  # dims: chain, draw, channel
                    "beta": np.array(
                        [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]]
                    ),  # dims: chain, draw, channel
                },
                "adstock_params": {
                    "alpha": np.array(
                        [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]]
                    )  # dims: chain, draw, channel
                },
                "channels": ["channel_1", "channel_2"],
                "channel_contribution": np.array(
                    [
                        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                    ]
                ),  # dims: chain, draw, channel, date
            },
            None,
            {"channel_1": 0.0, "channel_2": 0.0},
            0.0,
        ),
    ],
    ids=[
        "default_minimizer_kwargs",
        "manually_set_x0",
        "custom_minimizer_kwargs",
        "zero_total_budget",
    ],
)
def test_allocate_budget(
    total_budget,
    budget_bounds,
    x0,
    parameters,
    minimize_kwargs,
    expected_optimal,
    expected_response,
    mmm_wrapper,
):
    """Regression test for the post-migration optimization target.

    The old MMM tests optimized ``total_contribution`` (scaled and including
    non-media effects such as intercept). The multidimensional path now
    optimizes ``total_media_contribution_original_scale`` (media-only, original
    units), so the expected allocation/response values intentionally differ.
    """
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )

    # Allocate Budget
    optimal_budgets, optimization_res = optimizer.allocate_budget(
        total_budget=total_budget,
        budget_bounds=budget_bounds,
        x0=x0,
        minimize_kwargs=minimize_kwargs,
    )

    # Assert Results
    assert optimal_budgets.to_dataframe(name="_").to_dict()["_"] == pytest.approx(
        expected_optimal, abs=1e-12
    )
    assert -optimization_res.fun == pytest.approx(expected_response, abs=1e-2, rel=1e-2)


def test_budget_optimizer_clear_error_on_missing_response_variable(mmm_wrapper):
    """An unknown ``response_variable`` must raise a clear error listing the
    posterior variables available on the wrapped model."""
    with pytest.raises(ValueError, match=r"response_variable.*does_not_exist"):
        BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=4,
            response_variable="does_not_exist",
        )


def test_empty_constraints_auto_adds_default(mmm_wrapper):
    """Empty ``constraints`` should auto-add the default sum constraint."""
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=4,
        response_variable="total_media_contribution_original_scale",
    )
    assert "default" in optimizer._constraints


def test_non_empty_constraints_skips_default(mmm_wrapper):
    """A non-empty ``constraints`` means the caller is in charge: no default."""
    custom = [
        Constraint(
            key="cap",
            constraint_fun=lambda budgets_sym, total_budget_sym, optimizer: (
                budgets_sym.sum() - total_budget_sym
            ),
            constraint_type="eq",
        )
    ]
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=4,
        response_variable="total_media_contribution_original_scale",
        constraints=custom,
    )
    assert "default" not in optimizer._constraints
    assert "cap" in optimizer._constraints


def test_constraint_instance_round_trips_into_constraints(mmm_wrapper):
    """A ``Constraint`` passed via ``constraints`` lands in ``_constraints`` by key."""
    cap = Constraint(
        key="cap",
        constraint_fun=lambda budgets_sym, total_budget_sym, optimizer: (
            budgets_sym.sum() - total_budget_sym
        ),
        constraint_type="ineq",
    )
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=4,
        response_variable="total_media_contribution_original_scale",
        constraints=[cap],
    )
    # Stored object is the same instance, not a copy.
    assert optimizer._constraints["cap"] is cap


def test_constraints_empty_list_matches_default(mmm_wrapper):
    """An explicit empty list behaves like the default empty tuple."""
    opt_default = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=4,
        response_variable="total_media_contribution_original_scale",
    )
    opt_empty_list = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=4,
        response_variable="total_media_contribution_original_scale",
        constraints=[],
    )
    assert (
        set(opt_default._constraints) == set(opt_empty_list._constraints) == {"default"}
    )


def test_set_constraints_is_reentrant(mmm_wrapper):
    """Re-calling ``set_constraints`` clears prior state and recompiles."""
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=4,
        response_variable="total_media_contribution_original_scale",
    )
    assert set(optimizer._constraints) == {"default"}

    cap = Constraint(
        key="cap",
        constraint_fun=lambda budgets_sym, total_budget_sym, optimizer: (
            budgets_sym.sum() - total_budget_sym
        ),
        constraint_type="ineq",
    )
    optimizer.set_constraints([cap])

    # Old "default" is gone, only the new constraint remains, recompiled.
    assert set(optimizer._constraints) == {"cap"}
    assert len(optimizer._compiled_constraints) == 1


def test_duplicate_constraint_keys_raise(mmm_wrapper):
    """Two constraints sharing a key must raise, not silently clobber."""
    fun = lambda budgets_sym, total_budget_sym, optimizer: budgets_sym.sum()  # noqa: E731
    dup = [
        Constraint(key="cap", constraint_fun=fun, constraint_type="ineq"),
        Constraint(key="cap", constraint_fun=fun, constraint_type="ineq"),
    ]
    with pytest.raises(ValueError, match="Duplicate constraint key"):
        BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=4,
            response_variable="total_media_contribution_original_scale",
            constraints=dup,
        )


@patch("pymc_marketing.mmm.budget_optimizer.minimize")
def test_allocate_budget_custom_minimize_args(
    minimize_mock,
    mmm_wrapper,
) -> None:
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )

    total_budget = 100
    budget_bounds = {"channel_1": (0.0, 50.0), "channel_2": (0.0, 50.0)}
    minimize_kwargs = {
        "method": "SLSQP",
        "options": {"ftol": 1e-8, "maxiter": 1_002},
    }

    # The mocked minimize returns a Mock result.x, which fails the optimization
    # variables' shape validation when unpacking -- after minimize was called.
    with pytest.raises(ValueError, match=r"expected shape"):
        optimizer.allocate_budget(
            total_budget, budget_bounds, minimize_kwargs=minimize_kwargs
        )

    kwargs = minimize_mock.call_args_list[0].kwargs

    np.testing.assert_array_equal(actual=kwargs["x0"], desired=np.array([50.0, 50.0]))
    assert kwargs["bounds"] == [(0.0, 50.0), (0.0, 50.0)]
    assert kwargs["method"] == minimize_kwargs["method"]
    assert kwargs["options"] == minimize_kwargs["options"]


@pytest.mark.parametrize(
    "total_budget, budget_bounds, parameters, constraints",
    [
        (
            100,
            optimizer_xarray_builder(
                np.array([[0, 50], [0, 50]]),
                channel=["channel_1", "channel_2"],
                bound=["lower", "upper"],
            ),
            {
                "saturation_params": {
                    "lam": np.array(
                        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
                    ),  # dims: chain, draw, channel
                    "beta": np.array(
                        [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]]
                    ),  # dims: chain, draw, channel
                },
                "adstock_params": {
                    "alpha": np.array(
                        [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]]
                    )  # dims: chain, draw, channel
                },
                "channels": ["channel_1", "channel_2"],
            },
            # New-style custom constraint: channel_1 must be >= 60, which is infeasible
            [
                Constraint(
                    key="channel_1_min_constraint",
                    constraint_fun=lambda budgets_sym, total_budget_sym, optimizer: (
                        budgets_sym[0] - 60
                    ),
                    constraint_type="ineq",
                ),
            ],
        ),
    ],
)
def test_allocate_budget_infeasible_constraints(
    total_budget,
    budget_bounds,
    parameters,
    constraints,
    mmm_wrapper,
):
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        response_variable="total_media_contribution_original_scale",
        constraints=constraints,
        num_periods=30,
    )

    with pytest.raises(MinimizeException, match=r"Optimization failed"):
        optimizer.allocate_budget(total_budget, budget_bounds)


def mean_response_eq_constraint_fun(
    budgets_sym, total_budget_sym, optimizer, target_response
):
    """
    Enforces mean_response(budgets_sym) = target_response,
    i.e. returns (mean_resp - target_response).
    """
    resp_dist = optimizer.extract_response_distribution(
        "total_media_contribution_original_scale"
    )
    mean_resp = _check_samples_dimensionality(resp_dist).mean()
    return mean_resp - target_response


def minimize_budget_utility(samples, budgets):
    """
    A trivial "utility" that just tries to minimize total budget.
    Since the BudgetOptimizer by default *maximizes* the utility,
    we use the negative sign to effectively force minimization.
    """
    return -budgets.sum()


@pytest.mark.parametrize(
    "total_budget,target_response",
    [
        (10, 5.0),
        (50, 10.0),
    ],
    ids=["budget=10->resp=5", "budget=50->resp=10"],
)
def test_allocate_budget_custom_response_constraint(
    mmm_wrapper,
    total_budget,
    target_response,
):
    """
    Checks that a custom constraint can enforce the model's mean response
    to equal a target value, while we minimize the total budget usage.
    """

    def constraint_wrapper(budgets_sym, total_budget_sym, optimizer):
        return mean_response_eq_constraint_fun(
            budgets_sym, total_budget_sym, optimizer, target_response
        )

    constraints = [
        Constraint(
            key="target_response_constraint",
            constraint_fun=constraint_wrapper,
            constraint_type="eq",
        )
    ]

    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        response_variable="total_media_contribution_original_scale",
        utility_function=minimize_budget_utility,
        constraints=constraints,
        num_periods=30,
    )

    _allocation, res = optimizer.allocate_budget(
        total_budget=total_budget,
        budget_bounds=None,
    )

    resp_dist_sym = optimizer.extract_response_distribution(
        "total_media_contribution_original_scale"
    )
    resp_mean_sym = _check_samples_dimensionality(resp_dist_sym).mean()
    test_fn = pytensor.function([optimizer._budgets_flat], resp_mean_sym)
    final_resp = test_fn(res.x)

    np.testing.assert_allclose(final_resp, target_response, rtol=1e-2)


@pytest.mark.parametrize(
    "callback, total_budget",
    [
        # Basic cases
        (False, 100),  # Default behavior - no callback
        (True, 100),  # With callback
    ],
    ids=[
        "default_no_callback",
        "basic_with_callback",
    ],
)
def test_callback_functionality_parametrized(
    mmm_wrapper,
    callback,
    total_budget,
):
    """Test callback functionality with various parameter combinations."""
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )

    result = optimizer.allocate_budget(
        total_budget=total_budget,
        callback=callback,
    )

    # The result always unpacks to two elements regardless of callback
    assert isinstance(result, BudgetOptimizationResult)
    assert len(list(result)) == 2

    if callback:
        optimal_budgets, opt_result = result
        callback_info = result.callback_info

        # Verify callback info structure
        assert isinstance(callback_info, list)
        assert len(callback_info) > 0

        # Check first iteration
        first_iter = callback_info[0]
        assert "x" in first_iter
        assert "fun" in first_iter
        assert "jac" in first_iter

        # Check data types
        assert isinstance(first_iter["x"], np.ndarray)
        assert isinstance(first_iter["fun"], float | np.float64 | np.float32)
        assert isinstance(first_iter["jac"], np.ndarray)

        # Check dimensions
        assert first_iter["x"].shape == first_iter["jac"].shape

        # Check constraints (default constraint should be present)
        assert "constraint_info" in first_iter
        assert [c["key"] for c in first_iter["constraint_info"]] == ["default"]
        assert set(result.constraint_history) == {"default"}
        assert len(result.constraint_history["default"]) == len(callback_info)

        # Verify all iterations have same structure
        for iter_info in callback_info:
            assert set(iter_info.keys()) == set(first_iter.keys())

    else:
        # Unpack without callback
        optimal_budgets, opt_result = result
        assert result.callback_info is None
        assert result.constraint_history == {}

    # Common checks
    assert isinstance(optimal_budgets, xr.DataArray)
    assert hasattr(opt_result, "x")
    assert hasattr(opt_result, "success")

    # Check budget allocation sums to total
    assert np.abs(optimal_budgets.sum().item() - total_budget) < 1e-3


def test_constraint_history_keyed_by_constraint(mmm_wrapper):
    """Constraint diagnostics can be read by key instead of by position.

    The custom floor is never active (the equality pins the sum at the total
    budget), so this checks the keying, not the solver: every key is present,
    every key has one entry per iteration, and each entry is the same object
    as its positional counterpart.
    """

    def spend_floor(budgets_sym, total_budget_sym, optimizer):
        return budgets_sym.sum() - 10.0

    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
        constraints=[
            Constraint(
                key="spend_floor",
                constraint_type="ineq",
                constraint_fun=spend_floor,
            ),
            build_default_sum_constraint(),
        ],
    )
    result = optimizer.allocate_budget(total_budget=100.0, callback=True)

    history = result.constraint_history
    assert set(history) == {"spend_floor", "default"}
    for key, entries in history.items():
        assert len(entries) == len(result.callback_info)
        assert all(entry["key"] == key for entry in entries)

    # Regrouped entries are the same objects as the positional ones, found
    # by key rather than by the compile order this view exists to hide.
    last_iter = {
        info["key"]: info for info in result.callback_info[-1]["constraint_info"]
    }
    assert history["spend_floor"][-1] is last_iter["spend_floor"]
    assert history["default"][-1] is last_iter["default"]
    assert history["spend_floor"][-1]["type"] == "ineq"
    assert history["default"][-1]["type"] == "eq"
    assert np.isclose(history["default"][-1]["value"], 0.0, atol=1e-6)


def test_diversification_ratio_through_optimizer(mmm_wrapper):
    """The docstring recipe runs through BudgetOptimizer and actually optimizes.

    The utility sees the response as ``(sample, date, channel)``; reducing
    over ``date`` gives the ``(sample, channel)`` shape the ratio needs.

    ``total_budget=10`` is deliberate: at 100 the fixture's saturation is so
    flat that the gradient at the equal split is already within SLSQP's
    tolerance, and the solver stops at the initial guess, which every
    assertion here would then satisfy for free.
    """
    total_budget = 10.0
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="channel_contribution",
        utility_function=lambda samples, budgets: diversification_ratio(
            samples.sum(dim="date"), budgets
        ),
    )
    result = optimizer.allocate_budget(total_budget=total_budget)

    assert result.scipy_result.success
    assert np.isfinite(result.scipy_result.fun)
    np.testing.assert_allclose(result.budgets.sum(), total_budget, atol=1e-3)

    # The solution is not the initial guess (the equal split), and the
    # utility there is genuinely higher than at the initial guess.
    x0 = xr.full_like(result.budgets, total_budget / result.budgets.size)
    assert not np.allclose(result.budgets.values, x0.values)
    assert result.scipy_result.fun < optimizer.evaluate_plan(x0).objective
    # The objective is the negated utility, and DR is bounded below by 1.
    assert -result.scipy_result.fun >= 1.0 - 1e-6


def test_allocate_budget_result_object(mmm_wrapper):
    """allocate_budget returns a BudgetOptimizationResult with stable attributes."""
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )

    result = optimizer.allocate_budget(total_budget=100)

    assert isinstance(result, BudgetOptimizationResult)
    assert isinstance(result.budgets, xr.DataArray)
    assert hasattr(result.scipy_result, "x")
    assert result.optimized_vars == {}
    assert result.callback_info is None

    # Iteration contract: exactly (budgets, scipy_result)
    unpacked = list(result)
    assert len(unpacked) == 2
    assert unpacked[0] is result.budgets
    assert unpacked[1] is result.scipy_result


class TestEvaluatePlan:
    """Scoring a labelled plan through the public API.

    The compiled objective is the same callable SLSQP is handed; what this
    class pins is the contract around it -- which sign each output carries,
    what the gradient is labelled with, and which inputs are refused.
    """

    CHANNELS = ("channel_1", "channel_2")

    def _optimizer(self, mmm_wrapper, **kwargs):
        kwargs.setdefault(
            "response_variable", "total_media_contribution_original_scale"
        )
        return BudgetOptimizer(model=mmm_wrapper, num_periods=30, **kwargs)

    def _plan(self, *values):
        return xr.DataArray(
            list(values), dims=["channel"], coords={"channel": list(self.CHANNELS)}
        )

    def test_objective_is_the_value_the_solver_minimises(self, mmm_wrapper):
        """`objective` is on the solver's scale, so the two are comparable.

        Pinned against `scipy_result.fun` rather than against a recomputed
        number: if the method ever returned the utility under this name, an
        optimum would compare unequal to itself.
        """
        optimizer = self._optimizer(mmm_wrapper)
        result = optimizer.allocate_budget(total_budget=100)

        evaluation = optimizer.evaluate_plan(result.budgets)

        np.testing.assert_allclose(
            evaluation.objective, result.scipy_result.fun, rtol=1e-9
        )
        np.testing.assert_allclose(
            evaluation.utility, -result.scipy_result.fun, rtol=1e-9
        )

    def test_gradient_is_labelled_and_packs_back(self, mmm_wrapper):
        """The gradient comes back on the decision variables' own labels.

        Repacking it reproduces the flat gradient the solver is handed, which
        is what makes the labelled form usable for a decomposition: nothing was
        reordered or dropped on the way out.
        """
        optimizer = self._optimizer(mmm_wrapper)
        plan = self._plan(30.0, 70.0)

        evaluation = optimizer.evaluate_plan(plan)

        gradient = evaluation.objective_gradient
        assert set(gradient) == {"channel_data"}
        assert gradient["channel_data"].dims == ("channel",)
        _, flat_gradient = optimizer._objective_and_grad(
            optimizer.optimization_variables.pack(plan)
        )
        np.testing.assert_array_equal(
            optimizer.optimization_variables.pack(gradient), flat_gradient
        )
        np.testing.assert_array_equal(
            optimizer.optimization_variables.pack(evaluation.utility_gradient),
            -flat_gradient,
        )

    def test_a_raw_array_is_refused(self, mmm_wrapper):
        """A raw vector is refused because a short one would not be.

        The compiled objective runs with `trust_input=True`: a length-1 vector
        for a two-cell decision space broadcasts and returns the objective of a
        different plan, with no error. Labels make that unrepresentable.
        """
        optimizer = self._optimizer(mmm_wrapper)

        with pytest.raises(TypeError, match="labelled plan"):
            optimizer.evaluate_plan(np.array([30.0, 70.0]))

    def test_a_plan_missing_an_optimized_cell_is_refused(self, mmm_wrapper):
        """A plan that does not price every optimized cell is an error, not a guess."""
        optimizer = self._optimizer(mmm_wrapper)
        partial = xr.DataArray(
            [30.0], dims=["channel"], coords={"channel": ["channel_1"]}
        )

        with pytest.raises(ValueError, match="values missing"):
            optimizer.evaluate_plan(partial)

    def test_constraint_residuals_require_a_total_budget(self, mmm_wrapper):
        """Residuals are measured against the named budget, not ambient state.

        `_total_budget` is a shared variable holding 0.0 until the first
        `allocate_budget`, so a residual read off whatever it happens to hold
        would call a feasible plan infeasible by the whole budget, depending on
        call history the caller cannot see.
        """
        optimizer = self._optimizer(mmm_wrapper)
        plan = self._plan(30.0, 70.0)

        assert optimizer.evaluate_plan(plan).constraint_residuals is None

        residuals = optimizer.evaluate_plan(
            plan, total_budget=100.0
        ).constraint_residuals

        assert residuals["default"]["constraint_type"] == "eq"
        np.testing.assert_allclose(residuals["default"]["value"], 0.0, atol=1e-9)

    def test_measuring_constraints_leaves_the_shared_total_budget_alone(
        self, mmm_wrapper
    ):
        """Scoring a plan must not redefine the budget for later readers."""
        optimizer = self._optimizer(mmm_wrapper)
        optimizer.allocate_budget(total_budget=100)

        optimizer.evaluate_plan(self._plan(10.0, 10.0), total_budget=20.0)

        assert float(optimizer._total_budget.get_value()) == 100.0

    def test_feasible_reads_the_constraint_type(self, mmm_wrapper):
        """An equality is feasible near zero; an inequality is feasible from zero up."""
        optimizer = self._optimizer(mmm_wrapper)
        optimizer.set_constraints(
            [
                build_default_sum_constraint("default"),
                Constraint(
                    key="channel_1_floor",
                    constraint_type="ineq",
                    constraint_fun=lambda budgets, total, opt: (
                        opt.optimization_variables.variable_slice("channel_data").isel(
                            {FLAT_DIM: 0}
                        )
                        - 40.0
                    ),
                ),
            ]
        )

        on_budget_below_floor = self._plan(30.0, 70.0)
        on_budget_above_floor = self._plan(60.0, 40.0)
        over_budget = self._plan(60.0, 60.0)

        assert not optimizer.evaluate_plan(
            on_budget_below_floor, total_budget=100.0
        ).feasible()
        assert optimizer.evaluate_plan(
            on_budget_above_floor, total_budget=100.0
        ).feasible()
        assert not optimizer.evaluate_plan(over_budget, total_budget=100.0).feasible()

    def test_feasible_without_residuals_says_what_is_missing(self, mmm_wrapper):
        """Feasibility is undefined without the budget the constraints use."""
        optimizer = self._optimizer(mmm_wrapper)

        with pytest.raises(ValueError, match="total_budget"):
            optimizer.evaluate_plan(self._plan(30.0, 70.0)).feasible()

    def test_the_shared_budget_is_restored_when_a_constraint_raises(self, mmm_wrapper):
        """The restore is a `finally`, so a failed measurement must not leak either.

        The happy path is covered above; this is the path the `finally` exists
        for. A constraint that raises would otherwise leave the optimizer
        holding a budget nobody asked for, and the next `feasible()` or
        `callback=True` run would measure against it.
        """
        optimizer = self._optimizer(mmm_wrapper)
        optimizer.allocate_budget(total_budget=100)

        def explode(_x):
            raise RuntimeError("constraint blew up")

        optimizer._compiled_constraints[0]["fun"] = explode

        with pytest.raises(RuntimeError, match="constraint blew up"):
            optimizer.evaluate_plan(self._plan(10.0, 10.0), total_budget=20.0)

        assert float(optimizer._total_budget.get_value()) == 100.0


class TestEvaluateResponseDistribution:
    """The posterior response under a plan, not its reduction to a utility."""

    def _plan(self):
        return xr.DataArray(
            [30.0, 70.0],
            dims=["channel"],
            coords={"channel": ["channel_1", "channel_2"]},
        )

    def test_it_is_the_extracted_graph_evaluated_at_the_plan(self, mmm_wrapper):
        """The public evaluation is the graph a caller would compile by hand."""
        optimizer = BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
        )
        plan = self._plan()

        response = optimizer.evaluate_response_distribution(plan)

        graph = optimizer.extract_response_distribution(
            "total_media_contribution_original_scale"
        )
        expected = pytensor.function(
            [optimizer.optimization_variables.flat], graph.values
        )(optimizer.optimization_variables.pack(plan))
        assert isinstance(response, xr.DataArray)
        assert response.dims == ("sample",)
        np.testing.assert_allclose(response.to_numpy(), expected)

    def test_it_follows_set_posterior(self, mmm_wrapper, dummy_idata):
        """A cached response function must not outlive the posterior it was compiled on.

        The first `set_posterior` moves the draws from constants into shared
        variables and recompiles; a function compiled before that still holds
        the construction-time draws, and would keep reporting them silently.
        """
        optimizer = BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
        )
        plan = self._plan()
        before = optimizer.evaluate_response_distribution(plan)

        doubled = dummy_idata.copy()
        doubled["posterior"]["saturation_beta"] = (
            doubled["posterior"]["saturation_beta"] * 2
        )
        optimizer.set_posterior(doubled)

        after = optimizer.evaluate_response_distribution(plan)
        assert not np.allclose(after.to_numpy(), before.to_numpy())

    def test_a_raw_array_is_refused_here_too(self, mmm_wrapper):
        """The two sibling methods agree on their own contract.

        A correctly sized array packs without complaint, so the only thing
        standing between a caller and another plan's posterior -- built in
        coordinate order rather than flat order -- is this guard.
        """
        optimizer = BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
        )

        with pytest.raises(TypeError, match="labelled plan"):
            optimizer.evaluate_response_distribution(np.array([30.0, 70.0]))

    def test_a_non_default_variable_comes_back_labelled(self, mmm_wrapper):
        """A per-channel response is labelled with the model's own coords.

        The compiled function returns a bare array, so without the coords a
        caller selecting `channel="channel_2"` would be reading position 1 and
        hoping.
        """
        optimizer = BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
        )

        response = optimizer.evaluate_response_distribution(
            self._plan(), "channel_contribution"
        )

        assert set(response.dims) == {"sample", "date", "channel"}
        assert list(response.coords["channel"].values) == ["channel_1", "channel_2"]
        # Spending only on channel_1 leaves channel_2's contribution at zero,
        # which is what makes the labels worth checking.
        one_channel = optimizer.evaluate_response_distribution(
            xr.DataArray(
                [100.0, 0.0],
                dims=["channel"],
                coords={"channel": ["channel_1", "channel_2"]},
            ),
            "channel_contribution",
        )
        assert float(one_channel.sel(channel="channel_2").sum()) == 0.0
        assert float(one_channel.sel(channel="channel_1").sum()) > 0.0

    def test_a_response_the_plan_cannot_reach_warns_but_still_evaluates(
        self, mmm_wrapper
    ):
        """A plan-independent quantity is answered, and the caller is told once.

        The decision vector is an unused input of that graph, and compiling
        with PyTensor's default `on_unused_input` would refuse it outright
        rather than return the plan-independent value. Answering it silently
        is the other failure: a media quantity wired to its own copy of the
        spend looks exactly like this, and comparing two plans would report no
        difference with nothing to explain why.
        """
        optimizer = BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
        )

        with pytest.warns(UserWarning, match="does not depend on the decision"):
            response = optimizer.evaluate_response_distribution(
                self._plan(), "target_scale"
            )

        assert np.isfinite(response.to_numpy()).all()

    def test_a_reachable_response_does_not_warn(self, mmm_wrapper):
        """The warning has to stay quiet on the path everyone takes."""
        optimizer = BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            optimizer.evaluate_response_distribution(self._plan())

    def test_compile_kwargs_can_override_the_unused_input_default(self, mmm_wrapper):
        """`on_unused_input` is a default here, not a fixed argument.

        `compile_kwargs` is documented as forwarded to PyTensor's `function()`.
        Passing the same key used to collide -- `TypeError: got multiple values
        for keyword argument` -- which reads as a bug in unrelated code rather
        than as the setting taking effect.
        """
        optimizer = BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
            compile_kwargs={"on_unused_input": "raise"},
        )

        # The caller asked for strictness, so the plan-independent variable is
        # refused -- by PyTensor, on the caller's terms.
        with (
            pytest.warns(UserWarning, match="does not depend on the decision"),
            pytest.raises(UnusedInputError),
        ):
            optimizer.evaluate_response_distribution(self._plan(), "target_scale")

        # ... and the reachable path is unaffected by the override.
        assert np.isfinite(
            optimizer.evaluate_response_distribution(self._plan()).to_numpy()
        ).all()


def test_budget_optimizer_mu_effects_deprecated(mmm_wrapper):
    """Passing mu_effects warns and is ignored."""
    with pytest.warns(DeprecationWarning, match="no longer accepts mu_effects"):
        BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
            mu_effects=[],
        )


@pytest.mark.parametrize(
    "budget_distribution_over_period, num_periods, should_error, error_message",
    [
        # Valid case: uniform distribution
        (
            {
                "channel_1": [0.25, 0.25, 0.25, 0.25],
                "channel_2": [0.25, 0.25, 0.25, 0.25],
            },
            4,
            False,
            None,
        ),
        # Valid case: front-loaded distribution
        (
            {"channel_1": [0.7, 0.2, 0.1, 0.0], "channel_2": [0.4, 0.3, 0.2, 0.1]},
            4,
            False,
            None,
        ),
        # Invalid case: factors don't sum to 1
        (
            {"channel_1": [0.3, 0.3, 0.3, 0.3], "channel_2": [0.25, 0.25, 0.25, 0.25]},
            4,
            True,
            "budget_distribution_over_period must sum to 1 along the .date. dimension",
        ),
        # Invalid case: wrong number of periods
        (
            {"channel_1": [0.5, 0.5], "channel_2": [0.5, 0.5]},
            4,
            True,
            "budget_distribution_over_period .date. dimension must have length 4",
        ),
    ],
    ids=[
        "valid_uniform",
        "valid_front_loaded",
        "invalid_sum",
        "invalid_periods",
    ],
)
def test_budget_distribution_over_period(
    mmm_wrapper,
    budget_distribution_over_period,
    num_periods,
    should_error,
    error_message,
):
    """Test that budget_distribution_over_period correctly distributes budget over time."""
    channels = mmm_wrapper.channel_columns

    if budget_distribution_over_period is not None:
        budget_distribution_over_period_array = np.array(
            [budget_distribution_over_period[ch] for ch in channels]
        )
        budget_distribution_over_period_factors = xr.DataArray(
            budget_distribution_over_period_array,
            coords={
                "channel": channels,
                "date": list(range(len(budget_distribution_over_period["channel_1"]))),
            },
            dims=["channel", "date"],
        )
    else:
        budget_distribution_over_period_factors = None

    if should_error:
        with pytest.raises(ValueError, match=error_message):
            BudgetOptimizer(
                model=mmm_wrapper,
                num_periods=num_periods,
                budget_distribution_over_period=budget_distribution_over_period_factors,
                response_variable="total_media_contribution_original_scale",
            )
    else:
        optimizer = BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=num_periods,
            budget_distribution_over_period=budget_distribution_over_period_factors,
            response_variable="total_media_contribution_original_scale",
        )

        # Check that the time distribution factors were stored correctly
        if budget_distribution_over_period_factors is not None:
            assert optimizer._budget_distribution_over_period_tensor is not None
            # The tensor is now pre-processed and has shape (num_periods, num_optimized_budgets)
            num_optimized = optimizer.budgets_to_optimize.sum().item()
            expected_shape = (num_periods, num_optimized)
            assert (
                optimizer._budget_distribution_over_period_tensor.type.shape
                == expected_shape
            )
        else:
            assert optimizer._budget_distribution_over_period_tensor is None


def test_budget_distribution_over_period_wrong_dims(mmm_wrapper):
    """Test that budget_distribution_over_period with wrong dimensions raises error."""
    budget_distribution_over_period = xr.DataArray(
        [0.25, 0.25, 0.25, 0.25],
        coords={"date": list(range(4))},
        dims=["date"],
    )

    with pytest.raises(
        ValueError, match=r"budget_distribution_over_period must have dims"
    ):
        BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=4,
            budget_distribution_over_period=budget_distribution_over_period,
            response_variable="total_media_contribution_original_scale",
        )


def test_budget_distribution_over_period_applied_correctly(mmm_wrapper):
    """Test that budget distribution factors are correctly applied to budgets."""
    channels = mmm_wrapper.channel_columns

    budget_distribution_over_period_data = {
        "channel_1": [0.7, 0.2, 0.1, 0.0],
        "channel_2": [0.4, 0.3, 0.2, 0.1],
    }
    budget_distribution_over_period_array = np.array(
        [budget_distribution_over_period_data[ch] for ch in channels]
    )
    budget_distribution_over_period_factors = xr.DataArray(
        budget_distribution_over_period_array,
        coords={
            "channel": channels,
            "date": list(range(4)),
        },
        dims=["channel", "date"],
    )

    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=4,
        budget_distribution_over_period=budget_distribution_over_period_factors,
        response_variable="total_media_contribution_original_scale",
    )

    # Verify that the time distribution factors tensor was created correctly
    assert optimizer._budget_distribution_over_period_tensor is not None

    # Verify the values match what we provided (stored tensor is pre-processed and transposed)
    stored_values = optimizer._budget_distribution_over_period_tensor.eval()
    # The stored tensor has shape (num_periods, num_optimized_budgets)
    # and the original has shape (channels, periods), so we need to transpose
    np.testing.assert_array_almost_equal(
        stored_values, budget_distribution_over_period_array.T
    )


def test_budget_distribution_over_period_integration(mmm_wrapper):
    """Integration test: verify budget allocation with time distribution factors."""
    channels = mmm_wrapper.channel_columns

    num_periods = 4
    budget_distribution_over_period_data = {
        "channel_1": [0.7, 0.2, 0.1, 0.0],
        "channel_2": [0.25, 0.25, 0.25, 0.25],
    }
    budget_distribution_over_period_array = np.array(
        [budget_distribution_over_period_data[ch] for ch in channels]
    )
    budget_distribution_over_period_factors = xr.DataArray(
        budget_distribution_over_period_array,
        coords={
            "channel": channels,
            "date": list(range(num_periods)),
        },
        dims=["channel", "date"],
    )

    optimizer_with_factors = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=num_periods,
        budget_distribution_over_period=budget_distribution_over_period_factors,
        response_variable="total_media_contribution_original_scale",
    )

    optimizer_without_factors = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=num_periods,
        budget_distribution_over_period=None,
        response_variable="total_media_contribution_original_scale",
    )

    # Both should allocate budget successfully
    total_budget = 100
    budget_bounds = None

    result_with_factors, _ = optimizer_with_factors.allocate_budget(
        total_budget=total_budget,
        budget_bounds=budget_bounds,
    )

    result_without_factors, _ = optimizer_without_factors.allocate_budget(
        total_budget=total_budget,
        budget_bounds=budget_bounds,
    )

    # Both should sum to the total budget
    assert np.abs(result_with_factors.sum().item() - total_budget) < 1e-6
    assert np.abs(result_without_factors.sum().item() - total_budget) < 1e-6

    # Results should potentially be different due to time distribution
    # (though in practice they might be similar depending on the model)
    assert isinstance(result_with_factors, xr.DataArray)
    assert isinstance(result_without_factors, xr.DataArray)
    assert result_with_factors.dims == ("channel",)
    assert result_without_factors.dims == ("channel",)


def test_custom_protocol_model_budget_optimizer_works(mock_pymc_sample):
    """Validate the optimizer works with the built-in CustomModelWrapper.

    This serves as an example for users wanting to plug in their own PyMC models via
    ``CustomModelWrapper``, which satisfies the OptimizerCompatibleModelWrapper protocol.
    """
    # 1) Build and fit a tiny custom PyMC model
    rng = np.random.default_rng(0)
    num_obs = 12
    channels = ["C1", "C2", "C3"]
    X = rng.uniform(0.0, 1.0, size=(num_obs, len(channels)))
    true_beta = np.array([0.8, 0.4, 0.2])
    y = DataArray((X @ true_beta) + rng.normal(0.0, 0.05, size=num_obs), dims=("date",))

    coords = {"date": np.arange(num_obs), "channel": channels}
    with pm.Model(coords=coords) as train_model:
        pmd.Data("channel_data", X, dims=("date", "channel"))
        beta = pmd.Normal("beta", 0.0, 1.0, dims="channel")
        mu = (train_model["channel_data"] * beta).sum(dim="channel")
        pmd.Deterministic("total_media_contribution_original_scale", mu.sum(), dims=())
        pmd.Deterministic(
            "channel_contribution",
            train_model["channel_data"] * beta,
            dims=("date", "channel"),
        )
        sigma = pmd.HalfNormal("sigma", 0.2)
        pmd.Normal("y", mu=mu, sigma=sigma, observed=y, dims="date")

        idata = pm.sample(50, tune=50, chains=1, progressbar=False, random_seed=1)

    # 2) Wrap the model with CustomModelWrapper
    wrapper = CustomModelWrapper(
        base_model=train_model,
        idata=idata,
        channels=channels,
    )

    # Ensure the wrapper produces correctly shaped optimization models
    opt_model = wrapper._set_predictors_for_optimization(num_periods=6)
    assert tuple(opt_model.named_vars_to_dims["channel_data"]) == ("date", "channel")
    assert list(opt_model.coords["channel"]) == channels
    assert len(opt_model.coords["date"]) == 6

    # 3) Optimize budgets over a small future horizon
    optimizer = BudgetOptimizer(model=wrapper, num_periods=6)

    # Use dict bounds (single budget dimension)
    bounds = {c: (0.0, 50.0) for c in channels}

    optimal_budgets, result = optimizer.allocate_budget(
        total_budget=100.0, budget_bounds=bounds
    )

    # Assertions: types, dims, success, sum constraint
    assert isinstance(optimal_budgets, xr.DataArray)
    assert optimal_budgets.dims == ("channel",)
    assert list(optimal_budgets.coords["channel"].values) == channels
    assert result.success
    assert np.isclose(optimal_budgets.sum().item(), 100.0)


def test_shuffled_mask_labels_match_model_coords(mmm_wrapper):
    """A mask in a different coord order than the model must not shift labels.

    The mask is consumed positionally by the forward map (scatter into the
    model's tensor layout) and also supplies the labels for the inverse map,
    so it is reindexed to the model's coordinate order at construction. This
    pins the inverse map to the forward map: with per-channel bounds that make
    the optimum distinguishable, the value attributed to a channel must be the
    one its own bounds produced.
    """
    channels = list(mmm_wrapper.channel_columns)  # model order: channel_1, channel_2
    shuffled = list(reversed(channels))

    mask = xr.DataArray(
        np.ones(len(shuffled), dtype=bool),
        dims=("channel",),
        coords={"channel": shuffled},
    )
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        budgets_to_optimize=mask,
        response_variable="total_media_contribution_original_scale",
    )
    # The mask is realigned to the model's coordinate order.
    assert list(optimizer.budgets_to_optimize.coords["channel"].values) == channels

    # channel_1 is capped at 5, channel_2 must take the remaining 95.
    bounds = optimizer_xarray_builder(
        np.array([[0.0, 5.0], [0.0, 95.0]]),
        channel=channels,
        bound=["lower", "upper"],
    )
    result = optimizer.allocate_budget(total_budget=100.0, budget_bounds=bounds)

    assert float(result.budgets.sel(channel="channel_1")) <= 5.0 + 1e-6
    np.testing.assert_allclose(
        float(result.budgets.sel(channel="channel_2")), 95.0, atol=1e-4
    )


def test_partial_mask_result_is_invariant_to_coord_order(mmm_wrapper):
    """A partial mask must select the same cells however its coords are ordered.

    With a partial mask the label shift and the positional selection shift can
    cancel in the labelled output while the model optimizes the *other*
    channel's curve -- the reported allocation looks right but the objective
    behind it is wrong. Optimizing the same intent written in two coord orders
    must agree on both the allocation and the objective value.
    """
    channels = list(mmm_wrapper.channel_columns)  # [channel_1, channel_2]

    def optimize(coord_order):
        # Intent in every ordering: optimize channel_2 only.
        mask = xr.DataArray(
            np.array([c == "channel_2" for c in coord_order]),
            dims=("channel",),
            coords={"channel": coord_order},
        )
        optimizer = BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            budgets_to_optimize=mask,
            response_variable="total_media_contribution_original_scale",
        )
        bounds = optimizer_xarray_builder(
            np.array([[0.0, 100.0], [0.0, 100.0]]),
            channel=channels,
            bound=["lower", "upper"],
        )
        return optimizer.allocate_budget(total_budget=100.0, budget_bounds=bounds)

    in_model_order = optimize(channels)
    in_shuffled_order = optimize(list(reversed(channels)))

    xr.testing.assert_allclose(in_model_order.budgets, in_shuffled_order.budgets)
    np.testing.assert_allclose(
        in_model_order.scipy_result.fun, in_shuffled_order.scipy_result.fun, rtol=1e-8
    )
    # And the intent was honoured: the frozen channel got nothing.
    np.testing.assert_allclose(
        float(in_shuffled_order.budgets.sel(channel="channel_1")), 0.0, atol=1e-8
    )


def test_mask_missing_model_coords_raises(mmm_wrapper):
    """A mask that does not cover the model's coordinates is rejected.

    Reindexing such a mask would leave NaN, which `astype(bool)` would quietly
    turn into True -- optimizing a cell the user never named.
    """
    mask = xr.DataArray(
        np.array([True]),
        dims=("channel",),
        coords={"channel": ["channel_1"]},  # model also has channel_2
    )
    with pytest.raises(ValidationError, match="does not cover every model coordinate"):
        BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            budgets_to_optimize=mask,
            response_variable="total_media_contribution_original_scale",
        )


def test_integer_mask_is_coerced_to_bool(mmm_wrapper):
    """A 0/1 mask works: reindexing makes it float, so it is cast back."""
    mask = xr.DataArray(
        np.array([1, 0]),
        dims=("channel",),
        coords={"channel": list(mmm_wrapper.channel_columns)},
    )
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        budgets_to_optimize=mask,
        response_variable="total_media_contribution_original_scale",
    )
    assert optimizer.budgets_to_optimize.dtype == bool
    assert optimizer._variables.size == 1  # only channel_1 optimized


def test_allocate_budget_x0_dataarray(mmm_wrapper):
    """A labelled x0 warm start gives the same result as the flat vector."""
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )

    x0_flat = np.array([70.0, 30.0])
    x0_labelled = xr.DataArray(
        x0_flat,
        dims=("channel",),
        coords={"channel": ["channel_1", "channel_2"]},
    )

    result_flat = optimizer.allocate_budget(total_budget=100, x0=x0_flat)
    result_labelled = optimizer.allocate_budget(total_budget=100, x0=x0_labelled)
    result_dict = optimizer.allocate_budget(
        total_budget=100, x0={"channel_data": x0_labelled}
    )

    xr.testing.assert_allclose(result_flat.budgets, result_labelled.budgets)
    xr.testing.assert_allclose(result_flat.budgets, result_dict.budgets)


def test_shuffled_distribution_matches_model_order(mmm_wrapper):
    """A shuffled mask and distribution pair must not swap temporal profiles.

    The mask is realigned to the model's coordinate order, so the distribution
    has to be too: they are combined positionally, and aligning only one hands
    each channel another channel's spending profile.
    """
    channels = list(mmm_wrapper.channel_columns)  # [channel_1, channel_2]
    profile = {"channel_1": [0.8, 0.2], "channel_2": [0.2, 0.8]}

    def build(order):
        return BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=2,
            response_variable="total_media_contribution_original_scale",
            budgets_to_optimize=xr.DataArray(
                np.ones(2, dtype=bool), dims=("channel",), coords={"channel": order}
            ),
            budget_distribution_over_period=xr.DataArray(
                np.array([[profile[c][t] for c in order] for t in range(2)]),
                dims=("date", "channel"),
                coords={"date": [0, 1], "channel": order},
            ),
        )

    in_model_order = build(channels)._budget_distribution_over_period_tensor
    in_shuffled_order = build(
        list(reversed(channels))
    )._budget_distribution_over_period_tensor
    np.testing.assert_allclose(
        in_model_order.values.eval(), in_shuffled_order.values.eval()
    )


def test_mask_with_unknown_coords_raises(mmm_wrapper):
    """A mask naming a channel the model does not have is rejected.

    Reindexing drops unknown labels silently, so the cell would vanish and its
    budget be redistributed while the user believed it was considered.
    """
    mask = xr.DataArray(
        np.ones(3, dtype=bool),
        dims=("channel",),
        coords={"channel": [*mmm_wrapper.channel_columns, "channel_typo"]},
    )
    with pytest.raises(ValidationError, match="coordinates the model does not have"):
        BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            budgets_to_optimize=mask,
            response_variable="total_media_contribution_original_scale",
        )


def test_cost_per_unit_missing_coord_raises(mmm_wrapper):
    """cost_per_unit missing a model coordinate is caught, not turned into NaN."""
    cost = xr.DataArray(
        np.ones((30, 1)),
        dims=("date", "channel"),
        coords={"date": range(30), "channel": ["channel_1"]},  # channel_2 missing
    )
    with pytest.raises(ValidationError, match="does not cover every model coordinate"):
        BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            cost_per_unit=cost,
            response_variable="total_media_contribution_original_scale",
        )


def test_unknown_kwarg_raises(mmm_wrapper):
    """An unknown field name raises instead of being silently dropped.

    Regression for the ``custom_constraints`` -> ``constraints`` rename: a
    stale keyword used to be ignored, leaving the optimizer with only the
    default sum constraint and no error.
    """
    with pytest.raises(ValidationError, match="custom_constraints"):
        BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
            custom_constraints=[],
        )


def test_budget_bounds_missing_coord_raises(mmm_wrapper):
    """A bounds DataArray missing a model coordinate raises instead of NaN bounds."""
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )
    bounds = optimizer_xarray_builder(
        np.array([[0.0, 50.0]]), channel=["channel_1"], bound=["lower", "upper"]
    )
    with pytest.raises(ValueError, match="does not cover every model coordinate"):
        optimizer.allocate_budget(total_budget=100.0, budget_bounds=bounds)


def test_default_bounds_come_from_the_media_variable(mmm_wrapper):
    """With no user bounds, the variable's own defaults are used."""
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )
    with pytest.warns(UserWarning, match="No budget bounds provided"):
        result = optimizer.allocate_budget(total_budget=100.0)
    assert result.scipy_result.success
    np.testing.assert_allclose(float(result.budgets.sum()), 100.0, rtol=1e-6)


class _LeverEffect(MuEffect):
    """Test-only effect registering an optimizable pm.Data node.

    Deliberately a plain MuEffect: this PR wires levers by variable *name*, so
    the optimizer needs no knowledge of effect classes.
    """

    prefix: str = "promo"

    def create_data(self, mmm) -> None:
        model = mmm.model
        model.add_coord(self.prefix, ["evt1", "evt2"])
        pmd.Data(f"{self.prefix}_data", np.full(2, 0.10), dims=self.prefix)

    def create_effect(self, mmm):
        model = mmm.model
        data = model[f"{self.prefix}_data"]
        coef = pmd.HalfNormal(f"{self.prefix}_coef", sigma=1.0, dims=self.prefix)
        contribution = pmd.Deterministic(
            f"{self.prefix}_effect_contribution", data * coef, dims=self.prefix
        )
        # An objective that sees both blocks. The stock media objective is
        # media only, so a lever declared against it is (correctly) rejected by
        # the reachability guard.
        pmd.Deterministic(
            "joint_objective",
            model["channel_contribution"].sum() + contribution.sum(),
        )
        return contribution.sum(dim=self.prefix)

    def set_data(self, mmm, model, X) -> None:
        pass


def _fit_mmm_with_lever(mock_pymc_sample):
    date_range = pd.date_range("2023-01-01", periods=14, freq="W")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "date": date_range,
            "ch1": rng.uniform(100, 500, size=len(date_range)),
            "ch2": rng.uniform(100, 500, size=len(date_range)),
        }
    )
    y = pd.Series(rng.uniform(500, 1500, size=len(date_range)), name="target")
    mmm = MMM(
        date_column="date",
        channel_columns=["ch1", "ch2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    ).add_mu_effect(_LeverEffect())
    mmm.fit(X, y, random_seed=0)
    return mmm, date_range


def test_optimizable_vars_co_optimized_with_media(mock_pymc_sample):
    """A named pm.Data node is optimized alongside the budgets, in one solve."""
    mmm, date_range = _fit_mmm_with_lever(mock_pymc_sample)
    optimizer = mmm.budget_optimizer(
        start_date=date_range[-1] + pd.Timedelta(weeks=1),
        end_date=date_range[-1] + pd.Timedelta(weeks=4),
        optimizable_vars={"promo_data": [(0.0, 1.0), (0.0, 1.0)]},
        response_variable="joint_objective",
    )
    # One joint flat vector: media entries first, then the lever.
    assert optimizer._variables.slices["promo_data"] == slice(
        optimizer.budgets_to_optimize.sum().item(),
        optimizer.budgets_to_optimize.sum().item() + 2,
    )
    # The lever really is wired into the graph the objective is built from.
    assert optimizer._budgets_flat in ancestors([optimizer._pymc_model["promo_data"]])

    result = optimizer.allocate_budget(total_budget=100.0)
    assert result.scipy_result.success
    # Media still sums to the budget: the lever does not draw from the pot.
    np.testing.assert_allclose(float(result.budgets.sum()), 100.0, rtol=1e-6)
    # The lever's optimum comes back labelled.
    promo = result.optimized_vars["promo_data"]
    assert list(promo.coords["promo"].values) == ["evt1", "evt2"]
    assert ((promo.values >= 0.0) & (promo.values <= 1.0)).all()


def test_optimizable_vars_warm_start_at_current_value(mock_pymc_sample):
    """With maxiter=0 the solver returns x0, exposing the seeding convention."""
    mmm, date_range = _fit_mmm_with_lever(mock_pymc_sample)
    optimizer = mmm.budget_optimizer(
        start_date=date_range[-1] + pd.Timedelta(weeks=1),
        end_date=date_range[-1] + pd.Timedelta(weeks=4),
        optimizable_vars={"promo_data": [(0.0, 1.0), (0.0, 1.0)]},
        response_variable="joint_objective",
    )
    result = optimizer.allocate_budget(
        total_budget=100.0,
        minimize_kwargs={"options": {"maxiter": 0}},
        return_if_fail=True,
    )
    # Media spreads the budget uniformly; the lever starts at its model value.
    np.testing.assert_allclose(result.scipy_result.x[:2], [50.0, 50.0])
    np.testing.assert_allclose(result.scipy_result.x[2:], 0.10)


def test_optimizable_vars_unreachable_response_raises(mock_pymc_sample):
    """A lever the response variable cannot reach is rejected at construction."""
    mmm, date_range = _fit_mmm_with_lever(mock_pymc_sample)
    with pytest.raises(ValidationError, match="does not depend on optimizable_vars"):
        mmm.budget_optimizer(
            start_date=date_range[-1] + pd.Timedelta(weeks=1),
            end_date=date_range[-1] + pd.Timedelta(weeks=4),
            optimizable_vars={"promo_data": None},
            # channel_contribution is media only, so it cannot reach the lever.
            response_variable="channel_contribution",
        )


@pytest.mark.parametrize(
    "entry, match",
    [
        ({"not_a_variable": None}, "not a variable with named dims"),
        ({"promo_data": [(0.0, 1.0)]}, "bounds pairs"),
    ],
    ids=["unknown_name", "bounds_length_mismatch"],
)
def test_optimizable_vars_validation_raises(mock_pymc_sample, entry, match):
    mmm, date_range = _fit_mmm_with_lever(mock_pymc_sample)
    with pytest.raises(ValidationError, match=match):
        mmm.budget_optimizer(
            start_date=date_range[-1] + pd.Timedelta(weeks=1),
            end_date=date_range[-1] + pd.Timedelta(weeks=4),
            optimizable_vars=entry,
            response_variable="joint_objective",
        )


def test_optimized_vars_empty_without_optimizable_vars(mmm_wrapper):
    """Backward compatible: a plain optimization returns no extra variables."""
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )
    with pytest.warns(UserWarning, match="No budget bounds provided"):
        result = optimizer.allocate_budget(total_budget=100.0)
    assert result.optimized_vars == {}


class TestDateAxisMustMatchTheBlocks:
    """``carry_in + num_periods + adstock_periods`` has to equal the model's date axis.

    The substituted channel tensor is exactly that long, so a disagreement
    with the model's other date-indexed data only surfaces as a pytensor shape
    error inside scipy, several frames deep, and with no mention of what was
    miscounted. A model from ``create_optimization_model`` carries a leading
    block of history, which is the easiest way to get this wrong.
    """

    N_PERIODS = 8

    def test_forgetting_the_leading_block_is_caught_at_construction(
        self, funnel_identity_fitted_mmm
    ):
        mmm = funnel_identity_fitted_mmm
        lags = mmm.effective_carryover_lags()
        # The window opens right after training, so the leading block is there.
        t0 = pd.Timestamp(mmm.xarray_dataset.coords["date"].values[-1])
        model = mmm.create_optimization_model(
            start_date=t0 + pd.Timedelta(weeks=1),
            end_date=t0 + pd.Timedelta(weeks=self.N_PERIODS),
        )
        assert len(model.coords["date"]) == lags + self.N_PERIODS + lags
        n_decisions = self.N_PERIODS

        with pytest.raises(ValueError, match="Date length mismatch") as info:
            BudgetOptimizer(
                model=model,
                idata=mmm.idata,
                num_periods=n_decisions,
                adstock_periods=lags,
                response_variable="total_response_original_scale",
            )

        message = str(info.value)
        for block in ("carry_in_periods (0)", f"num_periods ({n_decisions})"):
            assert block in message
        assert f"adstock_periods ({lags})" in message
        assert "create_optimization_model" in message

    def test_a_model_with_no_coordinate_values_is_measured_from_the_tensor(self):
        """Dims declared without coords leave ``model.coords[date]`` as ``None``.

        The length that matters is the channel tensor's, which is what ``do``
        replaces, so that is what is measured; and a consistent model passes.
        """
        n_dates, channels = 6, ["a", "b"]
        with pm.Model(coords={"channel": channels}) as model:
            # A length without values: the dims API accepts it, and
            # ``model.coords["date"]`` is then ``None``.
            model.add_coord("date", length=n_dates)
            channel_data = pmd.Data(
                "channel_data", np.ones((n_dates, 2)), dims=("date", "channel")
            )
            beta = pmd.Normal("beta", 1.0, 0.1, dims="channel")
            pmd.Deterministic(
                "channel_contribution", channel_data * beta, dims=("date", "channel")
            )
            pmd.Deterministic(
                "total_media_contribution_original_scale",
                (channel_data * beta).sum(),
                dims=(),
            )
        assert model.coords["date"] is None
        prior = pm.sample_prior_predictive(draws=4, model=model, random_seed=1)
        idata = xr.DataTree.from_dict({"posterior": prior.prior})

        optimizer = BudgetOptimizer(
            model=model, idata=idata, num_periods=4, adstock_periods=2
        )
        assert optimizer.num_periods == 4

        with pytest.raises(ValueError, match="Date length mismatch"):
            BudgetOptimizer(model=model, idata=idata, num_periods=5, adstock_periods=2)


def test_budget_optimizer_has_no_marketing_imports():
    """The optimizer stays a graph-level tool: levers are wired by name only."""
    banned = ("pymc_marketing.mmm.additive_effect", "pymc_marketing.mmm.mmm")
    tree = ast.parse(inspect.getsource(budget_optimizer_module))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [m for m in imported if any(m.startswith(b) for b in banned)]
    assert not offenders, (
        f"budget_optimizer must not import marketing modules: {offenders}"
    )


def test_custom_constraint_can_bind_a_lever(mock_pymc_sample):
    """A lever is constrainable through the public variables accessor.

    Levers stay out of the default budget-sum constraint, so the only way to
    bound one jointly is a custom constraint reaching its segment of the flat
    vector. That has to be possible without private attributes.
    """
    mmm, date_range = _fit_mmm_with_lever(mock_pymc_sample)
    cap = 0.4

    total_lever_cap = Constraint(
        key="max_total_lever",
        constraint_type="ineq",
        constraint_fun=lambda budgets_sym, total_budget_sym, optimizer: (
            cap - optimizer.optimization_variables.variable_slice("promo_data").sum()
        ),
    )

    optimizer = mmm.budget_optimizer(
        start_date=date_range[-1] + pd.Timedelta(weeks=1),
        end_date=date_range[-1] + pd.Timedelta(weeks=4),
        optimizable_vars={"promo_data": [(0.0, 1.0), (0.0, 1.0)]},
        response_variable="joint_objective",
        constraints=[total_lever_cap, build_default_sum_constraint()],
    )
    result = optimizer.allocate_budget(total_budget=100.0)

    assert result.scipy_result.success
    # The lever cap binds: unconstrained, both entries would climb to 1.0.
    assert float(result.optimized_vars["promo_data"].sum()) <= cap + 1e-6
    # And the budget constraint is still honoured alongside it.
    np.testing.assert_allclose(float(result.budgets.sum()), 100.0, rtol=1e-6)


def test_spend_var_allocations_excludes_levers():
    """Only money is reported as money.

    ``optimized_vars`` carries both kinds, and a lever's units are its own: a
    discount depth added to a budget means nothing. Asserted on a result built
    by hand, because the discriminating case needs a lever *and* a spend
    variable present at once -- with only one kind present, returning
    everything would look correct.
    """
    result = BudgetOptimizationResult(
        budgets=xr.DataArray([1.0, 2.0], dims=("channel",)),
        scipy_result=OptimizeResult(success=True),
        optimized_vars={
            "lf_budget": xr.DataArray(7.0),
            "discount_depth": xr.DataArray(0.2),
        },
        spend_var_names=["lf_budget"],
    )

    assert set(result.spend_var_allocations) == {"lf_budget"}
    assert float(result.spend_var_allocations["lf_budget"]) == 7.0


def _scaled(posterior: xr.Dataset, beta_scale) -> xr.DataTree:
    scaled = posterior.assign(
        saturation_beta=lambda ds: ds["saturation_beta"] * beta_scale
    )
    return xr.DataTree.from_dict({"/posterior": scaled})


def test_set_posterior_rebinds_without_recompile(mmm_wrapper, dummy_idata):
    """``set_posterior`` swaps the draws under the compiled objective.

    The first call moves the draws into shared variables, which recompiles
    once; every later call keeps the compiled objective object.  After each
    swap the optimizer must agree with a fresh optimizer built on that
    posterior.  The second posterior also has a different number of draws.

    The budget is large enough that the optimum is interior: at a vertex the
    allocation only encodes which channel wins, so a wrong rebind would go
    unnoticed by the allocation assertion.
    """
    total_budget = 60.0
    posterior = dummy_idata["posterior"].to_dataset()
    first = _scaled(posterior.isel(draw=[0]), [3.0, 0.5])
    second = _scaled(posterior, [0.5, 3.0])

    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )
    assert optimizer._shared_posterior is None  # constants until asked
    constant_objective = optimizer._objective_and_grad
    baseline, _ = optimizer.allocate_budget(total_budget=total_budget)

    def fresh(idata):
        return BudgetOptimizer(
            model=CustomModelWrapper(
                base_model=mmm_wrapper.base_model,
                idata=idata,
                channels=mmm_wrapper.channel_columns,
            ),
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
        ).allocate_budget(total_budget=total_budget)

    optimizer.set_posterior(first)
    assert optimizer._objective_and_grad is not constant_objective  # one recompile
    shared_objective = optimizer._objective_and_grad
    assert optimizer.idata is first
    rebound, rebound_res = optimizer.allocate_budget(total_budget=total_budget)
    expected, expected_res = fresh(first)
    np.testing.assert_allclose(rebound.values, expected.values, rtol=1e-6)
    assert rebound_res.fun == pytest.approx(expected_res.fun, rel=1e-6)
    assert not np.allclose(rebound.values, baseline.values)

    optimizer.set_posterior(second)
    assert optimizer._objective_and_grad is shared_objective  # no recompile
    rebound, rebound_res = optimizer.allocate_budget(total_budget=total_budget)
    expected, expected_res = fresh(second)
    np.testing.assert_allclose(rebound.values, expected.values, rtol=1e-6)
    assert rebound_res.fun == pytest.approx(expected_res.fun, rel=1e-6)


def test_set_posterior_custom_constraint_follows_rebind(mmm_wrapper, dummy_idata):
    """A constraint built from ``extract_response_distribution`` reads the new draws."""

    def mean_response_floor(budgets_sym, total_budget_sym, optimizer):
        response = optimizer.extract_response_distribution(
            "total_media_contribution_original_scale"
        )
        return response.mean() - 1.0

    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
        constraints=[
            Constraint(
                key="floor", constraint_type="ineq", constraint_fun=mean_response_floor
            ),
            build_default_sum_constraint(),
        ],
    )
    x = np.array([1.0, 1.0])

    def compiled_floor():
        return next(c for c in optimizer._compiled_constraints if c["key"] == "floor")

    floor = compiled_floor()
    before = float(floor["fun"](x))

    posterior = dummy_idata["posterior"].to_dataset()
    optimizer.set_posterior(_scaled(posterior, 4.0))
    floor = compiled_floor()
    after_first = float(floor["fun"](x))
    assert after_first != pytest.approx(before)

    optimizer.set_posterior(_scaled(posterior, 8.0))
    assert compiled_floor() is floor
    assert float(floor["fun"](x)) != pytest.approx(after_first)


def test_set_posterior_accepts_bare_posterior_dataset(mmm_wrapper, dummy_idata):
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )
    optimizer.set_posterior(dummy_idata["posterior"].to_dataset().isel(draw=[1]))
    assert isinstance(optimizer.idata, xr.DataTree)
    assert optimizer.idata["posterior"].sizes["draw"] == 1


def test_set_posterior_keeps_the_auto_detected_mask(mmm_wrapper, dummy_idata):
    """The mask is fixed at construction; a posterior implying another is solved on it.

    The mask defines the decision vector the graphs were compiled for, so a
    rebind never re-derives it.  Changing the mask means building a new
    optimizer, which is what a caller who wants the other decision problem
    does anyway.
    """
    posterior = dummy_idata["posterior"].to_dataset()

    def with_contributions(per_channel):
        contribution = posterior["channel_contribution"] * xr.DataArray(
            per_channel, dims="channel", coords={"channel": posterior["channel"]}
        )
        return xr.DataTree.from_dict(
            {"/posterior": posterior.assign(channel_contribution=contribution)}
        )

    def optimizer_on(idata):
        return BudgetOptimizer(
            model=CustomModelWrapper(
                base_model=mmm_wrapper.base_model,
                idata=idata,
                channels=mmm_wrapper.channel_columns,
            ),
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
        )

    optimizer = optimizer_on(with_contributions([1.0, 0.0]))
    assert optimizer.budgets_to_optimize.values.tolist() == [True, False]

    swapped = with_contributions([0.0, 1.0])
    optimizer.set_posterior(swapped)
    assert optimizer.idata is swapped
    assert optimizer.budgets_to_optimize.values.tolist() == [True, False]  # pinned
    allocation, _ = optimizer.allocate_budget(total_budget=60.0)
    assert allocation.values.tolist() == [60.0, 0.0]  # the old decision vector

    # A fresh optimizer on the same posterior picks the other cell.
    assert optimizer_on(swapped).budgets_to_optimize.values.tolist() == [False, True]


def test_set_posterior_requires_every_bound_variable(mmm_wrapper, dummy_idata):
    """A posterior missing a bound variable is refused and nothing is committed.

    This is the first call, which recompiles: a failure part-way through must
    not leave a half-populated shared posterior behind, or every later call
    would take the rebind-only path and swap variables the compiled objective
    never reads.
    """
    total_budget = 60.0
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )
    construction_idata = optimizer.idata
    constant_objective = optimizer._objective_and_grad
    compiled_constraints = optimizer._compiled_constraints
    baseline, baseline_res = optimizer.allocate_budget(total_budget=total_budget)

    posterior = dummy_idata["posterior"].to_dataset()
    without_beta = xr.DataTree.from_dict(
        {"/posterior": posterior.drop_vars("saturation_beta")}
    )
    with pytest.raises(KeyError, match="saturation_beta"):
        optimizer.set_posterior(without_beta)

    # Nothing changed: no holder, same idata, same compiled graphs.
    assert optimizer._shared_posterior is None
    assert optimizer.idata is construction_idata
    assert optimizer._objective_and_grad is constant_objective
    assert optimizer._compiled_constraints is compiled_constraints
    after, after_res = optimizer.allocate_budget(total_budget=total_budget)
    np.testing.assert_allclose(after.values, baseline.values)
    assert after_res.fun == pytest.approx(baseline_res.fun)

    # And the next call takes the first-call path again, so the compiled
    # objective really does follow the new draws.
    good = _scaled(posterior, [3.0, 0.5])
    optimizer.set_posterior(good)
    assert optimizer._shared_posterior is not None
    assert optimizer.idata is good
    rebound, rebound_res = optimizer.allocate_budget(total_budget=total_budget)
    expected, expected_res = BudgetOptimizer(
        model=CustomModelWrapper(
            base_model=mmm_wrapper.base_model,
            idata=good,
            channels=mmm_wrapper.channel_columns,
        ),
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    ).allocate_budget(total_budget=total_budget)
    np.testing.assert_allclose(rebound.values, expected.values, rtol=1e-6)
    assert rebound_res.fun == pytest.approx(expected_res.fun, rel=1e-6)
    assert not np.allclose(rebound.values, baseline.values)


def test_set_posterior_accepts_a_posterior_without_channel_contribution(
    mmm_wrapper, dummy_idata
):
    """A posterior thinned to the free RVs the graphs read is accepted.

    Without ``channel_contribution`` the new posterior implies nothing about
    the auto-detected mask, so there is nothing to compare, and the
    posterior-derived mask in use is kept.
    """
    posterior = dummy_idata["posterior"].to_dataset()
    narrow = posterior.assign(
        channel_contribution=posterior["channel_contribution"]
        * xr.DataArray(
            [1.0, 0.0], dims="channel", coords={"channel": posterior["channel"]}
        )
    )
    optimizer = BudgetOptimizer(
        model=CustomModelWrapper(
            base_model=mmm_wrapper.base_model,
            idata=xr.DataTree.from_dict({"/posterior": narrow}),
            channels=mmm_wrapper.channel_columns,
        ),
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )
    assert optimizer._mask_auto_detected
    assert optimizer.budgets_to_optimize.values.tolist() == [True, False]

    thinned = _scaled(narrow.drop_vars("channel_contribution"), 4.0)
    optimizer.set_posterior(thinned)
    assert optimizer.idata is thinned
    assert optimizer.budgets_to_optimize.values.tolist() == [True, False]  # kept


def test_set_posterior_fallback_mask_is_not_compared(mmm_wrapper, dummy_idata):
    """A mask that fell back to every cell at construction stays every cell.

    When the construction posterior had no ``channel_contribution``, the mask
    is the all-ones fallback.  A later posterior that does carry the variable
    is solved on that mask like any other rebind.
    """
    posterior = dummy_idata["posterior"].to_dataset()
    without = posterior.drop_vars("channel_contribution")
    narrower = posterior.assign(
        channel_contribution=posterior["channel_contribution"]
        * xr.DataArray(
            [1.0, 0.0], dims="channel", coords={"channel": posterior["channel"]}
        )
    )
    optimizer = BudgetOptimizer(
        model=CustomModelWrapper(
            base_model=mmm_wrapper.base_model,
            idata=xr.DataTree.from_dict({"/posterior": without}),
            channels=mmm_wrapper.channel_columns,
        ),
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )
    assert optimizer._mask_auto_detected
    assert optimizer.budgets_to_optimize.values.tolist() == [True, True]

    optimizer.set_posterior(xr.DataTree.from_dict({"/posterior": narrower}))
    assert optimizer.budgets_to_optimize.values.tolist() == [True, True]  # pinned


def test_set_posterior_first_call_is_validated_like_later_calls(
    mmm_wrapper, dummy_idata
):
    """The first call goes through the same alignment and checks as every later one.

    The compile runs on the construction posterior and the new draws arrive
    through the validated rebind path, so a reordered ``channel`` axis is
    realigned on the first call too, and a resized axis is refused with a
    full rollback instead of being bound and failing later inside pytensor.
    """
    total_budget = 60.0
    posterior = dummy_idata["posterior"].to_dataset()
    ordered = _scaled(posterior, [3.0, 0.5])
    reordered = xr.DataTree.from_dict(
        {"/posterior": ordered["posterior"].to_dataset().isel(channel=[1, 0])}
    )
    assert reordered["posterior"]["channel"].values.tolist() == [
        "channel_2",
        "channel_1",
    ]

    def optimizer(**kwargs):
        return BudgetOptimizer(
            model=mmm_wrapper,
            num_periods=30,
            response_variable="total_media_contribution_original_scale",
            **kwargs,
        )

    reference, _ = optimizer().allocate_budget(total_budget=total_budget)

    first = optimizer()
    first.set_posterior(ordered)
    expected, expected_res = first.allocate_budget(total_budget=total_budget)
    assert not np.allclose(expected.values, reference.values)

    on_first_call = optimizer()
    on_first_call.set_posterior(reordered)
    got, got_res = on_first_call.allocate_budget(total_budget=total_budget)
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-6)
    assert got_res.fun == pytest.approx(expected_res.fun, rel=1e-6)

    on_second_call = optimizer()
    on_second_call.set_posterior(ordered)
    on_second_call.set_posterior(reordered)
    got, got_res = on_second_call.allocate_budget(total_budget=total_budget)
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-6)
    assert got_res.fun == pytest.approx(expected_res.fun, rel=1e-6)

    # A user-supplied mask bypasses the mask gate, so this is the shape check
    # in SharedPosterior doing the refusing -- and the rollback holding.
    mask = xr.DataArray(
        [True, True], dims="channel", coords={"channel": posterior["channel"]}
    )
    with_mask = optimizer(budgets_to_optimize=mask)
    construction_idata = with_mask.idata
    constant_objective = with_mask._objective_and_grad
    one_channel = xr.DataTree.from_dict({"/posterior": posterior.isel(channel=[0])})
    with pytest.raises(ValueError, match=r"channel labels"):
        with_mask.set_posterior(one_channel)
    assert with_mask._shared_posterior is None
    assert with_mask.idata is construction_idata
    assert with_mask._objective_and_grad is constant_objective
    after, _ = with_mask.allocate_budget(total_budget=total_budget)
    np.testing.assert_allclose(after.values, reference.values)
