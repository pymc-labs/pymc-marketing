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
from unittest.mock import patch

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytest
import xarray as xr
from pydantic import ValidationError
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
from pymc_marketing.mmm.utility import _check_samples_dimensionality


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

        # Verify all iterations have same structure
        for iter_info in callback_info:
            assert set(iter_info.keys()) == set(first_iter.keys())

    else:
        # Unpack without callback
        optimal_budgets, opt_result = result
        assert result.callback_info is None

    # Common checks
    assert isinstance(optimal_budgets, xr.DataArray)
    assert hasattr(opt_result, "x")
    assert hasattr(opt_result, "success")

    # Check budget allocation sums to total
    assert np.abs(optimal_budgets.sum().item() - total_budget) < 1e-3


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
