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
from pytensor.graph.traversal import ancestors
from xarray import DataArray

import pymc_marketing.mmm.budget_optimizer as budget_optimizer_module
from pymc_marketing.mmm import MMM, BudgetOptimizerWrapper
from pymc_marketing.mmm.additive_effect import (
    DiscountedEventEffect,
    MuEffect,
    OptimizableMuEffect,
)
from pymc_marketing.mmm.budget_optimizer import (
    BudgetOptimizer,
    CustomModelWrapper,
    MinimizeException,
    optimizer_xarray_builder,
)
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.constraints import Constraint
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

    with pytest.raises(
        ValueError, match=r"NumPy boolean array indexing assignment cannot assign"
    ):
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
    "callback, total_budget, expected_return_length",
    [
        # Basic cases
        (False, 100, 2),  # Default behavior - no callback
        (True, 100, 3),  # With callback
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
    expected_return_length,
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

    # Check return length
    assert len(result) == expected_return_length

    if callback:
        # Unpack with callback
        optimal_budgets, opt_result, callback_info = result

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

    # Common checks
    assert isinstance(optimal_budgets, xr.DataArray)
    assert hasattr(opt_result, "x")
    assert hasattr(opt_result, "success")

    # Check budget allocation sums to total
    assert np.abs(optimal_budgets.sum().item() - total_budget) < 1e-3


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


def test_budget_optimizer_with_optimizable_mu_effect(mock_pymc_sample):
    """A concrete OptimizableMuEffect flows into BudgetOptimizer via the MMM API.

    A minimal per-item lever (dim "promo") is added via `add_mu_effect`, the
    model is built/fit through the normal `MMM` API, and `mmm.budget_optimizer`
    wires `mu_effects` into `BudgetOptimizer` automatically. Checks the
    inert-lever guard (default media-only objective has no gradient for the
    lever -> raise), that the effect's own data node depends on the optimizer's
    flat decision vector, and that the lever stays out of the default sum
    constraint -- only media sums to `total_budget`.
    """

    class PromoEffect(OptimizableMuEffect):
        """A per-item lever contributing a constant boost to mu."""

        prefix: str = "promo"

        def create_data(self, mmm) -> None:
            model = mmm.model
            model.add_coord(self.prefix, ["evt1", "evt2"])
            pmd.Data(f"{self.prefix}_data", np.ones(2), dims=self.prefix)

        def create_effect(self, mmm):
            model = mmm.model
            data = model[f"{self.prefix}_data"]
            coef = pmd.HalfNormal(f"{self.prefix}_coef", sigma=1.0, dims=self.prefix)
            contribution = pmd.Deterministic(
                f"{self.prefix}_effect_contribution", data * coef, dims=self.prefix
            )
            return contribution.sum(dim=self.prefix)

        def set_data(self, mmm, model, X) -> None:
            pass

        @property
        def lever_bounds(self):
            return [(0.0, 1.0), (0.0, 1.0)]

        # The lever integrates by name: MMM.budget_optimizer translates this
        # effect into an optimizable_vars entry for its promo_data node.

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
    ).add_mu_effect(PromoEffect())

    mmm.fit(X, y, random_seed=0)

    # Inert-lever guard: the default media-only objective does not depend on
    # the promo lever, so its gradient is identically zero -- constructing the
    # optimizer raises instead of silently returning the seed as an "optimum".
    with pytest.raises(ValueError, match="does not depend on optimizable_vars"):
        mmm.budget_optimizer(
            start_date=date_range[-1] + pd.Timedelta(weeks=1),
            end_date=date_range[-1] + pd.Timedelta(weeks=4),
        )

    optimizer = mmm.budget_optimizer(
        start_date=date_range[-1] + pd.Timedelta(weeks=1),
        end_date=date_range[-1] + pd.Timedelta(weeks=4),
        response_variable="total_response_original_scale",
    )

    # The effect's own data node should now depend on the optimizer's flat
    # decision vector, not the ones it was created with.
    promo_data = optimizer._pymc_model["promo_data"]
    assert optimizer._budgets_flat in ancestors([promo_data])

    optimal_budgets, result = optimizer.allocate_budget(total_budget=100.0)

    assert result.success
    # The promo lever stays out of the sum constraint -- only the media
    # entries (the first two) sum to total_budget.
    assert np.isclose(result.x[:2].sum(), 100.0)
    # Media allocation comes back over the budget dims and sums to total_budget.
    assert np.isclose(float(optimal_budgets.sum()), 100.0)

    # The effect's optimal lever is decoded off the tail of result.x into a
    # DataArray over the effect's own dim/coords.
    promo_opt = result.optimized_vars["promo_data"]
    assert promo_opt.dims == ("promo",)
    assert list(promo_opt.coords["promo"].values) == ["evt1", "evt2"]
    np.testing.assert_allclose(promo_opt.values, result.x[2:])
    # The positive-coefficient contribution gives the objective a positive
    # gradient in promo_data, so the lever climbs to its upper bound.
    np.testing.assert_allclose(promo_opt.values, 1.0, atol=1e-6)

    # result.fun is in original objective units (the internal |f(x0)|
    # normalization is undone before returning): re-evaluating the raw
    # compiled objective at the solution must match.
    raw_obj, _ = optimizer._compiled_functions[optimizer.utility_function][
        "objective_and_grad"
    ](result.x.copy())
    np.testing.assert_allclose(float(result.fun), float(raw_obj), rtol=1e-10)

    # Default x0 spreads total_budget over the media head only (feasible
    # w.r.t. the sum constraint) and warm-starts effect levers at their
    # current model value (np.ones(2) here, within the (0, 1) bounds). With
    # maxiter=0 the solver returns x0 unchanged, exposing the seed.
    _, result_x0 = optimizer.allocate_budget(
        total_budget=100.0,
        minimize_kwargs={"options": {"maxiter": 0}},
        return_if_fail=True,
    )
    np.testing.assert_allclose(result_x0.x[:2], [50.0, 50.0])
    np.testing.assert_allclose(result_x0.x[2:], 1.0)


@pytest.mark.parametrize("link", ["identity", "log"])
def test_discounted_event_effect_optimization_end_to_end(mock_pymc_sample, link):
    """DiscountedEventEffect prescribes event-specific depths matching the FOC.

    Two events with (near-)pinned betas: both links use the same exact
    repricing multiplier (1 - d)(1 + d)^beta, so both prescribe the same
    analytic optimum d* = (beta - 1) / (beta + 1) -- the link-consistency
    property. The stronger event justifies the deeper discount.

    Also asserts media still sums to the budget (the discount stays out of the
    cash pot) and depths are decoded into result.optimized_vars.
    """
    from pymc_extras.prior import Prior

    date_range = pd.date_range("2023-01-01", periods=20, freq="W")
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "date": date_range,
            "ch1": rng.uniform(100, 500, size=len(date_range)),
            "ch2": rng.uniform(100, 500, size=len(date_range)),
        }
    )
    y = pd.Series(rng.uniform(500, 1500, size=len(date_range)), name="target")

    betas = np.array([2.5, 1.2])  # strong_sale, weak_sale
    effect = DiscountedEventEffect(
        df_events=pd.DataFrame(
            {
                "name": ["strong_sale", "weak_sale"],
                "start_date": ["2023-02-01", "2023-04-01"],
                "end_date": ["2023-03-01", "2023-04-28"],
                "discount_pct": [0.10, 0.10],
            }
        ),
        prefix="promo",
        # Near-degenerate prior pins beta so the optimum is analytic.
        beta_prior=Prior("Normal", mu=betas, sigma=0.001),
        discount_min=0.02,
        discount_max=0.45,
    )

    mmm = MMM(
        date_column="date",
        channel_columns=["ch1", "ch2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        link=link,
    ).add_mu_effect(effect)
    mmm.fit(X, y, random_seed=1)

    # The LinkSpec registers the full-response objective for both links.
    assert "total_response_original_scale" in mmm.model.named_vars

    # Retrospective window covering both event windows.
    optimizer = mmm.budget_optimizer(
        start_date=date_range[2],
        end_date=date_range[18],
        response_variable="total_response_original_scale",
    )
    optimal_budgets, result = optimizer.allocate_budget(total_budget=100.0)

    assert result.success
    # Media still sums to the budget; the discount stays out of the pot.
    assert np.isclose(float(optimal_budgets.sum()), 100.0)

    depths = result.optimized_vars["promo_data"]
    assert depths.dims == ("promo",)
    assert list(depths.coords["promo"].values) == ["strong_sale", "weak_sale"]
    depth_values = depths.values
    # Native bounds respected.
    assert ((depth_values >= 0.02 - 1e-8) & (depth_values <= 0.45 + 1e-8)).all()

    # Both links share the exact multiplier (1-d)(1+d)^beta, so the analytic
    # FOC applies to both: d* = (beta-1)/(beta+1).
    expected = (betas - 1.0) / (betas + 1.0)
    np.testing.assert_allclose(depth_values, expected, atol=0.02)
    # Event-specific: the stronger event justifies the deeper discount.
    assert depth_values[0] > depth_values[1]


def test_optimizable_vars_names_only(mock_pymc_sample):
    """`optimizable_vars` works from a variable name alone -- no effect object.

    Build a model containing an extra pm.Data node via a plain (non-optimizable)
    MuEffect, then co-optimize that node purely by name with native bounds. The
    optimizer never inspects the effect; it reads dims/coords off the graph.
    """

    class PlainPromoEffect(MuEffect):
        prefix: str = "promo"

        def create_data(self, mmm) -> None:
            model = mmm.model
            model.add_coord(self.prefix, ["evt1", "evt2"])
            pmd.Data(f"{self.prefix}_data", np.ones(2), dims=self.prefix)

        def create_effect(self, mmm):
            model = mmm.model
            data = model[f"{self.prefix}_data"]
            coef = pmd.HalfNormal(f"{self.prefix}_coef", sigma=1.0, dims=self.prefix)
            contribution = pmd.Deterministic(
                f"{self.prefix}_effect_contribution", data * coef, dims=self.prefix
            )
            return contribution.sum(dim=self.prefix)

        def set_data(self, mmm, model, X) -> None:
            pass

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
    ).add_mu_effect(PlainPromoEffect())
    mmm.fit(X, y, random_seed=0)

    optimizer = mmm.budget_optimizer(
        start_date=date_range[-1] + pd.Timedelta(weeks=1),
        end_date=date_range[-1] + pd.Timedelta(weeks=4),
        optimizable_vars={"promo_data": [(0.0, 1.0), (0.0, 1.0)]},
        response_variable="total_response_original_scale",
    )
    optimal_budgets, result = optimizer.allocate_budget(total_budget=100.0)

    assert result.success
    assert np.isclose(float(optimal_budgets.sum()), 100.0)
    promo_opt = result.optimized_vars["promo_data"]
    assert promo_opt.dims == ("promo",)
    assert ((promo_opt.values >= 0.0) & (promo_opt.values <= 1.0)).all()
    # The lever moved off its 0.0 seed: the positive-coefficient contribution
    # gives the objective a positive gradient in promo_data.
    assert (promo_opt.values > 1e-4).all()


def test_optimizable_vars_unknown_name_raises(dummy_df, dummy_idata):
    """A name that is not a dims-registered model variable raises clearly."""
    df_kwargs, X_dummy, y_dummy = dummy_df
    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )
    mmm.build_model(X=X_dummy, y=y_dummy)

    with pytest.raises(ValueError, match="not a variable with named dims"):
        BudgetOptimizer(
            model=mmm.model,
            idata=dummy_idata,
            num_periods=6,
            adstock_periods=4,
            optimizable_vars={"nonexistent_data": None},
        )


def test_budget_optimizer_has_no_marketing_imports():
    """budget_optimizer.py operates purely on the pm.Model graph.

    It must not import from the marketing layer (additive_effect, mmm):
    OptimizableMuEffect levers reach it only as `optimizable_vars` name/bounds
    entries, translated by MMM.budget_optimizer.
    """
    banned = ("pymc_marketing.mmm.additive_effect", "pymc_marketing.mmm.mmm")
    tree = ast.parse(inspect.getsource(budget_optimizer_module))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.startswith(banned):
                offenders.append(node.module)
        elif isinstance(node, ast.Import):
            offenders.extend(
                alias.name for alias in node.names if alias.name.startswith(banned)
            )
    assert not offenders, f"budget_optimizer imports marketing modules: {offenders}"


def test_optimizable_vars_bounds_length_mismatch_raises(mock_pymc_sample):
    """Bounds with the wrong number of entries for the variable raise clearly."""

    class PromoEffect(OptimizableMuEffect):
        prefix: str = "promo"

        def create_data(self, mmm) -> None:
            model = mmm.model
            model.add_coord(self.prefix, ["evt1", "evt2"])
            pmd.Data(f"{self.prefix}_data", np.ones(2), dims=self.prefix)

        def create_effect(self, mmm):
            model = mmm.model
            data = model[f"{self.prefix}_data"]
            coef = pmd.HalfNormal(f"{self.prefix}_coef", sigma=1.0, dims=self.prefix)
            contribution = pmd.Deterministic(
                f"{self.prefix}_effect_contribution", data * coef, dims=self.prefix
            )
            return contribution.sum(dim=self.prefix)

        def set_data(self, mmm, model, X) -> None:
            pass

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
    ).add_mu_effect(PromoEffect())
    mmm.fit(X, y, random_seed=0)

    # The mismatch is knowable at construction: it raises there, not at
    # allocate_budget time.
    with pytest.raises(ValueError, match="bounds have 1 entries"):
        mmm.budget_optimizer(
            start_date=date_range[-1] + pd.Timedelta(weeks=1),
            end_date=date_range[-1] + pd.Timedelta(weeks=4),
            optimizable_vars={"promo_data": [(0.0, 1.0)]},  # variable has 2 entries
        )


def test_optimizable_vars_multidim_var_raises(panel_fitted_mmm):
    """A variable with more than one non-date dim is rejected at construction."""
    date_range = pd.date_range("2023-01-01", periods=14, freq="W")
    with pytest.raises(ValueError, match="exactly one"):
        panel_fitted_mmm.budget_optimizer(
            start_date=date_range[-1] + pd.Timedelta(weeks=1),
            end_date=date_range[-1] + pd.Timedelta(weeks=4),
            # channel_data has (date, country, channel) -> two non-date dims
            optimizable_vars={"channel_data": None},
        )


def test_optimized_vars_empty_without_optimizable_vars(mock_pymc_sample):
    """Backward compat: plain optimizations return result.optimized_vars == {}."""
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
    )
    mmm.fit(X, y, random_seed=0)

    optimizer = mmm.budget_optimizer(
        start_date=date_range[-1] + pd.Timedelta(weeks=1),
        end_date=date_range[-1] + pd.Timedelta(weeks=4),
    )
    _, result = optimizer.allocate_budget(total_budget=100.0)
    assert result.success
    assert result.optimized_vars == {}


def test_optimizable_vars_empty_dict_opts_out(mock_pymc_sample):
    """Explicit optimizable_vars={} disables lever auto-injection.

    Re-planning media against a fixed discount calendar must be possible:
    with the opt-out, the default media-only response variable is valid again
    (no inert-lever raise) and no levers are optimized.
    """
    date_range = pd.date_range("2023-01-01", periods=20, freq="W")
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "date": date_range,
            "ch1": rng.uniform(100, 500, size=len(date_range)),
            "ch2": rng.uniform(100, 500, size=len(date_range)),
        }
    )
    y = pd.Series(rng.uniform(500, 1500, size=len(date_range)), name="target")
    effect = DiscountedEventEffect(
        df_events=pd.DataFrame(
            {
                "name": ["spring_sale"],
                "start_date": ["2023-02-01"],
                "end_date": ["2023-03-15"],
                "discount_pct": [0.10],
            }
        ),
        prefix="promo",
        discount_min=0.05,
        discount_max=0.45,
    )
    mmm = MMM(
        date_column="date",
        channel_columns=["ch1", "ch2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    ).add_mu_effect(effect)
    mmm.fit(X, y, random_seed=1)

    optimizer = mmm.budget_optimizer(
        start_date=date_range[3],
        end_date=date_range[10],
        optimizable_vars={},  # opt out: media-only, discount calendar fixed
    )
    optimal_budgets, result = optimizer.allocate_budget(total_budget=100.0)
    assert result.success
    assert np.isclose(float(optimal_budgets.sum()), 100.0)
    assert result.optimized_vars == {}


def test_optimizable_vars_out_of_window_lever_warns(mock_pymc_sample):
    """A lever whose event window is empty in the optimization window warns.

    The structural ancestry guard cannot see this (the lever is connected on
    the graph); the numeric gradient check at x0 catches it, and the lever
    comes back at its current model value rather than a bound.
    """
    date_range = pd.date_range("2023-01-01", periods=20, freq="W")
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "date": date_range,
            "ch1": rng.uniform(100, 500, size=len(date_range)),
            "ch2": rng.uniform(100, 500, size=len(date_range)),
        }
    )
    y = pd.Series(rng.uniform(500, 1500, size=len(date_range)), name="target")
    effect = DiscountedEventEffect(
        df_events=pd.DataFrame(
            {
                # early_event lies before the optimization window below.
                "name": ["early_event", "late_event"],
                "start_date": ["2023-01-01", "2023-03-01"],
                "end_date": ["2023-01-21", "2023-04-01"],
                "discount_pct": [0.10, 0.10],
            }
        ),
        prefix="promo",
        discount_min=0.02,
        discount_max=0.45,
    )
    mmm = MMM(
        date_column="date",
        channel_columns=["ch1", "ch2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    ).add_mu_effect(effect)
    mmm.fit(X, y, random_seed=1)

    optimizer = mmm.budget_optimizer(
        start_date=date_range[6],  # after early_event's window
        end_date=date_range[18],
        response_variable="total_response_original_scale",
    )
    with pytest.warns(UserWarning, match="early_event"):
        _, result = optimizer.allocate_budget(total_budget=100.0)

    depths = result.optimized_vars["promo_data"]
    # The inert lever is returned at its current model value (the historical
    # depth), not at a bound.
    np.testing.assert_allclose(float(depths.sel(promo="early_event")), 0.10)
    # The in-window lever is genuinely optimized.
    assert float(depths.sel(promo="late_event")) > 0.10


def test_optimize_budget_wires_effect_levers(mock_pymc_sample):
    """BudgetOptimizerWrapper.optimize_budget passes effect levers through.

    A model with a DiscountedEventEffect optimized via the legacy
    ``optimize_budget`` API must co-optimize the discount lever, not silently
    drop it.
    """
    date_range = pd.date_range("2023-01-01", periods=20, freq="W")
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "date": date_range,
            "ch1": rng.uniform(100, 500, size=len(date_range)),
            "ch2": rng.uniform(100, 500, size=len(date_range)),
        }
    )
    y = pd.Series(rng.uniform(500, 1500, size=len(date_range)), name="target")

    effect = DiscountedEventEffect(
        df_events=pd.DataFrame(
            {
                "name": ["spring_sale"],
                "start_date": ["2023-02-01"],
                "end_date": ["2023-03-15"],
                "discount_pct": [0.10],
            }
        ),
        prefix="promo",
        discount_min=0.05,
        discount_max=0.45,
    )
    mmm = MMM(
        date_column="date",
        channel_columns=["ch1", "ch2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    ).add_mu_effect(effect)
    mmm.fit(X, y, random_seed=1)

    with pytest.warns(DeprecationWarning, match="BudgetOptimizerWrapper"):
        wrapper = BudgetOptimizerWrapper(
            model=mmm,
            start_date=date_range[2],
            end_date=date_range[12],
        )

    # The default media-only response cannot reach the lever: the guard fires
    # instead of silently returning the seed.
    with pytest.raises(ValueError, match="does not depend on optimizable_vars"):
        wrapper.optimize_budget(budget=100.0)

    _optimal_budgets, result = wrapper.optimize_budget(
        budget=100.0,
        response_variable="total_response_original_scale",
    )
    assert result.success
    depths = result.optimized_vars["promo_data"]
    assert list(depths.coords["promo"].values) == ["spring_sale"]
    depth = float(depths.sel(promo="spring_sale"))
    assert 0.05 - 1e-8 <= depth <= 0.45 + 1e-8
