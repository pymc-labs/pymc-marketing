#   Copyright 2022 - 2025 The PyMC Labs Developers
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
from unittest.mock import patch

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr
from pymc.model.fgraph import clone_model as cm

from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.budget_optimizer import (
    BudgetOptimizer,
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
    # Data is not needed for optimization of this model
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

    y = np.ones(n)

    df_kwargs = {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "control_columns": ["event_1", "event_2", "t"],
    }

    return df_kwargs, df, y


@pytest.fixture(scope="module")
def dummy_idata(dummy_df) -> az.InferenceData:
    df_kwargs, df, y = dummy_df

    return az.from_dict(
        posterior={
            "saturation_lam": [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
            "saturation_beta": [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]],
            "adstock_alpha": [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]],
            "channel_contribution": np.array(
                [
                    [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                    [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                ]
            ),  # dims: chain, draw, channel, date
        },
        coords={
            "chain": [0, 1],
            "draw": [0, 1],
            "channel": df_kwargs["channel_columns"],
            "date": [0, 1],
        },
        dims={
            "saturation_lam": ["chain", "draw", "channel"],
            "saturation_beta": ["chain", "draw", "channel"],
            "adstock_alpha": ["chain", "draw", "channel"],
            "channel_contribution": ["chain", "draw", "channel", "date"],
        },
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
            {"channel_1": 54.78357587906867, "channel_2": 45.21642412093133},
            48.8,
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
            {"channel_1": 54.78357587906867, "channel_2": 45.21642412093133},
            48.8,
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
            48.8,
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
            {"channel_1": 0.0, "channel_2": 7.94e-13},
            2.38e-10,
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
    dummy_df,
    dummy_idata,
):
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )

    mmm.build_model(X=X_dummy, y=y_dummy)

    mmm.idata = dummy_idata

    # Create BudgetOptimizer Instance
    match = "Using default equality constraint"
    with pytest.warns(UserWarning, match=match):
        optimizer = BudgetOptimizer(
            model=mmm,
            num_periods=30,
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


@patch("pymc_marketing.mmm.budget_optimizer.minimize")
def test_allocate_budget_custom_minimize_args(
    minimize_mock, dummy_df, dummy_idata
) -> None:
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )
    mmm.build_model(X=X_dummy, y=y_dummy)
    mmm.idata = dummy_idata

    match = "Using default equality constraint"
    with pytest.warns(UserWarning, match=match):
        optimizer = BudgetOptimizer(
            model=mmm,
            num_periods=30,
        )

    total_budget = 100
    budget_bounds = {"channel_1": (0.0, 50.0), "channel_2": (0.0, 50.0)}
    minimize_kwargs = {
        "method": "SLSQP",
        "options": {"ftol": 1e-8, "maxiter": 1_002},
    }

    with pytest.raises(
        ValueError, match="NumPy boolean array indexing assignment cannot assign"
    ):
        optimizer.allocate_budget(
            total_budget, budget_bounds, minimize_kwargs=minimize_kwargs
        )

    kwargs = minimize_mock.call_args_list[0].kwargs

    np.testing.assert_array_equal(x=kwargs["x0"], y=np.array([50.0, 50.0]))
    assert kwargs["bounds"] == [(0.0, 50.0), (0.0, 50.0)]
    assert kwargs["method"] == minimize_kwargs["method"]
    assert kwargs["options"] == minimize_kwargs["options"]


@pytest.mark.parametrize(
    "total_budget, budget_bounds, parameters, custom_constraints",
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
                    constraint_fun=lambda budgets_sym,
                    total_budget_sym,
                    optimizer: budgets_sym[0] - 60,
                    constraint_type="ineq",
                ),
            ],
        ),
    ],
)
def test_allocate_budget_infeasible_constraints(
    total_budget, budget_bounds, parameters, custom_constraints, dummy_df, dummy_idata
):
    df_kwargs, X_dummy, y_dummy = dummy_df

    # Define the MMM model
    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )
    mmm.build_model(X=X_dummy, y=y_dummy)

    # Load necessary parameters into the model
    mmm.idata = dummy_idata

    # Instantiate BudgetOptimizer with custom constraints
    optimizer = BudgetOptimizer(
        model=mmm,
        response_variable="total_contribution",
        default_constraints=False,  # Avoid default equality constraints
        custom_constraints=custom_constraints,
        num_periods=30,
    )

    # Ensure optimization raises MinimizeException due to infeasible constraints
    with pytest.raises(MinimizeException, match="Optimization failed"):
        optimizer.allocate_budget(total_budget, budget_bounds)


def mean_response_eq_constraint_fun(
    budgets_sym, total_budget_sym, optimizer, target_response
):
    """
    Enforces mean_response(budgets_sym) = target_response,
    i.e. returns (mean_resp - target_response).
    """
    resp_dist = optimizer.extract_response_distribution("total_contribution")
    mean_resp = pt.mean(_check_samples_dimensionality(resp_dist))
    return mean_resp - target_response


def minimize_budget_utility(samples, budgets):
    """
    A trivial "utility" that just tries to minimize total budget.
    Since the BudgetOptimizer by default *maximizes* the utility,
    we use the negative sign to effectively force minimization.
    """
    return -pt.sum(budgets)


@pytest.mark.parametrize(
    "total_budget,target_response",
    [
        (10, 5.0),
        (50, 10.0),
    ],
    ids=["budget=10->resp=5", "budget=50->resp=10"],
)
def test_allocate_budget_custom_response_constraint(
    dummy_df, total_budget, target_response, dummy_idata
):
    """
    Checks that a custom constraint can enforce the model's mean response
    to equal a target value, while we minimize the total budget usage.
    """
    # Extract the dummy data and define the MMM model
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )
    mmm.build_model(X_dummy, y_dummy)

    # Provide some dummy posterior samples
    mmm.idata = dummy_idata

    def constraint_wrapper(budgets_sym, total_budget_sym, optimizer):
        return mean_response_eq_constraint_fun(
            budgets_sym, total_budget_sym, optimizer, target_response
        )

    custom_constraints = [
        Constraint(
            key="target_response_constraint",
            constraint_fun=constraint_wrapper,
            constraint_type="eq",
        )
    ]

    optimizer = BudgetOptimizer(
        model=mmm,
        response_variable="total_contribution",
        utility_function=minimize_budget_utility,
        default_constraints=False,
        custom_constraints=custom_constraints,
        num_periods=30,
    )

    allocation, res = optimizer.allocate_budget(
        total_budget=total_budget,
        budget_bounds=None,
    )

    resp_dist_sym = optimizer.extract_response_distribution("total_contribution")
    resp_mean_sym = pt.mean(_check_samples_dimensionality(resp_dist_sym))
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
    dummy_df,
    dummy_idata,
    callback,
    total_budget,
    expected_return_length,
):
    """Test callback functionality with various parameter combinations."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )

    mmm.build_model(X=X_dummy, y=y_dummy)
    mmm.idata = dummy_idata

    # Create BudgetOptimizer Instance
    match = "Using default equality constraint"
    with pytest.warns(UserWarning, match=match):
        optimizer = BudgetOptimizer(
            model=mmm,
            num_periods=30,
        )

    # Run allocation
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
    "callback",
    [
        False,  # Default no callback
        True,  # With callback
    ],
    ids=[
        "no_callback",
        "with_callback",
    ],
)
def test_mmm_optimize_budget_callback_parametrized(dummy_df, dummy_idata, callback):
    """Test that MMM.optimize_budget properly raises deprecation error."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )

    mmm.build_model(X=X_dummy, y=y_dummy)
    mmm.idata = dummy_idata

    with pytest.warns(
        DeprecationWarning,
        match="This method is deprecated and will be removed in a future version",
    ):
        result = mmm.optimize_budget(
            budget=100,
            num_periods=10,
            callback=callback,
        )

    # Check return value count
    if callback:
        assert len(result) == 3
        optimal_budgets, opt_result, callback_info = result

        # Validate callback info
        assert isinstance(callback_info, list)
        assert len(callback_info) > 0

        # Each iteration should have required keys
        for iter_info in callback_info:
            assert "x" in iter_info
            assert "fun" in iter_info
            assert "jac" in iter_info

        # Check that objective values are finite
        objectives = [iter_info["fun"] for iter_info in callback_info]
        assert all(np.isfinite(obj) for obj in objectives)

    else:
        assert len(result) == 2
        optimal_budgets, opt_result = result

    # Common validations
    assert isinstance(optimal_budgets, xr.DataArray)
    assert optimal_budgets.dims == ("channel",)
    assert len(optimal_budgets) == len(mmm.channel_columns)

    # Budget should sum to total (within tolerance)
    assert np.abs(optimal_budgets.sum().item() - 100) < 1e-6

    # Check optimization result
    assert hasattr(opt_result, "success")
    assert hasattr(opt_result, "x")
    assert hasattr(opt_result, "fun")


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
            "budget_distribution_over_period must sum to 1 along the date dimension",
        ),
        # Invalid case: wrong number of periods
        (
            {"channel_1": [0.5, 0.5], "channel_2": [0.5, 0.5]},
            4,
            True,
            "budget_distribution_over_period date dimension must have length 4",
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
    dummy_df,
    dummy_idata,
    budget_distribution_over_period,
    num_periods,
    should_error,
    error_message,
):
    """Test that budget_distribution_over_period correctly distributes budget over time."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )

    mmm.build_model(X=X_dummy, y=y_dummy)
    mmm.idata = dummy_idata

    # Create time distribution factors DataArray
    if budget_distribution_over_period is not None:
        budget_distribution_over_period_array = np.array(
            [budget_distribution_over_period[ch] for ch in df_kwargs["channel_columns"]]
        )
        budget_distribution_over_period_factors = xr.DataArray(
            budget_distribution_over_period_array,
            coords={
                "channel": df_kwargs["channel_columns"],
                "date": list(range(len(budget_distribution_over_period["channel_1"]))),
            },
            dims=["channel", "date"],
        )
    else:
        budget_distribution_over_period_factors = None

    if should_error:
        with pytest.raises(ValueError, match=error_message):
            BudgetOptimizer(
                model=mmm,
                num_periods=num_periods,
                budget_distribution_over_period=budget_distribution_over_period_factors,
                default_constraints=True,
            )
    else:
        # Create optimizer with time distribution factors
        match = "Using default equality constraint"
        with pytest.warns(UserWarning, match=match):
            optimizer = BudgetOptimizer(
                model=mmm,
                num_periods=num_periods,
                budget_distribution_over_period=budget_distribution_over_period_factors,
                default_constraints=True,
            )

        # Check that the time distribution factors were stored correctly
        if budget_distribution_over_period_factors is not None:
            assert optimizer._budget_distribution_over_period_tensor is not None
            # The tensor is now pre-processed and has shape (num_periods, num_optimized_budgets)
            num_optimized = optimizer.budgets_to_optimize.sum().item()
            expected_shape = (num_periods, num_optimized)
            assert (
                tuple(optimizer._budget_distribution_over_period_tensor.shape.eval())
                == expected_shape
            )
        else:
            assert optimizer._budget_distribution_over_period_tensor is None


def test_budget_distribution_over_period_wrong_dims(dummy_df, dummy_idata):
    """Test that budget_distribution_over_period with wrong dimensions raises error."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )

    mmm.build_model(X=X_dummy, y=y_dummy)
    mmm.idata = dummy_idata

    # Create time factors with wrong dimensions (missing channel dimension)
    budget_distribution_over_period = xr.DataArray(
        [0.25, 0.25, 0.25, 0.25],
        coords={"date": list(range(4))},
        dims=["date"],
    )

    with pytest.raises(
        ValueError, match="budget_distribution_over_period must have dims"
    ):
        BudgetOptimizer(
            model=mmm,
            num_periods=4,
            budget_distribution_over_period=budget_distribution_over_period,
            default_constraints=True,
        )


def test_budget_distribution_over_period_applied_correctly(dummy_df, dummy_idata):
    """Test that budget distribution factors are correctly applied to budgets."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )

    mmm.build_model(X=X_dummy, y=y_dummy)
    mmm.idata = dummy_idata

    # Create non-uniform time distribution factors
    budget_distribution_over_period_data = {
        "channel_1": [0.7, 0.2, 0.1, 0.0],
        "channel_2": [0.4, 0.3, 0.2, 0.1],
    }
    budget_distribution_over_period_array = np.array(
        [
            budget_distribution_over_period_data[ch]
            for ch in df_kwargs["channel_columns"]
        ]
    )
    budget_distribution_over_period_factors = xr.DataArray(
        budget_distribution_over_period_array,
        coords={
            "channel": df_kwargs["channel_columns"],
            "date": list(range(4)),
        },
        dims=["channel", "date"],
    )

    # Create optimizer with time distribution factors
    match = "Using default equality constraint"
    with pytest.warns(UserWarning, match=match):
        optimizer = BudgetOptimizer(
            model=mmm,
            num_periods=4,
            budget_distribution_over_period=budget_distribution_over_period_factors,
            default_constraints=True,
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


def test_budget_distribution_over_period_integration(dummy_df, dummy_idata):
    """Integration test: verify budget allocation with time distribution factors."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )

    mmm.build_model(X=X_dummy, y=y_dummy)
    mmm.idata = dummy_idata

    # Create front-loaded time distribution
    num_periods = 4
    budget_distribution_over_period_data = {
        "channel_1": [0.7, 0.2, 0.1, 0.0],  # Heavy front-loading
        "channel_2": [0.25, 0.25, 0.25, 0.25],  # Uniform distribution
    }
    budget_distribution_over_period_array = np.array(
        [
            budget_distribution_over_period_data[ch]
            for ch in df_kwargs["channel_columns"]
        ]
    )
    budget_distribution_over_period_factors = xr.DataArray(
        budget_distribution_over_period_array,
        coords={
            "channel": df_kwargs["channel_columns"],
            "date": list(range(num_periods)),
        },
        dims=["channel", "date"],
    )

    # Create two optimizers: one with and one without time distribution
    optimizer_with_factors = BudgetOptimizer(
        model=mmm,
        num_periods=num_periods,
        budget_distribution_over_period=budget_distribution_over_period_factors,
        default_constraints=True,
    )

    optimizer_without_factors = BudgetOptimizer(
        model=mmm,
        num_periods=num_periods,
        budget_distribution_over_period=None,
        default_constraints=True,
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


def test_custom_protocol_model_budget_optimizer_works():
    """Validate the optimizer works with a custom model that follows the protocol.

    This serves as an example for users wanting to plug in their own PyMC models.
    Requirements implemented here:
    - The model has a variable named 'channel_data' with dims ("date", "channel").
    - Deterministics 'channel_contribution' ("date", "channel") and 'total_contribution' ("date").
    - A wrapper object exposes: idata, channel_columns, _channel_scales, adstock.l_max, and
      a method `_set_predictors_for_optimization(num_periods) -> pm.Model` that returns a PyMC model
      where 'channel_data' is set for the optimization horizon.
    """
    # 1) Build and fit a tiny custom PyMC model
    rng = np.random.default_rng(0)
    num_obs = 12
    channels = ["C1", "C2", "C3"]
    X = rng.uniform(0.0, 1.0, size=(num_obs, len(channels)))
    true_beta = np.array([0.8, 0.4, 0.2])
    y = (X @ true_beta) + rng.normal(0.0, 0.05, size=num_obs)

    coords = {"date": np.arange(num_obs), "channel": channels}
    with pm.Model(coords=coords) as train_model:
        pm.Data("channel_data", X, dims=("date", "channel"))
        beta = pm.Normal("beta", 0.0, 1.0, dims="channel")
        mu = (train_model["channel_data"] * beta).sum(axis=-1)
        pm.Deterministic("total_contribution", mu.sum(), dims=())
        pm.Deterministic(
            "channel_contribution",
            train_model["channel_data"] * beta,
            dims=("date", "channel"),
        )
        sigma = pm.HalfNormal("sigma", 0.2)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="date")

        idata = pm.sample(50, tune=50, chains=1, progressbar=False, random_seed=1)

    # 2) Minimal wrapper satisfying the optimizer protocol
    class SimpleWrapper:
        def __init__(self, base_model: pm.Model, idata, channels):
            self._base_model = base_model
            self.idata = idata
            self.channel_columns = list(channels)
            self._channel_scales = 1.0
            self.adstock = type("Adstock", (), {"l_max": 0})()  # no carryover

        def _set_predictors_for_optimization(self, num_periods: int) -> pm.Model:
            m = cm(self._base_model)
            pm.set_data(
                {
                    "channel_data": np.zeros(
                        (num_periods, len(self.channel_columns)),
                        dtype=m["channel_data"].dtype,
                    )
                },
                coords={
                    "date": np.arange(num_periods),
                    "channel": self.channel_columns,
                },
                model=m,
            )
            return m

    wrapper = SimpleWrapper(base_model=train_model, idata=idata, channels=channels)

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
