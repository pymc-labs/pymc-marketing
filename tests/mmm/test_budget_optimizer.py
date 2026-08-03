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
import pytensor.xtensor as ptx
import pytest
import xarray as xr
from pydantic import ValidationError
from pytensor.graph.traversal import ancestors
from xarray import DataArray

import pymc_marketing.mmm.budget_optimizer as budget_optimizer_module
from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.additive_effect import (
    MuEffect,
    OptimizableMuEffect,
)
from pymc_marketing.mmm.budget_optimizer import (
    BudgetOptimizationResult,
    BudgetOptimizer,
    CustomModelWrapper,
    MinimizeException,
    optimizer_xarray_builder,
)
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.constraints import Constraint
from pymc_marketing.mmm.mmm import BudgetOptimizerWrapper
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

    # The mocked minimize returns a Mock result.x, which fails the decision
    # space's shape validation when unpacking — after minimize was called.
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


@pytest.fixture
def optimizer_with_zero_gradient(mmm_wrapper):
    """Optimizer whose compiled gradient is identically zero everywhere."""
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )
    optimizer._objective_and_grad = lambda x: (np.float64(0.0), np.zeros_like(x))
    return optimizer


def test_zero_gradient_block_warns(optimizer_with_zero_gradient):
    """A block the objective cannot see is reported by name."""
    bounds = optimizer_xarray_builder(
        np.array([[0, 50], [0, 50]]),
        channel=["channel_1", "channel_2"],
        bound=["lower", "upper"],
    )
    with pytest.warns(UserWarning, match=r"\['channel_data'\].*zero gradient"):
        optimizer_with_zero_gradient.allocate_budget(
            total_budget=100, budget_bounds=bounds
        )


def test_zero_gradient_pinned_block_does_not_warn(optimizer_with_zero_gradient):
    """Coordinates pinned by degenerate bounds (low == high) are not probed."""
    bounds = optimizer_xarray_builder(
        np.array([[50, 50], [50, 50]]),
        channel=["channel_1", "channel_2"],
        bound=["lower", "upper"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        optimizer_with_zero_gradient.allocate_budget(
            total_budget=100, budget_bounds=bounds
        )


def test_nonzero_gradient_does_not_warn(mmm_wrapper):
    """A block that reaches the response produces no zero-gradient warning."""
    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
    )
    bounds = optimizer_xarray_builder(
        np.array([[0, 100], [0, 100]]),
        channel=["channel_1", "channel_2"],
        bound=["lower", "upper"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        optimizer.allocate_budget(total_budget=100, budget_bounds=bounds)


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


class _PromoLeverEffect(OptimizableMuEffect):
    """Test-only lever: a constant per-event boost to mu, no date structure."""

    prefix: str = "promo"
    names: list[str] = ["spring_sale"]

    def create_data(self, mmm) -> None:
        model = mmm.model
        model.add_coord(self.prefix, self.names)
        pmd.Data(
            f"{self.prefix}_data", np.full(len(self.names), 0.10), dims=self.prefix
        )

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
        return [(0.05, 0.45)] * len(self.names)


class _WindowedLeverEffect(OptimizableMuEffect):
    """Test-only lever: per-event boost active only inside its date window."""

    prefix: str = "promo"
    events: dict[str, tuple[str, str]] = {
        "early_event": ("2023-01-01", "2023-01-21"),
        "late_event": ("2023-03-01", "2023-04-01"),
    }

    def _window(self, dates) -> np.ndarray:
        dates = pd.DatetimeIndex(dates)
        return np.column_stack(
            [
                ((dates >= start) & (dates <= end)).astype(float)
                for start, end in self.events.values()
            ]
        )

    def create_data(self, mmm) -> None:
        model = mmm.model
        model.add_coord(self.prefix, list(self.events))
        pmd.Data(
            f"{self.prefix}_window",
            self._window(model.coords["date"]),
            dims=("date", self.prefix),
        )
        pmd.Data(
            f"{self.prefix}_data", np.full(len(self.events), 0.10), dims=self.prefix
        )

    def create_effect(self, mmm):
        model = mmm.model
        data = model[f"{self.prefix}_data"]
        window = model[f"{self.prefix}_window"]
        coef = pmd.HalfNormal(f"{self.prefix}_coef", sigma=1.0, dims=self.prefix)
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution",
            (window * data * coef).sum(dim=self.prefix),
            dims="date",
        )

    def set_data(self, mmm, model, X) -> None:
        pm.set_data(
            {f"{self.prefix}_window": self._window(model.coords["date"])},
            model=model,
        )

    @property
    def lever_bounds(self):
        return [(0.02, 0.45)] * len(self.events)


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

    result = optimizer.allocate_budget(total_budget=100.0)

    assert result.scipy_result.success
    # The promo lever stays out of the sum constraint -- only the media
    # entries (the first two) sum to total_budget.
    assert np.isclose(result.scipy_result.x[:2].sum(), 100.0)
    # Media allocation comes back over the budget dims and sums to total_budget.
    assert np.isclose(float(result.budgets.sum()), 100.0)

    # The effect's optimal lever is decoded off the tail of result.scipy_result.x into a
    # DataArray over the effect's own dim/coords.
    promo_opt = result.optimized_vars["promo_data"]
    assert promo_opt.dims == ("promo",)
    assert list(promo_opt.coords["promo"].values) == ["evt1", "evt2"]
    np.testing.assert_allclose(promo_opt.values, result.scipy_result.x[2:])
    # The positive-coefficient contribution gives the objective a positive
    # gradient in promo_data, so the lever climbs to its upper bound.
    np.testing.assert_allclose(promo_opt.values, 1.0, atol=1e-6)

    # result.scipy_result.fun is in original objective units (the internal |f(x0)|
    # normalization is undone before returning): re-evaluating the raw
    # compiled objective at the solution must match.
    raw_obj, _ = optimizer._objective_and_grad(result.scipy_result.x.copy())
    np.testing.assert_allclose(
        float(result.scipy_result.fun), float(raw_obj), rtol=1e-10
    )

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
            # A names-only user supplies their own objective node --
            # total_response_original_scale is only registered for
            # OptimizableMuEffect models.
            pmd.Deterministic(f"{self.prefix}_objective", contribution.sum())
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
        response_variable="promo_objective",
    )
    result = optimizer.allocate_budget(total_budget=100.0)

    assert result.scipy_result.success
    assert np.isclose(float(result.budgets.sum()), 100.0)
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
    result = optimizer.allocate_budget(total_budget=100.0)
    assert result.scipy_result.success
    assert result.optimized_vars == {}


def test_direct_budget_optimizer_wrapper_infers_levers(mock_pymc_sample):
    """BudgetOptimizer(model=<wrapper>) infers effect levers, duck-typed.

    The legacy direct-construction path must not silently freeze the effect levers:
    _handle_legacy_model_arg pulls optimizable_vars off the wrapper
    via _effect_optimizable_vars (no marketing imports).
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
    effect = _PromoLeverEffect()
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
            model=mmm, start_date=date_range[2], end_date=date_range[12]
        )
    optimizer = BudgetOptimizer(
        model=wrapper,
        num_periods=wrapper.num_periods,
        response_variable="total_response_original_scale",
    )
    assert optimizer.optimizable_vars == {"promo_data": [(0.05, 0.45)]}
    assert [v.name for v in optimizer._variables.variables[1:]] == ["promo_data"]

    # An explicit opt-out on the direct path is respected, too.
    optimizer_off = BudgetOptimizer(
        model=wrapper, num_periods=wrapper.num_periods, optimizable_vars={}
    )
    assert optimizer_off._variables.variables[1:] == []


def test_optimizable_vars_empty_dict_opts_out(mock_pymc_sample):
    """Explicit optimizable_vars={} disables lever auto-injection.

    Re-planning media with the effect levers held fixed must be possible:
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
    effect = _PromoLeverEffect()
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
        optimizable_vars={},  # opt out: media-only, levers held fixed
    )
    result = optimizer.allocate_budget(total_budget=100.0)
    assert result.scipy_result.success
    assert np.isclose(float(result.budgets.sum()), 100.0)
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
    effect = _WindowedLeverEffect()
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
        result = optimizer.allocate_budget(total_budget=100.0)

    depths = result.optimized_vars["promo_data"]
    # The inert lever is returned at its current model value (the historical
    # depth), not at a bound.
    np.testing.assert_allclose(float(depths.sel(promo="early_event")), 0.10)
    # The in-window lever is genuinely optimized.
    assert float(depths.sel(promo="late_event")) > 0.10


def test_mixed_objective_stationary_warm_start_does_not_warn(mock_pymc_sample):
    """A lever at its optimum does not warn, on an objective with media AND lever.

    A quadratic effect (data - 0.5)^2 has an exactly-zero gradient at its
    optimum 0.5 in floating point. Warm-started there, the first evaluation
    flags it as suspicious; the perturbed second evaluation must clear it --
    no "not optimized" warning for a lever that is in fact perfectly
    optimized. The objective is the total response, which contains both the
    media contribution and the lever's, so neither block is invisible to the
    gradient guards (the media-blind variant of this test was the round-5
    counterexample: an unreachable media block silently returned its seed).
    """

    class QuadraticEffect(OptimizableMuEffect):
        prefix: str = "quad"

        @property
        def lever_bounds(self):
            return [(0.0, 1.0)]

        def create_data(self, mmm) -> None:
            model = mmm.model
            model.add_coord(self.prefix, ["k1"])
            pmd.Data(f"{self.prefix}_data", np.array([0.5]), dims=self.prefix)

        def create_effect(self, mmm):
            model = mmm.model
            data = model[f"{self.prefix}_data"]
            coef = pmd.HalfNormal(f"{self.prefix}_coef", sigma=1.0, dims=self.prefix)
            # Gradient wrt data is exactly 0.0 at data == 0.5, nonzero elsewhere.
            contribution = pmd.Deterministic(
                f"{self.prefix}_effect_contribution",
                -((data - 0.5) ** 2) * coef,
                dims=self.prefix,
            )
            pmd.Deterministic(f"{self.prefix}_objective", contribution.sum())
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
    ).add_mu_effect(QuadraticEffect())
    mmm.fit(X, y, random_seed=0)

    optimizer = mmm.budget_optimizer(
        start_date=date_range[-1] + pd.Timedelta(weeks=1),
        end_date=date_range[-1] + pd.Timedelta(weeks=4),
        response_variable="total_response_original_scale",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Silence the unrelated no-bounds default warning.
        warnings.filterwarnings("ignore", message="No budget bounds provided")
        result = optimizer.allocate_budget(total_budget=100.0)
    # The warm start already sat at the optimum and stays there.
    np.testing.assert_allclose(
        float(result.optimized_vars["quad_data"].values[0]), 0.5, atol=1e-6
    )
    # And the media block genuinely took part: the allocation satisfies the
    # sum constraint and moved off the uniform seed (regression: an
    # objective-only rescale once froze media at x0 while the levers moved).
    np.testing.assert_allclose(float(result.budgets.sum()), 100.0, rtol=1e-6)
    assert not np.allclose(result.scipy_result.x[:2], [50.0, 50.0], atol=1e-6)
    assert result.scipy_result.success


def test_levers_against_media_objective_warn_under_log(mock_pymc_sample):
    """Under log the media objective reaches the levers: warn, don't raise.

    total_media_contribution_original_scale is exp(mu) - exp(mu - mu_media)
    under the log link, and mu contains the lever contribution -- so the
    ancestry check passes and the levers would silently be tuned to maximize
    incremental media contribution. The optimizer warns instead.
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
    effect = _PromoLeverEffect()
    mmm = MMM(
        date_column="date",
        channel_columns=["ch1", "ch2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        link="log",
    ).add_mu_effect(effect)
    mmm.fit(X, y, random_seed=1)

    with pytest.warns(UserWarning, match="media-contribution objective"):
        mmm.budget_optimizer(
            start_date=date_range[3],
            end_date=date_range[10],
            # default response_variable: total_media_contribution_original_scale
        )


def test_optimize_budget_wires_effect_levers(mock_pymc_sample):
    """BudgetOptimizerWrapper.optimize_budget passes effect levers through.

    A model with an optimizable effect driven through the legacy
    ``optimize_budget`` API must co-optimize the effect lever, not silently
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

    effect = _PromoLeverEffect()
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

    result = wrapper.optimize_budget(
        budget=100.0,
        response_variable="total_response_original_scale",
    )
    assert result.scipy_result.success
    depths = result.optimized_vars["promo_data"]
    assert list(depths.coords["promo"].values) == ["spring_sale"]
    depth = float(depths.sel(promo="spring_sale"))
    assert 0.05 - 1e-8 <= depth <= 0.45 + 1e-8


@pytest.mark.parametrize(
    "x0_value, low, high, expected",
    [
        # Two finite bounds: the furthest one, which is how a dead zone is escaped.
        (50.0, 0.0, 100.0, 100.0),
        (90.0, 0.0, 100.0, 0.0),
        (7.0, 7.0, 7.0, None),
        (2.0, 0.0, None, 0.0),
        (2.0, None, 5.0, 5.0),
        (4.0, None, None, 8.0),
    ],
    ids=["interior", "near_upper", "pinned", "lower_only", "upper_only", "unbounded"],
)
def test_probe_coordinate_far_reaches_the_bounds(x0_value, low, high, expected):
    """The far probe leaves a dead zone the local step cannot escape."""
    probe = BudgetOptimizer._probe_coordinate(x0_value, low, high, far=True)
    if expected is None:
        assert probe is None
        return
    np.testing.assert_allclose(probe, expected, rtol=1e-9)
    if low is not None:
        assert probe >= low
    if high is not None:
        assert probe <= high


@pytest.mark.parametrize(
    "x0_value, low, high",
    [
        (1e6, 1e6 - 1.0, 1e6),
        (1e6, 1e6 - 5.0, 1e6),
        (1e-12, 0.0, 100.0),
    ],
    ids=["narrow_box_width_1", "narrow_box_width_5", "tiny_x0_wide_box"],
)
def test_probe_coordinate_never_gives_up_on_a_free_coordinate(x0_value, low, high):
    """A non-degenerate box always yields a distinct, in-bounds probe.

    Returning None here would be silently damning: the second gradient
    evaluation would land on x0 itself and the inert verdict would be
    automatic. Million-scale budgets with tight per-channel bounds are the
    normal case, not an exotic one.
    """
    for far in (False, True):
        probe = BudgetOptimizer._probe_coordinate(x0_value, low, high, far=far)
        assert probe is not None
        assert probe != x0_value
        assert low <= probe <= high


@pytest.mark.parametrize(
    "x0_value, low, high, expected",
    [
        # Two finite bounds: a small step up, local to x0 (not the box midpoint).
        (50.0, 0.0, 100.0, 50.05),
        # Pinned coordinate: no probe exists inside the feasible set.
        (7.0, 7.0, 7.0, None),
        # At the lower bound, the step must go up (down would leave the box).
        (0.0, 0.0, 100.0, 0.001),
        # At the upper bound, the step must go down.
        (100.0, 0.0, 100.0, 99.9),
        # Lower bound only.
        (2.0, 0.0, None, 2.002),
        # Upper bound only.
        (2.0, None, 5.0, 2.002),
        # Unbounded.
        (4.0, None, None, 4.004),
        # x0 at zero and unbounded: step falls back to unit scale.
        (0.0, None, None, 0.001),
    ],
    ids=[
        "interior",
        "pinned",
        "at_lower",
        "at_upper",
        "lower_only",
        "upper_only",
        "unbounded",
        "zero_unbounded",
    ],
)
def test_probe_coordinate_stays_local_and_in_bounds(x0_value, low, high, expected):
    """The probe perturbs relative to x0 and never leaves the box.

    A jump to the box midpoint asks the wrong question: for a saturating
    response already flat at x0, a distant point is flatter still, so an
    "inert" verdict gets confirmed by construction.
    """
    probe = BudgetOptimizer._probe_coordinate(x0_value, low, high)
    if expected is None:
        assert probe is None
        return
    np.testing.assert_allclose(probe, expected, rtol=1e-9)
    assert probe != x0_value
    if low is not None:
        assert probe >= low
    if high is not None:
        assert probe <= high


def test_dead_zone_objective_does_not_warn(mmm_wrapper):
    """A flat-here-but-responsive-further-out objective is not called inert.

    ``utility_function`` is user-supplied, so a utility counting only the
    response above a threshold creates a genuine dead zone: the gradient at x0
    is exactly zero and a purely local probe cannot escape it. The far probe
    can, so the variable must not be reported as inert.
    """
    channels = list(mmm_wrapper.channel_columns)

    def mean_response(samples, budgets):
        return samples.mean(dim="sample")

    # Measure the response at the warm start and far out, then put the
    # threshold between them so the dead zone is real but escapable.
    reference = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
        utility_function=mean_response,
    )

    def response_at(values):
        flat = reference._budgets_flat.type.filter(np.asarray(values, dtype=float))
        return -float(reference._objective_and_grad(flat)[0])

    near = response_at([1.0, 1.0])
    far = response_at([100.0, 100.0])
    assert far > near, "fixture response is not increasing in spend"
    threshold = 0.5 * (near + far)

    def threshold_utility(samples, budgets):
        return ptx.math.maximum(samples.mean(dim="sample") - threshold, 0.0)

    optimizer = BudgetOptimizer(
        model=mmm_wrapper,
        num_periods=30,
        response_variable="total_media_contribution_original_scale",
        utility_function=threshold_utility,
    )

    # The premise: the gradient really is exactly zero at the warm start.
    x0 = optimizer._budgets_flat.type.filter(np.array([1.0, 1.0]))
    _, g0 = optimizer._objective_and_grad(x0)
    assert np.all(np.asarray(g0) == 0.0), "fixture no longer has a dead zone at x0"

    bounds = optimizer_xarray_builder(
        np.array([[0.0, 100.0], [0.0, 100.0]]),
        channel=channels,
        bound=["lower", "upper"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message="No budget bounds provided")
        optimizer.allocate_budget(total_budget=2.0, budget_bounds=bounds)


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
    with pytest.raises(ValidationError, match="missing coordinates present in the"):
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
