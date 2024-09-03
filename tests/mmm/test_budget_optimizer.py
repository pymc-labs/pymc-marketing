#   Copyright 2024 The PyMC Labs Developers
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

import numpy as np
import pytest

from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer, MinimizeException
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation


@pytest.mark.parametrize(
    argnames="total_budget, budget_bounds, parameters, minimize_kwargs, expected_optimal, expected_response",
    argvalues=[
        (
            100,
            {"channel_1": (0, 50), "channel_2": (0, 50)},
            {
                "channel_1": {
                    "adstock_params": {"alpha": 0.5},
                    "saturation_params": {"lam": 10, "beta": 0.5},
                },
                "channel_2": {
                    "adstock_params": {"alpha": 0.7},
                    "saturation_params": {"lam": 20, "beta": 1.0},
                },
            },
            None,
            {"channel_1": 50.0, "channel_2": 50.0},
            49.5,
        ),
        (
            100,
            {"channel_1": (0, 50), "channel_2": (0, 50)},
            {
                "channel_1": {
                    "adstock_params": {"alpha": 0.5},
                    "saturation_params": {"lam": 10, "beta": 0.5},
                },
                "channel_2": {
                    "adstock_params": {"alpha": 0.7},
                    "saturation_params": {"lam": 20, "beta": 1.0},
                },
            },
            {
                "method": "SLSQP",
                "options": {"ftol": 1e-8, "maxiter": 1_002},
            },
            {"channel_1": 50.0, "channel_2": 50.0},
            49.5,
        ),
        # Add more test cases if needed
    ],
    ids=["default_minimizer_kwargs", "custom_minimizer_kwargs"],
)
def test_allocate_budget(
    total_budget,
    budget_bounds,
    parameters,
    minimize_kwargs,
    expected_optimal,
    expected_response,
):
    # Initialize Adstock and Saturation Transformations
    adstock = GeometricAdstock(l_max=4)
    saturation = LogisticSaturation()

    # Create BudgetOptimizer Instance
    optimizer = BudgetOptimizer(
        adstock=adstock,
        saturation=saturation,
        num_periods=30,
        parameters=parameters,
        adstock_first=True,
        scales=np.array([1, 1]),
    )

    # Allocate Budget
    match = "Using default equality constraint"
    with pytest.warns(UserWarning, match=match):
        optimal_budgets, total_response = optimizer.allocate_budget(
            total_budget=total_budget,
            budget_bounds=budget_bounds,
            minimize_kwargs=minimize_kwargs,
        )

    # Assert Results
    assert optimal_budgets == expected_optimal
    assert total_response == pytest.approx(expected_response, rel=1e-2)


@pytest.mark.parametrize(
    "total_budget, budget_bounds, parameters, expected_optimal, expected_response",
    [
        (
            0,
            {"channel_1": (0, 50), "channel_2": (0, 50)},
            {
                "channel_1": {
                    "adstock_params": {"alpha": 0.5},
                    "saturation_params": {"lam": 10, "beta": 0.5},
                },
                "channel_2": {
                    "adstock_params": {"alpha": 0.7},
                    "saturation_params": {"lam": 20, "beta": 1.0},
                },
            },
            {"channel_1": 0.0, "channel_2": 7.94e-13},
            2.38e-10,
        ),
    ],
)
def test_allocate_budget_zero_total(
    total_budget, budget_bounds, parameters, expected_optimal, expected_response
):
    adstock = GeometricAdstock(l_max=4)
    saturation = LogisticSaturation()

    optimizer = BudgetOptimizer(
        adstock=adstock,
        saturation=saturation,
        num_periods=30,
        parameters=parameters,
        adstock_first=True,
        scales=np.array([1, 1]),
    )
    match = "Using default equality constraint"
    with pytest.warns(UserWarning, match=match):
        optimal_budgets, total_response = optimizer.allocate_budget(
            total_budget, budget_bounds
        )
    assert optimal_budgets == pytest.approx(expected_optimal, rel=1e-2)
    assert total_response == pytest.approx(expected_response, abs=1e-1)


@patch("pymc_marketing.mmm.budget_optimizer.minimize")
def test_allocate_budget_custom_minimize_args(minimize_mock) -> None:
    total_budget = 100
    budget_bounds = {"channel_1": (0.0, 50.0), "channel_2": (0.0, 50.0)}
    parameters = {
        "channel_1": {
            "adstock_params": {"alpha": 0.5},
            "saturation_params": {"lam": 10, "beta": 0.5},
        },
        "channel_2": {
            "adstock_params": {"alpha": 0.7},
            "saturation_params": {"lam": 20, "beta": 1.0},
        },
    }
    minimize_kwargs = {
        "method": "SLSQP",
        "options": {"ftol": 1e-8, "maxiter": 1_002},
    }

    adstock = GeometricAdstock(l_max=4)
    saturation = LogisticSaturation()

    optimizer = optimizer = BudgetOptimizer(
        adstock=adstock,
        saturation=saturation,
        num_periods=30,
        parameters=parameters,
        adstock_first=True,
        scales=np.array([1, 1]),
    )
    match = "Using default equality constraint"
    with pytest.warns(UserWarning, match=match):
        optimizer.allocate_budget(
            total_budget, budget_bounds, minimize_kwargs=minimize_kwargs
        )

    kwargs = minimize_mock.call_args_list[0].kwargs

    np.testing.assert_array_equal(x=kwargs["x0"], y=np.array([50.0, 50.0]))
    assert kwargs["bounds"] == [(0.0, 50.0), (0.0, 50.0)]
    # default constraint constraints = {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}
    assert kwargs["constraints"]["type"] == "eq"
    assert (
        kwargs["constraints"]["fun"](np.array([total_budget / 2, total_budget / 2]))
        == 0.0
    )
    assert kwargs["constraints"]["fun"](np.array([100.0, 0.0])) == 0.0
    assert kwargs["constraints"]["fun"](np.array([0.0, 0.0])) == -total_budget
    assert kwargs["method"] == minimize_kwargs["method"]
    assert kwargs["options"] == minimize_kwargs["options"]


@pytest.mark.parametrize(
    "total_budget, budget_bounds, parameters, custom_constraints",
    [
        (
            100,
            {"channel_1": (0, 50), "channel_2": (0, 50)},
            {
                "channel_1": {
                    "adstock_params": {"alpha": 0.5},
                    "saturation_params": {"lam": 10, "beta": 0.5},
                },
                "channel_2": {
                    "adstock_params": {"alpha": 0.7},
                    "saturation_params": {"lam": 20, "beta": 1.0},
                },
            },
            {
                "type": "ineq",
                "fun": lambda x: x[0] - 60,
            },  # channel_1 must be >= 60, which is infeasible
        ),
    ],
)
def test_allocate_budget_infeasible_constraints(
    total_budget, budget_bounds, parameters, custom_constraints
):
    adstock = GeometricAdstock(l_max=4)
    saturation = LogisticSaturation()

    optimizer = optimizer = BudgetOptimizer(
        adstock=adstock,
        saturation=saturation,
        num_periods=30,
        parameters=parameters,
        adstock_first=True,
        scales=np.array([1, 1]),
    )

    with pytest.raises(MinimizeException, match="Optimization failed"):
        optimizer.allocate_budget(total_budget, budget_bounds, custom_constraints)
