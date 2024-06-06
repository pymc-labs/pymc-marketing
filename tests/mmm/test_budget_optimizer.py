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
import pytest

from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer
from pymc_marketing.mmm.components.adstock import _get_adstock_function
from pymc_marketing.mmm.components.saturation import _get_saturation_function


@pytest.mark.parametrize(
    "total_budget, budget_bounds, parameters, expected_optimal, expected_response",
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
            {"channel_1": 50.0, "channel_2": 50.0},
            49.5,
        ),
        # Add more test cases if needed
    ],
)
def test_allocate_budget(
    total_budget, budget_bounds, parameters, expected_optimal, expected_response
):
    # Initialize Adstock and Saturation Transformations
    adstock = _get_adstock_function(function="geometric", l_max=4)
    saturation = _get_saturation_function(function="logistic")

    # Create BudgetOptimizer Instance
    optimizer = BudgetOptimizer(adstock, saturation, 30, parameters, adstock_first=True)

    # Allocate Budget
    optimal_budgets, total_response = optimizer.allocate_budget(
        total_budget, budget_bounds
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
    adstock = _get_adstock_function(function="geometric", l_max=4)
    saturation = _get_saturation_function(function="logistic")
    optimizer = BudgetOptimizer(adstock, saturation, 30, parameters, adstock_first=True)
    optimal_budgets, total_response = optimizer.allocate_budget(
        total_budget, budget_bounds
    )
    assert optimal_budgets == pytest.approx(expected_optimal, rel=1e-2)
    assert total_response == pytest.approx(expected_response, rel=1e-2)


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
    adstock = _get_adstock_function(function="geometric", l_max=4)
    saturation = _get_saturation_function(function="logistic")
    optimizer = BudgetOptimizer(adstock, saturation, 30, parameters, adstock_first=True)

    with pytest.raises(Exception, match="Optimization failed"):
        optimizer.allocate_budget(total_budget, budget_bounds, custom_constraints)
