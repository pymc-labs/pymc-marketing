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

from pymc_marketing.mmm.budget_optimizer import (
    calculate_expected_contribution,
    objective_distribution,
    optimize_budget_distribution,
)


# Testing Calculate Expected Contribution
@pytest.mark.parametrize(
    "method,parameters,budget,expected",
    [
        (
            "michaelis-menten",
            {"channel1": (10, 5), "channel2": (20, 10)},
            {"channel1": 5, "channel2": 10},
            {"channel1": 5.0, "channel2": 10.0, "total": 15.0},
        ),
        # Add more cases
    ],
)
def test_calculate_expected_contribution(method, parameters, budget, expected):
    assert calculate_expected_contribution(method, parameters, budget) == expected


# Testing invalid method for Calculate Expected Contribution
def test_calculate_expected_contribution_invalid_method():
    with pytest.raises(ValueError):
        calculate_expected_contribution(
            "invalid", {"channel1": (10, 5)}, {"channel1": 5}
        )


# Testing Objective Distribution with valid inputs
@pytest.mark.parametrize(
    "x,method,channels,parameters,expected",
    [
        (
            [5, 10],
            "michaelis-menten",
            ["channel1", "channel2"],
            {"channel1": (10, 5), "channel2": (20, 10)},
            -15.0,
        ),
        (
            [1, 2],
            "sigmoid",
            ["channel1", "channel2"],
            {"channel1": (1, 0.5), "channel2": (1, 0.5)},
            -0.707,
        ),
        # Add more cases
    ],
)
def test_objective_distribution(x, method, channels, parameters, expected):
    assert objective_distribution(x, method, channels, parameters) == pytest.approx(
        expected, 0.001
    )


# Testing Objective Distribution with invalid method
def test_objective_distribution_invalid_method():
    with pytest.raises(ValueError):
        objective_distribution(
            [5, 10],
            "invalid",
            ["channel1", "channel2"],
            {"channel1": (10, 5), "channel2": (20, 10)},
        )


# Testing optimize_budget_distribution with valid inputs
@pytest.mark.parametrize(
    "method,total_budget,budget_ranges,parameters,channels,expected_sum",
    [
        (
            "michaelis-menten",
            100,
            {"channel1": (0, 50), "channel2": (0, 50)},
            {"channel1": (10, 5), "channel2": (20, 10)},
            ["channel1", "channel2"],
            100,
        ),
        (
            "sigmoid",
            10,
            None,
            {"channel1": (1, 0.5), "channel2": (1, 0.5)},
            ["channel1", "channel2"],
            2.0,
        ),  # Updated this line
        # Add more cases
    ],
)
def test_optimize_budget_distribution_valid(
    method, total_budget, budget_ranges, parameters, channels, expected_sum
):
    result = optimize_budget_distribution(
        method, total_budget, budget_ranges, parameters, channels
    )
    # Check if sum of budgets equals expected_sum
    assert sum(result.values()) == pytest.approx(expected_sum, 0.001)
    # Check if budgets are within their ranges
    if budget_ranges:
        for channel in channels:
            assert (
                budget_ranges[channel][0]
                <= result[channel]
                <= budget_ranges[channel][1]
            )
