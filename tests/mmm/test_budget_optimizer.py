import pandas as pd
import pytest

from pymc_marketing.mmm.budget_optimizer import (
    calculate_expected_contribution,
    optimize_budget_distribution,
)


@pytest.mark.parametrize(
    "method, total_budget, channels, parameters, budget_ranges, expected",
    [
        (
            "sigmoid",
            1000,
            ["channel1", "channel2"],
            {"channel1": (0.5, 0.5), "channel2": (0.5, 0.5)},
            {"channel1": (0, 1000), "channel2": (0, 1000)},
            pd.DataFrame(
                {
                    "estimated_contribution": {"channel1": 250, "channel2": 250},
                    "optimal_budget": {"channel1": 500, "channel2": 500},
                }
            ),
        ),
        (
            "michaelis-menten",
            2000,
            ["channel1", "channel2", "channel3"],
            {"channel1": (0.3, 0.3), "channel2": (0.3, 0.3), "channel3": (0.4, 0.4)},
            {"channel1": (0, 1000), "channel2": (0, 1000), "channel3": (0, 1000)},
            pd.DataFrame(
                {
                    "estimated_contribution": {
                        "channel1": 300,
                        "channel2": 300,
                        "channel3": 400,
                    },
                    "optimal_budget": {
                        "channel1": 600,
                        "channel2": 600,
                        "channel3": 800,
                    },
                }
            ),
        ),
    ],
)
def test_optimize_budget_distribution(
    method, total_budget, channels, parameters, budget_ranges, expected
):
    result = optimize_budget_distribution(
        method, total_budget, budget_ranges, parameters, channels
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "method, parameters, optimal_budget, expected",
    [
        (
            "sigmoid",
            {"channel1": (0.5, 0.5), "channel2": (0.5, 0.5)},
            {"channel1": 500, "channel2": 500},
            {"channel1": 250, "channel2": 250},
        ),
        (
            "michaelis-menten",
            {"channel1": (0.3, 0.3), "channel2": (0.3, 0.3), "channel3": (0.4, 0.4)},
            {"channel1": 600, "channel2": 600, "channel3": 800},
            {"channel1": 300, "channel2": 300, "channel3": 400},
        ),
    ],
)
def test_calculate_expected_contribution(method, parameters, optimal_budget, expected):
    result = calculate_expected_contribution(method, parameters, optimal_budget)
    assert result == expected
