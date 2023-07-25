import pandas as pd
import pytest

from pymc_marketing.mmm.budget_optimizer import budget_allocator


@pytest.mark.parametrize(
    "allocation_mode, total_budget, channels, parameters, budget_ranges, expected",
    [
        (
            "growth",
            1000,
            ["channel1", "channel2"],
            {"channel1": 0.5, "channel2": 0.5},
            {"channel1": (0, 1000), "channel2": (0, 1000)},
            pd.DataFrame(
                {
                    "estimated_contribution": {"channel1": 250, "channel2": 250},
                    "optimal_budget": {"channel1": 500, "channel2": 500},
                }
            ),
        ),
        (
            "growth",
            2000,
            ["channel1", "channel2", "channel3"],
            {"channel1": 0.3, "channel2": 0.3, "channel3": 0.4},
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
def test_budget_allocator(
    allocation_mode, total_budget, channels, parameters, budget_ranges, expected
):
    result = budget_allocator(
        allocation_mode, total_budget, channels, parameters, budget_ranges
    )
    pd.testing.assert_frame_equal(result, expected)
