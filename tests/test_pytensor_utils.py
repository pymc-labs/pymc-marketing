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

"""Tests for pytensor_utils module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytensor import function

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import (
    MMM,
    MultiDimensionalBudgetOptimizerWrapper,
)


@pytest.fixture
def sample_multidim_data():
    """Create sample data with country dimension for testing."""
    np.random.seed(42)
    n_obs_per_country = 12  # days of data per country
    countries = ["A", "B"]
    dates = pd.date_range(start="2023-01-01", periods=n_obs_per_country, freq="W-MON")

    # Create multi-indexed data with country and date
    data_list = []
    for country in countries:
        country_data = pd.DataFrame(
            {
                "date": dates,
                "country": country,
                "C1": np.random.randint(10, 50, n_obs_per_country),
                "C2": np.random.randint(5, 40, n_obs_per_country),
                "control": np.random.normal(10, 2, n_obs_per_country),
            }
        )

        # Create target with some relationship to channels
        country_data["y"] = (
            0.5 * country_data["C1"]
            + 0.3 * country_data["C2"]
            + 2 * country_data["control"]
            + (20 if country == "A" else 30)  # Different baseline per country
            + np.random.normal(0, 3, n_obs_per_country)
        )

        data_list.append(country_data)

    # Combine all country data
    data = pd.concat(data_list, ignore_index=True)
    return data


@pytest.fixture
def fitted_multidim_mmm(sample_multidim_data):
    """Create and fit a multidimensional MMM model."""
    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        control_columns=["control"],
        dims=("country",),
        target_column="y",
        adstock=GeometricAdstock(l_max=5),
        saturation=LogisticSaturation(),
    )

    X = sample_multidim_data.drop(columns=["y"])
    y = sample_multidim_data["y"]

    # Fit with minimal sampling for speed
    mmm.fit(X, y, draws=100, chains=2, random_seed=42, tune=100)

    return mmm


def test_extract_response_distribution_vs_sample_response(
    fitted_multidim_mmm, sample_multidim_data
):
    """Test that extract_response_distribution gives similar results to sample_response_distribution."""
    # Get date range from the data
    dates = sample_multidim_data["date"].unique()
    start_date = dates[-1] + pd.Timedelta(
        days=7
    )  # Start one week after last training date
    end_date = start_date + pd.Timedelta(weeks=8)  # 8 weeks of future data

    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print(f"Number of periods: {(end_date - start_date) // 7}")

    # Wrap the model in MultiDimensionalBudgetOptimizerWrapper
    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_multidim_mmm,
        start_date=start_date,
        end_date=end_date,
    )

    # Define allocation strategy: 2 units per channel per country
    allocation_values = np.array(
        [
            [2.0, 2.0],  # Country A: C1=2, C2=2
            [2.0, 2.0],  # Country B: C1=2, C2=2
        ]
    )

    allocation_strategy = xr.DataArray(
        allocation_values,
        dims=["country", "channel"],
        coords={
            "country": ["A", "B"],
            "channel": ["C1", "C2"],
        },
    )

    print(
        "\n=== Testing sample_response_distribution vs extract_response_distribution ==="
    )
    print(f"Allocation strategy:\n{allocation_strategy}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    # Create a BudgetOptimizer instance to mimic what happens internally
    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

    budget_optimizer = BudgetOptimizer(
        num_periods=optimizable_model.num_periods,
        model=optimizable_model,
        response_variable="total_media_contribution_original_scale",
    )
    # Compile channel data
    response_fun_inputs = budget_optimizer.extract_response_distribution("channel_data")

    input_fun = function([budget_optimizer._budgets_flat], response_fun_inputs)
    response_fun_inputs_values = input_fun(allocation_strategy.values.flatten())
    print(f"Shape of channel data: {response_fun_inputs_values.shape}")
    print("Dimension must match: (N_periods, N_country, N_channels)")
    assert response_fun_inputs_values.shape == (
        optimizable_model.num_periods + optimizable_model.adstock.l_max,
        len(optimizable_model.idata.posterior.coords["country"]),
        len(optimizable_model.idata.posterior.coords["channel"]),
    ), "Dimension mismatch"

    # Compile and evaluate to get the response values
    response_fun = budget_optimizer.extract_response_distribution(
        "total_media_contribution_original_scale"
    )

    eval_response_values = function([budget_optimizer._budgets_flat], response_fun)
    response_fun_values = eval_response_values(allocation_strategy.values.flatten())

    print("\n--- extract_response_distribution results ---")

    mean_from_extract = np.mean(response_fun_values)
    std_from_extract = np.std(response_fun_values)

    print(f"Mean total contribution: {mean_from_extract:.2f}")
    print(f"Std total contribution: {std_from_extract:.2f}")

    # Method 1: Use sample_response_distribution
    response_data = optimizable_model.sample_response_distribution(
        allocation_strategy=allocation_strategy,
        include_carryover=True,
        include_last_observations=False,
        noise_level=1e-17,
    )

    # Get the channels information
    data_values_for_model = xr.concat(
        [response_data[channel] for channel in optimizable_model.channel_columns],
        dim=pd.Index(optimizable_model.channel_columns, name="channel"),
    ).transpose(..., "channel")
    print(f"Shape of channel data: {data_values_for_model.shape}")
    print("Dimension must match: (N_periods, N_country, N_channels)")
    assert data_values_for_model.shape == (
        optimizable_model.num_periods + optimizable_model.adstock.l_max,
        len(optimizable_model.idata.posterior.coords["country"]),
        len(optimizable_model.idata.posterior.coords["channel"]),
    ), "Dimension mismatch"

    # Extract the total media contribution
    total_contribution_samples = response_data[
        "total_media_contribution_original_scale"
    ]

    print("\n--- sample_response_distribution results ---")
    print(f"Shape: {total_contribution_samples.shape}")
    print(f"Dims: {list(total_contribution_samples.dims)}")

    # Calculate mean and std from samples
    mean_from_samples = total_contribution_samples.mean().values
    std_from_samples = total_contribution_samples.std().values

    print(f"Mean total contribution: {mean_from_samples:.2f}")
    print(f"Std total contribution: {std_from_samples:.2f}")

    # Data inputs checks
    ## Calculate the diff between them per day.
    diff = response_fun_inputs_values - data_values_for_model.values
    print(f"Diff shape: {diff.shape}")
    ##assume dimension is (N_periods, 2, 2) pick the country 0 and channel 0
    diff_country_0_channel_0 = diff[:, 0, 0]
    print(f"Diff country 0 channel 0: {diff_country_0_channel_0}")

    # printing each day for channel 0 and country 0 on response_fun_inputs_values, and data_values_for_model
    print(f"Response fun inputs values: {response_fun_inputs_values[:, 0, 0]}")
    print(f"Data values for model: {data_values_for_model.values[:, 0, 0]}")

    print(f"Number of periods: {optimizable_model.num_periods}")
    print(f"Max adstock lag: {optimizable_model.adstock.l_max}")
    print(
        f"Number of periods + max adstock lag: {optimizable_model.num_periods + optimizable_model.adstock.l_max}"
    )

    ## Assert that response_fun_inputs_values[:, 0, 0] have length equal
    ## to optimizable_model.num_periods + optimizable_model.adstock.l_max

    ## Assert that data_values_for_model.values[:, 0, 0] have length equal
    ## to optimizable_model.num_periods + optimizable_model.adstock.l_max
    assert (
        len(response_fun_inputs_values[:, 0, 0])
        == optimizable_model.num_periods + optimizable_model.adstock.l_max
    ), "Response fun inputs values length mismatch"
    assert (
        len(data_values_for_model.values[:, 0, 0])
        == optimizable_model.num_periods + optimizable_model.adstock.l_max
    ), "Data values for model length mismatch"

    ## Assert that the count of values with zero its equal to optimizable_model.adstock.l_max
    assert (
        np.sum(response_fun_inputs_values[:, 0, 0] == 0)
        == optimizable_model.adstock.l_max
    ), "Posterior predictive dataset: Number of values with zero mismatch"
    assert (
        np.sum(data_values_for_model.values[:, 0, 0] == 0)
        == optimizable_model.adstock.l_max
    ), "Pytensor extraction function:Number of values with zero mismatch"

    ## Assert that both are the same
    assert np.allclose(response_fun_inputs_values, data_values_for_model.values), (
        "Data inputs are not the same across the two methods"
    )

    # Calculate expected number of periods correctly using pd.date_range
    expected_periods = len(pd.date_range(start=start_date, end=end_date, freq="W-MON"))
    assert optimizable_model.num_periods == expected_periods, (
        f"Number of periods mismatch: {optimizable_model.num_periods} != {expected_periods}"
    )

    # Compare the results
    print("\n--- Comparison ---")
    mean_diff = abs(mean_from_samples - mean_from_extract)
    std_diff = abs(std_from_samples - std_from_extract)

    print(f"Mean difference: {mean_diff:.4f}")
    print(f"Std difference: {std_diff:.4f}")

    # Calculate relative differences
    mean_rel_diff = (
        mean_diff / abs(mean_from_samples) * 100 if mean_from_samples != 0 else 0
    )
    std_rel_diff = std_diff / std_from_samples * 100 if std_from_samples != 0 else 0

    print(f"Mean relative difference: {mean_rel_diff:.2f}%")
    print(f"Std relative difference: {std_rel_diff:.2f}%")

    # Verify that both methods give similar results
    # Allow for larger tolerance since we're comparing with noisy data
    assert mean_rel_diff < 1.0, (
        f"Mean relative difference too large: {mean_rel_diff:.2f}%"
    )
    assert std_rel_diff < 1.0, f"Std relative difference too large: {std_rel_diff:.2f}%"

    # Additional checks
    assert np.all(np.isfinite(response_fun_values)), (
        "extract_response_distribution produced non-finite values"
    )
    assert np.all(np.isfinite(total_contribution_samples.values)), (
        "sample_response_distribution produced non-finite values"
    )

    # Both should have the same number of posterior samples
    expected_samples = 200  # 100 draws * 2 chains
    assert response_fun_values.shape[0] == expected_samples, (
        f"Expected {expected_samples} samples from extract_response_distribution"
    )
    assert total_contribution_samples.shape[0] == expected_samples, (
        "Sample count mismatch"
    )

    print("\n✓ Both methods produce consistent results!")
