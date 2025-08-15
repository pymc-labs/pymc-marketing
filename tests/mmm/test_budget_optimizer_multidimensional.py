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
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import (
    MMM,
    MultiDimensionalBudgetOptimizerWrapper,
)


@pytest.fixture(scope="module")
def dummy_df():
    n = 10
    # Data is not needed for optimization of this model
    df = pd.DataFrame(
        data={
            "date_week": pd.date_range(start=pd.Timestamp.today(), periods=n, freq="W"),
            "channel_1": np.linspace(0, 1, num=n),
            "channel_2": np.linspace(0, 1, num=n),
            # Dim
            "geo": np.random.choice(["A", "B"], size=n),
            "event_1": np.concatenate([np.zeros(n - 1), [1]]),
            "event_2": np.concatenate([[1], np.zeros(n - 1)]),
            "t": range(n),
        }
    )

    y = pd.Series(np.ones(n), name="y")

    df_kwargs = {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "dims": ("geo",),
        "control_columns": ["event_1", "event_2", "t"],
        "target_column": "y",
    }

    return df_kwargs, df, y


@pytest.fixture(scope="module")
def fitted_mmm(dummy_df):
    """Create and fit a model once for all tests."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )

    mmm.build_model(X=X_dummy, y=y_dummy)

    # Fit the model once
    mmm.fit(
        X=X_dummy,
        y=y_dummy,
        chains=2,
        target_accept=0.8,
        tune=50,
        draws=50,
        progressbar=False,  # Disable progress bar for cleaner test output
    )

    # Sample posterior predictive
    mmm.sample_posterior_predictive(
        X=X_dummy,
        extend_idata=True,
        combined=True,
        progressbar=False,
    )

    return mmm


def test_budget_optimizer_no_mask(dummy_df, fitted_mmm):
    df_kwargs, X_dummy, y_dummy = dummy_df

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=10),
    )

    optimal_budgets, result = optimizable_model.optimize_budget(
        budget=1,
        budgets_to_optimize=None,  # No mask provided
    )

    assert isinstance(optimal_budgets, xr.DataArray)
    assert optimal_budgets.shape == (2, 2)  # 2 channels, 2 geos
    assert result.success


def test_budget_optimizer_correct_mask(dummy_df, fitted_mmm):
    df_kwargs, X_dummy, y_dummy = dummy_df

    budgets_to_optimize = xr.DataArray(
        np.array([[True, False], [True, True]]),
        dims=["channel", "geo"],
        coords={
            "channel": ["channel_1", "channel_2"],
            "geo": ["A", "B"],
        },
    )

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=10),
    )

    optimal_budgets, result = optimizable_model.optimize_budget(
        budget=1,
        budgets_to_optimize=budgets_to_optimize,
    )

    assert isinstance(optimal_budgets, xr.DataArray)
    assert optimal_budgets.shape == (2, 2)  # 2 channels, 2 geos
    assert result.success


def test_budget_optimizer_incorrect_mask(dummy_df, fitted_mmm):
    df_kwargs, X_dummy, y_dummy = dummy_df

    # Simulate a case where the model has no information for one channel-geo combination
    # by creating a new model with modified data
    X_modified = X_dummy.copy()
    X_modified.loc[X_modified["geo"] == "A", "channel_2"] = 0.0

    # Create a new model with modified data (channel_2 in geo A has no spend)
    mmm_modified = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )
    mmm_modified.build_model(X=X_modified, y=y_dummy)
    mmm_modified.fit(
        X=X_modified,
        y=y_dummy,
        chains=2,
        target_accept=0.8,
        tune=50,
        draws=50,
        progressbar=False,
    )
    mmm_modified.sample_posterior_predictive(
        X=X_modified,
        extend_idata=True,
        combined=True,
        progressbar=False,
    )

    # Create a mask that tries to optimize all channels including the zero one
    budgets_to_optimize = xr.DataArray(
        np.array([[True, True], [True, True]]),
        dims=["channel", "geo"],
        coords={
            "channel": ["channel_1", "channel_2"],
            "geo": ["A", "B"],
        },
    )

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=mmm_modified,
        start_date=X_modified["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_modified["date_week"].max() + pd.Timedelta(weeks=10),
    )

    msg = (
        "budgets_to_optimize mask contains True values at coordinates where the model has no "
        "information."
    )
    with pytest.raises(ValueError, match=msg):
        optimizable_model.optimize_budget(
            budget=1,
            budgets_to_optimize=budgets_to_optimize,
        )


def test_time_distribution_by_geo_only(dummy_df, fitted_mmm):
    """Test time distribution factors that vary by geo only (same for all channels in a geo).

    Note: Even though the factors only vary by geo, we must specify all budget dimensions
    (channel and geo). The BudgetOptimizer validates that budget_distribution_over_period
    has dims ("date", *budget_dims) where budget_dims includes all dimensions from channel_data
    except "date".
    """
    df_kwargs, X_dummy, y_dummy = dummy_df

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=4),  # 4 weeks
    )

    # First, let's try with only geo dimension to demonstrate it fails
    time_factors_geo_only = xr.DataArray(
        np.array(
            [
                [0.7, 0.1],  # date 0: geo A front-loaded, geo B back-loaded
                [0.2, 0.2],  # date 1
                [0.1, 0.3],  # date 2
                [0.0, 0.4],  # date 3
            ]
        ),
        dims=["date", "geo"],
        coords={
            "date": [0, 1, 2, 3],
            "geo": ["A", "B"],
        },
    )

    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

    # This should raise ValueError because we need all budget dimensions
    with pytest.raises(
        ValueError,
        match="budget_distribution_over_period must have dims.*but got",
    ):
        BudgetOptimizer(
            model=optimizable_model,
            num_periods=4,
            budget_distribution_over_period=time_factors_geo_only,
            response_variable="total_media_contribution_original_scale",
            default_constraints=True,
        )

    # Now create the correct format with all dimensions (even though channels have same values)
    # Create time distribution factors that vary by geo only
    # Geo A: front-loaded, Geo B: back-loaded
    time_factors_data = np.array(
        [
            # date 0
            [
                [0.7, 0.7],  # geo A: channel_1, channel_2 (same values)
                [0.1, 0.1],  # geo B: channel_1, channel_2 (same values)
            ],
            # date 1
            [
                [0.2, 0.2],  # geo A: channel_1, channel_2
                [0.2, 0.2],  # geo B: channel_1, channel_2
            ],
            # date 2
            [
                [0.1, 0.1],  # geo A: channel_1, channel_2
                [0.3, 0.3],  # geo B: channel_1, channel_2
            ],
            # date 3
            [
                [0.0, 0.0],  # geo A: channel_1, channel_2
                [0.4, 0.4],  # geo B: channel_1, channel_2
            ],
        ]
    )

    budget_distribution_over_period = xr.DataArray(
        time_factors_data,
        dims=["date", "geo", "channel"],
        coords={
            "date": [0, 1, 2, 3],
            "geo": ["A", "B"],
            "channel": ["channel_1", "channel_2"],
        },
    )

    # Create the budget optimizer directly with time factors
    optimizer = BudgetOptimizer(
        model=optimizable_model,
        num_periods=4,
        budget_distribution_over_period=budget_distribution_over_period,
        response_variable="total_media_contribution_original_scale",
        # No custom utility function needed with fitted model!
        default_constraints=True,
    )

    optimal_budgets, result = optimizer.allocate_budget(
        total_budget=100,
    )

    assert isinstance(optimal_budgets, xr.DataArray)
    assert optimal_budgets.shape == (2, 2)  # 2 channels, 2 geos
    assert result.success
    assert np.abs(optimal_budgets.sum().item() - 100) < 1e-6


def test_time_distribution_by_channel_geo(dummy_df, fitted_mmm):
    """Test time distribution factors that vary by both channel and geo."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=4),  # 4 weeks
    )

    # Create time distribution factors that vary by both channel and geo
    # Each channel-geo combination has a unique pattern
    budget_distribution_over_period = xr.DataArray(
        np.array(
            [
                # date 0
                [
                    [0.7, 0.0],  # geo A: channel_1 front-loaded, channel_2 back-loaded
                    [0.25, 0.4],
                ],  # geo B: channel_1 uniform, channel_2 decreasing
                # date 1
                [
                    [0.2, 0.1],  # geo A
                    [0.25, 0.3],
                ],  # geo B
                # date 2
                [
                    [0.1, 0.3],  # geo A
                    [0.25, 0.2],
                ],  # geo B
                # date 3
                [
                    [0.0, 0.6],  # geo A
                    [0.25, 0.1],
                ],  # geo B
            ]
        ),
        dims=["date", "geo", "channel"],
        coords={
            "date": [0, 1, 2, 3],
            "geo": ["A", "B"],
            "channel": ["channel_1", "channel_2"],
        },
    )

    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

    # Create the budget optimizer directly with time factors
    optimizer = BudgetOptimizer(
        model=optimizable_model,
        num_periods=4,
        budget_distribution_over_period=budget_distribution_over_period,
        response_variable="total_media_contribution_original_scale",
        default_constraints=True,
    )

    optimal_budgets, result = optimizer.allocate_budget(
        total_budget=100,
    )

    assert isinstance(optimal_budgets, xr.DataArray)
    assert optimal_budgets.shape == (2, 2)  # 2 channels, 2 geos
    assert result.success
    assert np.abs(optimal_budgets.sum().item() - 100) < 1e-6


def test_time_distribution_with_zero_bounds(dummy_df, fitted_mmm):
    """Test time distribution with some channels having zero budget bounds."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=4),  # 4 weeks
    )

    # Create time distribution factors for all channel-geo combinations
    budget_distribution_over_period = xr.DataArray(
        np.array(
            [
                # date 0
                [
                    [0.7, 0.1],  # geo A: channel_1, channel_2
                    [0.25, 0.4],
                ],  # geo B: channel_1, channel_2
                # date 1
                [
                    [0.2, 0.2],  # geo A
                    [0.25, 0.3],
                ],  # geo B
                # date 2
                [
                    [0.1, 0.3],  # geo A
                    [0.25, 0.2],
                ],  # geo B
                # date 3
                [
                    [0.0, 0.4],  # geo A
                    [0.25, 0.1],
                ],  # geo B
            ]
        ),
        dims=["date", "geo", "channel"],
        coords={
            "date": [0, 1, 2, 3],
            "geo": ["A", "B"],
            "channel": ["channel_1", "channel_2"],
        },
    )

    # Set budget bounds: channel_2 in geo A gets zero budget
    budget_bounds = xr.DataArray(
        np.array(
            [
                # channel_1: normal bounds for both geos
                [[[0, 50], [0, 50]]],
                # channel_2: zero for geo A, normal for geo B
                [[[0, 0], [0, 50]]],
            ]
        ).squeeze(),
        dims=["channel", "geo", "bound"],
        coords={
            "channel": ["channel_1", "channel_2"],
            "geo": ["A", "B"],
            "bound": ["lower", "upper"],
        },
    )

    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

    # Create the budget optimizer with time factors
    optimizer = BudgetOptimizer(
        model=optimizable_model,
        num_periods=4,
        budget_distribution_over_period=budget_distribution_over_period,
        response_variable="total_media_contribution_original_scale",
        default_constraints=True,
    )

    optimal_budgets, result = optimizer.allocate_budget(
        total_budget=50,
        budget_bounds=budget_bounds,
    )

    assert isinstance(optimal_budgets, xr.DataArray)
    assert optimal_budgets.shape == (2, 2)  # 2 channels, 2 geos
    assert result.success

    # Check that channel_2 in geo A has zero budget
    assert optimal_budgets.sel(channel="channel_2", geo="A").item() == 0.0

    # Check total budget constraint
    assert np.abs(optimal_budgets.sum().item() - 50) < 1e-6


def test_budget_distribution_over_period_wrong_dims_multidimensional(
    dummy_df, fitted_mmm
):
    """Test that time distribution factors with wrong dimensions raise error in multidimensional case."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=4),
    )

    # Create time factors with missing geo dimension
    budget_distribution_over_period = xr.DataArray(
        [
            [0.25, 0.25],  # date 0
            [0.25, 0.25],  # date 1
            [0.25, 0.25],  # date 2
            [0.25, 0.25],
        ],  # date 3
        coords={
            "date": [0, 1, 2, 3],
            "channel": ["channel_1", "channel_2"],
        },
        dims=["date", "channel"],
    )

    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

    with pytest.raises(
        ValueError, match="budget_distribution_over_period must have dims"
    ):
        BudgetOptimizer(
            model=optimizable_model,
            num_periods=4,
            budget_distribution_over_period=budget_distribution_over_period,
            response_variable="total_media_contribution_original_scale",
            default_constraints=True,
        )


def test_time_distribution_multidim(dummy_df, fitted_mmm):
    """Test time distribution factors with fitted model."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=4),  # 4 weeks
    )

    # Create time distribution factors that vary by geo only
    budget_distribution_over_period = xr.DataArray(
        np.array(
            [
                # date 0
                [
                    [0.7, 0.7],  # geo A: both channels front-loaded
                    [0.1, 0.1],
                ],  # geo B: both channels back-loaded
                # date 1
                [
                    [0.2, 0.2],  # geo A
                    [0.2, 0.2],
                ],  # geo B
                # date 2
                [
                    [0.1, 0.1],  # geo A
                    [0.3, 0.3],
                ],  # geo B
                # date 3
                [
                    [0.0, 0.0],  # geo A
                    [0.4, 0.4],
                ],  # geo B
            ]
        ),
        dims=["date", "geo", "channel"],
        coords={
            "date": [0, 1, 2, 3],
            "geo": ["A", "B"],
            "channel": ["channel_1", "channel_2"],
        },
    )

    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

    # Create the budget optimizer with time factors
    with pytest.warns(UserWarning, match="Using default equality constraint"):
        optimizer = BudgetOptimizer(
            model=optimizable_model,
            num_periods=4,
            budget_distribution_over_period=budget_distribution_over_period,
            response_variable="total_media_contribution_original_scale",
            default_constraints=True,
        )

    optimal_budgets, result = optimizer.allocate_budget(
        total_budget=100,
    )

    assert isinstance(optimal_budgets, xr.DataArray)
    assert optimal_budgets.shape == (2, 2)  # 2 channels, 2 geos
    assert result.success
    assert np.abs(optimal_budgets.sum().item() - 100) < 1e-6


def test_time_distribution_channel_specific_pattern(dummy_df, fitted_mmm):
    """Test channel-specific time distribution patterns."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=4),
    )

    # Different patterns for each channel-geo combination
    budget_distribution_over_period = xr.DataArray(
        np.array(
            [
                # date 0
                [
                    [1.0, 0.0],  # geo A: channel_1 all front, channel_2 all back
                    [0.6, 0.0],
                ],  # geo B: channel_1 mostly front, channel_2 mostly back
                # date 1
                [
                    [0.0, 0.0],  # geo A
                    [0.3, 0.1],
                ],  # geo B
                # date 2
                [
                    [0.0, 0.0],  # geo A
                    [0.1, 0.3],
                ],  # geo B
                # date 3
                [
                    [0.0, 1.0],  # geo A
                    [0.0, 0.6],
                ],  # geo B
            ]
        ),
        dims=["date", "geo", "channel"],
        coords={
            "date": [0, 1, 2, 3],
            "geo": ["A", "B"],
            "channel": ["channel_1", "channel_2"],
        },
    )

    # Set specific budget bounds
    budget_bounds = xr.DataArray(
        np.array(
            [
                # channel_1: higher bounds
                [[[0, 80], [0, 80]]],
                # channel_2: lower bounds
                [[[0, 20], [0, 20]]],
            ]
        ).squeeze(),
        dims=["channel", "geo", "bound"],
        coords={
            "channel": ["channel_1", "channel_2"],
            "geo": ["A", "B"],
            "bound": ["lower", "upper"],
        },
    )

    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

    with pytest.warns(UserWarning, match="Using default equality constraint"):
        optimizer = BudgetOptimizer(
            model=optimizable_model,
            num_periods=4,
            budget_distribution_over_period=budget_distribution_over_period,
            response_variable="total_media_contribution_original_scale",
            default_constraints=True,
        )

    optimal_budgets, result = optimizer.allocate_budget(
        total_budget=80,
        budget_bounds=budget_bounds,
    )

    assert isinstance(optimal_budgets, xr.DataArray)
    assert optimal_budgets.shape == (2, 2)
    assert result.success

    # Check bounds are respected
    assert (optimal_budgets.sel(channel="channel_1") <= 80).all()
    assert (optimal_budgets.sel(channel="channel_2") <= 20).all()

    # Check total budget
    assert np.abs(optimal_budgets.sum().item() - 80) < 1e-6


def test_time_distribution_validation_multidim(dummy_df, fitted_mmm):
    """Test validation of time distribution factors in multidimensional case."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=4),
    )

    # Test 1: Factors don't sum to 1
    bad_factors = xr.DataArray(
        np.array(
            [
                # date 0
                [
                    [0.5, 0.25],  # geo A: channel_1 bad, channel_2 ok
                    [0.25, 0.5],
                ],  # geo B: channel_1 ok, channel_2 bad
                # date 1
                [[0.5, 0.25], [0.25, 0.5]],
                # date 2
                [[0.5, 0.25], [0.25, 0.5]],
                # date 3
                [[0.5, 0.25], [0.25, 0.5]],
            ]
        ),
        dims=["date", "geo", "channel"],
        coords={
            "date": [0, 1, 2, 3],
            "geo": ["A", "B"],
            "channel": ["channel_1", "channel_2"],
        },
    )

    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

    with pytest.raises(
        ValueError, match="budget_distribution_over_period must sum to 1"
    ):
        BudgetOptimizer(
            model=optimizable_model,
            num_periods=4,
            budget_distribution_over_period=bad_factors,
            response_variable="total_media_contribution_original_scale",
            default_constraints=True,
        )

    # Test 2: Wrong number of periods
    wrong_periods_factors = xr.DataArray(
        np.array(
            [
                # only 2 dates
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]],
            ]
        ),
        dims=["date", "geo", "channel"],
        coords={
            "date": [0, 1],
            "geo": ["A", "B"],
            "channel": ["channel_1", "channel_2"],
        },
    )

    with pytest.raises(
        ValueError,
        match="budget_distribution_over_period date dimension must have length 4",
    ):
        BudgetOptimizer(
            model=optimizable_model,
            num_periods=4,
            budget_distribution_over_period=wrong_periods_factors,
            response_variable="total_media_contribution_original_scale",
            default_constraints=True,
        )


def test_time_distribution_total_spend_preserved(dummy_df, fitted_mmm):
    """Test that total spend is the same with and without time distribution patterns."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    # Set up common parameters
    num_periods = 4
    total_budget = 100

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=num_periods),
    )

    # Run optimization WITHOUT time distribution pattern
    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

    with pytest.warns(UserWarning, match="Using default equality constraint"):
        optimizer_no_pattern = BudgetOptimizer(
            model=optimizable_model,
            num_periods=num_periods,
            budget_distribution_over_period=None,  # No pattern
            response_variable="total_media_contribution_original_scale",
            default_constraints=True,
        )

    optimal_budgets_no_pattern, result_no_pattern = (
        optimizer_no_pattern.allocate_budget(
            total_budget=total_budget,
        )
    )

    # Run optimization WITH time distribution pattern
    # Create a flighting pattern (e.g., 60% first period, 30% second, 10% third, 0% fourth)
    budget_distribution_over_period = xr.DataArray(
        np.array(
            [
                # date 0
                [[0.6, 0.6], [0.6, 0.6]],  # All channels/geos: 60%
                # date 1
                [[0.3, 0.3], [0.3, 0.3]],  # All channels/geos: 30%
                # date 2
                [[0.1, 0.1], [0.1, 0.1]],  # All channels/geos: 10%
                # date 3
                [[0.0, 0.0], [0.0, 0.0]],  # All channels/geos: 0%
            ]
        ),
        dims=["date", "geo", "channel"],
        coords={
            "date": [0, 1, 2, 3],
            "geo": ["A", "B"],
            "channel": ["channel_1", "channel_2"],
        },
    )

    with pytest.warns(UserWarning, match="Using default equality constraint"):
        optimizer_with_pattern = BudgetOptimizer(
            model=optimizable_model,
            num_periods=num_periods,
            budget_distribution_over_period=budget_distribution_over_period,
            response_variable="total_media_contribution_original_scale",
            default_constraints=True,
        )

    optimal_budgets_with_pattern, result_with_pattern = (
        optimizer_with_pattern.allocate_budget(
            total_budget=total_budget,
        )
    )

    # Both optimizations should succeed
    assert result_no_pattern.success
    assert result_with_pattern.success

    # Sample response distributions for both allocations
    response_no_pattern = optimizable_model.sample_response_distribution(
        allocation_strategy=optimal_budgets_no_pattern,
        include_carryover=False,  # Don't zero out dates for budget comparison
    )

    response_with_pattern = optimizable_model.sample_response_distribution(
        allocation_strategy=optimal_budgets_with_pattern,
        budget_distribution_over_period=budget_distribution_over_period,
        include_carryover=False,  # Don't zero out dates for budget comparison
    )

    # Print the flat pattern on response distributions
    print("\nResponse without pattern:")
    for channel in fitted_mmm.channel_columns:
        print(f"{channel} distribution by date:")
        print(response_no_pattern[channel].values)

    print("\nResponse with pattern:")
    for channel in fitted_mmm.channel_columns:
        print(f"{channel} distribution by date:")
        print(response_with_pattern[channel].values)

    # Extract channel spend from both response distributions
    # Access channel columns directly from the sampled allocation
    channel_columns = fitted_mmm.channel_columns

    # Calculate total spend for each pattern
    total_spend_no_pattern = 0
    total_spend_with_pattern = 0

    for channel in channel_columns:
        # Sum across all dimensions (date, geo)
        total_spend_no_pattern += response_no_pattern[channel].sum().item()
        total_spend_with_pattern += response_with_pattern[channel].sum().item()

    # Check that total spend is the same (within tolerance)
    # The tolerance accounts for numerical precision differences
    assert np.abs(total_spend_no_pattern - total_spend_with_pattern) < 0.1, (
        f"Total spend differs between patterns: "
        f"no pattern={total_spend_no_pattern}, "
        f"with pattern={total_spend_with_pattern}"
    )

    # Also verify that the optimal budgets sum to the total budget
    assert np.abs(optimal_budgets_no_pattern.sum().item() - total_budget) < 1e-6
    assert np.abs(optimal_budgets_with_pattern.sum().item() - total_budget) < 1e-6


def test_time_distribution_with_carryover_total_spend_preserved(dummy_df, fitted_mmm):
    """Test that total spend is preserved when using both carryover and time distribution patterns."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    # Set up common parameters
    num_periods = 4
    total_budget = 100

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=num_periods),
    )

    # Create a flighting pattern (e.g., 60% first period, 30% second, 10% third, 0% fourth)
    budget_distribution_over_period = xr.DataArray(
        np.array(
            [
                # date 0
                [[0.6, 0.6], [0.6, 0.6]],  # All channels/geos: 60%
                # date 1
                [[0.3, 0.3], [0.3, 0.3]],  # All channels/geos: 30%
                # date 2
                [[0.1, 0.1], [0.1, 0.1]],  # All channels/geos: 10%
                # date 3
                [[0.0, 0.0], [0.0, 0.0]],  # All channels/geos: 0%
            ]
        ),
        dims=["date", "geo", "channel"],
        coords={
            "date": [0, 1, 2, 3],
            "geo": ["A", "B"],
            "channel": ["channel_1", "channel_2"],
        },
    )

    # Create an optimal budget allocation
    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

    with pytest.warns(UserWarning, match="Using default equality constraint"):
        optimizer = BudgetOptimizer(
            model=optimizable_model,
            num_periods=num_periods,
            budget_distribution_over_period=budget_distribution_over_period,
            response_variable="total_media_contribution_original_scale",
            default_constraints=True,
        )

    optimal_budgets, result = optimizer.allocate_budget(
        total_budget=total_budget,
    )

    assert result.success

    # Test without carryover
    response_no_carryover = optimizable_model.sample_response_distribution(
        allocation_strategy=optimal_budgets,
        budget_distribution_over_period=budget_distribution_over_period,
        include_carryover=False,
    )

    # Test WITH carryover
    response_with_carryover = optimizable_model.sample_response_distribution(
        allocation_strategy=optimal_budgets,
        budget_distribution_over_period=budget_distribution_over_period,
        include_carryover=True,
    )

    # Extract channel spend
    channel_1_allocation = optimal_budgets.sel(channel="channel_1").sum().item()
    channel_1_spend_no_carryover = response_no_carryover["channel_1"].sum().item()
    channel_1_spend_with_carryover = response_with_carryover["channel_1"].sum().item()

    # Both scenarios should preserve total spend = allocation * num_periods
    assert (
        np.abs(channel_1_allocation * num_periods - channel_1_spend_no_carryover) < 0.1
    ), "Without carryover: spend should be allocation * num_periods"

    assert (
        np.abs(channel_1_allocation * num_periods - channel_1_spend_with_carryover)
        < 0.1
    ), "With carryover: spend should still be allocation * num_periods"


def test_budget_distribution_carryover_interaction_issue(dummy_df, fitted_mmm):
    """Test that budget distribution and carryover interaction preserves total spend correctly."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    # Set up a simple scenario
    num_periods = 4

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=num_periods),
    )

    # Create a simple allocation strategy - allocate 10 per channel per geo
    # Total allocation = 10 * 2 channels * 2 geos = 40
    allocation_strategy = xr.DataArray(
        np.full((2, 2), 10.0),
        dims=["channel", "geo"],
        coords={
            "channel": ["channel_1", "channel_2"],
            "geo": ["A", "B"],
        },
    )

    # Create a simple uniform budget distribution
    budget_distribution = xr.DataArray(
        np.full((num_periods, 2, 2), 1.0 / num_periods),
        dims=["date", "geo", "channel"],
        coords={
            "date": list(range(num_periods)),
            "geo": ["A", "B"],
            "channel": ["channel_1", "channel_2"],
        },
    )

    # Test without carryover
    response_no_carryover = optimizable_model.sample_response_distribution(
        allocation_strategy=allocation_strategy,
        budget_distribution_over_period=budget_distribution,
        include_carryover=False,
        noise_level=0.0,  # No noise to see exact values
    )

    # Test with carryover
    response_with_carryover = optimizable_model.sample_response_distribution(
        allocation_strategy=allocation_strategy,
        budget_distribution_over_period=budget_distribution,
        include_carryover=True,
        noise_level=0.0,  # No noise to see exact values
    )

    # Extract channel 1 spend
    channel_1_allocation = (
        allocation_strategy.sel(channel="channel_1").sum().item()
    )  # Should be 20
    channel_1_spend_no_carryover = response_no_carryover["channel_1"].sum().item()
    channel_1_spend_with_carryover = response_with_carryover["channel_1"].sum().item()

    # The key invariant: total spend should always equal allocation * num_periods
    # regardless of whether carryover is included or not
    assert (
        np.abs(channel_1_spend_no_carryover - channel_1_allocation * num_periods) < 0.1
    ), "Without carryover: total spend should equal allocation * num_periods"

    assert (
        np.abs(channel_1_spend_with_carryover - channel_1_allocation * num_periods)
        < 0.1
    ), "With carryover: total spend should still equal allocation * num_periods"


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
def test_multidimensional_optimize_budget_callback_parametrized(
    dummy_df, fitted_mmm, callback
):
    """Test callback functionality through MultiDimensionalBudgetOptimizerWrapper.optimize_budget interface."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=10),
    )

    # Test the MultiDimensionalBudgetOptimizerWrapper interface
    result = optimizable_model.optimize_budget(
        budget=100,
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
    assert optimal_budgets.dims == (
        "geo",
        "channel",
    )  # Multidimensional has geo dimension
    assert len(optimal_budgets.coords["channel"]) == len(fitted_mmm.channel_columns)

    # Budget should sum to total (within tolerance)
    assert np.abs(optimal_budgets.sum().item() - 100) < 1e-6

    # Check optimization result
    assert hasattr(opt_result, "success")
    assert hasattr(opt_result, "x")
    assert hasattr(opt_result, "fun")
