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
        start_date=X_dummy["date_week"].max() + pd.Timedelta(1, freq="1W"),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(2, freq="1W"),
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
        start_date=X_dummy["date_week"].max() + pd.Timedelta(1, freq="1W"),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(2, freq="1W"),
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
        start_date=X_modified["date_week"].max() + pd.Timedelta(1, freq="1W"),
        end_date=X_modified["date_week"].max() + pd.Timedelta(2, freq="1W"),
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
    """Test time distribution factors that vary by geo only (same for all channels in a geo)."""
    df_kwargs, X_dummy, y_dummy = dummy_df

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(1, freq="1W"),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(4, freq="1W"),  # 4 weeks
    )

    # Create time distribution factors that vary by geo only
    # Geo A: front-loaded, Geo B: back-loaded
    time_factors_data = np.array(
        [
            # date 0
            [
                [0.7, 0.7],  # geo A: channel_1, channel_2
                [0.1, 0.1],
            ],  # geo B: channel_1, channel_2
            # date 1
            [
                [0.2, 0.2],  # geo A: channel_1, channel_2
                [0.2, 0.2],
            ],  # geo B: channel_1, channel_2
            # date 2
            [
                [0.1, 0.1],  # geo A: channel_1, channel_2
                [0.3, 0.3],
            ],  # geo B: channel_1, channel_2
            # date 3
            [
                [0.0, 0.0],  # geo A: channel_1, channel_2
                [0.4, 0.4],
            ],  # geo B: channel_1, channel_2
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

    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

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
        start_date=X_dummy["date_week"].max() + pd.Timedelta(1, freq="1W"),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(4, freq="1W"),  # 4 weeks
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
        start_date=X_dummy["date_week"].max() + pd.Timedelta(1, freq="1W"),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(4, freq="1W"),  # 4 weeks
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
        start_date=X_dummy["date_week"].max() + pd.Timedelta(1, freq="1W"),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(4, freq="1W"),
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
        start_date=X_dummy["date_week"].max() + pd.Timedelta(1, freq="1W"),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(4, freq="1W"),  # 4 weeks
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
        start_date=X_dummy["date_week"].max() + pd.Timedelta(1, freq="1W"),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(4, freq="1W"),
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
        start_date=X_dummy["date_week"].max() + pd.Timedelta(1, freq="1W"),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(4, freq="1W"),
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
