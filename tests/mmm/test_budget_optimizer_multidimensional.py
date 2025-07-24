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


def test_budget_optimizer_no_mask(dummy_df):
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )
    mmm.build_model(X=X_dummy, y=y_dummy)
    mmm.fit(
        X=X_dummy,
        y=y_dummy,
        chains=2,
        target_accept=0.8,
        tune=50,
        draws=50,
    )
    mmm.sample_posterior_predictive(
        X=X_dummy,
        extend_idata=True,
        combined=True,
    )

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=mmm,
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


def test_budget_optimizer_correct_mask(
    dummy_df,
):
    df_kwargs, X_dummy, y_dummy = dummy_df

    budgets_to_optimize = xr.DataArray(
        np.array([[True, False], [True, True]]),
        dims=["channel", "geo"],
        coords={
            "channel": ["channel_1", "channel_2"],
            "geo": ["A", "B"],
        },
    )

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )
    mmm.build_model(X=X_dummy, y=y_dummy)
    mmm.fit(
        X=X_dummy,
        y=y_dummy,
        chains=2,
        target_accept=0.8,
        tune=50,
        draws=50,
    )
    mmm.sample_posterior_predictive(
        X=X_dummy,
        extend_idata=True,
        combined=True,
    )

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=mmm,
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


def test_budget_optimizer_incorrect_mask(
    dummy_df,
):
    df_kwargs, X_dummy, y_dummy = dummy_df

    # Remove spend for one channel in one geo
    X_dummy.loc[X_dummy["geo"] == "A", "channel_2"] = 0.0

    budgets_to_optimize = xr.DataArray(
        np.array([[True, True], [True, True]]),
        dims=["channel", "geo"],
        coords={
            "channel": ["channel_1", "channel_2"],
            "geo": ["A", "B"],
        },
    )

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )
    mmm.build_model(X=X_dummy, y=y_dummy)
    mmm.fit(
        X=X_dummy,
        y=y_dummy,
        chains=2,
        target_accept=0.8,
        tune=50,
        draws=50,
    )
    mmm.sample_posterior_predictive(
        X=X_dummy,
        extend_idata=True,
        combined=True,
    )

    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(1, freq="1W"),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(2, freq="1W"),
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
