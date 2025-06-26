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
from collections.abc import Callable

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pydantic import ValidationError
from pymc.model_graph import fast_eval
from pytensor.tensor.basic import TensorVariable
from scipy.optimize import OptimizeResult

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.additive_effect import EventAdditiveEffect, LinearTrendEffect
from pymc_marketing.mmm.events import EventEffect, GaussianBasis
from pymc_marketing.mmm.lift_test import _swap_columns_and_last_index_level
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.multidimensional import (
    MMM,
    MultiDimensionalBudgetOptimizerWrapper,
)
from pymc_marketing.mmm.scaling import Scaling, VariableScaling
from pymc_marketing.prior import Prior


@pytest.fixture
def mmm():
    return MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        dims=("country",),
        target_column="y",
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )


@pytest.fixture
def df() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=3, freq="W-MON").rename("date")
    df = pd.DataFrame(
        {
            ("A", "C1"): [1, 2, 3],
            ("B", "C1"): [4, 5, 6],
            ("A", "C2"): [7, 8, 9],
            ("B", "C2"): [10, 11, 12],
        },
        index=dates,
    )
    df.columns.names = ["country", "channel"]

    y = pd.DataFrame(
        {
            ("A", "y"): [1, 2, 3],
            ("B", "y"): [4, 5, 6],
        },
        index=dates,
    )
    y.columns.names = ["country", "channel"]

    return pd.concat(
        [
            df.stack("country", future_stack=True),
            y.stack("country", future_stack=True),
        ],
        axis=1,
    ).reset_index()


@pytest.fixture
def fit_mmm(df, mmm, mock_pymc_sample):
    X = df.drop(columns=["y"])
    y = df["y"]

    mmm.fit(X, y)

    return mmm


def test_simple_fit(fit_mmm):
    assert isinstance(fit_mmm.posterior, xr.Dataset)
    assert isinstance(fit_mmm.idata.constant_data, xr.Dataset)


def test_sample_prior_predictive(mmm: MMM, df: pd.DataFrame):
    X = df.drop(columns=["y"])
    y = df["y"]
    mmm.sample_prior_predictive(X, y)

    assert isinstance(mmm.prior, xr.Dataset)
    assert isinstance(mmm.prior_predictive, xr.Dataset)


def test_save_load(fit_mmm: MMM):
    file = "test.nc"
    fit_mmm.save(file)

    loaded = MMM.load(file)
    assert isinstance(loaded, MMM)


@pytest.fixture
def single_dim_data():
    """
    Generate a simple single-dimension (no extra dims) synthetic dataset.

    - date: 2023-01-01 to 2023-01-14 (14 days)
    - 2 channels: 'channel_1', 'channel_2'
    - target = sum of channels + random noise
    """
    date_range = pd.date_range("2023-01-01", periods=14)
    np.random.seed(42)

    # Generate random channel data
    channel_1 = np.random.randint(100, 500, size=len(date_range))
    channel_2 = np.random.randint(100, 500, size=len(date_range))
    channel_3 = np.nan

    df = pd.DataFrame(
        {
            "date": date_range,
            "channel_1": channel_1,
            "channel_2": channel_2,
            "channel_3": channel_3,
        }
    )
    # Target is sum of channels with noise
    df["target"] = (
        df["channel_1"]
        + df["channel_2"]
        + np.random.randint(100, 300, size=len(date_range))
    )
    X = df[["date", "channel_1", "channel_2", "channel_3"]].copy()

    return X, df["target"].copy()


@pytest.fixture
def multi_dim_data():
    """
    Generate a multi-dimensional dataset (e.g., includes 'country' dimension).

    - date: 2023-01-01 to 2023-01-07 (7 days)
    - countries: ["Venezuela", "Colombia", "Chile"]
    - 2 channels: 'channel_1', 'channel_2'
    - target = sum of channels + random noise
    """
    date_range = pd.date_range("2023-01-01", periods=7)
    countries = ["Venezuela", "Colombia", "Chile"]
    np.random.seed(123)

    records = []
    for country in countries:
        for date in date_range:
            channel_1 = np.random.randint(100, 500)
            channel_2 = np.random.randint(100, 500)
            channel_3 = np.nan
            target = channel_1 + channel_2 + np.random.randint(50, 150)
            records.append((date, country, channel_1, channel_2, channel_3, target))

    df = pd.DataFrame(
        records,
        columns=["date", "country", "channel_1", "channel_2", "channel_3", "target"],
    )

    X = df[["date", "country", "channel_1", "channel_2", "channel_3"]].copy()

    return X, df["target"].copy()


@pytest.mark.parametrize(
    "fixture_name, dims",
    [
        pytest.param("single_dim_data", (), id="Marginal model"),
        pytest.param("multi_dim_data", ("country",), id="County model"),
    ],
)
@pytest.mark.parametrize(
    "time_varying_intercept, time_varying_media, yearly_seasonality",
    [
        pytest.param(False, False, None, id="no tvps or fourier"),
        pytest.param(False, False, 4, id="no tvps with fourier"),
        pytest.param(True, False, None, id="tvp intercept only, no fourier"),
        pytest.param(False, True, 4, id="tvp media only with fourier"),
        pytest.param(True, True, 4, id="tvps and fourier"),
    ],
)
def test_fit(
    request,
    fixture_name,
    time_varying_intercept,
    time_varying_media,
    yearly_seasonality,
    dims,
    mock_pymc_sample,
):
    """Test that building the model works with different configurations (single-dim)."""
    X, y = request.getfixturevalue(fixture_name)

    adstock = GeometricAdstock(l_max=2)
    saturation = LogisticSaturation()

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=dims,
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=yearly_seasonality,
        time_varying_intercept=time_varying_intercept,
        time_varying_media=time_varying_media,
    )

    seed = sum(map(ord, "Fitting the MMMM"))
    random_seed = np.random.default_rng(seed)

    idata = mmm.fit(X, y, random_seed=random_seed)

    def normalization(data):
        return data.div(data.max())

    def unstack(data, name):
        if not name:
            return data

        return data.unstack(name)

    actual = mmm.target_data_scaled.eval()
    expected = (
        mmm.xarray_dataset._target.to_series()
        .pipe(normalization)
        .pipe(unstack, name=None if not dims else dims[0])
        .values
    )

    np.testing.assert_allclose(actual, expected)

    # Assertions
    assert hasattr(mmm, "model"), "Model attribute should be set after build_model."
    assert isinstance(mmm.model, pm.Model), "mmm.model should be a PyMC Model instance."
    for dim in dims:
        assert dim in mmm.model.coords, (
            f"Extra dimension '{dim}' should be in model coords."
        )

    # Basic checks to confirm presence of key variables
    var_names = mmm.model.named_vars.keys()
    assert "channel_data" in var_names
    assert "target_data" in var_names
    if time_varying_intercept:
        assert "intercept_latent_process" in var_names
    if time_varying_media:
        assert "media_latent_process" in var_names
    if yearly_seasonality is not None:
        assert "fourier_contribution" in var_names

    assert isinstance(idata, az.InferenceData), (
        "fit should return an InferenceData object."
    )
    assert hasattr(mmm, "idata"), (
        "MMM instance should store the inference data as 'idata'."
    )

    # Check presence of posterior group
    assert hasattr(mmm.idata, "posterior"), (
        "InferenceData should have a posterior group."
    )

    for dim in dims:
        assert dim in mmm.idata.posterior.dims, (
            f"Extra dimension '{dim}' should be in posterior dims."
        )


def test_sample_posterior_predictive_new_data(single_dim_data, mock_pymc_sample):
    """
    Test that sampling from the posterior predictive with new/unseen data
    properly creates a 'posterior_predictive' group in the InferenceData.
    """
    X, y = single_dim_data
    X_train = X.iloc[:-5]
    X_new = X.iloc[-5:]

    y_train = y.iloc[:-5]
    _ = y.iloc[-5:]

    # Build a small model
    adstock = GeometricAdstock(l_max=2)
    saturation = LogisticSaturation()
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=adstock,
        saturation=saturation,
    )

    # Fit with a fixed seed for reproducibility
    mmm.build_model(X_train, y_train)
    mmm.fit(X_train, y_train, draws=200, tune=100, chains=1, random_seed=42)

    mmm.sample_posterior_predictive(X_train, extend_idata=True, random_seed=42)

    def no_null_values(ds):
        return ds.y.isnull().mean()

    np.testing.assert_allclose(no_null_values(mmm.idata.posterior_predictive), 0)

    # Sample posterior predictive on new data
    out_of_sample_idata = mmm.sample_posterior_predictive(
        X_new, extend_idata=False, random_seed=42
    )

    # Check that posterior_predictive group was added
    assert hasattr(mmm.idata, "posterior_predictive"), (
        "After calling sample_posterior_predictive with new data, "
        "there should be a 'posterior_predictive' group in the inference data."
    )

    np.testing.assert_allclose(no_null_values(out_of_sample_idata), 0)

    # Check the shape of that group. We expect the new date dimension to match X_new length
    # plus no addition if we didn't set include_last_observations (which is False by default).
    assert "date" in out_of_sample_idata.dims, (
        "posterior_predictive should have a 'date' dimension."
    )
    assert out_of_sample_idata.coords["date"].values.shape == X_new.date.values.shape, (
        "The 'date' dimension in posterior_predictive should match new data length."
    )


def test_sample_posterior_predictive_same_data(single_dim_data, mock_pymc_sample):
    """
    Test that when we pass the SAME data used for training to sample_posterior_predictive:
      1) It does NOT overwrite the 'posterior' group.
      2) The deterministic variable (e.g. 'channel_contribution') from the
         posterior predictive matches the same data in the posterior group.
    """
    X, y = single_dim_data
    X_train = X.iloc[:-5]
    _ = X.iloc[-5:]

    y_train = y.iloc[:-5]
    _ = y.iloc[-5:]

    # Build a small model
    adstock = GeometricAdstock(l_max=2)
    saturation = LogisticSaturation()

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=adstock,
        saturation=saturation,
    )

    # Fit with a fixed random seed
    mmm.build_model(X_train, y_train)
    mmm.fit(X_train, y_train, draws=200, tune=100, chains=1, random_seed=123)

    # Just to confirm: 'posterior' group should exist before posterior predictive
    assert hasattr(mmm.idata, "posterior"), (
        "We expect a 'posterior' group after fitting."
    )

    # Sample posterior predictive with the SAME data
    out_of_sample_idata = mmm.sample_posterior_predictive(
        X_train,
        # y_train,
        extend_idata=False,
        combined=False,
        random_seed=123,
        include_last_observations=False,
        var_names=["channel_contribution", "intercept_contribution"],
    )

    # 1) Check that the 'posterior' group still exists
    assert hasattr(mmm.idata, "posterior"), (
        "The existing 'posterior' group should not be overwritten "
        "by posterior predictive sampling."
    )

    # 2) Compare 'channel_contribution' in the posterior vs. posterior_predictive
    # They should have the same shape if the data is the same
    assert (
        mmm.idata.posterior.channel_contribution.mean(dim=["draw", "chain"]).shape
        == out_of_sample_idata.channel_contribution.mean(dim=["draw", "chain"]).shape
    ), (
        "Shapes of posterior and posterior_predictive 'channel_contribution' "
        "must match when using the same data."
    )

    # They should be equal (or very close) because it's the same deterministic
    # transformation for the same data and the same random draws.
    assert np.allclose(
        mmm.idata.posterior.channel_contribution.mean(dim=["draw", "chain"]),
        out_of_sample_idata.channel_contribution.mean(dim=["draw", "chain"]),
        atol=1e-4,
    ), (
        "When passing identical data for posterior predictive, "
        "'channel_contribution' should match exactly (or within floating tolerance) "
        "the values in the 'posterior' group."
    )


def test_sample_posterior_predictive_same_data_with_include_last_observations(
    single_dim_data, mock_pymc_sample
):
    """
    Test that using include_last_observations=True with training data (overlapping dates)
    raises a ValueError with a clear error message.
    """
    X, y = single_dim_data
    X_train = X.iloc[:-5]
    y_train = y.iloc[:-5]

    # Build and fit the model
    adstock = GeometricAdstock(l_max=2)
    saturation = LogisticSaturation()

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=adstock,
        saturation=saturation,
    )

    mmm.build_model(X_train, y_train)
    mmm.fit(X_train, y_train, draws=200, tune=100, chains=1, random_seed=123)

    # Try to use include_last_observations=True with the same training data
    # This should raise a ValueError
    with pytest.raises(
        ValueError,
        match="Cannot use include_last_observations=True when input dates overlap with training dates",
    ):
        mmm.sample_posterior_predictive(
            X_train,  # Same training data
            include_last_observations=True,  # This should trigger the error
            extend_idata=False,
            random_seed=123,
        )


def test_sample_posterior_predictive_partial_overlap_with_include_last_observations(
    single_dim_data, mock_pymc_sample
):
    """
    Test that even partial date overlap with include_last_observations=True raises ValueError.
    """
    X, y = single_dim_data
    X_train = X.iloc[:-5]
    y_train = y.iloc[:-5]

    # Build and fit the model
    adstock = GeometricAdstock(l_max=2)
    saturation = LogisticSaturation()

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=adstock,
        saturation=saturation,
    )

    mmm.build_model(X_train, y_train)
    mmm.fit(X_train, y_train, draws=200, tune=100, chains=1, random_seed=123)

    # Create data that partially overlaps with training data
    # Take the last 3 training dates + 3 new future dates
    overlap_data = X.iloc[-8:-2]  # This will include some training dates

    # This should raise a ValueError due to partial overlap
    with pytest.raises(
        ValueError,
        match="Cannot use include_last_observations=True when input dates overlap with training dates",
    ):
        mmm.sample_posterior_predictive(
            overlap_data,
            include_last_observations=True,
            extend_idata=False,
            random_seed=123,
        )


def test_sample_posterior_predictive_no_overlap_with_include_last_observations(
    single_dim_data, mock_pymc_sample
):
    """
    Test that include_last_observations=True works correctly when there's no date overlap.
    """
    X, y = single_dim_data
    X_train = X.iloc[:-5]
    X_new = X.iloc[-5:]  # Non-overlapping future dates
    y_train = y.iloc[:-5]

    # Build and fit the model
    adstock = GeometricAdstock(l_max=2)
    saturation = LogisticSaturation()

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=adstock,
        saturation=saturation,
    )

    mmm.build_model(X_train, y_train)
    mmm.fit(X_train, y_train, draws=200, tune=100, chains=1, random_seed=123)

    # This should work fine since dates don't overlap
    try:
        result = mmm.sample_posterior_predictive(
            X_new,  # Non-overlapping dates
            include_last_observations=True,  # Should work fine
            extend_idata=False,
            random_seed=123,
        )

        # Verify that the result includes the expected dates
        # (should be l_max training dates + new prediction dates, then sliced to remove l_max)
        expected_dates = X_new["date"].values
        np.testing.assert_array_equal(result.coords["date"].values, expected_dates)

    except ValueError as e:
        pytest.fail(f"Unexpected error when using non-overlapping dates: {e}")


@pytest.fixture
def df_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_date": ["2025-01-01", "2024-12-25"],
            "end_date": ["2025-01-02", "2024-12-31"],
            "name": ["New Years", "Christmas Holiday"],
        }
    ).assign(random_column="random_value", another_extra_column="extra_value")


@pytest.fixture
def mock_mmm():
    coords = {"date": pd.date_range("2023-01-01", periods=7)}
    model = pm.Model(coords=coords)

    class MMM:
        pass

    mmm = MMM()
    mmm.model = model
    mmm.dims = ()

    return mmm


@pytest.fixture
def create_event_effect() -> Callable[[str], EventEffect]:
    def create(
        prefix: str = "holiday",
        sigma_dims: str | None = None,
        effect_size: Prior | None = None,
    ):
        basis = GaussianBasis()
        return EventEffect(
            basis=basis,
            effect_size=Prior("Normal"),
            dims=(prefix,),
        )

    return create


@pytest.fixture
def event_effect(create_event_effect) -> EventEffect:
    return create_event_effect()


def test_create_effect_mu_effect(
    mock_mmm,
    df_events,
    event_effect,
) -> None:
    effect = EventAdditiveEffect(
        df_events=df_events,
        prefix="holiday",
        effect=event_effect,
    )

    with mock_mmm.model:
        effect.create_data(mock_mmm)

    assert mock_mmm.model.coords["holiday"] == ("New Years", "Christmas Holiday")

    for named_var in ["days", "holiday_start_diff", "holiday_end_diff"]:
        assert named_var in mock_mmm.model.named_vars

    with mock_mmm.model:
        mu = effect.create_effect(mock_mmm)

    assert isinstance(mu, TensorVariable)

    for named_vars in ["holiday_sigma", "holiday_effect_size", "holiday_total_effect"]:
        assert named_vars in mock_mmm.model.named_vars

    coords = {"date": pd.date_range("2023-01-07", periods=7)}
    with pm.Model(coords=coords) as new_model:
        pm.Data("days", np.arange(7), dims="date")
        effect.set_data(None, new_model, None)  # type: ignore

    np.testing.assert_allclose(
        fast_eval(new_model["days"]),
        np.array([-725, -724, -723, -722, -721, -720, -719]),
    )


def test_mmm_with_events(
    df_events,
    create_event_effect,
    mmm,
    df,
    mock_pymc_sample,
) -> None:
    mmm.add_events(
        df_events,
        prefix="holiday",
        effect=create_event_effect(prefix="holiday"),
    )
    assert len(mmm.mu_effects) == 1

    mmm.add_events(
        df_events,
        prefix="another_event_type",
        effect=create_event_effect(prefix="another_event_type"),
    )
    assert len(mmm.mu_effects) == 2

    X = df.drop(columns=["y"])
    y = df["y"]
    mmm.build_model(X, y)

    seed = sum(map(ord, "Adding events"))
    random_seed = np.random.default_rng(seed)

    mmm.fit(X, y, random_seed=random_seed)

    assert "holiday_total_effect" in mmm.posterior
    assert "another_event_type_total_effect" in mmm.posterior

    kwargs = dict(
        extend_idata=False,
        var_names=["holiday_total_effect", "another_event_type_total_effect"],
        random_seed=random_seed,
    )

    in_sample = mmm.sample_posterior_predictive(X, **kwargs)
    np.testing.assert_array_equal(
        in_sample.coords["date"].to_numpy(),
        X["date"].unique(),
    )

    X_new = X.copy()
    diff = (X_new["date"].max() - X_new["date"].min()).days + 7
    X_new["date"] += pd.Timedelta(days=diff)

    out_of_sample = mmm.sample_posterior_predictive(X_new, **kwargs)

    np.testing.assert_array_equal(
        out_of_sample.coords["date"].to_numpy(),
        X_new["date"].unique(),
    )

    less_effect_for_out_of_sample = np.abs(in_sample.sum()) > np.abs(
        out_of_sample.sum()
    )

    assert less_effect_for_out_of_sample.to_pandas().all()


@pytest.mark.parametrize(
    "adstock, saturation, dims",
    [
        pytest.param(
            GeometricAdstock(l_max=2).set_dims_for_all_priors("country"),
            LogisticSaturation(),
            (),
            id="adstock",
        ),
        pytest.param(
            GeometricAdstock(l_max=2),
            LogisticSaturation().set_dims_for_all_priors("country"),
            (),
            id="saturation",
        ),
        pytest.param(
            GeometricAdstock(l_max=2).set_dims_for_all_priors("media"),
            LogisticSaturation(),
            (),
            id="adstock-wrong-media",
        ),
        pytest.param(
            GeometricAdstock(l_max=2),
            LogisticSaturation().set_dims_for_all_priors("media"),
            (),
            id="saturation-wrong-media",
        ),
        pytest.param(
            GeometricAdstock(l_max=2),
            LogisticSaturation().set_dims_for_all_priors(("media", "product")),
            ("country",),
            id="wrong-extra-dim",
        ),
    ],
)
def test_check_for_incompatible_dims(adstock, saturation, dims) -> None:
    kwargs = dict(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
    )
    with pytest.raises(ValueError):
        MMM(
            adstock=adstock,
            saturation=saturation,
            dims=dims,
            **kwargs,  # type: ignore
        )


@pytest.mark.parametrize("method", ["mean", "max"])
def test_different_target_scaling(method, multi_dim_data, mock_pymc_sample) -> None:
    X, y = multi_dim_data
    scaling = {"target": {"method": method, "dims": ()}}
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        scaling=scaling,
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
    )
    assert mmm.scaling.target == VariableScaling(method=method, dims=())
    mmm.fit(X, y)
    assert mmm.xarray_dataset._target.dims == ("date", "country")
    assert mmm.scalers._target.dims == ("country",)

    def max_abs(df: pd.DataFrame) -> pd.DataFrame:
        return df.div(df.max())

    def mean(df: pd.DataFrame) -> pd.DataFrame:
        return df.div(df.mean())

    normalization = {"mean": mean, "max": max_abs}[method]

    actual = mmm.target_data_scaled.eval()
    expected = (
        mmm.xarray_dataset._target.to_series()
        .unstack("country")
        .pipe(normalization)
        .values
    )

    np.testing.assert_allclose(actual, expected)


def test_target_scaling_raises() -> None:
    scaling = {"target": {"method": "mean", "dims": ("country",)}}
    match = "Target scaling dims"
    with pytest.raises(ValueError, match=match):
        MMM(
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            scaling=scaling,
            date_column="date",
            target_column="target",
            channel_columns=["channel_1", "channel_2", "channel_3"],
        )


@pytest.mark.parametrize("dims", [(), ("country",)], ids=["country-level", "global"])
def test_target_scaling_and_contributions(
    multi_dim_data,
    dims,
    mock_pymc_sample,
) -> None:
    X, y = multi_dim_data

    scaling = {"target": {"method": "mean", "dims": dims}}
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        scaling=scaling,
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
    )

    var_names = ["channel_contribution", "intercept_contribution", "y"]
    mmm.build_model(X, y)
    mmm.add_original_scale_contribution_variable(var=var_names)

    for var in var_names:
        new_var_name = f"{var}_original_scale"
        assert new_var_name in mmm.model.named_vars

    try:
        mmm.fit(X, y)
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


@pytest.mark.parametrize(
    "dims, expected_dims",
    [
        ((), ("country", "channel")),
        (("country",), ("channel",)),
        (("channel",), ("country",)),
    ],
    ids=["country-channel", "country", "channel"],
)
def test_channel_scaling(multi_dim_data, dims, expected_dims, mock_pymc_sample) -> None:
    X, y = multi_dim_data

    scaling = {"channel": {"method": "mean", "dims": dims}}
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        scaling=scaling,
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
    )

    mmm.fit(X, y)

    assert mmm.scalers._channel.dims == expected_dims


def test_scaling_dict_doesnt_mutate() -> None:
    scaling = {}
    dims = ("country",)
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        scaling=scaling,
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=dims,
    )

    assert scaling == {}
    assert mmm.scaling == Scaling(
        target=VariableScaling(method="max", dims=dims),
        channel=VariableScaling(method="max", dims=dims),
    )


def test_multidimensional_budget_optimizer_wrapper(fit_mmm, mock_pymc_sample):
    """Test the MultiDimensionalBudgetOptimizerWrapper functionality."""
    start_date = "2025-01-01"
    end_date = "2025-01-31"

    # Create the wrapper
    optimizer = MultiDimensionalBudgetOptimizerWrapper(
        model=fit_mmm, start_date=start_date, end_date=end_date
    )

    # Test basic attributes
    assert hasattr(optimizer, "model_class")
    assert hasattr(optimizer, "zero_data")
    assert hasattr(optimizer, "num_periods")
    assert optimizer.model_class == fit_mmm

    # Test attribute delegation
    assert optimizer.date_column == fit_mmm.date_column
    assert optimizer.channel_columns == fit_mmm.channel_columns
    assert optimizer.dims == fit_mmm.dims

    # Create a budget bounds DataArray
    budget = 1000
    countries = fit_mmm.xarray_dataset.country.values
    channels = fit_mmm.channel_columns
    budget_bounds = xr.DataArray(
        np.array([[[0, budget]] * len(channels)] * len(countries)),
        coords=[countries, channels, ["low", "high"]],
        dims=["country", "channel", "bound"],
    )

    # Run a real optimization
    allocation_xarray, scipy_opt_result = optimizer.optimize_budget(
        budget=budget, budget_bounds=budget_bounds
    )

    # Check the results
    assert set(allocation_xarray.dims) == set(
        (*fit_mmm.dims, "channel")
    )  # Check dims excluding 'date'
    assert allocation_xarray.shape == (
        len(countries),
        len(channels),
    )  # Check shape based on dims
    assert isinstance(scipy_opt_result, OptimizeResult)


@pytest.fixture
def df_lift_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel": ["channel_1", "channel_1"],
            "country": ["Venezuela", "Colombia"],
            "x": [1, 2],
            "delta_x": [1, 1],
            "delta_y": [1, 1],
            "sigma": [1, 1],
            "date": pd.to_datetime(["2023-01-02", "2023-01-04"]),
        }
    )


@pytest.mark.parametrize(
    "saturation_dims",
    [
        pytest.param((), id="scalar"),
        pytest.param(("channel",), id="vector"),
        pytest.param(("country", "channel"), id="matrix"),
    ],
)
@pytest.mark.parametrize(
    "target_scaling_dims",
    [
        pytest.param(("country",), id="through-country (default)"),
        pytest.param((), id="by-country"),
    ],
)
@pytest.mark.parametrize(
    "channel_scaling_dims",
    [
        pytest.param(("country",), id="through-country (default)"),
        pytest.param((), id="by-country"),
    ],
)
def test_add_lift_test_measurements(
    multi_dim_data,
    df_lift_test,
    saturation_dims,
    target_scaling_dims,
    channel_scaling_dims,
    mock_pymc_sample,
) -> None:
    X, y = multi_dim_data
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation().set_dims_for_all_priors(dims=saturation_dims),
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        scaling=Scaling(
            channel=VariableScaling(method="max", dims=channel_scaling_dims),
            target=VariableScaling(method="max", dims=target_scaling_dims),
        ),
        dims=("country",),
    )
    mmm.build_model(X, y)

    name = "lift_measurements"
    assert name not in mmm.model

    mmm.add_lift_test_measurements(
        df_lift_test,
        name=name,
    )

    assert name in mmm.model

    try:
        mmm.fit(X, y)
    except Exception as e:
        pytest.fail(f"Sampling failed with error: {e}")


def test_add_lift_test_measurements_no_model() -> None:
    adstock = GeometricAdstock(l_max=4)
    saturation = LogisticSaturation()
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=adstock,
        saturation=saturation,
    )
    with pytest.raises(RuntimeError, match="The model has not been built yet."):
        mmm.add_lift_test_measurements(
            pd.DataFrame(),
        )


def test_time_varying_media_with_lift_test(
    multi_dim_data, df_lift_test, mock_pymc_sample
) -> None:
    X, y = multi_dim_data
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_media=True,
    )
    mmm.build_model(X=X, y=y)
    try:
        mmm.add_lift_test_measurements(df_lift_test)
    except Exception as e:
        pytest.fail(
            f"add_lift_test_measurements for time_varying_media model failed with error {e}"
        )

    try:
        mmm.fit(X, y)
    except Exception as e:
        pytest.fail(f"Sampling failed with error: {e}")


def test_add_lift_test_measurements_missing_channel_column(multi_dim_data) -> None:
    """Test that KeyError is raised when 'channel' column is missing from lift test data."""
    X, y = multi_dim_data
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
    )
    mmm.build_model(X, y)

    # Create lift test data without 'channel' column
    df_lift_test_missing_channel = pd.DataFrame(
        {
            "country": ["Venezuela", "Colombia"],
            "x": [1, 2],
            "delta_x": [1, 1],
            "delta_y": [1, 1],
            "sigma": [1, 1],
            "date": pd.to_datetime(["2023-01-02", "2023-01-04"]),
        }
    )

    with pytest.raises(
        KeyError,
        match="The 'channel' column is required to map the lift measurements to the model.",
    ):
        mmm.add_lift_test_measurements(df_lift_test_missing_channel)


def test_add_lift_test_measurements_missing_dimension_column(multi_dim_data) -> None:
    """Test that KeyError is raised when dimension column is missing from lift test data."""
    X, y = multi_dim_data
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
    )
    mmm.build_model(X, y)

    # Create lift test data without 'country' column (which is in dims)
    df_lift_test_missing_dim = pd.DataFrame(
        {
            "channel": ["channel_1", "channel_1"],
            "x": [1, 2],
            "delta_x": [1, 1],
            "delta_y": [1, 1],
            "sigma": [1, 1],
            "date": pd.to_datetime(["2023-01-02", "2023-01-04"]),
        }
    )

    with pytest.raises(
        KeyError,
        match="The country column is required to map the lift measurements to the model.",
    ):
        mmm.add_lift_test_measurements(df_lift_test_missing_dim)


def test_add_lift_test_measurements_missing_multiple_dimension_columns() -> None:
    """Test that KeyError is raised when multiple dimension columns are missing from lift test data."""
    # Create multi-dimensional data with multiple dimensions
    date_range = pd.date_range("2023-01-01", periods=7)
    countries = ["Venezuela", "Colombia"]
    products = ["Product_A", "Product_B"]

    records = []
    for country in countries:
        for product in products:
            for date in date_range:
                channel_1 = np.random.randint(100, 500)
                channel_2 = np.random.randint(100, 500)
                target = channel_1 + channel_2 + np.random.randint(50, 150)
                records.append((date, country, product, channel_1, channel_2, target))

    df = pd.DataFrame(
        records,
        columns=["date", "country", "product", "channel_1", "channel_2", "target"],
    )

    X = df[["date", "country", "product", "channel_1", "channel_2"]].copy()
    y = df["target"].copy()

    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        dims=("country", "product"),
    )
    mmm.build_model(X, y)

    # Create lift test data missing both dimension columns
    df_lift_test_missing_dims = pd.DataFrame(
        {
            "channel": ["channel_1", "channel_1"],
            "x": [1, 2],
            "delta_x": [1, 1],
            "delta_y": [1, 1],
            "sigma": [1, 1],
            "date": pd.to_datetime(["2023-01-02", "2023-01-04"]),
        }
    )

    # Should raise KeyError for the first missing dimension (country)
    with pytest.raises(
        KeyError,
        match="The country column is required to map the lift measurements to the model.",
    ):
        mmm.add_lift_test_measurements(df_lift_test_missing_dims)


def test_add_lift_test_measurements_missing_single_dimension_from_multiple() -> None:
    """Test that KeyError is raised when one dimension column is missing from multi-dimensional lift test data."""
    # Create multi-dimensional data with multiple dimensions
    date_range = pd.date_range("2023-01-01", periods=7)
    countries = ["Venezuela", "Colombia"]
    products = ["Product_A", "Product_B"]

    records = []
    for country in countries:
        for product in products:
            for date in date_range:
                channel_1 = np.random.randint(100, 500)
                channel_2 = np.random.randint(100, 500)
                target = channel_1 + channel_2 + np.random.randint(50, 150)
                records.append((date, country, product, channel_1, channel_2, target))

    df = pd.DataFrame(
        records,
        columns=["date", "country", "product", "channel_1", "channel_2", "target"],
    )

    X = df[["date", "country", "product", "channel_1", "channel_2"]].copy()
    y = df["target"].copy()

    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        dims=("country", "product"),
    )
    mmm.build_model(X, y)

    # Create lift test data missing only 'product' column
    df_lift_test_missing_product = pd.DataFrame(
        {
            "channel": ["channel_1", "channel_1"],
            "country": ["Venezuela", "Colombia"],
            "x": [1, 2],
            "delta_x": [1, 1],
            "delta_y": [1, 1],
            "sigma": [1, 1],
            "date": pd.to_datetime(["2023-01-02", "2023-01-04"]),
        }
    )

    # Should raise KeyError for the missing 'product' dimension
    with pytest.raises(
        KeyError,
        match="The product column is required to map the lift measurements to the model.",
    ):
        mmm.add_lift_test_measurements(df_lift_test_missing_product)


@pytest.mark.parametrize(
    "saturation_dims",
    [
        pytest.param((), id="scalar"),
        pytest.param(("channel",), id="vector"),
    ],
)
def test_add_lift_test_measurements_no_dimensions_success(
    saturation_dims,
    single_dim_data,
    mock_pymc_sample,
) -> None:
    """Test that lift test measurements work correctly when no dimensions are specified."""
    X, y = single_dim_data
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation().set_dims_for_all_priors(dims=saturation_dims),
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=(),  # No dimensions
    )
    mmm.build_model(X, y)

    # Create lift test data with only required columns (no dimension columns)
    df_lift_test_no_dims = pd.DataFrame(
        {
            "channel": ["channel_1", "channel_1"],
            "x": [1, 2],
            "delta_x": [1, 1],
            "delta_y": [1, 1],
            "sigma": [1, 1],
        }
    )

    # Should work without errors since no dimensions are required
    try:
        mmm.add_lift_test_measurements(df_lift_test_no_dims)
        assert "lift_measurements" in mmm.model
    except Exception as e:
        pytest.fail(
            f"add_lift_test_measurements should work with no dimensions, but failed with error: {e}"
        )

    try:
        mmm.fit(X, y)
    except Exception as e:
        pytest.fail(f"Sampling failed with error: {e}")


@pytest.fixture
def df_lift_test_for_transform() -> pd.DataFrame:
    """Create a lift test DataFrame for testing transform functions."""
    return pd.DataFrame(
        {
            "channel": ["channel_1", "channel_1", "channel_2"],
            "country": ["Venezuela", "Colombia", "Chile"],
            "x": np.random.randint(100, 500),
            "delta_x": np.random.randint(10, 50, size=3),
            "delta_y": np.random.randint(100, 500, size=3),
            "sigma": np.random.randint(10, 50, size=3),
        }
    )


@pytest.mark.parametrize(
    "scaling_method, scaling_dims",
    [
        pytest.param("max", ("country",), id="max-scaling-across-country"),
        pytest.param("max", (), id="max-scaling-by-country"),
        pytest.param("mean", ("country",), id="mean-scaling-across-country"),
        pytest.param("mean", (), id="mean-scaling-by-country"),
    ],
)
def test_make_channel_transform_multi_dim(
    multi_dim_data,
    df_lift_test_for_transform,
    scaling_method,
    scaling_dims,
) -> None:
    """Test _make_channel_transform function with multi-dimensional data."""
    X, y = multi_dim_data

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        scaling=Scaling(
            channel=VariableScaling(method=scaling_method, dims=scaling_dims),
            target=VariableScaling(method="max", dims=()),
        ),
    )

    mmm.build_model(X.drop(columns=["channel_3"]), y)

    # Get the channel transform function
    channel_transform = mmm._make_channel_transform(df_lift_test_for_transform)

    # Replicate the tranforms of mmm.lift_test.scale_channel_lift_measurements
    index_cols = [*list(mmm.dims), "channel"]
    df_original = df_lift_test_for_transform.loc[
        :, [*index_cols, "x", "delta_x"]
    ].set_index(index_cols, append=True)
    df_to_rescale = (
        df_original.pipe(_swap_columns_and_last_index_level)
        .reindex(mmm.channel_columns, axis=1)
        .fillna(0)
    )

    # Apply the transform
    result = channel_transform(df_to_rescale.values)

    # Verify the result has the expected shape
    assert result.shape == df_to_rescale.values.shape

    # Verify the scaling is applied correctly
    if scaling_method == "max":
        for idx, channel in enumerate(df_to_rescale.columns):
            if scaling_dims == ("country",):
                # Through-country scaling: max across all countries for each channel
                assert np.all(
                    result[:, idx] == df_to_rescale.values[:, idx] / X[channel].max()
                )
            else:
                # By-country scaling: max for each country-channel combination
                for row_idx, (_, country, _) in enumerate(df_to_rescale.index):
                    assert result[row_idx, idx] == (
                        df_to_rescale.values[row_idx, idx]
                        / X.loc[X["country"] == country, channel].max()
                    )
    else:
        for idx, channel in enumerate(df_to_rescale.columns):
            if scaling_dims == ("country",):
                # Through-country scaling: mean across all countries for each channel
                assert np.all(
                    result[:, idx] == df_to_rescale.values[:, idx] / X[channel].mean()
                )
            else:
                # By-country scaling: mean for each country-channel combination
                for row_idx, (_, country, _) in enumerate(df_to_rescale.index):
                    assert result[row_idx, idx] == (
                        df_to_rescale.values[row_idx, idx]
                        / X.loc[X["country"] == country, channel].mean()
                    )


@pytest.mark.parametrize(
    "scaling_method",
    ["max", "mean"],
    ids=["max-scaling", "mean-scaling"],
)
def test_make_channel_transform_single_dim(
    single_dim_data,
    df_lift_test_for_transform,
    scaling_method,
) -> None:
    """Test _make_channel_transform function with single-dimensional data."""
    X, y = single_dim_data

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        dims=(),  # No extra dimensions
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        scaling=Scaling(
            channel=VariableScaling(method=scaling_method, dims=()),
            target=VariableScaling(method="max", dims=()),
        ),
    )

    mmm.build_model(X.drop(columns=["channel_3"]), y)

    # Get the channel transform function
    channel_transform = mmm._make_channel_transform(
        df_lift_test_for_transform.drop(columns=["country"])
    )

    # Replicate the tranforms of mmm.lift_test.scale_channel_lift_measurements
    index_cols = [*list(mmm.dims), "channel"]
    df_original = df_lift_test_for_transform.loc[
        :, [*index_cols, "x", "delta_x"]
    ].set_index(index_cols, append=True)
    df_to_rescale = (
        df_original.pipe(_swap_columns_and_last_index_level)
        .reindex(mmm.channel_columns, axis=1)
        .fillna(0)
    )

    # Apply the transform
    result = channel_transform(df_to_rescale.values)

    # Verify the result has the expected shape
    assert result.shape == df_to_rescale.values.shape

    # Verify the scaling is applied correctly
    if scaling_method == "max":
        for idx, channel in enumerate(df_to_rescale.columns):
            assert np.all(
                result[:, idx] == df_to_rescale.values[:, idx] / X[channel].max()
            )

    else:  # mean scaling
        for idx, channel in enumerate(df_to_rescale.columns):
            assert np.all(
                result[:, idx] == df_to_rescale.values[:, idx] / X[channel].mean()
            )


@pytest.mark.parametrize(
    "scaling_method, scaling_dims",
    [
        pytest.param("max", ("country",), id="max-scaling-through-country"),
        pytest.param("max", (), id="max-scaling-by-country"),
        pytest.param("mean", ("country",), id="mean-scaling-through-country"),
        pytest.param("mean", (), id="mean-scaling-by-country"),
    ],
)
def test_make_target_transform_multi_dim(
    multi_dim_data,
    df_lift_test_for_transform,
    scaling_method,
    scaling_dims,
) -> None:
    """Test _make_target_transform function with multi-dimensional data."""
    X, y = multi_dim_data

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        scaling=Scaling(
            channel=VariableScaling(method="max", dims=()),
            target=VariableScaling(method=scaling_method, dims=scaling_dims),
        ),
    )

    mmm.build_model(X, y)

    # Get the target transform function
    target_transform = mmm._make_target_transform(df_lift_test_for_transform)

    # Test with sample input data (delta_y values)
    input_data = df_lift_test_for_transform[
        "delta_y"
    ].values  # delta_y values from lift test

    # Apply the transform
    result = target_transform(input_data).flatten()

    # Verify the result has the expected shape (2D array)
    assert result.shape == input_data.shape

    # Verify the scaling is applied correctly
    if scaling_method == "max":
        if scaling_dims == ("country",):
            # Through-country scaling: max across all countries
            assert np.all(result == input_data / y.max())
        else:
            # per country scaling
            for idx, country in enumerate(df_lift_test_for_transform["country"].values):
                assert (
                    result[idx]
                    == input_data[idx] / y.loc[X["country"] == country].max()
                )
    else:  # mean scaling
        if scaling_dims == ("country",):
            # Through-country scaling: mean across all countries
            assert np.all(result == input_data / y.mean())
        else:
            # per country scaling
            for idx, country in enumerate(df_lift_test_for_transform["country"].values):
                assert (
                    result[idx]
                    == input_data[idx] / y.loc[X["country"] == country].mean()
                )


@pytest.mark.parametrize(
    "scaling_method",
    ["max", "mean"],
    ids=["max-scaling", "mean-scaling"],
)
def test_make_target_transform_single_dim(
    single_dim_data,
    df_lift_test_for_transform,
    scaling_method,
) -> None:
    """Test _make_target_transform function with single-dimensional data."""
    X, y = single_dim_data

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=(),  # No extra dimensions
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        scaling=Scaling(
            channel=VariableScaling(method=scaling_method, dims=()),
            target=VariableScaling(method=scaling_method, dims=()),
        ),
    )

    mmm.build_model(X, y.reset_index(drop=True))

    # Get the target transform function
    target_transform = mmm._make_target_transform(df_lift_test_for_transform)

    # Test with sample input data (delta_y values)
    input_data = df_lift_test_for_transform[
        "delta_y"
    ].values  # delta_y values from lift test

    # Apply the transform
    result = target_transform(input_data).flatten()

    # Verify the result has the expected shape (2D array)
    assert result.shape == input_data.shape

    # Verify the scaling is applied correctly
    if scaling_method == "max":
        assert np.all(result == input_data / y.max())
    else:  # mean scaling
        assert np.all(result == input_data / y.mean())


class TestPydanticValidation:
    """Test suite specifically for Pydantic validation in multidimensional MMM."""

    def test_empty_channel_columns_raises_validation_error(self):
        """Test that empty channel_columns raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MMM(
                date_column="date",
                channel_columns=[],  # Empty list should fail
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
            )

        # Check that the error message mentions the constraint
        error_msg = str(exc_info.value)
        assert "at least 1 item" in error_msg or "min_length" in error_msg

    def test_invalid_yearly_seasonality_raises_validation_error(self):
        """Test that yearly_seasonality <= 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MMM(
                date_column="date",
                channel_columns=["channel_1"],
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                yearly_seasonality=0,  # Should be > 0
            )

        error_msg = str(exc_info.value)
        assert "greater than 0" in error_msg

    def test_negative_yearly_seasonality_raises_validation_error(self):
        """Test that negative yearly_seasonality raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MMM(
                date_column="date",
                channel_columns=["channel_1"],
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                yearly_seasonality=-1,
            )

        error_msg = str(exc_info.value)
        assert "greater than 0" in error_msg

    def test_invalid_adstock_type_raises_validation_error(self):
        """Test that invalid adstock type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MMM(
                date_column="date",
                channel_columns=["channel_1"],
                target_column="target",
                adstock="not_an_adstock",  # Invalid type
                saturation=LogisticSaturation(),
            )

        error_msg = str(exc_info.value)
        assert "AdstockTransformation" in error_msg

    def test_invalid_saturation_type_raises_validation_error(self):
        """Test that invalid saturation type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MMM(
                date_column="date",
                channel_columns=["channel_1"],
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation="not_a_saturation",  # Invalid type
            )

        error_msg = str(exc_info.value)
        assert "SaturationTransformation" in error_msg

    def test_empty_control_columns_raises_validation_error(self):
        """Test that empty control_columns list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MMM(
                date_column="date",
                channel_columns=["channel_1"],
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                control_columns=[],  # Empty list should fail when not None
            )

        error_msg = str(exc_info.value)
        assert "at least 1 item" in error_msg or "min_length" in error_msg

    def test_invalid_scaling_type_raises_validation_error(self):
        """Test that invalid scaling type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MMM(
                date_column="date",
                channel_columns=["channel_1"],
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                scaling="invalid_scaling",  # Should be Scaling object or dict
            )

        error_msg = str(exc_info.value)
        assert "Scaling" in error_msg or "dict" in error_msg

    def test_valid_scaling_dict_accepted(self):
        """Test that valid scaling dict is accepted and converted."""
        scaling_dict = {
            "channel": {"method": "max", "dims": ()},
            "target": {"method": "max", "dims": ()},
        }
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1"],
            target_column="target",
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            scaling=scaling_dict,
        )
        assert isinstance(mmm.scaling, Scaling)
        assert mmm.scaling.model_dump() == scaling_dict

    def test_valid_scaling_object_accepted(self):
        """Test that valid Scaling object is accepted."""
        scaling_obj = Scaling(
            target=VariableScaling(method="max", dims=()),
            channel=VariableScaling(method="max", dims=()),
        )
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1"],
            target_column="target",
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            scaling=scaling_obj,
        )
        assert mmm.scaling == scaling_obj

    def test_dims_type_validation(self):
        """Test that dims validates as tuple of strings."""
        # Valid dims
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1"],
            target_column="target",
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            dims=("country", "product"),
        )
        assert mmm.dims == ("country", "product")

        # Test with single dimension
        mmm2 = MMM(
            date_column="date",
            channel_columns=["channel_1"],
            target_column="target",
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            dims=("country",),
        )
        assert mmm2.dims == ("country",)

    def test_invalid_boolean_types_raise_validation_error(self):
        """Test that non-boolean values for boolean fields raise ValidationError."""
        with pytest.raises(ValidationError):
            MMM(
                date_column="date",
                channel_columns=["channel_1"],
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                time_varying_intercept="yes",  # Should be boolean
            )

    def test_missing_required_fields_raise_validation_error(self):
        """Test that missing required fields raise ValidationError."""
        # Missing date_column
        with pytest.raises(ValidationError) as exc_info:
            MMM(
                channel_columns=["channel_1"],
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
            )
        error_msg = str(exc_info.value)
        assert "date_column" in error_msg

        # Missing channel_columns
        with pytest.raises(ValidationError) as exc_info:
            MMM(
                date_column="date",
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
            )
        error_msg = str(exc_info.value)
        assert "channel_columns" in error_msg

    def test_all_parameters_with_valid_values(self):
        """Test initialization with all parameters set to valid values."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2", "channel_3"],
            target_column="revenue",
            adstock=GeometricAdstock(l_max=10),
            saturation=LogisticSaturation(),
            time_varying_intercept=True,
            time_varying_media=True,
            dims=("country", "product"),
            scaling=Scaling(
                target=VariableScaling(method="mean", dims=("country",)),
                channel=VariableScaling(method="max", dims=("country", "channel")),
            ),
            model_config={"intercept": Prior("Normal", mu=0, sigma=2)},
            sampler_config={"draws": 1000, "chains": 4},
            control_columns=["holiday", "promotion"],
            yearly_seasonality=4,
            adstock_first=False,
        )

        # Verify all values were set correctly
        assert mmm.date_column == "date"
        assert mmm.channel_columns == ["channel_1", "channel_2", "channel_3"]
        assert mmm.target_column == "revenue"
        assert isinstance(mmm.adstock, GeometricAdstock)
        assert isinstance(mmm.saturation, LogisticSaturation)
        assert mmm.time_varying_intercept is True
        assert mmm.time_varying_media is True
        assert mmm.dims == ("country", "product")
        assert isinstance(mmm.scaling, Scaling)
        assert mmm.control_columns == ["holiday", "promotion"]
        assert mmm.yearly_seasonality == 4
        assert mmm.adstock_first is False

    def test_validation_error_provides_helpful_messages(self):
        """Test that validation errors provide clear, actionable messages."""
        with pytest.raises(ValidationError) as exc_info:
            MMM(
                date_column="date",
                channel_columns="not_a_list",  # Should be a list
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
            )

        # The error should mention that channel_columns should be a list
        error_msg = str(exc_info.value)
        assert "channel_columns" in error_msg
        assert "list" in error_msg.lower()


def test_mmm_linear_trend_different_dimensions_original_scale(
    multi_dim_data,
) -> None:
    X, y = multi_dim_data

    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        scaling={"target": {"method": "max", "dims": ()}},
    )
    trend = LinearTrend(
        n_changepoints=2,
        include_intercept=False,
        priors={
            "delta": Prior("Normal", dims="changepoint"),
        },
        dims=("geo",),
    )

    # Create the wrapper
    trend_effect = LinearTrendEffect(trend=trend, prefix="trend")
    mmm.mu_effects.append(trend_effect)

    mmm.build_model(X, y)

    mmm.add_original_scale_contribution_variable(var=["trend_effect_contribution"])

    mmm.sample_prior_predictive(
        X,
        var_names=["trend_effect_contribution_original_scale", "y"],
    )

    prior = mmm.prior
    variable = prior.trend_effect_contribution_original_scale

    assert variable.dims == ("chain", "draw", "date", "country")
    assert variable.sizes == {
        "chain": 1,
        "draw": 500,
        "date": 7,
        "country": 3,
    }


def test_set_xarray_data_preserves_dtypes(multi_dim_data, mock_pymc_sample):
    """Test that _set_xarray_data preserves the original data types from the model."""
    X, y = multi_dim_data

    # Build and fit the model
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        control_columns=None,  # Testing without control columns first
    )

    mmm.build_model(X, y)

    # Store original dtypes from the model
    original_channel_dtype = mmm.model.named_vars["channel_data"].type.dtype
    original_target_dtype = mmm.model.named_vars["target_data"].type.dtype

    # Create new data with different dtypes
    X_new = X.copy()
    # Convert channel columns to float32 (different from typical float64)
    for col in ["channel_1", "channel_2", "channel_3"]:
        X_new[col] = X_new[col].astype(np.float32)

    # Transform to xarray dataset without target (prediction scenario)
    dataset_xarray = mmm._posterior_predictive_data_transformation(
        X=X_new,
        y=None,  # Don't pass y for prediction
        include_last_observations=False,
    )

    # Verify that the input data has different dtypes
    assert dataset_xarray._channel.dtype == np.float32

    # Apply _set_xarray_data
    model = mmm._set_xarray_data(dataset_xarray, clone_model=True)

    # Check that the data in the model has been converted to the original dtypes
    assert model.named_vars["channel_data"].get_value().dtype == original_channel_dtype

    # Also verify the data shapes are preserved
    assert model.named_vars["channel_data"].get_value().shape == (
        len(X_new[mmm.date_column].unique()),
        len(mmm.xarray_dataset.coords["country"]),
        len(mmm.channel_columns),
    )

    # Now test with target data - create properly structured y data
    # Combine X and y to create a proper DataFrame structure
    df_with_target = X_new.copy()
    df_with_target["target"] = y.values  # Add target column

    # Convert target to float32 to test dtype conversion
    df_with_target["target"] = df_with_target["target"].astype(np.float32)

    # Extract y as a properly indexed Series
    y_new = df_with_target.set_index(["date", "country"])["target"]

    # Transform to xarray dataset with target
    dataset_xarray_with_target = mmm._posterior_predictive_data_transformation(
        X=X_new,
        y=y_new,
        include_last_observations=False,
    )

    # Verify that the target has different dtype
    assert dataset_xarray_with_target._target.dtype == np.float32

    # Apply _set_xarray_data with target
    model_with_target = mmm._set_xarray_data(
        dataset_xarray_with_target, clone_model=True
    )

    # Check that target dtype is preserved
    assert (
        model_with_target.named_vars["target_data"].get_value().dtype
        == original_target_dtype
    )
    assert model_with_target.named_vars["target_data"].get_value().shape == (
        len(X_new[mmm.date_column].unique()),
        len(mmm.xarray_dataset.coords["country"]),
    )


def test_set_xarray_data_with_control_columns_preserves_dtypes(multi_dim_data):
    """Test that _set_xarray_data preserves dtypes when control columns are present."""
    X, y = multi_dim_data

    # Add control columns with specific dtypes
    X["control_1"] = np.random.randn(len(X)).astype(np.float64)
    X["control_2"] = np.random.randn(len(X)).astype(np.float64)

    # Build model with control columns
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        control_columns=["control_1", "control_2"],
    )

    mmm.build_model(X, y)

    # Store original dtypes
    original_channel_dtype = mmm.model.named_vars["channel_data"].type.dtype
    original_control_dtype = mmm.model.named_vars["control_data"].type.dtype
    original_target_dtype = mmm.model.named_vars["target_data"].type.dtype

    # Create new data with different dtypes
    X_new = X.copy()
    # Convert all numeric columns to float32
    for col in X_new.select_dtypes(include=[np.number]).columns:
        X_new[col] = X_new[col].astype(np.float32)

    # First test without target (prediction scenario)
    dataset_xarray = mmm._posterior_predictive_data_transformation(
        X=X_new,
        y=None,
        include_last_observations=False,
    )

    # Apply _set_xarray_data
    model = mmm._set_xarray_data(dataset_xarray, clone_model=True)

    # Check that data types are preserved
    assert model.named_vars["channel_data"].get_value().dtype == original_channel_dtype
    assert model.named_vars["control_data"].get_value().dtype == original_control_dtype

    # Now test with target data - create properly structured y data
    df_with_target = X_new.copy()
    df_with_target["target"] = y.values
    df_with_target["target"] = df_with_target["target"].astype(np.float32)

    # Extract y as a properly indexed Series
    y_new = df_with_target.set_index(["date", "country"])["target"]

    # Transform to xarray dataset with target
    dataset_xarray_with_target = mmm._posterior_predictive_data_transformation(
        X=X_new,
        y=y_new,
        include_last_observations=False,
    )

    # Apply _set_xarray_data with target
    model_with_target = mmm._set_xarray_data(
        dataset_xarray_with_target, clone_model=True
    )

    # Check that all data types are preserved
    assert (
        model_with_target.named_vars["channel_data"].get_value().dtype
        == original_channel_dtype
    )
    assert (
        model_with_target.named_vars["control_data"].get_value().dtype
        == original_control_dtype
    )
    assert (
        model_with_target.named_vars["target_data"].get_value().dtype
        == original_target_dtype
    )


def test_set_xarray_data_without_target_preserves_dtypes(multi_dim_data):
    """Test that _set_xarray_data preserves dtypes when target is not provided."""
    X, y = multi_dim_data

    # Build the model
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
    )

    mmm.build_model(X, y)

    # Store original dtype
    original_channel_dtype = mmm.model.named_vars["channel_data"].type.dtype

    # Create new data without target
    X_new = X.copy()
    for col in ["channel_1", "channel_2", "channel_3"]:
        X_new[col] = X_new[col].astype(np.float32)

    # Transform to xarray dataset without y
    dataset_xarray = mmm._posterior_predictive_data_transformation(
        X=X_new,
        y=None,  # No target provided
        include_last_observations=False,
    )

    # Apply _set_xarray_data
    model = mmm._set_xarray_data(dataset_xarray, clone_model=True)

    # Check that channel data type is preserved
    assert model.named_vars["channel_data"].get_value().dtype == original_channel_dtype

    # Target data should remain unchanged from the original model
    # (no new target data was provided)


@pytest.mark.parametrize(
    "date_col_name",
    ["date_week", "week", "period", "timestamp", "time_period"],
    ids=["date_week", "week", "period", "timestamp", "time_period"],
)
def test_mmm_with_arbitrary_date_column_names_single_dim(
    single_dim_data, date_col_name, mock_pymc_sample
):
    """Test that MMM works with arbitrary date column names (single dimension data).

    This test validates the fix for hardcoded 'date' references by:
    1. Taking existing test data with 'date' column
    2. Renaming the date column to various arbitrary names
    3. Verifying MMM can build and fit models successfully
    4. Checking that internal coordinates use 'date' consistently
    """
    X_single, y_single = single_dim_data

    # Rename the date column to test arbitrary names
    X_renamed = X_single.rename(columns={"date": date_col_name})

    mmm = MMM(
        date_column=date_col_name,  # Use the renamed column
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    # This should work without any manual renaming
    mmm.build_model(X_renamed, y_single)

    # Verify internal coordinates use 'date' consistently
    assert "date" in mmm.model.coords, (
        f"Internal model coordinates should use 'date', not '{date_col_name}'"
    )
    assert "date" in mmm.xarray_dataset.coords, (
        f"Internal xarray coordinates should use 'date', not '{date_col_name}'"
    )

    # Verify model can be fitted
    idata = mmm.fit(X_renamed, y_single, draws=50, tune=25, chains=1)
    assert isinstance(idata, az.InferenceData)

    # Test posterior predictive sampling
    pred_data = mmm.sample_posterior_predictive(
        X_renamed, extend_idata=False, random_seed=42
    )
    assert "date" in pred_data.dims, (
        "Posterior predictive should use 'date' coordinate internally"
    )


@pytest.mark.parametrize(
    "date_col_name",
    ["date_week", "period", "timestamp"],
    ids=["date_week", "period", "timestamp"],
)
def test_mmm_with_arbitrary_date_column_names_multi_dim(
    multi_dim_data, date_col_name, mock_pymc_sample
):
    """Test that MMM works with arbitrary date column names (multi-dimensional data).

    This test validates complex features work with arbitrary date column names.
    """
    X_multi, y_multi = multi_dim_data

    # Rename the date column
    X_renamed = X_multi.rename(columns={"date": date_col_name})

    mmm_multi = MMM(
        date_column=date_col_name,  # Use the renamed column
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_intercept=True,  # Add complexity to test more code paths
        yearly_seasonality=2,
    )

    # Build and fit the model
    mmm_multi.build_model(X_renamed, y_multi)

    # Verify internal coordinates use 'date' consistently
    assert "date" in mmm_multi.model.coords
    assert "date" in mmm_multi.xarray_dataset.coords
    assert "country" in mmm_multi.model.coords  # Should preserve other dims

    # Verify model can be fitted with complex features
    idata_multi = mmm_multi.fit(X_renamed, y_multi, draws=50, tune=25, chains=1)
    assert isinstance(idata_multi, az.InferenceData)

    # Test that time-varying features work with arbitrary date names
    assert "intercept_latent_process" in mmm_multi.model.named_vars
    assert "fourier_contribution" in mmm_multi.model.named_vars

    # Test posterior predictive with new data having the same arbitrary date column name
    X_new = X_renamed.copy()
    diff_days = (X_new[date_col_name].max() - X_new[date_col_name].min()).days + 7
    X_new[date_col_name] += pd.Timedelta(days=diff_days)

    pred_data_multi = mmm_multi.sample_posterior_predictive(
        X_new, extend_idata=False, random_seed=42
    )
    assert "date" in pred_data_multi.dims
    assert "country" in pred_data_multi.dims


def test_date_column_validation_with_arbitrary_names(single_dim_data):
    """Test that proper validation occurs with arbitrary date column names."""
    X, y = single_dim_data

    # Test that specifying wrong date column name raises appropriate error
    X_renamed = X.rename(columns={"date": "week_ending"})

    mmm = MMM(
        date_column="wrong_column_name",  # This column doesn't exist
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    # Should raise an error because 'wrong_column_name' is not in the DataFrame
    with pytest.raises(ValueError, match="date_column 'wrong_column_name' not found"):
        mmm.build_model(X_renamed, y)


@pytest.mark.parametrize(
    "date_col_name",
    ["date_start", "end_date", "date_week_ending", "reporting_date"],
    ids=["date_start", "end_date", "date_week_ending", "reporting_date"],
)
def test_mixed_date_column_scenarios_variations(
    single_dim_data, date_col_name, mock_pymc_sample
):
    """Test edge cases with date column names that contain 'date' but aren't exactly 'date'."""
    X, y = single_dim_data

    X_test = X.rename(columns={"date": date_col_name})

    mmm = MMM(
        date_column=date_col_name,
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    mmm.build_model(X_test, y)

    # Verify scaling operations work correctly
    scales = mmm.get_scales_as_xarray()
    assert "channel_scale" in scales
    assert "target_scale" in scales

    # Verify the model builds successfully
    assert hasattr(mmm, "model")
    assert hasattr(mmm, "xarray_dataset")


def test_arbitrary_date_column_with_control_variables(
    single_dim_data, mock_pymc_sample
):
    """Test that the fix works with control columns and arbitrary date column names."""
    X, y = single_dim_data

    # Scenario: Ensure the fix works with control columns
    X_with_controls = X.rename(columns={"date": "time_stamp"})
    X_with_controls["control_1"] = np.random.normal(0, 1, len(X_with_controls))
    X_with_controls["control_2"] = np.random.normal(0, 1, len(X_with_controls))

    mmm_controls = MMM(
        date_column="time_stamp",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    mmm_controls.build_model(X_with_controls, y)

    # Verify control data was processed correctly
    assert "_control" in mmm_controls.xarray_dataset
    assert "control_data" in mmm_controls.model.named_vars

    idata = mmm_controls.fit(X_with_controls, y, draws=50, tune=25, chains=1)
    assert isinstance(idata, az.InferenceData)


@pytest.mark.parametrize(
    "model_config, expected_config, expected_rv",
    [
        pytest.param(
            {"intercept_tvp_config": {"ls_lower": 0.1, "ls_upper": None}},
            None,
            dict(name="intercept_latent_process_raw_ls_raw", kind="WeibullBetaRV"),
            id="weibull",
        ),
        pytest.param(
            {"intercept_tvp_config": {"ls_lower": 1, "ls_upper": 10}},
            None,
            dict(name="intercept_latent_process_raw_ls", kind="InvGammaRV"),
            id="inversegamma",
        ),
    ],
)
def test_specify_time_varying_configuration(
    single_dim_data,
    model_config,
    expected_config,
    expected_rv,
) -> None:
    X, y = single_dim_data
    expected_config = expected_config or model_config

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        model_config=model_config,
        time_varying_intercept=True,
    )

    assert isinstance(mmm.model_config["intercept_tvp_config"], dict)
    assert (
        mmm.model_config["intercept_tvp_config"]
        == expected_config["intercept_tvp_config"]
    )

    mmm.build_model(X, y)

    assert (
        mmm.model[expected_rv["name"]].owner.op.__class__.__name__
        == expected_rv["kind"]
    )
