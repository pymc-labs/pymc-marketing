#   Copyright 2025 - 2025 The PyMC Labs Developers
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
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from tests.conftest import mock_sample


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
def mock_pymc_sample() -> None:
    original_sample = pm.sample
    pm.sample = mock_sample

    yield

    pm.sample = original_sample


@pytest.fixture
def fit_mmm(df, mmm, mock_pymc_sample):
    X = df.drop(columns=["y"])
    y = df["y"]

    mmm.fit(X, y)

    return mmm


def test_fit(fit_mmm):
    assert isinstance(fit_mmm.posterior, xr.Dataset)
    assert isinstance(fit_mmm.idata.fit_data, xr.Dataset)


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

    df = pd.DataFrame(
        {
            "date": date_range,
            "channel_1": channel_1,
            "channel_2": channel_2,
        }
    )
    # Target is sum of channels with noise
    df["target"] = (
        df["channel_1"]
        + df["channel_2"]
        + np.random.randint(100, 300, size=len(date_range))
    )
    X = df[["date", "channel_1", "channel_2"]].copy()

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
            target = channel_1 + channel_2 + np.random.randint(50, 150)
            records.append((date, country, channel_1, channel_2, target))

    df = pd.DataFrame(
        records, columns=["date", "country", "channel_1", "channel_2", "target"]
    )

    X = df[["date", "country", "channel_1", "channel_2"]].copy()

    return X, df["target"].copy()


@pytest.mark.parametrize(
    "time_varying_intercept, time_varying_media, yearly_seasonality, dims",
    [
        (False, False, None, ()),  # no time-varying, no seasonality, no extra dims
        (False, False, 4, ()),  # no time-varying, has seasonality, no extra dims
        (
            True,
            False,
            None,
            (),
        ),  # time-varying intercept only, no seasonality, no extra dims
        (False, True, 4, ()),  # time-varying media only, has seasonality, no extra dims
        (True, True, 4, ()),  # both time-varying, has seasonality, no extra dims
    ],
)
def test_build_model_single_dim(
    single_dim_data,
    time_varying_intercept,
    time_varying_media,
    yearly_seasonality,
    dims,
):
    """Test that building the model works with different configurations (single-dim)."""
    X, y = single_dim_data
    adstock = GeometricAdstock(l_max=2)
    saturation = LogisticSaturation()

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        dims=dims,
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=yearly_seasonality,
        time_varying_intercept=time_varying_intercept,
        time_varying_media=time_varying_media,
    )

    mmm.build_model(X, y)

    # Assertions
    assert hasattr(mmm, "model"), "Model attribute should be set after build_model."
    assert isinstance(mmm.model, pm.Model), "mmm.model should be a PyMC Model instance."

    # Basic checks to confirm presence of key variables
    var_names = mmm.model.named_vars.keys()
    assert "channel_data" in var_names
    assert "target" in var_names
    if time_varying_intercept:
        assert "intercept_latent_process" in var_names
    if time_varying_media:
        assert "media_latent_process" in var_names
    if yearly_seasonality is not None:
        assert "fourier_contribution" in var_names


@pytest.mark.parametrize(
    "time_varying_intercept, time_varying_media, yearly_seasonality, dims",
    [
        (
            False,
            False,
            None,
            ("country",),
        ),  # no time-varying, no seasonality, 1 extra dim
        (
            True,
            False,
            4,
            ("country",),
        ),  # time-varying intercept only, has seasonality, 1 extra dim
        (
            False,
            True,
            4,
            ("country",),
        ),  # time-varying media only, has seasonality, 1 extra dim
        (
            True,
            True,
            2,
            ("country",),
        ),  # both time-varying, has seasonality, 1 extra dim
    ],
)
def test_build_model_multi_dim(
    multi_dim_data, time_varying_intercept, time_varying_media, yearly_seasonality, dims
):
    """Test building the model when extra dimensions (like 'country') are present."""
    X, y = multi_dim_data
    adstock = GeometricAdstock(l_max=2)
    saturation = LogisticSaturation()

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        dims=dims,
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=yearly_seasonality,
        time_varying_intercept=time_varying_intercept,
        time_varying_media=time_varying_media,
    )

    mmm.build_model(X, y)

    assert hasattr(mmm, "model"), "Model attribute should be set after build_model."
    assert isinstance(mmm.model, pm.Model), "mmm.model should be a PyMC Model instance."
    assert "country" in mmm.model.coords, (
        "Extra dimension 'country' should be in model coords."
    )


def test_fit_single_dim(single_dim_data):
    """Test fitting the model on a single-dimension dataset."""
    X, y = single_dim_data

    adstock = GeometricAdstock(l_max=2)
    saturation = LogisticSaturation()

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        dims=(),
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=None,  # disable yearly seasonality
        time_varying_intercept=False,
        time_varying_media=False,
    )

    # Build and fit
    mmm.build_model(X, y)

    # To keep tests fast, set small number of draws/tune
    idata = mmm.fit(X, y, draws=10, tune=10, chains=1)
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


def test_fit_multi_dim(multi_dim_data):
    """Test fitting the model on a multi-dimensional dataset (e.g. with 'country')."""
    X, y = multi_dim_data

    adstock = GeometricAdstock(l_max=2)
    saturation = LogisticSaturation()

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        dims=("country",),
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=2,
        time_varying_intercept=True,
        time_varying_media=True,
    )

    # Build and fit
    mmm.build_model(X, y)

    # Again, keep the sampler small for test speed
    idata = mmm.fit(X, y, draws=10, tune=10, chains=1)
    assert isinstance(idata, az.InferenceData), (
        "fit should return an InferenceData object."
    )
    assert hasattr(mmm, "idata"), (
        "MMM instance should store the inference data as 'idata'."
    )

    # Check if 'country' is in the posterior dimensions
    assert "country" in mmm.idata.posterior.dims, (
        "Posterior should have 'country' dimension."
    )


def test_sample_posterior_predictive_new_data(single_dim_data, new_data_single_dim):
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
        channel_columns=["channel_1", "channel_2"],
        adstock=adstock,
        saturation=saturation,
    )

    # Fit with a fixed seed for reproducibility
    mmm.build_model(X_train, y_train)
    mmm.fit(X_train, y_train, draws=200, tune=100, chains=1, random_seed=42)

    mmm.sample_posterior_predictive(X_train, extend_idata=True, random_seed=42)

    # Sample posterior predictive on new data
    out_of_sample_idata = mmm.sample_posterior_predictive(
        X_new, extend_idata=False, random_seed=42
    )

    # Check that posterior_predictive group was added
    assert hasattr(mmm.idata, "posterior_predictive"), (
        "After calling sample_posterior_predictive with new data, "
        "there should be a 'posterior_predictive' group in the inference data."
    )

    # Check the shape of that group. We expect the new date dimension to match X_new length
    # plus no addition if we didn't set include_last_observations (which is False by default).
    assert "date" in out_of_sample_idata.dims, (
        "posterior_predictive should have a 'date' dimension."
    )
    assert out_of_sample_idata.coords["date"].values.shape == X_new.date.values.shape, (
        "The 'date' dimension in posterior_predictive should match new data length."
    )


def test_sample_posterior_predictive_same_data(single_dim_data):
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
        channel_columns=["channel_1", "channel_2"],
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
