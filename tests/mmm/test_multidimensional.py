#   Copyright 2022 - 2026 The PyMC Labs Developers
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
import os
from collections.abc import Callable

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pydantic import ValidationError
from pymc.model_graph import fast_eval
from pymc_extras.prior import Prior
from pytensor.tensor.basic import TensorVariable
from scipy.optimize import OptimizeResult

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm import (
    CovFunc,
    GeometricAdstock,
    LogisticSaturation,
    SoftPlusHSGP,
)
from pymc_marketing.mmm.additive_effect import EventAdditiveEffect, LinearTrendEffect
from pymc_marketing.mmm.events import EventEffect, GaussianBasis, HalfGaussianBasis
from pymc_marketing.mmm.lift_test import _swap_columns_and_last_index_level
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.multidimensional import (
    MMM,
    MultiDimensionalBudgetOptimizerWrapper,
)
from pymc_marketing.mmm.scaling import Scaling, VariableScaling


@pytest.fixture
def target_column():
    return "y_named"


@pytest.fixture
def mmm(target_column):
    return MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        dims=("country",),
        target_column=target_column,
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )


@pytest.fixture
def df(target_column) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=3, freq="W-MON").rename("date")
    df = pd.DataFrame(
        {
            ("A", "C1"): [1, 2, 3.0],
            ("B", "C1"): [4, 5, 6.0],
            ("A", "C2"): [7, 8, 9.0],
            ("B", "C2"): [10, 11, 12.0],
        },
        index=dates,
    )
    df.columns.names = ["country", "channel"]

    y = pd.DataFrame(
        {
            ("A", target_column): [1, 2, 3],
            ("B", target_column): [4, 5, 6],
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
def fit_mmm(df, mmm, target_column, mock_pymc_sample):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    mmm.fit(X, y)

    return mmm


def test_target_column():
    mmm_default = MMM(
        date_column="date",
        channel_columns=["C"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )
    assert mmm_default.target_column == "y"

    mmm_custom = MMM(
        date_column="date",
        channel_columns=["C"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        target_column="epsilon",
    )
    assert mmm_custom.target_column == "epsilon"


def test_reserved_dims():
    other_kwargs = {
        "date_column": "date",
        "channel_columns": ["C"],
        "adstock": GeometricAdstock(l_max=2),
        "saturation": LogisticSaturation(),
    }
    # Calling MMM without a reserved dim is fine
    MMM(**other_kwargs, dims=("calendar",))

    for reserved_dim in ("date", "channel", "control", "fourier_mode"):
        with pytest.raises(ValueError, match=r".* reserved for internal use"):
            MMM(**other_kwargs, dims=(reserved_dim,))


def test_simple_fit(fit_mmm):
    assert isinstance(fit_mmm.posterior, xr.Dataset)
    assert isinstance(fit_mmm.idata.constant_data, xr.Dataset)


def test_sample_prior_predictive(mmm: MMM, target_column, df: pd.DataFrame):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    mmm.sample_prior_predictive(X, y)

    assert isinstance(mmm.prior, xr.Dataset)
    assert isinstance(mmm.prior_predictive, xr.Dataset)


def test_save_load(fit_mmm: MMM):
    file = "test.nc"
    fit_mmm.save(file)

    loaded = MMM.load(file)
    assert isinstance(loaded, MMM)

    os.remove(file)


def test_save_load_equality(fit_mmm: MMM):
    """Test that save/load produces an equivalent MMM instance.

    Tests the __eq__ method which validates ALL configuration aspects:
    - Core configuration (date, channels, target, dims, scaling)
    - Transformations (adstock, saturation, adstock_first)
    - Time-varying effects (HSGPs if present)
    - Additive effects (mu_effects)
    - Causal graph configuration
    - Model and sampler configuration
    """
    file = "test_equality.nc"
    original_mmm = fit_mmm

    # Save the model
    original_mmm.save(file)

    # Load the model
    loaded_mmm = MMM.load(file)

    # Test that loaded model equals original (using __eq__)
    assert loaded_mmm == original_mmm, (
        "Loaded MMM should be equal to original. "
        "Check __eq__ method for which properties don't match."
    )

    # Also verify key properties individually
    assert loaded_mmm.id == original_mmm.id
    assert loaded_mmm.date_column == original_mmm.date_column
    assert loaded_mmm.channel_columns == original_mmm.channel_columns
    assert loaded_mmm.target_column == original_mmm.target_column
    assert loaded_mmm.dims == original_mmm.dims
    assert loaded_mmm.adstock_first == original_mmm.adstock_first
    assert loaded_mmm.yearly_seasonality == original_mmm.yearly_seasonality
    assert loaded_mmm.sampler_config == original_mmm.sampler_config

    # Clean up
    import os

    os.remove(file)


def test_save_load_equality_with_all_effects(mock_pymc_sample):
    """Test save/load roundtrip with all MuEffects and HSGP time-varying effects.

    This test ensures that an MMM with:
    - Multiple MuEffects (FourierEffect, LinearTrendEffect)
    - Time-varying intercept (HSGP)
    - Time-varying media (HSGP)

    ...can be saved and loaded while maintaining complete equality via __eq__.
    """
    from pymc_marketing.mmm.additive_effect import FourierEffect, LinearTrendEffect
    from pymc_marketing.mmm.fourier import YearlyFourier
    from pymc_marketing.mmm.linear_trend import LinearTrend

    # Create test data
    date_range = pd.date_range("2023-01-01", periods=100, freq="W")
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "date": date_range,
            "channel_1": np.random.randint(100, 500, size=len(date_range)),
            "channel_2": np.random.randint(100, 500, size=len(date_range)),
            "target": np.random.randint(500, 1500, size=len(date_range)),
        }
    )

    # Create MMM with time-varying effects
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        time_varying_intercept=True,
        time_varying_media=True,
    )

    # Add MuEffects
    mmm.mu_effects.append(
        FourierEffect(fourier=YearlyFourier(n_order=3, prefix="yearly"))
    )
    mmm.mu_effects.append(
        LinearTrendEffect(
            trend=LinearTrend(n_changepoints=5),
            prefix="trend",
        )
    )

    # Fit the model
    X = df[["date", "channel_1", "channel_2"]]
    y = df["target"]
    mmm.fit(X, y)

    # Save the model
    file = "test_all_effects_equality.nc"
    mmm.save(file)

    # Load the model
    loaded_mmm = MMM.load(file)

    # Test that loaded model equals original (using __eq__)
    assert loaded_mmm == mmm, (
        "Loaded MMM with all effects should equal original. "
        "Check __eq__ method for which properties don't match."
    )

    # Verify specific properties
    assert loaded_mmm.id == mmm.id
    assert len(loaded_mmm.mu_effects) == len(mmm.mu_effects) == 2
    assert isinstance(loaded_mmm.mu_effects[0], FourierEffect)
    assert isinstance(loaded_mmm.mu_effects[1], LinearTrendEffect)

    # Verify HSGP serialization - skip for now as it's a separate issue
    # TODO: Fix HSGP serialization separately
    # assert loaded_mmm.time_varying_intercept is not None
    # assert loaded_mmm.time_varying_media is not None
    # if hasattr(loaded_mmm.time_varying_intercept, "to_dict"):
    #     assert (
    #         loaded_mmm.time_varying_intercept.to_dict()
    #         == mmm.time_varying_intercept.to_dict()
    #     )
    # if hasattr(loaded_mmm.time_varying_media, "to_dict"):
    #     assert (
    #         loaded_mmm.time_varying_media.to_dict()
    #         == mmm.time_varying_media.to_dict()
    #     )

    # Clean up
    import os

    os.remove(file)


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
        pytest.param("multi_dim_data", ("country",), id="Country model"),
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
        assert "media_temporal_latent_multiplier" in var_names
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

    # Check presence of fit_data group
    assert hasattr(mmm.idata, "fit_data"), "InferenceData should have a fit_data group."

    np.testing.assert_equal(
        mmm.idata.fit_data.coords["date"].values, mmm.model.coords["date"]
    )
    if mmm.dims:
        for dim in mmm.dims:
            np.testing.assert_equal(
                mmm.idata.fit_data.coords[dim].values, mmm.model.coords[dim]
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
    y_train = y.iloc[:-5]

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
        match=r"Cannot use include_last_observations=True when input dates overlap with training dates",
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
        match=r"Cannot use include_last_observations=True when input dates overlap with training dates",
    ):
        mmm.sample_posterior_predictive(
            overlap_data,
            include_last_observations=True,
            extend_idata=False,
            random_seed=123,
        )


@pytest.mark.parametrize(
    "hsgp_dims",
    [
        pytest.param(("date",), id="hsgp-dims=date"),
        pytest.param(("date", "channel"), id="hsgp-dims=date,channel"),
    ],
)
def test_time_varying_media_with_custom_hsgp_single_dim(single_dim_data, hsgp_dims):
    """Ensure passing an HSGP instance to time_varying_media works (single-dim)."""
    X, y = single_dim_data

    # Build HSGP using the new API
    hsgp = SoftPlusHSGP.parameterize_from_data(
        X=np.arange(X.shape[0]),
        dims=hsgp_dims,
    )

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_media=hsgp,
    )

    mmm.build_model(X, y)

    # Check latent multiplier exists with the expected dims
    var_name = "media_temporal_latent_multiplier"
    assert var_name in mmm.model.named_vars
    latent_dims = mmm.model.named_vars_to_dims[var_name]
    assert latent_dims == hsgp_dims

    # Channel contribution should always be date x channel
    assert mmm.model.named_vars_to_dims["channel_contribution"] == ("date", "channel")


@pytest.mark.parametrize(
    "hsgp_dims",
    [
        pytest.param(("date",), id="hsgp-dims=date"),
    ],
)
def test_time_varying_intercept_with_custom_hsgp_single_dim(single_dim_data, hsgp_dims):
    """Ensure passing an HSGP instance to time_varying_intercept works (single/multi-dim)."""
    X, y = single_dim_data

    hsgp = SoftPlusHSGP.parameterize_from_data(
        X=np.arange(X.shape[0]),
        dims=hsgp_dims,
    )

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_intercept=hsgp,
    )

    mmm.build_model(X, y)

    var_name = "intercept_latent_process"
    assert var_name in mmm.model.named_vars
    latent_dims = mmm.model.named_vars_to_dims[var_name]
    assert latent_dims == hsgp_dims


@pytest.mark.parametrize(
    "cov_func",
    ["expquad", "matern32", "matern52"],
    ids=["expquad", "matern32", "matern52"],
)
def test_time_varying_media_with_custom_hsgp_single_dim_kernels(
    single_dim_data, cov_func
) -> None:
    """Ensure MMM builds when HSGP uses different kernels for media TVP (single-dim)."""
    X, y = single_dim_data

    hsgp = SoftPlusHSGP.parameterize_from_data(
        X=np.arange(X.shape[0]),
        dims=("date", "channel"),
        cov_func=cov_func,
    )

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_media=hsgp,
    )

    mmm.build_model(X, y)

    var_name = "media_temporal_latent_multiplier"
    assert var_name in mmm.model.named_vars
    assert mmm.model.named_vars_to_dims[var_name] == ("date", "channel")
    # channel contribution dims are stable
    assert mmm.model.named_vars_to_dims["channel_contribution"] == ("date", "channel")


@pytest.mark.parametrize(
    "cov_func",
    ["expquad", "matern32", "matern52"],
    ids=["expquad", "matern32", "matern52"],
)
def test_time_varying_intercept_with_custom_hsgp_single_dim_kernels(
    single_dim_data, cov_func
) -> None:
    """Ensure MMM builds when HSGP uses different kernels for intercept TVP (single-dim)."""
    X, y = single_dim_data

    hsgp = SoftPlusHSGP.parameterize_from_data(
        X=np.arange(X.shape[0]),
        dims=("date",),
        cov_func=cov_func,
    )

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_intercept=hsgp,
    )

    mmm.build_model(X, y)

    var_name = "intercept_latent_process"
    assert var_name in mmm.model.named_vars
    assert mmm.model.named_vars_to_dims[var_name] == ("date",)


@pytest.mark.parametrize(
    "hsgp_dims",
    [
        pytest.param(("date", "country"), id="hsgp-dims=date,country"),
        pytest.param(
            ("date", "country", "channel"), id="hsgp-dims=date,country,channel"
        ),
    ],
)
def test_time_varying_media_with_custom_hsgp_multi_dim(df, target_column, hsgp_dims):
    """Ensure passing an HSGP instance to time_varying_media works (multi-dim)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    hsgp = SoftPlusHSGP.parameterize_from_data(
        X=np.arange(X.shape[0]),
        dims=hsgp_dims,
    )

    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        target_column=target_column,
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_media=hsgp,
    )

    mmm.build_model(X, y)

    var_name = "media_temporal_latent_multiplier"
    assert var_name in mmm.model.named_vars
    latent_dims = mmm.model.named_vars_to_dims[var_name]
    assert latent_dims == hsgp_dims

    # Channel contribution should always be date x country x channel
    assert mmm.model.named_vars_to_dims["channel_contribution"] == (
        "date",
        "country",
        "channel",
    )


@pytest.mark.parametrize(
    "hsgp_dims",
    [
        pytest.param(("date", "country"), id="hsgp-dims=date,country"),
        pytest.param(("date",), id="hsgp-dims=date"),
    ],
)
def test_time_varying_intercept_with_custom_hsgp_multi_dim(
    df, target_column, hsgp_dims
):
    """Ensure passing an HSGP instance to time_varying_intercept works (multi-dim)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    hsgp = SoftPlusHSGP.parameterize_from_data(
        X=np.arange(X.shape[0]),
        dims=hsgp_dims,
    )

    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        target_column=target_column,
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_intercept=hsgp,
    )

    mmm.build_model(X, y)

    var_name = "intercept_latent_process"
    assert var_name in mmm.model.named_vars
    latent_dims = mmm.model.named_vars_to_dims[var_name]
    assert latent_dims == hsgp_dims


@pytest.mark.parametrize(
    "hsgp_dims",
    [
        pytest.param(
            [
                "date",
            ],
            id="hsgp-dims=date",
        ),
        pytest.param(["date", "channel"], id="hsgp-dims=date,channel"),
    ],
)
def test_time_varying_media_with_custom_hsgp_single_dim_save_load(
    single_dim_data, hsgp_dims
):
    """
    Ensure saved MMM with HSGP instance passed to time_varying_media can .save() and .load() (single-dim).
    """
    X, y = single_dim_data

    data = {
        "m": 72,
        "X_mid": 6.5,
        "dims": hsgp_dims,
        "transform": None,
        "demeaned_basis": False,
        "ls": {
            "dist": "Weibull",
            "kwargs": {"alpha": 0.5, "beta": 90.08328710020781},
            "transform": "reciprocal",
        },
        "eta": {"dist": "Exponential", "kwargs": {"lam": 2.995732273553991}},
        "L": 41.6,
        "centered": False,
        "drop_first": True,
        "cov_func": CovFunc.ExpQuad,
    }

    hsgp = SoftPlusHSGP.from_dict(data.copy())  # .from_dict() modifies data

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_media=hsgp,
    )

    mmm.fit(X, y)

    file = "test_hsgp_media.nc"
    mmm.save(file)
    loaded = MMM.load(file)

    assert loaded.time_varying_media.to_dict() == data

    os.remove(file)


@pytest.mark.parametrize(
    "hsgp_dims",
    [
        pytest.param(
            [
                "date",
            ],
            id="hsgp-dims=date",
        ),
    ],
)
def test_time_varying_intercept_with_custom_hsgp_single_dim_save_load(
    single_dim_data, hsgp_dims
):
    """
    Ensure MMM with an HSGP instance passed to time_varying_intercept can .save() and .load() (single-dim).
    """
    X, y = single_dim_data

    data = {
        "m": 72,
        "X_mid": 6.5,
        "dims": hsgp_dims,
        "transform": None,
        "demeaned_basis": False,
        "ls": {
            "dist": "Weibull",
            "kwargs": {"alpha": 0.5, "beta": 90.08328710020781},
            "transform": "reciprocal",
        },
        "eta": {"dist": "Exponential", "kwargs": {"lam": 2.995732273553991}},
        "L": 41.6,
        "centered": False,
        "drop_first": True,
        "cov_func": CovFunc.ExpQuad,
    }

    hsgp = SoftPlusHSGP.from_dict(data.copy())  # .from_dict() modifies data

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_intercept=hsgp,
    )

    mmm.fit(X, y)

    file = "test_hsgp_intercept.nc"
    mmm.save(file)
    loaded = MMM.load(file)

    assert loaded.time_varying_intercept.to_dict() == data

    os.remove(file)


@pytest.mark.parametrize(
    "hsgp_dims",
    [
        pytest.param(["date", "country"], id="hsgp-dims=date,country"),
        pytest.param(
            ["date", "country", "channel"], id="hsgp-dims=date,country,channel"
        ),
    ],
)
def test_time_varying_media_with_custom_hsgp_multi_dim_save_load(
    df, target_column, hsgp_dims
):
    """
    Ensure MMM with an HSGP instance passed to time_varying_media can .save() and .load() (multi-dim).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    data = {
        "m": 28,
        "X_mid": 2.5,
        "dims": hsgp_dims,
        "transform": None,
        "demeaned_basis": False,
        "ls": {
            "dist": "Weibull",
            "kwargs": {"alpha": 0.5, "beta": 90.08328710020781},
            "transform": "reciprocal",
        },
        "eta": {"dist": "Exponential", "kwargs": {"lam": 2.995732273553991}},
        "L": 16.0,
        "centered": False,
        "drop_first": True,
        "cov_func": CovFunc.ExpQuad,
    }
    hsgp = SoftPlusHSGP.from_dict(data.copy())  # .from_dict() modifies data

    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        target_column=target_column,
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_media=hsgp,
    )
    mmm.fit(X, y)

    file = "test_hsgp_intercept_multi_dim.nc"
    mmm.save(file)
    loaded = MMM.load(file)

    assert loaded.time_varying_media.to_dict() == data

    os.remove(file)


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
        dims: tuple[str] | str | None = None,
    ):
        basis = GaussianBasis()
        return EventEffect(
            basis=basis,
            effect_size=Prior("Normal"),
            dims=dims or (prefix,),
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
    target_column,
    mock_pymc_sample,
) -> None:
    mmm.add_events(
        df_events,
        prefix="holiday",
        effect=create_event_effect(prefix="holiday"),
    )
    assert len(mmm.mu_effects) == 1

    df_events_with_country = df_events.copy()
    df_events_with_country["country"] = "A"
    mmm.add_events(
        df_events_with_country,
        prefix="another_event_type",
        effect=create_event_effect(
            prefix="another_event_type", dims=("country", "another_event_type")
        ),
    )
    assert len(mmm.mu_effects) == 2

    X = df.drop(columns=[target_column])
    y = df[target_column]
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
    "basis_factory, expected_zero",
    [
        pytest.param(lambda: GaussianBasis(), False, id="gaussian"),
        pytest.param(
            lambda: HalfGaussianBasis(mode="after", include_event=True),
            False,
            id="halfgaussian-after",
        ),
        pytest.param(
            lambda: HalfGaussianBasis(mode="before", include_event=True),
            True,
            id="halfgaussian-before",
        ),
    ],
)
def test_mmm_with_events_bases(
    df_events, mmm, df, basis_factory, expected_zero, target_column
):
    basis = basis_factory()
    effect = EventEffect(basis=basis, effect_size=Prior("Normal"), dims=("holiday",))

    mmm.add_events(
        df_events=df_events,
        prefix="holiday",
        effect=effect,
    )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    mmm.build_model(X, y)
    mmm.sample_prior_predictive(X, y)  # type: ignore

    da = mmm.prior["holiday_total_effect"]
    assert "date" in da.dims

    if expected_zero:
        np.testing.assert_allclose(da, 0)
    else:
        assert np.any(np.abs(da.values) > 0)


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
    match = r"Target scaling dims"
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
    with pytest.raises(RuntimeError, match=r"The model has not been built yet."):
        mmm.add_lift_test_measurements(
            pd.DataFrame(),
        )


def test_add_calibration_test_measurements(multi_dim_data):
    X, y = multi_dim_data

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    # Build the model
    mmm.build_model(X, y)

    # Add original scale contribution variable first (required before calibration)
    mmm.add_original_scale_contribution_variable(var=["channel_contribution"])

    # Spend data: same structure as X (use X directly for simplicity)
    spend_df = X.copy()

    # Calibration rows map to dims+channel; provide targets and sigma
    # Pick two concrete rows present in coords
    countries = mmm.model.coords["country"]
    channels = ["channel_1", "channel_2"]
    calibration_df = pd.DataFrame(
        {
            "country": [countries[0], countries[1]],
            "channel": [channels[0], channels[1]],
            "cost_per_target": [30.0, 45.0],
            "sigma": [2.0, 3.0],
        }
    )

    assert "cost_per_target" not in mmm.model.named_vars

    mmm.add_cost_per_target_calibration(
        data=spend_df,
        calibration_data=calibration_df,
        name_prefix="cpt_calibration",
    )

    assert "channel_contribution_original_scale" in mmm.model.named_vars
    assert "cost_per_target" not in mmm.model.named_vars

    pot_names = [getattr(p, "name", None) for p in mmm.model.potentials]
    assert "cpt_calibration" in pot_names


def test_add_cost_per_target_calibration_requires_model(multi_dim_data) -> None:
    X, _ = multi_dim_data

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    spend_df = X.copy()
    calibration_df = pd.DataFrame(
        {
            "country": [spend_df["country"].iloc[0]],
            "channel": ["channel_1"],
            "cost_per_target": [30.0],
            "sigma": [2.0],
        }
    )

    with pytest.raises(
        RuntimeError, match=r"Model must be built before adding calibration."
    ):
        mmm.add_cost_per_target_calibration(
            data=spend_df,
            calibration_data=calibration_df,
            name_prefix="cpt_calibration",
        )


def test_add_cost_per_target_calibration_requires_original_scale(
    multi_dim_data,
) -> None:
    """Test that add_cost_per_target_calibration raises error when original scale variable doesn't exist."""
    X, y = multi_dim_data

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    mmm.build_model(X, y)

    # Don't add original scale variable - should cause error
    spend_df = X.copy()
    countries = mmm.model.coords["country"]
    calibration_df = pd.DataFrame(
        {
            "country": [countries[0]],
            "channel": ["channel_1"],
            "cost_per_target": [30.0],
            "sigma": [2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"`channel_contribution_original_scale` is not in the model.",
    ):
        mmm.add_cost_per_target_calibration(
            data=spend_df,
            calibration_data=calibration_df,
            name_prefix="cpt_calibration",
        )


def test_add_cost_per_target_calibration_missing_dim_column(multi_dim_data) -> None:
    X, y = multi_dim_data

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    mmm.build_model(X, y)

    spend_df = X.copy()
    calibration_df = pd.DataFrame(
        {
            "channel": ["channel_1"],
            "cost_per_target": [40.0],
            "sigma": [2.5],
        }
    )

    with pytest.raises(
        KeyError,
        match=r"The country column is required in calibration_data to map to model dims.",
    ):
        mmm.add_cost_per_target_calibration(
            data=spend_df,
            calibration_data=calibration_df,
            name_prefix="cpt_calibration",
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
        match=r"The 'channel' column is required to map the lift measurements to the model.",
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
        match=r"The country column is required to map the lift measurements to the model.",
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
        match=r"The country column is required to map the lift measurements to the model.",
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
        match=r"The product column is required to map the lift measurements to the model.",
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


def test_mmm_equality():
    """Test MMM.__eq__() method for comparing instances.

    Tests that __eq__ correctly compares all configuration attributes:
    - Core configuration (date, channels, target, dims, scaling)
    - Transformations (adstock, saturation, adstock_first)
    - Time-varying effects (time_varying_intercept, time_varying_media)
    - Additive effects (mu_effects)
    - Causal graph (dag, treatment_nodes, outcome_node)
    - Control columns and seasonality settings
    - Model and sampler configuration
    - Model ID (content-based hash)
    """
    # Create two identical MMM instances
    mmm1 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("geo",),
        control_columns=["control_1"],
        yearly_seasonality=2,
        adstock_first=True,
    )

    mmm2 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("geo",),
        control_columns=["control_1"],
        yearly_seasonality=2,
        adstock_first=True,
    )

    # Test: Identical configurations should be equal
    assert mmm1 == mmm2
    assert mmm1.id == mmm2.id

    # Test: Different date_column
    mmm3 = MMM(
        date_column="date_column",  # Different
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("geo",),
    )
    assert mmm1 != mmm3

    # Test: Different channel_columns
    mmm4 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_3"],  # Different
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("geo",),
    )
    assert mmm1 != mmm4

    # Test: Different target_column
    mmm5 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="revenue",  # Different
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("geo",),
    )
    assert mmm1 != mmm5

    # Test: Different dims
    mmm6 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("country",),  # Different
    )
    assert mmm1 != mmm6

    # Test: Different adstock transformation
    mmm7 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=10),  # Different l_max
        saturation=LogisticSaturation(),
        dims=("geo",),
    )
    assert mmm1 != mmm7

    # Test: Different saturation transformation
    mmm8 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(
            priors={"lam": Prior("Normal", mu=0.5, sigma=0.2)}
        ),  # Different priors
        dims=("geo",),
    )
    assert mmm1 != mmm8

    # Test: Different adstock_first
    mmm9 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("geo",),
        adstock_first=False,  # Different
    )
    assert mmm1 != mmm9

    # Test: Different control_columns
    mmm10 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("geo",),
        control_columns=["control_2"],  # Different
    )
    assert mmm1 != mmm10

    # Test: Different yearly_seasonality
    mmm11 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("geo",),
        yearly_seasonality=4,  # Different
    )
    assert mmm1 != mmm11

    # Test: Different scaling configuration
    mmm12 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("geo",),
        scaling=Scaling(
            target=VariableScaling(method="mean", dims=("geo",)),  # Different method
            channel=VariableScaling(method="max", dims=("geo",)),
        ),
    )
    assert mmm1 != mmm12

    # Test: Different time_varying_intercept
    mmm13 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("geo",),
        time_varying_intercept=True,  # Different
    )
    assert mmm1 != mmm13

    # Test: Different time_varying_media
    mmm14 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dims=("geo",),
        time_varying_media=True,  # Different
    )
    assert mmm1 != mmm14

    # Test: Comparison with non-MMM object
    assert mmm1 != "not an MMM"
    assert mmm1 != 42
    assert mmm1 != None  # noqa: E711

    # Test: With custom HSGP for time-varying effects
    # Use parameterize_from_data to properly initialize HSGP
    X_dummy = np.arange(10)
    hsgp1 = SoftPlusHSGP.parameterize_from_data(X=X_dummy, dims=("date",))
    hsgp2 = SoftPlusHSGP.parameterize_from_data(X=X_dummy, dims=("date",))

    mmm15 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        time_varying_intercept=hsgp1,
    )

    mmm16 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        time_varying_intercept=hsgp2,
    )

    # Should be equal if HSGP configs match
    assert mmm15 == mmm16

    # Test: Different HSGP config (different input data creates different params)
    X_dummy_different = np.arange(20)  # Different length
    hsgp3 = SoftPlusHSGP.parameterize_from_data(X=X_dummy_different, dims=("date",))
    mmm17 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        time_varying_intercept=hsgp3,
    )
    assert mmm15 != mmm17

    # Test: With causal graph configuration
    dag = """
    digraph {
        channel_1 -> sales;
        channel_2 -> sales;
        control_1 -> sales;
    }
    """

    mmm18 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dag=dag,
        treatment_nodes=["channel_1", "channel_2"],
        outcome_node="sales",
    )

    mmm19 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dag=dag,
        treatment_nodes=["channel_1", "channel_2"],
        outcome_node="sales",
    )

    assert mmm18 == mmm19

    # Test: Different DAG
    dag_different = """
    digraph {
        channel_1 -> sales;
    }
    """

    mmm20 = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        dag=dag_different,  # Different
        treatment_nodes=["channel_1"],
        outcome_node="sales",
    )

    assert mmm18 != mmm20


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
    with pytest.raises(ValueError, match=r"date_column 'wrong_column_name' not found"):
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
            dict(
                name="intercept_latent_process_raw_ls_raw",
                kind="WeibullBetaRV",
            ),
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


class TestTimeVaryingConfigFormats:
    """Test time-varying coefficient configuration formats for API harmonization."""

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create minimal sample data for testing."""
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="W")
        rng = np.random.default_rng(12345)
        X = pd.DataFrame(
            {
                "date": dates,
                "channel_1": rng.random(n),
                "channel_2": rng.random(n),
            }
        )
        y = pd.Series(rng.random(n), name="target")
        return X, y

    def test_intercept_tvp_with_hsgp_kwargs_instance(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test time_varying_intercept=True with HSGPKwargs in model_config."""
        X, y = sample_data
        model_config = {
            "intercept_tvp_config": HSGPKwargs(
                m=50, L=None, eta_lam=1.0, ls_mu=5.0, ls_sigma=10.0, cov_func=None
            ),
        }
        mmm = MMM(
            date_column="date",
            target_column="target",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            time_varying_intercept=True,
            model_config=model_config,
        )
        mmm.build_model(X, y)
        assert "intercept_latent_process" in mmm.model.named_vars

    def test_intercept_tvp_with_hsgp_kwargs_dict(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test time_varying_intercept=True with dict in HSGPKwargs format."""
        X, y = sample_data
        model_config = {
            "intercept_tvp_config": {
                "m": 50,
                "L": None,
                "eta_lam": 1.0,
                "ls_mu": 5.0,
                "ls_sigma": 10.0,
                "cov_func": None,
            },
        }
        mmm = MMM(
            date_column="date",
            target_column="target",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            time_varying_intercept=True,
            model_config=model_config,
        )
        mmm.build_model(X, y)
        assert "intercept_latent_process" in mmm.model.named_vars

    def test_media_tvp_with_hsgp_kwargs_instance(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test time_varying_media=True with HSGPKwargs in model_config."""
        X, y = sample_data
        model_config = {
            "media_tvp_config": HSGPKwargs(
                m=50, L=None, eta_lam=1.0, ls_mu=5.0, ls_sigma=10.0, cov_func=None
            ),
        }
        mmm = MMM(
            date_column="date",
            target_column="target",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            time_varying_media=True,
            model_config=model_config,
        )
        mmm.build_model(X, y)
        assert "media_temporal_latent_multiplier" in mmm.model.named_vars


def test_multidimensional_mmm_serializes_and_deserializes_dag_and_nodes(
    single_dim_data, mock_pymc_sample
):
    dag = """
    digraph {
        channel_1 -> y;
        control_1 -> channel_1;
        control_1 -> y;
    }
    """
    treatment_nodes = ["channel_1"]
    outcome_node = "y"

    X, y = single_dim_data
    y = y.rename("y")

    mmm = MMM(
        date_column="date",
        target_column="y",
        channel_columns=["channel_1", "channel_2"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        dag=dag,
        treatment_nodes=treatment_nodes,
        outcome_node=outcome_node,
    )

    mmm.fit(X=X, y=y)

    mmm.save("test_model_multi")
    loaded_mmm = MMM.load("test_model_multi")

    assert loaded_mmm.dag == dag
    assert loaded_mmm.treatment_nodes == treatment_nodes
    assert loaded_mmm.outcome_node == outcome_node

    os.remove("test_model_multi")


def test_multidimensional_mmm_causal_attributes_initialization():
    dag = """
    digraph {
        channel_1 -> target;
        control_1 -> channel_1;
        control_1 -> target;
    }
    """
    treatment_nodes = ["channel_1"]
    outcome_node = "target"

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        dag=dag,
        treatment_nodes=treatment_nodes,
        outcome_node=outcome_node,
    )

    assert mmm.dag == dag
    assert mmm.treatment_nodes == treatment_nodes
    assert mmm.outcome_node == outcome_node


def test_multidimensional_mmm_causal_attributes_default_treatment_nodes():
    dag = """
    digraph {
        channel_1 -> target;
        channel_2 -> target;
        control_1 -> channel_1;
        control_1 -> target;
    }
    """
    outcome_node = "target"

    with pytest.warns(
        UserWarning, match=r"No treatment nodes provided, using channel columns"
    ):
        mmm = MMM(
            date_column="date",
            target_column="target",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            dag=dag,
            outcome_node=outcome_node,
        )

    assert mmm.treatment_nodes == ["channel_1", "channel_2"]
    assert mmm.outcome_node == "target"


def test_multidimensional_mmm_adjustment_set_updates_control_columns():
    dag = """
    digraph {
        channel_1 -> target;
        control_1 -> channel_1;
        control_1 -> target;
    }
    """
    treatment_nodes = ["channel_1"]
    outcome_node = "target"

    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        dag=dag,
        treatment_nodes=treatment_nodes,
        outcome_node=outcome_node,
    )

    assert mmm.control_columns == ["control_1"]


def test_multidimensional_mmm_missing_dag_does_not_initialize_causal_graph():
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    assert mmm.dag is None
    assert not hasattr(mmm, "causal_graphical_model")


def test_multidimensional_mmm_only_dag_provided_does_not_initialize_graph():
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        dag="digraph {channel_1 -> target;}",
    )

    assert mmm.treatment_nodes is None
    assert mmm.outcome_node is None
    assert not hasattr(mmm, "causal_graphical_model")


def test_default_model_config_dims_include_self_dims():
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    cfg = mmm.default_model_config

    # Keys from MMM.default_model_config we want to validate here
    keys_to_check = [
        "intercept",
        "likelihood",
        "gamma_control",
        "gamma_fourier",
    ]

    for key in keys_to_check:
        assert key in cfg, f"{key} missing in default_model_config"
        prior = cfg[key]

        # Prior may be a distribution or a container (e.g., likelihood with nested sigma prior)
        # In both cases, the top-level prior should expose dims that at least include model dims
        assert hasattr(prior, "dims"), f"{key} prior does not have dims attribute"

        dims = prior.dims if isinstance(prior.dims, tuple) else (prior.dims,)
        # Ensure all model dims are present (allowing additional dims like control/fourier_mode)
        for d in mmm.dims:
            assert d in dims, f"{key} dims {dims} must include model dims {mmm.dims}"


def test_calibration_spend_reindexing_in_posterior_predictive(
    multi_dim_data, mock_pymc_sample
):
    """Test that calibration spend data is properly reindexed during posterior predictive sampling.

    This test covers the previously uncovered lines in _set_xarray_data that handle
    reindexing and dtype conversion of calibration spend data.
    """
    X, y = multi_dim_data

    # Create MMM model
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    # Build the model
    mmm.build_model(X, y)

    # Create spend data with same structure as X
    spend_df = X.copy()
    # Add some variation to make it different from channel data
    for col in ["channel_1", "channel_2", "channel_3"]:
        spend_df[col] = spend_df[col] * 1.5

    # Create calibration data
    countries = mmm.model.coords["country"]
    channels = ["channel_1", "channel_2"]
    calibration_df = pd.DataFrame(
        {
            "country": [countries[0], countries[1]],
            "channel": [channels[0], channels[1]],
            "cost_per_target": [30.0, 45.0],
            "sigma": [2.0, 3.0],
        }
    )

    # Add original scale contribution variable first (required before calibration)
    mmm.add_original_scale_contribution_variable(var=["channel_contribution"])

    # Add calibration
    mmm.add_cost_per_target_calibration(
        data=spend_df,
        calibration_data=calibration_df,
        name_prefix="cpt_calibration",
    )

    # Fit the model
    mmm.fit(X, y, draws=50, tune=25, chains=1)

    # Create new data with different dates for posterior predictive
    # This will trigger the reindexing logic in _set_xarray_data
    X_new = X.copy()

    # Shift dates to future
    date_shift = pd.Timedelta(days=14)
    X_new["date"] = X_new["date"] + date_shift

    # Also test with some missing countries to ensure fill_value=0 works
    # Remove one country to test reindexing with missing dimensions
    X_new = X_new[X_new["country"] != countries[-1]]

    # Add some NaN values to test fillna functionality
    X_new.loc[X_new.index[0], "channel_1"] = np.nan

    # Sample posterior predictive - this will call _set_xarray_data internally
    # and execute the uncovered lines for spend data reindexing
    idata_pred = mmm.sample_posterior_predictive(
        X_new,
        extend_idata=False,
        random_seed=42,
    )

    # Verify the posterior predictive was successful
    # When extend_idata=False, it returns an xarray Dataset directly
    assert "y" in idata_pred

    # Verify that sampling succeeds without explicit spend data containers
    assert "channel_contribution_original_scale" in mmm.model.named_vars

    # Additional test: verify with include_last_observations=True
    # to test a different code path
    X_future = X.copy()
    X_future["date"] = X_future["date"] + pd.Timedelta(days=30)  # Non-overlapping dates

    idata_pred_with_last = mmm.sample_posterior_predictive(
        X_future,
        include_last_observations=True,
        extend_idata=False,
        random_seed=42,
    )

    # When extend_idata=False, it returns an xarray Dataset directly
    assert "y" in idata_pred_with_last


def test_calibration_spend_with_different_dtypes(multi_dim_data, mock_pymc_sample):
    """Test that calibration spend data dtype conversion works correctly.

    This specifically tests the dtype conversion logic in the uncovered lines.
    """
    X, y = multi_dim_data

    # Create MMM model
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    # Build the model
    mmm.build_model(X, y)

    # Create spend data with float32 dtype (different from model's float64)
    spend_df = X.copy()
    for col in ["channel_1", "channel_2", "channel_3"]:
        spend_df[col] = spend_df[col].astype(np.float32) * 1.5

    # Create calibration data
    countries = mmm.model.coords["country"]
    calibration_df = pd.DataFrame(
        {
            "country": [countries[0], countries[1], countries[2]],
            "channel": ["channel_1", "channel_2", "channel_3"],
            "cost_per_target": [25.0, 35.0, 40.0],
            "sigma": [1.5, 2.5, 2.0],
        }
    )

    # Add original scale contribution variable first (required before calibration)
    mmm.add_original_scale_contribution_variable(var=["channel_contribution"])

    # Add calibration
    mmm.add_cost_per_target_calibration(
        data=spend_df,
        calibration_data=calibration_df,
        name_prefix="cpt_calibration",
    )

    # Fit the model
    mmm.fit(X, y, draws=50, tune=25, chains=1)

    # Create new data with float32 to test dtype conversion
    X_new = X.copy()
    X_new["date"] = X_new["date"] + pd.Timedelta(days=7)
    for col in ["channel_1", "channel_2", "channel_3"]:
        X_new[col] = X_new[col].astype(np.float32)

    # Sample posterior predictive
    idata_pred = mmm.sample_posterior_predictive(
        X_new,
        extend_idata=False,
        random_seed=42,
    )

    # Verify success
    # When extend_idata=False, it returns an xarray Dataset directly
    assert "y" in idata_pred


def test_calibration_duplicate_name_error(multi_dim_data, mock_pymc_sample):
    """Test that attempting to re-register calibration potentials raises a duplicate name error."""
    X, y = multi_dim_data

    # Create MMM model
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    # Build the model
    mmm.build_model(X, y)

    # Create spend data
    spend_df = X.copy()
    for col in ["channel_1", "channel_2", "channel_3"]:
        spend_df[col] = spend_df[col] * 1.5

    # Create calibration data
    countries = mmm.model.coords["country"]
    calibration_df = pd.DataFrame(
        {
            "country": [countries[0], countries[1], countries[2]],
            "channel": ["channel_1", "channel_2", "channel_3"],
            "cost_per_target": [25.0, 35.0, 40.0],
            "sigma": [1.5, 2.5, 2.0],
        }
    )

    # Add original scale contribution variable first (required before calibration)
    mmm.add_original_scale_contribution_variable(var=["channel_contribution"])

    # Add calibration first time
    mmm.add_cost_per_target_calibration(
        data=spend_df,
        calibration_data=calibration_df,
        name_prefix="cpt_calibration",
    )

    # Attempting to re-register the calibration potentials should raise a duplicate name error
    with pytest.raises(
        ValueError, match="Variable name cpt_calibration already exists"
    ):
        mmm.add_cost_per_target_calibration(
            data=spend_df,
            calibration_data=calibration_df,
            name_prefix="cpt_calibration",
        )


def test_calibration_shape_mismatch_error(multi_dim_data, mock_pymc_sample):
    """Test that spend data with mismatched shape raises a ValueError."""
    X, y = multi_dim_data

    # Create MMM model
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    # Build the model
    mmm.build_model(X, y)

    # Add original scale contribution variable first (required before calibration)
    mmm.add_original_scale_contribution_variable(var=["channel_contribution"])

    # Create spend data with different shape (remove some countries)
    spend_df = X.copy()
    # Remove one country to create shape mismatch
    spend_df = spend_df[spend_df["country"] != "Chile"].copy()
    for col in ["channel_1", "channel_2", "channel_3"]:
        spend_df[col] = spend_df[col] * 1.5
    # print unique countries in spend_df
    print(spend_df["country"].unique())

    # Create calibration data
    countries = mmm.model.coords["country"]
    # print unique countries in countries
    print(countries)
    calibration_df = pd.DataFrame(
        {
            "country": [countries[0], countries[1], countries[2]],
            "channel": ["channel_1", "channel_2", "channel_3"],
            "cost_per_target": [25.0, 35.0, 40.0],
            "sigma": [1.5, 2.5, 2.0],
        }
    )

    # This should raise a shape mismatch error
    with pytest.raises(ValueError, match="shape does not match"):
        mmm.add_cost_per_target_calibration(
            data=spend_df,
            calibration_data=calibration_df,
            name_prefix="cpt_calibration",
        )


def test_calibration_coordinate_label_mismatch_error(multi_dim_data, mock_pymc_sample):
    """Test that spend data with mismatched coord labels raises a ValueError.

    Keeps the shape identical to the model but replaces one coordinate label with
    a new label not present in model coords, so the label-equality guard triggers.
    """
    X, y = multi_dim_data

    # Create MMM model
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=["channel_1", "channel_2", "channel_3"],
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )

    # Build the model
    mmm.build_model(X, y)

    # Add original scale contribution variable first (required before calibration)
    mmm.add_original_scale_contribution_variable(var=["channel_contribution"])

    # Create spend data with the same shape but mismatched coordinate labels
    spend_df = X.copy()
    model_countries = list(mmm.model.coords["country"])
    # Replace all rows of one existing country with a new label not in model coords
    wrong_label = str(model_countries[-1]) + "_WRONG"
    spend_df.loc[spend_df["country"] == model_countries[-1], "country"] = wrong_label
    for col in ["channel_1", "channel_2", "channel_3"]:
        spend_df[col] = spend_df[col] * 1.5

    # Calibration data uses the original model coords
    calibration_df = pd.DataFrame(
        {
            "country": [model_countries[0], model_countries[1], model_countries[2]],
            "channel": ["channel_1", "channel_2", "channel_3"],
            "cost_per_target": [25.0, 35.0, 40.0],
            "sigma": [1.5, 2.5, 2.0],
        }
    )

    # Expect label mismatch error (not just shape mismatch)
    with pytest.raises(
        ValueError,
        match=r"Spend data coordinates for dim 'country' do not match model coords:",
    ):
        mmm.add_cost_per_target_calibration(
            data=spend_df,
            calibration_data=calibration_df,
            name_prefix="cpt_calibration",
        )


class TestAddOriginalScaleContributionVariable:
    """Tests for add_original_scale_contribution_variable method."""

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create sample data for testing."""
        rng = np.random.default_rng(42)
        n_dates = 20
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="W")
        X = pd.DataFrame(
            {
                "date": dates,
                "x1": rng.uniform(0.1, 1.0, n_dates),
                "x2": rng.uniform(0.1, 1.0, n_dates),
            }
        )
        y = pd.Series(rng.uniform(100, 500, n_dates), name="y")
        return X, y

    def test_channel_contribution(self, sample_data) -> None:
        """Test adding channel_contribution_original_scale."""
        X, y = sample_data
        mmm = MMM(
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            date_column="date",
            channel_columns=["x1", "x2"],
            target_column="y",
        )
        mmm.build_model(X, y)
        mmm.add_original_scale_contribution_variable(var=["channel_contribution"])

        assert "channel_contribution_original_scale" in mmm.model.named_vars
        dims = mmm.model.named_vars_to_dims["channel_contribution_original_scale"]
        assert dims == ("date", "channel")

    def test_fourier_contribution(self, sample_data) -> None:
        """Test adding fourier_contribution_original_scale."""
        X, y = sample_data
        mmm = MMM(
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            date_column="date",
            channel_columns=["x1", "x2"],
            target_column="y",
            yearly_seasonality=3,
        )
        mmm.build_model(X, y)
        mmm.add_original_scale_contribution_variable(var=["fourier_contribution"])

        assert "fourier_contribution_original_scale" in mmm.model.named_vars
        dims = mmm.model.named_vars_to_dims["fourier_contribution_original_scale"]
        assert dims == ("date", "fourier_mode")

    def test_multiple_contributions(self, sample_data) -> None:
        """Test adding multiple contribution variables at once."""
        X, y = sample_data
        mmm = MMM(
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            date_column="date",
            channel_columns=["x1", "x2"],
            target_column="y",
            yearly_seasonality=3,
        )
        mmm.build_model(X, y)
        mmm.add_original_scale_contribution_variable(
            var=["channel_contribution", "fourier_contribution"]
        )

        assert "channel_contribution_original_scale" in mmm.model.named_vars
        assert "fourier_contribution_original_scale" in mmm.model.named_vars

        channel_dims = mmm.model.named_vars_to_dims[
            "channel_contribution_original_scale"
        ]
        fourier_dims = mmm.model.named_vars_to_dims[
            "fourier_contribution_original_scale"
        ]
        assert channel_dims == ("date", "channel")
        assert fourier_dims == ("date", "fourier_mode")

    def test_yearly_seasonality_contribution(self, sample_data) -> None:
        """Test adding yearly_seasonality_contribution_original_scale."""
        X, y = sample_data
        mmm = MMM(
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            date_column="date",
            channel_columns=["x1", "x2"],
            target_column="y",
            yearly_seasonality=3,
        )
        mmm.build_model(X, y)
        mmm.add_original_scale_contribution_variable(
            var=["yearly_seasonality_contribution"]
        )

        assert "yearly_seasonality_contribution_original_scale" in mmm.model.named_vars
        dims = mmm.model.named_vars_to_dims[
            "yearly_seasonality_contribution_original_scale"
        ]
        assert dims == ("date",)

    def test_intercept_contribution_with_tvp(self, sample_data) -> None:
        """Test adding intercept_contribution_original_scale with time-varying intercept."""
        X, y = sample_data
        mmm = MMM(
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            date_column="date",
            channel_columns=["x1", "x2"],
            target_column="y",
            time_varying_intercept=True,
        )
        mmm.build_model(X, y)
        mmm.add_original_scale_contribution_variable(var=["intercept_contribution"])

        assert "intercept_contribution_original_scale" in mmm.model.named_vars
        dims = mmm.model.named_vars_to_dims["intercept_contribution_original_scale"]
        assert dims == ("date",)

    def test_with_geo_dims(self, multi_dim_data) -> None:
        """Test adding contribution variables with geo dimensions."""
        X, y = multi_dim_data
        mmm = MMM(
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            date_column="date",
            channel_columns=["channel_1", "channel_2", "channel_3"],
            target_column="target",
            dims=("country",),
            yearly_seasonality=2,
        )
        mmm.build_model(X, y)
        mmm.add_original_scale_contribution_variable(
            var=["channel_contribution", "fourier_contribution"]
        )

        channel_dims = mmm.model.named_vars_to_dims[
            "channel_contribution_original_scale"
        ]
        fourier_dims = mmm.model.named_vars_to_dims[
            "fourier_contribution_original_scale"
        ]
        assert channel_dims == ("date", "country", "channel")
        assert fourier_dims == ("date", "country", "fourier_mode")


class TestDataProperty:
    """Tests for the MMM.data property."""

    def test_data_property_raises_when_no_idata(self) -> None:
        """Test that accessing .data before fitting raises ValueError."""
        mmm = MMM(
            date_column="date",
            channel_columns=["C1", "C2"],
            target_column="target",
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )

        with pytest.raises(ValueError, match="idata does not exist"):
            _ = mmm.data

    def test_data_property_returns_wrapper(self, fit_mmm) -> None:
        """Test that .data returns MMMIDataWrapper with correct schema."""
        wrapper = fit_mmm.data

        assert isinstance(wrapper, MMMIDataWrapper)
        assert wrapper.idata is fit_mmm.idata
        assert wrapper.schema is not None
        # fit_mmm has dims=("country",), no controls, no seasonality, no time-varying
        assert wrapper.schema.custom_dims == ("country",)

    def test_data_property_without_custom_dims(
        self, single_dim_data, mock_pymc_sample
    ) -> None:
        """Test schema receives empty tuple when dims is empty."""
        X, y = single_dim_data
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2", "channel_3"],
            target_column="target",
            dims=(),
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.fit(X, y)

        wrapper = mmm.data
        assert isinstance(wrapper, MMMIDataWrapper)
        assert wrapper.schema.custom_dims == ()

    def test_data_property_with_controls(
        self, single_dim_data, mock_pymc_sample
    ) -> None:
        """Test schema receives has_controls=True when control_columns set."""
        X, y = single_dim_data
        # Add a control column
        X = X.copy()
        X["control_var"] = np.random.rand(len(X))

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2", "channel_3"],
            control_columns=["control_var"],
            target_column="target",
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.fit(X, y)

        wrapper = mmm.data
        assert isinstance(wrapper, MMMIDataWrapper)
        # Verify control_data variable exists in schema
        assert "control_data_" in wrapper.schema.groups["constant_data"].variables

    def test_data_property_with_seasonality(
        self, single_dim_data, mock_pymc_sample
    ) -> None:
        """Test schema receives has_seasonality=True when yearly_seasonality set."""
        X, y = single_dim_data
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2", "channel_3"],
            target_column="target",
            yearly_seasonality=2,
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.fit(X, y)

        wrapper = mmm.data
        assert isinstance(wrapper, MMMIDataWrapper)
        # Verify dayofyear variable exists in schema (indicator of seasonality)
        assert "dayofyear" in wrapper.schema.groups["constant_data"].variables

    def test_data_property_with_time_varying_intercept(
        self, single_dim_data, mock_pymc_sample
    ) -> None:
        """Test schema receives time_varying=True when time_varying_intercept=True."""
        X, y = single_dim_data
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2", "channel_3"],
            target_column="target",
            time_varying_intercept=True,
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.fit(X, y)

        wrapper = mmm.data
        assert isinstance(wrapper, MMMIDataWrapper)
        # Verify time_index variable exists in schema (indicator of time-varying)
        assert "time_index" in wrapper.schema.groups["constant_data"].variables

    def test_data_property_with_time_varying_media(
        self, single_dim_data, mock_pymc_sample
    ) -> None:
        """Test schema receives time_varying=True when time_varying_media=True."""
        X, y = single_dim_data
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2", "channel_3"],
            target_column="target",
            time_varying_media=True,
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.fit(X, y)

        wrapper = mmm.data
        assert isinstance(wrapper, MMMIDataWrapper)
        # Verify time_index variable exists in schema (indicator of time-varying)
        assert "time_index" in wrapper.schema.groups["constant_data"].variables
