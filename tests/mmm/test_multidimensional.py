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
from pymc.model_graph import fast_eval
from pytensor.tensor.basic import TensorVariable

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.events import EventEffect, GaussianBasis
from pymc_marketing.mmm.multidimensional import (
    MMM,
    create_event_mu_effect,
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

    return X, df.set_index(["date"])["target"].copy()


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

    actual = mmm.model["target_scaled"].eval()
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
    effect = create_event_mu_effect(df_events, prefix="holiday", effect=event_effect)

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

    actual = mmm.model["target_scaled"].eval()
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
