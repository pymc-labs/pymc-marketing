#   Copyright 2024 The PyMC Labs Developers
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
import pytensor.tensor as pt
import pytest
from pymc.model_graph import fast_eval
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from pymc_marketing.mmm.components.saturation import (
    HillSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
    SaturationTransformation,
)
from pymc_marketing.mmm.lift_test import (
    MissingLiftTestError,
    NonMonotonicLiftError,
    add_lift_measurements_to_likelihood,
    add_lift_measurements_to_likelihood_from_saturation,
    check_increasing_assumption,
    create_time_varying_saturation,
    index_variable,
    indices_from_lift_tests,
    lift_test_indices,
    scale_channel_lift_measurements,
    scale_target_for_lift_measurements,
)
from pymc_marketing.mmm.transformers import logistic_saturation, michaelis_menten


@pytest.fixture(scope="function")
def df_lift_tests() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-03"],
            "channel": ["organic", "paid"],
        }
    )


def test_lift_test_indices(df_lift_tests) -> None:
    coords = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "channel": ["organic", "paid", "social"],
    }
    model = pm.Model(coords=coords)

    indices = lift_test_indices(df_lift_tests, model)

    assert indices["date"].tolist() == [0, 2]
    assert indices["channel"].tolist() == [0, 1]


def test_lift_test_indices_additional_model_coords(df_lift_tests) -> None:
    """Models will likely have additional coords that are not for test."""
    coords = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "channel": ["organic", "paid", "social"],
    }
    model = pm.Model(coords=coords)

    indices = lift_test_indices(df_lift_tests[["date"]], model)

    assert indices["date"].tolist() == [0, 2]


def test_lift_test_indices_another_dim(df_lift_tests) -> None:
    coords = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "channel": ["organic", "paid", "social"],
        "brand": ["brand1", "brand2"],
    }

    df_additional_brand = df_lift_tests.assign(brand=["brand1", "brand1"])

    model = pm.Model(coords=coords)

    indices = lift_test_indices(df_additional_brand, model)

    assert indices["date"].tolist() == [0, 2]
    assert indices["channel"].tolist() == [0, 1]
    assert indices["brand"].tolist() == [0, 0]


@pytest.mark.parametrize(
    "dates",
    [
        pd.date_range("2023-01-01", periods=3, freq="D"),
        pd.date_range("2023-01-01", periods=3, freq="W"),
        pd.date_range("2023-01-01", periods=3, freq="W-MON"),
        pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
        pd.date_range("2023-01-01", periods=3, freq="D")
        .to_numpy()
        .astype("datetime64"),
        pd.date_range("2023-01-01", periods=3, freq="D")
        .to_numpy()
        .astype("datetime64[D]"),
        pd.date_range("2023-01-01", periods=3, freq="D")
        .to_numpy()
        .astype("datetime64[s]"),
    ],
)
def test_lift_test_indices_with_dates(df_lift_tests, dates) -> None:
    coords = {
        "actual_date": dates,
        "channel": ["organic", "paid", "social"],
    }
    model = pm.Model(coords=coords)

    df_actual_dates = df_lift_tests.assign(
        actual_date=dates[[0, 2]],
    )

    indices = lift_test_indices(
        df_actual_dates.loc[:, ["actual_date", "channel"]], model
    )

    assert indices["actual_date"].tolist() == [0, 2]
    assert indices["channel"].tolist() == [0, 1]


def test_lift_test_missing_coords(df_lift_tests) -> None:
    with pytest.raises(KeyError):
        df_lift_tests.pipe(lift_test_indices, model=pm.Model())


def test_lift_test_additional_df_column(df_lift_tests) -> None:
    coords = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "channel": ["organic", "paid", "social"],
    }
    df_additional_col = df_lift_tests.assign(additional_column=[1, 2])

    with pytest.raises(KeyError):
        lift_test_indices(df_additional_col, model=pm.Model(coords=coords))


def test_lift_tests_missing(df_lift_tests) -> None:
    coords = {
        "date": ["2020-01-03", "2020-01-04", "2020-01-05"],
        "channel": ["organic", "paid", "social"],
    }
    model = pm.Model(coords=coords)
    with pytest.raises(MissingLiftTestError) as err:
        lift_test_indices(df_lift_tests, model)

    assert err.value.missing_values.tolist() == [0]


@pytest.fixture
def df_lift_tests_with_numerics(df_lift_tests) -> pd.DataFrame:
    return df_lift_tests.assign(
        x=100,
        delta_x=50,
        delta_y=0.1,
        sigma=0.15,
    )


def add_menten_empirical_lift_measurements_to_likelihood(
    df_lift_test: pd.DataFrame,
    alpha_name: str,
    lam_name: str,
    dist=pm.Gamma,
    model: pm.Model | None = None,
    name: str = "lift_measurements",
) -> None:
    """Ported over to test."""
    variable_mapping = {
        "alpha": alpha_name,
        "lam": lam_name,
    }

    add_lift_measurements_to_likelihood(
        df_lift_test,
        variable_mapping,
        saturation_function=michaelis_menten,
        model=model,
        dist=dist,
        name=name,
    )


def add_logistic_empirical_lift_measurements_to_likelihood(
    df_lift_test: pd.DataFrame,
    lam_name: str,
    beta_name: str,
    dist: pm.Distribution = pm.Gamma,
    model: pm.Model | None = None,
    name: str = "lift_measurements",
) -> None:
    """Ported over to test."""
    variable_mapping = {
        "lam": lam_name,
        "beta": beta_name,
    }

    def saturation_function(x, beta, lam):
        return beta * logistic_saturation(x, lam)

    add_lift_measurements_to_likelihood(
        df_lift_test,
        variable_mapping,
        saturation_function=saturation_function,
        model=model,
        dist=dist,
        name=name,
    )


@pytest.mark.parametrize("dist", [pm.Normal, pm.Gamma])
@pytest.mark.parametrize("alpha_dims", ["channel", ("date", "channel"), "date"])
def test_add_menten_empirical_lift_measurements_to_likelihood(
    df_lift_tests_with_numerics,
    dist,
    alpha_dims,
) -> None:
    coords = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "channel": ["organic", "paid", "social"],
    }
    with pm.Model(coords=coords) as model:
        pm.HalfNormal("alpha", dims=alpha_dims)
        pm.HalfNormal("lam", dims="channel")

        add_menten_empirical_lift_measurements_to_likelihood(
            df_lift_tests_with_numerics,
            alpha_name="alpha",
            lam_name="lam",
            dist=dist,
        )

    assert "lift_measurements" in model


@pytest.mark.parametrize("dist", [pm.Normal, pm.Gamma])
@pytest.mark.parametrize("lam_dims", ["channel", ("date", "channel"), "date"])
def test_add_logistic_empirical_lift_measurements_to_likelihood(
    df_lift_tests_with_numerics,
    dist,
    lam_dims,
) -> None:
    coords = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "channel": ["organic", "paid", "social"],
    }
    with pm.Model(coords=coords) as model:
        pm.HalfNormal("lam", dims=lam_dims)
        pm.HalfNormal("beta", dims="channel")

        add_logistic_empirical_lift_measurements_to_likelihood(
            df_lift_tests_with_numerics,
            lam_name="lam",
            beta_name="beta",
            dist=dist,
        )

    assert "lift_measurements" in model


def test_add_menten_empirical_lift_measurements_explicit_coords(
    df_lift_tests_with_numerics,
) -> None:
    """Test that the function works with 2D and 3D coords."""
    df_lift_tests_with_numerics = df_lift_tests_with_numerics.assign(
        brand=["brand1", "brand1"]
    )
    coords = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "channel": ["organic", "paid", "social"],
        "brand": ["brand1", "brand2"],
    }
    with pm.Model(coords=coords) as model:
        pm.HalfNormal("alpha", dims=("date", "channel", "brand"))
        pm.HalfNormal("lam", dims=("channel", "brand"))

        add_menten_empirical_lift_measurements_to_likelihood(
            df_lift_tests_with_numerics,
            alpha_name="alpha",
            lam_name="lam",
            dist=pm.Gamma,
        )

    assert "lift_measurements" in model


@pytest.fixture
def create_mock_idata():
    def _create_mock_idata(channels) -> az.InferenceData:
        rng = np.random.default_rng(42)
        n_chains = 1
        n_samples = 100
        n_channels = len(channels)
        datadict = {
            "lam": np.abs(rng.normal(size=(n_chains, n_samples, n_channels))),
            "alpha": np.abs(rng.normal(size=(n_chains, n_samples, n_channels))),
        }
        coords = {"channel": channels}
        dims = {"lam": ["channel"], "alpha": ["channel"]}

        return az.convert_to_inference_data(datadict, coords=coords, dims=dims)

    return _create_mock_idata


def test_add_lift_measurements_before_new_data(
    df_lift_tests_with_numerics, create_mock_idata
) -> None:
    """Adding lift measurements doesn't affect posterior predictive with new dates."""
    channels = ["organic", "paid", "social"]
    coords = {
        "channel": channels,
    }
    coords_mutable = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
    }
    with pm.Model(coords=coords, coords_mutable=coords_mutable) as model:
        alpha = pm.HalfNormal("alpha", dims="channel")
        lam = pm.HalfNormal("lam", dims="channel")

        X = pm.Data("X", np.ones((3, 3)), dims=("date", "channel"), mutable=True)
        pm.Deterministic(
            "random_operation",
            X + alpha + lam,
            dims=("date", "channel"),
        )

        add_menten_empirical_lift_measurements_to_likelihood(
            df_lift_tests_with_numerics,
            alpha_name="alpha",
            lam_name="lam",
            dist=pm.Normal,
        )

        idata = create_mock_idata(channels=channels)

    new_dates = ["2020-01-04", "2020-01-05"]
    new_coords = {
        "date": new_dates,
    }
    with model:
        new_shape = (2, 3)
        new_data = np.ones(new_shape)
        pm.set_data(new_data={"X": new_data}, coords=new_coords)

        posterior_predictive = pm.sample_posterior_predictive(
            idata,
            var_names=["random_operation"],
        )

    assert (
        posterior_predictive.posterior_predictive.random_operation.mean(
            ("chain", "draw")
        ).shape
        == new_shape
    )


@pytest.fixture
def model_with_3_dims() -> pm.Model:
    coords = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "channel": ["organic", "paid", "social"],
        "brand": ["brand1", "brand2"],
    }
    model = pm.Model(coords=coords)
    with model:
        alpha = pm.HalfNormal("alpha", dims="date")
        pm.HalfNormal("beta", dims="channel")
        pm.HalfNormal("gamma", dims=("date", "channel"))
        pm.HalfNormal("delta", dims=("date", "channel", "brand"))

        pm.Deterministic(
            "alpha_modified",
            alpha + pt.ones_like(alpha),
            dims="date",
        )

    return model


@pytest.mark.parametrize(
    "var_names, expected_unique",
    [
        (["alpha"], ["date"]),
        (["alpha_modified"], ["date"]),
        (["beta"], ["channel"]),
        (["alpha", "beta"], ["date", "channel"]),
        (["alpha_modified", "beta"], ["date", "channel"]),
        (["alpha", "beta", "gamma"], ["date", "channel"]),
    ],
)
def test_indices_from_lift_tests(
    model_with_3_dims,
    df_lift_tests_with_numerics,
    var_names,
    expected_unique,
) -> None:
    indices = indices_from_lift_tests(
        df_lift_tests_with_numerics, model_with_3_dims, var_names
    )

    assert len(indices) == len(expected_unique)
    for expected in expected_unique:
        assert expected in indices


def test_indices_from_lift_tests_missing_column(
    df_lift_tests_with_numerics, model_with_3_dims
) -> None:
    with pytest.raises(KeyError, match="The required coordinates are"):
        indices_from_lift_tests(
            df_lift_tests_with_numerics, model_with_3_dims, ["delta"]
        )


@pytest.mark.parametrize(
    "var_dims, var_data, indices, expected",
    [
        (["channel"], [0, 1, 2, 3, 4, 5, 6, 7, 8], {"channel": [0, 1, 2]}, [0, 1, 2]),
        (
            ["channel", "date"],
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
            {"channel": [0, 1, 2], "date": [0, 1, 2]},
            [0, 4, 8],
        ),
        (
            ["channel", "date", "brand"],
            [
                [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
            ],
            {"channel": [0, 1, 2], "date": [0, 1, 2], "brand": [0, 1, 2]},
            [0, 13, 26],
        ),
    ],
)
def test_index_variable(var_dims, var_data, indices, expected) -> None:
    var = pt.as_tensor_variable(var_data)
    result = index_variable(var_dims=var_dims, var=var, indices=indices)

    np.testing.assert_allclose(
        fast_eval(result),
        expected,
    )


@pytest.fixture
def mock_channel_pipeline() -> Pipeline:
    pipeline = Pipeline(steps=[("scaler", MaxAbsScaler())])

    max_value = np.array([1, 2, 3])
    n_channels = len(max_value)
    pipeline.fit(np.ones((5, n_channels)) * max_value)

    return pipeline


def test_scale_channel_lift_measurements(mock_channel_pipeline) -> None:
    df_lift_test = pd.DataFrame(
        {
            "channel": ["organic", "organic", "social"],
            "x": [1, 2, 3],
        }
    ).assign(delta_x=1)
    channel_columns = ["organic", "paid", "social"]

    result = scale_channel_lift_measurements(
        df_lift_test=df_lift_test,
        channel_col="channel",
        channel_columns=channel_columns,
        transform=mock_channel_pipeline.transform,
    )

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {
                "channel": ["organic", "organic", "social"],
                "x": [1.0, 2.0, 1.0],
                "delta_x": [1.0, 1.0, 1 / 3],
            }
        ),
    )


@pytest.fixture
def mock_target_pipeline() -> Pipeline:
    pipeline = Pipeline(steps=[("scaler", MaxAbsScaler())])

    max_value = 3

    pipeline.fit(np.ones((5, 1)) * max_value)

    return pipeline


def test_scale_target_for_lift_measurements(mock_target_pipeline) -> None:
    target = pd.Series([0, 3, 6, 9])

    result = scale_target_for_lift_measurements(
        target=target,
        transform=mock_target_pipeline.transform,
    )

    pd.testing.assert_series_equal(
        result,
        pd.Series([0, 1, 2, 3], dtype="float64"),
    )


def test_works_with_negative_delta(df_lift_tests_with_numerics) -> None:
    df_lift_tests_with_numerics_negative = df_lift_tests_with_numerics.assign(
        delta_x=lambda row: row["delta_x"] * -1,
        delta_y=lambda row: row["delta_y"] * -1,
    )

    alpha_dims = "date"
    dist = pm.Gamma

    coords = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "channel": ["organic", "paid", "social"],
    }
    with pm.Model(coords=coords) as model:
        pm.HalfNormal("alpha", dims=alpha_dims)
        pm.HalfNormal("lam", dims="channel")

        add_menten_empirical_lift_measurements_to_likelihood(
            df_lift_tests_with_numerics_negative,
            alpha_name="alpha",
            lam_name="lam",
            dist=dist,
        )

    assert "lift_measurements" in model

    try:
        with model:
            pm.sample(draws=10, tune=10)
    except pm.SamplingError:
        pytest.fail("Negative delta values caused a sampling error.")


def test_check_increasing_assumption() -> None:
    df = pd.DataFrame(
        {
            "delta_x": [1, 2, 3],
            "delta_y": [1, -2, 3],
        }
    )

    with pytest.raises(NonMonotonicLiftError):
        check_increasing_assumption(df)


def saturation_functions() -> list[SaturationTransformation]:
    transformations = [
        LogisticSaturation(),
        MichaelisMentenSaturation(),
        HillSaturation(),
    ]
    for transformation in transformations:
        for config in transformation.function_priors.values():
            config["dims"] = "channel"

    return transformations


@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
def test_create_time_varying_saturation_scales(saturation) -> None:
    kwargs = {name: 0.5 for name in saturation.default_priors.keys()}

    xx = np.linspace(0, 1, 20)
    yy = saturation.function(xx, **kwargs)
    if isinstance(yy, pt.TensorVariable):
        yy = fast_eval(yy)

    MULTIPLIER = 3.5
    function, _ = create_time_varying_saturation(saturation, "time_varying_name")
    yy_tv = function(xx, **kwargs, time_varying=MULTIPLIER)
    if isinstance(yy_tv, pt.TensorVariable):
        yy_tv = fast_eval(yy_tv)

    np.testing.assert_allclose(MULTIPLIER * yy, yy_tv)


@pytest.mark.parametrize(
    "time_varying_variable_name",
    [
        "time_varying",
        "media_latent_multiplier",
    ],
)
@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
def test_create_time_varying_saturation_correct_variable_mapping(
    time_varying_variable_name, saturation
) -> None:
    _, variable_mapping = create_time_varying_saturation(
        saturation, time_varying_variable_name
    )

    assert variable_mapping == {
        **saturation.variable_mapping,
        "time_varying": time_varying_variable_name,
    }


@pytest.mark.parametrize("time_varying_var_name", ["tvp", None])
@pytest.mark.parametrize("saturation", saturation_functions())
def test_add_lift_measurements_to_likelihood_from_saturation(
    time_varying_var_name, saturation
) -> None:
    df_lift_tests = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "delta_x": [0.1, 0.2, 0.3],
            "sigma": [0.1, 0.2, 0.3],
            "delta_y": [0.1, 0.2, 0.3],
            "channel": ["organic", "paid", "social"],
        }
    )

    if time_varying_var_name is not None:
        df_lift_tests["date"] = [1, 2, 3]

    coords = {
        "channel": ["organic", "paid", "social"],
        "date": [1, 2, 3, 4],
    }
    with pm.Model(coords=coords) as model:
        saturation._create_distributions(dims="channel")

        if time_varying_var_name is not None:
            tvp_raw = pm.Normal("tvp_raw", dims="date")
            pm.Deterministic(time_varying_var_name, pt.exp(tvp_raw), dims="date")

    assert "lift_measurements" not in model

    add_lift_measurements_to_likelihood_from_saturation(
        df_lift_tests,
        saturation=saturation,
        time_varying_var_name=time_varying_var_name,
        model=model,
    )

    assert "lift_measurements" in model


def test_tvp_needs_date_in_lift_tests() -> None:
    df_lift_tests = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "delta_x": [0.1, 0.2, 0.3],
            "sigma": [0.1, 0.2, 0.3],
            "delta_y": [0.1, 0.2, 0.3],
            "channel": ["organic", "paid", "social"],
        }
    )

    saturation = saturation_functions()[0]

    time_varying_var_name = "tvp"
    coords = {
        "channel": ["organic", "paid", "social"],
        "date": [1, 2, 3, 4],
    }
    with pm.Model(coords=coords) as model:
        saturation._create_distributions(dims="channel")

        tvp_raw = pm.Normal("tvp_raw", dims="date")
        pm.Deterministic(time_varying_var_name, pt.exp(tvp_raw), dims="date")

    assert "lift_measurements" not in model

    with pytest.raises(KeyError, match="The required coordinates are"):
        add_lift_measurements_to_likelihood_from_saturation(
            df_lift_tests,
            saturation=saturation,
            time_varying_var_name=time_varying_var_name,
            model=model,
        )


def test_tvp_needs_exact_date() -> None:
    df_lift_tests = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "delta_x": [0.1, 0.2, 0.3],
            "sigma": [0.1, 0.2, 0.3],
            "delta_y": [0.1, 0.2, 0.3],
            "channel": ["organic", "paid", "social"],
            "date": [1, 2, 5],
        }
    )
    saturation = saturation_functions()[0]

    time_varying_var_name = "tvp"
    coords = {
        "channel": ["organic", "paid", "social"],
        "date": [1, 2, 3, 4],
    }
    with pm.Model(coords=coords) as model:
        saturation._create_distributions(dims="channel")

        tvp_raw = pm.Normal("tvp_raw", dims="date")
        pm.Deterministic(time_varying_var_name, pt.exp(tvp_raw), dims="date")

    assert "lift_measurements" not in model

    with pytest.raises(
        MissingLiftTestError, match=r"Some lift test values are not in the model: \[2\]"
    ):
        add_lift_measurements_to_likelihood_from_saturation(
            df_lift_tests,
            saturation=saturation,
            time_varying_var_name=time_varying_var_name,
            model=model,
        )
