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
import pymc as pm
import pytensor.tensor as pt
import pytest
from pymc.model_graph import fast_eval
from pytensor.tensor.variable import TensorVariable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from pymc_marketing.mmm import MMM, GeometricAdstock
from pymc_marketing.mmm.components.saturation import (
    HillSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
    SaturationTransformation,
)
from pymc_marketing.mmm.lift_test import (
    NonMonotonicError,
    UnalignedValuesError,
    add_cost_per_target_potentials,
    add_lift_measurements_to_likelihood_from_saturation,
    assert_monotonic,
    create_time_varying_saturation,
    create_variable_indexer,
    exact_row_indices,
    scale_channel_lift_measurements,
    scale_lift_measurements,
    scale_target_for_lift_measurements,
)


@pytest.fixture(scope="module")
def df_lift_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "geo": ["A", "C"],
            "channel": [1, 2],
        }
    )


@pytest.fixture(scope="module")
def model() -> pm.Model:
    coords = {
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "channel": [1, 2, 3],
        "geo": ["A", "B", "C"],
    }
    return pm.Model(coords=coords)


def convert_to_lists(indices):
    return {key: value.tolist() for key, value in indices.items()}


@pytest.mark.parametrize(
    "columns, expected",
    [
        (["date"], {"date": [0, 2]}),
        (["channel"], {"channel": [0, 1]}),
        (["geo"], {"geo": [0, 2]}),
        (["date", "channel"], {"date": [0, 2], "channel": [0, 1]}),
        (
            ["date", "channel", "geo"],
            {"date": [0, 2], "channel": [0, 1], "geo": [0, 2]},
        ),
    ],
)
def test_exact_row_indices(df_lift_test, model, columns, expected) -> None:
    indices = exact_row_indices(df_lift_test[columns], model)

    assert convert_to_lists(indices) == expected


@pytest.fixture(scope="module")
def df_lift_test_unaligned() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "geo": ["A", "Z"],
            "channel": [1000, 2],
        }
    )


def test_exact_row_indices_raises(df_lift_test_unaligned, model) -> None:
    match = r"The following rows of the DataFrame"
    with pytest.raises(UnalignedValuesError, match=match) as res:
        exact_row_indices(df_lift_test_unaligned, model)

    e = res.value

    assert e.unaligned_values == {"geo": [1], "channel": [0]}
    assert e.unaligned_rows == [0, 1]


@pytest.fixture(scope="module")
def fixed_model() -> pm.Model:
    coords = {
        "geo": ["A", "B", "C"],
        "channel": [1, 2],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.DiracDelta("alpha", [10, 20], dims="channel")
        beta = pm.DiracDelta("beta", [1, 2, 3], dims="geo")
        gamma = pm.Deterministic(
            "gamma", alpha + beta[:, None], dims=("geo", "channel")
        )
        delta = pm.DiracDelta("delta", [1, 2, 3], dims="date")
        pm.Deterministic(
            "epsilon",
            delta[:, None, None] * gamma,
            dims=("date", "geo", "channel"),
        )
    return model


@pytest.fixture(scope="module")
def indices():
    return {
        "geo": [0, 2, 1],
        "channel": [0, 1, 0],
        "date": [2, 1, 0],
    }


@pytest.fixture(scope="module")
def variable_indexer(fixed_model, indices):
    return create_variable_indexer(fixed_model, indices)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("alpha", [10, 20, 10]),
        ("beta", [1, 3, 2]),
        ("gamma", [11, 23, 12]),
        ("delta", [3, 2, 1]),
        ("epsilon", [33, 46, 12]),
    ],
)
def test_variable_indexer(variable_indexer, name, expected) -> None:
    np.testing.assert_allclose(
        fast_eval(variable_indexer(name)),
        expected,
    )


def test_variable_indexer_missing_variable(variable_indexer) -> None:
    with pytest.raises(KeyError, match=r"The variable 'missing' is not in the model"):
        variable_indexer("missing")


def test_lift_test_missing_coords(df_lift_test) -> None:
    """Test that KeyError is raised when coords are missing from the model."""
    with pytest.raises(
        KeyError, match=r"The coords \['date', 'geo', 'channel'\] are not in the model"
    ):
        df_lift_test.pipe(exact_row_indices, model=pm.Model())


def test_lift_test_missing_single_coord(fixed_model) -> None:
    """Test that KeyError is raised with correct message for a single missing coord."""
    df_test = pd.DataFrame(
        {
            "date": fixed_model.coords["date"][:2],
            "channel": fixed_model.coords["channel"],
            "missing_coord": ["X", "Y"],  # This coord won't be in the model
        }
    )

    match = r"The coord \['missing_coord'\] is not in the model"
    with pytest.raises(KeyError, match=match):
        exact_row_indices(df_test, fixed_model)


def test_lift_test_missing_multiple_coords(fixed_model) -> None:
    """Test that KeyError is raised with correct message for multiple missing coords."""
    df_test = pd.DataFrame(
        {
            "channel": fixed_model.coords["channel"],
            "missing1": ["A", "B"],
            "missing2": ["X", "Y"],
        }
    )

    match = r"The coords \['missing1', 'missing2'\] are not in the model"
    with pytest.raises(KeyError, match=match):
        exact_row_indices(df_test, fixed_model)


@pytest.fixture(scope="module")
def df_lift_test_with_numerics(df_lift_test) -> pd.DataFrame:
    return df_lift_test.assign(
        x=100,
        delta_x=50,
        delta_y=0.1,
        sigma=0.15,
    )


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


def test_works_with_negative_delta(df_lift_test_with_numerics) -> None:
    df_lift_test_with_numerics_negative = df_lift_test_with_numerics.assign(
        delta_x=lambda row: row["delta_x"] * -1,
        delta_y=lambda row: row["delta_y"] * -1,
    )

    alpha_dims = "date"
    dist = pm.Gamma

    coords = {
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "channel": [0, 1, 2],
    }
    with pm.Model(coords=coords) as model:
        pm.HalfNormal("saturation_alpha", dims=alpha_dims)
        pm.HalfNormal("saturation_lam", dims="channel")

        add_lift_measurements_to_likelihood_from_saturation(
            df_lift_test=df_lift_test_with_numerics_negative,
            saturation=MichaelisMentenSaturation(),
            dist=dist,
        )

    assert "lift_measurements" in model

    try:
        with model:
            pm.sample(draws=10, tune=10)
    except pm.SamplingError:
        pytest.fail("Negative delta values caused a sampling error.")


def test_check_increasing_assumption() -> None:
    delta_x = pd.Series([1, 2, 3])
    delta_y = pd.Series([1, -2, 3])

    match = r"The data is not monotonic."
    with pytest.raises(NonMonotonicError, match=match):
        assert_monotonic(delta_x, delta_y)


def saturation_functions() -> list[SaturationTransformation]:
    transformations = [
        LogisticSaturation(),
        MichaelisMentenSaturation(),
        HillSaturation(),
    ]
    for transformation in transformations:
        transformation.set_dims_for_all_priors("channel")

    return transformations


@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
def test_create_time_varying_saturation_scales(saturation) -> None:
    kwargs = {name: 0.5 for name in saturation.default_priors.keys()}

    xx = np.linspace(0, 1, 20)
    yy = saturation.function(xx, **kwargs)
    if isinstance(yy, TensorVariable):
        yy = fast_eval(yy)

    MULTIPLIER = 3.5
    function, _ = create_time_varying_saturation(saturation, "time_varying_name")
    yy_tv = function(xx, **kwargs, time_varying=MULTIPLIER)
    if isinstance(yy_tv, TensorVariable):
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

    match = r"The value"
    with pytest.raises(KeyError, match=match):
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

    match = "The following rows of the DataFrame are not aligned"
    with pytest.raises(
        UnalignedValuesError,
        match=match,
    ) as res:
        add_lift_measurements_to_likelihood_from_saturation(
            df_lift_tests,
            saturation=saturation,
            time_varying_var_name=time_varying_var_name,
            model=model,
        )

    assert res.value.unaligned_values == {"date": [2]}


def test_scale_lift_measurements(df_lift_test_with_numerics) -> None:
    result = scale_lift_measurements(
        df_lift_test=df_lift_test_with_numerics,
        channel_col="channel",
        channel_columns=[0, 1, 2],
        channel_transform=lambda x: x * 2,
        target_transform=lambda x: x / 2,
    )

    expected = df_lift_test_with_numerics.assign(
        x=lambda row: row["x"] * 2.0,
        delta_x=lambda row: row["delta_x"] * 2.0,
        delta_y=lambda row: row["delta_y"] / 2,
        sigma=lambda row: row["sigma"] / 2,
    ).loc[
        :,
        ["channel", "x", "delta_x", "delta_y", "sigma"]
        + (["date"] if "date" in df_lift_test_with_numerics.columns else []),
    ]

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_like=True,
    )


@pytest.fixture
def dummy_mmm_model():
    # Create sample data for dummy model
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.date_range("2024-01-01", periods=52)),
            "organic": np.random.rand(52) * 520,
            "paid": np.random.rand(52) * 200,
            "social": np.random.rand(52) * 50,
            "y": np.random.rand(52) * 500,  # target variable
        }
    )
    X = df[["date", "organic", "paid", "social"]]
    y = df["y"]
    # Initialize model
    model = MMM(
        date_column="date",
        adstock=GeometricAdstock(l_max=6),
        saturation=LogisticSaturation(),
        channel_columns=["organic", "paid", "social"],
        time_varying_media=True,  # trigger the condition
    )
    # Build the model
    model.build_model(X, y)
    return model


def test_adds_date_column_if_missing(dummy_mmm_model):
    df_lift_test = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "delta_x": [0.1, 0.2, 0.3],
            "sigma": [0.1, 0.2, 0.3],
            "delta_y": [0.1, 0.2, 0.3],
            "channel": ["organic", "paid", "social"],
        }
    )

    # Make sure the column is missing initially
    assert "date" not in df_lift_test.columns

    # Run the method (it should handle date patching internally)
    dummy_mmm_model.add_lift_test_measurements(df_lift_test)

    # Check if the date was added inside the function
    assert dummy_mmm_model._last_lift_test_df["date"].notna().all()


def test_add_cost_per_target_potentials(dummy_mmm_model):
    model = dummy_mmm_model

    # Create a simple constant cost_per_target tensor over (date, channel)
    dates = model.model.coords["date"]
    channels = model.model.coords["channel"]
    const_cpt = pt.as_tensor_variable(
        np.full((len(dates), len(channels)), 30.0, dtype=float)
    )

    # Calibration DataFrame: rows map to existing channels (no extra dims in this fixture)
    calibration_df = pd.DataFrame(
        {
            "channel": [channels[0], channels[1]],
            "cost_per_target": [30.0, 45.0],
            "sigma": [2.0, 3.0],
        }
    )

    # Add potentials using tensor pathway
    add_cost_per_target_potentials(
        calibration_df=calibration_df,
        model=model.model,
        cpt_value=const_cpt,
        name_prefix="cpt_calibration",
    )

    # Check aggregated potential was added with the expected base name
    pot_names = [getattr(p, "name", None) for p in model.model.potentials]
    assert "cpt_calibration" in pot_names


def test_add_cost_per_target_potentials_missing_columns(dummy_mmm_model):
    """Test that KeyError is raised when required columns are missing from calibration_df."""
    model = dummy_mmm_model.model
    channels = model.coords["channel"]

    dates = model.coords["date"]
    const_cpt = pt.as_tensor_variable(
        np.full((len(dates), len(channels)), 30.0, dtype=float)
    )

    # Test missing 'sigma' column
    calibration_df = pd.DataFrame(
        {
            "channel": [channels[0], channels[1]],
            "cost_per_target": [30.0, 45.0],
            # Missing 'sigma' column
        }
    )

    match = r"Missing required columns in calibration_df: \['sigma'\]"
    with pytest.raises(KeyError, match=match):
        add_cost_per_target_potentials(
            calibration_df=calibration_df,
            model=model,
            cpt_value=const_cpt,
            name_prefix="cpt_calibration",
        )

    # Test missing multiple columns
    calibration_df_minimal = pd.DataFrame({"channel": [channels[0], channels[1]]})

    match = (
        r"Missing required columns in calibration_df: \['cost_per_target', 'sigma'\]"
    )
    with pytest.raises(KeyError, match=match):
        add_cost_per_target_potentials(
            calibration_df=calibration_df_minimal,
            model=model,
            cpt_value=const_cpt,
            name_prefix="cpt_calibration",
        )
