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
import inspect
from inspect import signature

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from pydantic import ValidationError
from pymc_extras.deserialize import (
    DESERIALIZERS,
    register_deserialization,
)
from pymc_extras.prior import Prior
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

import pymc_marketing.mmm.components.saturation as saturation_module
from pymc_marketing.mmm.components.saturation import (
    SATURATION_TRANSFORMATIONS,
    LogisticSaturation,
    MichaelisMentenSaturation,
    SaturationTransformation,
    saturation_from_dict,
)
from pymc_marketing.serialization import serialization

ALL_SATURATION_CLASSES: list[type[SaturationTransformation]] = [
    cls
    for _, cls in inspect.getmembers(saturation_module, inspect.isclass)
    if issubclass(cls, SaturationTransformation) and cls is not SaturationTransformation
]


@pytest.fixture
def model() -> pm.Model:
    coords = {"channel": ["a", "b", "c"]}
    return pm.Model(coords=coords)


def saturation_functions():
    return [
        pytest.param(saturation(), id=name)
        for name, saturation in SATURATION_TRANSFORMATIONS.items()
    ]


@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
@pytest.mark.parametrize(
    "x, dims",
    [
        pytest.param(np.linspace(0, 1, 100), ("time",), id="vector"),
        pytest.param(np.ones((100, 3)), ("time", "channel"), id="matrix"),
    ],
)
def test_apply_method(
    model,
    saturation: SaturationTransformation,
    x,
    dims,
) -> None:
    x = as_xtensor(x, dims=dims)

    with model:
        y = saturation.apply(x)

    assert isinstance(y, XTensorVariable)
    assert y.eval().shape == x.type.shape


@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
def test_default_prefix(saturation: SaturationTransformation) -> None:
    assert saturation.prefix == "saturation"
    for value in saturation.variable_mapping.values():
        assert value.startswith("saturation_")


@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
def test_support_for_lift_test_integrations(
    saturation: SaturationTransformation,
) -> None:
    function_parameters = signature(saturation.function).parameters

    for key in saturation.variable_mapping.keys():
        assert isinstance(key, str)
        assert key in function_parameters

    assert len(saturation.variable_mapping) == len(function_parameters) - 2


@pytest.mark.parametrize("saturation", saturation_functions())
def test_sample_curve(saturation: SaturationTransformation) -> None:
    prior = saturation.sample_prior()
    assert isinstance(prior, xr.Dataset)
    curve = saturation.sample_curve(prior)
    assert isinstance(curve, xr.DataArray)
    assert curve.name == "saturation"
    assert curve.shape == (1, 500, 100)


@pytest.mark.parametrize("saturation", saturation_functions())
@pytest.mark.parametrize("num_points", [50, 200, 1000])
def test_sample_curve_num_points(
    saturation: SaturationTransformation,
    num_points,
) -> None:
    """Test that num_points parameter controls the number of points in the curve."""
    prior = saturation.sample_prior()
    curve = saturation.sample_curve(prior, num_points=num_points)
    assert isinstance(curve, xr.DataArray)
    assert curve.name == "saturation"
    assert curve.shape == (1, 500, num_points)


@pytest.mark.parametrize(
    argnames="num_points", argvalues=[0, -1], ids=["zero", "negative"]
)
def test_sample_curve_with_bad_num_points(num_points) -> None:
    """Test that invalid num_points raises ValidationError."""
    saturation = LogisticSaturation()
    prior = saturation.sample_prior()

    with pytest.raises(ValidationError):
        saturation.sample_curve(prior, num_points=num_points)


def create_mock_parameters(
    coords: dict[str, list],
    variable_dim_mapping: dict[str, tuple[str]],
) -> xr.Dataset:
    dim_sizes = {coord: len(values) for coord, values in coords.items()}
    return xr.Dataset(
        {
            name: xr.DataArray(
                np.ones(tuple(dim_sizes[coord] for coord in dims)),
                dims=dims,
                coords={coord: coords[coord] for coord in dims},
            )
            for name, dims in variable_dim_mapping.items()
        }
    )


@pytest.fixture
def mock_menten_parameters() -> xr.Dataset:
    coords = {
        "chain": np.arange(1),
        "draw": np.arange(500),
    }

    variable_dim_mapping = {
        "saturation_alpha": ("chain", "draw"),
        "saturation_lam": ("chain", "draw"),
        "another_random_variable": ("chain", "draw"),
    }

    return create_mock_parameters(coords, variable_dim_mapping)


def test_sample_curve_additional_dataset_variables(mock_menten_parameters) -> None:
    """Case when the parameter dataset has additional variables."""
    saturation = MichaelisMentenSaturation()

    try:
        curve = saturation.sample_curve(parameters=mock_menten_parameters)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

    assert isinstance(curve, xr.DataArray)
    assert curve.name == "saturation"


@pytest.fixture
def mock_menten_parameters_with_additional_dim() -> xr.Dataset:
    coords = {
        "chain": np.arange(1),
        "draw": np.arange(500),
        "channel": ["C1", "C2", "C3"],
        "random_dim": ["R1", "R2"],
    }
    variable_dim_mapping = {
        "saturation_alpha": ("chain", "draw", "channel"),
        "saturation_lam": ("chain", "draw", "channel"),
        "another_random_variable": ("chain", "draw", "channel", "random_dim"),
    }

    return create_mock_parameters(coords, variable_dim_mapping)


def test_sample_curve_with_additional_dims(
    mock_menten_parameters_with_additional_dim,
) -> None:
    dummy_distribution = Prior("HalfNormal", dims="channel")
    priors = {
        "alpha": dummy_distribution,
        "lam": dummy_distribution,
    }
    saturation = MichaelisMentenSaturation(priors=priors)

    curve = saturation.sample_curve(
        parameters=mock_menten_parameters_with_additional_dim
    )

    assert curve.coords["channel"].to_numpy().tolist() == ["C1", "C2", "C3"]
    assert "random_dim" not in curve.coords


@pytest.mark.parametrize(
    argnames="max_value", argvalues=[0, -1], ids=["zero", "negative"]
)
def test_sample_curve_with_bad_max_value(max_value) -> None:
    dummy_distribution = Prior("HalfNormal", dims="channel")
    priors = {
        "alpha": dummy_distribution,
        "lam": dummy_distribution,
    }
    saturation = MichaelisMentenSaturation(priors=priors)

    with pytest.raises(ValidationError):
        saturation.sample_curve(
            parameters=mock_menten_parameters_with_additional_dim, max_value=max_value
        )


def test_saturation_from_dict() -> None:
    data = {
        "lookup_name": "michaelis_menten",
        "priors": {
            "alpha": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
            "lam": {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 1},
            },
        },
    }

    with pytest.warns(FutureWarning, match="saturation_from_dict is deprecated"):
        saturation = saturation_from_dict(data)
    assert saturation == MichaelisMentenSaturation(
        priors={
            "alpha": Prior("HalfNormal", sigma=1),
            "lam": Prior("HalfNormal", sigma=1),
        }
    )


@pytest.mark.parametrize(
    "lookup_name, saturation_cls",
    list(SATURATION_TRANSFORMATIONS.items()),
)
def test_saturation_from_dict_without_priors(lookup_name, saturation_cls) -> None:
    data = {
        "lookup_name": lookup_name,
    }

    with pytest.warns(FutureWarning, match="saturation_from_dict is deprecated"):
        saturation = saturation_from_dict(data)
    assert saturation.default_priors == {
        k: Prior.from_dict(v) for k, v in saturation.to_dict()["priors"].items()
    }


class ArbitraryObject:
    def __init__(self, msg: str, value: int) -> None:
        self.msg = msg
        self.value = value
        self.dims = ()

    def create_variable(self, name: str):
        return pm.Normal(name, mu=0, sigma=1)


@pytest.fixture
def register_arbitrary_deserialization():
    register_deserialization(
        lambda data: isinstance(data, dict) and data.keys() == {"msg", "value"},
        lambda data: ArbitraryObject(**data),
    )

    yield

    DESERIALIZERS.pop()


def test_deserialization(
    register_arbitrary_deserialization,
) -> None:
    data = {
        "lookup_name": "logistic",
        "prefix": "new",
        "priors": {
            "alpha": {"msg": "hello", "value": 1},
        },
    }

    with pytest.warns(FutureWarning, match="saturation_from_dict is deprecated"):
        instance = saturation_from_dict(data)
    assert isinstance(instance, LogisticSaturation)
    assert instance.prefix == "new"

    alpha = instance.function_priors["alpha"]
    assert isinstance(alpha, ArbitraryObject)
    assert alpha.msg == "hello"
    assert alpha.value == 1


class TestSaturationRoundtrips:
    """Every SaturationTransformation subclass round-trips with all params."""

    @pytest.mark.parametrize(
        "sat_cls", ALL_SATURATION_CLASSES, ids=lambda c: c.__name__
    )
    def test_roundtrip_all_parameters(self, sat_cls):
        custom_priors = {
            name: Prior("HalfNormal", sigma=0.5) for name in sat_cls.default_priors
        }
        kwargs: dict = {
            "prefix": "custom_sat",
            "priors": custom_priors,
        }

        original = sat_cls(**kwargs)
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is sat_cls
        assert restored.prefix == "custom_sat"
        for prior_name, prior in custom_priors.items():
            assert restored.function_priors[prior_name] == prior
        assert restored == original


@pytest.mark.parametrize(
    "type_key",
    [
        "pymc_marketing.mmm.components.saturation.LogisticSaturation",
        "pymc_marketing.mmm.components.saturation.TanhSaturation",
        "pymc_marketing.mmm.components.saturation.TanhSaturationBaselined",
        "pymc_marketing.mmm.components.saturation.HillSaturation",
        "pymc_marketing.mmm.components.saturation.HillSaturationSigmoid",
        "pymc_marketing.mmm.components.saturation.MichaelisMentenSaturation",
        "pymc_marketing.mmm.components.saturation.RootSaturation",
        "pymc_marketing.mmm.components.saturation.InverseScaledLogisticSaturation",
        "pymc_marketing.mmm.components.saturation.NoSaturation",
    ],
    ids=lambda s: s.rsplit(".", 1)[-1],
)
def test_type_registered(type_key):
    assert type_key in serialization._registry, f"{type_key} not registered"
