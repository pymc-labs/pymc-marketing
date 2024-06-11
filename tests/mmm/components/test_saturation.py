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
from inspect import signature

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr

from pymc_marketing.mmm.components.saturation import (
    HillSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
    TanhSaturation,
    TanhSaturationBaselined,
    _get_saturation_function,
)


@pytest.fixture
def model() -> pm.Model:
    coords = {"channel": ["a", "b", "c"]}
    return pm.Model(coords=coords)


def saturation_functions():
    return [
        LogisticSaturation(),
        TanhSaturation(),
        TanhSaturationBaselined(),
        MichaelisMentenSaturation(),
        HillSaturation(),
    ]


@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
@pytest.mark.parametrize(
    "x, dims",
    [
        (np.linspace(0, 1, 100), None),
        (np.ones((100, 3)), "channel"),
    ],
)
def test_apply_method(model, saturation, x, dims) -> None:
    with model:
        y = saturation.apply(x, dims=dims)

    assert isinstance(y, pt.TensorVariable)
    assert y.eval().shape == x.shape


@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
def test_default_prefix(saturation) -> None:
    assert saturation.prefix == "saturation"
    for value in saturation.variable_mapping.values():
        assert value.startswith("saturation_")


@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
def test_support_for_lift_test_integrations(saturation) -> None:
    function_parameters = signature(saturation.function).parameters

    for key in saturation.variable_mapping.keys():
        assert isinstance(key, str)
        assert key in function_parameters

    assert len(saturation.variable_mapping) == len(function_parameters) - 1


@pytest.mark.parametrize(
    "name, saturation_cls",
    [
        ("logistic", LogisticSaturation),
        ("tanh", TanhSaturation),
        ("tanh_baselined", TanhSaturationBaselined),
        ("michaelis_menten", MichaelisMentenSaturation),
        ("hill", HillSaturation),
    ],
)
def test_get_saturation_function(name, saturation_cls) -> None:
    saturation = _get_saturation_function(name)

    assert isinstance(saturation, saturation_cls)


@pytest.mark.parametrize("saturation", saturation_functions())
def test_get_saturation_function_passthrough(saturation) -> None:
    id_before = id(saturation)
    id_after = id(_get_saturation_function(saturation))

    assert id_after == id_before


def test_get_saturation_function_unknown() -> None:
    with pytest.raises(
        ValueError, match="Unknown saturation function: unknown. Choose from"
    ):
        _get_saturation_function("unknown")


@pytest.mark.parametrize("saturation", saturation_functions())
def test_sample_curve(saturation) -> None:
    prior = saturation.sample_prior()
    assert isinstance(prior, xr.Dataset)
    curve = saturation.sample_curve(prior)
    assert isinstance(curve, xr.DataArray)
    assert curve.name == "saturation"
    assert curve.shape == (1, 500, 100)
