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
import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from pydantic import ValidationError

from pymc_marketing.deserialize import (
    DESERIALIZERS,
    deserialize,
    register_deserialization,
)
from pymc_marketing.mmm import (
    AdstockTransformation,
    DelayedAdstock,
    GeometricAdstock,
    NoAdstock,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
    adstock_from_dict,
)
from pymc_marketing.mmm.transformers import ConvMode
from pymc_marketing.prior import Prior


def adstocks() -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformations = [
            DelayedAdstock(l_max=10),
            GeometricAdstock(l_max=10),
            WeibullPDFAdstock(l_max=10),
            WeibullCDFAdstock(l_max=10),
            NoAdstock(l_max=1),
        ]

    return [
        pytest.param(adstock, id=adstock.lookup_name) for adstock in transformations
    ]


@pytest.fixture
def model() -> pm.Model:
    coords = {"channel": ["a", "b", "c"]}
    return pm.Model(coords=coords)


x = np.zeros(20)
x[0] = 1


@pytest.mark.parametrize(
    "adstock",
    adstocks(),
)
@pytest.mark.parametrize(
    "x, dims",
    [
        (x, None),
        (np.broadcast_to(x, (3, 20)).T, "channel"),
    ],
)
def test_apply(model, adstock, x, dims) -> None:
    with model:
        y = adstock.apply(x, dims=dims)

    assert isinstance(y, pt.TensorVariable)
    assert y.eval().shape == x.shape


@pytest.mark.parametrize(
    "adstock",
    adstocks(),
)
def test_default_prefix(adstock) -> None:
    assert adstock.prefix == "adstock"
    for value in adstock.variable_mapping.values():
        assert value.startswith("adstock_")


def test_adstock_no_negative_lmax():
    with pytest.raises(ValidationError, match=".*Input should be greater than 0.*"):
        DelayedAdstock(l_max=-1)


@pytest.mark.parametrize(
    "adstock",
    adstocks(),
)
def test_adstock_sample_curve(adstock) -> None:
    if adstock.lookup_name == "no_adstock":
        raise pytest.skip(reason="NoAdstock has no parameters to sample.")

    prior = adstock.sample_prior()
    assert isinstance(prior, xr.Dataset)
    curve = adstock.sample_curve(prior)
    assert isinstance(curve, xr.DataArray)
    assert curve.name == "adstock"
    assert curve.shape == (1, 500, adstock.l_max)


@pytest.mark.parametrize("deserialize_func", [adstock_from_dict, deserialize])
def test_adstock_from_dict(deserialize_func) -> None:
    data = {
        "lookup_name": "geometric",
        "l_max": 10,
        "prefix": "test",
        "mode": "Before",
        "priors": {
            "alpha": {
                "dist": "Beta",
                "kwargs": {
                    "alpha": 1,
                    "beta": 2,
                },
            },
        },
    }

    adstock = deserialize_func(data)
    assert adstock == GeometricAdstock(
        l_max=10,
        prefix="test",
        priors={
            "alpha": Prior("Beta", alpha=1, beta=2),
        },
        mode=ConvMode.Before,
    )


@pytest.mark.parametrize(
    "adstock",
    adstocks(),
)
@pytest.mark.parametrize("deserialize_func", [adstock_from_dict, deserialize])
def test_adstock_from_dict_without_priors(adstock, deserialize_func) -> None:
    data = {
        "lookup_name": adstock.lookup_name,
        "l_max": 10,
        "prefix": "test",
        "mode": "Before",
    }

    adstock = deserialize_func(data)
    assert adstock.default_priors == {
        k: Prior.from_dict(v) for k, v in adstock.to_dict()["priors"].items()
    }


class AnotherNewTransformation(AdstockTransformation):
    lookup_name: str = "another_new_transformation"
    default_priors = {}

    def function(self, x):
        return x


@pytest.mark.parametrize("deserialize_func", [adstock_from_dict, deserialize])
def test_automatic_register_adstock_transformation(deserialize_func) -> None:
    data = {
        "lookup_name": "another_new_transformation",
        "l_max": 10,
        "normalize": False,
        "mode": "Before",
        "priors": {},
    }
    adstock = deserialize_func(data)
    assert adstock == AnotherNewTransformation(
        l_max=10, mode=ConvMode.Before, normalize=False, priors={}
    )


def test_repr() -> None:
    assert repr(GeometricAdstock(l_max=10)) == (
        "GeometricAdstock(prefix='adstock', l_max=10, "
        "normalize=True, "
        "mode='After', "
        "priors={'alpha': Prior(\"Beta\", alpha=1, beta=3)}"
        ")"
    )


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
        "lookup_name": "geometric",
        "prefix": "new",
        "l_max": 10,
        "priors": {
            "alpha": {"msg": "hello", "value": 1},
        },
    }

    instance = deserialize(data)
    assert isinstance(instance, GeometricAdstock)
    assert instance.prefix == "new"
    assert instance.l_max == 10

    alpha = instance.function_priors["alpha"]
    assert isinstance(alpha, ArbitraryObject)
    assert alpha.msg == "hello"
    assert alpha.value == 1


def test_deserialize_new_transformation() -> None:
    class NewAdstock(AdstockTransformation):
        lookup_name = "new_adstock"

        def function(self, x):
            return x

        default_priors = {}

    data = {
        "lookup_name": "new_adstock",
        "l_max": 10,
    }

    instance = deserialize(data)
    assert isinstance(instance, NewAdstock)
    assert instance.l_max == 10
