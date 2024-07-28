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
import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from pydantic import ValidationError

from pymc_marketing.mmm.components.adstock import (
    ADSTOCK_TRANSFORMATIONS,
    AdstockTransformation,
    DelayedAdstock,
    GeometricAdstock,
    WeibullAdstock,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
    _get_adstock_function,
    adstock_from_dict,
    register_adstock_transformation,
)
from pymc_marketing.mmm.transformers import ConvMode
from pymc_marketing.prior import Prior


def adstocks() -> list[AdstockTransformation]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [
            DelayedAdstock(l_max=10),
            GeometricAdstock(l_max=10),
            WeibullAdstock(l_max=10, kind="PDF"),
            WeibullAdstock(l_max=10, kind="CDF"),
            WeibullPDFAdstock(l_max=10),
            WeibullCDFAdstock(l_max=10),
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


@pytest.mark.parametrize(
    "name, adstock_cls, kwargs",
    [
        ("delayed", DelayedAdstock, {"l_max": 10}),
        ("geometric", GeometricAdstock, {"l_max": 10}),
        ("weibull", WeibullAdstock, {"l_max": 10}),
    ],
)
def test_get_adstock_function(name, adstock_cls, kwargs):
    # Test for a warning
    with pytest.warns(DeprecationWarning, match="The preferred method of initializing"):
        adstock = _get_adstock_function(name, **kwargs)

    assert isinstance(adstock, adstock_cls)


def test_adstock_no_negative_lmax():
    with pytest.raises(
        ValidationError,
        match="1 validation error for __init__\\nl_max\\n  Input should be greater than 0",
    ):
        DelayedAdstock(l_max=-1)


@pytest.mark.parametrize(
    "adstock",
    adstocks(),
)
def test_get_adstock_function_passthrough(adstock) -> None:
    id_before = id(adstock)
    id_after = id(_get_adstock_function(adstock))

    assert id_after == id_before


def test_get_adstock_function_unknown():
    with pytest.raises(
        ValueError, match="Unknown adstock function: Unknown. Choose from"
    ):
        _get_adstock_function(function="Unknown")


def test_get_adstock_function_unknown_wrong_type():
    with pytest.raises(ValueError, match="Unknown adstock function: 1."):
        _get_adstock_function(function=1)


@pytest.mark.parametrize(
    "adstock",
    adstocks(),
)
def test_adstock_sample_curve(adstock) -> None:
    prior = adstock.sample_prior()
    assert isinstance(prior, xr.Dataset)
    curve = adstock.sample_curve(prior)
    assert isinstance(curve, xr.DataArray)
    assert curve.name == "adstock"
    assert curve.shape == (1, 500, adstock.l_max)


def test_adstock_from_dict() -> None:
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

    adstock = adstock_from_dict(data)
    assert adstock == GeometricAdstock(
        l_max=10,
        prefix="test",
        priors={
            "alpha": Prior("Beta", alpha=1, beta=2),
        },
        mode=ConvMode.Before,
    )


def test_register_adstock_transformation() -> None:
    class NewTransformation(AdstockTransformation):
        lookup_name: str = "new_transformation"
        default_priors = {}

        def function(self, x):
            return x

    register_adstock_transformation(NewTransformation)
    assert "new_transformation" in ADSTOCK_TRANSFORMATIONS

    data = {
        "lookup_name": "new_transformation",
        "l_max": 10,
        "normalize": False,
        "mode": "Before",
        "priors": {},
    }
    adstock = adstock_from_dict(data)
    assert adstock == NewTransformation(
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
