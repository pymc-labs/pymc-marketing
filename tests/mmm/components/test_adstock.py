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

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from pydantic import ValidationError
from pymc_extras.prior import Prior
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

import pymc_marketing.mmm.components.adstock as adstock_module
from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
    DelayedAdstock,
    GeometricAdstock,
    NoAdstock,
)
from pymc_marketing.mmm.transformers import ConvMode
from pymc_marketing.serialization import serialization

ALL_ADSTOCK_CLASSES: list[type[AdstockTransformation]] = [
    cls
    for _, cls in inspect.getmembers(adstock_module, inspect.isclass)
    if issubclass(cls, AdstockTransformation) and cls is not AdstockTransformation
]


def adstocks() -> list:
    return [
        pytest.param(adstock_cls(l_max=10), id=adstock_cls.__name__)
        for adstock_cls in ALL_ADSTOCK_CLASSES
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
        pytest.param(x, ("time",), id="vector"),
        pytest.param(np.broadcast_to(x, (3, 20)).T, ("channel", "time"), id="matrix"),
    ],
)
def test_apply(model, adstock: AdstockTransformation, x, dims) -> None:
    x = as_xtensor(x, dims=dims)
    with model:
        y = adstock.apply(x, core_dim="time")

    assert isinstance(y, XTensorVariable)
    assert y.eval().shape == x.type.shape


@pytest.mark.parametrize(
    "adstock",
    adstocks(),
)
def test_default_prefix(adstock: AdstockTransformation) -> None:
    assert adstock.prefix == "adstock"
    for value in adstock.variable_mapping.values():
        assert value.startswith("adstock_")


def test_adstock_no_negative_lmax():
    with pytest.raises(ValidationError, match=r".*Input should be greater than 0.*"):
        DelayedAdstock(l_max=-1)


@pytest.mark.parametrize(
    "adstock",
    adstocks(),
)
def test_adstock_sample_curve(adstock: AdstockTransformation) -> None:
    if isinstance(adstock, NoAdstock):
        raise pytest.skip(reason="NoAdstock has no parameters to sample.")

    prior = adstock.sample_prior()
    assert isinstance(prior, xr.Dataset)
    curve = adstock.sample_curve(prior)
    assert isinstance(curve, xr.DataArray)
    assert curve.name == "adstock"
    assert curve.shape == (1, 500, adstock.l_max)


def test_repr() -> None:
    assert repr(GeometricAdstock(l_max=10)) == (
        "GeometricAdstock(prefix='adstock', l_max=10, "
        "normalize=True, "
        "mode='After', "
        "priors={'alpha': Prior(\"Beta\", alpha=1, beta=3)}"
        ")"
    )


class TestAdstockRoundtrips:
    """Every AdstockTransformation subclass round-trips with all params."""

    @pytest.mark.parametrize(
        "adstock_cls", ALL_ADSTOCK_CLASSES, ids=lambda c: c.__name__
    )
    def test_roundtrip_all_parameters(self, adstock_cls):
        custom_priors = {
            name: Prior("HalfNormal", sigma=0.5) for name in adstock_cls.default_priors
        }
        kwargs: dict = {
            "l_max": 7,
            "normalize": False,
            "mode": ConvMode.Before,
            "prefix": "custom_prefix",
            "priors": custom_priors,
        }

        original = adstock_cls(**kwargs)
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is adstock_cls
        assert restored.l_max == 7
        assert restored.normalize is False
        assert restored.mode == ConvMode.Before
        assert restored.prefix == "custom_prefix"
        for prior_name, prior in custom_priors.items():
            assert restored.function_priors[prior_name] == prior
        assert restored == original


class TestGeometricAdstockHalfLife:
    """The half-life parametrisation replaces alpha with a prior on the half-life."""

    def test_halflife_prior_implies_parametrization(self) -> None:
        adstock = GeometricAdstock(
            l_max=10, priors={"halflife": Prior("Gamma", mu=3, sigma=1)}
        )

        assert adstock.parametrization == "halflife"
        assert adstock.function_priors == {"halflife": Prior("Gamma", mu=3, sigma=1)}
        assert adstock.variable_mapping == {"halflife": "adstock_halflife"}
        assert adstock.model_config == {
            "adstock_halflife": Prior("Gamma", mu=3, sigma=1)
        }

    def test_parametrization_without_prior_uses_default(self) -> None:
        adstock = GeometricAdstock(l_max=10, parametrization="halflife")

        assert adstock.function_priors == GeometricAdstock.halflife_priors

    def test_class_default_priors_unchanged(self) -> None:
        GeometricAdstock(l_max=10, parametrization="halflife")

        assert GeometricAdstock.default_priors == {
            "alpha": Prior("Beta", alpha=1, beta=3)
        }
        assert GeometricAdstock(l_max=10).parametrization == "alpha"

    def test_alpha_prior_rejected(self) -> None:
        with pytest.raises(ValueError, match="Provide a 'halflife' prior instead"):
            GeometricAdstock(
                l_max=10,
                parametrization="halflife",
                priors={"alpha": Prior("Beta", alpha=1, beta=3)},
            )

    def test_matches_equivalent_alpha(self, model) -> None:
        """A half-life of one period decays to alpha = 0.5 ** (1 / 1)."""
        spike = as_xtensor(x, dims=("time",))

        with model:
            halflife = GeometricAdstock(l_max=20, priors={"halflife": 1.0}).apply(
                spike, core_dim="time"
            )
            alpha = GeometricAdstock(l_max=20, priors={"alpha": 0.5}).apply(
                spike, core_dim="time"
            )

        np.testing.assert_allclose(halflife.eval(), alpha.eval())

    def test_sample_prior_and_curve(self) -> None:
        adstock = GeometricAdstock(l_max=10, parametrization="halflife")

        prior = adstock.sample_prior()
        assert "adstock_halflife" in prior

        curve = adstock.sample_curve(prior)
        assert curve.shape == (1, 500, adstock.l_max)

    def test_roundtrip(self) -> None:
        original = GeometricAdstock(
            l_max=7, priors={"halflife": Prior("Gamma", mu=3, sigma=1)}
        )

        data = serialization.serialize(original)
        assert data["parametrization"] == "halflife"

        restored = serialization.deserialize(data)
        assert restored.parametrization == "halflife"
        assert restored == original

    def test_alpha_parametrization_not_serialized(self) -> None:
        assert "parametrization" not in serialization.serialize(
            GeometricAdstock(l_max=7)
        )


@pytest.mark.parametrize(
    "type_key",
    [
        "pymc_marketing.mmm.components.adstock.GeometricAdstock",
        "pymc_marketing.mmm.components.adstock.DelayedAdstock",
        "pymc_marketing.mmm.components.adstock.WeibullCDFAdstock",
        "pymc_marketing.mmm.components.adstock.WeibullPDFAdstock",
        "pymc_marketing.mmm.components.adstock.BinomialAdstock",
        "pymc_marketing.mmm.components.adstock.NoAdstock",
    ],
    ids=lambda s: s.rsplit(".", 1)[-1],
)
def test_type_registered(type_key):
    assert type_key in serialization._registry, f"{type_key} not registered"
