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
from itertools import pairwise

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from pydantic import ValidationError
from pymc_extras.prior import Prior
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable
from scipy import stats

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
        with pytest.raises(
            ValueError, match=r"Priors for 'alpha' are not used when.*'halflife'"
        ):
            GeometricAdstock(
                l_max=10,
                parametrization="halflife",
                priors={"alpha": Prior("Beta", alpha=1, beta=3)},
            )

    def test_halflife_prior_rejected(self) -> None:
        """An explicit alpha parametrization is not silently overridden."""
        with pytest.raises(
            ValueError, match=r"Priors for 'halflife' are not used when.*'alpha'"
        ):
            GeometricAdstock(
                l_max=10,
                parametrization="alpha",
                priors={"halflife": Prior("Gamma", mu=3, sigma=1)},
            )

    def test_both_priors_rejected(self) -> None:
        """Neither prior is silently dropped when the parametrization is inferred.

        Accepting both would ignore one of them and then serialise both
        alongside the inferred parametrization, so that a constructible
        transformation fails to deserialise.
        """
        with pytest.raises(
            ValueError, match=r"Priors for 'alpha' are not used when.*'halflife'"
        ):
            GeometricAdstock(
                l_max=10,
                priors={
                    "alpha": Prior("Beta", alpha=1, beta=3),
                    "halflife": Prior("Gamma", mu=3, sigma=1),
                },
            )

    def test_conflicting_prior_assignment_rejected(self) -> None:
        """Assignment is the other route to a state that cannot be deserialised."""
        adstock = GeometricAdstock(l_max=10, parametrization="halflife")

        with pytest.raises(
            ValueError, match=r"Priors for 'alpha' are not used when.*'halflife'"
        ):
            adstock.function_priors = {"alpha": Prior("Beta", alpha=1, beta=3)}

        assert adstock.function_priors == GeometricAdstock.halflife_priors

    @pytest.mark.parametrize(
        "quantile, tolerance",
        [(0.05, 0.005), (0.25, 0.005), (0.5, 0.005), (0.75, 0.01), (0.95, 0.02)],
    )
    def test_default_prior_matches_alpha_default(
        self, quantile: float, tolerance: float
    ) -> None:
        """The default half-life prior implies the default alpha prior.

        Retuning either default should be a deliberate act, so pin the implied
        quantiles of alpha against those of ``Beta(1, 3)``. Both are available in
        closed form, since ``alpha = 0.5 ** (1 / h)`` is monotone in ``h``.
        """
        halflife_prior = GeometricAdstock.halflife_priors["halflife"]
        assert halflife_prior.distribution == "InverseGamma"

        halflife = stats.invgamma(
            a=halflife_prior.parameters["alpha"],
            scale=halflife_prior.parameters["beta"],
        ).ppf(quantile)
        implied_alpha = 0.5 ** (1 / halflife)

        alpha_prior = GeometricAdstock.default_priors["alpha"]
        assert alpha_prior.distribution == "Beta"

        expected = stats.beta(
            a=alpha_prior.parameters["alpha"], b=alpha_prior.parameters["beta"]
        ).ppf(quantile)
        assert implied_alpha == pytest.approx(expected, abs=tolerance)

    def test_update_priors(self) -> None:
        adstock = GeometricAdstock(l_max=10, parametrization="halflife")
        prior = Prior("InverseGamma", alpha=4, beta=2)

        adstock.update_priors({"adstock_halflife": prior})

        assert adstock.function_priors["halflife"] == prior

    @pytest.mark.parametrize("l_max", [10, 30], ids=["l_max=10", "l_max=30"])
    @pytest.mark.parametrize("halflife", [2.0, 5.0, 8.0])
    @pytest.mark.parametrize("normalize", [False, True], ids=["raw", "normalized"])
    def test_halflife_exact_under_truncation(
        self, model, l_max: int, halflife: float, normalize: bool
    ) -> None:
        """Truncation and normalization leave the weight ratio at one half."""
        spike = as_xtensor(x, dims=("time",))

        with model:
            weights = (
                GeometricAdstock(
                    l_max=l_max, normalize=normalize, priors={"halflife": halflife}
                )
                .apply(spike, core_dim="time")
                .eval()
            )

        assert weights[int(halflife)] / weights[0] == pytest.approx(0.5)

    def test_function_requires_exactly_one_parameter(self) -> None:
        adstock = GeometricAdstock(l_max=10)

        with pytest.raises(ValueError, match="exactly one"):
            adstock.function(as_xtensor(x, dims=("time",)), dim="time")

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


class TestDelayedAdstockHalfLife:
    """The half-life parametrisation replaces alpha with a prior on the half-life.

    ``theta`` is shared by both parametrisations, so only the width of the
    response around the peak is being re-expressed.
    """

    def test_halflife_prior_implies_parametrization(self) -> None:
        adstock = DelayedAdstock(
            l_max=12, priors={"halflife": Prior("LogNormal", mu=1, sigma=0.3)}
        )

        assert adstock.parametrization == "halflife"
        assert adstock.function_priors == {
            "halflife": Prior("LogNormal", mu=1, sigma=0.3),
            "theta": Prior("HalfNormal", sigma=1),
        }
        assert adstock.variable_mapping == {
            "halflife": "adstock_halflife",
            "theta": "adstock_theta",
        }
        assert adstock.model_config == {
            "adstock_halflife": Prior("LogNormal", mu=1, sigma=0.3),
            "adstock_theta": Prior("HalfNormal", sigma=1),
        }

    def test_parametrization_without_prior_uses_default(self) -> None:
        adstock = DelayedAdstock(l_max=12, parametrization="halflife")

        assert adstock.function_priors == DelayedAdstock.halflife_priors

    def test_class_default_priors_unchanged(self) -> None:
        DelayedAdstock(l_max=12, parametrization="halflife")

        assert DelayedAdstock.default_priors == {
            "alpha": Prior("Beta", alpha=1, beta=3),
            "theta": Prior("HalfNormal", sigma=1),
        }
        assert DelayedAdstock(l_max=12).parametrization == "alpha"

    def test_alpha_prior_rejected(self) -> None:
        with pytest.raises(
            ValueError, match=r"Priors for 'alpha' are not used when.*'halflife'"
        ):
            DelayedAdstock(
                l_max=12,
                parametrization="halflife",
                priors={"alpha": Prior("Beta", alpha=1, beta=3)},
            )

    def test_halflife_prior_rejected(self) -> None:
        """An explicit alpha parametrization is not silently overridden."""
        with pytest.raises(
            ValueError, match=r"Priors for 'halflife' are not used when.*'alpha'"
        ):
            DelayedAdstock(
                l_max=12,
                parametrization="alpha",
                priors={"halflife": Prior("LogNormal", mu=1, sigma=0.3)},
            )

    def test_both_priors_rejected(self) -> None:
        """Neither prior is silently dropped when the parametrization is inferred."""
        with pytest.raises(
            ValueError, match=r"Priors for 'alpha' are not used when.*'halflife'"
        ):
            DelayedAdstock(
                l_max=12,
                priors={
                    "alpha": Prior("Beta", alpha=1, beta=3),
                    "halflife": Prior("LogNormal", mu=1, sigma=0.3),
                },
            )

    def test_conflicting_prior_assignment_rejected(self) -> None:
        """Assignment is the other route to a state that cannot be deserialised."""
        adstock = DelayedAdstock(l_max=12, parametrization="halflife")

        with pytest.raises(
            ValueError, match=r"Priors for 'alpha' are not used when.*'halflife'"
        ):
            adstock.function_priors = {"alpha": Prior("Beta", alpha=1, beta=3)}

        assert adstock.function_priors == DelayedAdstock.halflife_priors

    def test_theta_prior_shared_by_both_parametrizations(self) -> None:
        """``theta`` is orthogonal to the width, so it is not an alternative."""
        theta = Prior("HalfNormal", sigma=2)

        for parametrization in ("alpha", "halflife"):
            adstock = DelayedAdstock(
                l_max=12, parametrization=parametrization, priors={"theta": theta}
            )

            assert adstock.function_priors["theta"] == theta

    @pytest.mark.parametrize(
        "quantile, tolerance",
        [(0.05, 0.005), (0.25, 0.005), (0.5, 0.005), (0.75, 0.01), (0.95, 0.02)],
    )
    def test_default_prior_matches_alpha_default(
        self, quantile: float, tolerance: float
    ) -> None:
        """The default half-life prior implies the default alpha prior.

        Retuning either default should be a deliberate act, so pin the implied
        quantiles of alpha against those of ``Beta(1, 3)``. Both are available in
        closed form, since ``alpha = 2 ** (-1 / h**2)`` is monotone in ``h``.
        """
        halflife_prior = DelayedAdstock.halflife_priors["halflife"]
        assert halflife_prior.distribution == "InverseGamma"

        halflife = stats.invgamma(
            a=halflife_prior.parameters["alpha"],
            scale=halflife_prior.parameters["beta"],
        ).ppf(quantile)
        implied_alpha = 2.0 ** (-1.0 / halflife**2)

        alpha_prior = DelayedAdstock.default_priors["alpha"]
        assert alpha_prior.distribution == "Beta"

        expected = stats.beta(
            a=alpha_prior.parameters["alpha"], b=alpha_prior.parameters["beta"]
        ).ppf(quantile)
        assert implied_alpha == pytest.approx(expected, abs=tolerance)

    def test_update_priors(self) -> None:
        adstock = DelayedAdstock(l_max=12, parametrization="halflife")
        prior = Prior("InverseGamma", alpha=4, beta=2)

        adstock.update_priors({"adstock_halflife": prior})

        assert adstock.function_priors["halflife"] == prior

    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("theta", [0.0, 2.0, 5.0])
    @pytest.mark.parametrize("l_max", [8, 12, 20], ids=lambda v: f"l_max={v}")
    @pytest.mark.parametrize("normalize", [False, True], ids=["raw", "normalized"])
    def test_matches_equivalent_alpha(
        self, model, alpha: float, theta: float, l_max: int, normalize: bool
    ) -> None:
        """The half-life parametrisation is an exact change of coordinates.

        ``h = sqrt(log(0.5) / log(alpha))`` inverts ``alpha = 2 ** (-1 / h**2)``,
        so both parametrisations must build the same kernel.
        """
        halflife = np.sqrt(np.log(0.5) / np.log(alpha))
        spike = as_xtensor(x, dims=("time",))
        kwargs = {"l_max": l_max, "normalize": normalize}

        with model:
            from_halflife = DelayedAdstock(
                **kwargs, priors={"halflife": halflife, "theta": theta}
            ).apply(spike, core_dim="time")
            from_alpha = DelayedAdstock(
                **kwargs, priors={"alpha": alpha, "theta": theta}
            ).apply(spike, core_dim="time")

        np.testing.assert_allclose(from_halflife.eval(), from_alpha.eval())

    @pytest.mark.parametrize("normalize", [False, True], ids=["raw", "normalized"])
    def test_half_peak_interpretation(self, model, normalize: bool) -> None:
        """A half-life away from the peak, effectiveness is half of the peak.

        Normalization divides every weight by the same sum, so it moves the
        absolute scale but not the ratio to the peak.
        """
        theta, halflife = 4, 2
        spike = as_xtensor(x, dims=("time",))

        with model:
            weights = (
                DelayedAdstock(
                    l_max=12,
                    normalize=normalize,
                    priors={"halflife": halflife, "theta": theta},
                )
                .apply(spike, core_dim="time")
                .eval()
            )

        assert weights[theta - halflife] / weights[theta] == pytest.approx(0.5)
        assert weights[theta + halflife] / weights[theta] == pytest.approx(0.5)

        if not normalize:
            assert weights[theta] == pytest.approx(1.0)
            assert weights[theta - halflife] == pytest.approx(0.5)
            assert weights[theta + halflife] == pytest.approx(0.5)

    @pytest.mark.parametrize("distance", [1, 2, 3])
    def test_symmetric_about_theta(self, model, distance: int) -> None:
        theta = 5
        spike = as_xtensor(x, dims=("time",))

        with model:
            weights = (
                DelayedAdstock(
                    l_max=12, normalize=False, priors={"halflife": 2.5, "theta": theta}
                )
                .apply(spike, core_dim="time")
                .eval()
            )

        assert weights[theta - distance] == pytest.approx(weights[theta + distance])

    def test_larger_halflife_widens_the_kernel(self, model) -> None:
        """A larger half-life holds more of the peak at any distance from it."""
        theta = 6
        spike = as_xtensor(x, dims=("time",))
        halflives = [0.5, 1.0, 2.0, 4.0]

        with model:
            kernels = [
                DelayedAdstock(
                    l_max=13,
                    normalize=False,
                    priors={"halflife": halflife, "theta": theta},
                )
                .apply(spike, core_dim="time")
                .eval()
                for halflife in halflives
            ]

        for narrow, wide in pairwise(kernels):
            assert narrow[theta] == pytest.approx(wide[theta])
            for distance in (1, 2, 3):
                assert wide[theta + distance] > narrow[theta + distance]
                assert wide[theta - distance] > narrow[theta - distance]

    def test_function_requires_exactly_one_parameter(self) -> None:
        adstock = DelayedAdstock(l_max=12)
        spike = as_xtensor(x, dims=("time",))

        with pytest.raises(ValueError, match="exactly one"):
            adstock.function(spike, dim="time")

        with pytest.raises(ValueError, match="exactly one"):
            adstock.function(spike, alpha=0.5, halflife=2.0, dim="time")

    def test_sample_prior_and_curve(self) -> None:
        adstock = DelayedAdstock(l_max=12, parametrization="halflife")

        prior = adstock.sample_prior()
        assert "adstock_halflife" in prior
        assert "adstock_theta" in prior

        curve = adstock.sample_curve(prior)
        assert curve.shape == (1, 500, adstock.l_max)

    def test_roundtrip(self) -> None:
        original = DelayedAdstock(
            l_max=7, priors={"halflife": Prior("LogNormal", mu=1, sigma=0.3)}
        )

        data = serialization.serialize(original)
        assert data["parametrization"] == "halflife"

        restored = serialization.deserialize(data)
        assert restored.parametrization == "halflife"
        assert restored == original

    def test_alpha_parametrization_not_serialized(self) -> None:
        assert "parametrization" not in serialization.serialize(DelayedAdstock(l_max=7))

    @pytest.mark.parametrize(
        "x, dims",
        [
            pytest.param(x, ("time",), id="vector"),
            pytest.param(np.broadcast_to(x, (3, 20)), ("channel", "time"), id="matrix"),
        ],
    )
    def test_apply_with_channel_specific_priors(self, model, x, dims) -> None:
        """Half-life broadcasts over dims the same way alpha does.

        A channel-specific half-life lifts a bare time series up to
        ``(channel, time)``, so compare named sizes rather than raw shapes.
        """
        adstock = DelayedAdstock(
            l_max=12,
            priors={
                "halflife": Prior("InverseGamma", alpha=9, beta=5.75, dims="channel"),
                "theta": Prior("HalfNormal", sigma=1, dims="channel"),
            },
        )

        with model:
            y = adstock.apply(as_xtensor(x, dims=dims), core_dim="time")

        assert isinstance(y, XTensorVariable)
        assert dict(zip(y.dims, y.eval().shape, strict=True)) == {
            "channel": 3,
            "time": 20,
        }


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
