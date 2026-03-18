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
"""Comprehensive round-trip tests verifying every component class
serializes and deserializes correctly through the TypeRegistry.

Each test uses non-default parameter values, asserts exact type preservation,
checks every individual field, and finishes with `restored == original`."""

from __future__ import annotations

import inspect

import pytest
from pymc_extras.prior import Prior

import pymc_marketing.mmm.components.adstock as adstock_module
import pymc_marketing.mmm.components.saturation as saturation_module
from pymc_marketing.mmm.components.adstock import AdstockTransformation
from pymc_marketing.mmm.components.saturation import SaturationTransformation
from pymc_marketing.mmm.transformers import ConvMode
from pymc_marketing.serialization import DeferredFactory, registry

ALL_ADSTOCK_CLASSES: list[type[AdstockTransformation]] = [
    cls
    for _, cls in inspect.getmembers(adstock_module, inspect.isclass)
    if issubclass(cls, AdstockTransformation) and cls is not AdstockTransformation
]

ALL_SATURATION_CLASSES: list[type[SaturationTransformation]] = [
    cls
    for _, cls in inspect.getmembers(saturation_module, inspect.isclass)
    if issubclass(cls, SaturationTransformation) and cls is not SaturationTransformation
]


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
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is adstock_cls
        assert restored.l_max == 7
        assert restored.normalize is False
        assert restored.mode == ConvMode.Before
        assert restored.prefix == "custom_prefix"
        for prior_name, prior in custom_priors.items():
            assert restored.function_priors[prior_name] == prior
        assert restored == original


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
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is sat_cls
        assert restored.prefix == "custom_sat"
        for prior_name, prior in custom_priors.items():
            assert restored.function_priors[prior_name] == prior
        assert restored == original


class TestHSGPRoundtrips:
    def test_hsgp_all_parameters(self):
        from pymc_marketing.hsgp_kwargs import CovFunc
        from pymc_marketing.mmm.hsgp import HSGP

        original = HSGP(
            m=15,
            L=2.5,
            eta=Prior("Exponential", lam=2.0),
            ls=Prior("InverseGamma", alpha=3.0, beta=2.0),
            dims=("time", "geo"),
            centered=True,
            drop_first=False,
            cov_func=CovFunc.Matern52,
            demeaned_basis=True,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is HSGP
        assert restored.m == 15
        assert restored.L == 2.5
        assert isinstance(restored.dims, tuple)
        assert restored.dims == ("time", "geo")
        assert restored.centered is True
        assert restored.drop_first is False
        assert restored.cov_func == CovFunc.Matern52
        assert restored.demeaned_basis is True
        assert restored == original

    def test_hsgp_with_deferred_factory(self):
        from pymc_marketing.mmm.hsgp import HSGP

        deferred_eta = DeferredFactory(
            factory="pymc_marketing.mmm.hsgp.create_eta_prior",
            kwargs={"upper": 5.0, "mass": 0.95},
        )
        original = HSGP(
            m=12,
            L=2.0,
            eta=deferred_eta,
            ls=1.0,
            dims=("time", "geo"),
            centered=True,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is HSGP
        assert isinstance(restored.eta, DeferredFactory)
        assert restored.eta.factory == deferred_eta.factory
        assert restored.eta.kwargs == deferred_eta.kwargs
        assert restored.m == 12
        assert restored.L == 2.0
        assert restored.dims == ("time", "geo")
        assert restored.eta.resolve() is not None

    def test_hsgp_periodic_all_parameters(self):
        from pymc_marketing.mmm.hsgp import HSGPPeriodic

        original = HSGPPeriodic(
            m=15,
            scale=Prior("Exponential", lam=1.5),
            ls=Prior("InverseGamma", alpha=2.0, beta=1.0),
            period=7.0,
            dims=("time", "geo"),
            demeaned_basis=True,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is HSGPPeriodic
        assert restored.m == 15
        assert restored.period == 7.0
        assert isinstance(restored.dims, tuple)
        assert restored.dims == ("time", "geo")
        assert restored.demeaned_basis is True
        assert restored == original


class TestFourierRoundtrips:
    @pytest.mark.parametrize(
        "cls_name",
        ["YearlyFourier", "MonthlyFourier", "WeeklyFourier"],
    )
    def test_fourier_roundtrip_all_parameters(self, cls_name):
        import pymc_marketing.mmm.fourier as fourier_mod

        cls = getattr(fourier_mod, cls_name)
        original = cls(
            n_order=5,
            prefix="custom_fourier",
            prior=Prior("Laplace", mu=0.5, b=2.0),
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is cls
        assert restored.n_order == 5
        assert restored.prefix == "custom_fourier"
        assert restored.days_in_period == cls(n_order=1).days_in_period
        assert restored == original


class TestMuEffectRoundtrips:
    def test_fourier_effect_all_fourier_types(self):
        from pymc_marketing.mmm.additive_effect import FourierEffect
        from pymc_marketing.mmm.fourier import MonthlyFourier

        original = FourierEffect(
            fourier=MonthlyFourier(
                n_order=5,
                prefix="custom_fourier",
                prior=Prior("Laplace", mu=0.5, b=2.0),
            ),
            date_dim_name="custom_date",
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is FourierEffect
        assert type(restored.fourier) is MonthlyFourier
        assert restored.fourier.n_order == 5
        assert restored.fourier.prefix == "custom_fourier"
        assert restored.date_dim_name == "custom_date"
        assert restored == original

    def test_linear_trend_effect_all_parameters(self):
        from pymc_marketing.mmm.additive_effect import LinearTrendEffect
        from pymc_marketing.mmm.linear_trend import LinearTrend

        original = LinearTrendEffect(
            trend=LinearTrend(
                n_changepoints=8,
                include_intercept=True,
                priors={
                    "delta": Prior("Laplace", mu=0, b=0.5, dims="changepoint"),
                    "k": Prior("Normal", mu=0.1, sigma=0.1),
                },
            ),
            prefix="custom_trend",
            date_dim_name="custom_date",
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is LinearTrendEffect
        assert restored.prefix == "custom_trend"
        assert restored.date_dim_name == "custom_date"
        assert type(restored.trend) is LinearTrend
        assert restored.trend.n_changepoints == 8
        assert restored.trend.include_intercept is True
        assert restored == original

    def test_custom_mu_effect_roundtrip(self):
        """User-defined MuEffect subclasses auto-register and round-trip."""
        from pymc_marketing.mmm.additive_effect import MuEffect

        class UserEffect(MuEffect):
            my_param: float = 3.14
            my_str: str = "default"

            def create_data(self, mmm):
                pass

            def create_effect(self, mmm):
                pass

            def set_data(self, mmm, model, X):
                pass

        original = UserEffect(my_param=2.71, my_str="custom_value")
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is UserEffect
        assert restored.my_param == 2.71
        assert restored.my_str == "custom_value"
        assert restored == original


class TestMediaTransformationRoundtrips:
    def test_full_media_config_list_all_parameters(self):
        from pymc_marketing.mmm.components.adstock import (
            DelayedAdstock,
            GeometricAdstock,
        )
        from pymc_marketing.mmm.components.saturation import (
            LogisticSaturation,
            TanhSaturation,
        )
        from pymc_marketing.mmm.media_transformation import (
            MediaConfig,
            MediaConfigList,
            MediaTransformation,
        )

        mt1 = MediaTransformation(
            adstock=GeometricAdstock(
                l_max=8,
                normalize=False,
                mode=ConvMode.Before,
                prefix="geo_adstock",
                priors={"alpha": Prior("Beta", alpha=2.0, beta=5.0)},
            ),
            saturation=LogisticSaturation(
                prefix="log_sat",
                priors={
                    "lam": Prior("Gamma", alpha=2, beta=2),
                    "beta": Prior("HalfNormal", sigma=3),
                },
            ),
            adstock_first=False,
            dims=("channel",),
        )
        mt2 = MediaTransformation(
            adstock=DelayedAdstock(l_max=6),
            saturation=TanhSaturation(),
            adstock_first=True,
        )
        mc1 = MediaConfig(
            name="online", columns=["tv", "radio"], media_transformation=mt1
        )
        mc2 = MediaConfig(name="offline", columns=["print"], media_transformation=mt2)
        original = MediaConfigList([mc1, mc2])
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is MediaConfigList
        assert len(restored.media_configs) == 2

        r1 = restored.media_configs[0]
        assert r1.name == "online"
        assert r1.columns == ["tv", "radio"]
        assert type(r1.media_transformation.adstock) is GeometricAdstock
        assert r1.media_transformation.adstock.l_max == 8
        assert r1.media_transformation.adstock.normalize is False
        assert r1.media_transformation.adstock_first is False

        r2 = restored.media_configs[1]
        assert r2.name == "offline"
        assert type(r2.media_transformation.adstock) is DelayedAdstock

        assert restored == original
