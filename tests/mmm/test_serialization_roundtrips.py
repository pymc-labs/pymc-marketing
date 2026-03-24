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
from pymc_marketing.serialization import DeferredFactory, serialization

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
            transform="sigmoid",
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is HSGP
        assert restored.m == 15
        assert restored.L == 2.5
        assert isinstance(restored.dims, tuple)
        assert restored.dims == ("time", "geo")
        assert restored.centered is True
        assert restored.drop_first is False
        assert restored.cov_func == CovFunc.Matern52
        assert restored.demeaned_basis is True
        assert restored.transform == "sigmoid"
        assert restored == original

    def test_softplus_hsgp_all_parameters(self):
        from pymc_marketing.hsgp_kwargs import CovFunc
        from pymc_marketing.mmm.hsgp import SoftPlusHSGP

        original = SoftPlusHSGP(
            m=20,
            L=3.0,
            eta=Prior("Exponential", lam=1.0),
            ls=2.0,
            dims=("time", "geo"),
            centered=True,
            drop_first=False,
            cov_func=CovFunc.Matern52,
            demeaned_basis=True,
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is SoftPlusHSGP
        assert restored.m == 20
        assert restored.L == 3.0
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
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is HSGP
        assert isinstance(restored.eta, DeferredFactory)
        assert restored.eta.factory == deferred_eta.factory
        assert restored.eta.kwargs == deferred_eta.kwargs
        assert restored.m == 12
        assert restored.L == 2.0
        assert restored.dims == ("time", "geo")
        assert restored.eta.resolve() is not None

    def test_hsgp_with_deferred_factory_all_parameters(self):
        from pymc_marketing.mmm.hsgp import HSGP

        deferred_eta = DeferredFactory(
            factory="pymc_marketing.mmm.hsgp.create_eta_prior",
            kwargs={"upper": 5.0, "mass": 0.95},
        )
        deferred_ls = DeferredFactory(
            factory="pymc_marketing.mmm.hsgp.create_constrained_inverse_gamma_prior",
            kwargs={"upper": 30.0, "lower": 1.0, "mass": 0.9},
        )
        original = HSGP(
            m=12,
            L=2.0,
            eta=deferred_eta,
            ls=deferred_ls,
            dims=("time", "geo"),
            centered=True,
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is HSGP
        assert isinstance(restored.eta, DeferredFactory)
        assert isinstance(restored.ls, DeferredFactory)
        assert restored.eta.factory == deferred_eta.factory
        assert restored.eta.kwargs == deferred_eta.kwargs
        assert restored.ls.factory == deferred_ls.factory
        assert restored.ls.kwargs == deferred_ls.kwargs
        assert restored.m == 12
        assert restored.L == 2.0
        assert restored.dims == ("time", "geo")
        assert restored.centered is True
        assert restored.eta.resolve() is not None
        assert restored == original

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
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

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
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is cls
        assert restored.n_order == 5
        assert restored.prefix == "custom_fourier"
        assert restored.days_in_period == cls(n_order=1).days_in_period
        assert restored == original


class TestMuEffectRoundtrips:
    @pytest.mark.parametrize(
        "fourier_cls_name",
        ["YearlyFourier", "MonthlyFourier", "WeeklyFourier"],
    )
    def test_fourier_effect_all_fourier_types(self, fourier_cls_name):
        import pymc_marketing.mmm.fourier as fourier_mod
        from pymc_marketing.mmm.additive_effect import FourierEffect

        fourier_cls = getattr(fourier_mod, fourier_cls_name)
        original = FourierEffect(
            fourier=fourier_cls(
                n_order=5,
                prefix="custom_fourier",
                prior=Prior("Laplace", mu=0.5, b=2.0),
            ),
            date_dim_name="custom_date",
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is FourierEffect
        assert type(restored.fourier) is fourier_cls
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
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

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
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

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
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

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


class TestHSGPKwargsRoundtrips:
    """Round-trip tests for HSGPKwargs (moved from test_hsgp.py)."""

    def test_to_dict_includes_type_key(self):
        from pymc_marketing.hsgp_kwargs import HSGPKwargs

        obj = HSGPKwargs(m=200, L=None, eta_lam=1.0, ls_mu=5.0, ls_sigma=5.0)
        data = obj.to_dict()
        assert "__type__" in data
        expected = f"{HSGPKwargs.__module__}.{HSGPKwargs.__qualname__}"
        assert data["__type__"] == expected

    def test_registered_in_type_registry(self):
        from pymc_marketing.hsgp_kwargs import HSGPKwargs

        type_key = f"{HSGPKwargs.__module__}.{HSGPKwargs.__qualname__}"
        assert type_key in serialization._registry

    def test_roundtrip_all_parameters(self):
        from pymc_marketing.hsgp_kwargs import CovFunc, HSGPKwargs

        original = HSGPKwargs(
            m=150,
            L=2.5,
            eta_lam=0.5,
            ls_mu=3.0,
            ls_sigma=2.0,
            cov_func=CovFunc.Matern32,
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is HSGPKwargs
        assert restored.m == 150
        assert restored.L == 2.5
        assert restored.eta_lam == 0.5
        assert restored.ls_mu == 3.0
        assert restored.ls_sigma == 2.0
        assert restored.cov_func == CovFunc.Matern32
        assert restored == original


class TestBasisAndEventEffectRoundtrips:
    """Round-trip tests for Basis and EventEffect (moved from test_events.py)."""

    @pytest.mark.parametrize(
        "basis_cls,kwargs",
        [
            (
                "GaussianBasis",
                {
                    "prefix": "custom_basis",
                    "priors": {"sigma": Prior("Gamma", mu=5, sigma=2)},
                },
            ),
            (
                "HalfGaussianBasis",
                {
                    "mode": "before",
                    "include_event": False,
                    "prefix": "hg_basis",
                    "priors": {"sigma": Prior("Gamma", mu=5, sigma=2)},
                },
            ),
            (
                "AsymmetricGaussianBasis",
                {
                    "event_in": "before",
                    "prefix": "ag_basis",
                    "priors": {
                        "sigma_before": Prior("Gamma", mu=2, sigma=0.5),
                        "sigma_after": Prior("Gamma", mu=5, sigma=1),
                        "a_after": Prior("Normal", mu=0.5, sigma=0.3),
                    },
                },
            ),
        ],
        ids=["GaussianBasis", "HalfGaussianBasis", "AsymmetricGaussianBasis"],
    )
    def test_basis_roundtrip_all_parameters(self, basis_cls, kwargs):
        import pymc_marketing.mmm.events as events_mod

        cls = getattr(events_mod, basis_cls)
        original = cls(**kwargs)
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is cls
        assert restored.prefix == kwargs["prefix"]
        for prior_name, prior in kwargs["priors"].items():
            assert restored.function_priors[prior_name] == prior

        if basis_cls == "HalfGaussianBasis":
            assert restored.mode == kwargs["mode"]
            assert restored.include_event == kwargs["include_event"]
        elif basis_cls == "AsymmetricGaussianBasis":
            assert restored.event_in == kwargs["event_in"]

        assert restored == original

    @pytest.mark.parametrize(
        "basis_cls_name",
        ["GaussianBasis", "HalfGaussianBasis", "AsymmetricGaussianBasis"],
    )
    def test_basis_to_dict_includes_type_key(self, basis_cls_name):
        import pymc_marketing.mmm.events as events_mod

        cls = getattr(events_mod, basis_cls_name)
        obj = cls()
        data = obj.to_dict()
        assert "__type__" in data
        expected = f"{cls.__module__}.{cls.__qualname__}"
        assert data["__type__"] == expected

    def test_event_effect_roundtrip_all_parameters(self):
        from pymc_marketing.mmm.events import EventEffect, GaussianBasis

        basis = GaussianBasis(
            prefix="ev_basis",
            priors={"sigma": Prior("Gamma", mu=5, sigma=2)},
        )
        effect_size = Prior("Normal", mu=0.5, sigma=2.0)
        original = EventEffect(
            basis=basis,
            effect_size=effect_size,
            dims=("date", "event"),
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is EventEffect
        assert type(restored.basis) is GaussianBasis
        assert restored.dims == original.dims
        assert restored.basis.prefix == "ev_basis"
        assert restored == original


class TestEventAdditiveEffectRoundtrips:
    """Round-trip tests for EventAdditiveEffect (moved from test_additive_effect.py)."""

    def test_to_dict_serializes_all_fields(self):
        import pandas as pd

        from pymc_marketing.mmm.additive_effect import EventAdditiveEffect
        from pymc_marketing.mmm.events import EventEffect, GaussianBasis

        df = pd.DataFrame(
            {
                "name": ["event1"],
                "start_date": ["2024-01-01"],
                "end_date": ["2024-01-07"],
            }
        )
        basis = GaussianBasis(
            prefix="ev_basis",
            priors={"sigma": Prior("Gamma", mu=5, sigma=2)},
        )
        effect = EventEffect(
            basis=basis,
            effect_size=Prior("Normal", mu=0.5, sigma=2.0),
            dims="custom_events",
        )
        eae = EventAdditiveEffect(
            df_events=df,
            prefix="custom_events",
            effect=effect,
            reference_date="2024-06-01",
            date_dim_name="custom_date",
        )
        data = eae.to_dict()
        assert "__type__" in data
        assert "effect" in data
        assert "df_events_group" in data
        assert data["df_events_group"] == "supplementary_data_custom_events"
        assert data["prefix"] == "custom_events"
        assert data["reference_date"] == "2024-06-01"
        assert data["date_dim_name"] == "custom_date"

    def test_roundtrip_all_parameters_with_mock_context(self):
        import pandas as pd
        import xarray as xr

        from pymc_marketing.mmm.additive_effect import EventAdditiveEffect
        from pymc_marketing.mmm.events import EventEffect, GaussianBasis
        from pymc_marketing.serialization import DeserializationContext

        df = pd.DataFrame(
            {
                "name": ["ev1", "ev2", "ev3"],
                "start_date": ["2024-01-01", "2024-06-01", "2024-12-01"],
                "end_date": ["2024-01-07", "2024-06-07", "2024-12-07"],
            }
        )
        basis = GaussianBasis(
            prefix="ev_basis",
            priors={"sigma": Prior("Gamma", mu=5, sigma=2)},
        )
        effect = EventEffect(
            basis=basis,
            effect_size=Prior("Normal", mu=0.5, sigma=2.0),
            dims="custom_events",
        )
        original = EventAdditiveEffect(
            df_events=df,
            prefix="custom_events",
            effect=effect,
            reference_date="2024-06-01",
            date_dim_name="custom_date",
        )

        data = serialization.serialize(original)

        ds = xr.Dataset.from_dataframe(df.set_index("name"))
        fake_idata_dict = {"supplementary_data_custom_events": ds}

        class MockIdata:
            def __getitem__(self, key):
                return fake_idata_dict[key]

        ctx = DeserializationContext(idata=MockIdata())
        restored = serialization.deserialize(data, context=ctx)

        assert type(restored) is EventAdditiveEffect
        assert len(restored.df_events) == 3
        assert restored.prefix == "custom_events"
        assert restored.reference_date == "2024-06-01"
        assert restored.date_dim_name == "custom_date"
        assert type(restored.effect) is EventEffect
        assert type(restored.effect.basis) is GaussianBasis


class TestLinearTrendRoundtrips:
    """Round-trip tests for LinearTrend (moved from test_linear_trend.py)."""

    def test_to_dict_includes_type_key(self):
        from pymc_marketing.mmm.linear_trend import LinearTrend

        lt = LinearTrend(n_changepoints=5)
        data = lt.to_dict()
        assert "__type__" in data
        expected = f"{LinearTrend.__module__}.{LinearTrend.__qualname__}"
        assert data["__type__"] == expected

    def test_registered_in_type_registry(self):
        from pymc_marketing.mmm.linear_trend import LinearTrend

        type_key = f"{LinearTrend.__module__}.{LinearTrend.__qualname__}"
        assert type_key in serialization._registry

    def test_roundtrip_all_parameters(self):
        from pymc_marketing.mmm.linear_trend import LinearTrend

        original = LinearTrend(
            n_changepoints=8,
            include_intercept=True,
            dims=("geo",),
            priors={
                "delta": Prior("Laplace", mu=0, b=0.5, dims="changepoint"),
                "k": Prior("Normal", mu=0.1, sigma=0.1),
            },
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is LinearTrend
        assert restored.n_changepoints == 8
        assert restored.include_intercept is True
        assert restored.dims == ("geo",)
        assert restored.priors["delta"] == Prior(
            "Laplace", mu=0, b=0.5, dims="changepoint"
        )
        assert restored.priors["k"] == Prior("Normal", mu=0.1, sigma=0.1)
        assert restored == original


class TestScalingRoundtrips:
    """Round-trip tests for Scaling (moved from test_linear_trend.py)."""

    def test_variable_scaling_roundtrip_all_parameters(self):
        from pymc_marketing.mmm.scaling import VariableScaling

        original = VariableScaling(method="mean", dims=("geo", "channel"))
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is VariableScaling
        assert restored.method == "mean"
        assert restored.dims == ("geo", "channel")
        assert restored == original

    def test_scaling_roundtrip_all_parameters(self):
        from pymc_marketing.mmm.scaling import Scaling, VariableScaling

        original = Scaling(
            target=VariableScaling(method="max", dims="geo"),
            channel=VariableScaling(method="mean", dims=("geo", "channel")),
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is Scaling
        assert type(restored.target) is VariableScaling
        assert type(restored.channel) is VariableScaling
        assert restored.target.method == "max"
        assert restored.target.dims == ("geo",)
        assert restored.channel.method == "mean"
        assert restored.channel.dims == ("geo", "channel")
        assert restored == original


class TestRegistrationValidation:
    """Verify all built-in types are registered and have __type__ in to_dict()."""

    @pytest.mark.parametrize(
        "cls",
        [
            "pymc_marketing.mmm.components.adstock.GeometricAdstock",
            "pymc_marketing.mmm.components.adstock.DelayedAdstock",
            "pymc_marketing.mmm.components.adstock.WeibullCDFAdstock",
            "pymc_marketing.mmm.components.adstock.WeibullPDFAdstock",
            "pymc_marketing.mmm.components.adstock.BinomialAdstock",
            "pymc_marketing.mmm.components.adstock.NoAdstock",
            "pymc_marketing.mmm.components.saturation.LogisticSaturation",
            "pymc_marketing.mmm.components.saturation.TanhSaturation",
            "pymc_marketing.mmm.components.saturation.TanhSaturationBaselined",
            "pymc_marketing.mmm.components.saturation.HillSaturation",
            "pymc_marketing.mmm.components.saturation.HillSaturationSigmoid",
            "pymc_marketing.mmm.components.saturation.MichaelisMentenSaturation",
            "pymc_marketing.mmm.components.saturation.RootSaturation",
            "pymc_marketing.mmm.components.saturation.InverseScaledLogisticSaturation",
            "pymc_marketing.mmm.components.saturation.NoSaturation",
            "pymc_marketing.mmm.hsgp.HSGP",
            "pymc_marketing.mmm.hsgp.HSGPPeriodic",
            "pymc_marketing.mmm.hsgp.SoftPlusHSGP",
            "pymc_marketing.mmm.fourier.YearlyFourier",
            "pymc_marketing.mmm.fourier.MonthlyFourier",
            "pymc_marketing.mmm.fourier.WeeklyFourier",
            "pymc_marketing.mmm.events.GaussianBasis",
            "pymc_marketing.mmm.events.HalfGaussianBasis",
            "pymc_marketing.mmm.events.AsymmetricGaussianBasis",
            "pymc_marketing.mmm.events.EventEffect",
            "pymc_marketing.mmm.additive_effect.FourierEffect",
            "pymc_marketing.mmm.additive_effect.LinearTrendEffect",
            "pymc_marketing.mmm.additive_effect.EventAdditiveEffect",
            "pymc_marketing.mmm.linear_trend.LinearTrend",
            "pymc_marketing.mmm.scaling.Scaling",
            "pymc_marketing.mmm.scaling.VariableScaling",
            "pymc_marketing.mmm.media_transformation.MediaTransformation",
            "pymc_marketing.mmm.media_transformation.MediaConfig",
            "pymc_marketing.mmm.media_transformation.MediaConfigList",
            "pymc_marketing.hsgp_kwargs.HSGPKwargs",
        ],
        ids=lambda s: s.rsplit(".", 1)[-1],
    )
    def test_type_registered(self, cls):
        assert cls in serialization._registry, f"{cls} not registered in TypeRegistry"
