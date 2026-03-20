# Phase 1: Component-Layer Serialization — Implementation Plan (Part 3: Tasks 13–18)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every MMM component class self-sufficiently serializable through a unified `TypeRegistry`, so that `registry.serialize(obj)` and `registry.deserialize(data)` work for all component types.

**Tech Stack:** Python 3.11+, Pydantic v2, PyMC, pymc_extras (Prior, deserialize), ArviZ (InferenceData), pytest

**Python executable:** `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python`

**Pre-commit:** Run after every file modification: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pre_commit run --files <changed_files>`

**Design doc:** `docs/plans/2026-03-05-serialization-overhaul-design.md` — Phase 1 (component layer, patterns 1–3)

**This is Part 3 of 3.** See also: [Part 1 (Tasks 1–6)](2026-03-16-phase1-part1-core-and-components.md) | [Part 2 (Tasks 7–12)](2026-03-16-phase1-part2-effects-and-media.md)

**Prerequisites:** Tasks 1–12 from Parts 1 and 2 must be complete before starting this part. All component registrations (Adstock, Saturation, Events, HSGP, HSGPKwargs, Fourier, MediaTransformation, LinearTrend, Scaling, MuEffect + FourierEffect) must be in place.

---

## Task 13: LinearTrendEffect — Fix Field + Custom Serialization

**Files:**
- Modify: `pymc_marketing/mmm/additive_effect.py`
- Test: `tests/mmm/test_additive_effect.py`

### Step 1: Write the failing test

```python
class TestLinearTrendEffectSerialization:
    """Tests for TypeRegistry-based round-trip serialization of LinearTrendEffect."""

    def test_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.mmm.additive_effect import LinearTrendEffect
        from pymc_marketing.mmm.linear_trend import LinearTrend
        from pymc_marketing.serialization import registry

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
        assert restored.trend.priors["delta"] == Prior("Laplace", mu=0, b=0.5, dims="changepoint")
        assert restored == original
```

### Step 2: Run test to verify it fails

### Step 3: Implement

**3a. Fix `linear_trend_first_date`** — make it a proper Pydantic field:

```python
class LinearTrendEffect(MuEffect):
    trend: InstanceOf[LinearTrend]
    prefix: str
    date_dim_name: str = Field("date")
    linear_trend_first_date: Any = Field(default=None, exclude=True)
```

Remove `model_config = {"extra": "allow"}` and the `__init__` override.

**3b. Add custom `to_dict()`/`from_dict()`:**

```python
    def to_dict(self) -> dict[str, Any]:
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "trend": self.trend.to_dict(),
            "prefix": self.prefix,
            "date_dim_name": self.date_dim_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearTrendEffect":
        from pymc_marketing.serialization import registry

        work = {k: v for k, v in data.items() if k != "__type__"}
        trend_data = work["trend"]
        if "__type__" in trend_data:
            trend = registry.deserialize(trend_data)
        else:
            from pymc_extras.deserialize import deserialize
            trend_dict = trend_data.copy()
            if "priors" in trend_dict and trend_dict["priors"]:
                trend_dict["priors"] = {
                    k: deserialize(v) for k, v in trend_dict["priors"].items()
                }
            from pymc_marketing.mmm.linear_trend import LinearTrend
            trend = LinearTrend.model_validate(trend_dict)
        return cls(
            trend=trend,
            prefix=work["prefix"],
            date_dim_name=work.get("date_dim_name", "date"),
        )
```

### Step 4: Run pre-commit and tests

Verify `create_data` still sets `linear_trend_first_date` correctly:

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_additive_effect.py -v --no-header -x`

### Step 5: Commit

```bash
git add pymc_marketing/mmm/additive_effect.py tests/mmm/test_additive_effect.py
git commit -m "feat: fix LinearTrendEffect field + add custom serialization"
```

---

## Task 14: EventAdditiveEffect — Supplementary Data Mechanism

**Files:**
- Modify: `pymc_marketing/mmm/additive_effect.py`
- Test: `tests/mmm/test_additive_effect.py`

### Step 1: Write the failing tests

```python
class TestEventAdditiveEffectSerialization:
    """Tests for TypeRegistry-based round-trip serialization of EventAdditiveEffect."""

    def test_to_dict_serializes_all_fields(self):
        import pandas as pd

        from pymc_extras.prior import Prior

        from pymc_marketing.mmm.additive_effect import EventAdditiveEffect
        from pymc_marketing.mmm.events import EventEffect, GaussianBasis

        df = pd.DataFrame({
            "name": ["event1"],
            "start_date": ["2024-01-01"],
            "end_date": ["2024-01-07"],
        })
        basis = GaussianBasis(
            prefix="ev_basis",
            priors={"sigma": Prior("Gamma", mu=5, sigma=2)},
        )
        effect = EventEffect(
            basis=basis,
            effect_size=Prior("Normal", mu=0.5, sigma=2.0),
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
        assert data["df_events_group"] == "supplementary_data/custom_events"
        assert data["prefix"] == "custom_events"
        assert data["reference_date"] == "2024-06-01"
        assert data["date_dim_name"] == "custom_date"

    def test_roundtrip_all_parameters_with_mock_context(self):
        import pandas as pd
        import xarray as xr

        from pymc_extras.prior import Prior

        from pymc_marketing.mmm.additive_effect import EventAdditiveEffect
        from pymc_marketing.mmm.events import EventEffect, GaussianBasis
        from pymc_marketing.serialization import (
            DeserializationContext,
            registry,
        )

        df = pd.DataFrame({
            "name": ["ev1", "ev2", "ev3"],
            "start_date": ["2024-01-01", "2024-06-01", "2024-12-01"],
            "end_date": ["2024-01-07", "2024-06-07", "2024-12-07"],
        })
        basis = GaussianBasis(
            prefix="ev_basis",
            priors={"sigma": Prior("Gamma", mu=5, sigma=2)},
        )
        effect = EventEffect(
            basis=basis,
            effect_size=Prior("Normal", mu=0.5, sigma=2.0),
        )
        original = EventAdditiveEffect(
            df_events=df,
            prefix="custom_events",
            effect=effect,
            reference_date="2024-06-01",
            date_dim_name="custom_date",
        )

        data = registry.serialize(original)

        # Simulate idata with supplementary data
        ds = xr.Dataset.from_dataframe(df.set_index("name"))
        fake_idata_dict = {"supplementary_data/custom_events": ds}

        class MockIdata:
            def __getitem__(self, key):
                return fake_idata_dict[key]

        ctx = DeserializationContext(idata=MockIdata())
        restored = registry.deserialize(data, context=ctx)

        assert type(restored) is EventAdditiveEffect
        assert len(restored.df_events) == 3
        assert restored.prefix == "custom_events"
        assert restored.reference_date == "2024-06-01"
        assert restored.date_dim_name == "custom_date"
        assert type(restored.effect) is EventEffect
        assert type(restored.effect.basis) is GaussianBasis
```

### Step 2: Run test to verify it fails

### Step 3: Implement

**3a. Add custom `to_dict()` to `EventAdditiveEffect`:**

```python
def to_dict(self) -> dict[str, Any]:
    return {
        "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
        "prefix": self.prefix,
        "reference_date": self.reference_date,
        "date_dim_name": self.date_dim_name,
        "effect": self.effect.to_dict(),
        "df_events_group": f"supplementary_data/{self.prefix}",
    }
```

**3b. Register a custom deserializer:**

At the bottom of `additive_effect.py` (after class definitions):

```python
from pymc_marketing.serialization import DeserializationContext, registry


def _deserialize_event_additive_effect(
    data: dict[str, Any], context: DeserializationContext | None
) -> EventAdditiveEffect:
    from pymc_marketing.serialization import SerializationError

    group_name = data["df_events_group"]

    if context is None or context.idata is None:
        raise SerializationError(
            f"Cannot deserialize EventAdditiveEffect: no InferenceData "
            f"provided. The df_events DataFrame is stored in idata group "
            f"'{group_name}' and requires a DeserializationContext with idata."
        )

    try:
        ds = context.idata[group_name]
        df_events = ds.to_dataframe().reset_index()
    except (KeyError, AttributeError) as e:
        raise SerializationError(
            f"Cannot read supplementary data group '{group_name}' from "
            f"InferenceData: {e}"
        ) from e

    effect_data = data["effect"]
    if "__type__" in effect_data:
        effect = registry.deserialize(effect_data)
    else:
        effect = EventEffect.from_dict(
            effect_data.get("data", effect_data)
        )

    return EventAdditiveEffect(
        df_events=df_events,
        effect=effect,
        prefix=data["prefix"],
        reference_date=data.get("reference_date", "2025-01-01"),
        date_dim_name=data.get("date_dim_name", "date"),
    )


registry.register(
    f"{EventAdditiveEffect.__module__}.{EventAdditiveEffect.__qualname__}",
    EventAdditiveEffect,
    deserializer=_deserialize_event_additive_effect,
)
```

**Important**: This `registry.register()` call with a custom deserializer must come AFTER the `SerializableMixin.__init_subclass__` auto-registration (which happens at class definition time). Since this explicit call uses the same type key, it will overwrite the auto-registered entry with one that includes the custom deserializer.

### Step 4: Run pre-commit and tests

### Step 5: Commit

```bash
git add pymc_marketing/mmm/additive_effect.py tests/mmm/test_additive_effect.py
git commit -m "feat: add EventAdditiveEffect serialization with supplementary data"
```

---

## Task 15: Register SpecialPrior Hierarchy (Dual Registration)

**Files:**
- Modify: `pymc_marketing/special_priors.py`
- Test: `tests/test_special_priors.py`

The 4 `register_deserialization()` calls in `special_priors.py` are **kept** (they feed `pymc_extras`). We additionally register these types in the `TypeRegistry` so MMM's new serialization path can also resolve them.

### Step 1: Write the failing tests

Add to `tests/test_special_priors.py`:

```python
class TestSpecialPriorTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of SpecialPrior classes."""

    @pytest.mark.parametrize(
        "cls_name",
        ["LogNormalPrior", "LaplacePrior", "MaskedPrior"],
    )
    def test_registered_in_type_registry(self, cls_name):
        from pymc_marketing import special_priors
        from pymc_marketing.serialization import registry

        cls = getattr(special_priors, cls_name)
        type_key = f"{cls.__module__}.{cls.__qualname__}"
        assert type_key in registry._registry, f"{cls_name} not registered"

    def test_log_normal_roundtrip_all_parameters(self):
        from pymc_marketing.serialization import registry
        from pymc_marketing.special_priors import LogNormalPrior

        original = LogNormalPrior(mu=2.0, sigma=0.8, dims=("channel",), centered=False)
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is LogNormalPrior
        assert restored.parameters["mean"] == 2.0
        assert restored.parameters["std"] == 0.8
        assert restored.dims == ("channel",)
        assert restored.centered is False

    def test_laplace_prior_roundtrip_all_parameters(self):
        from pymc_marketing.serialization import registry
        from pymc_marketing.special_priors import LaplacePrior

        original = LaplacePrior(mu=1.5, b=0.3, dims=("geo",), centered=False)
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is LaplacePrior
        assert restored.parameters["mu"] == 1.5
        assert restored.parameters["b"] == 0.3
        assert restored.dims == ("geo",)
        assert restored.centered is False

    def test_masked_prior_roundtrip_all_parameters(self):
        import numpy as np
        import xarray as xr

        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry
        from pymc_marketing.special_priors import MaskedPrior

        mask = xr.DataArray(
            np.array([[True, False], [True, True]]),
            dims=("geo", "channel"),
            coords={"geo": ["a", "b"], "channel": ["tv", "radio"]},
        )
        original = MaskedPrior(
            prior=Prior("Normal", mu=0, sigma=1, dims=("geo", "channel")),
            mask=mask,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is MaskedPrior
        assert restored.dims == original.dims
        xr.testing.assert_equal(restored.mask, original.mask)
```

### Step 2: Run test to verify it fails

### Step 3: Implement

Add `@registry.register` to each SpecialPrior class and update `to_dict()` to include `__type__`. Since these classes already have `to_dict()`/`from_dict()`, we just need to:

1. Import registry: `from pymc_marketing.serialization import registry`
2. Add `@registry.register` to `LogNormalPrior`, `LaplacePrior`, `MaskedPrior`, `SpecialPrior` (base)
3. Update each `to_dict()` to include `"__type__"` key
4. Update `from_dict()` to strip `"__type__"` key

Example for `LogNormalPrior.to_dict()`:

```python
def to_dict(self) -> dict:
    data = super().to_dict()  # existing implementation
    data["__type__"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
    return data
```

And in `from_dict()`, add `data.pop("__type__", None)` at the start.

### Step 4: Run pre-commit and tests

### Step 5: Commit

```bash
git add pymc_marketing/special_priors.py tests/test_special_priors.py
git commit -m "feat: add dual TypeRegistry registration for SpecialPrior hierarchy"
```

---

## Task 16: Comprehensive Round-Trip Tests

**Files:**
- Create: `tests/mmm/test_serialization_roundtrips.py`

### Step 1: Write comprehensive registry round-trip tests

```python
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

from pymc_marketing.mmm.components.adstock import AdstockTransformation
from pymc_marketing.mmm.components.saturation import SaturationTransformation
from pymc_marketing.mmm.transformers import ConvMode
from pymc_marketing.serialization import DeferredFactory, registry
import pymc_marketing.mmm.components.adstock as adstock_module
import pymc_marketing.mmm.components.saturation as saturation_module

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
            m=15, L=2.5,
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
            m=12, L=2.0, eta=deferred_eta, ls=1.0,
            dims=("time", "geo"), centered=True,
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
                l_max=8, normalize=False, mode=ConvMode.Before,
                prefix="geo_adstock",
                priors={"alpha": Prior("Beta", alpha=2.0, beta=5.0)},
            ),
            saturation=LogisticSaturation(
                prefix="log_sat",
                priors={"lam": Prior("Gamma", alpha=2, beta=2), "beta": Prior("HalfNormal", sigma=3)},
            ),
            adstock_first=False,
            dims=("channel",),
        )
        mt2 = MediaTransformation(
            adstock=DelayedAdstock(l_max=6),
            saturation=TanhSaturation(),
        )
        mc1 = MediaConfig(name="online", columns=["tv", "radio"], media_transformation=mt1)
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
```

### Step 2: Run tests

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_serialization_roundtrips.py -v --no-header -x`

Expected: ALL PASS (if all previous tasks are complete)

### Step 3: Commit

```bash
git add tests/mmm/test_serialization_roundtrips.py
git commit -m "test: add comprehensive TypeRegistry round-trip tests for all components"
```

---

## Task 17: Update test_serialization_issues.py Expectations

**Files:**
- Modify: `tests/mmm/test_serialization_issues.py`

### Step 1: Review which issues are now fixed

After all previous tasks, the following issues from `test_serialization_issues.py` should now be fixable through the registry path:

1. **HSGP dims tuple→list** — Fixed by `from_dict()` normalization (Task 6)
2. **HSGP Prior equality** — Addressable via `DeferredFactory` (Task 6)
3. **Custom MuEffect deserialization** — Fixed by `SerializableMixin` auto-registration (Task 11)
4. **LinearTrendEffect model_dump** — Fixed by custom `to_dict()` (Task 13)
5. **EventAdditiveEffect deserialization** — Fixed by supplementary data mechanism (Task 14)

### Step 2: Add new tests that verify fixes through the registry

Add a new test class that validates fixes through `registry.serialize()`/`registry.deserialize()`:

```python
class TestSerializationFixesThroughRegistry:
    """Verify that known serialization issues are fixed via the TypeRegistry."""

    def test_hsgp_dims_preserved_as_tuple(self):
        """Issue: HSGP dims tuple→list after JSON roundtrip."""
        import json

        from pymc_marketing.mmm.hsgp import HSGP
        from pymc_marketing.serialization import registry

        original = HSGP(m=10, L=1.5, eta=1.0, ls=1.0, dims=("time", "geo"))
        data = registry.serialize(original)
        json_str = json.dumps(data)
        data_back = json.loads(json_str)
        restored = registry.deserialize(data_back)
        assert isinstance(restored.dims, tuple)
        assert restored.dims == ("time", "geo")

    def test_custom_mu_effect_roundtrips(self):
        """Issue: Custom MuEffects not in _MUEFFECT_DESERIALIZERS."""
        from pymc_marketing.mmm.additive_effect import MuEffect
        from pymc_marketing.serialization import registry

        class MyCustomEffect(MuEffect):
            param: float = 1.0

            def create_data(self, mmm):
                pass

            def create_effect(self, mmm):
                pass

            def set_data(self, mmm, model, X):
                pass

        original = MyCustomEffect(param=42.0)
        data = registry.serialize(original)
        restored = registry.deserialize(data)
        assert isinstance(restored, MyCustomEffect)
        assert restored.param == 42.0

    def test_linear_trend_effect_serializes_without_error(self):
        """Issue: LinearTrendEffect model_dump fails with Prior fields."""
        from pymc_marketing.mmm.additive_effect import LinearTrendEffect
        from pymc_marketing.mmm.linear_trend import LinearTrend
        from pymc_marketing.serialization import registry

        effect = LinearTrendEffect(
            trend=LinearTrend(n_changepoints=5), prefix="trend"
        )
        data = registry.serialize(effect)
        assert "__type__" in data
        restored = registry.deserialize(data)
        assert isinstance(restored, LinearTrendEffect)

    def test_deferred_factory_replaces_tensor_priors(self):
        """Issue: HSGP Prior equality breaks after roundtrip."""
        from pymc_marketing.mmm.hsgp import HSGP, create_eta_prior
        from pymc_marketing.serialization import DeferredFactory, registry

        deferred_eta = DeferredFactory(
            factory="pymc_marketing.mmm.hsgp.create_eta_prior",
            kwargs={"upper": 5.0, "mass": 0.95},
        )
        hsgp = HSGP(m=10, L=1.5, eta=deferred_eta, ls=1.0, dims="time")
        data = registry.serialize(hsgp)
        restored = registry.deserialize(data)
        assert isinstance(restored.eta, DeferredFactory)
        resolved = restored.eta.resolve()
        assert resolved is not None
```

### Step 3: Run all tests

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_serialization_issues.py -v --no-header`

### Step 4: Commit

```bash
git add tests/mmm/test_serialization_issues.py
git commit -m "test: add registry-based fix verification for known serialization issues"
```

---

## Task 18: Final Verification — Full Test Suite

### Step 1: Run the full component-related test suite

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/test_serialization.py tests/test_special_priors.py tests/mmm/components/ tests/mmm/test_hsgp.py tests/mmm/test_fourier.py tests/mmm/test_events.py tests/mmm/test_media_transformation.py tests/mmm/test_additive_effect.py tests/mmm/test_linear_trend.py tests/mmm/test_serialization_roundtrips.py tests/mmm/test_serialization_issues.py -v --no-header -x 2>&1 | tail -30`

Expected: ALL PASS (no regressions, new tests pass)

### Step 2: Run the broader test suite to verify no regressions

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/ -x --no-header -q --ignore=tests/mmm/test_mmm.py --ignore=tests/mmm/test_multidimensional.py 2>&1 | tail -20`

(Excluding `test_mmm.py` and `test_multidimensional.py` since they test the orchestrator and may be slow)

### Step 3: Final commit

```bash
git add -A
git commit -m "feat: Phase 1 complete — all components support TypeRegistry serialization"
```

---

## Summary of What Phase 1 Delivers

| Component | `__type__` in `to_dict()` | `from_dict()` classmethod | `@registry.register` | DeferredFactory | Supplementary Data |
|---|---|---|---|---|---|
| Adstock (6 classes) | Yes | Yes (on `AdstockTransformation`) | Yes | — | — |
| Saturation (9 classes) | Yes | Yes (on `SaturationTransformation`) | Yes | — | — |
| Basis (3 classes) | Yes | Yes (on `Basis`) | Yes | — | — |
| EventEffect | Yes | Yes (updated) | Yes | — | — |
| HSGP / HSGPPeriodic / SoftPlusHSGP | Yes | Yes (updated, dims fix) | Yes | Yes (eta/ls) | — |
| HSGPKwargs | Yes | Yes (new) | Yes | — | — |
| Fourier (3 classes) | Yes | Yes (updated) | Yes | — | — |
| MediaTransformation / Config / ConfigList | Yes | Yes (updated) | Yes | — | — |
| LinearTrend | Yes | Yes (new) | Yes | — | — |
| Scaling / VariableScaling | Yes | Yes (via mixin) | Yes (via mixin) | — | — |
| MuEffect (abstract) | — | — | Not registered (abstract) | — | — |
| FourierEffect | Yes | Yes (custom) | Yes (via mixin) | — | — |
| LinearTrendEffect | Yes | Yes (custom) | Yes (via mixin) | — | — |
| EventAdditiveEffect | Yes | Yes (custom deserializer) | Yes (custom) | — | Yes |
| SpecialPrior hierarchy (4 classes) | Yes | Yes (updated) | Yes (dual) | — | — |

## What's Left for Phase 2 (Orchestrator Plan)

1. Update `_serializable_model_config` in `multidimensional.py` to use `registry.serialize()`
2. Simplify `create_idata_attrs()` / `attrs_to_init_kwargs()` — route through `registry.serialize()`/`registry.deserialize()`
3. Actually write supplementary data groups to idata during `save()`
4. Add `__serialization_version__` attr to idata
5. Remove the broad `except Exception` catch-all in `build_from_idata()`
6. Remove old infrastructure: `RegistrationMeta`, `lookup_name` from `to_dict()`, `singledispatch`, `_MUEFFECT_DESERIALIZERS`, per-type lookup dicts, standalone `*_from_dict` functions, 11 MMM-specific `register_deserialization` calls
7. `pymc_marketing/migrate.py` — version-aware migration for old `.nc` files
8. Base-class hooks (`ModelBuilder._json_default`, `ModelBuilder._model_config_formatting`)
