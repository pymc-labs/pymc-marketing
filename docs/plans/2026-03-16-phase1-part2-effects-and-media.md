# Phase 1: Component-Layer Serialization — Implementation Plan (Part 2: Tasks 7–12)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every MMM component class self-sufficiently serializable through a unified `TypeRegistry`, so that `registry.serialize(obj)` and `registry.deserialize(data)` work for all component types.

**Tech Stack:** Python 3.11+, Pydantic v2, PyMC, pymc_extras (Prior, deserialize), ArviZ (InferenceData), pytest

**Python executable:** `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python`

**Pre-commit:** Run after every file modification: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pre_commit run --files <changed_files>`

**Design doc:** `docs/plans/2026-03-05-serialization-overhaul-design.md` — Phase 1 (component layer, patterns 1–3)

**This is Part 2 of 3.** See also: [Part 1 (Tasks 1–6)](2026-03-16-phase1-part1-core-and-components.md) | [Part 3 (Tasks 13–18)](2026-03-16-phase1-part3-verification-and-special.md)

**Prerequisites:** Tasks 1–6 from Part 1 must be complete before starting this part. The `pymc_marketing/serialization.py` module, `TypeRegistry`, `SerializableMixin`, `DeferredFactory`, and registrations for Adstock, Saturation, Events, and HSGP classes must all be in place.

---

## Task 7: Register HSGPKwargs

**Files:**
- Modify: `pymc_marketing/hsgp_kwargs.py`
- Test: `tests/mmm/test_hsgp.py` (or a new test)

### Step 1: Write the failing test

Add to `tests/mmm/test_hsgp.py` (or create test in appropriate location):

```python
class TestHSGPKwargsTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of HSGPKwargs."""

    def test_to_dict_includes_type_key(self):
        from pymc_marketing.hsgp_kwargs import HSGPKwargs

        obj = HSGPKwargs(m=200, L=None, eta_lam=1.0, ls_mu=5.0, ls_sigma=5.0)
        data = obj.to_dict()
        assert "__type__" in data
        expected = f"{HSGPKwargs.__module__}.{HSGPKwargs.__qualname__}"
        assert data["__type__"] == expected

    def test_registered_in_type_registry(self):
        from pymc_marketing.hsgp_kwargs import HSGPKwargs
        from pymc_marketing.serialization import registry

        type_key = f"{HSGPKwargs.__module__}.{HSGPKwargs.__qualname__}"
        assert type_key in registry._registry

    def test_roundtrip_all_parameters(self):
        from pymc_marketing.hsgp_kwargs import CovFunc, HSGPKwargs
        from pymc_marketing.serialization import registry

        original = HSGPKwargs(
            m=150,
            L=2.5,
            eta_lam=0.5,
            ls_mu=3.0,
            ls_sigma=2.0,
            cov_func=CovFunc.Matern32,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is HSGPKwargs
        assert restored.m == 150
        assert restored.L == 2.5
        assert restored.eta_lam == 0.5
        assert restored.ls_mu == 3.0
        assert restored.ls_sigma == 2.0
        assert restored.cov_func == CovFunc.Matern32
        assert restored == original
```

### Step 2: Run test to verify it fails

### Step 3: Implement

**3a. Add `to_dict()` and `from_dict()` to `HSGPKwargs`, register:**

```python
from pymc_marketing.serialization import registry

@registry.register
class HSGPKwargs(BaseModel):
    # ... existing fields ...

    def to_dict(self) -> dict:
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            **self.model_dump(mode="json"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HSGPKwargs":
        filtered = {k: v for k, v in data.items() if k != "__type__"}
        return cls.model_validate(filtered)
```

### Step 4: Run pre-commit and tests

### Step 5: Commit

```bash
git add pymc_marketing/hsgp_kwargs.py tests/mmm/test_hsgp.py
git commit -m "feat: add TypeRegistry support to HSGPKwargs"
```

---

## Task 8: Register Fourier Classes

**Files:**
- Modify: `pymc_marketing/mmm/fourier.py`
- Test: `tests/mmm/test_fourier.py`

### Step 1: Write the failing tests

Add to `tests/mmm/test_fourier.py`:

```python
class TestFourierTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of Fourier classes."""

    @pytest.mark.parametrize(
        "fourier_cls", [YearlyFourier, MonthlyFourier, WeeklyFourier],
        ids=lambda c: c.__name__,
    )
    def test_to_dict_includes_type_key(self, fourier_cls):
        obj = fourier_cls(n_order=2)
        data = obj.to_dict()
        assert "__type__" in data
        expected = f"{fourier_cls.__module__}.{fourier_cls.__qualname__}"
        assert data["__type__"] == expected

    @pytest.mark.parametrize(
        "fourier_cls", [YearlyFourier, MonthlyFourier, WeeklyFourier],
        ids=lambda c: c.__name__,
    )
    def test_registered_in_type_registry(self, fourier_cls):
        from pymc_marketing.serialization import registry

        type_key = f"{fourier_cls.__module__}.{fourier_cls.__qualname__}"
        assert type_key in registry._registry, f"{fourier_cls.__name__} not registered"

    @pytest.mark.parametrize(
        "fourier_cls", [YearlyFourier, MonthlyFourier, WeeklyFourier],
        ids=lambda c: c.__name__,
    )
    def test_roundtrip_all_parameters(self, fourier_cls):
        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry

        original = fourier_cls(
            n_order=5,
            prefix="custom_fourier",
            prior=Prior("Laplace", mu=0.5, b=2.0),
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is fourier_cls
        assert restored.n_order == 5
        assert restored.prefix == "custom_fourier"
        assert restored.days_in_period == fourier_cls(n_order=1).days_in_period
        assert restored == original
```

### Step 2: Run test to verify it fails

### Step 3: Implement

**3a. Update `FourierBase.to_dict()`** (line ~755) to include `__type__`:

```python
def to_dict(self) -> dict[str, Any]:
    return {
        "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
        "class": self.__class__.__name__,
        "data": self.model_dump(mode="json"),
    }
```

**3b. Update `FourierBase.from_dict()`** to handle `__type__`:

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> Self:
    inner = data.get("data", data)
    if "__type__" in inner:
        inner = {k: v for k, v in inner.items() if k != "__type__"}
    inner["prior"] = deserialize(inner["prior"])
    return cls(**inner)
```

**3c. Register all Fourier classes:**

```python
from pymc_marketing.serialization import registry

@registry.register
class YearlyFourier(FourierBase):
    ...

@registry.register
class MonthlyFourier(FourierBase):
    ...

@registry.register
class WeeklyFourier(FourierBase):
    ...
```

### Step 4: Run pre-commit and tests

### Step 5: Commit

```bash
git add pymc_marketing/mmm/fourier.py tests/mmm/test_fourier.py
git commit -m "feat: add TypeRegistry support to Fourier classes"
```

---

## Task 9: Register MediaTransformation / MediaConfig / MediaConfigList

**Files:**
- Modify: `pymc_marketing/mmm/media_transformation.py`
- Test: `tests/mmm/test_media_transformation.py`

### Step 1: Write the failing tests

Add to `tests/mmm/test_media_transformation.py`:

```python
class TestMediaTransformationTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of Media classes."""

    def test_media_transformation_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry

        adstock = GeometricAdstock(
            l_max=8,
            normalize=False,
            mode=ConvMode.Before,
            prefix="custom_adstock",
            priors={"alpha": Prior("Beta", alpha=2.0, beta=5.0)},
        )
        saturation = LogisticSaturation(
            prefix="custom_sat",
            priors={"lam": Prior("Gamma", alpha=2, beta=2), "beta": Prior("HalfNormal", sigma=3)},
        )
        original = MediaTransformation(
            adstock=adstock,
            saturation=saturation,
            adstock_first=False,
            dims=("channel",),
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is MediaTransformation
        assert type(restored.adstock) is GeometricAdstock
        assert type(restored.saturation) is LogisticSaturation
        assert restored.adstock_first is False
        assert restored.dims == ("channel",)
        assert restored.adstock.l_max == 8
        assert restored.adstock.normalize is False
        assert restored.adstock.mode == ConvMode.Before
        assert restored.adstock.prefix == "custom_adstock"
        assert restored.saturation.prefix == "custom_sat"
        assert restored == original

    def test_media_config_roundtrip_all_parameters(self):
        from pymc_marketing.serialization import registry

        mt = MediaTransformation(
            adstock=GeometricAdstock(l_max=6),
            saturation=LogisticSaturation(),
            adstock_first=True,
        )
        original = MediaConfig(
            name="online_channels",
            columns=["tv", "radio", "digital"],
            media_transformation=mt,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is MediaConfig
        assert restored.name == "online_channels"
        assert restored.columns == ["tv", "radio", "digital"]
        assert type(restored.media_transformation.adstock) is GeometricAdstock
        assert restored.media_transformation.adstock.l_max == 6
        assert restored == original

    def test_media_config_list_roundtrip_multiple_configs(self):
        from pymc_marketing.serialization import registry

        mt1 = MediaTransformation(
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mt2 = MediaTransformation(
            adstock=DelayedAdstock(l_max=6),
            saturation=TanhSaturation(),
            adstock_first=False,
        )
        mc1 = MediaConfig(name="online", columns=["tv", "radio"], media_transformation=mt1)
        mc2 = MediaConfig(name="offline", columns=["print"], media_transformation=mt2)
        original = MediaConfigList([mc1, mc2])
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is MediaConfigList
        assert len(restored.media_configs) == 2
        assert restored.media_configs[0].name == "online"
        assert restored.media_configs[0].columns == ["tv", "radio"]
        assert restored.media_configs[1].name == "offline"
        assert type(restored.media_configs[1].media_transformation.adstock) is DelayedAdstock
        assert restored == original
```

### Step 2: Run test to verify it fails

### Step 3: Implement

**3a. Update `MediaTransformation.to_dict()`:**

```python
def to_dict(self) -> dict:
    return {
        "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
        "adstock": self.adstock.to_dict(),
        "saturation": self.saturation.to_dict(),
        "adstock_first": self.adstock_first,
        "dims": self.dims,
    }
```

**3b. Update `MediaTransformation.from_dict()` to use registry for nested types:**

```python
@classmethod
def from_dict(cls, data) -> MediaTransformation:
    from pymc_marketing.serialization import registry

    work = data.copy()
    work.pop("__type__", None)

    adstock_data = work["adstock"]
    saturation_data = work["saturation"]

    if "__type__" in adstock_data:
        adstock = registry.deserialize(adstock_data)
    else:
        adstock = adstock_from_dict(adstock_data)

    if "__type__" in saturation_data:
        saturation = registry.deserialize(saturation_data)
    else:
        saturation = saturation_from_dict(saturation_data)

    return cls(
        adstock=adstock,
        saturation=saturation,
        adstock_first=work["adstock_first"],
        dims=work.get("dims"),
    )
```

**3c. Update `MediaConfig.to_dict()` and `from_dict()` similarly.**

**3d. Update `MediaConfigList.to_dict()` and `from_dict()`:**

```python
def to_dict(self) -> dict:
    return {
        "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
        "media_configs": [config.to_dict() for config in self.media_configs],
    }

@classmethod
def from_dict(cls, data: dict | list) -> MediaConfigList:
    if isinstance(data, list):
        return cls([MediaConfig.from_dict(config) for config in data])
    configs = data.get("media_configs", [])
    return cls([MediaConfig.from_dict(config) for config in configs])
```

**3e. Register all three classes** with `@registry.register`.

### Step 4: Run pre-commit and tests

### Step 5: Commit

```bash
git add pymc_marketing/mmm/media_transformation.py tests/mmm/test_media_transformation.py
git commit -m "feat: add TypeRegistry support to MediaTransformation/Config/ConfigList"
```

---

## Task 10: Add SerializableMixin to LinearTrend + Scaling

**Files:**
- Modify: `pymc_marketing/mmm/linear_trend.py`
- Modify: `pymc_marketing/mmm/scaling.py`
- Test: `tests/mmm/test_linear_trend.py`

### Step 1: Write the failing tests

Add to `tests/mmm/test_linear_trend.py`:

```python
class TestLinearTrendTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of LinearTrend."""

    def test_to_dict_includes_type_key(self):
        from pymc_marketing.mmm.linear_trend import LinearTrend

        lt = LinearTrend(n_changepoints=5)
        data = lt.to_dict()
        assert "__type__" in data
        expected = f"{LinearTrend.__module__}.{LinearTrend.__qualname__}"
        assert data["__type__"] == expected

    def test_registered_in_type_registry(self):
        from pymc_marketing.mmm.linear_trend import LinearTrend
        from pymc_marketing.serialization import registry

        type_key = f"{LinearTrend.__module__}.{LinearTrend.__qualname__}"
        assert type_key in registry._registry

    def test_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.mmm.linear_trend import LinearTrend
        from pymc_marketing.serialization import registry

        original = LinearTrend(
            n_changepoints=8,
            include_intercept=True,
            dims=("geo",),
            priors={
                "delta": Prior("Laplace", mu=0, b=0.5, dims="changepoint"),
                "k": Prior("Normal", mu=0.1, sigma=0.1),
            },
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is LinearTrend
        assert restored.n_changepoints == 8
        assert restored.include_intercept is True
        assert restored.dims == ("geo",)
        assert restored.priors["delta"] == Prior("Laplace", mu=0, b=0.5, dims="changepoint")
        assert restored.priors["k"] == Prior("Normal", mu=0.1, sigma=0.1)
        assert restored == original


class TestScalingTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of Scaling."""

    def test_variable_scaling_roundtrip_all_parameters(self):
        from pymc_marketing.mmm.scaling import VariableScaling
        from pymc_marketing.serialization import registry

        original = VariableScaling(method="mean", dims=("geo", "channel"))
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is VariableScaling
        assert restored.method == "mean"
        assert restored.dims == ("geo", "channel")
        assert restored == original

    def test_scaling_roundtrip_all_parameters(self):
        from pymc_marketing.mmm.scaling import Scaling, VariableScaling
        from pymc_marketing.serialization import registry

        original = Scaling(
            target=VariableScaling(method="max", dims="geo"),
            channel=VariableScaling(method="mean", dims=("geo", "channel")),
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is Scaling
        assert type(restored.target) is VariableScaling
        assert type(restored.channel) is VariableScaling
        assert restored.target.method == "max"
        assert restored.target.dims == ("geo",)
        assert restored.channel.method == "mean"
        assert restored.channel.dims == ("geo", "channel")
        assert restored == original
```

### Step 2: Run tests to verify they fail

### Step 3: Implement

**3a. Add `SerializableMixin` to `LinearTrend`:**

`LinearTrend` has `Prior` fields that `model_dump(mode="json")` can't handle. So we need a **custom `to_dict()`** that serializes priors explicitly:

```python
from pymc_marketing.serialization import SerializableMixin, registry

@registry.register
class LinearTrend(BaseModel):
    # ... existing fields ...

    def to_dict(self) -> dict:
        data = self.model_dump(mode="json", exclude={"priors"})
        data["__type__"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        if self.priors is not None:
            data["priors"] = {k: v.to_dict() for k, v in self.priors.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "LinearTrend":
        from pymc_extras.deserialize import deserialize

        work = {k: v for k, v in data.items() if k != "__type__"}
        if "priors" in work and work["priors"] is not None:
            work["priors"] = {
                k: deserialize(v) for k, v in work["priors"].items()
            }
        if "dims" in work and isinstance(work["dims"], list):
            work["dims"] = tuple(work["dims"])
        return cls.model_validate(work)
```

**3b. Add `SerializableMixin` to `Scaling`/`VariableScaling`:**

These are pure Pydantic models with no Prior fields — `SerializableMixin` works as-is:

```python
from pymc_marketing.serialization import SerializableMixin

class VariableScaling(SerializableMixin, BaseModel):
    # ... existing fields unchanged ...

class Scaling(SerializableMixin, BaseModel):
    # ... existing fields unchanged ...
```

Note: `Scaling.to_dict()` (from mixin) calls `model_dump(mode="json")` which produces nested dicts for `target`/`channel`. Since `VariableScaling` also has `SerializableMixin`, its `model_dump` produces plain dicts. `Scaling.from_dict()` calls `model_validate()` which reconstructs `VariableScaling` instances from nested dicts via Pydantic's normal validation. This works because `VariableScaling` is a Pydantic model and Pydantic knows how to construct it from a dict.

**Important**: the mixin's `to_dict()` calls `model_dump(mode="json")` which does NOT produce `__type__` keys in nested sub-models. This means nested `VariableScaling` objects inside `Scaling.to_dict()` won't have `__type__`. That's fine because `Scaling.from_dict()` uses `model_validate()` which reconstructs them via Pydantic's field types, not via registry dispatch. Registry dispatch is only needed at the top level.

### Step 4: Run pre-commit and tests

### Step 5: Commit

```bash
git add pymc_marketing/mmm/linear_trend.py pymc_marketing/mmm/scaling.py tests/mmm/test_linear_trend.py
git commit -m "feat: add TypeRegistry support to LinearTrend and Scaling"
```

---

## Task 11: Add SerializableMixin to MuEffect Base Class

**Files:**
- Modify: `pymc_marketing/mmm/additive_effect.py`
- Test: `tests/mmm/test_additive_effect.py`

This is a key change: `MuEffect` (the abstract base) gains `SerializableMixin`. Since `MuEffect` is abstract (`ABC`), the mixin's `__init_subclass__` skips registering it (the `inspect.isabstract` check). Concrete subclasses like `FourierEffect`, `LinearTrendEffect`, and `EventAdditiveEffect` are auto-registered.

### Step 1: Write the failing tests

Add to `tests/mmm/test_additive_effect.py`:

```python
class TestMuEffectTypeRegistry:
    def test_mu_effect_not_registered(self):
        """MuEffect itself (abstract) should not be in the registry."""
        from pymc_marketing.mmm.additive_effect import MuEffect
        from pymc_marketing.serialization import registry

        type_key = f"{MuEffect.__module__}.{MuEffect.__qualname__}"
        assert type_key not in registry._registry

    def test_fourier_effect_registered(self):
        from pymc_marketing.mmm.additive_effect import FourierEffect
        from pymc_marketing.serialization import registry

        type_key = f"{FourierEffect.__module__}.{FourierEffect.__qualname__}"
        assert type_key in registry._registry

    def test_linear_trend_effect_registered(self):
        from pymc_marketing.mmm.additive_effect import LinearTrendEffect
        from pymc_marketing.serialization import registry

        type_key = f"{LinearTrendEffect.__module__}.{LinearTrendEffect.__qualname__}"
        assert type_key in registry._registry

    def test_event_additive_effect_registered(self):
        from pymc_marketing.mmm.additive_effect import EventAdditiveEffect
        from pymc_marketing.serialization import registry

        type_key = f"{EventAdditiveEffect.__module__}.{EventAdditiveEffect.__qualname__}"
        assert type_key in registry._registry

    def test_custom_mu_effect_auto_registered(self):
        """User-defined MuEffect subclasses should be auto-registered."""
        from pymc_marketing.mmm.additive_effect import MuEffect
        from pymc_marketing.serialization import registry

        class CustomTestEffect(MuEffect):
            my_val: int = 42

            def create_data(self, mmm):
                pass

            def create_effect(self, mmm):
                pass

            def set_data(self, mmm, model, X):
                pass

        type_key = f"{CustomTestEffect.__module__}.{CustomTestEffect.__qualname__}"
        assert type_key in registry._registry
```

### Step 2: Run test to verify it fails

### Step 3: Implement

Add `SerializableMixin` to the `MuEffect` class hierarchy:

```python
from pymc_marketing.serialization import SerializableMixin

class MuEffect(SerializableMixin, ABC, BaseModel):
    """Abstract base class for arbitrary additive mu effects."""

    @abstractmethod
    def create_data(self, mmm: Model) -> None: ...

    @abstractmethod
    def create_effect(self, mmm: Model) -> XTensorVariable: ...

    @abstractmethod
    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None: ...
```

The `SerializableMixin` must come **before** `ABC` and `BaseModel` in the MRO. Since:
- `SerializableMixin` provides `to_dict()`/`from_dict()`/`__init_subclass__()`
- `ABC` provides abstract method enforcement
- `BaseModel` provides Pydantic functionality

The `__init_subclass__` hook on `SerializableMixin` calls `inspect.isabstract(cls)`, which returns `True` for `MuEffect` (it has unresolved abstract methods), so `MuEffect` itself is NOT registered. Concrete subclasses that implement all abstract methods ARE registered.

### Step 4: Run pre-commit and tests

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_additive_effect.py -v --no-header -x`

Expected: ALL PASS

### Step 5: Commit

```bash
git add pymc_marketing/mmm/additive_effect.py tests/mmm/test_additive_effect.py
git commit -m "feat: add SerializableMixin to MuEffect base class"
```

---

## Task 12: FourierEffect — Custom to_dict/from_dict

**Files:**
- Modify: `pymc_marketing/mmm/additive_effect.py`
- Test: `tests/mmm/test_additive_effect.py`

`FourierEffect` needs custom overrides because the mixin's `model_dump(mode="json")` cannot preserve the `FourierBase` subclass discriminator.

### Step 1: Write the failing test

```python
class TestFourierEffectSerialization:
    """Tests for TypeRegistry-based round-trip serialization of FourierEffect."""

    @pytest.mark.parametrize(
        "fourier_cls", [YearlyFourier, MonthlyFourier, WeeklyFourier],
        ids=lambda c: c.__name__,
    )
    def test_fourier_effect_roundtrip_all_fourier_types(self, fourier_cls):
        from pymc_extras.prior import Prior

        from pymc_marketing.mmm.additive_effect import FourierEffect
        from pymc_marketing.serialization import registry

        original = FourierEffect(
            fourier=fourier_cls(
                n_order=5,
                prefix="custom_fourier",
                prior=Prior("Laplace", mu=0.5, b=2.0),
            ),
            date_dim_name="custom_date",
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is FourierEffect
        assert type(restored.fourier) is fourier_cls
        assert restored.fourier.n_order == 5
        assert restored.fourier.prefix == "custom_fourier"
        assert restored.date_dim_name == "custom_date"
        assert restored == original
```

### Step 2: Run test to verify it fails

### Step 3: Implement

Add `to_dict()` and `from_dict()` overrides to `FourierEffect`:

```python
class FourierEffect(MuEffect):
    fourier: InstanceOf[FourierBase]
    date_dim_name: str = Field("date")

    def to_dict(self) -> dict[str, Any]:
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "fourier": self.fourier.to_dict(),
            "date_dim_name": self.date_dim_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FourierEffect":
        from pymc_marketing.serialization import registry

        work = {k: v for k, v in data.items() if k != "__type__"}
        fourier_data = work["fourier"]
        if "__type__" in fourier_data:
            fourier = registry.deserialize(fourier_data)
        else:
            from pymc_extras.deserialize import deserialize
            fourier = deserialize(fourier_data)
        return cls(fourier=fourier, date_dim_name=work.get("date_dim_name", "date"))

    # ... existing create_data, create_effect, set_data unchanged ...
```

### Step 4: Run pre-commit and tests

### Step 5: Commit

```bash
git add pymc_marketing/mmm/additive_effect.py tests/mmm/test_additive_effect.py
git commit -m "feat: add custom to_dict/from_dict for FourierEffect"
```
