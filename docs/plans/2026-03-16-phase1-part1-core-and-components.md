# Phase 1: Component-Layer Serialization — Implementation Plan (Part 1: Tasks 1–6)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every MMM component class self-sufficiently serializable through a unified `TypeRegistry`, so that `registry.serialize(obj)` and `registry.deserialize(data)` work for all component types.

**Architecture:** A new `pymc_marketing/serialization.py` module introduces a `TypeRegistry` singleton, `Serializable` protocol, `SerializableMixin` (auto-registers Pydantic subclasses), and `DeferredFactory` (stores recipes for non-serializable state). Each component gains a `__type__` key in its `to_dict()` output, a `from_dict()` classmethod (if missing), and a `@registry.register` registration. The old serialization infrastructure (RegistrationMeta, lookup dicts, singledispatch, `_MUEFFECT_DESERIALIZERS`) is **kept temporarily** — it will be removed in Phase 2 when the orchestrator is switched to the new system.

**Tech Stack:** Python 3.11+, Pydantic v2, PyMC, pymc_extras (Prior, deserialize), ArviZ (InferenceData), pytest

**Python executable:** `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python`

**Pre-commit:** Run after every file modification: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pre_commit run --files <changed_files>`

**Design doc:** `docs/plans/2026-03-05-serialization-overhaul-design.md` — Phase 1 (component layer, patterns 1–3)

**This is Part 1 of 3.** See also: [Part 2 (Tasks 7–12)](2026-03-16-phase1-part2-effects-and-media.md) | [Part 3 (Tasks 13–18)](2026-03-16-phase1-part3-verification-and-special.md)

---

## Approach: Additive, Not Subtractive

Phase 1 **adds** the new TypeRegistry system alongside the existing one. Both coexist:

- `to_dict()` outputs include `__type__` **alongside** existing keys (`lookup_name`, `class`, `hsgp_class`)
- `from_dict()` classmethods handle the new `__type__`-based format
- `@registry.register` decorators are added to all component classes
- Old infrastructure stays: `RegistrationMeta`, lookup dicts, `register_deserialization` calls, `singledispatch` handlers, `_MUEFFECT_DESERIALIZERS`

This ensures existing tests pass unchanged while new tests verify the registry round-trips work. Phase 2 will switch orchestrator callers to the new system and remove the old infrastructure.

---

## Task 1: Create Core Infrastructure Tests

**Files:**
- Create: `tests/test_serialization.py`

### Step 1: Write the failing tests

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
"""Tests for core serialization infrastructure."""

from __future__ import annotations

from typing import Any, Self

import pytest
from pydantic import BaseModel, Field


class TestSerializationError:
    def test_is_exception(self):
        from pymc_marketing.serialization import SerializationError

        assert issubclass(SerializationError, Exception)

    def test_message(self):
        from pymc_marketing.serialization import SerializationError

        err = SerializationError("test message")
        assert str(err) == "test message"


class TestSerializableProtocol:
    def test_structural_typing(self):
        from pymc_marketing.serialization import Serializable

        class HasMethods:
            def to_dict(self) -> dict[str, Any]:
                return {}

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> Self:
                return cls()

        assert isinstance(HasMethods(), Serializable)

    def test_not_serializable(self):
        from pymc_marketing.serialization import Serializable

        class NoMethods:
            pass

        assert not isinstance(NoMethods(), Serializable)


class TestDeserializationContext:
    def test_default_none(self):
        from pymc_marketing.serialization import DeserializationContext

        ctx = DeserializationContext()
        assert ctx.idata is None

    def test_with_idata(self):
        from pymc_marketing.serialization import DeserializationContext

        ctx = DeserializationContext(idata="fake_idata")
        assert ctx.idata == "fake_idata"


class TestDeferredFactory:
    def test_to_dict(self):
        from pymc_marketing.serialization import DeferredFactory

        df = DeferredFactory(factory="builtins.int", kwargs={"x": "42"})
        result = df.to_dict()
        assert result == {
            "__deferred__": True,
            "factory": "builtins.int",
            "kwargs": {"x": "42"},
        }

    def test_from_dict(self):
        from pymc_marketing.serialization import DeferredFactory

        data = {
            "__deferred__": True,
            "factory": "builtins.int",
            "kwargs": {"x": "42"},
        }
        df = DeferredFactory.from_dict(data)
        assert df.factory == "builtins.int"
        assert df.kwargs == {"x": "42"}

    def test_resolve(self):
        from pymc_marketing.serialization import DeferredFactory

        df = DeferredFactory(factory="builtins.int", kwargs={"x": "42"})
        result = df.resolve()
        assert result == 42

    def test_roundtrip(self):
        from pymc_marketing.serialization import DeferredFactory

        original = DeferredFactory(factory="builtins.int", kwargs={"x": "42"})
        data = original.to_dict()
        restored = DeferredFactory.from_dict(data)
        assert restored.resolve() == original.resolve()


class TestTypeRegistry:
    def test_register_decorator(self):
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()

        @reg.register
        class Foo:
            def to_dict(self):
                return {
                    "__type__": f"{Foo.__module__}.{Foo.__qualname__}",
                    "x": 1,
                }

            @classmethod
            def from_dict(cls, data):
                return cls()

        type_key = f"{Foo.__module__}.{Foo.__qualname__}"
        assert type_key in reg._registry

    def test_register_with_key(self):
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()

        class Bar:
            def to_dict(self):
                return {"__type__": "custom.Bar", "y": 2}

            @classmethod
            def from_dict(cls, data):
                return cls()

        reg.register("custom.Bar", Bar)
        assert "custom.Bar" in reg._registry

    def test_serialize(self):
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()

        @reg.register
        class Baz:
            def to_dict(self):
                return {
                    "__type__": f"{Baz.__module__}.{Baz.__qualname__}",
                    "val": 99,
                }

            @classmethod
            def from_dict(cls, data):
                return cls()

        result = reg.serialize(Baz())
        assert result["__type__"] == f"{Baz.__module__}.{Baz.__qualname__}"
        assert result["val"] == 99

    def test_deserialize_standard(self):
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()

        @reg.register
        class Qux:
            def __init__(self, val=0):
                self.val = val

            def to_dict(self):
                return {
                    "__type__": f"{Qux.__module__}.{Qux.__qualname__}",
                    "val": self.val,
                }

            @classmethod
            def from_dict(cls, data):
                return cls(val=data.get("val", 0))

        data = {
            "__type__": f"{Qux.__module__}.{Qux.__qualname__}",
            "val": 42,
        }
        obj = reg.deserialize(data)
        assert isinstance(obj, Qux)
        assert obj.val == 42

    def test_deserialize_deferred(self):
        from pymc_marketing.serialization import DeferredFactory, TypeRegistry

        reg = TypeRegistry()
        data = {
            "__deferred__": True,
            "factory": "builtins.int",
            "kwargs": {"x": "7"},
        }
        result = reg.deserialize(data)
        assert isinstance(result, DeferredFactory)
        assert result.resolve() == 7

    def test_deserialize_custom_deserializer(self):
        from pymc_marketing.serialization import DeserializationContext, TypeRegistry

        reg = TypeRegistry()

        class Special:
            def __init__(self, extra=None):
                self.extra = extra

            def to_dict(self):
                return {"__type__": "test.Special", "data": "x"}

        def custom_deser(data, context):
            return Special(extra=context.idata if context else None)

        reg.register("test.Special", Special, deserializer=custom_deser)

        ctx = DeserializationContext(idata="my_idata")
        result = reg.deserialize({"__type__": "test.Special"}, context=ctx)
        assert isinstance(result, Special)
        assert result.extra == "my_idata"

    def test_deserialize_unknown_type_raises(self):
        from pymc_marketing.serialization import SerializationError, TypeRegistry

        reg = TypeRegistry()
        with pytest.raises(SerializationError, match="Unknown type"):
            reg.deserialize({"__type__": "nonexistent.Class"})

    def test_deserialize_missing_type_key_raises(self):
        from pymc_marketing.serialization import SerializationError, TypeRegistry

        reg = TypeRegistry()
        with pytest.raises(SerializationError, match="__type__"):
            reg.deserialize({"no_type": "here"})

    def test_serialize_unregistered_raises(self):
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()

        class Unknown:
            def to_dict(self):
                return {"__type__": "test.Unknown"}

        with pytest.raises((KeyError, TypeError)):
            reg.serialize(Unknown())


class TestSerializableMixin:
    def test_auto_registration(self):
        from pymc_marketing.serialization import TypeRegistry

        test_reg = TypeRegistry()

        from pymc_marketing.serialization import SerializableMixin

        class AutoReg(SerializableMixin, BaseModel):
            x: int = 1

        type_key = f"{AutoReg.__module__}.{AutoReg.__qualname__}"
        # Auto-registered in the module-level registry
        from pymc_marketing.serialization import registry

        assert type_key in registry._registry

    def test_to_dict(self):
        from pymc_marketing.serialization import SerializableMixin

        class MyModel(SerializableMixin, BaseModel):
            name: str = "test"
            value: int = 42

        obj = MyModel()
        result = obj.to_dict()
        assert "__type__" in result
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_from_dict(self):
        from pymc_marketing.serialization import SerializableMixin

        class MyModel2(SerializableMixin, BaseModel):
            name: str = "test"
            value: int = 42

        data = {"__type__": "whatever", "name": "hello", "value": 99}
        obj = MyModel2.from_dict(data)
        assert obj.name == "hello"
        assert obj.value == 99

    def test_roundtrip_via_registry(self):
        from pymc_marketing.serialization import SerializableMixin, registry

        class RoundTripper(SerializableMixin, BaseModel):
            a: str = "foo"
            b: float = 3.14

        original = RoundTripper(a="bar", b=2.71)
        data = registry.serialize(original)
        restored = registry.deserialize(data)
        assert isinstance(restored, RoundTripper)
        assert restored.a == "bar"
        assert restored.b == 2.71

    def test_abstract_subclass_not_registered(self):
        """Abstract subclasses should not be registered."""
        from abc import ABC, abstractmethod

        from pymc_marketing.serialization import SerializableMixin, registry

        class AbstractEffect(SerializableMixin, ABC, BaseModel):
            @abstractmethod
            def do_thing(self):
                ...

        type_key = f"{AbstractEffect.__module__}.{AbstractEffect.__qualname__}"
        assert type_key not in registry._registry

        class ConcreteEffect(AbstractEffect):
            val: int = 1

            def do_thing(self):
                return self.val

        concrete_key = f"{ConcreteEffect.__module__}.{ConcreteEffect.__qualname__}"
        assert concrete_key in registry._registry
```

### Step 2: Run tests to verify they fail

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/test_serialization.py -v --no-header 2>&1 | head -80`

Expected: ERRORS — `ModuleNotFoundError: No module named 'pymc_marketing.serialization'`

### Step 3: Implement — see Task 2

### Step 4: Run tests after Task 2 implementation

### Step 5: Commit after Task 2

---

## Task 2: Create `pymc_marketing/serialization.py`

**Files:**
- Create: `pymc_marketing/serialization.py`

### Step 1: Implement the module

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
"""Unified serialization infrastructure for pymc-marketing.

This module provides the ``TypeRegistry``, ``Serializable`` protocol,
``SerializableMixin``, ``DeferredFactory``, and ``DeserializationContext``
that replace the scattered serialization patterns across MMM components.

Every serializable object produces a JSON-safe dict with a ``__type__`` key
(fully-qualified class path). The ``TypeRegistry`` dispatches deserialization
from that key alone.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field


class SerializationError(Exception):
    """Raised when serialization or deserialization fails."""


@runtime_checkable
class Serializable(Protocol):
    """Structural protocol for serializable objects."""

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...


@dataclass
class DeserializationContext:
    """Runtime state passed to custom deserializers.

    Attributes
    ----------
    idata : InferenceData or None
        The InferenceData object being loaded from, used by deserializers
        that need to read supplementary data groups (e.g., EventAdditiveEffect
        reads df_events from a named idata group).
    """

    idata: Any = None


def _import_from_dotted_path(path: str) -> Any:
    """Import an object from a fully-qualified dotted path."""
    module_path, _, attr_name = path.rpartition(".")
    if not module_path:
        raise ImportError(f"Cannot import from path: {path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


class DeferredFactory(BaseModel):
    """Serializable recipe for creating objects with non-serializable state.

    Instead of storing a live object (e.g., a Prior with PyTensor tensor
    parameters), store the factory function path and its scalar arguments.
    Call ``resolve()`` at build_model() time to create the actual object.
    """

    factory: str = Field(
        ..., description="Fully-qualified dotted path to the factory function"
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Scalar keyword arguments for the factory"
    )

    def resolve(self) -> Any:
        """Import the factory function and call it with kwargs."""
        fn = _import_from_dotted_path(self.factory)
        return fn(**self.kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "__deferred__": True,
            "factory": self.factory,
            "kwargs": self.kwargs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeferredFactory:
        return cls(factory=data["factory"], kwargs=data.get("kwargs", {}))


@dataclass
class _RegistryEntry:
    cls: type
    serializer: Any = None
    deserializer: Any = None


class TypeRegistry:
    """Centralized registry for serializable types.

    Replaces scattered ``register_deserialization`` calls, ``RegistrationMeta``
    metaclasses, ``singledispatch`` handlers, and lookup dicts.

    Usage::

        # As a bare decorator (type_key auto-derived):
        @registry.register
        class MyClass:
            def to_dict(self): ...
            @classmethod
            def from_dict(cls, data): ...

        # With explicit type_key + custom deserializer:
        registry.register("mod.MyClass", MyClass, deserializer=my_deser_fn)

    """

    def __init__(self) -> None:
        self._registry: dict[str, _RegistryEntry] = {}

    def register(
        self,
        cls_or_key: type | str | None = None,
        cls: type | None = None,
        *,
        serializer: Any = None,
        deserializer: Any = None,
    ):
        """Register a class for serialization/deserialization.

        Can be used as a bare decorator, a decorator factory, or a direct call.
        """
        if cls_or_key is None:
            return lambda c: self.register(c, serializer=serializer, deserializer=deserializer)

        if isinstance(cls_or_key, type):
            actual_cls = cls_or_key
            type_key = f"{actual_cls.__module__}.{actual_cls.__qualname__}"
        elif isinstance(cls_or_key, str):
            type_key = cls_or_key
            actual_cls = cls
            if actual_cls is None:
                raise TypeError(
                    f"When registering with a string key ({type_key!r}), "
                    "the class must be provided as the second argument."
                )
        else:
            raise TypeError(
                f"First argument must be a class or string, got {type(cls_or_key)}"
            )

        self._registry[type_key] = _RegistryEntry(
            cls=actual_cls, serializer=serializer, deserializer=deserializer
        )
        return actual_cls

    def serialize(self, obj: Serializable) -> dict[str, Any]:
        """Serialize an object to a JSON-safe dict with ``__type__`` key."""
        type_key = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
        if type_key not in self._registry:
            raise KeyError(
                f"Type {type_key!r} is not registered in the TypeRegistry. "
                f"Use @registry.register to register it."
            )
        entry = self._registry[type_key]
        if entry.serializer is not None:
            return entry.serializer(obj)
        return obj.to_dict()

    def deserialize(
        self,
        data: dict[str, Any],
        context: DeserializationContext | None = None,
    ) -> Any:
        """Deserialize a dict back to an object.

        Three-tier dispatch:
        1. If ``__deferred__`` is True, return an unresolved ``DeferredFactory``.
        2. If a custom deserializer was registered, call it with ``(data, context)``.
        3. Otherwise, look up the class by ``__type__`` and call ``cls.from_dict(data)``.
        """
        if not isinstance(data, dict):
            raise SerializationError(
                f"Expected a dict for deserialization, got {type(data).__name__}"
            )

        if data.get("__deferred__"):
            return DeferredFactory.from_dict(data)

        type_key = data.get("__type__")
        if type_key is None:
            raise SerializationError(
                "Dict is missing '__type__' key. Cannot determine which class "
                "to deserialize to. Ensure the object was serialized with "
                "registry.serialize() or a to_dict() that includes '__type__'."
            )

        if type_key not in self._registry:
            raise SerializationError(
                f"Unknown type {type_key!r}. The class may not have been "
                f"registered with @registry.register, or the module defining "
                f"it may not have been imported. "
                f"Registered types: {sorted(self._registry.keys())}"
            )

        entry = self._registry[type_key]

        if entry.deserializer is not None:
            return entry.deserializer(data, context)

        return entry.cls.from_dict(data)


registry = TypeRegistry()


class SerializableMixin:
    """Mixin that auto-implements Serializable for Pydantic BaseModel classes.

    - Provides default ``to_dict()`` / ``from_dict()`` via
      ``model_dump(mode="json")`` / ``model_validate()``.
    - Auto-registers concrete subclasses in the module-level ``registry``
      via ``__init_subclass__`` (no decorator needed).
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            type_key = f"{cls.__module__}.{cls.__qualname__}"
            registry.register(type_key, cls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            **self.model_dump(mode="json"),  # type: ignore[attr-defined]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        filtered = {k: v for k, v in data.items() if k != "__type__"}
        return cls.model_validate(filtered)  # type: ignore[attr-defined]
```

### Step 2: Run pre-commit

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pre_commit run --files pymc_marketing/serialization.py tests/test_serialization.py`

### Step 3: Run tests

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/test_serialization.py -v --no-header -x`

Expected: ALL PASS

### Step 4: Run existing tests to verify no regressions

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/components/ tests/mmm/test_hsgp.py tests/mmm/test_fourier.py tests/mmm/test_events.py tests/mmm/test_media_transformation.py tests/mmm/test_additive_effect.py -x --no-header -q`

Expected: ALL PASS (no regressions — we only added a new module)

### Step 5: Commit

```bash
git add pymc_marketing/serialization.py tests/test_serialization.py
git commit -m "feat: add core serialization infrastructure (TypeRegistry, DeferredFactory, SerializableMixin)"
```

---

## Task 3: Register Transformation Base + Adstock Classes

**Files:**
- Modify: `pymc_marketing/mmm/components/base.py` (lines 169–185 — `to_dict()`)
- Modify: `pymc_marketing/mmm/components/adstock.py` (lines 81–142 — class definitions + `to_dict()`)
- Test: `tests/mmm/components/test_adstock.py`

### Step 1: Write the failing tests

Add to existing `tests/mmm/components/test_adstock.py`:

```python
class TestAdstockTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of adstock classes."""

    @pytest.mark.parametrize(
        "adstock_cls", ALL_ADSTOCK_CLASSES, ids=lambda c: c.__name__
    )
    def test_to_dict_includes_type_key(self, adstock_cls):
        """to_dict() output must include __type__ with fully-qualified class path."""
        obj = adstock_cls(l_max=4)
        data = obj.to_dict()
        assert "__type__" in data
        expected = f"{adstock_cls.__module__}.{adstock_cls.__qualname__}"
        assert data["__type__"] == expected

    @pytest.mark.parametrize(
        "adstock_cls", ALL_ADSTOCK_CLASSES, ids=lambda c: c.__name__
    )
    def test_registered_in_type_registry(self, adstock_cls):
        from pymc_marketing.serialization import registry

        type_key = f"{adstock_cls.__module__}.{adstock_cls.__qualname__}"
        assert type_key in registry._registry, f"{adstock_cls.__name__} not registered"

    @pytest.mark.parametrize(
        "adstock_cls", ALL_ADSTOCK_CLASSES, ids=lambda c: c.__name__
    )
    def test_roundtrip_all_parameters(self, adstock_cls):
        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry

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
```

### Step 2: Run tests to verify they fail

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/components/test_adstock.py::TestAdstockTypeRegistry -v --no-header -x`

Expected: FAIL — `__type__` not in `to_dict()` output

### Step 3: Implement the changes

**3a. Update `Transformation.to_dict()` in `base.py`** to include `__type__`:

In `pymc_marketing/mmm/components/base.py`, find the existing `to_dict()` method (line ~169) and add `__type__` to the returned dict:

```python
def to_dict(self) -> dict[str, Any]:
    return {
        "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
        "lookup_name": self.lookup_name,
        "prefix": self.prefix,
        "priors": {
            key: _serialize_value(value)
            for key, value in self.function_priors.items()
        },
    }
```

**3b. Add `from_dict()` classmethod on `AdstockTransformation`** in `adstock.py`:

After the existing `to_dict()` method (~line 142), add:

```python
@classmethod
def from_dict(cls, data: dict) -> "AdstockTransformation":
    """Reconstruct an adstock transformation from a dict."""
    data = data.copy()
    data.pop("__type__", None)
    data.pop("lookup_name", None)

    if "priors" in data:
        from pymc_extras.deserialize import deserialize

        data["priors"] = {k: deserialize(v) for k, v in data["priors"].items()}

    if "mode" in data:
        data["mode"] = data["mode"]

    return cls(**data)
```

**3c. Register all adstock classes** with `@registry.register`:

At the top of `adstock.py`, add the import:

```python
from pymc_marketing.serialization import registry
```

Then add `@registry.register` decorator to each concrete class: `BinomialAdstock`, `GeometricAdstock`, `DelayedAdstock`, `WeibullPDFAdstock`, `WeibullCDFAdstock`, `NoAdstock`.

Example for `GeometricAdstock`:

```python
@registry.register
class GeometricAdstock(AdstockTransformation):
    lookup_name = "geometric"
    # ... rest unchanged
```

### Step 4: Run pre-commit, then run tests

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pre_commit run --files pymc_marketing/mmm/components/base.py pymc_marketing/mmm/components/adstock.py tests/mmm/components/test_adstock.py`

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/components/test_adstock.py -v --no-header -x`

Expected: ALL PASS (new and existing tests)

### Step 5: Commit

```bash
git add pymc_marketing/mmm/components/base.py pymc_marketing/mmm/components/adstock.py tests/mmm/components/test_adstock.py
git commit -m "feat: add __type__ and TypeRegistry support to Transformation/Adstock classes"
```

---

## Task 4: Register Saturation Classes

**Files:**
- Modify: `pymc_marketing/mmm/components/saturation.py`
- Test: `tests/mmm/components/test_saturation.py`

### Step 1: Write the failing tests

Add to `tests/mmm/components/test_saturation.py`:

First, add to the top of the file (alongside existing imports):

```python
import inspect

import pymc_marketing.mmm.components.saturation as saturation_module

ALL_SATURATION_CLASSES: list[type[SaturationTransformation]] = [
    cls
    for _, cls in inspect.getmembers(saturation_module, inspect.isclass)
    if issubclass(cls, SaturationTransformation) and cls is not SaturationTransformation
]
```

Then add the test class:

```python
class TestSaturationTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of saturation classes."""

    @pytest.mark.parametrize(
        "sat_cls", ALL_SATURATION_CLASSES, ids=lambda c: c.__name__
    )
    def test_to_dict_includes_type_key(self, sat_cls):
        """to_dict() output must include __type__ with fully-qualified class path."""
        obj = sat_cls()
        data = obj.to_dict()
        assert "__type__" in data
        expected = f"{sat_cls.__module__}.{sat_cls.__qualname__}"
        assert data["__type__"] == expected

    @pytest.mark.parametrize(
        "sat_cls", ALL_SATURATION_CLASSES, ids=lambda c: c.__name__
    )
    def test_registered_in_type_registry(self, sat_cls):
        from pymc_marketing.serialization import registry

        type_key = f"{sat_cls.__module__}.{sat_cls.__qualname__}"
        assert type_key in registry._registry, f"{sat_cls.__name__} not registered"

    @pytest.mark.parametrize(
        "sat_cls", ALL_SATURATION_CLASSES, ids=lambda c: c.__name__
    )
    def test_roundtrip_all_parameters(self, sat_cls):
        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry

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
```

### Step 2: Run tests to verify they fail

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/components/test_saturation.py::TestSaturationTypeRegistry -v --no-header -x`

Expected: FAIL — `__type__` not in output

### Step 3: Implement

Same pattern as adstock:

**3a. Add `from_dict()` classmethod on `SaturationTransformation`:**

```python
@classmethod
def from_dict(cls, data: dict) -> "SaturationTransformation":
    """Reconstruct a saturation transformation from a dict."""
    data = data.copy()
    data.pop("__type__", None)
    data.pop("lookup_name", None)

    if "priors" in data:
        from pymc_extras.deserialize import deserialize

        data["priors"] = {k: deserialize(v) for k, v in data["priors"].items()}

    return cls(**data)
```

**3b. Register all saturation classes** with `@registry.register`:

Add `from pymc_marketing.serialization import registry` import and `@registry.register` to each concrete class.

### Step 4: Run pre-commit and tests

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pre_commit run --files pymc_marketing/mmm/components/saturation.py tests/mmm/components/test_saturation.py`

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/components/test_saturation.py -v --no-header -x`

Expected: ALL PASS

### Step 5: Commit

```bash
git add pymc_marketing/mmm/components/saturation.py tests/mmm/components/test_saturation.py
git commit -m "feat: add __type__ and TypeRegistry support to Saturation classes"
```

---

## Task 5: Register Basis + EventEffect Classes

**Files:**
- Modify: `pymc_marketing/mmm/events.py`
- Test: `tests/mmm/test_events.py`

### Step 1: Write the failing tests

Add to `tests/mmm/test_events.py`:

```python
class TestEventsTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of Basis and EventEffect."""

    @pytest.mark.parametrize(
        "basis_cls,kwargs",
        [
            (GaussianBasis, {"prefix": "custom_basis", "priors": {"sigma": Prior("Gamma", mu=5, sigma=2)}}),
            (HalfGaussianBasis, {"mode": "before", "include_event": False, "prefix": "hg_basis", "priors": {"sigma": Prior("Gamma", mu=5, sigma=2)}}),
            (AsymmetricGaussianBasis, {"event_in": "before", "prefix": "ag_basis", "priors": {"sigma_before": Prior("Gamma", mu=2, sigma=0.5), "sigma_after": Prior("Gamma", mu=5, sigma=1), "a_after": Prior("Normal", mu=0.5, sigma=0.3)}}),
        ],
        ids=["GaussianBasis", "HalfGaussianBasis", "AsymmetricGaussianBasis"],
    )
    def test_basis_roundtrip_all_parameters(self, basis_cls, kwargs):
        from pymc_marketing.serialization import registry

        original = basis_cls(**kwargs)
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is basis_cls
        assert restored.prefix == kwargs["prefix"]
        for prior_name, prior in kwargs["priors"].items():
            assert restored.function_priors[prior_name] == prior
        assert restored == original

    @pytest.mark.parametrize(
        "basis_cls",
        [GaussianBasis, HalfGaussianBasis, AsymmetricGaussianBasis],
        ids=lambda c: c.__name__,
    )
    def test_basis_to_dict_includes_type_key(self, basis_cls):
        obj = basis_cls()
        data = obj.to_dict()
        assert "__type__" in data
        expected = f"{basis_cls.__module__}.{basis_cls.__qualname__}"
        assert data["__type__"] == expected

    @pytest.mark.parametrize(
        "basis_cls",
        [GaussianBasis, HalfGaussianBasis, AsymmetricGaussianBasis],
        ids=lambda c: c.__name__,
    )
    def test_basis_registered_in_type_registry(self, basis_cls):
        from pymc_marketing.serialization import registry

        type_key = f"{basis_cls.__module__}.{basis_cls.__qualname__}"
        assert type_key in registry._registry, f"{basis_cls.__name__} not registered"

    def test_event_effect_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry

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
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is EventEffect
        assert type(restored.basis) is GaussianBasis
        assert restored.dims == original.dims
        assert restored.basis.prefix == "ev_basis"
        assert restored == original
```

### Step 2: Run tests to verify they fail

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_events.py::TestEventsTypeRegistry -v --no-header -x`

Expected: FAIL

### Step 3: Implement

**3a. Add `from_dict()` classmethod on `Basis`:**

```python
@classmethod
def from_dict(cls, data: dict) -> "Basis":
    """Reconstruct a basis from a dict."""
    data = data.copy()
    data.pop("__type__", None)
    data.pop("lookup_name", None)

    if "priors" in data:
        from pymc_extras.deserialize import deserialize

        data["priors"] = {k: deserialize(v) for k, v in data["priors"].items()}

    return cls(**data)
```

**3b. Register basis classes** with `@registry.register`:

Add `from pymc_marketing.serialization import registry` and `@registry.register` to `GaussianBasis`, `HalfGaussianBasis`, `AsymmetricGaussianBasis`.

**3c. Update `EventEffect.to_dict()`** to include `__type__`:

```python
def to_dict(self) -> dict:
    return {
        "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
        "class": "EventEffect",
        "data": {
            "basis": self.basis.to_dict(),
            "effect_size": self.effect_size.to_dict(),
            "dims": self.dims,
        },
    }
```

**3d. Update `EventEffect.from_dict()`** to handle `__type__`:

```python
@classmethod
def from_dict(cls, data: dict) -> "EventEffect":
    data_inner = data.get("data", data)
    if "__type__" in data_inner:
        data_inner = {k: v for k, v in data_inner.items() if k != "__type__"}
    return cls(
        basis=deserialize(data_inner["basis"]),
        effect_size=deserialize(data_inner["effect_size"]),
        dims=data_inner["dims"],
    )
```

**3e. Register `EventEffect`:**

```python
@registry.register
class EventEffect(BaseModel):
    ...
```

### Step 4: Run pre-commit and tests

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pre_commit run --files pymc_marketing/mmm/events.py tests/mmm/test_events.py`

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_events.py -v --no-header -x`

Expected: ALL PASS

### Step 5: Commit

```bash
git add pymc_marketing/mmm/events.py tests/mmm/test_events.py
git commit -m "feat: add __type__ and TypeRegistry support to Basis/EventEffect classes"
```

---

## Task 6: Register HSGP Classes + DeferredFactory Integration

**Files:**
- Modify: `pymc_marketing/mmm/hsgp.py`
- Test: `tests/mmm/test_hsgp.py`

This is the most complex task. It involves three changes:
1. Add `__type__` to `to_dict()` and register with `@registry.register`
2. Fix `dims` tuple→list in `from_dict()`
3. Add `DeferredFactory` support for `eta`/`ls` fields

### Step 1: Write the failing tests

Add to `tests/mmm/test_hsgp.py`:

```python
class TestHSGPTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of HSGP classes."""

    @pytest.mark.parametrize(
        "hsgp_cls", [HSGP, HSGPPeriodic, SoftPlusHSGP], ids=lambda c: c.__name__
    )
    def test_to_dict_includes_type_key(self, hsgp_cls):
        if hsgp_cls is HSGPPeriodic:
            obj = hsgp_cls(m=10, scale=1.0, ls=1.0, period=365.25, dims="time")
        else:
            obj = hsgp_cls(m=10, L=1.5, eta=1.0, ls=1.0, dims="time")
        data = obj.to_dict()
        assert "__type__" in data
        expected = f"{hsgp_cls.__module__}.{hsgp_cls.__qualname__}"
        assert data["__type__"] == expected

    @pytest.mark.parametrize(
        "hsgp_cls", [HSGP, HSGPPeriodic, SoftPlusHSGP], ids=lambda c: c.__name__
    )
    def test_registered_in_type_registry(self, hsgp_cls):
        from pymc_marketing.serialization import registry

        type_key = f"{hsgp_cls.__module__}.{hsgp_cls.__qualname__}"
        assert type_key in registry._registry, f"{hsgp_cls.__name__} not registered"

    def test_hsgp_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.hsgp_kwargs import CovFunc
        from pymc_marketing.serialization import registry

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

    def test_hsgp_periodic_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry

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

    def test_softplus_hsgp_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry

        original = SoftPlusHSGP(
            m=20,
            L=3.0,
            eta=Prior("Exponential", lam=1.0),
            ls=2.0,
            dims="time",
            centered=True,
            drop_first=False,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is SoftPlusHSGP
        assert restored.m == 20
        assert restored.L == 3.0
        assert restored.centered is True
        assert restored.drop_first is False
        assert restored == original


class TestDeferredFactoryInHSGP:
    def test_deferred_factory_in_eta(self):
        from pymc_marketing.serialization import DeferredFactory

        deferred = DeferredFactory(
            factory="pymc_marketing.mmm.hsgp.create_eta_prior",
            kwargs={"upper": 5.0, "mass": 0.95},
        )
        hsgp = HSGP(m=10, L=1.5, eta=deferred, ls=1.0, dims="time")
        data = hsgp.to_dict()
        assert data["eta"]["__deferred__"] is True
        assert data["eta"]["factory"] == "pymc_marketing.mmm.hsgp.create_eta_prior"

    def test_deferred_roundtrip_all_parameters(self):
        from pymc_marketing.serialization import DeferredFactory, registry

        deferred_eta = DeferredFactory(
            factory="pymc_marketing.mmm.hsgp.create_eta_prior",
            kwargs={"upper": 5.0, "mass": 0.95},
        )
        deferred_ls = DeferredFactory(
            factory="pymc_marketing.mmm.hsgp.create_ls_prior",
            kwargs={"mu": 3.0, "sigma": 1.0},
        )
        original = HSGP(
            m=12, L=2.0, eta=deferred_eta, ls=deferred_ls,
            dims=("time", "geo"), centered=True,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is HSGP
        assert isinstance(restored.eta, DeferredFactory)
        assert isinstance(restored.ls, DeferredFactory)
        assert restored.eta.factory == deferred_eta.factory
        assert restored.eta.kwargs == deferred_eta.kwargs
        assert restored.ls.factory == deferred_ls.factory
        assert restored.m == 12
        assert restored.L == 2.0
        assert restored.dims == ("time", "geo")
        assert restored.centered is True
        assert restored.eta.resolve() is not None
```

### Step 2: Run tests to verify they fail

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_hsgp.py::TestHSGPTypeRegistry tests/mmm/test_hsgp.py::TestDeferredFactoryInHSGP -v --no-header -x`

Expected: FAIL

### Step 3: Implement

**3a. Update `HSGPBase.to_dict()`** (line ~346 in `hsgp.py`) to include `__type__`:

```python
def to_dict(self) -> dict:
    data = self.model_dump()

    def handle_prior(value):
        return value if not hasattr(value, "to_dict") else value.to_dict()

    result = {key: handle_prior(value) for key, value in data.items()}
    result["__type__"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
    return result
```

**3b. Fix `dims` tuple→list in `HSGP.from_dict()`** (line ~928):

```python
@classmethod
def from_dict(cls, data) -> HSGP:
    data = data.copy()
    data.pop("__type__", None)
    data.pop("hsgp_class", None)

    if "dims" in data and isinstance(data["dims"], list):
        data["dims"] = tuple(data["dims"])

    for key in ["eta", "ls"]:
        value = data.get(key)
        if isinstance(value, dict):
            if value.get("__deferred__"):
                from pymc_marketing.serialization import DeferredFactory
                data[key] = DeferredFactory.from_dict(value)
            else:
                data[key] = Prior.from_dict(value)

    return cls(**data)
```

**3c. Fix `dims` in `HSGPPeriodic.from_dict()`** (line ~1281):

Same pattern — add `data.pop("__type__", None)`, `data.pop("hsgp_class", None)`, and `list→tuple` for dims.

**3d. Add `DeferredFactory` to HSGP field types:**

Update the `eta` and `ls` field annotations to accept `DeferredFactory`:

```python
from pymc_marketing.serialization import DeferredFactory

# On HSGP class:
ls: InstanceOf[VariableFactory] | DeferredFactory | float = Field(...)
eta: InstanceOf[VariableFactory] | DeferredFactory | float = Field(...)
```

And in `HSGPBase.to_dict()`, handle `DeferredFactory` in the `handle_prior` function:

```python
def handle_prior(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value
```

(This already handles `DeferredFactory` since it has `to_dict()`.)

**3e. Register HSGP classes:**

```python
from pymc_marketing.serialization import registry

@registry.register
class HSGP(HSGPBase):
    ...

@registry.register
class HSGPPeriodic(HSGPBase):
    ...

@registry.register
class SoftPlusHSGP(HSGP):
    ...
```

### Step 4: Run pre-commit and tests

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pre_commit run --files pymc_marketing/mmm/hsgp.py tests/mmm/test_hsgp.py`

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_hsgp.py -v --no-header -x`

Expected: ALL PASS

### Step 5: Commit

```bash
git add pymc_marketing/mmm/hsgp.py tests/mmm/test_hsgp.py
git commit -m "feat: add TypeRegistry + DeferredFactory + dims fix for HSGP classes"
```
