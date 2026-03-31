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

        df = DeferredFactory(factory="builtins.dict", kwargs={"a": 1, "b": 2})
        result = df.to_dict()
        assert result == {
            "__deferred__": True,
            "factory": "builtins.dict",
            "kwargs": {"a": 1, "b": 2},
        }

    def test_from_dict(self):
        from pymc_marketing.serialization import DeferredFactory

        data = {
            "__deferred__": True,
            "factory": "builtins.dict",
            "kwargs": {"a": 1, "b": 2},
        }
        df = DeferredFactory.from_dict(data)
        assert df.factory == "builtins.dict"
        assert df.kwargs == {"a": 1, "b": 2}

    def test_resolve(self):
        from pymc_marketing.serialization import DeferredFactory

        df = DeferredFactory(factory="builtins.dict", kwargs={"a": 1, "b": 2})
        result = df.resolve()
        assert result == {"a": 1, "b": 2}

    def test_roundtrip(self):
        from pymc_marketing.serialization import DeferredFactory

        original = DeferredFactory(factory="builtins.dict", kwargs={"a": 1, "b": 2})
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
                return {"x": 1}

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
                return {"y": 2}

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
                return {"val": 99}

            @classmethod
            def from_dict(cls, data):
                return cls()

        result = reg.serialize(Baz())
        assert result == {
            "__type__": f"{Baz.__module__}.{Baz.__qualname__}",
            "val": 99,
        }

    def test_deserialize_standard(self):
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()

        @reg.register
        class Qux:
            def __init__(self, val=0):
                self.val = val

            def to_dict(self):
                return {"val": self.val}

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
            "factory": "builtins.dict",
            "kwargs": {"x": 7},
        }
        result = reg.deserialize(data)
        assert isinstance(result, DeferredFactory)
        assert result.resolve() == {"x": 7}

    def test_deserialize_custom_deserializer(self):
        from pymc_marketing.serialization import DeserializationContext, TypeRegistry

        reg = TypeRegistry()

        class Special:
            def __init__(self, extra=None):
                self.extra = extra

            def to_dict(self):
                return {"data": "x"}

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
                return {}

        with pytest.raises((KeyError, TypeError)):
            reg.serialize(Unknown())


class TestSerializableBaseModel:
    def test_auto_registration(self):
        from pymc_marketing.serialization import SerializableBaseModel

        class AutoReg(SerializableBaseModel):
            x: int = 1

        type_key = f"{AutoReg.__module__}.{AutoReg.__qualname__}"
        from pymc_marketing.serialization import serialization

        assert type_key in serialization._registry

    def test_to_dict(self):
        from pymc_marketing.serialization import SerializableBaseModel

        class MyModel(SerializableBaseModel):
            name: str = "test"
            value: int = 42

        obj = MyModel()
        result = obj.to_dict()
        assert result == {
            "__type__": f"{MyModel.__module__}.{MyModel.__qualname__}",
            "name": "test",
            "value": 42,
        }

    def test_from_dict(self):
        from pymc_marketing.serialization import SerializableBaseModel

        class MyModel2(SerializableBaseModel):
            name: str = "test"
            value: int = 42

        data = {"__type__": "whatever", "name": "hello", "value": 99}
        obj = MyModel2.from_dict(data)
        assert obj.name == "hello"
        assert obj.value == 99

    def test_roundtrip_via_registry(self):
        from pymc_marketing.serialization import SerializableBaseModel, serialization

        class RoundTripper(SerializableBaseModel):
            a: str = "foo"
            b: float = 3.14

        original = RoundTripper(a="bar", b=2.71)
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)
        assert isinstance(restored, RoundTripper)
        assert restored.a == "bar"
        assert restored.b == 2.71

    def test_abstract_subclass_not_registered(self):
        """Abstract subclasses should not be registered."""
        from abc import ABC, abstractmethod

        from pymc_marketing.serialization import SerializableBaseModel, serialization

        class AbstractEffect(SerializableBaseModel, ABC):
            @abstractmethod
            def do_thing(self): ...

        type_key = f"{AbstractEffect.__module__}.{AbstractEffect.__qualname__}"
        assert type_key not in serialization._registry

        class ConcreteEffect(AbstractEffect):
            val: int = 1

            def do_thing(self):
                return self.val

        concrete_key = f"{ConcreteEffect.__module__}.{ConcreteEffect.__qualname__}"
        assert concrete_key in serialization._registry
