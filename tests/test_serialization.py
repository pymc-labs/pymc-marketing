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

from pymc_marketing.serialization import serialization


def _make_effect_class() -> type:
    """Return a fresh class so registering it (which patches ``to_dict``) stays test-local."""

    class MyEffect:
        def __init__(self, a: int = 1) -> None:
            self.a = a

        def to_dict(self) -> dict[str, Any]:
            return {"a": self.a}

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> MyEffect:
            return cls(data["a"])

    return MyEffect


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

    def test_register_returns_decorator_when_none(self):
        """register(None) should return a decorator lambda."""
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()

        # register(None) returns a decorator
        decorator = reg.register(None)
        assert callable(decorator)

        # The decorator should register the class
        class MyClass:
            def to_dict(self):
                return {}

            @classmethod
            def from_dict(cls, data):
                return cls()

        registered = decorator(MyClass)
        assert registered is MyClass

        type_key = f"{MyClass.__module__}.{MyClass.__qualname__}"
        assert type_key in reg._registry

    def test_register_string_key_without_class_raises(self):
        """register(str_key) without class should raise TypeError."""
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()
        with pytest.raises(TypeError, match="class must be provided"):
            reg.register("some.string.key")

    def test_string_key_is_written_and_read(self):
        """A class registered only under a string key round-trips under that key (GH-3330)."""
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()
        MyEffect = _make_effect_class()
        reg.register("my_package.MyEffect", MyEffect)
        effect = MyEffect(3)

        expected = {"__type__": "my_package.MyEffect", "a": 3}
        assert reg.is_registered(effect)
        assert reg.serialize(effect) == expected
        assert effect.to_dict() == expected
        restored = reg.deserialize(effect.to_dict())
        assert isinstance(restored, MyEffect)
        assert restored.a == 3

    @pytest.mark.parametrize(
        "primary_is_string", [True, False], ids=["string", "class"]
    )
    def test_first_registered_key_is_written_and_later_keys_still_load(
        self, primary_is_string
    ):
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()
        MyEffect = _make_effect_class()
        if primary_is_string:
            primary = "my_package.MyEffect"
            reg.register(primary, MyEffect)
        else:
            primary = f"{MyEffect.__module__}.{MyEffect.__qualname__}"
            reg.register(MyEffect)
        reg.register("old_package.MyEffect", MyEffect)

        assert reg.serialize(MyEffect(3)) == {"__type__": primary, "a": 3}
        restored = reg.deserialize({"__type__": "old_package.MyEffect", "a": 4})
        assert isinstance(restored, MyEffect)
        assert restored.a == 4

    def test_serialize_writes_its_own_key_when_another_registry_wrapped_the_class(
        self,
    ):
        from pymc_marketing.serialization import TypeRegistry

        first, second = TypeRegistry(), TypeRegistry()
        MyEffect = _make_effect_class()
        first.register(MyEffect)
        second.register("my_package.MyEffect", MyEffect)

        data = second.serialize(MyEffect(2))
        assert data == {"__type__": "my_package.MyEffect", "a": 2}
        assert second.deserialize(data).a == 2

    def test_key_registered_for_another_class_is_no_longer_written_by_the_first(self):
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()
        MyEffect = _make_effect_class()
        reg.register("my_package.MyEffect", MyEffect)

        class OtherEffect(MyEffect):
            pass

        reg.register("my_package.MyEffect", OtherEffect)

        assert not reg.is_registered(MyEffect(1))
        with pytest.raises(KeyError):
            reg.serialize(MyEffect(1))
        data = reg.serialize(OtherEffect(2))
        assert data == {"__type__": "my_package.MyEffect", "a": 2}
        assert type(reg.deserialize(data)) is OtherEffect

    def test_model_config_uses_serializer_of_string_keyed_class(self):
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()
        MyEffect = _make_effect_class()
        reg.register(
            "my_package.MyEffect",
            MyEffect,
            serializer=lambda obj: {"__type__": "my_package.MyEffect", "b": obj.a},
        )

        serialized = reg.serialize_model_config({"effect": MyEffect(3)})
        assert serialized == {"effect": {"__type__": "my_package.MyEffect", "b": 3}}

    def test_subclass_of_string_keyed_class_uses_its_own_key(self):
        from pymc_marketing.serialization import TypeRegistry

        reg = TypeRegistry()
        MyEffect = _make_effect_class()
        reg.register("my_package.MyEffect", MyEffect)

        class SubEffect(MyEffect):
            pass

        assert not reg.is_registered(SubEffect(5))

        reg.register(SubEffect)
        data = SubEffect(5).to_dict()
        assert data["__type__"] == f"{SubEffect.__module__}.{SubEffect.__qualname__}"
        assert type(reg.deserialize(data)) is SubEffect


class TestSerializableBaseModel:
    def test_auto_registration(self):
        from pymc_marketing.serialization import SerializableBaseModel

        class AutoReg(SerializableBaseModel):
            x: int = 1

        type_key = f"{AutoReg.__module__}.{AutoReg.__qualname__}"

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
        from pymc_marketing.serialization import SerializableBaseModel

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

        from pymc_marketing.serialization import SerializableBaseModel

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

    def test_string_key_is_written_instead_of_auto_registered_module_path(self):
        """Auto-registration does not pin the written key; an explicit string key does (GH-3330)."""
        from pymc_marketing.serialization import SerializableBaseModel

        class StableKeyModel(SerializableBaseModel):
            a: int = 1

        module_key = f"{StableKeyModel.__module__}.{StableKeyModel.__qualname__}"
        serialization.register("tests.serialization.StableKeyModel", StableKeyModel)

        data = serialization.serialize(StableKeyModel(a=2))
        assert data == {"__type__": "tests.serialization.StableKeyModel", "a": 2}
        assert serialization.deserialize(data) == StableKeyModel(a=2)
        # Dicts saved under the module path before the key was registered still load.
        legacy = {"__type__": module_key, "a": 3}
        assert serialization.deserialize(legacy) == StableKeyModel(a=3)

    def test_registering_the_class_first_keeps_writing_the_module_path(self):
        """A key registered after ``register(cls)`` only loads (GH-3330)."""
        from pymc_marketing.serialization import SerializableBaseModel

        class MovedModel(SerializableBaseModel):
            a: int = 1

        old_key = "tests.serialization.old_module.MovedModel"
        serialization.register(MovedModel)
        serialization.register(old_key, MovedModel)

        module_key = f"{MovedModel.__module__}.{MovedModel.__qualname__}"
        data = serialization.serialize(MovedModel(a=2))
        assert data == {"__type__": module_key, "a": 2}
        assert serialization.deserialize({"__type__": old_key, "a": 3}) == MovedModel(
            a=3
        )


class TestSerializeModelConfig:
    """Tests for single-pass model_config serialization with shared memo."""

    def test_shared_decomposition_is_embedded_inline(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.r2d2 import R2D2

        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.2),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        config = {
            "gamma_control": r2d2.split("control"),
            "likelihood": Prior("Normal", sigma=r2d2.error_sigma),
        }

        serialized = serialization.serialize_model_config(config)

        assert "gamma_control" in serialized
        assert "likelihood" in serialized

        gamma_decomp = serialized["gamma_control"]["decomposition"]
        likelihood_sigma = serialized["likelihood"]["kwargs"]["sigma"]
        likelihood_decomp = likelihood_sigma["decomposition"]
        assert gamma_decomp["__type__"].endswith("R2D2")
        assert likelihood_decomp["__type__"].endswith("R2D2")
        assert gamma_decomp == likelihood_decomp

    def test_prior_with_r2d2_sigma_survives_round_trip(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.r2d2 import R2D2, R2D2Sigma

        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.2),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        config = {
            "gamma_control": r2d2.split("control"),
            "likelihood": Prior("Normal", sigma=r2d2.error_sigma),
        }

        serialized = serialization.serialize_model_config(config)
        restored = serialization.deserialize_model_config(serialized)

        assert "gamma_control" in restored
        assert "likelihood" in restored
        assert isinstance(restored["gamma_control"].decomposition, R2D2)
        assert isinstance(restored["likelihood"].parameters["sigma"], R2D2Sigma)

    def test_shared_decomposition_preserved_after_round_trip(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.r2d2 import R2D2

        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.2),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        config = {
            "gamma_control": r2d2.split("control"),
            "likelihood": Prior("Normal", sigma=r2d2.error_sigma),
        }

        serialized = serialization.serialize_model_config(config)
        restored = serialization.deserialize_model_config(serialized)

        split_decomp = restored["gamma_control"].decomposition
        sigma_obj = restored["likelihood"].parameters["sigma"]
        sigma_decomp = sigma_obj.decomposition

        assert split_decomp is sigma_decomp, (
            "Shared decomposition should be the same object after round-trip"
        )

    def test_plain_values_passthrough(self):
        import numpy as np

        config = {
            "int_val": 42,
            "str_val": "hello",
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None,
            "list_val": [1, 2, 3],
            "dict_val": {"a": 1, "b": 2},
        }

        serialized = serialization.serialize_model_config(config)
        restored = serialization.deserialize_model_config(serialized)

        assert restored["int_val"] == config["int_val"]
        assert restored["str_val"] == config["str_val"]
        assert restored["float_val"] == config["float_val"]
        assert restored["bool_val"] == config["bool_val"]
        assert restored["none_val"] == config["none_val"]
        np.testing.assert_array_equal(restored["list_val"], config["list_val"])
        assert restored["dict_val"] == config["dict_val"]

    def test_numpy_arrays_converted(self):
        import numpy as np

        config = {
            "arr": np.array([1.0, 2.0, 3.0]),
            "scalar": np.float64(4.0),
        }

        serialized = serialization.serialize_model_config(config)
        restored = serialization.deserialize_model_config(serialized)

        np.testing.assert_array_equal(restored["arr"], [1.0, 2.0, 3.0])
        assert restored["scalar"] == 4.0

    @pytest.mark.parametrize(
        "case",
        [
            "no-transform",
            "transform",
            "nested-transform",
            "non-centered-transform",
            "array-parameter",
            "censored-parameter",
        ],
    )
    def test_prior_survives_json_round_trip(self, case):
        """Regression test for #3225: a ``Prior`` and its ``transform`` survive save and load."""
        import json

        import numpy as np
        from pymc_extras.prior import Censored, Prior

        prior = {
            "no-transform": Prior("Normal", mu=0, sigma=1, dims="channel"),
            "transform": Prior(
                "Normal", mu=0, sigma=1, dims="channel", transform="abs"
            ),
            "nested-transform": Prior(
                "Normal", mu=Prior("Normal", transform="exp"), sigma=1, dims="channel"
            ),
            "non-centered-transform": Prior(
                "Normal",
                mu=0,
                sigma=Prior("HalfNormal"),
                dims=("geo", "channel"),
                centered=False,
                transform="abs",
            ),
            "array-parameter": Prior(
                "Normal",
                mu=np.array([1.0, 2.0]),
                sigma=1,
                dims="channel",
                transform="exp",
            ),
            "censored-parameter": Prior(
                "Normal",
                mu=0,
                sigma=Censored(Prior("Normal", sigma=1), lower=0),
                transform="exp",
            ),
        }[case]

        serialized = json.loads(
            json.dumps(serialization.serialize_model_config({"prior": prior}))
        )
        restored = serialization.deserialize_model_config(serialized)["prior"]

        assert restored == prior

    @pytest.mark.parametrize("factor_dtype", ["float64", "float32", "int64"])
    def test_prior_with_scaled_parameter_survives_json_round_trip(self, factor_dtype):
        """A ``Scaled`` parameter is written in the format ``Prior.from_dict`` reads.

        ``float32`` and ``int64`` factors are numpy scalars that ``json`` cannot encode.
        """
        import json

        import numpy as np
        from pymc_extras.prior import Prior, Scaled

        inner = Prior("HalfNormal", sigma=1, transform="exp")
        factor = np.dtype(factor_dtype).type(2)
        prior = Prior("Normal", mu=0, sigma=Scaled(inner, factor=factor))

        serialized = json.loads(
            json.dumps(serialization.serialize_model_config({"prior": prior}))
        )
        restored = serialization.deserialize_model_config(serialized)["prior"]

        restored_sigma = restored.parameters["sigma"]
        assert isinstance(restored_sigma, Scaled)
        assert restored_sigma.dist == inner
        assert restored_sigma.factor == 2.0

    def test_prior_without_transform_writes_no_transform_key(self):
        """A ``Prior`` without a transform is written without a ``transform`` key."""
        from pymc_extras.prior import Prior

        serialized = serialization.serialize_model_config(
            {"prior": Prior("Normal", mu=0, sigma=1, dims="channel")}
        )

        assert "transform" not in serialized["prior"]
