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
``SerializableBaseModel``, ``DeferredFactory``, and ``DeserializationContext``
that replace the scattered serialization patterns across MMM components.

Every serializable object produces a JSON-safe dict with a ``__type__`` key
(fully-qualified class path). The ``TypeRegistry`` dispatches deserialization
from that key alone.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field


class SerializationError(Exception):
    """Raised when serialization or deserialization fails."""


@runtime_checkable
class Serializable(Protocol):
    """Structural protocol for serializable objects."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this object to a JSON-safe dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct an instance from a dictionary."""
        ...


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
        """Serialize the deferred factory to a dict."""
        return {
            "__deferred__": True,
            "factory": self.factory,
            "kwargs": self.kwargs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeferredFactory:
        """Reconstruct a DeferredFactory from a dict."""
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
        @serialization.register
        class MyClass:
            def to_dict(self): ...
            @classmethod
            def from_dict(cls, data): ...


        # With explicit type_key + custom deserializer:
        serialization.register("mod.MyClass", MyClass, deserializer=my_deser_fn)

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
            return lambda c: self.register(
                c, serializer=serializer, deserializer=deserializer
            )

        if isinstance(cls_or_key, type):
            actual_cls = cls_or_key
            type_key = f"{actual_cls.__module__}.{actual_cls.__qualname__}"
        elif isinstance(cls_or_key, str):
            type_key = cls_or_key
            if cls is None:
                raise TypeError(
                    f"When registering with a string key ({type_key!r}), "
                    "the class must be provided as the second argument."
                )
            actual_cls = cls
        else:
            raise TypeError(
                f"First argument must be a class or string, got {type(cls_or_key)}"
            )

        self._registry[type_key] = _RegistryEntry(
            cls=actual_cls, serializer=serializer, deserializer=deserializer
        )

        # Inject __type__ into to_dict() unless a custom serializer handles it,
        # or the class already inherits a wrapped to_dict.
        if serializer is None:
            resolved = getattr(actual_cls, "to_dict", None)
            if resolved is not None and not getattr(resolved, "_type_injected", False):
                original_to_dict = resolved  # resolved through MRO

                def _wrapped_to_dict(self, _orig=original_to_dict):
                    type_key = (
                        f"{self.__class__.__module__}.{self.__class__.__qualname__}"
                    )
                    return {"__type__": type_key, **_orig(self)}

                _wrapped_to_dict._type_injected = True  # type: ignore[attr-defined]
                actual_cls.to_dict = _wrapped_to_dict  # type: ignore[attr-defined]

        return actual_cls

    def serialize(self, obj: Serializable) -> dict[str, Any]:
        """Serialize an object to a JSON-safe dict with ``__type__`` key."""
        type_key = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
        if type_key not in self._registry:
            raise KeyError(
                f"Type {type_key!r} is not registered in the TypeRegistry. "
                f"Use @serialization.register to register it."
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
                "serialization.serialize() or a to_dict() that includes '__type__'."
            )

        if type_key not in self._registry:
            raise SerializationError(
                f"Unknown type {type_key!r}. The class may not have been "
                f"registered with @serialization.register, or the module defining "
                f"it may not have been imported. "
                f"Registered types: {sorted(self._registry.keys())}"
            )

        entry = self._registry[type_key]

        if entry.deserializer is not None:
            return entry.deserializer(data, context)

        return entry.cls.from_dict(data)  # type: ignore[attr-defined]


serialization = TypeRegistry()


class SerializableBaseModel(BaseModel):
    """Base model that auto-implements Serializable for Pydantic BaseModel subclasses.

    - Provides default ``to_dict()`` / ``from_dict()`` via
      ``model_dump(mode="json")`` / ``model_validate()``.
    - Auto-registers concrete subclasses in the module-level ``serialization``
      via ``__init_subclass__`` (no decorator needed).
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete subclasses in the module-level serialization."""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            type_key = f"{cls.__module__}.{cls.__qualname__}"
            serialization.register(type_key, cls)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict via Pydantic model_dump. ``__type__`` is injected by the registry wrapper."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct from a dict via Pydantic model_validate."""
        filtered = {k: v for k, v in data.items() if k != "__type__"}
        return cls.model_validate(filtered)
