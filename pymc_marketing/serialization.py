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
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field

# Context variable to thread serialization memo through nested to_dict() calls
_current_serialize_memo: ContextVar[_SerializeMemo | None] = ContextVar(
    "current_serialize_memo", default=None
)

# Context variable to thread deserialization memo through nested from_dict() calls
_current_deserialize_memo: ContextVar[_DeserializeMemo | None] = ContextVar(
    "current_deserialize_memo", default=None
)


def get_current_serialize_memo() -> _SerializeMemo | None:
    """Get the current serialization memo, or None if not in a serialization pass."""
    return _current_serialize_memo.get()


def get_current_deserialize_memo() -> _DeserializeMemo | None:
    """Get the current deserialization memo, or None if not in a deserialization pass."""
    return _current_deserialize_memo.get()


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
    idata : xr.DataTree or None
        The DataTree object being loaded from, used by deserializers
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


@dataclass
class _SerializeMemo:
    """Per-pass memo for serialization (replaces singleton tracker for serialization)."""

    _seen: dict[int, int] = field(default_factory=dict)
    _ref_counter: int = 0

    def track(self, obj: Any) -> None:
        """Track an object for deduplication."""
        self._seen[id(obj)] = self._ref_counter
        self._ref_counter += 1

    def is_seen(self, obj: Any) -> bool:
        """Check if object has been serialized before."""
        return id(obj) in self._seen

    def get_ref_id(self, obj: Any) -> int:
        """Get the reference ID for a tracked object."""
        return self._seen[id(obj)]


@dataclass
class _DeserializeMemo:
    """Per-pass memo for deserialization (replaces singleton tracker for deserialization)."""

    _deserialized: dict[int, Any] = field(default_factory=dict)
    _ref_counter: int = 0

    def store(self, obj: Any) -> int:
        """Store deserialized object and return its reference ID."""
        ref_id = self._ref_counter
        self._deserialized[ref_id] = obj
        self._ref_counter += 1
        return ref_id

    def resolve_ref(self, ref_id: int) -> Any:
        """Resolve a reference ID to a deserialized object."""
        return self._deserialized[ref_id]

    def has_ref(self, ref_id: int) -> bool:
        """Check if a reference ID exists."""
        return ref_id in self._deserialized


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

    Reference deduplication:
        When the same object appears multiple times during serialization,
        subsequent occurrences emit a ``{"$ref": N}`` reference instead of
        the full object. On deserialization, references are resolved to the
        first occurrence. This is transparent to registered classes.
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
        """Serialize an object to a JSON-safe dict with ``__type__`` key.

        Supports reference deduplication: if the same object appears multiple
        times during serialization, subsequent occurrences emit ``{"$ref": N}``
        references instead of duplicating the full object.
        """
        # Use fresh tracker per serialize pass (no cross-call leaking)
        memo = _SerializeMemo()
        token = _current_serialize_memo.set(memo)
        try:
            return self._serialize_with_refs(obj, memo)
        finally:
            _current_serialize_memo.reset(token)

    def serialize_batch(self, objs: list[Serializable]) -> list[dict[str, Any]]:
        """Serialize multiple objects in one pass, sharing reference tracking.

        Objects that appear multiple times across the batch are deduplicated:
        the first occurrence is emitted in full, subsequent occurrences emit
        ``{"$ref": N}`` references.
        """
        memo = _SerializeMemo()
        token = _current_serialize_memo.set(memo)
        try:
            return [self._serialize_with_refs(obj, memo) for obj in objs]
        finally:
            _current_serialize_memo.reset(token)

    def _serialize_with_refs(self, obj: Any, memo: _SerializeMemo) -> dict[str, Any]:
        """Serialize with reference tracking for deduplication."""
        # Emit reference for previously seen objects (include __type__ for parent dispatch)
        if memo.is_seen(obj):
            type_key = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
            return {"$ref": memo.get_ref_id(obj), "__type__": type_key}

        type_key = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
        if type_key not in self._registry:
            raise KeyError(
                f"Type {type_key!r} is not registered in the TypeRegistry. "
                f"Use @serialization.register to register it."
            )

        entry = self._registry[type_key]
        if entry.serializer is not None:
            serialized = entry.serializer(obj)
        else:
            serialized = obj.to_dict()

        # Store reference for deduplication
        memo.track(obj)

        return serialized

    def deserialize(
        self,
        data: dict[str, Any],
        context: DeserializationContext | None = None,
    ) -> Any:
        """Deserialize a dict back to an object.

        Three-tier dispatch:
        1. If ``$ref`` key exists, return previously deserialized object.
        2. If ``__deferred__`` is True, return an unresolved ``DeferredFactory``.
        3. Otherwise, look up the class by ``__type__`` and call ``cls.from_dict(data)``.
        """
        if not isinstance(data, dict):
            raise SerializationError(
                f"Expected a dict for deserialization, got {type(data).__name__}"
            )

        # Use fresh tracker per deserialize pass (no cross-call leaking)
        memo = _DeserializeMemo()
        token = _current_deserialize_memo.set(memo)
        try:
            return self._deserialize_with_refs(data, context, memo)
        finally:
            _current_deserialize_memo.reset(token)

    def deserialize_batch(
        self,
        items: list[dict[str, Any]],
        context: DeserializationContext | None = None,
    ) -> list[Any]:
        """Deserialize multiple objects in one pass, sharing reference tracking.

        Objects that were serialized with ``serialize_batch`` will have their
        shared references correctly resolved.
        """
        memo = _DeserializeMemo()
        token = _current_deserialize_memo.set(memo)
        try:
            return [self._deserialize_with_refs(item, context, memo) for item in items]
        finally:
            _current_deserialize_memo.reset(token)

    def _deserialize_with_refs(
        self,
        data: dict[str, Any],
        context: DeserializationContext | None,
        memo: _DeserializeMemo,
    ) -> Any:
        """Deserialize with reference tracking for deduplication."""
        # Handle references to previously deserialized objects
        if "$ref" in data:
            ref_id = data["$ref"]
            if memo.has_ref(ref_id):
                return memo.resolve_ref(ref_id)
            raise SerializationError(
                f"Reference $ref={ref_id} not found. "
                f"The referenced object may not have been deserialized yet."
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

        # Resolve any nested $ref dicts before passing to from_dict()
        resolved_data = self._resolve_nested_refs(data, memo)

        result = entry.cls.from_dict(resolved_data)  # type: ignore[attr-defined]

        # Store deserialized object for reference resolution
        memo.store(result)

        return result

    def _resolve_nested_refs(
        self, data: dict[str, Any], memo: _DeserializeMemo
    ) -> dict[str, Any]:
        """Recursively resolve $ref dicts in nested data structures."""
        resolved = {}
        for key, value in data.items():
            if isinstance(value, dict) and "$ref" in value:
                ref_id = value["$ref"]
                if memo.has_ref(ref_id):
                    resolved[key] = memo.resolve_ref(ref_id)
                else:
                    raise SerializationError(
                        f"Reference $ref={ref_id} not found in nested data. "
                        f"The referenced object may not have been deserialized yet."
                    )
            elif isinstance(value, dict):
                resolved[key] = self._resolve_nested_refs(value, memo)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_nested_refs(item, memo)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                resolved[key] = value
        return resolved

    def serialize_model_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Serialize a model_config dict in a single pass with shared memo.

        All values in the dict are serialized with the same ``_SerializeMemo``,
        so shared objects (e.g., an ``R2D2Decomposition`` referenced by both
        an ``R2D2Split`` and an ``R2D2Sigma``) are deduplicated via ``$ref``.

        Parameters
        ----------
        config : dict
            Raw model configuration dictionary with live objects
            (Priors, R2D2 splits, numpy arrays, etc.).

        Returns
        -------
        dict
            JSON-safe dict suitable for ``json.dumps``. Shared references
            become ``{"$ref": N, "__type__": ...}`` dicts.
        """
        memo = _SerializeMemo()
        token = _current_serialize_memo.set(memo)
        try:
            return {k: self._serialize_config_value(v, memo) for k, v in config.items()}
        finally:
            _current_serialize_memo.reset(token)

    def _serialize_config_value(self, value: Any, memo: _SerializeMemo) -> Any:
        """Serialize a single config value with shared memo support."""
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_config_value(v, memo) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_config_value(v, memo) for k, v in value.items()}

        from pymc_extras.prior import Prior

        if isinstance(value, Prior):
            result: dict[str, Any] = {
                "dist": value.distribution,
                "kwargs": {
                    k: self._serialize_config_value(v, memo)
                    for k, v in value.parameters.items()
                },
            }
            if value.dims is not None:
                result["dims"] = value.dims
            if not value.centered:
                result["centered"] = False
            return result

        type_key = f"{value.__class__.__module__}.{value.__class__.__qualname__}"
        if type_key in self._registry:
            return self._serialize_with_refs(value, memo)

        if hasattr(value, "to_dict"):
            base = value.to_dict()
            if isinstance(base, dict):
                return {
                    k: self._serialize_config_value(v, memo) for k, v in base.items()
                }
            return base

        return value

    def deserialize_model_config(self, data: dict[str, Any]) -> dict[str, Any]:
        """Deserialize a model_config dict in a single pass with shared memo.

        All values in the dict are deserialized with the same
        ``_DeserializeMemo``, so ``$ref`` references are resolved across keys.

        Parameters
        ----------
        data : dict
            JSON-parsed model configuration dict, as produced by
            :meth:`serialize_model_config`.

        Returns
        -------
        dict
            Dict with live objects (Priors, R2D2 splits, etc.).
        """
        memo = _DeserializeMemo()
        token = _current_deserialize_memo.set(memo)
        try:
            return {k: self._deserialize_config_value(v, memo) for k, v in data.items()}
        finally:
            _current_deserialize_memo.reset(token)

    def _deserialize_config_value(self, value: Any, memo: _DeserializeMemo) -> Any:
        """Deserialize a single config value with shared memo support."""
        import numpy as np

        if isinstance(value, dict):
            if "$ref" in value:
                ref_id = value["$ref"]
                if memo.has_ref(ref_id):
                    return memo.resolve_ref(ref_id)
                raise SerializationError(
                    f"Reference $ref={ref_id} not found in config. "
                    "The referenced object may not have been deserialized yet."
                )

            if "__type__" in value:
                return self._deserialize_with_refs(value, None, memo)

            if isinstance(value.get("dist"), str):
                prior = self._deserialize_prior_spec(value, memo)
                if prior is not None:
                    return prior

            return {
                k: self._deserialize_config_value(v, memo) for k, v in value.items()
            }

        if isinstance(value, list):
            result = [self._deserialize_config_value(v, memo) for v in value]
            if result and all(isinstance(v, (int, float)) for v in result):
                return np.array(result)
            return result

        return value

    def _deserialize_prior_spec(
        self, data: dict[str, Any], memo: _DeserializeMemo
    ) -> Any | None:
        """Deserialize a Prior spec dict and recurse into kwargs."""
        from pymc_extras.deserialize import DeserializableError, deserialize

        token = _current_deserialize_memo.set(memo)
        try:
            prior = deserialize(data)
        except DeserializableError as err:
            if err.__cause__ is not None:
                raise
            return None
        finally:
            _current_deserialize_memo.reset(token)

        return prior


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
