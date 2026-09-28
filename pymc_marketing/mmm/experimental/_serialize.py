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
"""Equations as a JSON table of nodes in which each shared term appears once.

Every term (``ModelTerm``, ``Sum``, ``Product``, and graph terms) becomes one node
``{"__type__": ..., "fields": ...}``; a field holding another term stores
``{"$ref": node_id}``, so a term shared by several equations stays one object after
loading. Configuration values are stored inline: ``Prior`` recipes keep the labels of
their ``DataArray`` parameters, transformations and Fourier bases store their priors
the same way, and other registered objects use their own ``to_dict``.

Node classes are resolved only through ``pymc_marketing.serialization``: a class must
be registered to be saved or loaded. Dataclass terms are rebuilt from their init
fields; other terms from their public attributes, passed back as keyword arguments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pymc_extras.prior import Prior, VariableFactory

from pymc_marketing.mmm.components.base import Transformation
from pymc_marketing.mmm.experimental._graph import Data, Equation, GraphTerm
from pymc_marketing.mmm.experimental._terms import MediaTransform, Seasonality
from pymc_marketing.mmm.fourier import FourierBase
from pymc_marketing.serialization import (
    DeferredFactory,
    SerializationError,
    serialization,
)
from pymc_marketing.terms import (
    ModelTerm,
    Product,
    Sum,
    _deserialize_child,
    _func_name,
    _resolve_func,
    _serialize_child,
)

FORMAT = "pymc_marketing.experimental.GAM/1"


def _standalone(*_: Any) -> Any:
    raise SerializationError(
        "Experimental graph terms are saved only as part of a model; use GAM.save."
    )


for _graph_term in (Data, Equation, MediaTransform, Seasonality):
    serialization.register(
        _graph_term, serializer=_standalone, deserializer=_standalone
    )


def _type_key(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _node_fields(term: Any) -> dict[str, Any]:
    if is_dataclass(term) and not isinstance(term, GraphTerm):
        return {
            field.name: getattr(term, field.name)
            for field in fields(term)
            if field.init
        }
    return {key: value for key, value in vars(term).items() if not key.startswith("_")}


def _encode_labels(index: pd.Index) -> dict[str, Any]:
    if isinstance(index, pd.DatetimeIndex):
        return {"datetime": True, "values": [label.isoformat() for label in index]}
    return {"datetime": False, "values": index.tolist()}


def _encode_array(array: xr.DataArray) -> dict[str, Any]:
    return {
        "values": array.values.tolist(),
        "dtype": str(array.dtype),
        "dims": list(array.dims),
        "coords": {
            dim: _encode_labels(array.get_index(dim))
            for dim in array.dims
            if dim in array.coords
        },
    }


def _decode_array(data: Mapping[str, Any]) -> xr.DataArray:
    coords = {
        dim: pd.DatetimeIndex(labels["values"])
        if labels["datetime"]
        else labels["values"]
        for dim, labels in data["coords"].items()
    }
    return xr.DataArray(
        np.asarray(data["values"], dtype=data["dtype"]),
        dims=data["dims"],
        coords=coords,
    )


class _Writer:
    def __init__(self) -> None:
        self.ids: dict[int, str] = {}
        self.nodes: dict[str, dict[str, Any]] = {}

    def node(self, term: Any) -> dict[str, str]:
        key = id(term)
        if key not in self.ids:
            if not serialization.is_registered(term):
                raise SerializationError(
                    f"Cannot save term {type(term).__name__}; register its class with "
                    "@serialization.register."
                )
            identifier = f"n{len(self.ids)}"
            self.ids[key] = identifier
            encoded = {}
            for name, value in _node_fields(term).items():
                try:
                    encoded[name] = self.value(value)
                except SerializationError as error:
                    raise SerializationError(
                        f"Cannot save {type(term).__name__}.{name}: {error}"
                    ) from error
            self.nodes[identifier] = {"__type__": _type_key(term), "fields": encoded}
        return {"$ref": self.ids[key]}

    def value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (ModelTerm, Sum, Product)):
            return self.node(value)
        if type(value) is Prior:
            return {
                "$prior": {
                    "distribution": value.distribution,
                    "parameters": {
                        name: self.value(item)
                        for name, item in value.parameters.items()
                    },
                    "dims": self.value(value.dims),
                    "core_dims": self.value(value.core_dims),
                    "centered": value.centered,
                    "transform": value.transform,
                }
            }
        if isinstance(value, xr.DataArray):
            return {"$array": _encode_array(value)}
        if isinstance(value, np.ndarray):
            return {"$ndarray": {"values": value.tolist(), "dtype": str(value.dtype)}}
        if isinstance(value, Transformation):
            if not serialization.is_registered(value):
                raise SerializationError(
                    f"Register {type(value).__name__} with @serialization.register."
                )
            data = serialization.serialize(value)
            data["priors"] = {
                name: self.value(item) for name, item in value.function_priors.items()
            }
            return {"$transformation": data}
        if isinstance(value, FourierBase):
            data = {"__type__": _type_key(value), **value.model_dump(mode="json")}
            data["prior"] = self.value(value.prior)
            return {"$fourier": data}
        if isinstance(value, (VariableFactory, DeferredFactory)):
            return {"$value": _serialize_child(value)}
        if isinstance(value, tuple):
            return {"$tuple": [self.value(item) for item in value]}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise SerializationError("Saved mappings need string keys.")
            return {"$map": {key: self.value(item) for key, item in value.items()}}
        if callable(value) and not isinstance(value, type):
            return {"$func": _func_name(value)}
        raise SerializationError(f"Cannot save a value of type {type(value).__name__}.")


class _Reader:
    def __init__(self, nodes: Mapping[str, Any]) -> None:
        self.nodes = nodes
        self.built: dict[str, Any] = {}
        self.active: set[str] = set()

    def node(self, identifier: str) -> Any:
        if identifier in self.built:
            return self.built[identifier]
        if identifier in self.active:
            raise SerializationError("The saved model contains a dependency cycle.")
        if identifier not in self.nodes:
            raise SerializationError(f"The saved model has no node {identifier!r}.")
        entry = self.nodes[identifier]
        cls = serialization.lookup(entry["__type__"])
        if not issubclass(cls, (ModelTerm, Sum, Product)):
            raise SerializationError(f"{entry['__type__']!r} is not a model term.")
        self.active.add(identifier)
        try:
            term = cls(
                **{name: self.value(item) for name, item in entry["fields"].items()}
            )
        finally:
            self.active.remove(identifier)
        self.built[identifier] = term
        return term

    def value(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if not isinstance(value, dict):
            return value
        if len(value) != 1:
            raise SerializationError(f"Unrecognized saved value {value!r}.")
        ((tag, payload),) = value.items()
        if tag == "$ref":
            return self.node(payload)
        if tag == "$prior":
            return Prior(
                payload["distribution"],
                dims=self.value(payload["dims"]),
                core_dims=self.value(payload["core_dims"]),
                centered=payload["centered"],
                transform=payload["transform"],
                **{
                    name: self.value(item)
                    for name, item in payload["parameters"].items()
                },
            )
        if tag == "$array":
            return _decode_array(payload)
        if tag == "$ndarray":
            return np.asarray(payload["values"], dtype=payload["dtype"])
        if tag == "$transformation":
            priors = {
                name: self.value(item) for name, item in payload["priors"].items()
            }
            cls = serialization.lookup(payload["__type__"])
            return cls.from_dict({**payload, "priors": priors})  # type: ignore[attr-defined]
        if tag == "$fourier":
            data = {key: item for key, item in payload.items() if key != "__type__"}
            data["prior"] = self.value(data["prior"])
            return serialization.lookup(payload["__type__"])(**data)
        if tag == "$value":
            return _deserialize_child(payload)
        if tag == "$tuple":
            return tuple(self.value(item) for item in payload)
        if tag == "$map":
            return {key: self.value(item) for key, item in payload.items()}
        if tag == "$func":
            return _resolve_func(payload)
        raise SerializationError(f"Unrecognized saved value tag {tag!r}.")


def spec_to_dict(roots: Sequence[Equation]) -> dict[str, Any]:
    """Serialize equations to a JSON-safe node table, sharing terms by identity.

    Parameters
    ----------
    roots : sequence of Equation
        Observed equations of one model, in declaration order.

    Returns
    -------
    dict
        ``{"format", "roots", "nodes"}``, JSON-serializable.

    Raises
    ------
    SerializationError
        If a term or value cannot be saved, for example a ``Transform`` of a lambda.
    """
    writer = _Writer()
    return {
        "format": FORMAT,
        "roots": [writer.node(root)["$ref"] for root in roots],
        "nodes": writer.nodes,
    }


def spec_from_dict(data: Mapping[str, Any]) -> tuple[Equation, ...]:
    """Rebuild equations from a node table written by ``spec_to_dict``.

    Parameters
    ----------
    data : mapping
        Saved node table.

    Returns
    -------
    tuple of Equation
        Root equations; terms referenced from several places are one object.

    Raises
    ------
    SerializationError
        If the format is unknown or a node is invalid.
    """
    if data.get("format") != FORMAT:
        raise SerializationError(
            f"Unsupported saved model format {data.get('format')!r}; expected {FORMAT!r}."
        )
    reader = _Reader(data["nodes"])
    roots = tuple(reader.node(identifier) for identifier in data["roots"])
    if not all(isinstance(root, Equation) for root in roots):
        raise SerializationError("Saved model roots must be equations.")
    return roots
