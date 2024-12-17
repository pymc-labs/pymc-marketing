#   Copyright 2024 The PyMC Labs Developers
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
"""Deserialize a dictionary into a PyMC-Marketing object."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

IsType = Callable[[dict], bool]
Deserialize = Callable[[dict], Any]


@dataclass
class Deserializer:
    """Object to store a deserialization mapping."""

    is_type: IsType
    deserialize: Deserialize


DESERIALIZERS: list[Deserializer] = []


def deserialize(data: dict) -> Any:
    """Deserialize a dictionary into a Python object.

    Parameters
    ----------
    data : dict
        The dictionary to deserialize.

    Returns
    -------
    Any
        The deserialized object.

    """
    for mapping in DESERIALIZERS:
        try:
            is_type = mapping.is_type(data)
        except Exception:
            is_type = False

        if is_type:
            return mapping.deserialize(data)
    else:
        raise ValueError(f"Couldn't deserialize {data}")


def register_deserialization(is_type: IsType, deserialize: Deserialize) -> None:
    """Register a deserialization mapping.

    Parameters
    ----------
    is_type : Callable[[Any], bool]
        Function to determine if the data is of the correct type.
    deserialize : Callable[[dict], Any]
        Function to deserialize the data.

    """
    mapping = Deserializer(is_type=is_type, deserialize=deserialize)
    DESERIALIZERS.append(mapping)
