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
"""Deserialize into a PyMC-Marketing object.

This is a two step process:

1. Determine if the data is of the correct type.
2. Deserialize the data into a python object for PyMC-Marketing.

Examples
--------
Custom class deserialization:

.. code-block:: python

    from pymc_marketing.deserialize import register_deserialization

    class MyClass:
        def __init__(self, value: int):
            self.value = value

        def to_dict(self) -> dict:
            # Example of what the to_dict method might look like.
            return {"value": self.value}

    register_deserialization(
        is_type=lambda data: data.keys() == {"value"},
        deserialize=lambda data: MyClass(value=data["value"]),
    )

"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

IsType = Callable[[Any], bool]
Deserialize = Callable[[Any], Any]


@dataclass
class Deserializer:
    """Object to store a deserialization mapping."""

    is_type: IsType
    deserialize: Deserialize


DESERIALIZERS: list[Deserializer] = []


class DeserializableError(Exception):
    """Error raised when data cannot be deserialized."""

    def __init__(self, data: Any):
        self.data = data
        super().__init__(
            f"Couldn't deserialize {data}. Use register_deserialization to add a deserialization mapping."
        )


def deserialize(data: Any) -> Any:
    """Deserialize a dictionary into a Python object.

    Parameters
    ----------
    data : Any
        The data to deserialize.

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

        if not is_type:
            continue

        try:
            return mapping.deserialize(data)
        except Exception as e:
            raise DeserializableError(data) from e
    else:
        raise DeserializableError(data)


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
