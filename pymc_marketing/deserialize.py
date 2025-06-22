#   Copyright 2022 - 2025 The PyMC Labs Developers
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

This is used to deserialize JSON data into PyMC-Marketing objects
throughout the package.

Examples
--------
Make use of the already registered PyMC-Marketing deserializers:

.. code-block:: python

    from pymc_marketing.deserialize import deserialize

    prior_class_data = {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}}
    prior = deserialize(prior_class_data)
    # Prior("Normal", mu=0, sigma=1)

Register custom class deserialization:

.. code-block:: python

    from pymc_marketing.deserialize import register_deserialization


    class MyClass:
        def __init__(self, value: int):
            self.value = value

        def to_dict(self) -> dict:
            # Example of what the to_dict method might look like.
            return {"value": self.value}


    register_deserialization(
        is_type=lambda data: data.keys() == {"value"}
        and isinstance(data["value"], int),
        deserialize=lambda data: MyClass(value=data["value"]),
    )

Deserialize data into that custom class:

.. code-block:: python

    from pymc_marketing.deserialize import deserialize

    data = {"value": 42}
    obj = deserialize(data)
    assert isinstance(obj, MyClass)


"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

IsType = Callable[[Any], bool]
Deserialize = Callable[[Any], Any]


@dataclass
class Deserializer:
    """Object to store information required for deserialization.

    All deserializers should be stored via the :func:`register_deserialization` function
    instead of creating this object directly.

    Attributes
    ----------
    is_type : IsType
        Function to determine if the data is of the correct type.
    deserialize : Deserialize
        Function to deserialize the data.

    Examples
    --------
    .. code-block:: python

        from typing import Any


        class MyClass:
            def __init__(self, value: int):
                self.value = value


        from pymc_marketing.deserialize import Deserializer


        def is_type(data: Any) -> bool:
            return data.keys() == {"value"} and isinstance(data["value"], int)


        def deserialize(data: dict) -> MyClass:
            return MyClass(value=data["value"])


        deserialize_logic = Deserializer(is_type=is_type, deserialize=deserialize)

    """

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

    Use the :func:`register_deserialization` function to add custom deserializations.

    Deserialization is a two step process due to the dynamic nature of the data:

    1. Determine if the data is of the correct type.
    2. Deserialize the data into a Python object.

    Each registered deserialization is checked in order until one is found that can
    deserialize the data. If no deserialization is found, a :class:`DeserializableError` is raised.

    A :class:`DeserializableError` is raised when the data fails to be deserialized
    by any of the registered deserializers.

    Parameters
    ----------
    data : Any
        The data to deserialize.

    Returns
    -------
    Any
        The deserialized object.

    Raises
    ------
    DeserializableError
        Raised when the data doesn't match any registered deserializations
        or fails to be deserialized.

    Examples
    --------
    Deserialize a :class:`pymc_marketing.prior.Prior` object:

    .. code-block:: python

        from pymc_marketing.deserialize import deserialize

        data = {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}}
        prior = deserialize(data)
        # Prior("Normal", mu=0, sigma=1)

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
    """Register an arbitrary deserialization.

    Use the :func:`deserialize` function to then deserialize data using all registered
    deserialize functions.

    Classes from PyMC-Marketing have their deserialization mappings registered
    automatically. However, custom classes will need to be registered manually
    using this function before they can be deserialized.

    Parameters
    ----------
    is_type : Callable[[Any], bool]
        Function to determine if the data is of the correct type.
    deserialize : Callable[[dict], Any]
        Function to deserialize the data of that type.

    Examples
    --------
    Register a custom class deserialization:

    .. code-block:: python

        from pymc_marketing.deserialize import register_deserialization


        class MyClass:
            def __init__(self, value: int):
                self.value = value

            def to_dict(self) -> dict:
                # Example of what the to_dict method might look like.
                return {"value": self.value}


        register_deserialization(
            is_type=lambda data: data.keys() == {"value"}
            and isinstance(data["value"], int),
            deserialize=lambda data: MyClass(value=data["value"]),
        )

    Use that custom class deserialization:

    .. code-block:: python

        from pymc_marketing.deserialize import deserialize

        data = {"value": 42}
        obj = deserialize(data)
        assert isinstance(obj, MyClass)

    """
    mapping = Deserializer(is_type=is_type, deserialize=deserialize)
    DESERIALIZERS.append(mapping)
