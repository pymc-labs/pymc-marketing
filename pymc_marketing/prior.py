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
"""Class that represents a prior distribution.

The `Prior` class is a wrapper around PyMC distributions that allows the user
to create outside of the PyMC model.

.. note::

    This module has been deprecated and is moved to `pymc_extras.prior`.

This is the alternative to using the dictionaries in PyMC-Marketing models.

Examples
--------
Create a normal prior.

.. code-block:: python

    from pymc_extras.prior import Prior

    normal = Prior("Normal")

Create a hierarchical normal prior by using distributions for the parameters
and specifying the dims.

.. code-block:: python

    hierarchical_normal = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="channel",
    )

Create a non-centered hierarchical normal prior with the `centered` parameter.

.. code-block:: python

    non_centered_hierarchical_normal = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="channel",
        # Only change needed to make it non-centered
        centered=False,
    )

Create a hierarchical beta prior by using Beta distribution, distributions for
the parameters, and specifying the dims.

.. code-block:: python

    hierarchical_beta = Prior(
        "Beta",
        alpha=Prior("HalfNormal"),
        beta=Prior("HalfNormal"),
        dims="channel",
    )

Create a transformed hierarchical normal prior by using the `transform`
parameter. Here the "sigmoid" transformation comes from `pm.math`.

.. code-block:: python

    transformed_hierarchical_normal = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        transform="sigmoid",
        dims="channel",
    )

Create a prior with a custom transform function by registering it with
`register_tensor_transform`.

.. code-block:: python

    from pymc_extras.prior import register_tensor_transform


    def custom_transform(x):
        return x**2


    register_tensor_transform("square", custom_transform)

    custom_distribution = Prior("Normal", transform="square")

"""

from __future__ import annotations

import copy
import functools
import warnings
from typing import Any

from pymc_extras import prior
from pymc_extras.deserialize import deserialize, register_deserialization


def is_alternative_prior(data: Any) -> bool:
    """Check if the data is a dictionary representing a Prior (alternative check)."""
    return isinstance(data, dict) and "distribution" in data


def deserialize_alternative_prior(data: dict[str, Any]) -> prior.Prior:
    """Alternative deserializer that recursively handles all nested parameters.

    This implementation is more general and handles cases where any parameter
    might be a nested prior, and also extracts centered and transform parameters.

    Examples
    --------
    This handles cases like:

    .. code-block:: yaml

        distribution: Gamma
        alpha: 1
        beta:
            distribution: HalfNormal
            sigma: 1
            dims: channel
        dims: [brand, channel]

    """
    data = copy.deepcopy(data)

    distribution = data.pop("distribution")
    dims = data.pop("dims", None)
    centered = data.pop("centered", True)
    transform = data.pop("transform", None)
    parameters = data

    # Recursively deserialize any nested parameters
    parameters = {
        key: value if not isinstance(value, dict) else deserialize(value)
        for key, value in parameters.items()
    }

    return prior.Prior(
        distribution,
        transform=transform,
        centered=centered,
        dims=dims,
        **parameters,
    )


# Register the alternative prior deserializer for more complex nested cases
register_deserialization(is_alternative_prior, deserialize_alternative_prior)


def warn_class_deprecation(func):
    """Warn about the deprecation of this module."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        name = self.__class__.__name__
        warnings.warn(
            f"The {name} class has moved to pymc_extras.prior module and will be removed in a future release. "
            f"Import it from `from pymc_extras.prior import {name}`. ",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(self, *args, **kwargs)

    return wrapper


def warn_function_deprecation(func):
    """Warn about the deprecation of this function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        warnings.warn(
            f"The {name} function has moved to pymc_extras.prior module and will be removed in a future release. "
            f"Import it from `from pymc_extras.prior import {name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


class Prior(prior.Prior):
    """Backwards-compatible wrapper for the Prior class."""

    @warn_class_deprecation
    def __init__(self, *args, **kwargs):
        """Initialize the Prior class with the given arguments."""
        super().__init__(*args, **kwargs)


class Censored(prior.Censored):
    """Backwards-compatible wrapper for the CensoredPrior class."""

    @warn_class_deprecation
    def __init__(self, *args, **kwargs):
        """Initialize the CensoredPrior class with the given arguments."""
        super().__init__(*args, **kwargs)


class Scaled(prior.Scaled):
    """Backwards-compatible wrapper for the ScaledPrior class."""

    @warn_class_deprecation
    def __init__(self, *args, **kwargs):
        """Initialize the ScaledPrior class with the given arguments."""
        super().__init__(*args, **kwargs)


sample_prior = warn_function_deprecation(prior.sample_prior)
create_dim_handler = warn_function_deprecation(prior.create_dim_handler)
handle_dims = warn_function_deprecation(prior.handle_dims)
register_tensor_transform = warn_function_deprecation(prior.register_tensor_transform)
