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
"""Custom deserializers for the MMM project."""

from __future__ import annotations

import copy
import logging
from typing import Any

# Add these imports to register any built-in deserializers
from pymc_marketing.deserialize import deserialize, register_deserialization
from pymc_marketing.prior import Prior

logger = logging.getLogger(__name__)


def is_alternative_prior(data: Any) -> bool:
    """Check if the data is a dictionary representing a Prior (alternative check)."""
    return isinstance(data, dict) and "distribution" in data


def deserialize_alternative_prior(data: dict[str, Any]) -> Prior:
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

    return Prior(
        distribution,
        transform=transform,
        centered=centered,
        dims=dims,
        **parameters,
    )


def is_priors_dict(data: Any) -> bool:
    """Check if the data is a dictionary of priors."""
    if not isinstance(data, dict):
        return False

    # Check if any value is a prior-like dictionary
    for _key, value in data.items():
        if isinstance(value, dict) and "distribution" in value:
            return True
    return False


def deserialize_priors_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a dictionary of priors."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            if "distribution" in value:
                result[key] = deserialize(value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result


# Register the alternative prior deserializer for more complex nested cases
register_deserialization(is_alternative_prior, deserialize_alternative_prior)

# Register the priors dictionary deserializer
register_deserialization(is_priors_dict, deserialize_priors_dict)
