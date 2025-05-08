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

import logging
from typing import Any

# Add these imports to register any built-in deserializers
from pymc_marketing.deserialize import register_deserialization
from pymc_marketing.prior import Prior

logger = logging.getLogger(__name__)


def is_prior_dict(data: Any) -> bool:
    """Check if the data is a dictionary representing a Prior."""
    if not isinstance(data, dict):
        return False
    return "distribution" in data


def deserialize_prior(data: dict[str, Any]) -> Prior:
    """Deserialize a Prior from the dictionary representation."""
    # Make a copy to avoid modifying the original
    data_copy = data.copy()

    # Extract distribution
    distribution = data_copy.pop("distribution")

    # Process nested priors in the data before creating the Prior
    for key, value in data_copy.items():
        if isinstance(value, dict) and is_prior_dict(value):
            data_copy[key] = deserialize_prior(value)
        elif (
            isinstance(value, dict)
            and "target_class" in value
            and value["target_class"] == "pymc_marketing.prior.Prior"
        ):
            data_copy[key] = deserialize_standard_prior(value)

    # Create Prior
    return Prior(distribution, **data_copy)


def is_likelihood_dict(data: Any) -> bool:
    """Check if the data is a likelihood dictionary with a nested sigma Prior."""
    if not isinstance(data, dict):
        return False
    # Likelihood dict typically has sigma and dims keys
    return (
        "distribution" in data
        and data["distribution"] == "Normal"
        and "sigma" in data
        and isinstance(data["sigma"], dict)
    )


def deserialize_likelihood(data: dict[str, Any]) -> Prior:
    """
    Deserialize a likelihood dictionary as a Normal prior with nested sigma.

    The expected format is:
    {
        "distribution": "Normal",
        "sigma": {
            "distribution": "Gamma",
            ...
        },
        "dims": ...
    }
    """
    # Make a copy to avoid modifying the original
    data_copy = data.copy()

    # Extract distribution
    distribution = data_copy.pop("distribution")

    # Handle special sigma case - deserialize it if it contains a distribution
    if (
        "sigma" in data_copy
        and isinstance(data_copy["sigma"], dict)
        and "distribution" in data_copy["sigma"]
    ):
        data_copy["sigma"] = deserialize_prior(data_copy["sigma"])

    # Create Prior
    logger.info(f"Creating Prior with distribution={distribution}, kwargs={data_copy}")
    return Prior(distribution, **data_copy)


def is_standard_prior_dict(data: Any) -> tuple[bool, str]:
    """
    Check if the data is a standard dictionary in the format used by factories.py.

    Returns a tuple of (is_match, target_class_name)
    """
    if not isinstance(data, dict):
        return False, ""

    if (
        "target_class" in data
        and data["target_class"] == "pymc_marketing.prior.Prior"
        and "kwargs" in data
    ):
        return True, "Prior"

    return False, ""


def deserialize_standard_prior(data: dict[str, Any]) -> Prior:
    """
    Deserialize a prior from the standard format used by factories.py.

    The expected format is:
    {
        "target_class": "pymc_marketing.prior.Prior",
        "kwargs": {
            "args": ["Distribution"],
            "param1": value1,
            ...
        }
    }
    """
    kwargs = data.get("kwargs", {})

    # Get distribution from args
    args = kwargs.get("args", ["Normal"])
    distribution = args[0]

    # Create a new kwargs dict without args
    new_kwargs = {k: v for k, v in kwargs.items() if k != "args"}

    # Process nested priors in kwargs
    for key, value in new_kwargs.items():
        if isinstance(value, dict):
            if "distribution" in value:
                new_kwargs[key] = deserialize_prior(value)
            elif (
                "target_class" in value
                and value["target_class"] == "pymc_marketing.prior.Prior"
            ):
                new_kwargs[key] = deserialize_standard_prior(value)

    # Create Prior
    logger.info(f"Creating Prior with distribution={distribution}, kwargs={new_kwargs}")
    return Prior(distribution, **new_kwargs)


def is_priors_dict(data: Any) -> bool:
    """Check if the data is a dictionary of priors."""
    if not isinstance(data, dict):
        return False

    # Check if any value is a prior-like dictionary
    for _key, value in data.items():
        if isinstance(value, dict) and (
            "distribution" in value
            or (
                "target_class" in value
                and value["target_class"] == "pymc_marketing.prior.Prior"
            )
        ):
            return True
    return False


def deserialize_priors_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a dictionary of priors."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            if "distribution" in value:
                result[key] = deserialize_prior(value)
            elif (
                "target_class" in value
                and value["target_class"] == "pymc_marketing.prior.Prior"
            ):
                result[key] = deserialize_standard_prior(value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def register_custom_deserializers():
    """Register all custom deserializers."""
    # Register the simple prior deserializer for distribution-based format
    register_deserialization(is_prior_dict, deserialize_prior)

    # Register the nested likelihood deserializer for distribution-based format
    register_deserialization(is_likelihood_dict, deserialize_likelihood)

    # Register the standard deserializer used by factories.py
    register_deserialization(
        lambda x: is_standard_prior_dict(x)[0], deserialize_standard_prior
    )

    # Register the priors dictionary deserializer
    register_deserialization(is_priors_dict, deserialize_priors_dict)
