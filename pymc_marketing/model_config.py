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
"""Model configuration utilities."""

import warnings
from collections.abc import Sequence
from typing import Any

from pymc_extras.deserialize import deserialize
from pymc_extras.prior import Prior, VariableFactory

from pymc_marketing.hsgp_kwargs import HSGPKwargs


class ModelConfigError(Exception):
    """Exception raised for errors in model configuration."""


ModelConfig = dict[str, VariableFactory | HSGPKwargs | Prior | Any]


def parse_model_config(
    model_config: ModelConfig,
    hsgp_kwargs_fields: list[str] | None = None,
    non_distributions: list[str] | None = None,
) -> ModelConfig:
    """Parse the model config dictionary.

    Parameters
    ----------
    model_config : dict
        The model configuration dictionary.
    hsgp_kwargs_fields : list[str], optional
        A list of keys to parse as HSGP kwargs.
    non_distributions : list[str], optional
        A list of keys to ignore when parsing the model configuration
        dictionary due to them not being distributions.

    Returns
    -------
    dict
        The parsed model configuration dictionary.

    Examples
    --------
    Parse all keys in model configuration but ignore the key "tvp_intercept".

    .. code-block:: python

        from pymc_marketing.hsgp_kwargs import HSGPKwargs
        from pymc_marketing.model_config import parse_model_config
        from pymc_extras.prior import Prior

        model_config = {
            "alpha": {
                "dist": "Normal",
                "kwargs": {
                    "mu": 0,
                    "sigma": 1,
                },
            },
            "beta": Prior("HalfNormal"),
            "intercept_tvp_config": {
                "m": 200,
                "L": 119.17,
                "eta_lam": 1.0,
                "ls_mu": 5.0,
                "ls_sigma": 10.0,
                "cov_func": None,
            },
            "other_intercept": {
                "key": "Some other non-distribution configuration",
            },
        }

        parsed_model_config = parse_model_config(
            model_config,
            hsgp_kwargs_fields=["intercept_tvp_config"],
            non_distributions=["other_intercept"],
        )
        # {'alpha': Prior("Normal", mu=0, sigma=1),
        #  'beta': Prior("HalfNormal"),
        #  'intercept_tvp_config': HSGPKwargs(m=200, L=119.17, eta_lam=1.0, ls_mu=5.0, ls_sigma=10.0, cov_func=None),
        #  'other_intercept': {'key': 'Some other non-distribution configuration'}}

    Parsing with an error:

    .. code-block:: python

        from pymc_marketing.model_config import (
            parse_model_config,
            ModelConfigError,
        )

        model_config = {
            "alpha": {"key": "Non distribution"},
            "beta": {"dist": "UnknownDistribution"},
            "gamma": "Completely wrong",
        }

        try:
            parse_model_config(model_config)
        except ModelConfigError as e:
            print(e)

    """
    non_distributions = non_distributions or []
    hsgp_kwargs_fields = hsgp_kwargs_fields or []

    # Convert to sets for O(1) lookup
    non_distributions_set = set(non_distributions)
    hsgp_kwargs_set = set(hsgp_kwargs_fields)

    parse_errors = []

    def handle_prior_config(name, prior_config):
        # Early return for non-distribution fields - must be first check
        if name in non_distributions_set or name in hsgp_kwargs_set:
            return prior_config

        if isinstance(prior_config, Prior) or isinstance(prior_config, VariableFactory):
            return prior_config

        # Skip deserialization for non-dict, non-string sequence types (lists, tuples, etc.)
        # These are not distribution configurations and should never be deserialized
        if isinstance(prior_config, Sequence) and not isinstance(prior_config, str):
            return prior_config

        # Skip deserialization for other non-dict types (strings, numbers, etc.)
        # These are not distribution configurations
        if not isinstance(prior_config, dict):
            return prior_config

        try:
            dist = deserialize(prior_config)
        except Exception as e:
            parse_errors.append(f"Parameter {name}: {e}")
        else:
            msg = (
                f"{name} is automatically converted to {dist}. "
                "Use the Prior class to avoid this warning."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

            return dist

    def handle_hggp_kwargs(name, config):
        if name not in hsgp_kwargs_set:
            return config

        if isinstance(config, HSGPKwargs):
            return config

        try:
            hsgp_kwargs = HSGPKwargs.model_validate(config)
            return hsgp_kwargs
        except Exception as e:
            parse_errors.append(f"Parameter {name}: {e}")

    # Parse the model configuration to extrat the `Prior` objects.
    result: ModelConfig = {
        name: handle_prior_config(name, prior_config)
        for name, prior_config in model_config.items()
    }
    # Parse the model configuration to extract the `HSGPKwargs` objects.
    result = {name: handle_hggp_kwargs(name, config) for name, config in result.items()}

    if parse_errors:
        combined_errors = ", ".join(parse_errors)
        msg = (
            f"{len(parse_errors)} errors occurred while "
            "parsing model configuration. "
            f"Errors: {combined_errors}"
        )
        raise ModelConfigError(msg)

    return result
