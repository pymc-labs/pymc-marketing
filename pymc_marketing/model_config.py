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
"""Model configuration utilities.

TODO: Write some old format parsing and deprecate the old format.

"""

import warnings
from typing import Any

from pymc_marketing.prior import Prior


class ModelConfigError(Exception):
    """Exception raised for errors in model configuration."""


ModelConfig = dict[str, Prior | Any]


def parse_model_config(
    model_config: ModelConfig, non_distributions: list[str] | None = None
) -> ModelConfig:
    """ "Parse the model config dictionary.

    Parameters
    ----------
    model_config : dict
        The model configuration dictionary.
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

        from pymc_marketing.model_config import parse_model_config
        from pymc_marketing.prior import Prior

        model_config = {
            "alpha": {
                "dist": "Normal",
                "kwargs": {
                    "mu": 0,
                    "sigma": 1,
                },
            },
            "beta": Prior("HalfNormal"),
            "tvp_intercept": {
                "key": "Some other non-distribution configuration",
            },
        }

        parsed_model_config = parse_model_config(
            model_config,
            non_distributions=["tvp_intercept"],
        )
        # {'alpha': Prior("Normal", mu=0, sigma=1),
        #  'beta': Prior("HalfNormal"),
        #  'tvp_intercept': {'key': 'Some other non-distribution configuration'}}

    """
    non_distributions = non_distributions or []

    def handle_prior_config(name, prior_config):
        if name in non_distributions:
            return prior_config

        if isinstance(prior_config, Prior):
            return prior_config

        dist = Prior.from_json(prior_config)
        msg = (
            f"{name} is automatically converted to {dist}. "
            "Use the Prior class to avoid this warning."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        return dist

    return {
        name: handle_prior_config(name, prior_config)
        for name, prior_config in model_config.items()
    }
