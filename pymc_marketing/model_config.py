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
"""Model configuration utilities."""

from typing import Any

from pymc_extras.prior import Prior, VariableFactory

from pymc_marketing.hsgp_kwargs import HSGPKwargs


class ModelConfigError(Exception):
    """Exception raised for errors in model configuration."""


ModelConfig = dict[str, VariableFactory | HSGPKwargs | Prior | Any]


def parse_model_config(
    model_config: ModelConfig,
    hsgp_kwargs_fields: list[str] | None = None,
) -> ModelConfig:
    """Parse the model config dictionary.

    Parameters
    ----------
    model_config : dict
        The model configuration dictionary.
    hsgp_kwargs_fields : list[str], optional
        A list of keys to parse as HSGP kwargs.

    Returns
    -------
    dict
        The parsed model configuration dictionary.

    Examples
    --------
    Parse the HSGP kwargs field in a model configuration.

    .. code-block:: python

        from pymc_marketing.hsgp_kwargs import HSGPKwargs
        from pymc_marketing.model_config import parse_model_config
        from pymc_extras.prior import Prior

        model_config = {
            "alpha": Prior("Normal", mu=0, sigma=1),
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
        )
        # {'alpha': Prior("Normal", mu=0, sigma=1),  # unchanged
        #  'beta': Prior("HalfNormal"),  # unchanged
        #  'intercept_tvp_config': HSGPKwargs(m=200, L=119.17, eta_lam=1.0, ls_mu=5.0, ls_sigma=10.0, cov_func=None),
        #  'other_intercept': {'key': 'Some other non-distribution configuration'}}

    """
    hsgp_kwargs_fields = hsgp_kwargs_fields or []

    # Convert to a set for O(1) lookup
    hsgp_kwargs_set = set(hsgp_kwargs_fields)

    parse_errors = []

    def handle_hggp_kwargs(name, config):
        if name not in hsgp_kwargs_set:
            return config

        if isinstance(config, HSGPKwargs):
            return config

        # Only convert to HSGPKwargs if the config is a dict that has HSGPKwargs keys
        # Don't convert old-style configs with ls_lower/ls_upper (parameterize_from_data format)
        if not isinstance(config, dict):
            return config

        hsgp_keys = {"m", "L", "eta_lam", "ls_mu", "ls_sigma", "cov_func"}
        old_style_keys = {"ls_lower", "ls_upper"}

        # If config has old-style keys, keep it as dict for parameterize_from_data
        if config.keys() & old_style_keys:
            return config

        # Only convert if config has at least one HSGPKwargs key
        if not (config.keys() & hsgp_keys):
            return config

        try:
            hsgp_kwargs = HSGPKwargs.model_validate(config)
            return hsgp_kwargs
        except Exception as e:
            parse_errors.append(f"Parameter {name}: {e}")
            return config

    def check_legacy_prior_spec(name, config):
        if (
            name not in hsgp_kwargs_set
            and isinstance(config, dict)
            and ("dist" in config or "distribution" in config)
        ):
            parse_errors.append(
                f"Parameter {name!r} looks like a legacy dict-format prior spec. "
                "Dict-format priors were removed in v1.0.0; use "
                "pymc_extras.prior.Prior instead, e.g. "
                'Prior("Normal", mu=0, sigma=1) or Prior.from_dict(...)'
            )
        return config

    # Priors already arrive as `Prior`/`VariableFactory` objects and pass through
    # untouched; only the `HSGPKwargs` fields need converting from dicts. Dicts
    # that look like removed dict-format prior specs are rejected with a
    # migration hint instead of failing later with an opaque AttributeError.
    result: ModelConfig = {
        name: check_legacy_prior_spec(name, handle_hggp_kwargs(name, config))
        for name, config in model_config.items()
    }

    if parse_errors:
        combined_errors = ", ".join(parse_errors)
        msg = (
            f"{len(parse_errors)} errors occurred while "
            "parsing model configuration. "
            f"Errors: {combined_errors}"
        )
        raise ModelConfigError(msg)

    return result
