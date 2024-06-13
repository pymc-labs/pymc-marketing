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

Example model configuration for model with alpha and beta parameters

```python
model_config = {
    "alpha": {
        "dist": "Normal",
        "kwargs": {
            "mu": 0,
            "sigma": 1,
        },
    },
    "beta": {
        "dist": "Normal",
        "kwargs": {
            "mu": 0,
            "sigma": 1,
        },
        "dims": "geo",
    },
}
```

"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
from pymc.distributions.shape_utils import Dims
from pytensor import tensor as pt


class UnsupportedShapeError(Exception):
    """Error for when the shape of the hierarchical variable is not supported."""


def handle_scalar(var: pt.TensorLike, dims: Dims, desired_dims: Dims) -> pt.TensorLike:
    return var


def handle_1d(var: pt.TensorLike, dims: Dims, desired_dims: Dims) -> pt.TensorLike:
    return var


def handle_2d(var: pt.TensorLike, dims: Dims, desired_dims: Dims) -> pt.TensorLike:
    if dims == desired_dims:
        return var

    if dims[::-1] == desired_dims:
        return var.T

    if dims[0] == desired_dims[-2]:
        return var[:, None]

    return var


HANDLE_MAPPING = {0: handle_scalar, 1: handle_1d, 2: handle_2d}

DimHandler = Callable[[pt.TensorLike, Dims], pt.TensorLike]


def create_dim_handler(desired_dims: Dims) -> DimHandler:
    desired_dims = desired_dims if isinstance(desired_dims, tuple) else (desired_dims,)
    ndims = len(desired_dims)
    if ndims > 2:
        raise UnsupportedShapeError(
            "At most two dims can be specified. Raise an issue if support for more dims is needed."
        )

    handle = HANDLE_MAPPING[ndims]

    def handle_shape(
        var: pt.TensorLike,
        dims: Dims,
    ) -> pt.TensorLike:
        """Handle the shape for a hierarchical parameter."""
        dims = desired_dims if dims is None else dims
        dims = dims if isinstance(dims, tuple) else (dims,)

        if not set(dims).issubset(set(desired_dims)):
            raise UnsupportedShapeError("The dims of the variable are not supported.")

        return handle(var, dims, desired_dims)

    return handle_shape


def get_distribution(name: str) -> type[pm.Distribution]:
    """Retrieve a PyMC distribution class by name.

    Parameters
    ----------
    name : str
        Name of a PyMC distribution.

    Returns
    -------
    pm.Distribution
        A PyMC distribution class that can be used to instantiate a random
        variable.

    Raises
    ------
    ValueError
        If the specified distribution name does not correspond to any
        distribution in PyMC.
    """
    if not hasattr(pm, name):
        raise ValueError(f"Distribution {name} does not exist in PyMC.")

    return getattr(pm, name)


def handle_nested_distribution(
    name: str,
    param: str,
    parameter_config: dict[str, Any],
    dim_handler: DimHandler,
):
    param_name = f"{name}_{param}"
    kwargs = {
        key: value
        for key, value in parameter_config.items()
        if key not in ["dist", "kwargs"]
    }
    var = create_distribution(
        param_name,
        parameter_config["dist"],
        parameter_config["kwargs"],
        **kwargs,
    )
    return dim_handler(var, parameter_config.get("dims"))


class ModelConfigError(Exception):
    def __init__(self, param: str) -> None:
        self.param = param
        self.message = (
            f"Invalid parameter configuration for '{param}'."
            " It must be either a dictionary with 'dist' and 'kwargs' keys or a numeric value."
        )

        super().__init__(self.message)


def handle_parameter_configurations(
    name: str,
    param: str,
    parameter_config: dict[str, Any],
    dim_handler: DimHandler,
) -> Any:
    is_nested_distribution = (
        isinstance(parameter_config, dict)
        and "dist" in parameter_config
        and "kwargs" in parameter_config
    )
    if is_nested_distribution:
        return handle_nested_distribution(
            name,
            param,
            parameter_config,
            dim_handler=dim_handler,
        )

    if isinstance(parameter_config, int | float | np.ndarray | pt.TensorVariable):
        return parameter_config

    raise ModelConfigError(f"{name}_{param}")


def handle_parameter_distributions(
    name: str,
    param_distributions: dict[str, dict[str, Any]],
    dim_handler: DimHandler,
) -> dict[str, Any]:
    return {
        param: handle_parameter_configurations(
            name,
            param,
            parameter_config,
            dim_handler=dim_handler,
        )
        for param, parameter_config in param_distributions.items()
    }


def create_distribution(
    name: str,
    distribution_name: str,
    distribution_kwargs: dict[str, Any],
    **kwargs,
) -> pt.TensorVariable:
    dim_handler = create_dim_handler(kwargs.get("dims"))
    parameter_distributions = handle_parameter_distributions(
        name, distribution_kwargs, dim_handler=dim_handler
    )
    distribution = get_distribution(name=distribution_name)
    return distribution(name, **parameter_distributions, **kwargs)


def create_distribution_from_config(name: str, config) -> pt.TensorVariable:
    parameter_config = config[name]
    try:
        dist_name = parameter_config["dist"]
        dist_kwargs = parameter_config["kwargs"]
    except KeyError:
        raise ModelConfigError(name)

    return create_distribution(
        name,
        dist_name,
        dist_kwargs,
        dims=parameter_config.get("dims"),
    )


LIKELIHOOD_DISTRIBUTIONS: set[str] = {
    "Normal",
    "Beta",
    "StudentT",
    "Laplace",
    "Logistic",
    "LogNormal",
    "Wald",
    "TruncatedNormal",
    "Gamma",
    "AsymmetricLaplace",
    "VonMises",
}


class UnsupportedDistributionError(Exception):
    pass


def create_likelihood_distribution(
    name: str,
    param_config: dict,
    mu: pt.TensorVariable,
    observed: np.ndarray | pd.Series,
    dims: str,
) -> pt.TensorVariable:
    """
    Create and return a likelihood distribution for the model.

    This method prepares the distribution and its parameters as specified in the
    configuration dictionary, validates them, and constructs the likelihood
    distribution using PyMC.

    Parameters
    ----------
    dist : Dict
        A configuration dictionary that must contain a 'dist' key with the name of
        the distribution and a 'kwargs' key with parameters for the distribution.
    observed : Union[np.ndarray, pd.Series]
        The observed data to which the likelihood distribution will be fitted.
    dims : str
        The dimensions of the data.

    Returns
    -------
    TensorVariable
        The likelihood distribution constructed with PyMC.

    Raises
    ------
    ValueError
        If 'kwargs' key is missing in `dist`, or the parameter configuration does
        not contain 'dist' and 'kwargs' keys, or if 'mu' is present in the nested
        'kwargs'
    """
    if param_config["dist"] not in LIKELIHOOD_DISTRIBUTIONS:
        raise UnsupportedDistributionError(
            f"""
            The distribution used for the likelihood is not allowed.
            Please, use one of the following distributions: {list(LIKELIHOOD_DISTRIBUTIONS)}.
            """
        )

    if "mu" in param_config["kwargs"]:
        raise ValueError(
            "The 'mu' key is not allowed directly within 'kwargs' of the main distribution as it is reserved."
        )

    param_config["kwargs"]["mu"] = mu

    kwargs = {
        key: value
        for key, value in param_config.items()
        if key not in ["dist", "kwargs"]
    }
    kwargs["dims"] = dims
    kwargs["observed"] = observed

    return create_distribution(
        name=name,
        distribution_name=param_config["dist"],
        distribution_kwargs=param_config["kwargs"],
        **kwargs,
    )
