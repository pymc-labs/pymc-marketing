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

Example configuration for a scalar parameter:

.. code-block:: python

    scalar_parameter = {
        "dist": "Normal",
        "kwargs": {
            "mu": 0,
            "sigma": 1,
        },
    }

Example configuration of a 1D parameter:

.. code-block:: python

    vector_parameter = {
        "dist": "Normal",
        "kwargs": {
            "mu": 0,
            "sigma": 1,
        },
        # dims need to be specified now!
        "dims": "channel",
    }

Example configuration of 1D parameter with a hierarchical distribution for
only the mu:

.. code-block:: python

    hierarchical_parameter = {
        "dist": "Normal",
        "kwargs": {
            # Replace the scalar parameter values with additional distributions
            "mu": {
                "dist": "Normal",
                "kwargs": {
                    "mu": 0,
                    "sigma": 1,
                },
            },
            # Common sigma for all channels
            "sigma": 1,
        },
        "dims": "channel",
    }

Example parameter configuration with a hierarchical non-centered distribution:

.. code-block:: python

    hierarchical_non_centered_parameter = {
            "dist": "Normal",
            "kwargs": {
                "mu": {"dist": "HalfNormal", "kwargs": {"sigma": 2},},
                "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 1},},
            },
            "dims": ("channel"),
            "centered": False,
        }

Example configuration of a 2D parameter:

.. code-block:: python

    matrix_parameter = {
        "dist": "Normal",
        "kwargs": {
            "mu": 0,
            "sigma": 1,
        },
        # dims need to be specified now!
        "dims": ("channel", "geo"),
    }

Model configuration with all of these variables:

.. code-block:: python

    model_config = {
        "alpha": scalar_parameter,
        "beta": vector_parameter,
        "gamma": hierarchical_parameter,
        "delta": matrix_parameter,
    }

Creating variables from the configuration:

.. code-block:: python

    import pymc as pm

    from pymc_marketing.model_config import create_distribution_from_config

    coords = {
        "channel": ["A", "B", "C"],
        "geo": ["Region1", "Region2"],
    }
    with pm.Model(coords=coords) as model:
        alpha = create_distribution_from_config("alpha", model_config)
        beta = create_distribution_from_config("beta", model_config)
        gamma = create_distribution_from_config("gamma", model_config)
        delta = create_distribution_from_config("delta", model_config)

"""

from collections.abc import Callable
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
from pymc.distributions.shape_utils import Dims
from pytensor import tensor as pt


class UnsupportedShapeError(Exception):
    """Error for when the shape of the hierarchical variable is not supported."""


def handle_scalar(var: pt.TensorLike, dims: Dims, desired_dims: Dims) -> pt.TensorLike:
    """Broadcast a scalar to the desired dims."""
    return var


def handle_1d(var: pt.TensorLike, dims: Dims, desired_dims: Dims) -> pt.TensorLike:
    """Broadcast a 1D variable to the desired dims."""
    return var


def handle_2d(var: pt.TensorLike, dims: Dims, desired_dims: Dims) -> pt.TensorLike:
    """Broadcast a 2D variable to the desired dims."""
    if dims == desired_dims:
        return var

    if dims[::-1] == desired_dims:
        return var.T

    if dims[0] == desired_dims[-2]:
        return var[:, None]

    return var


HANDLE_MAPPING = MappingProxyType({0: handle_scalar, 1: handle_1d, 2: handle_2d})

DimHandler = Callable[[pt.TensorLike, Dims], pt.TensorLike]


def create_dim_handler(desired_dims: Dims) -> DimHandler:
    """Create a function that maps variable shapes to the desired dims.

    Parameters
    ----------
    desired_dims : str, sequence[str]
        The desired dimensions which the variable can broadcast with.

    Examples
    --------
    Map variable to "channel" dim:

    .. code-block:: python

        import numpy as np

        from pymc_marketing.model_config import create_dim_handler

        handle_channel = create_dim_handler("channel")
        handle_channel(np.array([1, 2, 3]), "channel")

    Map variable to "channel" and "geo" dims:

    .. code-block:: python

        import numpy as np

        from pymc_marketing.model_config import create_dim_handler

        handle_channel_geo = create_dim_handler(("channel", "geo"))

        # Transpose the array to match the desired dims
        handle_channel_geo(np.array([[1, 2, 3], [4, 5, 6]]), ("geo", "channel"))

        # Add a new axis to the array to match the desired dims
        handle_channel_geo(np.array([1, 2, 3]), "channel")

    """
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
) -> pt.TensorVariable:
    """Handle a nested distribution configuration.

    Parameters
    ----------
    name : str
        Name of parent variable.
    param : str
        Name of the parameter.
    parameter_config : Dict
        A configuration dictionary with 'dist' and 'kwargs' keys.
    dim_handler : Callable
        A function that maps variable shapes to the desired dims.

    Returns
    -------
    TensorVariable
        A PyMC random variable.

    """
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
    """Error for invalid model configuration."""

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
    """Handle the parameter configuration for a variable.

    Parameters
    ----------
    name : str
        Name of the variable.
    param : str
        Name of the parameter.
    parameter_config : Dict
        A configuration dictionary with 'dist' and 'kwargs' keys.
    dim_handler : Callable
        A function that maps variable shapes to the desired dims.

    Returns
    -------
    Any
        The parameter value or a PyMC random variable.

    """
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
    """Loop over the parameter configurations and handle them.

    name: str
        name of the variable
    param_distributions: Dict
        a dictionary with parameter names mapping to parameter configurations
    dim_handler: Callable
        a function that maps variable shapes to the desired dims

    Returns
    -------
    Dict
        a dictionary with parameter names mapping to parameter values or PyMC random variables

    """
    return {
        param: handle_parameter_configurations(
            name,
            param,
            parameter_config,
            dim_handler=dim_handler,
        )
        for param, parameter_config in param_distributions.items()
    }


class NestedDistributionError(Exception):
    """Error for when a nested distribution is detected where it is not allowed."""

    def __init__(self, param: str) -> None:
        self.param = param
        self.message = (
            f"Nested distribution detected in '{param}', which is not allowed."
        )
        super().__init__(self.message)


def check_for_deeper_nested_distribution(
    param_config: dict[str, Any], param_name: str
) -> None:
    """Check if the parameter configuration contains a deeper nested distribution."""
    if (
        isinstance(param_config, dict)
        and "dist" in param_config
        and "kwargs" in param_config
    ):
        for _key, value in param_config["kwargs"].items():
            if isinstance(value, dict) and "dist" in value and "kwargs" in value:
                raise NestedDistributionError(param_name)


class NonCenterInvalidDistributionError(Exception):
    """Error for when an invalid distribution is used for non-centered hierarchical distribution."""

    def __init__(self, name: str) -> None:
        self.param = name
        self.message = f"""
        Invalid distribution '{name}' for non-centered hierarchical distribution.
        Only 'Normal' is allowed.
        """
        super().__init__(self.message)


def create_hierarchical_non_center(
    name: str,
    distribution_kwargs: dict[str, Any],
    **kwargs,
) -> pt.TensorVariable:
    """
    Create a hierarchical non-centered distribution.

    This function constructs a hierarchical non-centered distribution using the provided
    distribution parameters for offset, mu, and sigma. It returns a deterministic variable
    representing the hierarchical non-centered distribution.

    Parameters
    ----------
    name : str
        The name of the variable.
    distribution_kwargs : dict[str, Any]
        A dictionary containing the distribution parameters for 'offset', 'mu', and 'sigma'.
    **kwargs
        Additional keyword arguments, including 'dims' for specifying desired dimensions.

    Returns
    -------
    pt.TensorVariable
        A PyMC deterministic variable representing the hierarchical non-centered distribution.

    """
    desired_dims = kwargs.get("dims", ())
    dim_handler = create_dim_handler(desired_dims)

    mu_dist = distribution_kwargs["mu"]
    mu_dims = mu_dist.get("dims", ())
    sigma_dist = distribution_kwargs["sigma"]
    sigma_dims = sigma_dist.get("dims", ())

    offset = pm.Normal(name=f"{name}_offset", mu=0, sigma=1, dims=desired_dims)

    check_for_deeper_nested_distribution(mu_dist, f"{name}_mu")

    mu_global = create_distribution(
        f"{name}_mu",
        mu_dist["dist"],
        mu_dist["kwargs"],
        dims=mu_dims,
    )
    mu_global = dim_handler(mu_global, mu_dims)

    check_for_deeper_nested_distribution(sigma_dist, f"{name}_sigma")

    sigma_global = create_distribution(
        f"{name}_sigma",
        sigma_dist["dist"],
        sigma_dist["kwargs"],
        dims=sigma_dims,
    )
    sigma_global = dim_handler(sigma_global, sigma_dims)

    return pm.Deterministic(
        name=name, var=mu_global + offset * sigma_global, dims=desired_dims
    )


def create_distribution(
    name: str,
    distribution_name: str,
    distribution_kwargs: dict[str, Any],
    centered: bool | None = None,
    **kwargs,
) -> pt.TensorVariable:
    """Create a PyMC distribution with the specified parameters.

    Parameters
    ----------
    name : str
        Name of the variable.
    distribution_name : str
        Name of the PyMC distribution.
    distribution_kwargs : Dict
        parameters for the distribution including any nested distributions
    **kwargs
        Additional keyword arguments for the distribution.

    Returns
    -------
    TensorVariable
        A PyMC random variable.
    """
    if centered is False:
        if distribution_name != "Normal":
            raise NonCenterInvalidDistributionError(distribution_name)
        return create_hierarchical_non_center(name, distribution_kwargs, **kwargs)

    dim_handler = create_dim_handler(kwargs.get("dims", ()))
    parameter_distributions = handle_parameter_distributions(
        name, distribution_kwargs, dim_handler=dim_handler
    )
    distribution = get_distribution(name=distribution_name)
    return distribution(name, **parameter_distributions, **kwargs)


def create_distribution_from_config(name: str, config) -> pt.TensorVariable:
    """Wrapper around create_distribution that uses a configuration dictionary.

    Parameters
    ----------
    name : str
        Name of the variable.
    config : Dict
        A configuration with the name mapping to parameter configuration.

    Returns
    -------
    TensorVariable
        A PyMC random variable.

    Examples
    --------

    .. code-block:: python

        import pymc as pm

        from pymc_marketing.model_config import create_distribution_from_config

        distribution = {
            "dist": "Normal",
            "kwargs": {"mu": 0, "sigma": 1},
        }
        config = {
            "alpha": distribution,
            "beta": distribution,
        }

        with pm.Model():
            alpha = create_distribution_from_config("alpha", config)

    """
    parameter_config = config[name]
    centered_flag = parameter_config.get("centered", True)
    try:
        dist_name = parameter_config["dist"]
        dist_kwargs = parameter_config["kwargs"]
    except KeyError:
        raise ModelConfigError(name)

    return create_distribution(
        name,
        dist_name,
        dist_kwargs,
        centered=centered_flag,
        dims=parameter_config.get("dims"),
    )


LIKELIHOOD_DISTRIBUTIONS: set[str] = {
    "AsymmetricLaplace",
    "Beta",
    "Gamma",
    "Laplace",
    "LogNormal",
    "Logistic",
    "Normal",
    "StudentT",
    "TruncatedNormal",
    "VonMises",
    "Wald",
}


class UnsupportedDistributionError(Exception):
    """Error for when an unsupported distribution is used."""

    pass


class MuAlreadyExistsError(Exception):
    """Error for when 'mu' is present in the nested 'kwargs'."""

    def __init__(self, param_config) -> None:
        self.param_config = param_config
        self.message = ("The mu parameter is already defined in the kwargs",)
        super().__init__(self.message)


def create_likelihood_distribution(
    name: str,
    param_config: dict,
    mu: pt.TensorVariable,
    observed: np.ndarray | pd.Series,
    dims: Dims,
) -> pt.TensorVariable:
    """Create observed variable for the model.

    This method prepares the distribution and its parameters as specified in the
    configuration dictionary, validates them, and constructs the likelihood
    distribution using PyMC.

    Parameters
    ----------
    name : str
        Name of the variable.
    param_config : Dict
        A configuration dictionary that must contain a 'dist' key with the name of
        the distribution and a 'kwargs' key with parameters for the distribution.
    mu : TensorVariable
        The mean of the likelihood distribution.
    observed : Union[np.ndarray, pd.Series]
        The observed data to which the likelihood distribution will be fitted.
    dims : str, sequence[str]
        The dimensions of the data.

    Returns
    -------
    TensorVariable
        The likelihood distribution constructed with PyMC.

    Raises
    ------
    UnsupportedDistributionError
        If 'kwargs' key is missing in `dist`, or the parameter configuration does
        not contain 'dist' and 'kwargs' keys, or if 'mu' is present in the nested
        'kwargs'
    """
    param_config = deepcopy(param_config)

    if param_config["dist"] not in LIKELIHOOD_DISTRIBUTIONS:
        raise UnsupportedDistributionError(
            f"""
            The distribution used for the likelihood is not allowed.
            Please, use one of the following distributions: {list(LIKELIHOOD_DISTRIBUTIONS)}.
            """
        )

    if "mu" in param_config["kwargs"]:
        raise MuAlreadyExistsError(param_config)

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
