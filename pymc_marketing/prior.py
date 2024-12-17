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
"""Class that represents a prior distribution.

The `Prior` class is a wrapper around PyMC distributions that allows the user
to create outside of the PyMC model.

This is the alternative to using the dictionaries in PyMC-Marketing models.

Examples
--------
Create a normal prior.

.. code-block:: python

    from pymc_marketing.prior import Prior

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

    from pymc_marketing.prior import register_tensor_transform

    def custom_transform(x):
        return x ** 2

    register_tensor_transform("square", custom_transform)

    custom_distribution = Prior("Normal", transform="square")

"""

from __future__ import annotations

import copy
from collections.abc import Callable
from inspect import signature
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pydantic import validate_call
from pymc.distributions.shape_utils import Dims


class UnsupportedShapeError(Exception):
    """Error for when the shapes from variables are not compatible."""


class UnsupportedDistributionError(Exception):
    """Error for when an unsupported distribution is used."""


class UnsupportedParameterizationError(Exception):
    """The follow parameterization is not supported."""


class MuAlreadyExistsError(Exception):
    """Error for when 'mu' is present in Prior."""

    def __init__(self, distribution: Prior) -> None:
        self.distribution = distribution
        self.message = f"The mu parameter is already defined in {distribution}"
        super().__init__(self.message)


class UnknownTransformError(Exception):
    """Error for when an unknown transform is used."""


def _remove_leading_xs(args: list[str | int]) -> list[str | int]:
    """Remove leading 'x' from the args."""
    while args and args[0] == "x":
        args.pop(0)

    return args


def handle_dims(x: pt.TensorLike, dims: Dims, desired_dims: Dims) -> pt.TensorVariable:
    """Take a tensor of dims `dims` and align it to `desired_dims`.

    Doesn't check for validity of the dims

    Examples
    --------
    1D to 2D with new dim

    .. code-block:: python

        x = np.array([1, 2, 3])
        dims = "channel"

        desired_dims = ("channel", "group")

        handle_dims(x, dims, desired_dims)

    """
    x = pt.as_tensor_variable(x)

    if np.ndim(x) == 0:
        return x

    dims = dims if isinstance(dims, tuple) else (dims,)
    desired_dims = desired_dims if isinstance(desired_dims, tuple) else (desired_dims,)

    if difference := set(dims).difference(desired_dims):
        raise UnsupportedShapeError(
            f"Dims {dims} of data are not a subset of the desired dims {desired_dims}. "
            f"{difference} is missing from the desired dims."
        )

    aligned_dims = np.array(dims)[:, None] == np.array(desired_dims)

    missing_dims = aligned_dims.sum(axis=0) == 0
    new_idx = aligned_dims.argmax(axis=0)

    args = [
        "x" if missing else idx
        for (idx, missing) in zip(new_idx, missing_dims, strict=False)
    ]
    args = _remove_leading_xs(args)
    return x.dimshuffle(*args)


DimHandler = Callable[[pt.TensorLike, Dims], pt.TensorLike]


def create_dim_handler(desired_dims: Dims) -> DimHandler:
    """Wrap the `handle_dims` function to act like the previous `create_dim_handler` function."""

    def func(x: pt.TensorLike, dims: Dims) -> pt.TensorVariable:
        return handle_dims(x, dims, desired_dims)

    return func


def _dims_to_str(obj: tuple[str, ...]) -> str:
    if len(obj) == 1:
        return f'"{obj[0]}"'

    return (
        "(" + ", ".join(f'"{i}"' if isinstance(i, str) else str(i) for i in obj) + ")"
    )


def _get_pymc_distribution(name: str) -> type[pm.Distribution]:
    if not hasattr(pm, name):
        raise UnsupportedDistributionError(
            f"PyMC doesn't have a distribution of name {name!r}"
        )

    return getattr(pm, name)


Transform = Callable[[pt.TensorLike], pt.TensorLike]

CUSTOM_TRANSFORMS: dict[str, Transform] = {}


def register_tensor_transform(name: str, transform: Transform) -> None:
    """Register a tensor transform function to be used in the `Prior` class.

    Parameters
    ----------
    name : str
        The name of the transform.
    func : Callable[[pt.TensorLike], pt.TensorLike]
        The function to apply to the tensor.

    Examples
    --------
    Register a custom transform function.

    .. code-block:: python

        from pymc_marketing.prior import (
            Prior,
            register_tensor_transform,
        )

        def custom_transform(x):
            return x ** 2

        register_tensor_transform("square", custom_transform)

        custom_distribution = Prior("Normal", transform="square")

    """
    CUSTOM_TRANSFORMS[name] = transform


def _get_transform(name: str):
    if name in CUSTOM_TRANSFORMS:
        return CUSTOM_TRANSFORMS[name]

    for module in (pt, pm.math):
        if hasattr(module, name):
            break
    else:
        module = None

    if not module:
        msg = (
            f"Neither pytensor.tensor nor pymc.math have the function {name!r}. "
            "If this is a custom function, register it with the "
            "`pymc_marketing.prior.register_tensor_transform` function before "
            "previous function call."
        )

        raise UnknownTransformError(msg)

    return getattr(module, name)


def _get_pymc_parameters(distribution: pm.Distribution) -> set[str]:
    return set(signature(distribution.dist).parameters.keys()) - {"kwargs", "args"}


class Prior:
    """A class to represent a prior distribution.

    This is the alternative to using the dictionaries in PyMC-Marketing models
    but provides added flexibility and functionality.

    Make use of the various helper methods to understand the distributions
    better.

    - `preliz` attribute to get the equivalent distribution in `preliz`
    - `sample_prior` method to sample from the prior
    - `graph` get a dummy model graph with the distribution
    - `constrain` to shift the distribution to a different range

    Parameters
    ----------
    distribution : str
        The name of PyMC distribution.
    dims : Dims, optional
        The dimensions of the variable, by default None
    centered : bool, optional
        Whether the variable is centered or not, by default True.
        Only allowed for Normal distribution.
    transform : str, optional
        The name of the transform to apply to the variable after it is
        created, by default None or no transform. The transformation must
        be registered with `register_tensor_transform` function or
        be available in either `pytensor.tensor` or `pymc.math`.

    """

    # Taken from https://en.wikipedia.org/wiki/Location%E2%80%93scale_family
    non_centered_distributions: dict[str, dict[str, float]] = {
        "Normal": {"mu": 0, "sigma": 1},
        "StudentT": {"mu": 0, "sigma": 1},
        "ZeroSumNormal": {"sigma": 1},
    }

    pymc_distribution: type[pm.Distribution]
    pytensor_transform: Callable[[pt.TensorLike], pt.TensorLike] | None

    @validate_call
    def __init__(
        self,
        distribution: str,
        *,
        dims: Dims | None = None,
        centered: bool = True,
        transform: str | None = None,
        **parameters,
    ) -> None:
        self.distribution = distribution
        self.parameters = parameters
        self.dims = dims
        self.centered = centered
        self.transform = transform

        self._checks()

    @property
    def distribution(self) -> str:
        """The name of the PyMC distribution."""
        return self._distribution

    @distribution.setter
    def distribution(self, distribution: str) -> None:
        if hasattr(self, "_distribution"):
            raise AttributeError("Can't change the distribution")

        self._distribution = distribution
        self.pymc_distribution = _get_pymc_distribution(distribution)

    @property
    def transform(self) -> str | None:
        """The name of the transform to apply to the variable after it is created."""
        return self._transform

    @transform.setter
    def transform(self, transform: str | None) -> None:
        self._transform = transform
        self.pytensor_transform = not transform or _get_transform(transform)  # type: ignore

    @property
    def dims(self) -> Dims:
        """The dimensions of the variable."""
        return self._dims

    @dims.setter
    def dims(self, dims) -> None:
        if isinstance(dims, str):
            dims = (dims,)

        self._dims = dims or ()

        self._param_dims_work()
        self._unique_dims()

    def __getitem__(self, key: str) -> Prior | Any:
        """Return the parameter of the prior."""
        return self.parameters[key]

    def _checks(self) -> None:
        if not self.centered:
            self._correct_non_centered_distribution()

        self._parameters_are_at_least_subset_of_pymc()
        self._convert_lists_to_numpy()
        self._parameters_are_correct_type()

    def _parameters_are_at_least_subset_of_pymc(self) -> None:
        pymc_params = _get_pymc_parameters(self.pymc_distribution)
        if not set(self.parameters.keys()).issubset(pymc_params):
            msg = (
                f"Parameters {set(self.parameters.keys())} "
                "are not a subset of the pymc distribution "
                f"parameters {set(pymc_params)}"
            )
            raise ValueError(msg)

    def _convert_lists_to_numpy(self) -> None:
        def convert(x):
            if not isinstance(x, list):
                return x

            return np.array(x)

        self.parameters = {
            key: convert(value) for key, value in self.parameters.items()
        }

    def _parameters_are_correct_type(self) -> None:
        supported_types = (int, float, np.ndarray, Prior, pt.TensorVariable)

        incorrect_types = {
            param: type(value)
            for param, value in self.parameters.items()
            if not isinstance(value, supported_types)
        }
        if incorrect_types:
            msg = (
                "Parameters must be one of the following types: "
                f"(int, float, np.array, Prior, pt.TensorVariable). Incorrect parameters: {incorrect_types}"
            )
            raise ValueError(msg)

    def _correct_non_centered_distribution(self) -> None:
        if (
            not self.centered
            and self.distribution not in self.non_centered_distributions
        ):
            raise UnsupportedParameterizationError(
                f"{self.distribution!r} is not supported for non-centered parameterization. "
                f"Choose from {list(self.non_centered_distributions.keys())}"
            )

        required_parameters = set(
            self.non_centered_distributions[self.distribution].keys()
        )

        if set(self.parameters.keys()) < required_parameters:
            msg = " and ".join([f"{param!r}" for param in required_parameters])
            raise ValueError(
                f"Must have at least {msg} parameter for non-centered for {self.distribution!r}"
            )

    def _unique_dims(self) -> None:
        if not self.dims:
            return

        if len(self.dims) != len(set(self.dims)):
            raise ValueError("Dims must be unique")

    def _param_dims_work(self) -> None:
        other_dims = set()
        for value in self.parameters.values():
            if isinstance(value, Prior):
                other_dims.update(value.dims)

        if not other_dims.issubset(self.dims):
            raise UnsupportedShapeError(
                f"Parameter dims {other_dims} are not a subset of the prior dims {self.dims}"
            )

    def __str__(self) -> str:
        """Return a string representation of the prior."""
        param_str = ", ".join(
            [f"{param}={value}" for param, value in self.parameters.items()]
        )
        param_str = "" if not param_str else f", {param_str}"

        dim_str = f", dims={_dims_to_str(self.dims)}" if self.dims else ""
        centered_str = f", centered={self.centered}" if not self.centered else ""
        transform_str = f', transform="{self.transform}"' if self.transform else ""
        return f'Prior("{self.distribution}"{param_str}{dim_str}{centered_str}{transform_str})'

    def __repr__(self) -> str:
        """Return a string representation of the prior."""
        return f"{self}"

    def _create_parameter(self, param, value, name):
        if not isinstance(value, Prior):
            return value

        child_name = f"{name}_{param}"
        return self.dim_handler(value.create_variable(child_name), value.dims)

    def _create_centered_variable(self, name: str):
        parameters = {
            param: self._create_parameter(param, value, name)
            for param, value in self.parameters.items()
        }
        return self.pymc_distribution(name, **parameters, dims=self.dims)

    def _create_non_centered_variable(self, name: str) -> pt.TensorVariable:
        def handle_variable(var_name: str):
            parameter = self.parameters[var_name]
            if not isinstance(parameter, Prior):
                return parameter

            return self.dim_handler(
                parameter.create_variable(f"{name}_{var_name}"),
                parameter.dims,
            )

        defaults = self.non_centered_distributions[self.distribution]
        other_parameters = {
            param: handle_variable(param)
            for param in self.parameters.keys()
            if param not in defaults
        }
        offset = self.pymc_distribution(
            f"{name}_offset",
            **defaults,
            **other_parameters,
            dims=self.dims,
        )
        if "mu" in self.parameters:
            mu = (
                handle_variable("mu")
                if isinstance(self.parameters["mu"], Prior)
                else self.parameters["mu"]
            )
        else:
            mu = 0

        sigma = (
            handle_variable("sigma")
            if isinstance(self.parameters["sigma"], Prior)
            else self.parameters["sigma"]
        )

        return pm.Deterministic(
            name,
            mu + sigma * offset,
            dims=self.dims,
        )

    def create_variable(self, name: str) -> pt.TensorVariable:
        """Create a PyMC variable from the prior.

        Must be used in a PyMC model context.

        Parameters
        ----------
        name : str
            The name of the variable.

        Returns
        -------
        pt.TensorVariable
            The PyMC variable.

        Examples
        --------
        Create a hierarchical normal variable in larger PyMC model.

        .. code-block:: python

            dist = Prior(
                "Normal",
                mu=Prior("Normal"),
                sigma=Prior("HalfNormal"),
                dims="channel",
            )

            coords = {"channel": ["C1", "C2", "C3"]}
            with pm.Model(coords=coords):
                var = dist.create_variable("var")

        """
        self.dim_handler = create_dim_handler(self.dims)

        if self.transform:
            var_name = f"{name}_raw"

            def transform(var):
                return pm.Deterministic(
                    name, self.pytensor_transform(var), dims=self.dims
                )
        else:
            var_name = name

            def transform(var):
                return var

        create_variable = (
            self._create_centered_variable
            if self.centered
            else self._create_non_centered_variable
        )
        var = create_variable(name=var_name)
        return transform(var)

    @property
    def preliz(self):
        """Create an equivalent preliz distribution.

        Helpful to visualize a distribution when it is univariate.

        Returns
        -------
        preliz.distributions.Distribution

        Examples
        --------
        Create a preliz distribution from a prior.

        .. code-block:: python

            from pymc_marketing.prior import Prior

            dist = Prior("Gamma", alpha=5, beta=1)
            dist.preliz.plot_pdf()

        """
        import preliz as pz

        return getattr(pz, self.distribution)(**self.parameters)

    def to_json(self) -> dict[str, Any]:
        """Convert the prior to the previous dictionary format.

        Returns
        -------
        dict[str, Any]
            The dictionary format of the prior.

        Examples
        --------
        Convert a prior to the dictionary format.

        .. code-block:: python

            from pymc_marketing.prior import Prior

            dist = Prior("Normal", mu=0, sigma=1)

            dist.to_json()
            # {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}}

        Convert a hierarchical prior to the dictionary format.

        .. code-block:: python

            dist = Prior(
                "Normal",
                mu=Prior("Normal"),
                sigma=Prior("HalfNormal"),
                dims="channel",
            )

            dist.to_json()
            # {
            #     "dist": "Normal",
            #     "kwargs": {
            #         "mu": {"dist": "Normal"},
            #         "sigma": {"dist": "HalfNormal"},
            #     },
            #     "dims": "channel",
            # }

        """
        json: dict[str, Any] = {
            "dist": self.distribution,
        }
        if self.parameters:

            def handle_value(value):
                if isinstance(value, Prior):
                    return value.to_json()

                if isinstance(value, pt.TensorVariable):
                    value = value.eval()

                if isinstance(value, np.ndarray):
                    return value.tolist()

                return value

            json["kwargs"] = {
                param: handle_value(value) for param, value in self.parameters.items()
            }
        if not self.centered:
            json["centered"] = False

        if self.dims:
            json["dims"] = self.dims

        if self.transform:
            json["transform"] = self.transform

        return json

    @classmethod
    def from_json(cls, json) -> Prior:
        """Create a Prior from the dictionary format.

        Parameters
        ----------
        json : dict[str, Any]
            The dictionary format of the prior.

        Returns
        -------
        Prior
            The prior distribution.

        Examples
        --------
        Convert prior in the dictionary format to a Prior instance.

        .. code-block:: python

            from pymc_marketing.prior import Prior

            json = {
                "dist": "Normal",
                "kwargs": {"mu": 0, "sigma": 1},
            }

            dist = Prior.from_json(json)
            dist
            # Prior("Normal", mu=0, sigma=1)

        """
        if not isinstance(json, dict):
            msg = (
                "Must be a dictionary representation of a prior distribution. "
                f"Not of type: {type(json)}"
            )
            raise ValueError(msg)

        dist = json["dist"]
        kwargs = json.get("kwargs", {})

        def handle_value(value):
            if isinstance(value, dict):
                return cls.from_json(value)

            if isinstance(value, list):
                return np.array(value)

            return value

        kwargs = {param: handle_value(value) for param, value in kwargs.items()}
        centered = json.get("centered", True)
        dims = json.get("dims")
        if isinstance(dims, list):
            dims = tuple(dims)
        transform = json.get("transform")

        return cls(dist, dims=dims, centered=centered, transform=transform, **kwargs)

    def constrain(
        self, lower: float, upper: float, mass: float = 0.95, kwargs=None
    ) -> Prior:
        """Create a new prior with a given mass constrained within the given bounds.

        Wrapper around `preliz.maxent`.

        Parameters
        ----------
        lower : float
            The lower bound.
        upper : float
            The upper bound.
        mass: float = 0.95
            The mass of the distribution to keep within the bounds.
        kwargs : dict
            Additional arguments to pass to `pz.maxent`.

        Returns
        -------
        Prior
            The maximum entropy prior with a mass constrained to the given bounds.

        Examples
        --------
        Create a Beta distribution that is constrained to have 95% of the mass
        between 0.5 and 0.8.

        .. code-block:: python

            dist = Prior(
                "Beta",
            ).constrain(lower=0.5, upper=0.8)

        Create a Beta distribution with mean 0.6, that is constrained to
        have 95% of the mass between 0.5 and 0.8.

        .. code-block:: python

            dist = Prior(
                "Beta",
                mu=0.6,
            ).constrain(lower=0.5, upper=0.8)

        """
        from preliz import maxent

        if self.transform:
            raise ValueError("Can't constrain a transformed variable")

        if kwargs is None:
            kwargs = {}
            kwargs.setdefault("plot", False)

        if kwargs["plot"]:
            new_parameters = maxent(self.preliz, lower, upper, mass, **kwargs)[
                0
            ].params_dict
        else:
            new_parameters = maxent(
                self.preliz, lower, upper, mass, **kwargs
            ).params_dict

        return Prior(
            self.distribution,
            dims=self.dims,
            transform=self.transform,
            centered=self.centered,
            **new_parameters,
        )

    def __eq__(self, other) -> bool:
        """Check if two priors are equal."""
        if not isinstance(other, Prior):
            return False

        try:
            np.testing.assert_equal(self.parameters, other.parameters)
        except AssertionError:
            return False

        return (
            self.distribution == other.distribution
            and self.dims == other.dims
            and self.centered == other.centered
            and self.transform == other.transform
        )

    def sample_prior(
        self, coords=None, name: str = "var", **sample_prior_predictive_kwargs
    ) -> xr.Dataset:
        """Sample the prior distribution for the variable.

        Parameters
        ----------
        coords : dict[str, list[str]], optional
            The coordinates for the variable, by default None.
            Only required if the dims are specified.
        name : str, optional
            The name of the variable, by default "var".
        sample_prior_predictive_kwargs : dict
            Additional arguments to pass to `pm.sample_prior_predictive`.

        Returns
        -------
        xr.Dataset
            The dataset of the prior samples.

        Example
        -------
        Sample from a hierarchical normal distribution.

        .. code-block:: python

            dist = Prior(
                "Normal",
                mu=Prior("Normal"),
                sigma=Prior("HalfNormal"),
                dims="channel",
            )

            coords = {"channel": ["C1", "C2", "C3"]}
            prior = dist.sample_prior(coords=coords)

        """
        coords = coords or {}

        if missing_keys := set(self.dims) - set(coords.keys()):
            raise KeyError(f"Coords are missing the following dims: {missing_keys}")

        with pm.Model(coords=coords):
            self.create_variable(name)

            return pm.sample_prior_predictive(**sample_prior_predictive_kwargs).prior

    def __deepcopy__(self, memo) -> Prior:
        """Return a deep copy of the prior."""
        if id(self) in memo:
            return memo[id(self)]

        copy_obj = Prior(
            self.distribution,
            dims=copy.copy(self.dims),
            centered=self.centered,
            transform=self.transform,
            **copy.deepcopy(self.parameters),
        )
        memo[id(self)] = copy_obj
        return copy_obj

    def deepcopy(self) -> Prior:
        """Return a deep copy of the prior."""
        return copy.deepcopy(self)

    def to_graph(self):
        """Generate a graph of the variables.

        Examples
        --------
        Create the graph for a 2D transformed hierarchical distribution.

        .. code-block:: python

            from pymc_marketing.prior import Prior

            mu = Prior(
                "Normal",
                mu=Prior("Normal"),
                sigma=Prior("HalfNormal"),
                dims="channel",
            )
            sigma = Prior("HalfNormal", dims="channel")
            dist = Prior(
                "Normal",
                mu=mu,
                sigma=sigma,
                dims=("channel", "geo"),
                centered=False,
                transform="sigmoid",
            )

            dist.to_graph()

        .. image:: /_static/example-graph.png
            :alt: Example graph

        """
        coords = {name: ["DUMMY"] for name in self.dims}
        with pm.Model(coords=coords) as model:
            self.create_variable("var")

        return pm.model_to_graphviz(model)

    def create_likelihood_variable(
        self,
        name: str,
        mu: pt.TensorLike,
        observed: pt.TensorLike,
    ) -> pt.TensorVariable:
        """Create a likelihood variable from the prior.

        Will require that the distribution has a `mu` parameter
        and that it has not been set in the parameters.

        Parameters
        ----------
        name : str
            The name of the variable.
        mu : pt.TensorLike
            The mu parameter for the likelihood.
        observed : pt.TensorLike
            The observed data.

        Returns
        -------
        pt.TensorVariable
            The PyMC variable.

        Examples
        --------
        Create a likelihood variable in a larger PyMC model.

        .. code-block:: python

            import pymc as pm

            dist = Prior("Normal", sigma=Prior("HalfNormal"))

            with pm.Model():
                # Create the likelihood variable
                mu = pm.Normal("mu", mu=0, sigma=1)
                dist.create_likelihood_variable("y", mu=mu, observed=observed)

        """
        if "mu" not in _get_pymc_parameters(self.pymc_distribution):
            raise UnsupportedDistributionError(
                f"Likelihood distribution {self.distribution!r} is not supported."
            )

        if "mu" in self.parameters:
            raise MuAlreadyExistsError(self)

        distribution = self.deepcopy()
        distribution.parameters["mu"] = mu
        distribution.parameters["observed"] = observed
        return distribution.create_variable(name)
