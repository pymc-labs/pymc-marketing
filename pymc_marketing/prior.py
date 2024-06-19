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
from __future__ import annotations

import copy
from inspect import signature
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc.distributions.shape_utils import Dims


class UnsupportedShapeError(Exception):
    """Error for when the shape of the hierarchical variable is not supported."""


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

    aligned_dims = np.array(dims)[:, None] == np.array(desired_dims)

    missing_dims = aligned_dims.sum(axis=0) == 0
    new_idx = aligned_dims.argmax(axis=0)

    args = [
        "x" if missing else idx
        for (idx, missing) in zip(new_idx, missing_dims, strict=False)
    ]
    return x.dimshuffle(*args)


def create_dim_handler(desired_dims: Dims):
    """Wrapper to act like the previous `create_dim_handler` function."""

    def func(x: pt.TensorLike, dims: Dims) -> pt.TensorVariable:
        return handle_dims(x, dims, desired_dims)

    return func


def dims_to_str(obj: tuple[str, ...]) -> str:
    if len(obj) == 1:
        return f'"{obj[0]}"'

    return (
        "(" + ", ".join(f'"{i}"' if isinstance(i, str) else str(i) for i in obj) + ")"
    )


def get_pymc_distribution(name: str) -> type[pm.Distribution]:
    if not hasattr(pm, name):
        raise UnsupportedDistributionError(
            f"pymc doesn't have a distribution of name {name!r}"
        )

    return getattr(pm, name)


def get_transform(name: str):
    for module in (pt, pm.math):
        if hasattr(module, name):
            break
    else:
        module = None

    if not module:
        raise UnknownTransformError(
            f"Neither pytensor or pm.math have the function {name!r}"
        )

    return getattr(module, name)


def pymc_parameters(distribution: pm.Distribution) -> set[str]:
    return set(signature(distribution.dist).parameters.keys()) - {"kwargs", "args"}


class Prior:
    """A class to represent a prior distribution."""

    def __init__(
        self,
        distribution: str,
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
        return self._distribution

    @distribution.setter
    def distribution(self, distribution: str) -> None:
        if not isinstance(distribution, str):
            raise ValueError("Distribution must be a string")

        self._distribution = distribution
        self.pymc_distribution = get_pymc_distribution(distribution)

    @property
    def transform(self) -> str | None:
        return self._transform

    @transform.setter
    def transform(self, transform: str | None) -> None:
        if not isinstance(transform, str) and transform is not None:
            raise ValueError("Transform must be a string or None")

        self._transform = transform
        self.pytensor_transform = not transform or get_transform(transform)

    @property
    def dims(self) -> Dims:
        return self._dims

    @dims.setter
    def dims(self, dims) -> None:
        if isinstance(dims, str):
            dims = (dims,)

        self._dims = dims or ()

        self._param_dims_work()
        self._unique_dims()

    def __getitem__(self, key: str) -> Prior | Any:
        return self.parameters[key]

    def _checks(self) -> None:
        if not self.centered:
            self._correct_non_centered_distribution()

        self._parameters_are_at_least_subset_of_pymc()

    def _parameters_are_at_least_subset_of_pymc(self) -> None:
        pymc_params = pymc_parameters(self.pymc_distribution)
        if not set(self.parameters.keys()).issubset(pymc_params):
            msg = (
                f"Parameters {set(self.parameters.keys())} "
                "are not a subset of the pymc distribution "
                "parameters {set(pymc_params)}"
            )
            raise ValueError(msg)

    def _correct_non_centered_distribution(self) -> None:
        if not self.centered and self.distribution != "Normal":
            raise UnsupportedParameterizationError(
                f"Must be a Normal distribution to be non-centered not {self.distribution!r}"
            )

        if not set(self.parameters.keys()) == {"mu", "sigma"}:
            raise ValueError()

        if not any(isinstance(value, Prior) for value in self.parameters.values()):
            raise ValueError()

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
        param_str = ", ".join(
            [f"{param}={value}" for param, value in self.parameters.items()]
        )
        param_str = "" if not param_str else f", {param_str}"

        dim_str = f", dims={dims_to_str(self.dims)}" if self.dims else ""
        centered_str = f", centered={self.centered}" if not self.centered else ""
        transform_str = f', transform="{self.transform}"' if self.transform else ""
        return f'Prior("{self.distribution}"{param_str}{dim_str}{centered_str}{transform_str})'

    def __repr__(self) -> str:
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
            return self.dim_handler(
                parameter.create_variable(f"{name}_{var_name}"), parameter.dims
            )

        offset = pm.Normal(f"{name}_offset", mu=0, sigma=1, dims=self.dims)
        mu = (
            handle_variable("mu")
            if isinstance(self.parameters["mu"], Prior)
            else self.parameters["mu"]
        )
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
        """Create a PyMC variable from the prior."""
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
        import preliz as pz

        return getattr(pz, self.distribution)(**self.parameters)

    def to_json(self) -> dict[str, Any]:
        json: dict[str, Any] = {
            "dist": self.distribution,
        }
        if self.parameters:

            def handle_value(value):
                if isinstance(value, Prior):
                    return value.to_json()

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

    def constrain(self, lower: float, upper: float, **kwargs) -> Prior:
        if self.transform:
            raise ValueError("Can't constrain a transformed variable")

        new_parameters = pm.find_constrained_prior(
            self.pymc_distribution,
            lower=lower,
            upper=upper,
            init_guess=self.parameters,
            **kwargs,
        )

        return Prior(
            self.distribution,
            dims=self.dims,
            transform=self.transform,
            centered=self.centered,
            **new_parameters,
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Prior):
            return False

        return (
            self.distribution == other.distribution
            and self.dims == other.dims
            and self.centered == other.centered
            and self.parameters == other.parameters
            and self.transform == other.transform
        )

    def sample_prior(
        self, coords=None, name: str = "var", **sample_prior_predictive_kwargs
    ) -> xr.Dataset:
        with pm.Model(coords=coords):
            self.create_variable(name)

            return pm.sample_prior_predictive(**sample_prior_predictive_kwargs).prior

    def __deepcopy__(self, memo) -> Prior:
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

    def copy(self) -> Prior:
        return copy.deepcopy(self)

    def graph(self):
        """Generate a graph of the variables."""
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
        if "mu" not in pymc_parameters(self.pymc_distribution):
            raise UnsupportedDistributionError(
                f"Likelihood distribution {self.distribution!r} is not supported."
            )

        if "mu" in self.parameters:
            raise MuAlreadyExistsError(self)

        distribution = self.copy()
        distribution.parameters["mu"] = mu
        distribution.parameters["observed"] = observed
        return distribution.create_variable(name)
