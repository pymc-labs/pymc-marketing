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
"""Dims related functionality."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy, deepcopy
from typing import Any, TypeAlias

import numpy as np
import pymc as pm
import pymc.dims as pmd
from pydantic import validate_call
from pymc.distributions.shape_utils import Dims
from pymc_extras.deserialize import deserialize, register_deserialization
from pymc_extras.prior import (
    MuAlreadyExistsError,
    Prior,
    UnsupportedDistributionError,
    UnsupportedParameterizationError,
    VariableFactory,
    _dims_to_str,
    _get_pymc_parameters,
)
from pytensor import Variable
from pytensor import tensor as pt
from pytensor.tensor import TensorVariable
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable
from xarray import DataArray

XTensorLike: TypeAlias = Any


def XData(name: str, value: DataArray) -> XTensorVariable:
    """Register an xarray DataArray as pmd.Data."""
    return pmd.Data(name, value.values, dims=value.dims)


def get_xdata(model: pm.Model, name: str) -> XTensorVariable:
    """Get an XData variable from a PyMC model."""
    # Work-around for pmd.Data not registering the XTensorVariable directly in the model, but only the TensorVariable
    # TODO: Fix on PyMC, then this will simply be `model[name]`
    return as_xtensor(model[name], dims=model.named_vars_to_dims[name])


class XPrior:
    """Temporary alternative to Prior that works with dimmed distributions/variables."""

    # Taken from https://en.wikipedia.org/wiki/Location%E2%80%93scale_family
    non_centered_distributions: dict[str, dict[str, float]] = {
        "Normal": {"mu": 0, "sigma": 1},
        "StudentT": {"mu": 0, "sigma": 1},
        "ZeroSumNormal": {"sigma": 1},
    }
    """Available non-centered distributions and their default parameters."""

    pymc_distribution: type[pmd.DimDistribution]
    """The PyMC distribution class."""

    pytensor_transform: Callable[[XTensorLike], XTensorLike] | None
    """The PyTensor transform function."""

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
        try:
            self.pymc_distribution = getattr(pmd, distribution)
        except AttributeError:
            raise UnsupportedDistributionError(
                f"PyMC.dims doesn't have a distribution of name {distribution!r}"
            )

        self._distribution = distribution
        self.parameters = parameters
        self.dims = dims
        self.centered = centered
        self.transform = transform
        self._checks()

    @property
    def distribution(self) -> str:
        """The name of the PyMC distribution."""
        return self._distribution

    @staticmethod
    def _is_xprior_type(data):
        return "xdist" in data

    @property
    def transform(self) -> str | None:
        """The name of the transform to apply to the variable after it is created."""
        return self._transform

    @transform.setter
    def transform(self, transform: str | None) -> None:
        self._transform = transform
        self.pytensor_transform = not transform  # type: ignore
        # or _get_transform(transform)

    @property
    def dims(self) -> Dims:
        """The dimensions of the variable."""
        return self._dims

    @dims.setter
    def dims(self, dims) -> None:
        if isinstance(dims, str):
            dims = (dims,)

        if isinstance(dims, list):
            dims = tuple(dims)

        self._dims = dims or ()

    def __str__(self) -> str:
        """Return a string representation of the prior."""
        param_str = ", ".join(
            [f"{param}={value}" for param, value in self.parameters.items()]
        )
        param_str = "" if not param_str else f", {param_str}"

        dim_str = f", dims={_dims_to_str(self.dims)}" if self.dims else ""
        centered_str = f", centered={self.centered}" if not self.centered else ""
        transform_str = f', transform="{self.transform}"' if self.transform else ""
        return f'{self.__class__.__name__}("{self.distribution}"{param_str}{dim_str}{centered_str}{transform_str})'

    def __repr__(self) -> str:
        """Return a string representation of the prior."""
        return self.__str__()

    def __eq__(self, other) -> bool:
        """Check if two priors are equal."""
        if type(self) is not type(other):
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

    def __getitem__(self, key: str) -> XPrior | Any:
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
        supported_types = (
            int,
            float,
            np.ndarray,
            XPrior,
            TensorVariable,
            XTensorVariable,
            VariableFactory,
        )

        incorrect_types = {
            param: type(value)
            for param, value in self.parameters.items()
            if not isinstance(value, supported_types)
        }
        if incorrect_types:
            msg = (
                "Parameters must be one of the following types: "
                f"(int, float, np.array, XPrior, TensorVariable, XTensorVariable). "
                f"Incorrect parameters: {incorrect_types}"
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

    def _create_parameter(self, param, value, name):
        if not hasattr(value, "create_variable"):
            return value

        child_name = f"{name}_{param}"
        return value.create_variable(child_name)

    def _create_centered_variable(self, name: str):
        parameters = {
            param: self._create_parameter(param, value, name)
            for param, value in self.parameters.items()
        }
        return self.pymc_distribution(name, **parameters, dims=self.dims)

    def _create_non_centered_variable(self, name: str) -> pt.TensorVariable:
        def handle_variable(var_name: str):
            parameter = self.parameters[var_name]
            if not hasattr(parameter, "create_variable"):
                return parameter

            return parameter.create_variable(f"{name}_{var_name}")

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
                if isinstance(self.parameters["mu"], XPrior)
                else self.parameters["mu"]
            )
        else:
            mu = 0

        sigma = (
            handle_variable("sigma")
            if isinstance(self.parameters["sigma"], XPrior)
            else self.parameters["sigma"]
        )

        return pmd.Deterministic(
            name,
            mu + sigma * offset,
        )

    def create_variable(self, name: str) -> XTensorVariable:
        """Create the XPrior variable."""
        if self.transform:
            var_name = f"{name}_raw"

            def transform(var):
                return pmd.Deterministic(name, self.pytensor_transform(var))
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict."""
        data: dict[str, Any] = {
            "xdist": self.distribution,
        }
        if self.parameters:

            def handle_value(value):
                if isinstance(value, XPrior):
                    return value.to_dict()

                if isinstance(value, Variable):
                    value = value.eval()

                if isinstance(value, np.ndarray):
                    return value.tolist()

                if hasattr(value, "to_dict"):
                    return value.to_dict()

                return value

            data["kwargs"] = {
                param: handle_value(value) for param, value in self.parameters.items()
            }
        if not self.centered:
            data["centered"] = False

        if self.dims:
            data["dims"] = self.dims

        if self.transform:
            data["transform"] = self.transform

        return data

    @classmethod
    def from_dict(cls, data) -> XPrior:
        """Create XPrior from serialized dict."""
        if not isinstance(data, dict):
            msg = (
                "Must be a dictionary representation of a prior distribution. "
                f"Not of type: {type(data)}"
            )
            raise ValueError(msg)

        dist = data["xdist"]
        kwargs = data.get("kwargs", {})

        def handle_value(value):
            if isinstance(value, dict):
                return deserialize(value)

            if isinstance(value, list):
                return np.array(value)

            return value

        kwargs = {param: handle_value(value) for param, value in kwargs.items()}
        centered = data.get("centered", True)
        dims = data.get("dims")
        if isinstance(dims, list):
            dims = tuple(dims)
        transform = data.get("transform")

        return cls(dist, dims=dims, centered=centered, transform=transform, **kwargs)

    @classmethod
    def from_prior(cls, prior: Prior) -> XPrior:
        """Create XPrior from Prior object."""
        prior_dict = prior.to_dict()

        def _dist_to_xdist(d):
            return {
                "xdist" if key == "dist" else key: _dist_to_xdist(value)
                if isinstance(value, dict)
                else value
                for key, value in d.items()
            }

        x_prior_dict = _dist_to_xdist(prior_dict)
        return cls.from_dict(x_prior_dict)

    def __deepcopy__(self, memo) -> XPrior:
        """Return a deep copy of the prior."""
        if id(self) in memo:
            return memo[id(self)]

        copy_obj = XPrior(
            self.distribution,
            dims=copy(self.dims),
            centered=self.centered,
            transform=self.transform,
            **deepcopy(self.parameters),
        )
        memo[id(self)] = copy_obj
        return copy_obj

    def deepcopy(self) -> XPrior:
        """Return a deep copy of the prior."""
        return deepcopy(self)

    def create_likelihood_variable(
        self,
        name: str,
        mu: XTensorLike,
        observed: XTensorLike,
    ) -> XTensorVariable:
        """Create an observed variables from XPrior."""
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


register_deserialization(is_type=XPrior._is_xprior_type, deserialize=XPrior.from_dict)
