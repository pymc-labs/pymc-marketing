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
"""Base class for adstock and saturation functions used in MMM.

Use the subclasses directly for custom transformations:

- Adstock Transformations: :class:`pymc_marketing.mmm.components.adstock.AdstockTransformation`
- Saturation Transformations: :class:`pymc_marketing.mmm.components.saturation.SaturationTransformation`

"""

import warnings
from collections.abc import Iterable
from copy import deepcopy
from inspect import signature
from typing import Any

import numpy as np
import numpy.typing as npt
import pymc as pm
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import InstanceOf
from pymc.distributions.shape_utils import Dims
from pytensor import tensor as pt
from pytensor.tensor.variable import TensorVariable

from pymc_marketing.model_config import parse_model_config
from pymc_marketing.plot import (
    SelToString,
    plot_curve,
    plot_hdi,
    plot_samples,
)
from pymc_marketing.prior import Prior, VariableFactory, create_dim_handler

# "x" for saturation, "time since exposure" for adstock
NON_GRID_NAMES: frozenset[str] = frozenset({"x", "time since exposure"})

SupportedPrior = (
    InstanceOf[Prior] | float | InstanceOf[TensorVariable] | InstanceOf[VariableFactory]
)


class ParameterPriorException(Exception):
    """Error when the functions and specified priors don't match up."""

    def __init__(self, priors: set[str], parameters: set[str]) -> None:
        self.priors = priors
        self.parameters = parameters

        msg = "The function parameters and priors don't line up."

        if self.priors:
            msg = f"{msg} Missing default prior: {self.priors}."

        if self.parameters:
            msg = f"{msg} Missing function parameter: {self.parameters}."

        super().__init__(msg)


RESERVED_DATA_PARAMETER_NAMES = {"x", "data"}


class MissingDataParameter(Exception):
    """Error if the function doesn't have a data parameter."""

    def __init__(self) -> None:
        msg = (
            f"The function must have a data parameter."
            " The first parameter is assumed to be the data"
            f" with name being one of: {RESERVED_DATA_PARAMETER_NAMES}"
        )

        super().__init__(msg)


class Transformation:
    """Base class for adstock and saturation functions.

    The subclasses will need to implement the following attributes:

    - function: The function that will be applied to the data.
    - prefix: The prefix for the variables that will be created.
    - default_priors: The default priors for the parameters of the function.

    In order to make a new saturation or adstock function, use the specific subclasses:

    - :class:`pymc_marketing.mmm.components.saturation.SaturationTransformation`
    - :class:`pymc_marketing.mmm.components.adstock.AdstockTransformation`

    View the documentation for those classes for more information.

    Parameters
    ----------
    priors : dict[str, Prior | float | TensorVariable | VariableFactory], optional
        Dictionary with the priors for the parameters of the function. The keys should be the
        parameter names and the values the priors. If not provided, it will use the default
        priors from the subclass.
    prefix : str, optional
        The prefix for the variables that will be created. If not provided, it will use the prefix
        from the subclass.

    """

    prefix: str
    default_priors: dict[str, Prior]
    function: Any
    lookup_name: str

    def __init__(
        self,
        priors: dict[str, Prior | float | TensorVariable | VariableFactory]
        | None = None,
        prefix: str | None = None,
    ) -> None:
        self._checks()
        self.function_priors = priors  # type: ignore
        self.prefix = prefix or self.prefix

    def __repr__(self) -> str:
        """Representation of the transformation."""
        return (
            f"{self.__class__.__name__}("
            f"prefix={self.prefix!r}, "
            f"priors={self.function_priors}"
            ")"
        )

    def set_dims_for_all_priors(self, dims: Dims):
        """Set the dims for all priors.

        Convenience method to loop through all the priors and set the dims.

        Parameters
        ----------
        dims : Dims
            The dims for the priors.

        Returns
        -------
        Transformation
        """
        for prior in self.function_priors.values():
            prior.dims = dims

        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert the transformation to a dictionary.

        Returns
        -------
        dict
            The dictionary defining the transformation.

        """
        return {
            "lookup_name": self.lookup_name,
            "prefix": self.prefix,
            "priors": {
                key: _serialize_value(value)
                for key, value in self.function_priors.items()
            },
        }

    def __eq__(self, other: Any) -> bool:
        """Check if two transformations are equal."""
        if not isinstance(other, self.__class__):
            return False

        return self.to_dict() == other.to_dict()

    @property
    def function_priors(self) -> dict[str, Prior]:
        """Get the priors for the function."""
        return self._function_priors

    @function_priors.setter
    def function_priors(self, priors: dict[str, Any | Prior] | None) -> None:
        priors = priors or {}

        non_distributions = [
            key
            for key, value in priors.items()
            if not isinstance(value, Prior) and not isinstance(value, dict)
        ]

        priors = parse_model_config(priors, non_distributions=non_distributions)
        self._function_priors = {**deepcopy(self.default_priors), **priors}

    def update_priors(self, priors: dict[str, Prior]) -> None:
        """Update the priors for a function after initialization.

        Uses {prefix}_{parameter_name} as the key for the priors instead of the parameter name
        in order to be used in the larger MMM.

        Parameters
        ----------
        priors : dict[str, Prior]
            Dictionary with the new priors for the parameters of the function.

        Examples
        --------
        Update the priors for a transformation after initialization.

        .. code-block:: python

            from pymc_marketing.mmm.components.base import Transformation
            from pymc_marketing.prior import Prior


            class MyTransformation(Transformation):
                lookup_name: str = "my_transformation"
                prefix: str = "transformation"
                function = lambda x, lam: x * lam
                default_priors = {"lam": Prior("Gamma", alpha=3, beta=1)}


            transformation = MyTransformation()
            transformation.update_priors(
                {"transformation_lam": Prior("HalfNormal", sigma=1)},
            )

        """
        new_priors = {
            parameter_name: priors[variable_name]
            for parameter_name, variable_name in self.variable_mapping.items()
            if variable_name in priors
        }
        if not new_priors:
            available_priors = list(self.variable_mapping.values())
            warnings.warn(
                f"No priors were updated. Available parameters are {available_priors}",
                UserWarning,
                stacklevel=2,
            )

        self.function_priors.update(new_priors)

    @property
    def model_config(self) -> dict[str, Any]:
        """Mapping from variable name to prior for the model."""
        return {
            variable_name: self.function_priors[parameter_name]
            for parameter_name, variable_name in self.variable_mapping.items()
        }

    def _checks(self) -> None:
        self._has_all_attributes()
        self._function_works_on_instances()
        self._has_defaults_for_all_arguments()

    def _has_all_attributes(self) -> None:
        if not hasattr(self, "prefix"):
            raise NotImplementedError("prefix must be implemented in the subclass")

        if not hasattr(self, "default_priors"):
            raise NotImplementedError(
                "default_priors must be implemented in the subclass"
            )

        if not hasattr(self, "function"):
            raise NotImplementedError("function must be implemented in the subclass")

        if not hasattr(self, "lookup_name"):
            raise NotImplementedError("lookup_name must be implemented in the subclass")

    def _has_defaults_for_all_arguments(self) -> None:
        function_signature = signature(self.function)

        # Remove the first one as assumed to be the data
        parameters_that_need_priors = set(
            list(function_signature.parameters.keys())[1:]
        )
        parameters_with_priors = set(self.default_priors.keys())

        missing_priors = parameters_that_need_priors - parameters_with_priors
        missing_parameters = parameters_with_priors - parameters_that_need_priors

        if missing_priors or missing_parameters:
            raise ParameterPriorException(missing_priors, missing_parameters)

    def _function_works_on_instances(self) -> None:
        class_function = self.__class__.function
        function_parameters = list(signature(class_function).parameters)

        is_method = function_parameters[0] == "self"
        data_parameter_idx = 1 if is_method else 0

        has_data_parameter = (
            function_parameters[data_parameter_idx] in RESERVED_DATA_PARAMETER_NAMES
        )
        if not has_data_parameter:
            raise MissingDataParameter()

        if is_method:
            return

        self.function = class_function

    @property
    def variable_mapping(self) -> dict[str, str]:
        """Mapping from parameter name to variable name in the model."""
        return {
            parameter: f"{self.prefix}_{parameter}"
            for parameter in self.default_priors.keys()
        }

    @property
    def combined_dims(self) -> tuple[str, ...]:
        """Get the combined dims for all the parameters."""
        return tuple(self._infer_output_core_dims())

    def _infer_output_core_dims(self) -> tuple[str, ...]:
        parameter_dims = sorted(
            [
                (dims,) if isinstance(dims, str) else dims
                for dist in self.function_priors.values()
                if (dims := getattr(dist, "dims", None)) is not None
            ],
            key=len,
            reverse=True,
        )
        return tuple(list({str(dim): None for dims in parameter_dims for dim in dims}))

    def _create_distributions(
        self, dims: Dims | None = None
    ) -> dict[str, TensorVariable]:
        dim_handler = create_dim_handler(dims or self._infer_output_core_dims())

        def create_variable(parameter_name: str, variable_name: str) -> TensorVariable:
            dist = self.function_priors[parameter_name]
            if not hasattr(dist, "create_variable"):
                return dist

            var = dist.create_variable(variable_name)
            return dim_handler(var, dist.dims)

        return {
            parameter_name: create_variable(parameter_name, variable_name)
            for parameter_name, variable_name in self.variable_mapping.items()
        }

    def sample_prior(
        self, coords: dict | None = None, **sample_prior_predictive_kwargs
    ) -> xr.Dataset:
        """Sample the priors for the transformation.

        Parameters
        ----------
        coords : dict, optional
            The coordinates for the associated with dims
        **sample_prior_predictive_kwargs
            Keyword arguments for the pm.sample_prior_predictive function.

        Returns
        -------
        xr.Dataset
            The dataset with the sampled priors.

        """
        coords = coords or {}
        dims = tuple(coords.keys())
        with pm.Model(coords=coords):
            self._create_distributions(dims=dims)
            return pm.sample_prior_predictive(**sample_prior_predictive_kwargs).prior

    def plot_curve(
        self,
        curve: xr.DataArray,
        n_samples: int = 10,
        hdi_probs: float | list[float] | None = None,
        random_seed: np.random.Generator | None = None,
        subplot_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        hdi_kwargs: dict | None = None,
        axes: npt.NDArray[Axes] | None = None,
        same_axes: bool = False,
        colors: Iterable[str] | None = None,
        legend: bool | None = None,
        sel_to_string: SelToString | None = None,
    ) -> tuple[Figure, npt.NDArray[Axes]]:
        """Plot curve HDI and samples.

        Parameters
        ----------
        curve : xr.DataArray
            The curve to plot.
        n_samples : int, optional
            Number of samples
        hdi_probs : float | list[float], optional
            HDI probabilities. Defaults to None which uses arviz default for
            stats.ci_prob which is 94%
        random_seed : int | random number generator, optional
            Random number generator. Defaults to None
        subplot_kwargs : dict, optional
            Keyword arguments for plt.subplots
        sample_kwargs : dict, optional
            Keyword arguments for the plot_curve_sample function. Defaults to None.
        hdi_kwargs : dict, optional
            Keyword arguments for the plot_curve_hdi function. Defaults to None.
        axes : npt.NDArray[plt.Axes], optional
            The exact axes to plot on. Overrides any subplot_kwargs
        same_axes : bool, optional
            If the axes should be the same for all plots. Defaults to False.
        colors : Iterable[str], optional
            The colors to use for the plot. Defaults to None.
        legend : bool, optional
            If the legend should be shown. Defaults to None.
        sel_to_string : SelToString, optional
            The function to convert the selection to a string. Defaults to None.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]

        """
        return plot_curve(
            curve,
            non_grid_names=set(NON_GRID_NAMES),
            n_samples=n_samples,
            hdi_probs=hdi_probs,
            random_seed=random_seed,
            subplot_kwargs=subplot_kwargs,
            sample_kwargs=sample_kwargs,
            hdi_kwargs=hdi_kwargs,
            axes=axes,
            same_axes=same_axes,
            colors=colors,
            legend=legend,
            sel_to_string=sel_to_string,
        )

    def _sample_curve(
        self,
        var_name: str,
        parameters: xr.Dataset,
        x: pt.TensorLike,
        coords: dict[str, Any],
    ) -> xr.DataArray:
        output_core_dims = self._infer_output_core_dims()

        keys = list(coords.keys())
        if len(keys) != 1:
            msg = "The coords should only have one key."
            raise ValueError(msg)
        x_dim = keys[0]

        # Allow broadcasting
        x = np.expand_dims(
            x,
            axis=tuple(range(1, len(output_core_dims) + 1)),
        )

        coords.update(
            {
                dim: np.asarray(coord)
                for dim, coord in parameters.coords.items()
                if dim not in ["chain", "draw"]
            }
        )

        with pm.Model(coords=coords):
            pm.Deterministic(
                var_name,
                self.apply(x, dims=output_core_dims),
                dims=(x_dim, *output_core_dims),
            )

            return pm.sample_posterior_predictive(
                parameters,
                var_names=[var_name],
            ).posterior_predictive[var_name]

    def plot_curve_samples(
        self,
        curve: xr.DataArray,
        n: int = 10,
        rng: np.random.Generator | None = None,
        plot_kwargs: dict | None = None,
        subplot_kwargs: dict | None = None,
        axes: npt.NDArray[Axes] | None = None,
    ) -> tuple[Figure, npt.NDArray[Axes]]:
        """Plot samples from the curve.

        Parameters
        ----------
        curve : xr.DataArray
            The curve to plot.
        n : int, optional
            The number of samples to plot. Defaults to 10.
        rng : np.random.Generator, optional
            The random number generator to use. Defaults to None.
        plot_kwargs : dict, optional
            Keyword arguments for the DataFrame plot function. Defaults to None.
        subplot_kwargs : dict, optional
            Keyword arguments for plt.subplots
        axes : npt.NDArray[plt.Axes], optional
            The exact axes to plot on. Overrides any subplot_kwargs

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
        plt.Axes
            The axes with the plot.

        """
        return plot_samples(
            curve,
            non_grid_names=set(NON_GRID_NAMES),
            n=n,
            rng=rng,
            axes=axes,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
        )

    def plot_curve_hdi(
        self,
        curve: xr.DataArray,
        hdi_kwargs: dict | None = None,
        plot_kwargs: dict | None = None,
        subplot_kwargs: dict | None = None,
        axes: npt.NDArray[Axes] | None = None,
    ) -> tuple[Figure, npt.NDArray[Axes]]:
        """Plot the HDI of the curve.

        Parameters
        ----------
        curve : xr.DataArray
            The curve to plot.
        hdi_kwargs : dict, optional
            Keyword arguments for the az.hdi function. Defaults to None.
        plot_kwargs : dict, optional
            Keyword arguments for the fill_between function. Defaults to None.
        subplot_kwargs : dict, optional
            Keyword arguments for plt.subplots
        axes : npt.NDArray[plt.Axes], optional
            The exact axes to plot on. Overrides any subplot_kwargs

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]

        """
        return plot_hdi(
            curve,
            non_grid_names=set(NON_GRID_NAMES),
            axes=axes,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
            hdi_kwargs=hdi_kwargs,
        )

    def apply(self, x: pt.TensorLike, dims: Dims | None = None) -> TensorVariable:
        """Call within a model context.

        Used internally of the MMM to apply the transformation to the data.

        Parameters
        ----------
        x : pt.TensorLike
            The data to be transformed.
        dims : str, sequence[str], optional
            The dims of the parameters. Defaults to None. Not the dims of the
            data!

        Returns
        -------
        pt.TensorVariable
            The transformed data.

        Examples
        --------
        Call the function for custom use-case

        .. code-block:: python

            import pymc as pm

            transformation = ...

            coords = {"channel": ["TV", "Radio", "Digital"]}
            with pm.Model(coords=coords):
                transformed_data = transformation.apply(data, dims="channel")

        """
        kwargs = self._create_distributions(dims=dims)
        return self.function(x, **kwargs)


def _serialize_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()

    if isinstance(value, TensorVariable):
        value = value.eval()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


class DuplicatedTransformationError(Exception):
    """Exception when a transformation is duplicated."""

    def __init__(self, name: str, lookup_name: str):
        self.name = name
        self.lookup_name = lookup_name
        super().__init__(f"Duplicate {name}. The name {lookup_name!r} already exists.")


def create_registration_meta(subclasses: dict[str, Any]) -> type[type]:
    """Create a metaclass for registering subclasses.

    Parameters
    ----------
    subclasses : dict[str, type[Transformation]]
        The subclasses to register.

    Returns
    -------
    type
        The metaclass for registering subclasses.

    """

    class RegistrationMeta(type):
        def __new__(cls, name, bases, attrs):
            new_cls = super().__new__(cls, name, bases, attrs)

            if "lookup_name" not in attrs:
                return new_cls

            base_name = bases[0].__name__

            lookup_name = attrs["lookup_name"]
            if lookup_name in subclasses:
                raise DuplicatedTransformationError(base_name, lookup_name)

            subclasses[lookup_name] = new_cls

            return new_cls

    return RegistrationMeta
