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
"""Base class for adstock and saturation functions used in MMM.

Use the subclasses directly for custom transformations:

- Adstock Transformations: :class:`pymc_marketing.mmm.components.adstock.AdstockTransformation`
- Saturation Transformations: :class:`pymc_marketing.mmm.components.saturation.SaturationTransformation`

"""

import warnings
from collections.abc import Generator, MutableMapping, Sequence
from copy import deepcopy
from inspect import signature
from itertools import product
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pymc as pm
import xarray as xr
from pymc.distributions.shape_utils import Dims
from pytensor import tensor as pt

from pymc_marketing.model_config import (
    DimHandler,
    create_dim_handler,
    create_distribution,
)

Values = Sequence[Any] | npt.NDArray[Any]
Coords = dict[str, Values]

# chain and draw from sampling
# "x" for saturation, "time since exposure" for adstock
NON_GRID_NAMES = {"chain", "draw", "x", "time since exposure"}


def get_plot_coords(coords: Coords) -> Coords:
    plot_coord_names = list(key for key in coords.keys() if key not in NON_GRID_NAMES)
    return {name: np.array(coords[name]) for name in plot_coord_names}


def get_total_coord_size(coords: Coords) -> int:
    total_size: int = (
        1 if coords == {} else np.prod([len(values) for values in coords.values()])  # type: ignore
    )
    if total_size >= 12:
        warnings.warn("Large number of coordinates!", stacklevel=2)

    return total_size


def set_subplot_kwargs_defaults(
    subplot_kwargs: MutableMapping[str, Any],
    total_size: int,
) -> None:
    if "ncols" in subplot_kwargs and "nrows" in subplot_kwargs:
        raise ValueError("Only specify one")

    if "ncols" not in subplot_kwargs and "nrows" not in subplot_kwargs:
        subplot_kwargs["ncols"] = total_size

    if "ncols" in subplot_kwargs:
        subplot_kwargs["nrows"] = total_size // subplot_kwargs["ncols"]
    elif "nrows" in subplot_kwargs:
        subplot_kwargs["ncols"] = total_size // subplot_kwargs["nrows"]


def selections(
    coords: Coords,
) -> Generator[dict[str, Any], None, None]:
    """Helper to create generator of selections."""
    coord_names = coords.keys()
    for values in product(*coords.values()):
        yield {name: value for name, value in zip(coord_names, values, strict=True)}


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
    priors : dict, optional
        Dictionary with the priors for the parameters of the function. The keys should be the
        parameter names and the values should be dictionaries with the distribution and kwargs.
    prefix : str, optional
        The prefix for the variables that will be created. If not provided, it will use the prefix
        from the subclass.

    """

    prefix: str
    default_priors: dict[str, Any]
    function: Any
    lookup_name: str

    def __init__(self, priors: dict | None = None, prefix: str | None = None) -> None:
        self._checks()
        priors = priors or {}
        self.function_priors = {**deepcopy(self.default_priors), **priors}
        self.prefix = prefix or self.prefix

    def update_priors(self, priors: dict[str, Any]) -> None:
        """Helper to update the priors for a function after initialization.

        Uses {prefix}_{parameter_name} as the key for the priors instead of the parameter name
        in order to be used in the larger MMM.

        Parameters
        ----------
        priors : dict
            Dictionary with the new priors for the parameters of the function.

        Examples
        --------
        Update the priors for a transformation after initialization.

        from pymc_marketing.mmm.components.base import Transformation

        .. code-block:: python

            class MyTransformation(Transformation):
                prefix: str = "transformation"
                function = lambda x, lam: x * lam
                default_priors = {"lam": {"dist": "Gamma", "kwargs": {"alpha": 3, "beta": 1}}}

            transformation = MyTransformation()
            transformation.update_priors(
                {"transformation_lam": {"dist": "HalfNormal", "kwargs": {"sigma": 1}}}
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

    def _create_distributions(
        self, dims: Dims | None = None
    ) -> dict[str, pt.TensorVariable]:
        dim_handler: DimHandler = create_dim_handler(dims)
        distributions: dict[str, pt.TensorVariable] = {}
        for parameter_name, variable_name in self.variable_mapping.items():
            parameter_prior = self.function_priors[parameter_name]

            var_dims = parameter_prior.get("dims")
            var = create_distribution(
                name=variable_name,
                distribution_name=parameter_prior["dist"],
                distribution_kwargs=parameter_prior["kwargs"],
                dims=var_dims,
            )

            distributions[parameter_name] = dim_handler(var, var_dims)

        return distributions

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
        subplot_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        hdi_kwargs: dict | None = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """Plot curve HDI and samples.

        Parameters
        ----------
        curve : xr.DataArray
            The curve to plot.
        subplot_kwargs : dict, optional
            Keyword arguments for plt.subplots
        sample_kwargs : dict, optional
            Keyword arguments for the plot_curve_sample function. Defaults to None.
        hdi_kwargs : dict, optional
            Keyword arguments for the plot_curve_hdi function. Defaults to None.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]

        """
        hdi_kwargs = hdi_kwargs or {}
        sample_kwargs = sample_kwargs or {}

        if "subplot_kwargs" not in hdi_kwargs:
            hdi_kwargs["subplot_kwargs"] = subplot_kwargs

        fig, axes = self.plot_curve_hdi(curve, **hdi_kwargs)
        fig, axes = self.plot_curve_samples(curve, axes=axes, **sample_kwargs)

        return fig, axes

    def _sample_curve(
        self,
        var_name: str,
        parameters: xr.Dataset,
        x: pt.TensorLike,
        coords: dict[str, Any],
    ) -> xr.DataArray:
        required_vars = list(self.variable_mapping.values())

        keys = list(coords.keys())
        if len(keys) != 1:
            msg = "The coords should only have one key."
            raise ValueError(msg)
        x_dim = keys[0]

        function_parameters = parameters[required_vars]

        parameter_coords = function_parameters.coords

        additional_coords = {
            coord: parameter_coords[coord].to_numpy()
            for coord in parameter_coords.keys()
            if coord not in {"chain", "draw"}
        }

        dims = tuple(additional_coords.keys())
        # Allow broadcasting
        x = np.expand_dims(
            x,
            axis=tuple(range(1, len(dims) + 1)),
        )

        coords.update(additional_coords)

        with pm.Model(coords=coords):
            pm.Deterministic(
                var_name,
                self.apply(x, dims=dims),
                dims=(x_dim, *dims),
            )

            return pm.sample_posterior_predictive(
                function_parameters,
                var_names=[var_name],
            ).posterior_predictive[var_name]

    def plot_curve_samples(
        self,
        curve: xr.DataArray,
        n: int = 10,
        rng: np.random.Generator | None = None,
        plot_kwargs: dict | None = None,
        subplot_kwargs: dict | None = None,
        axes: npt.NDArray[plt.Axes] | None = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
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
        plot_coords = get_plot_coords(curve.coords)
        total_size = get_total_coord_size(plot_coords)

        if axes is None:
            subplot_kwargs = subplot_kwargs or {}
            set_subplot_kwargs_defaults(subplot_kwargs, total_size)
            fig, axes = plt.subplots(**subplot_kwargs)
        else:
            fig = plt.gcf()

        plot_kwargs = plot_kwargs or {}
        plot_kwargs["alpha"] = plot_kwargs.get("alpha", 0.3)
        plot_kwargs["legend"] = False

        for i, (ax, sel) in enumerate(
            zip(np.ravel(axes), selections(plot_coords), strict=False)
        ):
            color = f"C{i}"

            df_curve = curve.sel(sel).to_series().unstack()
            df_sample = df_curve.sample(n=n, random_state=rng)

            df_sample.T.plot(ax=ax, color=color, **plot_kwargs)
            title = ", ".join(f"{name}={value}" for name, value in sel.items())
            ax.set_title(title)

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        return fig, axes

    def plot_curve_hdi(
        self,
        curve: xr.DataArray,
        hdi_kwargs: dict | None = None,
        plot_kwargs: dict | None = None,
        subplot_kwargs: dict | None = None,
        axes: npt.NDArray[plt.Axes] | None = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
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
        plot_coords = get_plot_coords(curve.coords)
        total_size = get_total_coord_size(plot_coords)

        hdi_kwargs = hdi_kwargs or {}
        conf = az.hdi(curve, **hdi_kwargs)[curve.name]

        if axes is None:
            subplot_kwargs = subplot_kwargs or {}
            set_subplot_kwargs_defaults(subplot_kwargs, total_size)
            fig, axes = plt.subplots(**subplot_kwargs)
        else:
            fig = plt.gcf()

        plot_kwargs = plot_kwargs or {}
        plot_kwargs["alpha"] = plot_kwargs.get("alpha", 0.3)

        for i, (ax, sel) in enumerate(
            zip(np.ravel(axes), selections(plot_coords), strict=False)
        ):
            color = f"C{i}"
            df_conf = conf.sel(sel).to_series().unstack()

            ax.fill_between(
                x=df_conf.index,
                y1=df_conf["lower"],
                y2=df_conf["higher"],
                color=color,
                **plot_kwargs,
            )
            title = ", ".join(f"{name}={value}" for name, value in sel.items())
            ax.set_title(title)

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        return fig, axes

    def apply(self, x: pt.TensorLike, dims: Dims | None = None) -> pt.TensorVariable:
        """Called within a model context.

        Used internally of the MMM to apply the transformation to the data.

        Parameters
        ----------
        x : pt.TensorLike
            The data to be transformed.
        dims : str, sequence[str], optional
            The name of the dimension associated with the columns of the
            data. Defaults to None

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
