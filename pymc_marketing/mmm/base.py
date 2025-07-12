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
"""Base class for Marketing Mix Models (MMM)."""

import warnings
from collections.abc import Callable
from inspect import (
    getattr_static,
    isdatadescriptor,
    isgetsetdescriptor,
    ismemberdescriptor,
    ismethoddescriptor,
)
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xarray import DataArray, Dataset

from pymc_marketing.mmm.utils import (
    apply_sklearn_transformer_across_dim,
    transform_1d_array,
)
from pymc_marketing.mmm.validating import (
    ValidateChannelColumns,
    ValidateDateColumn,
    ValidateTargetColumn,
)
from pymc_marketing.model_builder import ModelBuilder

__all__ = ["BaseValidateMMM", "MMMModelBuilder"]

from pydantic import Field, validate_call


class MMMModelBuilder(ModelBuilder):
    """Base class for Marketing Mix Models (MMM)."""

    model: pm.Model
    _model_type = "BaseMMM"
    version = "0.0.2"

    @validate_call
    def __init__(
        self,
        date_column: str = Field(..., description="Column name of the date variable."),
        channel_columns: list[str] = Field(
            min_length=1, description="Column names of the media channel variables."
        ),
        model_config: dict | None = Field(None, description="Model configuration."),
        sampler_config: dict | None = Field(None, description="Sampler configuration."),
    ) -> None:
        self.date_column: str = date_column
        self.channel_columns: list[str] | tuple[str] = channel_columns

        self.n_channel: int = len(channel_columns)

        self.X: pd.DataFrame
        self.y: pd.Series | np.ndarray

        self._time_resolution: int
        self._time_index: NDArray[np.int_]
        self._time_index_mid: int
        self._fit_result: az.InferenceData
        self._posterior_predictive: az.InferenceData
        super().__init__(model_config=model_config, sampler_config=sampler_config)

    @property
    def methods(self) -> list[Any]:
        """Get all methods of the object."""
        maybe_methods = [getattr_static(self, attr) for attr in dir(self)]
        return [
            method
            for method in maybe_methods
            if callable(method)
            and not (
                ismethoddescriptor(method)
                or isdatadescriptor(method)
                or isgetsetdescriptor(method)
                or ismemberdescriptor(method)
            )
        ]

    @property
    def validation_methods(
        self,
    ) -> tuple[
        list[
            Callable[["MMMModelBuilder", pd.DataFrame | pd.Series | np.ndarray], None]
        ],
        list[
            Callable[["MMMModelBuilder", pd.DataFrame | pd.Series | np.ndarray], None]
        ],
    ]:
        """A property that provides validation methods for features ("X") and the target variable ("y").

        This property scans the methods of the object and returns those marked for validation.
        The methods are marked by having a _tags dictionary attribute,with either "validation_X" or "validation_y"
        set to True. The "validation_X" tag indicates a method used for validating features, and "validation_y"
        indicates a method used for validating the target variable.

        Returns
        -------
        tuple of list of Callable[["MMMModelBuilder", pd.DataFrame], None]
            A tuple where the first element is a list of methods for "X" validation, and the second element is
            a list of methods for "y" validation.

        """
        return (
            [
                method
                for method in self.methods
                if getattr(method, "_tags", {}).get("validation_X", False)
            ],
            [
                method
                for method in self.methods
                if getattr(method, "_tags", {}).get("validation_y", False)
            ],
        )

    def validate(
        self, target: str, data: pd.DataFrame | pd.Series | np.ndarray
    ) -> None:
        """Validate the input data based on the specified target type.

        This function loops over the validation methods specified for
        the target type and applies them to the input data.

        Parameters
        ----------
        target : str
            The type of target to be validated.
            Expected values are "X" for features and "y" for the target variable.
        data : Union[pd.DataFrame, pd.Series, np.ndarray]
            The input data to be validated.

        Raises
        ------
        ValueError
            If the target type is not "X" or "y", a ValueError will be raised.

        """
        if target not in ["X", "y"]:
            raise ValueError("Target must be either 'X' or 'y'")
        if target == "X":
            validation_methods = self.validation_methods[0]
        elif target == "y":
            validation_methods = self.validation_methods[1]

        for method in validation_methods:
            method(self, data)

    @property
    def preprocessing_methods(
        self,
    ) -> tuple[
        list[
            Callable[
                ["MMMModelBuilder", pd.DataFrame | pd.Series | np.ndarray],
                pd.DataFrame | pd.Series | np.ndarray,
            ]
        ],
        list[
            Callable[
                ["MMMModelBuilder", pd.DataFrame | pd.Series | np.ndarray],
                pd.DataFrame | pd.Series | np.ndarray,
            ]
        ],
    ]:
        """A property that provides preprocessing methods for features ("X") and the target variable ("y").

        This property scans the methods of the object and returns those marked for preprocessing.
        The methods are marked by having a _tags dictionary attribute, with either "preprocessing_X"
        or "preprocessing_y" set to True. The "preprocessing_X" tag indicates a method used for preprocessing
        features, and "preprocessing_y" indicates a method used for preprocessing the target variable.

        Returns
        -------
        tuple of list of Callable[["MMMModelBuilder", pd.DataFrame], pd.DataFrame]
            A tuple where the first element is a list of methods for "X" preprocessing, and the second element is a
            list of methods for "y" preprocessing.

        """
        return (
            [
                method
                for method in self.methods
                if getattr(method, "_tags", {}).get("preprocessing_X", False)
            ],
            [
                method
                for method in self.methods
                if getattr(method, "_tags", {}).get("preprocessing_y", False)
            ],
        )

    def preprocess(
        self, target: str, data: pd.DataFrame | pd.Series | np.ndarray
    ) -> pd.DataFrame | pd.Series | np.ndarray:
        """Preprocess the provided data according to the specified target.

        This method applies preprocessing methods to the data ("X" or "y"), which are specified in the
        preprocessing_methods property of this object. It iteratively applies each method in the appropriate
        list (either for "X" or "y") to the data.

        Parameters
        ----------
        target : str
            Indicates whether the data represents features ("X") or the target variable ("y").

        data : Union[pd.DataFrame, pd.Series, np.ndarray]
            The data to be preprocessed.

        Returns
        -------
        Union[pd.DataFrame, pd.Series, np.ndarray]
            The preprocessed data.

        Raises
        ------
        ValueError
            If the target is neither "X" nor "y".

        Example
        -------
        >>> data = pd.DataFrame({"x1": [1, 2, 3], "y": [4, 5, 6]})
        >>> self.preprocess("X", data)

        """
        data_cp = data.copy()
        if target == "X":
            for method in self.preprocessing_methods[0]:
                data_cp = method(self, data_cp)
        elif target == "y":
            for method in self.preprocessing_methods[1]:
                data_cp = method(self, data_cp)
        else:
            raise ValueError("Target must be either 'X' or 'y'")
        return data_cp

    def get_target_transformer(self) -> Pipeline:
        """Return the target transformer pipeline used for preprocessing the target variable.

        Returns
        -------
        Pipeline

        """
        try:
            return self.target_transformer  # type: ignore
        except AttributeError:
            identity_transformer = FunctionTransformer()
            return Pipeline(steps=[("scaler", identity_transformer)])

    def _get_group_predictive_data(
        self,
        group: Literal["prior_predictive", "posterior_predictive"],
        original_scale: bool = False,
    ) -> Dataset:
        """Get the prior or posterior predictive data."""
        try:
            group_data: Dataset = getattr(self, group)

        except Exception as e:
            raise RuntimeError(
                f"Make sure the model has been fitted and the {group} has been sampled!"
            ) from e

        if original_scale:
            group_data = apply_sklearn_transformer_across_dim(
                data=group_data,
                func=self.get_target_transformer().inverse_transform,
                dim_name="date",
            )
        return group_data

    def _get_prior_predictive_data(self, original_scale: bool = False) -> Dataset:
        return self._get_group_predictive_data(
            group="prior_predictive", original_scale=original_scale
        )

    def _get_posterior_predictive_data(self, original_scale: bool = False) -> Dataset:
        return self._get_group_predictive_data(
            group="posterior_predictive", original_scale=original_scale
        )

    def _add_mean_to_plot(
        self,
        ax: plt.Axes,
        group: Literal["prior_predictive", "posterior_predictive"],
        original_scale: bool = False,
        color="blue",
        linestyle="-",
        **kwargs,
    ) -> plt.Axes:
        """Add mean prediction to existing plot."""
        group_data: Dataset = self._get_group_predictive_data(
            group=group, original_scale=original_scale
        )

        mean_prediction = group_data[self.output_var].mean(dim=["chain", "draw"])

        ax.plot(
            np.asarray(group_data.date),
            mean_prediction,
            color=color,
            linestyle=linestyle,
            label="Mean Prediction",
        )
        return ax

    def _add_hdi_to_plot(
        self,
        ax: plt.Axes,
        group: Literal["prior_predictive", "posterior_predictive"],
        original_scale: bool = False,
        hdi_prob: float = 0.94,
        color: str = "C0",
        alpha: float = 0.2,
        **kwargs,
    ) -> plt.Axes:
        """Add HDI to existing plot."""
        group_data: Dataset = self._get_group_predictive_data(
            group=group, original_scale=original_scale
        )

        likelihood_hdi: DataArray = az.hdi(ary=group_data, hdi_prob=hdi_prob)[
            self.output_var
        ]

        ax.fill_between(
            x=group_data.date,
            y1=likelihood_hdi[:, 0],
            y2=likelihood_hdi[:, 1],
            color=color,
            alpha=alpha,
            label=f"{hdi_prob:.0%} HDI",
            **kwargs,
        )
        return ax

    def _add_gradient_to_plot(
        self,
        ax: plt.Axes,
        group: Literal["prior_predictive", "posterior_predictive"],
        original_scale: bool = False,
        n_percentiles: int = 30,
        palette: str = "Blues",
        **kwargs,
    ) -> plt.Axes:
        """
        Add a gradient representation of the prior or posterior predictive distribution to an existing plot.

        This method creates a shaded area plot where the color intensity represents
        the density of the posterior predictive distribution.

        Parameters
        ----------
        ax : plt.Axes
            The matplotlib axes object to add the gradient to.
        group : Literal["prior_predictive", "posterior_predictive"]
            The group of data to plot.
        original_scale : bool, optional
            If True, use the original scale of the data. Default is False.
        n_percentiles : int, optional
            Number of percentile ranges to use for the gradient. Default is 30.
        palette : str, optional
            Color palette to use for the gradient. Default is "Blues".
        **kwargs
            Additional keyword arguments passed to ax.fill_between().

        Returns
        -------
        plt.Axes
            The matplotlib axes object with the gradient added.
        """
        # Get posterior predictive data and flatten it
        group_data: Dataset = self._get_group_predictive_data(
            group=group, original_scale=original_scale
        )
        group_data_flattened = group_data.stack(sample=("chain", "draw")).to_dataarray()
        dates = group_data.date.values

        # Set up color map and ranges
        cmap = plt.get_cmap(palette)
        color_range = np.linspace(0.3, 1.0, n_percentiles // 2)
        percentile_ranges = np.linspace(3, 97, n_percentiles)

        # Create gradient by filling between percentile ranges
        for i in range(len(percentile_ranges) - 1):
            lower_percentile = np.percentile(
                group_data_flattened, percentile_ranges[i], axis=2
            ).squeeze()
            upper_percentile = np.percentile(
                group_data_flattened, percentile_ranges[i + 1], axis=2
            ).squeeze()
            if i < n_percentiles // 2:
                color_val = color_range[i]
            else:
                color_val = color_range[n_percentiles - i - 2]
            alpha_val = 0.2 + 0.8 * (
                1 - abs(2 * i / n_percentiles - 1)
            )  # Higher alpha in the middle
            ax.fill_between(
                x=dates,
                y1=lower_percentile,
                y2=upper_percentile,
                color=cmap(color_val),
                alpha=alpha_val,
                **kwargs,
            )

        return ax

    def _plot_group_predictive(
        self,
        group: Literal["prior_predictive", "posterior_predictive"],
        original_scale: bool = False,
        hdi_list: list[float] | None = None,
        add_mean: bool = True,
        add_gradient: bool = False,
        ax: plt.Axes = None,
        **plt_kwargs: Any,
    ) -> plt.Figure:
        """
        Plot the prior or posterior predictive distribution from the model fit.

        This function creates a visualization of the model's prior or posterior predictive distribution,
        allowing for comparison with observed data. It can include highest density intervals (HDI),
        mean predictions, and a gradient representation of the full distribution.

        Parameters
        ----------
        group : Literal["prior_predictive", "posterior_predictive"]
            The group of data to plot.
        original_scale : bool, optional
            If True, plot in the original scale of the target variable.
            If False, plot in the transformed scale used for modeling. Default is False.
        hdi_list : list of float, optional
            List of HDI levels to plot. Default is [0.94] Provide an empty list to omit plotting the HDI.
        add_mean : bool, optional
            If True, add the mean prediction to the plot. Default is True.
        add_gradient : bool, optional
            If True, add a gradient representation of the full posterior distribution. Default is False.
        ax : plt.Axes, optional
            A matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        **plt_kwargs : dict
            Additional keyword arguments to pass to plt.subplots() when creating a new figure.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If the length of the target variable doesn't match the length
            of the date column in the posterior predictive data.

        Notes
        -----
        This function visualizes the model's predictions against the observed data.
        The observed data is always plotted as a black line.
        Depending on the parameters, it can also show:
        - HDI (Highest Density Intervals) at 94% and 50% levels
        - Mean prediction line
        - Gradient representation of the full posterior distribution

        If predicting out-of-sample, ensure that `self.y` is overwritten with the
        corresponding non-transformed target variable.
        """
        group_data: Dataset = self._get_group_predictive_data(
            group=group, original_scale=original_scale
        )

        target_to_plot = np.asarray(
            self.y
            if original_scale
            else transform_1d_array(self.get_target_transformer().transform, self.y)
        )

        if len(target_to_plot) != len(group_data.date):
            raise ValueError(
                "The length of the target variable doesn't match the length of the date column. "
                "If you are predicting out-of-sample, please overwrite `self.y` with the "
                "corresponding (non-transformed) target variable."
            )

        if ax is None:
            fig, ax = plt.subplots(**plt_kwargs)
        else:
            fig = ax.figure

        if hdi_list is None:
            hdi_list = [0.94, 0.5]

        if hdi_list and not add_gradient:
            alpha_list = np.linspace(0.2, 0.4, len(hdi_list), dtype=float)
            for hdi_prob, alpha in zip(hdi_list, alpha_list, strict=True):
                ax = self._add_hdi_to_plot(
                    ax=ax,
                    group=group,
                    original_scale=original_scale,
                    hdi_prob=hdi_prob,
                    alpha=alpha,
                )

        if add_mean:
            ax = self._add_mean_to_plot(
                ax=ax, group=group, original_scale=original_scale, color="blue"
            )

        if add_gradient:
            ax = self._add_gradient_to_plot(
                ax=ax,
                group=group,
                original_scale=original_scale,
                n_percentiles=30,
                palette="Blues",
            )

        ax.plot(
            np.asarray(group_data.date),
            target_to_plot,
            color="black",
            label="Observed",
        )
        ax.legend()
        ax.set(
            title=f"{group} predictive check",
            xlabel="date",
            ylabel=self.output_var,
        )

        return fig

    def plot_prior_predictive(
        self,
        original_scale: bool = False,
        hdi_list: list[float] | None = None,
        add_mean: bool = True,
        add_gradient: bool = False,
        ax: plt.Axes = None,
        **plt_kwargs: Any,
    ) -> plt.Figure:
        """
        Plot the prior predictive distribution from the model fit.

        This function creates a visualization of the model's prior predictive distribution,
        allowing for comparison with observed data. It can include highest density intervals (HDI),
        mean predictions, and a gradient representation of the full distribution.

        Parameters
        ----------
        original_scale : bool, optional
            If True, plot in the original scale of the target variable.
            If False, plot in the transformed scale used for modeling. Default is False.
        hdi_list : list of float, optional
            List of HDI levels to plot. Default is [0.94] Provide an empty list to omit plotting the HDI.
        add_mean : bool, optional
            If True, add the mean prediction to the plot. Default is True.
        add_gradient : bool, optional
            If True, add a gradient representation of the full posterior distribution. Default is False.
        ax : plt.Axes, optional
            A matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        **plt_kwargs : dict
            Additional keyword arguments to pass to plt.subplots() when creating a new figure.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If the length of the target variable doesn't match the length
            of the date column in the posterior predictive data.

        Notes
        -----
        This function visualizes the model's predictions against the observed data.
        The observed data is always plotted as a black line.
        Depending on the parameters, it can also show:
        - HDI (Highest Density Intervals) at 94% and 50% levels
        - Mean prediction line
        - Gradient representation of the full posterior distribution
        """
        return self._plot_group_predictive(
            group="prior_predictive",
            original_scale=original_scale,
            hdi_list=hdi_list,
            add_mean=add_mean,
            add_gradient=add_gradient,
            ax=ax,
            **plt_kwargs,
        )

    def plot_posterior_predictive(
        self,
        original_scale: bool = False,
        hdi_list: list[float] | None = None,
        add_mean: bool = True,
        add_gradient: bool = False,
        ax: plt.Axes = None,
        **plt_kwargs: Any,
    ) -> plt.Figure:
        """
        Plot the posterior predictive distribution from the model fit.

        This function creates a visualization of the model's posterior predictive distribution,
        allowing for comparison with observed data. It can include highest density intervals (HDI),
        mean predictions, and a gradient representation of the full distribution.

        Parameters
        ----------
        original_scale : bool, optional
            If True, plot in the original scale of the target variable.
            If False, plot in the transformed scale used for modeling. Default is False.
        hdi_list : list of float, optional
            List of HDI levels to plot. Default is [0.94] Provide an empty list to omit plotting the HDI.
        add_mean : bool, optional
            If True, add the mean prediction to the plot. Default is True.
        add_gradient : bool, optional
            If True, add a gradient representation of the full posterior distribution. Default is False.
        ax : plt.Axes, optional
            A matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        **plt_kwargs : dict
            Additional keyword arguments to pass to plt.subplots() when creating a new figure.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If the length of the target variable doesn't match the length
            of the date column in the posterior predictive data.

        Notes
        -----
        This function visualizes the model's predictions against the observed data.
        The observed data is always plotted as a black line.
        Depending on the parameters, it can also show:
        - HDI (Highest Density Intervals) at 94% and 50% levels
        - Mean prediction line
        - Gradient representation of the full posterior distribution

        If predicting out-of-sample, ensure that `self.y` is overwritten with the
        corresponding non-transformed target variable.
        """
        return self._plot_group_predictive(
            group="posterior_predictive",
            original_scale=original_scale,
            hdi_list=hdi_list,
            add_mean=add_mean,
            add_gradient=add_gradient,
            ax=ax,
            **plt_kwargs,
        )

    def get_errors(self, original_scale: bool = False) -> DataArray:
        """Get model errors posterior distribution.

        errors = true values - predicted

        Parameters
        ----------
        original_scale : bool, optional
            Whether to plot in the original scale.

        Returns
        -------
        DataArray

        """
        try:
            posterior_predictive_data: Dataset = self.posterior_predictive

        except Exception as e:
            raise RuntimeError(
                "Make sure the model has been fitted and the posterior_predictive has been sampled!"
            ) from e

        target_array = np.asarray(
            transform_1d_array(self.get_target_transformer().transform, self.y)
        )

        if len(target_array) != len(posterior_predictive_data.date):
            raise ValueError(
                "The length of the target variable doesn't match the length of the date column. "
                "If you are computing out-of-sample errors, please overwrite `self.y` with the "
                "corresponding (non-transformed) target variable."
            )

        target = (
            pd.Series(target_array, index=self.posterior_predictive.date)
            .rename_axis("date")
            .to_xarray()
        )

        errors = (
            (target - posterior_predictive_data)[self.output_var]
            .rename("errors")
            .transpose(..., "date")
        )

        if original_scale:
            return apply_sklearn_transformer_across_dim(
                data=errors,
                func=self.get_target_transformer().inverse_transform,
                dim_name="date",
            )

        return errors

    def plot_errors(
        self, original_scale: bool = False, ax: plt.Axes = None, **plt_kwargs: Any
    ) -> plt.Figure:
        """Plot model errors by taking the difference between true values and predicted.

        errors = true values - predicted

        Parameters
        ----------
        original_scale : bool, optional
            Whether to plot in the original scale.
        ax : plt.Axes, optional
            Matplotlib axis object.
        **plt_kwargs
            Keyword arguments passed to `plt.subplots`.

        Returns
        -------
        plt.Figure

        """
        errors = self.get_errors(original_scale=original_scale)

        if ax is None:
            fig, ax = plt.subplots(**plt_kwargs)
        else:
            fig = ax.figure

        for hdi_prob, alpha in zip((0.94, 0.50), (0.2, 0.4), strict=True):
            errors_hdi = az.hdi(ary=errors, hdi_prob=hdi_prob)

            ax.fill_between(
                x=self.posterior_predictive.date,
                y1=errors_hdi["errors"].sel(hdi="lower"),
                y2=errors_hdi["errors"].sel(hdi="higher"),
                color="C3",
                alpha=alpha,
                label=f"${100 * hdi_prob}\\%$ HDI",
            )

        ax.plot(
            self.posterior_predictive.date,
            errors.mean(dim=("chain", "draw")).to_numpy(),
            color="C3",
            label="Errors Mean",
        )

        ax.axhline(y=0.0, linestyle="--", color="black", label="zero")
        ax.legend()
        ax.set(
            title="Errors Posterior Distribution",
            xlabel="date",
            ylabel="true - predictions",
        )
        return fig

    def _format_model_contributions(self, var_contribution: str) -> DataArray:
        contributions = az.extract(
            self.fit_result,
            var_names=[var_contribution],
            combined=False,
        )
        contracted_dims = [
            d for d in contributions.dims if d not in ["chain", "draw", "date"]
        ]
        return contributions.sum(contracted_dims) if contracted_dims else contributions

    def plot_components_contributions(self, **plt_kwargs: Any) -> plt.Figure:
        """Plot the target variable and the posterior predictive model components.

        We can plot the target variable and the posterior predictive model components in
        the scaled space or the original space.

        **plt_kwargs
            Additional keyword arguments to pass to `plt.subplots`.

        Returns
        -------
        plt.Figure

        """
        channel_contribution = self._format_model_contributions(
            var_contribution="channel_contribution"
        )
        means = [channel_contribution.mean(["chain", "draw"])]
        contribution_vars = [
            az.hdi(channel_contribution, hdi_prob=0.94).channel_contribution
        ]

        for arg, var_contribution in zip(
            ["control_columns", "yearly_seasonality"],
            ["control_contribution", "fourier_contribution"],
            strict=True,
        ):
            if getattr(self, arg, None):
                contributions = self._format_model_contributions(
                    var_contribution=var_contribution
                )
                means.append(contributions.mean(["chain", "draw"]))
                contribution_vars.append(
                    az.hdi(contributions, hdi_prob=0.94)[var_contribution]
                )

        fig, ax = plt.subplots(**plt_kwargs)

        for i, (mean, hdi, var_contribution) in enumerate(
            zip(
                means,
                contribution_vars,
                [
                    "channel_contribution",
                    "control_contribution",
                    "fourier_contribution",
                ],
                strict=False,
            )
        ):
            if self.X is not None:
                ax.fill_between(
                    x=self.X[self.date_column],
                    y1=hdi.isel(hdi=0),
                    y2=hdi.isel(hdi=1),
                    color=f"C{i}",
                    alpha=0.25,
                    label=f"$94\\%$ HDI ({var_contribution})",
                )
                ax.plot(
                    np.asarray(self.X[self.date_column]),
                    np.asarray(mean),
                    color=f"C{i}",
                )
        if self.X is not None:
            intercept = az.extract(
                self.fit_result, var_names=["intercept"], combined=False
            )

            if intercept.ndim == 2:
                # Intercept has a stationary prior
                intercept_hdi = np.repeat(
                    a=az.hdi(intercept).intercept.data[None, ...],
                    repeats=self.X[self.date_column].shape[0],
                    axis=0,
                )
            elif intercept.ndim == 3:
                # Intercept has a time-varying prior
                intercept_hdi = az.hdi(intercept).intercept.data

            ax.plot(
                np.asarray(self.X[self.date_column]),
                np.full(len(self.X[self.date_column]), intercept.mean().data),
                color=f"C{i + 1}",
            )
            ax.fill_between(
                x=self.X[self.date_column],
                y1=intercept_hdi[:, 0],
                y2=intercept_hdi[:, 1],
                color=f"C{i + 1}",
                alpha=0.25,
                label="$94\\%$ HDI (intercept)",
            )
            ax.plot(
                np.asarray(self.X[self.date_column]),
                np.asarray(self.preprocessed_data["y"]),  # type: ignore
                label="scaled target",
                color="black",
            )
            ax.legend(title="components", loc="center left", bbox_to_anchor=(1, 0.5))
            ax.set(
                title="Posterior Predictive Model Components",
                xlabel="date",
                ylabel=self.output_var,
            )
        return fig

    def compute_channel_contribution_original_scale(
        self, prior: bool = False
    ) -> DataArray:
        """Compute the channel contributions in the original scale of the target variable.

        Parameters
        ----------
        prior : bool, optional
            Whether to use the prior or posterior, by default False (posterior)

        Returns
        -------
        DataArray

        """
        _data = self.prior if prior else self.fit_result
        channel_contribution = az.extract(
            data=_data, var_names=["channel_contribution"], combined=False
        )

        # sklearn preprocessers expect 2-D arrays of (obs, features)
        # We need to treat all entries of channel_contribution as independent obs
        # so we flatten it, then apply the transform, and finally reshape back into its
        # original form
        return DataArray(
            np.reshape(
                self.get_target_transformer().inverse_transform(
                    channel_contribution.data.flatten()[:, None]
                ),
                channel_contribution.shape,
            ),
            dims=channel_contribution.dims,
            coords=channel_contribution.coords,
        )

    def compute_mean_contributions_over_time(
        self, original_scale: bool = False
    ) -> pd.DataFrame:
        """Get the contributions of each channel over time.

        Parameters
        ----------
        original_scale : bool, optional
            Whether to return the contributions in the original scale of the target
            variable. If False, the contributions are returned in the scale of the
            transformed target variable. Defaults to False.

        Returns
        -------
        pd.DataFrame
            A dataframe with the mean contributions of each channel and control variables over time.

        """
        contributions_channel_over_time = (
            az.extract(
                self.fit_result,
                var_names=["channel_contribution"],
                combined=True,
            )
            .mean("sample")
            .to_dataframe()
            .squeeze()
            .unstack()
        )
        contributions_channel_over_time.columns = self.channel_columns

        if getattr(self, "control_columns", None):
            contributions_control_over_time = (
                az.extract(
                    self.fit_result,
                    var_names=["control_contribution"],
                    combined=True,
                )
                .mean("sample")
                .to_dataframe()
                .squeeze()
                .unstack()
            )
        else:
            contributions_control_over_time = pd.DataFrame(
                index=contributions_channel_over_time.index
            )

        if getattr(self, "yearly_seasonality", None):
            contributions_fourier_over_time = pd.DataFrame(
                az.extract(
                    self.fit_result,
                    var_names=["fourier_contribution"],
                    combined=True,
                )
                .mean("sample")
                .to_dataframe()
                .squeeze()
                .unstack()
                .sum(axis=1),
                columns=["yearly_seasonality"],
            )
        else:
            contributions_fourier_over_time = pd.DataFrame(
                index=contributions_channel_over_time.index
            )

        contributions_intercept_over_time = (
            az.extract(
                self.fit_result,
                var_names=["intercept"],
                combined=True,
            )
            .mean("sample")
            .to_numpy()
        )

        all_contributions_over_time = (
            contributions_channel_over_time.join(contributions_control_over_time)
            .join(contributions_fourier_over_time)
            .assign(intercept=contributions_intercept_over_time)
        )

        if original_scale:
            all_contributions_over_time = pd.DataFrame(
                data=self.get_target_transformer().inverse_transform(
                    all_contributions_over_time
                ),
                columns=all_contributions_over_time.columns,
                index=all_contributions_over_time.index,
            )
            all_contributions_over_time.columns = (
                all_contributions_over_time.columns.map(
                    lambda x: f"channel_{x}" if isinstance(x, int) else x
                )
            )
        return all_contributions_over_time

    def plot_grouped_contribution_breakdown_over_time(
        self,
        stack_groups: dict[str, list[str]] | None = None,
        original_scale: bool = False,
        area_kwargs: dict[str, Any] | None = None,
        **plt_kwargs: Any,
    ) -> plt.Figure:
        """Plot a time series area chart for all channel contributions.

        Since a chart like this can become quite crowded if you have many channels or
        control variables, you can group certain variables together using the
        `stack_groups` keyword.

        Parameters
        ----------
        stack_groups : dict of {str: list of str}, optional
            Specifies which variables to group together.
            Example: passing
                {
                    "Baseline": ["intercept"],
                    "Offline": ["TV", "Radio"],
                    "Online": ["Banners"]
                }
            results in a chart with three colors, one for Baseline, one for Online,
            and one for Offline. If `stack_groups` is None, the chart would have four
            colors since TV and Radio would be separated.

            Note: If you only pass {"Baseline": "intercept", "Online": ["Banners"]},
            you will not see the TV and Radio channels in the chart.
        original_scale : bool, by default False
            If True, the contributions are plotted in the original scale of the target.

        Returns
        -------
        plt.Figure
            Matplotlib figure with the plot.

        """
        all_contributions_over_time = self.compute_mean_contributions_over_time(
            original_scale=original_scale
        )

        if stack_groups is not None:
            grouped_buffer = []
            for group, columns in stack_groups.items():
                grouped = (
                    all_contributions_over_time.filter(columns)
                    .sum(axis="columns")
                    .rename(group)
                )
                grouped_buffer.append(grouped)

            all_contributions_over_time = pd.concat(grouped_buffer, axis="columns")

        fig, ax = plt.subplots(**plt_kwargs)
        area_params = dict(stacked=True, ax=ax)
        if area_kwargs is not None:
            area_params.update(area_kwargs)
        try:
            all_contributions_over_time.plot.area(**area_params)
        except ValueError:
            warnings.warn(
                """
                Each contribution value must be either all positive or all negative.
                Try deselecting variables with negative contributions.
                """,
                stacklevel=2,
            )
            return fig
        ax.legend(title="groups", loc="center left", bbox_to_anchor=(1, 0.5))
        return fig

    def get_channel_contribution_share_samples(self, prior: bool = False) -> DataArray:
        """Get the share of channel contributions in the original scale of the target variable.

        Parameters
        ----------
        prior : bool, optional
            Whether to use the prior or posterior, by default False (posterior)

        Returns
        -------
        DataArray
            The share of channel contributions in the original scale of the target variable.

        """
        channel_contribution_original_scale_samples: DataArray = (
            self.compute_channel_contribution_original_scale(prior=prior)
        )
        numerator: DataArray = channel_contribution_original_scale_samples.sum(["date"])
        denominator: DataArray = numerator.sum("channel")
        return numerator / denominator

    def plot_channel_contribution_share_hdi(
        self, hdi_prob: float = 0.94, prior: bool = False, **plot_kwargs: Any
    ) -> plt.Figure:
        """Plot the share of channel contributions in a forest plot.

        Parameters
        ----------
        hdi_prob : float, optional
            HDI value to be displayed, by default 0.94
        prior : bool, optional
            Whether to use the prior or posterior, by default False (posterior)
        **plot_kwargs
            Additional keyword arguments to pass to `az.plot_forest`.

        Returns
        -------
        plt.Figure

        """
        channel_contribution_share: DataArray = (
            self.get_channel_contribution_share_samples(prior=prior)
        )

        ax, *_ = az.plot_forest(
            data=channel_contribution_share,
            combined=True,
            hdi_prob=hdi_prob,
            **plot_kwargs,
        )
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y: 0.0%}"))
        fig: plt.Figure = plt.gcf()
        fig.suptitle("channel Contribution Share", fontsize=16, y=1.05)
        return fig

    def _process_decomposition_components(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data to compute the sum of contributions by component and calculate their percentages.

        The output dataframe will have columns for "component", "contribution", and "percentage".

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing the contribution by component from the function "compute_mean_contributions_over_time".

        Returns
        -------
        pd.DataFrame
            A dataframe with contributions summed up by component, sorted by contribution in ascending order.
            With an additional column showing the percentage contribution of each component.

        """
        dataframe = data.copy()
        stack_dataframe = dataframe.stack().reset_index()
        stack_dataframe.columns = pd.Index(["date", "component", "contribution"])
        stack_dataframe.set_index(["date", "component"], inplace=True)
        dataframe = stack_dataframe.groupby("component").sum()
        dataframe.sort_values(by="contribution", ascending=True, inplace=True)
        dataframe.reset_index(inplace=True)

        total_contribution = dataframe["contribution"].sum()
        dataframe["percentage"] = (dataframe["contribution"] / total_contribution) * 100

        return dataframe

    def plot_prior_vs_posterior(
        self,
        var_name: str,
        alphabetical_sort: bool = True,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot the prior vs posterior distribution for a specified variable in a 3 columngrid layout.

        This function generates KDE plots for each MMM channel, showing the prior predictive
        and posterior distributions with their respective means highlighted.
        It sorts the plots either alphabetically or based on the difference between the
        posterior and prior means, with the largest difference (posterior - prior) at the top.

        Parameters
        ----------
        var_name: str
            The variable to analyze (e.g., 'adstock_alpha').
        alphabetical_sort: bool, optional
            Whether to sort the channels alphabetically (True) or by the difference
            between the posterior and prior means (False). Default is True.
        figsize : tuple of int, optional
            Figure size in inches. If None, it will be calculated based on the number of channels.

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure object

        Raises
        ------
        ValueError
            If the required attributes (prior, posterior) were not found.
        ValueError
            If var_name is not a string.
        """
        if not hasattr(self, "fit_result") or not hasattr(self, "prior"):
            raise ValueError(
                "Required attributes (fit_result, prior) not found. "
                "Ensure you've called model.fit() and model.sample_prior_predictive()"
            )

        if not isinstance(var_name, str):
            raise ValueError(
                "var_name must be a string. Please provide a single variable name."
            )

        # Determine the number of channels and set up the grid
        num_channels = len(self.channel_columns)
        num_cols = 1
        num_rows = num_channels

        if figsize is None:
            figsize = (12, 4 * num_rows)

        # Calculate prior and posterior means for sorting
        channel_means = []
        for channel in self.channel_columns:
            prior_mean = self.prior[var_name].sel(channel=channel).mean().values
            posterior_mean = (
                self.fit_result[var_name].sel(channel=channel).mean().values
            )
            difference = posterior_mean - prior_mean
            channel_means.append((channel, prior_mean, posterior_mean, difference))

        # Choose how to sort the channels
        if alphabetical_sort:
            sorted_channels = sorted(channel_means, key=lambda x: x[0])
        else:
            # Otherwise, sort on difference between posterior and prior means
            sorted_channels = sorted(channel_means, key=lambda x: x[3], reverse=True)

        fig, axs = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=figsize,
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        axs = axs.flatten()  # Flatten the array for easy iteration

        # Plot for each channel
        for i, (channel, prior_mean, posterior_mean, difference) in enumerate(
            sorted_channels
        ):
            # Extract prior samples for the current channel
            prior_samples = self.prior[var_name].sel(channel=channel).values.flatten()

            # Plot the prior predictive distribution
            sns.kdeplot(
                prior_samples,
                ax=axs[i],
                label="Prior Predictive",
                color="C0",
                fill=True,
            )

            # Add a vertical line for the mean of the prior distribution
            axs[i].axvline(
                prior_mean,
                color="C0",
                linestyle="--",
                linewidth=2,
                label=f"Prior Mean: {prior_mean:.2f}",
            )

            # Extract posterior samples for the current channel
            posterior_samples = (
                self.fit_result[var_name].sel(channel=channel).values.flatten()
            )

            # Plot the prior predictive distribution
            sns.kdeplot(
                posterior_samples,
                ax=axs[i],
                label="Posterior Predictive",
                color="C1",
                fill=True,
                alpha=0.15,
            )

            # Add a vertical line for the mean of the posterior distribution
            axs[i].axvline(
                posterior_mean,
                color="C1",
                linestyle="--",
                linewidth=2,
                label=f"Posterior Mean: {posterior_mean:.2f} (Diff: {difference:.2f})",
            )

            # Set titles and labels
            axs[i].set_title(channel)  # Subplot title is just the channel name
            axs[i].set_xlabel(var_name.capitalize())
            axs[i].set_ylabel("Density")
            axs[i].legend(loc="upper right")

        # Set the overall figure title
        fig.suptitle(
            f"Prior vs Posterior Distributions | {var_name}",
            fontsize=16,
            horizontalalignment="center",
        )

        return fig

    def plot_waterfall_components_decomposition(
        self,
        original_scale: bool = True,
        figsize: tuple[int, int] = (14, 7),
        **kwargs,
    ) -> plt.Figure:
        """Create a waterfall plot.

        The plot shows the decomposition of the target into its components.

        Parameters
        ----------
        original_scale : bool, optional
            If True, the contributions are plotted in the original scale of the target.
        figsize : tuple[int, int], optional
            The size of the figure. The default is (14, 7).
        **kwargs
            Additional keyword arguments to pass to the matplotlib `subplots` function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.

        """
        dataframe = self.compute_mean_contributions_over_time(
            original_scale=original_scale
        )

        dataframe = self._process_decomposition_components(data=dataframe)
        total_contribution = dataframe["contribution"].sum()

        fig, ax = plt.subplots(figsize=figsize, layout="constrained", **kwargs)

        cumulative_contribution = 0

        for index, row in dataframe.iterrows():
            color = "C0" if row["contribution"] >= 0 else "C3"

            bar_start = (
                cumulative_contribution + row["contribution"]
                if row["contribution"] < 0
                else cumulative_contribution
            )
            ax.barh(
                row["component"],
                row["contribution"],
                left=bar_start,
                color=color,
                alpha=0.5,
            )

            if row["contribution"] > 0:
                cumulative_contribution += row["contribution"]

            label_pos = bar_start + (row["contribution"] / 2)

            if row["contribution"] < 0:
                label_pos = bar_start - (row["contribution"] / 2)

            ax.text(
                label_pos,
                index,
                f"{row['contribution']:,.0f}\n({row['percentage']:.1f}%)",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

        ax.set_title("Response Decomposition Waterfall by Components")
        ax.set_xlabel("Cumulative Contribution")
        ax.set_ylabel("Components")

        xticks = np.linspace(0, total_contribution, num=11)
        xticklabels = [f"{(x / total_contribution) * 100:.0f}%" for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_yticks(np.arange(len(dataframe)))
        ax.set_yticklabels(dataframe["component"])

        return fig


class BaseValidateMMM(
    MMMModelBuilder,
    ValidateTargetColumn,
    ValidateDateColumn,
    ValidateChannelColumns,
):
    """Base class with some validation of the inputs."""
