"""Base class for Marketing Mix Models (MMM)."""

import warnings
from inspect import (
    getattr_static,
    isdatadescriptor,
    isgetsetdescriptor,
    ismemberdescriptor,
    ismethoddescriptor,
)
from itertools import repeat
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xarray import DataArray, Dataset

from pymc_marketing.mmm.budget_optimizer import budget_allocator
from pymc_marketing.mmm.transformers import michaelis_menten
from pymc_marketing.mmm.utils import (
    estimate_menten_parameters,
    estimate_sigmoid_parameters,
    find_sigmoid_inflection_point,
    sigmoid_saturation,
    standardize_scenarios_dict_keys,
)
from pymc_marketing.mmm.validating import (
    ValidateChannelColumns,
    ValidateDateColumn,
    ValidateTargetColumn,
)
from pymc_marketing.model_builder import ModelBuilder

__all__ = ("BaseMMM", "MMM")


class BaseMMM(ModelBuilder):
    model: pm.Model
    _model_type = "BaseMMM"
    version = "0.0.2"

    def __init__(
        self,
        date_column: str,
        channel_columns: Union[List[str], Tuple[str]],
        model_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[Union[pd.Series, np.ndarray]] = None
        self.date_column: str = date_column
        self.channel_columns: Union[List[str], Tuple[str]] = channel_columns
        self.n_channel: int = len(channel_columns)
        self._fit_result: Optional[az.InferenceData] = None
        self._posterior_predictive: Optional[az.InferenceData] = None
        super().__init__(model_config=model_config, sampler_config=sampler_config)

    @property
    def methods(self) -> List[Any]:
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
    ) -> Tuple[
        List[Callable[["BaseMMM", Union[pd.DataFrame, pd.Series, np.ndarray]], None]],
        List[Callable[["BaseMMM", Union[pd.DataFrame, pd.Series, np.ndarray]], None]],
    ]:
        """
        A property that provides validation methods for features ("X") and the target variable ("y").

        This property scans the methods of the object and returns those marked for validation.
        The methods are marked by having a _tags dictionary attribute, with either "validation_X" or "validation_y" set to True.
        The "validation_X" tag indicates a method used for validating features, and "validation_y" indicates a method used for validating the target variable.

        Returns
        -------
        tuple of list of Callable[["BaseMMM", pd.DataFrame], None]
            A tuple where the first element is a list of methods for "X" validation, and the second element is a list of methods for "y" validation.

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
        self, target: str, data: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> None:
        """
        Validates the input data based on the specified target type.

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
    ) -> Tuple[
        List[
            Callable[
                ["BaseMMM", Union[pd.DataFrame, pd.Series, np.ndarray]],
                Union[pd.DataFrame, pd.Series, np.ndarray],
            ]
        ],
        List[
            Callable[
                ["BaseMMM", Union[pd.DataFrame, pd.Series, np.ndarray]],
                Union[pd.DataFrame, pd.Series, np.ndarray],
            ]
        ],
    ]:
        """
        A property that provides preprocessing methods for features ("X") and the target variable ("y").

        This property scans the methods of the object and returns those marked for preprocessing.
        The methods are marked by having a _tags dictionary attribute, with either "preprocessing_X" or "preprocessing_y" set to True.
        The "preprocessing_X" tag indicates a method used for preprocessing features, and "preprocessing_y" indicates a method used for preprocessing the target variable.

        Returns
        -------
        tuple of list of Callable[["BaseMMM", pd.DataFrame], pd.DataFrame]
            A tuple where the first element is a list of methods for "X" preprocessing, and the second element is a list of methods for "y" preprocessing.
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
        self, target: str, data: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Preprocess the provided data according to the specified target.

        This method applies preprocessing methods to the data ("X" or "y"), which are specified in the preprocessing_methods property of this object.
        It iteratively applies each method in the appropriate list (either for "X" or "y") to the data.

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
        try:
            return self.target_transformer  # type: ignore
        except AttributeError:
            identity_transformer = FunctionTransformer()
            return Pipeline(steps=[("scaler", identity_transformer)])

    @property
    def prior(self) -> Dataset:
        if self.idata is None or "prior" not in self.idata:
            raise RuntimeError(
                "The model hasn't been fit yet, call .sample_prior_predictive() with extend_idata=True first"
            )
        return self.idata["prior"]

    @property
    def prior_predictive(self) -> az.InferenceData:
        if self.idata is None or "prior_predictive" not in self.idata:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
        return self.idata["prior_predictive"]

    @property
    def fit_result(self) -> Dataset:
        if self.idata is None or "posterior" not in self.idata:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
        return self.idata["posterior"]

    @property
    def posterior_predictive(self) -> Dataset:
        if self.idata is None or "posterior_predictive" not in self.idata:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
        return self.idata["posterior_predictive"]

    def plot_prior_predictive(
        self, samples: int = 1_000, **plt_kwargs: Any
    ) -> plt.Figure:
        prior_predictive_data: az.InferenceData = self.prior_predictive

        likelihood_hdi_94: DataArray = az.hdi(ary=prior_predictive_data, hdi_prob=0.94)[
            self.output_var
        ]
        likelihood_hdi_50: DataArray = az.hdi(ary=prior_predictive_data, hdi_prob=0.50)[
            self.output_var
        ]

        fig, ax = plt.subplots(**plt_kwargs)
        if self.X is not None and self.y is not None:
            ax.fill_between(
                x=np.asarray(self.X[self.date_column]),
                y1=likelihood_hdi_94[:, 0],
                y2=likelihood_hdi_94[:, 1],
                color="C0",
                alpha=0.2,
                label=r"$94\%$ HDI",
            )

            ax.fill_between(
                x=np.asarray(self.X[self.date_column]),
                y1=likelihood_hdi_50[:, 0],
                y2=likelihood_hdi_50[:, 1],
                color="C0",
                alpha=0.3,
                label=r"$50\%$ HDI",
            )

            ax.plot(
                np.asarray(self.X[self.date_column]),
                np.asarray(self.preprocessed_data["y"]),  # type: ignore
                color="black",
            )
            ax.set(
                title="Prior Predictive Check", xlabel="date", ylabel=self.output_var
            )
        else:
            raise RuntimeError(
                "The model hasn't been fit yet, call .fit() first with X and y data."
            )
        return fig

    def plot_posterior_predictive(
        self, original_scale: bool = False, **plt_kwargs: Any
    ) -> plt.Figure:
        posterior_predictive_data: Dataset = self.posterior_predictive
        likelihood_hdi_94: DataArray = az.hdi(
            ary=posterior_predictive_data, hdi_prob=0.94
        )[self.output_var]
        likelihood_hdi_50: DataArray = az.hdi(
            ary=posterior_predictive_data, hdi_prob=0.50
        )[self.output_var]

        if original_scale:
            likelihood_hdi_94 = self.get_target_transformer().inverse_transform(
                Xt=likelihood_hdi_94
            )
            likelihood_hdi_50 = self.get_target_transformer().inverse_transform(
                Xt=likelihood_hdi_50
            )

        fig, ax = plt.subplots(**plt_kwargs)
        if self.X is not None and self.y is not None:
            ax.fill_between(
                x=self.X[self.date_column],
                y1=likelihood_hdi_94[:, 0],
                y2=likelihood_hdi_94[:, 1],
                color="C0",
                alpha=0.2,
                label="$94\%$ HDI",  # noqa: W605
            )

            ax.fill_between(
                x=self.X[self.date_column],
                y1=likelihood_hdi_50[:, 0],
                y2=likelihood_hdi_50[:, 1],
                color="C0",
                alpha=0.3,
                label="$50\%$ HDI",  # noqa: W605
            )

            target_to_plot: np.ndarray = np.asarray(
                self.y if original_scale else self.preprocessed_data["y"]  # type: ignore
            )
            ax.plot(
                np.asarray(self.X[self.date_column]),
                target_to_plot,
                color="black",
            )
            ax.set(
                title="Posterior Predictive Check",
                xlabel="date",
                ylabel=self.output_var,
            )
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
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
        channel_contributions = self._format_model_contributions(
            var_contribution="channel_contributions"
        )
        means = [channel_contributions.mean(["chain", "draw"])]
        contribution_vars = [
            az.hdi(channel_contributions, hdi_prob=0.94).channel_contributions
        ]

        for arg, var_contribution in zip(
            ["control_columns", "yearly_seasonality"],
            ["control_contributions", "fourier_contributions"],
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
            )
        ):
            if self.X is not None:
                ax.fill_between(
                    x=self.X[self.date_column],
                    y1=hdi.isel(hdi=0),
                    y2=hdi.isel(hdi=1),
                    color=f"C{i}",
                    alpha=0.25,
                    label=f"$94\%$ HDI ({var_contribution})",  # noqa: W605
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
            intercept_hdi = np.repeat(
                a=az.hdi(intercept).intercept.data[None, ...],
                repeats=self.X[self.date_column].shape[0],
                axis=0,
            )
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
                label="$94\%$ HDI (intercept)",  # noqa: W605
            )
            ax.plot(
                np.asarray(self.X[self.date_column]),
                np.asarray(self.preprocessed_data["y"]),  # type: ignore
                color="black",
            )
            ax.legend(title="components", loc="center left", bbox_to_anchor=(1, 0.5))
            ax.set(
                title="Posterior Predictive Model Components",
                xlabel="date",
                ylabel=self.output_var,
            )
        return fig

    def plot_channel_parameter(self, param_name: str, **plt_kwargs: Any) -> plt.Figure:
        if param_name not in ["alpha", "lam", "beta_channel"]:
            raise ValueError(f"Invalid parameter name: {param_name}")

        param_samples_df = pd.DataFrame(
            data=az.extract(data=self.fit_result, var_names=[param_name]).T,
            columns=self.channel_columns,
        )

        fig, ax = plt.subplots(**plt_kwargs)
        sns.violinplot(data=param_samples_df, orient="h", ax=ax)
        ax.set(
            title=f"Posterior Distribution: {param_name} Parameter",
            xlabel=param_name,
            ylabel="channel",
        )
        return fig

    def compute_channel_contribution_original_scale(self) -> DataArray:
        channel_contribution = az.extract(
            data=self.fit_result, var_names=["channel_contributions"], combined=False
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

    def _estimate_budget_contribution_fit(
        self, channel: str, budget: float, method: str = "sigmoid"
    ) -> Tuple:
        """
        Estimate the lower and upper bounds of the contribution fit for a given channel and budget.
        This function computes the quantiles (0.05 & 0.95) of the channel contributions, estimates
        the parameters of the fit function based on the specified method (either 'sigmoid' or 'michaelis-menten'),
        and calculates the lower and upper bounds of the contribution fit.

        The function is used in the `plot_budget_scenearios` function to estimate the contribution fit for each channel
        and budget scenario. The estimated fit is then used to plot the contribution optimization bounds for each scenario.

        Parameters
        ----------
        method : str
            The method used to fit the contribution & spent non-linear relationship. It can be either 'sigmoid' or 'michaelis-menten'.
        channel : str
            The name of the channel for which the contribution fit is being estimated.
        budget : float
            The budget for the channel.

        Returns
        -------
        tuple
            A tuple containing the lower and upper bounds of the contribution fit.

        Raises
        ------
        ValueError
            If the method is not 'sigmoid' or 'michaelis-menten'.
        """
        channel_contributions_quantiles = (
            self.compute_channel_contribution_original_scale().quantile(
                q=[0.05, 0.95], dim=["chain", "draw"]
            )
        )

        # Estimate parameters based on the method
        if method == "sigmoid":
            estimate_function = estimate_sigmoid_parameters
            fit_function = sigmoid_saturation
        elif method == "michaelis-menten":
            estimate_function = estimate_menten_parameters
            fit_function = michaelis_menten
        else:
            raise ValueError("`method` must be either 'michaelis-menten' or 'sigmoid'.")

        alpha_limit_upper, lam_constant_upper = estimate_function(
            channel, self.X, channel_contributions_quantiles.sel(quantile=0.95)
        )
        alpha_limit_lower, lam_constant_lower = estimate_function(
            channel, self.X, channel_contributions_quantiles.sel(quantile=0.05)
        )

        y_fit_lower = fit_function(budget, alpha_limit_lower, lam_constant_lower)
        y_fit_upper = fit_function(budget, alpha_limit_upper, lam_constant_upper)

        return y_fit_lower, y_fit_upper

    def _plot_scenario(
        self,
        ax,
        data,
        label,
        color,
        offset,
        bar_width,
        upper_bound=None,
        lower_bound=None,
        contribution=False,
    ):
        """
        Plot a single scenario (bar-plot) on a given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to plot the scenario.
        data : dict
            Dictionary containing the data for the scenario. Keys are the names of the channels and values are the corresponding values.
        label : str
            Label for the scenario.
        color : str
            Color to use for the bars in the plot.
        offset : float
            Offset to apply to the positions of the bars in the plot.
        bar_width: float
            Bar width.
        upper_bound : dict, optional
            Dictionary containing the upper bounds for the data. Keys should match those in the `data` dictionary. Only used if `contribution` is True.
        lower_bound : dict, optional
            Dictionary containing the lower bounds for the data. Keys should match those in the `data` dictionary. Only used if `contribution` is True.
        contribution : bool, optional
            If True, plot the upper and lower bounds for the data. Default is False.

        Returns
        -------
        None
            The function adds a plot to the provided axes object in-place and doesn't return any object.
        """
        keys = sorted(k for k in data.keys() if k != "total")
        positions = [i + offset for i in range(len(keys))]
        values = [data[k] for k in keys]

        if contribution:
            upper_values = [upper_bound[k] for k in keys]
            lower_values = [lower_bound[k] for k in keys]

            ax.barh(positions, upper_values, height=bar_width, alpha=0.25, color=color)

            ax.barh(
                positions,
                values,
                height=bar_width,
                color=color,
                alpha=0.25,
            )

            ax.barh(positions, lower_values, height=bar_width, alpha=0.35, color=color)
        else:
            ax.barh(
                positions,
                values,
                height=bar_width,
                label=label,
                color=color,
                alpha=0.85,
            )

    def plot_budget_scenearios(
        self, *, base_data: Dict, method: str = "sigmoid", **kwargs
    ) -> plt.Figure:
        """
        Experimental: Plots the budget and contribution bars side by side for multiple scenarios.

        Parameters
        ----------
        base_data : dict
            Base dictionary containing 'budget' and 'contribution'.
        method : str
            The method to use for estimating contribution fit ('sigmoid' or 'michaelis-menten').
        scenarios_data : list of dict, optional
            Additional dictionaries containing other scenarios.

        Returns
        -------
        matplotlib.figure.Figure
            The resulting figure object.

        """

        scenarios_data = kwargs.get("scenarios_data", [])
        for scenario in scenarios_data:
            standardize_scenarios_dict_keys(scenario, ["contribution", "budget"])

        standardize_scenarios_dict_keys(base_data, ["contribution", "budget"])

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        scenarios = [base_data] + list(scenarios_data)
        num_scenarios = len(scenarios)
        bar_width = (
            0.8 / num_scenarios
        )  # bar width calculated based on the number of scenarios
        num_channels = len(base_data["contribution"]) - 1

        # Generate upper_bound and lower_bound dictionaries for each scenario
        upper_bounds, lower_bounds = [], []
        for scenario in scenarios:
            upper_bound, lower_bound = {}, {}
            for channel, budget in scenario["budget"].items():
                if channel != "total":
                    y_fit_lower, y_fit_upper = self._estimate_budget_contribution_fit(
                        method=method, channel=channel, budget=budget
                    )
                    upper_bound[channel] = y_fit_upper
                    lower_bound[channel] = y_fit_lower
            upper_bounds.append(upper_bound)
            lower_bounds.append(lower_bound)

        # Plot all scenarios
        for i, (scenario, upper_bound, lower_bound) in enumerate(
            zip(scenarios, upper_bounds, lower_bounds)
        ):
            color = f"C{i}"
            offset = i * bar_width - 0.4 + bar_width / 2
            label = f"Scenario {i+1}" if i else "Initial"
            self._plot_scenario(
                axes[0], scenario["budget"], label, color, offset, bar_width
            )
            self._plot_scenario(
                axes[1],
                scenario["contribution"],
                label,
                color,
                offset,
                bar_width,
                upper_bound,
                lower_bound,
                True,
            )

        axes[0].set_title("Budget Optimization")
        axes[0].set_xlabel("Budget")
        axes[0].set_yticks(range(num_channels))
        axes[0].set_yticklabels(
            [k for k in sorted(base_data["budget"].keys()) if k != "total"]
        )

        axes[1].set_title("Contribution Optimization")
        axes[1].set_xlabel("Contribution")
        axes[1].set_yticks(range(num_channels))
        axes[1].set_yticklabels(
            [k for k in sorted(base_data["contribution"].keys()) if k != "total"]
        )

        fig.suptitle("Budget and Contribution Optimization", fontsize=16, y=1.18)
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4)

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        return fig

    def _plot_response_curve_fit(
        self,
        x: np.ndarray,
        ax: plt.Axes,
        channel: str,
        color_index: int,
        xlim_max: int,
        method: str = "sigmoid",
        label: str = "Fit Curve",
    ) -> None:
        """
        Plot the curve fit for the given channel based on the estimation of the parameters.

        The function computes the mean channel contributions, estimates the parameters based on the specified method (either 'sigmoid' or 'michaelis-menten'), and plots
        the curve fit. An inflection point on the curve is also highlighted.

        Parameters
        ----------
        x : np.ndarray
            The x-axis data, usually representing the amount of input (e.g., substrate concentration in enzymology terms).
        ax : plt.Axes
            The matplotlib axes object where the plot should be drawn.
        channel : str
            The name of the channel for which the curve fit is being plotted.
        color_index : int
            An index used for color selection to ensure distinct colors for multiple plots.
        xlim_max: int
            The maximum value to be plot on the X-axis
        method: str
            The method used to fit the contribution & spent non-linear relationship. It can be either 'sigmoid' or 'michaelis-menten'.

        Returns
        -------
        None
            The function modifies the given axes object in-place and doesn't return any object.
        """
        channel_contributions = self.compute_channel_contribution_original_scale().mean(
            ["chain", "draw"]
        )

        channel_contributions_quantiles = (
            self.compute_channel_contribution_original_scale().quantile(
                q=[0.05, 0.95], dim=["chain", "draw"]
            )
        )

        if self.X is not None:
            x_mean = np.max(self.X[channel])

        # Estimate parameters based on the method
        if method == "sigmoid":
            alpha_limit, lam_constant = estimate_sigmoid_parameters(
                channel=channel,
                original_dataframe=self.X,
                contributions=channel_contributions,
            )
            alpha_limit_upper, lam_constant_upper = estimate_sigmoid_parameters(
                channel=channel,
                original_dataframe=self.X,
                contributions=channel_contributions_quantiles.sel(quantile=0.95),
            )
            alpha_limit_lower, lam_constant_lower = estimate_sigmoid_parameters(
                channel=channel,
                original_dataframe=self.X,
                contributions=channel_contributions_quantiles.sel(quantile=0.05),
            )

            x_inflection, y_inflection = find_sigmoid_inflection_point(
                alpha=alpha_limit, lam=lam_constant
            )
            fit_function = sigmoid_saturation
        elif method == "michaelis-menten":
            alpha_limit, lam_constant = estimate_menten_parameters(
                channel=channel,
                original_dataframe=self.X,
                contributions=channel_contributions,
            )
            alpha_limit_upper, lam_constant_upper = estimate_menten_parameters(
                channel=channel,
                original_dataframe=self.X,
                contributions=channel_contributions_quantiles.sel(quantile=0.95),
            )
            alpha_limit_lower, lam_constant_lower = estimate_menten_parameters(
                channel=channel,
                original_dataframe=self.X,
                contributions=channel_contributions_quantiles.sel(quantile=0.05),
            )

            y_inflection = michaelis_menten(lam_constant, alpha_limit, lam_constant)
            x_inflection = lam_constant
            fit_function = michaelis_menten
        else:
            raise ValueError("`method` must be either 'michaelis-menten' or 'sigmoid'.")

        # Set x_limit based on the method or xlim_max
        if xlim_max is not None:
            x_limit = xlim_max
        else:
            x_limit = x_mean

        # Generate x_fit and y_fit
        x_fit = np.linspace(0, x_limit, 1000)
        y_fit = fit_function(x_fit, alpha_limit, lam_constant)
        y_fit_lower = fit_function(x_fit, alpha_limit_lower, lam_constant_lower)
        y_fit_upper = fit_function(x_fit, alpha_limit_upper, lam_constant_upper)

        ax.fill_between(
            x_fit, y_fit_lower, y_fit_upper, color=f"C{color_index}", alpha=0.25
        )
        ax.plot(x_fit, y_fit, color=f"C{color_index}", label=label, alpha=0.6)
        ax.plot(
            x_inflection,
            y_inflection,
            color=f"C{color_index}",
            markerfacecolor="white",
        )

        ax.text(
            x_mean,
            ax.get_ylim()[1] / 1.25,
            f"α: {alpha_limit:.5f}",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )

        ax.set(xlabel="Spent", ylabel="Contribution")
        ax.legend()

    def optimize_channel_budget_for_maximum_contribution(
        self,
        method: str,
        total_budget: int,
        budget_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        *,
        parameters: Dict[str, Tuple[float, float]],
    ) -> pd.DataFrame:
        """
        Experimental: Optimize the allocation of a given total budget across multiple channels to maximize the expected contribution.

        The optimization is based on the method provided, where each channel's contribution
        follows a saturating function of its allocated budget. The function seeks the budget allocation
        that maximizes the total expected contribution across all channels. The method can be either 'sigmoid' or 'michaelis-menten'.

        Parameters
        ----------
        total_budget : int, required
            The total budget to be distributed across channels.
        method : str, required
            The method used to fit the contribution & spent non-linear relationship. It can be either 'sigmoid' or 'michaelis-menten'.
        parameters : Dict, required
            A dictionary where keys are channel names and values are tuples (L, k) representing the
            parameters for each channel based on the method used.
        budget_bounds : Dict, optional
            An optional dictionary defining the minimum and maximum budget for each channel.
            If not provided, the budget for each channel is constrained between 0 and its L value.

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the allocated budget and contribution information.

        Raises
        ------
        ValueError
            If any of the required parameters are not provided or have an incorrect type.
        """
        if not isinstance(budget_bounds, (dict, type(None))):
            raise TypeError("`budget_ranges` should be a dictionary or None.")

        if not isinstance(total_budget, (int, float)):
            raise ValueError(
                "The 'total_budget' parameter must be an integer or float."
            )

        if not parameters:
            raise ValueError(
                "The 'parameters' argument (keyword-only) must be provided and non-empty."
            )

        warnings.warn("This budget allocator method is experimental", UserWarning)

        return budget_allocator(
            method=method,
            total_budget=total_budget,
            channels=list(self.channel_columns),
            parameters=parameters,
            budget_ranges=budget_bounds,
        )

    def compute_channel_curve_optimization_parameters_original_scale(
        self, method: str = "sigmoid"
    ) -> Dict:
        """
        Experimental: Estimate the parameters for the saturating function of each channel's contribution.

        The function estimates the parameters (alpha, constant) for each channel based on the specified method (either 'sigmoid' or 'michaelis-menten').
        These parameters represent the maximum possible contribution (alpha) and the constant parameter which vary their definition based on the function (constant) for each channel.

        Parameters
        ----------
        method : str, required
            The method used to fit the contribution & spent non-linear relationship. It can be either 'sigmoid' or 'michaelis-menten'.

        Returns
        -------
        Dict
            A dictionary where keys are channel names and values are tuples (L, k) representing the
            parameters for each channel based on the method used.
        """
        warnings.warn(
            "The curve optimization parameters method is experimental", UserWarning
        )

        channel_contributions = self.compute_channel_contribution_original_scale().mean(
            ["chain", "draw"]
        )

        if method == "michaelis-menten":
            fit_function = estimate_menten_parameters
        elif method == "sigmoid":
            fit_function = estimate_sigmoid_parameters
        else:
            raise ValueError("`method` must be either 'michaelis-menten' or 'sigmoid'.")

        return {
            channel: fit_function(channel, self.X, channel_contributions)
            for channel in self.channel_columns
        }

    def plot_direct_contribution_curves(
        self,
        show_fit: bool = False,
        xlim_max=None,
        method: str = "sigmoid",
        channels: Optional[List[str]] = None,
        same_axes: bool = False,
    ) -> plt.Figure:
        """
        Plots the direct contribution curves for each marketing channel. The term "direct" refers to the fact
        we plot costs vs immediate returns and we do not take into account the lagged
        effects of the channels e.g. adstock transformations.

        Parameters
        ----------
        show_fit : bool, optional
            If True, the function will also plot the curve fit based on the specified method. Defaults to False.
        xlim_max : int, optional
            The maximum value to be plot on the X-axis. If not provided, the maximum value in the data will be used.
        method : str, optional
            The method used to fit the contribution & spent non-linear relationship. It can be either 'sigmoid' or 'michaelis-menten'. Defaults to 'sigmoid'.
        channels : List[str], optional
            A list of channels to plot. If not provided, all channels will be plotted.
        same_axes : bool, optional
            If True, all channels will be plotted on the same axes. Defaults to False.

        Returns
        -------
        plt.Figure
            A matplotlib Figure object with the direct contribution curves.
        """
        channels_to_plot = self.channel_columns if channels is None else channels

        if not all(channel in self.channel_columns for channel in channels_to_plot):
            unknown_channels = set(channels_to_plot) - set(self.channel_columns)
            raise ValueError(
                f"The provided channels must be a subset of the available channels. Got {unknown_channels}"
            )

        if len(channels_to_plot) != len(set(channels_to_plot)):
            raise ValueError("The provided channels must be unique.")

        channel_contributions = self.compute_channel_contribution_original_scale().mean(
            ["chain", "draw"]
        )

        if same_axes:
            nrows = 1
            figsize = (12, 4)

            def label_func(channel):
                return f"{channel} Data Points"

            def legend_title_func(channel):
                return "Legend"
        else:
            nrows = len(channels_to_plot)
            figsize = (12, 4 * len(channels_to_plot))

            def label_func(channel):
                return "Data Points"

            def legend_title_func(channel):
                return f"{channel} Legend"

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex=False,
            sharey=False,
            figsize=figsize,
            layout="constrained",
        )

        axes_channels = (
            zip(repeat(axes), channels_to_plot)
            if same_axes
            else zip(np.ravel(axes), channels_to_plot)
        )

        for i, (ax, channel) in enumerate(axes_channels):
            if self.X is not None:
                x = self.X[channels_to_plot].to_numpy()[:, i]
                y = channel_contributions.sel(channel=channel).to_numpy()

                label = label_func(channel)
                ax.scatter(x, y, label=label, color=f"C{i}")

                if show_fit:
                    label = f"{channel} Fit Curve" if same_axes else "Fit Curve"
                    self._plot_response_curve_fit(
                        x=x,
                        ax=ax,
                        channel=channel,
                        color_index=i,
                        xlim_max=xlim_max,
                        method=method,
                        label=label,
                    )

                title = legend_title_func(channel)
                ax.legend(
                    loc="upper left",
                    facecolor="white",
                    title=title,
                    fontsize="small",
                )

                ax.set(xlabel="Spent", ylabel="Contribution")

        fig.suptitle("Direct response curves", fontsize=16)
        return fig

    def _get_distribution(self, dist: Dict) -> Callable:
        """
        Retrieve a PyMC distribution callable based on the provided dictionary.

        Parameters
        ----------
        dist : Dict
            A dictionary containing the key 'dist' which should correspond to the
            name of a PyMC distribution.

        Returns
        -------
        Callable
            A PyMC distribution callable that can be used to instantiate a random
            variable.

        Raises
        ------
        ValueError
            If the specified distribution name in the dictionary does not correspond
            to any distribution in PyMC.
        """
        try:
            prior_distribution = getattr(pm, dist["dist"])
        except AttributeError:
            raise ValueError(f"Distribution {dist['dist']} does not exist in PyMC")
        return prior_distribution

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
                var_names=["channel_contributions"],
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
                    var_names=["control_contributions"],
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
            contributions_fourier_over_time = (
                az.extract(
                    self.fit_result,
                    var_names=["fourier_contributions"],
                    combined=True,
                )
                .mean("sample")
                .to_dataframe()
                .squeeze()
                .unstack()
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
        stack_groups: Optional[Dict[str, List[str]]] = None,
        original_scale: bool = False,
        area_kwargs: Optional[Dict[str, Any]] = None,
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
        all_contributions_over_time.plot.area(**area_params)
        ax.legend(title="groups", loc="center left", bbox_to_anchor=(1, 0.5))
        return fig

    def _get_channel_contributions_share_samples(self) -> DataArray:
        channel_contribution_original_scale_samples: DataArray = (
            self.compute_channel_contribution_original_scale()
        )
        numerator: DataArray = channel_contribution_original_scale_samples.sum(["date"])
        denominator: DataArray = numerator.sum("channel")
        return numerator / denominator

    def plot_channel_contribution_share_hdi(
        self, hdi_prob: float = 0.94, **plot_kwargs: Any
    ) -> plt.Figure:
        channel_contributions_share: DataArray = (
            self._get_channel_contributions_share_samples()
        )

        ax, *_ = az.plot_forest(
            data=channel_contributions_share,
            combined=True,
            hdi_prob=hdi_prob,
            **plot_kwargs,
        )
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y: 0.0%}"))
        fig: plt.Figure = plt.gcf()
        fig.suptitle("channel Contribution Share", fontsize=16, y=1.05)
        return fig

    def graphviz(self, **kwargs):
        return pm.model_to_graphviz(self.model, **kwargs)


class MMM(BaseMMM, ValidateTargetColumn, ValidateDateColumn, ValidateChannelColumns):
    pass
