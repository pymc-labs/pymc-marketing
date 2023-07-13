from inspect import (
    getattr_static,
    isdatadescriptor,
    isgetsetdescriptor,
    ismemberdescriptor,
    ismethoddescriptor,
)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from pymc_experimental.model_builder import ModelBuilder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xarray import DataArray, Dataset

from pymc_marketing.mmm.validating import (
    ValidateChannelColumns,
    ValidateDateColumn,
    ValidateTargetColumn,
)

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
        self.y: Optional[pd.Series] = None
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
        List[Callable[["BaseMMM", Union[pd.DataFrame, pd.Series]], None]],
        List[Callable[["BaseMMM", Union[pd.DataFrame, pd.Series]], None]],
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

    def validate(self, target: str, data: Union[pd.DataFrame, pd.Series]) -> None:
        """
        Validates the input data based on the specified target type.

        This function loops over the validation methods specified for
        the target type and applies them to the input data.

        Parameters
        ----------
        target : str
            The type of target to be validated.
            Expected values are "X" for features and "y" for the target variable.
        data : Union[pd.DataFrame, pd.Series]
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
                ["BaseMMM", Union[pd.DataFrame, pd.Series]],
                Union[pd.DataFrame, pd.Series],
            ]
        ],
        List[
            Callable[
                ["BaseMMM", Union[pd.DataFrame, pd.Series]],
                Union[pd.DataFrame, pd.Series],
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
        self, target: str, data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Preprocess the provided data according to the specified target.

        This method applies preprocessing methods to the data ("X" or "y"), which are specified in the preprocessing_methods property of this object.
        It iteratively applies each method in the appropriate list (either for "X" or "y") to the data.

        Parameters
        ----------
        target : str
            Indicates whether the data represents features ("X") or the target variable ("y").

        data : pd.DataFrame
            The data to be preprocessed.

        Returns
        -------
        Union[pd.DataFrame, pd.Series]
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
        if target == "X":
            for method in self.preprocessing_methods[0]:
                data = method(self, data)
        elif target == "y":
            for method in self.preprocessing_methods[1]:
                data = method(self, data)
        else:
            raise ValueError("Target must be either 'X' or 'y'")
        return data

    def get_target_transformer(self) -> Pipeline:
        try:
            return self.target_transformer  # type: ignore
        except AttributeError:
            identity_transformer = FunctionTransformer()
            return Pipeline(steps=[("scaler", identity_transformer)])

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
            "likelihood"
        ]
        likelihood_hdi_50: DataArray = az.hdi(ary=prior_predictive_data, hdi_prob=0.50)[
            "likelihood"
        ]

        fig, ax = plt.subplots(**plt_kwargs)
        if self.X is not None and self.y is not None:
            ax.fill_between(
                x=np.asarray(self.X[self.date_column]),
                y1=likelihood_hdi_94[:, 0],
                y2=likelihood_hdi_94[:, 1],
                color="C0",
                alpha=0.2,
                label="94% HDI",
            )

            ax.fill_between(
                x=np.asarray(self.X[self.date_column]),
                y1=likelihood_hdi_50[:, 0],
                y2=likelihood_hdi_50[:, 1],
                color="C0",
                alpha=0.3,
                label="50% HDI",
            )

            ax.plot(
                np.asarray(self.X[self.date_column]),
                np.asarray(self.preprocessed_data["y"]),
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
        )["likelihood"]
        likelihood_hdi_50: DataArray = az.hdi(
            ary=posterior_predictive_data, hdi_prob=0.50
        )["likelihood"]

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
                label="94% HDI",
            )

            ax.fill_between(
                x=self.X[self.date_column],
                y1=likelihood_hdi_50[:, 0],
                y2=likelihood_hdi_50[:, 1],
                color="C0",
                alpha=0.3,
                label="50% HDI",
            )

            target_to_plot: np.ndarray = np.asarray(
                self.y if original_scale else self.preprocessed_data["y"]
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
                    label=f"$94 %$ HDI ({var_contribution})",
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
                label="$94 %$ HDI (intercept)",
            )
            ax.plot(
                np.asarray(self.X[self.date_column]),
                np.asarray(self.preprocessed_data["y"]),
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
            title=f"Posterior Predictive {param_name} Parameter",
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

    def plot_direct_contribution_curves(self) -> plt.Figure:
        """Plots the direct contribution curves. The term "direct" refers to the fact
        we plots costs vs immediate returns and we do not take into account the lagged
        effects of the channels e.g. adstock transformations.

        Returns
        -------
        plt.Figure
            Direct contribution curves.
        """
        channel_contributions = self.compute_channel_contribution_original_scale().mean(
            ["chain", "draw"]
        )
        fig, axes = plt.subplots(
            nrows=self.n_channel,
            ncols=1,
            sharex=False,
            sharey=False,
            figsize=(12, 4 * self.n_channel),
            layout="constrained",
        )

        for i, channel in enumerate(self.channel_columns):
            ax = axes[i]
            if self.X is not None:
                sns.regplot(
                    x=self.X[self.channel_columns].to_numpy()[:, i],
                    y=channel_contributions.sel(channel=channel),
                    color=f"C{i}",
                    order=2,
                    ci=None,
                    line_kws={
                        "linestyle": "--",
                        "alpha": 0.5,
                        "label": "quadratic fit",
                    },
                    ax=ax,
                )
            ax.legend(loc="upper left")
            ax.set(title=f"{channel}", xlabel="total_cost_eur")

        fig.suptitle("Contribution Plots", fontsize=16)
        return fig

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
