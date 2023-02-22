from abc import abstractmethod
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
from pymc.util import RandomState
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xarray import DataArray

from pymc_marketing.mmm.validating import (
    ValidateChannelColumns,
    ValidateDateColumn,
    ValidateTargetColumn,
)

__all__ = ("BaseMMM", "MMM")


class BaseMMM:
    model: pm.Model

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        date_column: str,
        channel_columns: Union[List[str], Tuple[str]],
        validate_data: bool = True,
        **kwargs,
    ) -> None:
        self.data: pd.DataFrame = data
        self.target_column: str = target_column
        self.date_column: str = date_column
        self.channel_columns: Union[List[str], Tuple[str]] = channel_columns
        self.n_obs: int = data.shape[0]
        self.n_channel: int = len(channel_columns)
        self._fit_result: Optional[az.InferenceData] = None
        self._posterior_predictive: Optional[az.InferenceData] = None

        if validate_data:
            self.validate(self.data)
        self.preprocessed_data = self.preprocess(self.data.copy())

        self.build_model(
            data=self.preprocessed_data,
            **kwargs,
        )

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
    def validation_methods(self) -> List[Callable[["BaseMMM", pd.DataFrame], None]]:
        return [
            method
            for method in self.methods
            if getattr(method, "_tags", {}).get("validation", False)
        ]

    @property
    def preprocessing_methods(
        self,
    ) -> List[Callable[["BaseMMM", pd.DataFrame], pd.DataFrame]]:
        return [
            method
            for method in self.methods
            if getattr(method, "_tags", {}).get("preprocessing", False)
        ]

    def get_target_transformer(self) -> Pipeline:
        try:
            return self.target_transformer  # type: ignore
        except AttributeError:
            identity_transformer = FunctionTransformer()
            return Pipeline(steps=[("scaler", identity_transformer)])

    def validate(self, data: pd.DataFrame):
        for method in self.validation_methods:
            method(self, data)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        for method in self.preprocessing_methods:
            data = method(self, data)
        return data

    @abstractmethod
    def build_model(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_prior_predictive_data(self, *args, **kwargs) -> az.InferenceData:
        try:
            return self._prior_predictive
        except AttributeError:
            with self.model:
                self._prior_predictive: az.InferenceData = pm.sample_prior_predictive(
                    *args, **kwargs
                )
            return self._prior_predictive

    def fit(
        self,
        progressbar: bool = True,
        random_seed: RandomState = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        with self.model:
            self._fit_result = pm.sample(
                progressbar=progressbar, random_seed=random_seed, *args, **kwargs
            )
            self._posterior_predictive = pm.sample_posterior_predictive(
                trace=self._fit_result, progressbar=progressbar, random_seed=random_seed
            )

    @property
    def fit_result(self) -> az.InferenceData:
        if self._fit_result is None:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
        return self._fit_result

    @property
    def posterior_predictive(self) -> az.InferenceData:
        if self._posterior_predictive is None:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
        return self._posterior_predictive

    def plot_prior_predictive(
        self, samples: int = 1_000, **plt_kwargs: Any
    ) -> plt.Figure:
        prior_predictive_data: az.InferenceData = self.get_prior_predictive_data(
            samples=samples
        )

        likelihood_hdi_94: DataArray = az.hdi(
            ary=prior_predictive_data["prior_predictive"], hdi_prob=0.94
        )["likelihood"]
        likelihood_hdi_50: DataArray = az.hdi(
            ary=prior_predictive_data["prior_predictive"], hdi_prob=0.50
        )["likelihood"]

        fig, ax = plt.subplots(**plt_kwargs)

        ax.fill_between(
            x=self.data[self.date_column],
            y1=likelihood_hdi_94[:, 0],
            y2=likelihood_hdi_94[:, 1],
            color="C0",
            alpha=0.2,
            label="94% HDI",
        )

        ax.fill_between(
            x=self.data[self.date_column],
            y1=likelihood_hdi_50[:, 0],
            y2=likelihood_hdi_50[:, 1],
            color="C0",
            alpha=0.3,
            label="50% HDI",
        )

        ax.plot(
            self.data[self.date_column],
            self.preprocessed_data[self.target_column],
            color="black",
        )
        ax.set(title="Prior Predictive Check", xlabel="date", ylabel=self.target_column)
        return fig

    def plot_posterior_predictive(
        self, original_scale: bool = False, **plt_kwargs: Any
    ) -> plt.Figure:
        posterior_predictive_data: az.InferenceData = self.posterior_predictive

        likelihood_hdi_94: DataArray = az.hdi(
            ary=posterior_predictive_data["posterior_predictive"], hdi_prob=0.94
        )["likelihood"]
        likelihood_hdi_50: DataArray = az.hdi(
            ary=posterior_predictive_data["posterior_predictive"], hdi_prob=0.50
        )["likelihood"]

        if original_scale:
            likelihood_hdi_94 = self.get_target_transformer().inverse_transform(
                Xt=likelihood_hdi_94
            )
            likelihood_hdi_50 = self.get_target_transformer().inverse_transform(
                Xt=likelihood_hdi_50
            )

        fig, ax = plt.subplots(**plt_kwargs)

        ax.fill_between(
            x=self.data[self.date_column],
            y1=likelihood_hdi_94[:, 0],
            y2=likelihood_hdi_94[:, 1],
            color="C0",
            alpha=0.2,
            label="94% HDI",
        )

        ax.fill_between(
            x=self.data[self.date_column],
            y1=likelihood_hdi_50[:, 0],
            y2=likelihood_hdi_50[:, 1],
            color="C0",
            alpha=0.3,
            label="50% HDI",
        )

        target_to_plot: pd.Series = (
            self.data[self.target_column]
            if original_scale
            else self.preprocessed_data[self.target_column]
        )
        ax.plot(self.data[self.date_column], target_to_plot, color="black")
        ax.set(
            title="Posterior Predictive Check",
            xlabel="date",
            ylabel=self.target_column,
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
            ax.fill_between(
                x=self.data[self.date_column],
                y1=hdi.isel(hdi=0),
                y2=hdi.isel(hdi=1),
                color=f"C{i}",
                alpha=0.25,
                label=f"$94 %$ HDI ({var_contribution})",
            )
            sns.lineplot(
                x=self.data[self.date_column],
                y=mean,
                color=f"C{i}",
                ax=ax,
            )

        intercept = az.extract(self.fit_result, var_names=["intercept"], combined=False)
        intercept_hdi = np.repeat(
            a=az.hdi(intercept).intercept.data[None, ...],
            repeats=self.n_obs,
            axis=0,
        )
        sns.lineplot(
            x=self.data[self.date_column],
            y=intercept.mean().data,
            color=f"C{i + 1}",
            ax=ax,
        )
        ax.fill_between(
            x=self.data[self.date_column],
            y1=intercept_hdi[:, 0],
            y2=intercept_hdi[:, 1],
            color=f"C{i + 1}",
            alpha=0.25,
            label="$94 %$ HDI (intercept)",
        )
        ax.plot(
            self.data[self.date_column],
            self.preprocessed_data[self.target_column],
            color="black",
        )
        ax.legend(title="components", loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(
            title="Posterior Predictive Model Components",
            xlabel="date",
            ylabel=self.target_column,
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

    def plot_contribution_curves(self) -> plt.Figure:
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
            sns.regplot(
                x=self.data[self.channel_columns].to_numpy()[:, i],
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
