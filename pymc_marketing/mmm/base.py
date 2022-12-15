from abc import abstractmethod
from inspect import (
    getattr_static,
    isdatadescriptor,
    isgetsetdescriptor,
    ismemberdescriptor,
    ismethoddescriptor,
)
from typing import Any, Callable, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import seaborn as sns
from pymc.util import RandomState
from xarray import DataArray, Dataset

from pymc_marketing.mmm.validating import (
    ValidateChannelColumns,
    ValidateDateColumn,
    ValidateTargetColumn,
)

__all__ = ("BaseMMM", "MMM")


class BaseMMM:
    def __init__(
        self,
        data_df: pd.DataFrame,
        target_column: str,
        date_column: str,
        channel_columns: Union[List[str], Tuple[str]],
        validate_data: bool = True,
        **kwargs,
    ) -> None:
        self.data_df: pd.DataFrame = data_df
        self.target_column: str = target_column
        self.date_column: str = date_column
        self.channel_columns: Union[List[str], Tuple[str]] = channel_columns
        self.n_obs: int = data_df.shape[0]
        self.n_channel: int = len(channel_columns)
        self._fit_result: Optional[az.InferenceData] = None
        self._posterior_predictive: Optional[az.InferenceData] = None

        if validate_data:
            self.validate(self.data_df)
        self.preprocessed_data = self.preprocess(self.data_df.copy())

        self.model: pm.Model = pm.Model()
        self.build_model(
            data_df=self.preprocessed_data,
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
    def validation_methods(self) -> List[Callable[[pd.DataFrame], None]]:
        return [
            method
            for method in self.methods
            if getattr(method, "_tags", {}).get("validation", False)
        ]

    @property
    def preprocessing_methods(self) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:
        return [
            method
            for method in self.methods
            if getattr(method, "_tags", {}).get("preprocessing", False)
        ]

    def validate(self, data_df: pd.DataFrame):
        for method in self.validation_methods:
            method(self, data_df)

    def preprocess(self, data_df: pd.DataFrame) -> pd.DataFrame:
        for method in self.preprocessing_methods:
            data_df = method(self, data_df)
        return data_df

    @abstractmethod
    def build_model(*args, **kwargs):
        raise NotImplementedError()

    def get_prior_predictive_data(self, *args, **kwargs) -> az.InferenceData:
        with self.model:
            prior_predictive: az.InferenceData = pm.sample_prior_predictive(
                *args, **kwargs
            )
        return prior_predictive

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
            x=self.data_df[self.date_column],
            y1=likelihood_hdi_94[:, 0],
            y2=likelihood_hdi_94[:, 1],
            color="C0",
            alpha=0.2,
            label="94% HDI",
        )

        ax.fill_between(
            x=self.data_df[self.date_column],
            y1=likelihood_hdi_50[:, 0],
            y2=likelihood_hdi_50[:, 1],
            color="C0",
            alpha=0.3,
            label="50% HDI",
        )

        ax.plot(
            self.data_df[self.date_column],
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
            likelihood_hdi_94 = self.target_transformer.inverse_transform(
                Xt=likelihood_hdi_94
            )
            likelihood_hdi_50 = self.target_transformer.inverse_transform(
                Xt=likelihood_hdi_50
            )

        fig, ax = plt.subplots(**plt_kwargs)

        ax.fill_between(
            x=self.data_df[self.date_column],
            y1=likelihood_hdi_94[:, 0],
            y2=likelihood_hdi_94[:, 1],
            color="C0",
            alpha=0.2,
            label="94% HDI",
        )

        ax.fill_between(
            x=self.data_df[self.date_column],
            y1=likelihood_hdi_50[:, 0],
            y2=likelihood_hdi_50[:, 1],
            color="C0",
            alpha=0.3,
            label="50% HDI",
        )

        target_to_plot: pd.Series = (
            self.data_df[self.target_column]
            if original_scale
            else self.preprocessed_data[self.target_column]
        )
        ax.plot(self.data_df[self.date_column], target_to_plot, color="black")
        ax.set(
            title="Posterior Predictive Check",
            xlabel="date",
            ylabel=self.target_column,
        )
        return fig

    def plot_components_contributions(self, **plt_kwargs: Any) -> plt.Figure:
        model_hdi: Dataset = az.hdi(ary=self.fit_result, hdi_prob=0.94)
        # ? Should this be passed as an argument?
        contribution_vars: List[str] = ["channel_contribution"]

        if self.control_columns:
            contribution_vars.append("control_contribution")

        fig, ax = plt.subplots(**plt_kwargs)

        for i, var_contribution in enumerate(contribution_vars):
            ax.fill_between(
                x=self.data_df[self.date_column],
                y1=model_hdi[var_contribution][:, 0],
                y2=model_hdi[var_contribution][:, 1],
                color=f"C{i}",
                alpha=0.25,
                label=f"$94 %$ HDI ({var_contribution})",
            )
            sns.lineplot(
                x=self.data_df[self.date_column],
                y=az.extract(self.fit_result, var_names=[var_contribution]).mean(
                    axis=1
                ),
                color=f"C{i}",
                ax=ax,
            )

        intercept_hdi: npt.NDArray[np.float_] = np.repeat(
            a=model_hdi["intercept"].to_numpy()[None, ...],
            repeats=self.n_obs,
            axis=0,
        )
        sns.lineplot(
            x=self.data_df[self.date_column],
            y=az.extract(self.fit_result, var_names=["intercept"]).mean().item(),
            color=f"C{i + 1}",
            ax=ax,
        )
        ax.fill_between(
            x=self.data_df[self.date_column],
            y1=intercept_hdi[:, 0],
            y2=intercept_hdi[:, 1],
            color=f"C{i + 1}",
            alpha=0.25,
            label="$94 %$ HDI (intercept)",
        )
        ax.plot(
            self.data_df[self.date_column],
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
        beta_channel_samples_extended: DataArray = az.extract(
            data=self.fit_result, var_names=["beta_channel"], combined=False
        ).expand_dims({"date": self.n_obs}, axis=2)

        channel_transformed: DataArray = az.extract(
            data=self.fit_result,
            var_names=["channel_adstock_saturated"],
            combined=False,
        )

        normalization_factor: float = self.target_transformer.named_steps[
            "scaler"
        ].scale_.item()
        return (
            beta_channel_samples_extended * channel_transformed
        ) / normalization_factor

    def plot_contribution_curves(self) -> plt.Figure:
        beta_adstock_saturated_inverse_transform: DataArray = (
            az.extract(data=self.fit_result, var_names=["beta_channel"])
            / self.target_transformer.named_steps["scaler"].scale_.item()
        )

        channel_adstock_saturated_effect_original_scale: DataArray = (
            beta_adstock_saturated_inverse_transform
            * az.extract(data=self.fit_result, var_names=["channel_adstock_saturated"])
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
                x=self.data_df[self.channel_columns].to_numpy()[:, i],
                y=channel_adstock_saturated_effect_original_scale[i, :, :].mean(axis=0),
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

    def _get_channel_contributions_share_samples(self) -> DataArray:
        channel_contribution_original_scale_samples: DataArray = (
            self.compute_channel_contribution_original_scale().stack(
                samples=("chain", "draw")
            )
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
            data=channel_contributions_share.unstack(),
            combined=True,
            hdi_prob=hdi_prob,
            backend_kwargs=plot_kwargs,
        )
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y: 0.0%}"))
        fig: plt.Figure = plt.gcf()
        fig.suptitle("channel Contribution Share", fontsize=16, y=1.05)
        return fig


class MMM(BaseMMM, ValidateTargetColumn, ValidateDateColumn, ValidateChannelColumns):
    pass
