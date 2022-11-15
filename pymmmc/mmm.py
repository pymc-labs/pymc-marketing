from typing import Any, Dict, List, Optional, Sequence, Union

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import seaborn as sns
from aesara.compile.sharedvalue import SharedVariable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from xarray import DataArray, Dataset

from pymmmc.transformers import geometric_adstock_vectorized, logistic_saturation

RandomSeed = Optional[Union[int, Sequence[int], np.ndarray]]
RandomState = Union[RandomSeed, np.random.RandomState, np.random.Generator]


class MMM:
    def __init__(
        self,
        data_df: pd.DataFrame,
        y_column: str,
        date_column: str,
        channel_columns: List[str],
        control_columns: Optional[List[str]] = None,
        adstock_max_lag: int = 4,
    ) -> None:
        self.data_df: pd.DataFrame = data_df.copy()
        self.y_column: str = y_column
        self.date_column: str = date_column
        self.channel_columns: List[str] = channel_columns
        self.n_obs: int = data_df.shape[0]
        self.n_channel: int = len(channel_columns)
        self.model: pm.Model = pm.Model()
        self._fit_result: Optional[az.InferenceData] = None
        self._posterior_predictive: Optional[az.InferenceData] = None
        self.control_columns: Optional[List[str]] = control_columns
        self.adstock_max_lag: int = adstock_max_lag

        self._validate_control_columns()
        self._preprocess_data()

        self._build_model(
            date_data=self.data_df[self.date_column],
            target_data=self.target_data_transformed,
            channel_data=self.channel_data_transformed,
            control_data=self.control_data_transformed,
            adstock_max_lag=self.adstock_max_lag,
        )

    def _validate_input_data(self) -> None:
        self._validate_target()
        self._validate_date_col()
        self._validate_channel_columns()

    def _validate_target(self) -> None:
        if self.y_column not in self.data_df.columns:
            raise ValueError(f"target {self.y_column} not in data_df")

    def _validate_date_col(self) -> None:
        if self.date_column not in self.data_df.columns:
            raise ValueError(f"date_col {self.date_column} not in data_df")
        if not self.data_df[self.date_column].is_unique:
            raise ValueError(f"date_col {self.date_column} has repeated values")

    def _validate_channel_columns(self) -> None:
        if self.channel_columns is None:
            raise ValueError("channel_columns must not be None")
        if not isinstance(self.channel_columns, list):
            raise ValueError("channel_columns must be a list or tuple")
        if len(self.channel_columns) == 0:
            raise ValueError("channel_columns must not be empty")
        if not set(self.channel_columns).issubset(self.data_df.columns):
            raise ValueError(f"channel_columns {self.channel_columns} not in data_df")
        if len(set(self.channel_columns)) != len(self.channel_columns):
            raise ValueError(
                f"channel_columns {self.channel_columns} contains duplicates"
            )
        if (self.data_df[self.channel_columns] < 0).any().any():
            raise ValueError(
                f"channel_columns {self.channel_columns} contains negative values"
            )

    def _validate_control_columns(self) -> None:
        if self.control_columns is not None and not set(self.control_columns).issubset(
            self.data_df.columns
        ):
            raise ValueError(f"control_columns {self.control_columns} not in data_df")

    def _preprocess_data(self) -> None:
        self._preprocess_target_data()
        self._preprocess_channel_data()
        self._preprocess_control_data()

    def _preprocess_target_data(self) -> None:
        target_vector: npt.NDArray[np.float_] = (
            self.data_df[self.y_column].to_numpy().reshape(-1, 1)
        )

        transformers = [("scaler", MinMaxScaler())]
        pipeline: Pipeline = Pipeline(steps=transformers)
        self.target_transformer: Pipeline = pipeline.fit(X=target_vector)
        self.target_data_transformed: pd.Series = pd.Series(
            data=self.target_transformer.transform(X=target_vector).flatten(),
        )

    def _preprocess_channel_data(self) -> None:
        channel_data: pd.DataFrame = self.data_df[self.channel_columns]
        # potentially add more transformations (e.g. log)
        transformers = [("scaler", MaxAbsScaler())]
        pipeline: Pipeline = Pipeline(steps=transformers)
        self.channel_transformer: Pipeline = pipeline.fit(X=channel_data.to_numpy())
        self.channel_data_transformed: pd.DataFrame = pd.DataFrame(
            data=self.channel_transformer.transform(channel_data.to_numpy()),
            columns=self.channel_columns,
        )

    def _preprocess_control_data(self) -> None:
        self.control_data_transformed: Optional[pd.DataFrame] = None
        self.control_transformer: Optional[Pipeline] = None
        if self.control_columns:
            control_data: pd.DataFrame = self.data_df[self.control_columns]
            # potentially add more transformations (e.g. log)
            transformers = [("scaler", MinMaxScaler())]
            pipeline: Pipeline = Pipeline(steps=transformers)
            self.control_transformer = pipeline.fit(X=control_data.to_numpy())
            self.control_data_transformed = pd.DataFrame(
                data=self.control_transformer.transform(control_data.to_numpy()),
                columns=self.control_columns,
            )

    def _build_model(
        self,
        date_data: pd.Series,
        target_data: pd.Series,
        channel_data: pd.DataFrame,
        control_data: Optional[pd.DataFrame] = None,
        adstock_max_lag: int = 4,
    ) -> None:
        coords: Dict[str, Any] = {
            "date": date_data,
            "channel": channel_data.columns,
        }

        if control_data is not None:
            coords["control_names"] = control_data.columns

        with pm.Model(coords=coords) as self.model:
            channel_data_: SharedVariable = pm.MutableData(
                name="channel_data",
                value=channel_data,
                dims=("date", "channel"),
            )

            target_: SharedVariable = pm.MutableData(
                name="target", value=target_data, dims="date"
            )

            intercept = pm.Normal(name="intercept", mu=0, sigma=2)

            beta_channel = pm.HalfNormal(
                name="beta_channel", sigma=2, dims="channel"
            )  # ? Allow prior depend on channel costs?

            alpha = pm.Beta(name="alpha", alpha=1, beta=3, dims="channel")

            lam = pm.Gamma(name="lam", alpha=3, beta=1, dims="channel")

            sigma = pm.HalfNormal(name="sigma", sigma=2)

            channel_adstock = pm.Deterministic(
                name="channel_adstock",
                var=geometric_adstock_vectorized(
                    x=channel_data_,
                    alpha=alpha,
                    l_max=adstock_max_lag,
                    normalize=True,
                ),
                dims=("date", "channel"),
            )
            channel_adstock_saturated = pm.Deterministic(
                name="channel_adstock_saturated",
                var=logistic_saturation(x=channel_adstock, lam=lam),
                dims=("date", "channel"),
            )
            channel_contribution = pm.Deterministic(
                name="channel_contribution",
                var=pm.math.dot(channel_adstock_saturated, beta_channel),
                dims="date",
            )

            mu_var = intercept + channel_contribution

            if control_data is not None:
                control_data_: SharedVariable = pm.MutableData(
                    name="control_data", value=control_data, dims=("date", "control")
                )

                gamma_control = pm.Normal(
                    name="gamma_control", mu=0, sigma=2, dims="control"
                )

                control_contribution = pm.Deterministic(
                    name="control_contribution",
                    var=pm.math.dot(control_data_, gamma_control),
                    dims="date",
                )

                mu_var += control_contribution

            mu = pm.Deterministic(name="mu", var=mu_var, dims="date")

            pm.Normal(
                name="likelihood",
                mu=mu,
                sigma=sigma,
                observed=target_,
                dims="date",
            )

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

        sns.lineplot(
            x=self.data_df[self.date_column],
            y=self.target_data_transformed,
            color="black",
            ax=ax,
        )
        ax.set(title="Prior Predictive Check", xlabel="date", ylabel=self.y_column)
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
            self.data_df[self.y_column]
            if original_scale
            else self.target_data_transformed
        )
        sns.lineplot(
            x=self.data_df[self.date_column], y=target_to_plot, color="black", ax=ax
        )
        ax.set(
            title="Posterior Predictive Check",
            xlabel="date",
            ylabel=self.y_column,
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
        )
        ax.fill_between(
            x=self.data_df[self.date_column],
            y1=intercept_hdi[:, 0],
            y2=intercept_hdi[:, 1],
            color=f"C{i + 1}",
            alpha=0.25,
            label="$94 %$ HDI (intercept)",
        )
        sns.lineplot(
            x=self.data_df[self.date_column],
            y=self.target_data_transformed,
            color="black",
            ax=ax,
        )
        ax.legend(title="components", loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(
            title="Posterior Predictive Model Components",
            xlabel="date",
            ylabel=self.y_column,
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
