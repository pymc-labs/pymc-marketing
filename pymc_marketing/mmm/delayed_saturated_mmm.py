from typing import Any, Dict, List, Optional, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import seaborn as sns
from pymc.distributions.shape_utils import change_dist_size
from pytensor.tensor import TensorVariable

from pymc_marketing.mmm.base import MMM
from pymc_marketing.mmm.preprocessing import MaxAbsScaleChannels, MaxAbsScaleTarget
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from pymc_marketing.mmm.utils import generate_fourier_modes
from pymc_marketing.mmm.validating import ValidateControlColumns

__all__ = ["DelayedSaturatedMMM"]


class BaseDelayedSaturatedMMM(MMM):
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        date_column: str,
        channel_columns: List[str],
        channel_prior: Optional[TensorVariable] = None,
        validate_data: bool = True,
        control_columns: Optional[List[str]] = None,
        adstock_max_lag: int = 4,
        yearly_seasonality: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Media Mix Model with delayed adstock and logistic saturation class (see [1]_).

        Parameters
        ----------
        data : pd.DataFrame
            Training data.
        target_column : str
            Column name of the target variable.
        date_column : str
            Column name of the date variable.
        channel_columns : List[str]
            Column names of the media channel variables.
        channel_prior : Optional[TensorVariable], optional
            Prior distribution for the channel coefficients, by default None which
            corresponds to a HalfNormal distribution with sigma=2 (so that all
            contributions are positive). The prior distribution is specified by the
            `dist` API. For example, if you `pm.HalfNormal.dist(sigma=4, shape=2)`.
        validate_data : bool, optional
            Whether to validate the data upon initialization, by default True.
        control_columns : Optional[List[str]], optional
            Column names of control variables to be added as additional regressors, by default None
        adstock_max_lag : int, optional
            Number of lags to consider in the adstock transformation, by default 4
        yearly_seasonality : Optional[int], optional
            Number of Fourier modes to model yearly seasonality, by default None.

        References
        ----------
        .. [1] Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).
        """
        self.control_columns = control_columns
        self.adstock_max_lag = adstock_max_lag
        self.yearly_seasonality = yearly_seasonality
        super().__init__(
            data=data,
            target_column=target_column,
            date_column=date_column,
            channel_columns=channel_columns,
            channel_prior=channel_prior,
            validate_data=validate_data,
            adstock_max_lag=adstock_max_lag,
        )

    def _preprocess_channel_prior(self) -> TensorVariable:
        return (
            pm.HalfNormal.dist(sigma=2, shape=len(self.channel_columns))
            if self.channel_prior is None
            else change_dist_size(
                dist=self.channel_prior, new_size=len(self.channel_columns)
            )
        )

    def build_model(
        self,
        data: pd.DataFrame,
        adstock_max_lag: int = 4,
    ) -> None:
        date_data = data[self.date_column]
        target_data = data[self.target_column]
        channel_data = data[self.channel_columns]

        coords: Dict[str, Any] = {
            "date": date_data,
            "channel": channel_data.columns,
        }

        if self.control_columns is not None:
            control_data: Optional[pd.DataFrame] = data[self.control_columns]
            coords["control"] = data[self.control_columns].columns
        else:
            control_data = None

        if self.yearly_seasonality is not None:
            fourier_features = self._get_fourier_models_data()
            coords["fourier_mode"] = fourier_features.columns.to_numpy()

        else:
            fourier_features = None

        with pm.Model(coords=coords) as self.model:
            channel_data_ = pm.MutableData(
                name="channel_data",
                value=channel_data,
                dims=("date", "channel"),
            )

            target_ = pm.MutableData(name="target", value=target_data, dims="date")

            intercept = pm.Normal(name="intercept", mu=0, sigma=2)

            channel_prior = self._preprocess_channel_prior()
            beta_channel = self.model.register_rv(
                rv_var=channel_prior, name="beta_channel", dims="channel"
            )

            alpha = pm.Beta(name="alpha", alpha=1, beta=3, dims="channel")

            lam = pm.Gamma(name="lam", alpha=3, beta=1, dims="channel")

            sigma = pm.HalfNormal(name="sigma", sigma=2)

            channel_adstock = pm.Deterministic(
                name="channel_adstock",
                var=geometric_adstock(
                    x=channel_data_,
                    alpha=alpha,
                    l_max=adstock_max_lag,
                    normalize=True,
                    axis=0,
                ),
                dims=("date", "channel"),
            )
            channel_adstock_saturated = pm.Deterministic(
                name="channel_adstock_saturated",
                var=logistic_saturation(x=channel_adstock, lam=lam),
                dims=("date", "channel"),
            )
            channel_contributions = pm.Deterministic(
                name="channel_contributions",
                var=channel_adstock_saturated * beta_channel,
                dims=("date", "channel"),
            )

            mu_var = intercept + channel_contributions.sum(axis=-1)

            if control_data is not None:
                control_data_ = pm.MutableData(
                    name="control_data", value=control_data, dims=("date", "control")
                )

                gamma_control = pm.Normal(
                    name="gamma_control", mu=0, sigma=2, dims="control"
                )

                control_contributions = pm.Deterministic(
                    name="control_contributions",
                    var=control_data_ * gamma_control,
                    dims=("date", "control"),
                )

                mu_var += control_contributions.sum(axis=-1)

            if fourier_features is not None:
                fourier_data_ = pm.MutableData(
                    name="fourier_data",
                    value=fourier_features,
                    dims=("date", "fourier_mode"),
                )

                gamma_fourier = pm.Laplace(
                    name="gamma_fourier", mu=0, b=1, dims="fourier_mode"
                )

                fourier_contribution = pm.Deterministic(
                    name="fourier_contributions",
                    var=fourier_data_ * gamma_fourier,
                    dims=("date", "fourier_mode"),
                )

                mu_var += fourier_contribution.sum(axis=-1)

            mu = pm.Deterministic(name="mu", var=mu_var, dims="date")

            pm.Normal(
                name="likelihood",
                mu=mu,
                sigma=sigma,
                observed=target_,
                dims="date",
            )

    def _get_fourier_models_data(self) -> pd.DataFrame:
        """Generates fourier modes to model seasonality.

        References
        ----------
        https://www.pymc.io/projects/examples/en/latest/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html
        """
        if self.yearly_seasonality is None:
            raise ValueError("yearly_seasonality must be specified.")

        date_data: pd.Series = pd.to_datetime(
            arg=self.data[self.date_column], format="%Y-%m-%d"
        )
        periods: npt.NDArray[np.float_] = date_data.dt.dayofyear.to_numpy() / 365.25
        return generate_fourier_modes(
            periods=periods,
            n_order=self.yearly_seasonality,
        )

    def channel_contributions_forward_pass(
        self, channel_data: Union[pd.DataFrame, pd.Series, npt.NDArray[np.float_]]
    ) -> npt.NDArray[np.float_]:
        """Evaluate the channel contribution for a given channel data and a fitted model, ie. the forward pass.

        Parameters
        ----------
        channel_data : array-like
            Input channel data.

        Returns
        -------
        array-like
            Transformed channel data.
        """
        alpha_posterior = (
            az.extract(self.fit_result, group="posterior", var_names=["alpha"])
            .to_numpy()
            .T
        )

        lam_posterior = (
            az.extract(self.fit_result, group="posterior", var_names=["lam"])
            .to_numpy()
            .T
        )
        lam_posterior_expanded = np.expand_dims(a=lam_posterior, axis=1)

        beta_channel_posterior = (
            az.extract(self.fit_result, group="posterior", var_names=["beta_channel"])
            .to_numpy()
            .T
        )
        beta_channel_posterior_expanded = np.expand_dims(
            a=beta_channel_posterior, axis=1
        )

        geometric_adstock_posterior = geometric_adstock(
            x=channel_data,
            alpha=alpha_posterior,
            l_max=self.adstock_max_lag,
            normalize=True,
            axis=0,
        )

        logistic_saturation_posterior = logistic_saturation(
            x=geometric_adstock_posterior,
            lam=lam_posterior_expanded,
        )

        channel_contribution_forward_pass = (
            beta_channel_posterior_expanded * logistic_saturation_posterior
        )
        return channel_contribution_forward_pass.eval()


class DelayedSaturatedMMM(
    MaxAbsScaleTarget,
    MaxAbsScaleChannels,
    ValidateControlColumns,
    BaseDelayedSaturatedMMM,
):
    def channel_contributions_forward_pass(
        self, channel_data: Union[pd.DataFrame, pd.Series, npt.NDArray[np.float_]]
    ) -> npt.NDArray[np.float_]:
        """Evaluate the channel contribution for a given channel data and a fitted model, ie. the forward pass.

        We return the contribution in the original scale of the target variable.

        Parameters
        ----------
        channel_data : array-like
            Input channel data.

        Returns
        -------
        array-like
            Transformed channel data.
        """
        channel_contribution_forward_pass = super().channel_contributions_forward_pass(
            channel_data=channel_data
        )
        target_transformed_vectorized = np.vectorize(
            self.target_transformer.inverse_transform,
            excluded=[1, 2],
            signature="(m, n) -> (m, n)",
        )
        return target_transformed_vectorized(channel_contribution_forward_pass)

    def get_channel_contributions_forward_pass_grid(
        self, start: float, stop: float, num: int
    ) -> npt.NDArray[np.float_]:
        """Generate a grid of scaled channel contributions for a given grid of share values.

        Parameters
        ----------
        start : float
            Start of the grid. It must be equal or greater than 0.
        stop : float
            End of the grid. It must be greater than start.
        num : int
            Number of points in the grid.

        Returns
        -------
        array-like
            Grid of channel contributions.
        """
        share_grid = np.linspace(start=start, stop=stop, num=num)

        channel_contributions = []
        for delta in share_grid:
            channel_data = (
                delta
                * self.max_abs_scale_channel_data(data=self.data)[
                    self.channel_columns
                ].to_numpy()
            )
            channel_contribution_forward_pass = self.channel_contributions_forward_pass(
                channel_data=channel_data
            )
            channel_contributions.append(channel_contribution_forward_pass)
        return np.array(channel_contributions)

    def plot_channel_contributions_grid(
        self, start: float, stop: float, num: int, **plt_kwargs: Any
    ) -> plt.Figure:
        """Plots a grid of scaled channel contributions for a given grid of share values.

        Parameters
        ----------
        start : float
            Start of the grid. It must be equal or greater than 0.
        stop : float
            End of the grid. It must be greater than start.
        num : int
            Number of points in the grid.

        Returns
        -------
        plt.Figure
            Plot of grid of channel contributions.
        """
        if start < 0:
            raise ValueError("start must be greater than or equal to 0.")

        share_grid = np.linspace(start=start, stop=stop, num=num)
        contributions = self.get_channel_contributions_forward_pass_grid(
            start=start, stop=stop, num=num
        )

        fig, ax = plt.subplots(**plt_kwargs)

        for i, x in enumerate(self.channel_columns):
            hdi_contribution = az.hdi(ary=contributions[:, :, :, i].sum(axis=-1).T)

            ax.fill_between(
                x=share_grid,
                y1=hdi_contribution[:, 0],
                y2=hdi_contribution[:, 1],
                color=f"C{i}",
                label=f"{x} $94%$ HDI contribution",
                alpha=0.4,
            )

            sns.lineplot(
                x=share_grid,
                y=contributions[:, :, :, i].sum(axis=-1).mean(axis=1),
                color=f"C{i}",
                marker="o",
                label=f"{x} contribution mean",
                ax=ax,
            )

        ax.axvline(x=1, color="black", linestyle="--", label=r"$\delta = 1$")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(
            title="Channel contribution as a function of cost share",
            xlabel=r"$\delta$",
            ylabel="contribution (sales)",
        )
