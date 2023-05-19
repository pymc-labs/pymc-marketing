from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
from pymc.distributions.shape_utils import change_dist_size
from pytensor.tensor import TensorVariable

import theano.tensor as tt

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
        channel_priors: Optional[Dict[str, pm.Distribution]] = None,
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
        self.channel_priors = channel_priors
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

            if self.channel_priors is None:
                beta_channel = pm.HalfNormal(name="beta_channel", sigma=2, dims="channel")
            else:
                beta_channel = []
                for channel in self.channel_columns:
                    if channel in self.channel_priors:
                        beta_channel.append(self.channel_priors[channel])
                    else:
                        beta_channel.append(pm.HalfNormal(name=f"beta_{channel}", sigma=2))
                beta_channel = tt.stack(beta_channel, axis=-1)  
            # ? Allow prior depend on channel costs?

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


class DelayedSaturatedMMM(
    MaxAbsScaleTarget,
    MaxAbsScaleChannels,
    ValidateControlColumns,
    BaseDelayedSaturatedMMM,
):
    ...
