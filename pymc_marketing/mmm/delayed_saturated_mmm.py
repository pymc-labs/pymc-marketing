from typing import Any, Dict, List, Optional

import arviz as az
import pandas as pd
import pymc as pm
from xarray import DataArray

from pymc_marketing.mmm.base import MMM
from pymc_marketing.mmm.preprocessing import MaxAbsScaleChannels, MixMaxScaleTarget
from pymc_marketing.mmm.transformers import (
    geometric_adstock_vectorized,
    logistic_saturation,
)
from pymc_marketing.mmm.validating import ValidateControlColumns


class DelayedSaturatedMMM(
    MMM, MixMaxScaleTarget, MaxAbsScaleChannels, ValidateControlColumns
):
    def __init__(
        self,
        data_df: pd.DataFrame,
        target_column: str,
        date_column: str,
        channel_columns: List[str],
        validate_data: bool = True,
        control_columns: Optional[List[str]] = None,
        adstock_max_lag: int = 4,
        **kwargs,
    ) -> None:
        self.control_columns = control_columns
        self.adstock_max_lag = adstock_max_lag
        super().__init__(
            data_df=data_df,
            target_column=target_column,
            date_column=date_column,
            channel_columns=channel_columns,
            validate_data=validate_data,
            adstock_max_lag=adstock_max_lag,
        )

    def build_model(
        self,
        data_df: pd.DataFrame,
        adstock_max_lag: int = 4,
    ) -> None:
        date_data = data_df[self.date_column]
        target_data = data_df[self.target_column]
        channel_data = data_df[self.channel_columns]
        if self.control_columns is not None:
            control_data: Optional[pd.DataFrame] = data_df[self.control_columns]
        else:
            control_data = None
        coords: Dict[str, Any] = {
            "date": date_data,
            "channel": channel_data.columns,
        }

        if control_data is not None:
            coords["control_names"] = control_data.columns

        with pm.Model(coords=coords) as self.model:
            channel_data_ = pm.MutableData(
                name="channel_data",
                value=channel_data,
                dims=("date", "channel"),
            )

            target_ = pm.MutableData(name="target", value=target_data, dims="date")

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

            mu = pm.Deterministic(name="mu", var=mu_var, dims="date")

            pm.Normal(
                name="likelihood",
                mu=mu,
                sigma=sigma,
                observed=target_,
                dims="date",
            )

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
