from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pymc as pm

from pymc_marketing.mmm.base import MMM
from pymc_marketing.mmm.preprocessing import MaxAbsScaleChannels, MaxAbsScaleTarget
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from pymc_marketing.mmm.validating import ValidateControlColumns


class DelayedSaturatedMMM(
    MMM, MaxAbsScaleTarget, MaxAbsScaleChannels, ValidateControlColumns
):
    def __init__(
        self,
        data: pd.DataFrame,
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
            data=data,
            target_column=target_column,
            date_column=date_column,
            channel_columns=channel_columns,
            validate_data=validate_data,
            adstock_max_lag=adstock_max_lag,
        )

    def build_model(
        self,
        model_data: dict,
        model_config: dict,
    ) -> None:
        with pm.Model(coords=model_config.coords) as self.model:
            channel_data_ = pm.MutableData(
                name="channel_data",
                value=model_data.channel_data,
                dims=model_data.channel_data["dims"],
            )

            target_ = pm.MutableData(
                name="target",
                value=model_data.target_data,
                dims=model_config.target["dims"],
            )

            intercept = pm.Normal(
                name="intercept",
                mu=model_config.intercept["mu"],
                sigma=model_config.intercept["sigma"],
            )

            beta_channel = pm.HalfNormal(
                name="beta_channel",
                sigma=model_config.beta_channel["sigma"],
                dims=model_config.beta_channel["dims"],
            )  # ? Allow prior depend on channel costs?

            alpha = pm.Beta(
                name="alpha",
                alpha=model_config.aplha["alpha"],
                beta=model_config.aplha["beta"],
                dims=model_config.alpha["dims"],
            )

            lam = pm.Gamma(
                name="lam",
                alpha=model_config.lam["alpha"],
                beta=model_config.lam["beta"],
                dims=model_config.lam["dims"],
            )

            sigma = pm.HalfNormal(name="sigma", sigma=model_config.sigma["sigma"])

            channel_adstock = pm.Deterministic(
                name="channel_adstock",
                var=geometric_adstock(
                    x=channel_data_,
                    alpha=alpha,
                    l_max=model_config.channel_adstock["l_max"],
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

            if model_data.control_data is not None:
                control_data_ = pm.MutableData(
                    name="control_data",
                    value=model_data.control_data["value"],
                    dims=model_data.control_data["dims"],
                )

                gamma_control = pm.Normal(
                    name="gamma_control",
                    mu=model_config.gamma_control["mu"],
                    sigma=model_config.gamma_control["sigma"],
                    dims=model_config.gamma_control["dims"],
                )

                control_contributions = pm.Deterministic(
                    name="control_contributions",
                    var=control_data_ * gamma_control,
                    dims=model_config.control_contributions["dims"],
                )

                mu_var += control_contributions.sum(axis=-1)

            mu = pm.Deterministic(name="mu", var=mu_var, dims=model_config.mu["dims"])

            pm.Normal(
                name="likelihood",
                mu=mu,
                sigma=sigma,
                observed=target_,
                dims=model_config.likelihood["dims"],
            )

    @classmethod
    def create_sample_input(
        self, data, adstock_max_lag: int = 4
    ) -> tuple(Dict[dict, Union[dict, str]]):
        """
        Needs to be implemented by the user in the inherited class.
        Returns examples for data, model_config and sampler_config.
        This is useful for understanding the required
        data structures for the user model.
        """

        date_data = data[self.date_column]
        target_data = data[self.target_column]
        channel_data = data[self.channel_columns]
        if self.control_columns is not None:
            control_data: Optional[pd.DataFrame] = data[self.control_columns]
        else:
            control_data = None
        coords: Dict[str, Any] = {
            "date": date_data,
            "channel": channel_data.columns,
        }
        model_config = {
            "intercept": {"type": "dist", "mu": 0, "sigma": 2},
            "beta_channel": {"type": "dist", "sigma": 2, "dims": ("channel",)},
            "alpha": {"type": "dist", "alpha": 1, "beta": 3, "dims": ("channel",)},
            "lam": {"type": "dist", "alpha": 3, "beta": 1, "dims": ("channel",)},
            "sigma": {"type": "dist", "sigma": 2},
        }
        model_config["channel_adstock"] = (
            {
                "type": "deterministic",
                "apply_function": geometric_adstock,
                "var": {
                    "x": channel_data,
                    "alpha": model_config["alpha"],
                    "l_max": adstock_max_lag,
                    "normalize": True,
                },
            },
        )
        model_config["channel_addstock_saturated"] = (
            {
                "type": "deterministic",
                "apply_function": logistic_saturation,
                "var": {
                    "x": model_config["channel_adstock"],
                    "lam": model_config["lam"],
                },
                "dims": ("date", "channel"),
            },
        )
        if control_data is not None:
            coords["control"] = control_data.columns
        model_data = {
            "target": {"value": target_data, "dims": ("date",)},
            "channel_data": {"value": channel_data, "dims": ("date", "channel")},
        }
        if control_data is not None:
            model_data["control_data"] = {
                "value": control_data,
                "dims": ("date", "control"),
            }
            model_config["gamma_control"] = {
                "type": "dist",
                "mu": 0,
                "sigma": 2,
                "dims": ("control",),
            }
            model_config["control_contributions"] = {
                "type": "deterministic",
                "dims": ("date", "control"),
            }
            model_config["mu"] = {"dims": ("date",)}
            model_config["likelihood"] = {
                "dims": ("date",),
            }
            model_config["coords"] = coords

            self.sampler_config = {"progressbar": True, "random_seed": None}
        return model_data, model_config

    def _data_setter(self, data: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]]):
        """
        Sets new data in the model.

        Parameters
        ----------
        data : Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to set as idata for the model
        """

        with self.model:
            try:
                new_channel_data = data["channel_data"]
            except KeyError as e:
                print("New data must contain channel_data!", e)
            try:
                target = data["target"]
            except KeyError as e:
                print("New data must contain target", e)
            pm.set_data(
                {
                    "channel_data": new_channel_data,
                    "target": target,
                }
            )
