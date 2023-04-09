from typing import Any, Dict, List, Optional

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
        data: pd.DataFrame,
        adstock_max_lag: int = 4,
    ) -> None:
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

        if control_data is not None:
            coords["control"] = control_data.columns

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

            mu = pm.Deterministic(name="mu", var=mu_var, dims="date")

            pm.Normal(
                name="likelihood",
                mu=mu,
                sigma=sigma,
                observed=target_,
                dims="date",
            )

            self.data, self.model_config = self._assmemble_model_info(
                data, coords, adstock_max_lag, control_data
            )

        def _assmemble_model_info(
            self, data, coords, adstock_max_lag, control_data
        ) -> tuple(Dict, Dict):
            target_data = data[self.target_column]
            channel_data = data[self.channel_columns]
            model_data = {
                "target": {"value": target_data, "dims": ("date",)},
                "channel_data": {"value": channel_data, "dims": ("date", "channel")},
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
            model_config["mu_var"] = (
                {
                    # got to figure out how to save information about interactions between model variables
                    # or functions being applied to them
                },
            )
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
                    # TODO find a way to save interactions
                    "var": {},
                    "dims": ("date", "control"),
                }
                model_config["mu"] = {"var": model_config["mu_var"], "dims": ("date",)}
                model_config["likelihood"] = {
                    "mu": model_config["mu"],
                    "sigma": model_config["sigma"],
                    "observed": model_data["target"],
                    "dims": ("date",),
                }

            return model_data, model_config
