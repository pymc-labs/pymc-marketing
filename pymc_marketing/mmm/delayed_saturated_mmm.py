from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm

from pymc_marketing.mmm.base import MMM
from pymc_marketing.mmm.preprocessing import MaxAbsScaleChannels, MaxAbsScaleTarget
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from pymc_marketing.mmm.utils import generate_fourier_modes
from pymc_marketing.mmm.validating import ValidateControlColumns


class DelayedSaturatedMMM(
    MMM, MaxAbsScaleTarget, MaxAbsScaleChannels, ValidateControlColumns
):
    def __init__(
        self,
        target_column: str,
        date_column: str,
        channel_columns: List[str],
        data: Optional[pd.DataFrame] = None,
        model_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
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
        self.data = self.generate_model_data(data=data)
        if self.yearly_seasonality is not None:
            fourier_features = self._get_fourier_models_data()

            self.data["fourier_features"] = (
                {
                    "value": fourier_features,
                    "fourier_mode": fourier_features.columns.to_numpy(),
                    "dims": ("date", "fourier_mode"),
                },
            )
        else:
            self.data["fourier_features"]["fourier_mode"] = None
        self.data["coords"]["fourier_mode"] = self.data["fourier_features"][
            "fourier_mode"
        ]
        super().__init__(
            data=self.data,
            target_column=target_column,
            date_column=date_column,
            channel_columns=channel_columns,
            validate_data=validate_data,
            adstock_max_lag=adstock_max_lag,
        )
        if model_config is None:
            self.model_config = self.default_model_config
        if sampler_config is None:
            self.sampler_config = self.default_sampler_config

    @property
    def default_sampler_config(self) -> Dict:
        return {"progressbar": True, "random_seed": 1234}

    @classmethod
    def generate_model_data(cls, data=None) -> pd.DataFrame:
        if data is None:
            data = {
                """
                Needs to be defined, should provide basic required data structure
                that will allow users to use functions in order to learn about the class
                and it's funcitons
                """
            }
        date_data = data[cls.date_column]
        target_data = data[cls.target_column]
        channel_data = data[cls.channel_columns]
        coords: Dict[str, Any] = {
            "date": date_data,
            "channels": channel_data.columns,
        }

        if cls.control_columns is not None:
            control_data: Optional[pd.DataFrame] = data[cls.control_columns]
            coords["control"] = data[cls.control_columns].columns
        else:
            control_data = None
        model_data = {
            "channel_data_": {
                "type": "MutableData",
                "value": channel_data,
                "dims": ("date", "channel"),
            },
            "target_": {
                "type": "MutableData",
                "value": target_data,
                "dims": "date",
            },
            "control_data": {
                "value": control_data,
                "dims": ("date", "control"),
            },
            "coords": coords,
        }
        cls.data = model_data
        return model_data

    @property
    def default_model_config(self) -> Dict:
        model_config: Dict = {
            "intercept": {"type": "dist", "mu": 0, "sigma": 2},
            "beta_channel": {"type": "dist", "sigma": 2, "dims": ("channel",)},
            "alpha": {"type": "dist", "alpha": 1, "beta": 3, "dims": ("channel",)},
            "lam": {"type": "dist", "alpha": 3, "beta": 1, "dims": ("channel",)},
            "sigma": {"type": "dist", "sigma": 2},
            "gamma_control": {
                "type": "dist",
                "mu": 0,
                "sigma": 2,
                "dims": ("control",),
            },
            "control_contributions": {
                "type": "deterministic",
                "dims": ("date", "control"),
            },
            "mu": {"dims": ("date",)},
            "likelihood": {"dims": ("date",)},
            "gamma_fourier": {"mu": 0, "b": 1, "dims": "fourier_mode"},
            "fourier_contributions": {"dims": ("date", "fourier_mode")},
        }
        return model_config

    def build_model(
        self,
        data: pd.DataFrame = None,
        model_config: Optional[Dict] = None,
        adstock_max_lag: int = 4,
    ) -> None:
        if model_config is None:
            model_config = self.default_model_config
        with pm.Model(coords=data["coords"]) as self.model:
            channel_data_ = pm.MutableData(
                name="channel_data",
                value=self.data["channel_data_"]["value"],
                dims=self.data["channel_data_"]["dims"],
            )

            target_ = pm.MutableData(
                name="target",
                value=self.data["target_data_"]["value"],
                dims=self.data["target_data_"]["dims"],
            )

            intercept = pm.Normal(
                name="intercept",
                mu=model_config["intercept"]["mu"],
                sigma=model_config["intercept"]["sigma"],
            )

            beta_channel = pm.HalfNormal(
                name="beta_channel",
                sigma=model_config["beta_channel"]["sigma"],
                dims=model_config["beta_channel"]["dims"],
            )  # ? Allow prior depend on channel costs?

            alpha = pm.Beta(
                name="alpha",
                alpha=model_config["alpha"]["alpha"],
                beta=model_config["alpha"]["beta"],
                dims=model_config["alpha"]["dims"],
            )

            lam = pm.Gamma(
                name="lam",
                alpha=model_config["lam"]["alpha"],
                beta=model_config["lam"]["beta"],
                dims=model_config["lam"]["dims"],
            )

            sigma = pm.HalfNormal(name="sigma", sigma=model_config["sigma"]["sigma"])

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

            if self.data["control_data_"]["value"] is not None:
                control_data_ = pm.MutableData(
                    name="control_data",
                    value=self.data["control_data"]["value"],
                    dims=self.data["control_data"]["dims"],
                )

                gamma_control = pm.Normal(
                    name="gamma_control",
                    mu=model_config["gamma_control"]["mu"],
                    sigma=model_config["gamma_control"]["sigma"],
                    dims=model_config["gamma_control"]["dims"],
                )

                control_contributions = pm.Deterministic(
                    name="control_contributions",
                    var=control_data_ * gamma_control,
                    dims=model_config["control_contributions"]["dims"],
                )

                mu_var += control_contributions.sum(axis=-1)

            if self.data["fourier_features"]["value"] is not None:
                fourier_data_ = pm.MutableData(
                    name="fourier_data",
                    value=self.data["fourier_features"]["value"],
                    dims=self.data["fourier_features"]["dims"],
                )

                gamma_fourier = pm.Laplace(
                    name="gamma_fourier",
                    mu=model_config["gamma_fourier"]["mu"],
                    b=model_config["gamma_fourier"]["b"],
                    dims=model_config["gamma_fourier"]["dims"],
                )

                fourier_contribution = pm.Deterministic(
                    name="fourier_contributions",
                    var=fourier_data_ * gamma_fourier,
                    dims=model_config["fourier_contributions"]["dims"],
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

    @property
    def _serializable_model_config(self) -> Dict[str, Any]:
        serializable_config = self.model_config.copy()
        serializable_config["channel_adstock"]["var"]["x"] = serializable_config[
            "channel_adstock"
        ]["var"]["x"].to_dict()
        serializable_config["coords"]["date"] = serializable_config["coords"][
            "date"
        ].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        serializable_config["coords"]["date"] = serializable_config["coords"][
            "date"
        ].to_dict()
        serializable_config["coords"]["channel"] = serializable_config["coords"][
            "channel"
        ].to_list()
        if "control" in serializable_config["coords"].keys():
            serializable_config["coords"]["control"] = serializable_config["coords"][
                "control"
            ].to_list()

        return serializable_config

    def _data_setter(
        self,
        data: Dict[str, Union[np.ndarray[Any, Any], Any, Any]],
        x_only: bool = True,
    ) -> None:
        """
        Sets new data in the model.

            Parameters
        ----------
        data : Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to set as idata for the model
        """

        with self.model:
            new_channel_data = None
        try:
            new_channel_data = data["channel_data"]["value"]
        except KeyError as e:
            raise RuntimeError("New data must contain channel_data!", e)
        target = None
        try:
            target = data["target"]["value"]
        except KeyError as e:
            raise RuntimeError("New data must contain target", e)

        with self.model:
            pm.set_data(
                {
                    "channel_data": new_channel_data,
                    "target": target,
                }
            )
