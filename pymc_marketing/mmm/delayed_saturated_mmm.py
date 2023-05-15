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

__all__ = ["DelayedSaturatedMMM"]


class BaseDelayedSaturatedMMM(MMM):
    def __init__(
        self,
        date_column: str,
        channel_columns: List[str],
        adstock_max_lag: int,
        model_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
        validate_data: bool = True,
        control_columns: Optional[List[str]] = None,
        yearly_seasonality: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Media Mix Model with delayed adstock and logistic saturation class (see [1]_).

        Parameters
        ----------
        date_column : str
            Column name of the date variable.
        channel_columns : List[str]
            Column names of the media channel variables.
        model_config : Dictionary, optional
            dictionary of parameters that initialise model configuration. Class-default defined by the user default_model_config method.
        sampler_config : Dictionary, optional
            dictionary of parameters that initialise sampler configuration. Class-default defined by the user default_sampler_config method.
        validate_data : bool, optional
            Whether to validate the data before fitting to model, by default True.
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
        self.date_column = date_column
        self.validate_data = validate_data

        super().__init__(
            date_column=date_column,
            channel_columns=channel_columns,
            model_config=model_config,
            sampler_config=sampler_config,
            adstock_max_lag=adstock_max_lag,
        )

    @property
    def default_sampler_config(self) -> Dict:
        return {"progressbar": True, "random_seed": 1234}

    @property
    def output_var(self):
        return "y"

    def generate_and_preprocess_model_data(
        self, X: Union[pd.DataFrame, pd.Series], y: pd.Series
    ) -> None:
        """
        Applies preprocessing to the data before fitting the model.
        if validate is True, it will check if the data is valid for the model.
        sets self.model_coords based on provided dataset

        Parameters
        ----------
        X : array, shape (n_obs, n_features)
        y : array, shape (n_obs,)
        """
        date_data = X[self.date_column]
        channel_data = X[self.channel_columns]
        coords: Dict[str, Any] = {
            "date": date_data,
            "channel": self.channel_columns,
        }

        new_X_dict = {
            self.date_column: date_data,
        }
        X_data = pd.DataFrame.from_dict(new_X_dict)
        X_data = pd.concat([X_data, channel_data], axis=1)
        control_data: Optional[Union[pd.DataFrame, pd.Series]] = None
        if self.control_columns is not None:
            control_data = X[self.control_columns]
            coords["control"] = self.control_columns
            X_data = pd.concat([X_data, control_data], axis=1)

        fourier_features: Optional[pd.DataFrame] = None
        if self.yearly_seasonality is not None:
            fourier_features = self._get_fourier_models_data(X=X)
            self.fourier_columns = fourier_features.columns
            coords["fourier_mode"] = fourier_features.columns.to_numpy()
            X_data = pd.concat([X_data, fourier_features], axis=1)

        self.model_coords = coords
        if self.validate_data:
            self.validate("X", X_data)
            self.validate("y", y)
        self.preprocessed_data: Dict[str, Union[pd.DataFrame, pd.Series]] = {
            "X": self.preprocess("X", X_data.copy()),
            "y": self.preprocess("y", y.copy()),
        }
        self.X: pd.DataFrame = X_data
        self.y: pd.Series = y

    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> None:
        model_config = self.model_config
        self.generate_and_preprocess_model_data(X, y)
        with pm.Model(coords=self.model_coords) as self.model:
            channel_data_ = pm.MutableData(
                name="channel_data",
                value=self.preprocessed_data["X"][self.channel_columns].to_numpy(),
                dims=("date", "channel"),
            )

            target_ = pm.MutableData(
                name="target",
                value=self.preprocessed_data["y"],
                dims="date",
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
            )
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
                    l_max=self.adstock_max_lag,
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
            if (
                self.control_columns is not None
                and len(self.control_columns) > 0
                and all(
                    column in self.preprocessed_data["X"].columns
                    for column in self.control_columns
                )
            ):
                control_data_ = pm.MutableData(
                    name="control_data",
                    value=self.preprocessed_data["X"][self.control_columns],
                    dims=("date", "control"),
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
                    dims=("date", "control"),
                )

                mu_var += control_contributions.sum(axis=-1)
            if (
                hasattr(self, "fourier_columns")
                and self.fourier_columns is not None
                and len(self.fourier_columns) > 0
                and all(
                    column in self.preprocessed_data["X"].columns
                    for column in self.fourier_columns
                )
            ):
                fourier_data_ = pm.MutableData(
                    name="fourier_data",
                    value=self.preprocessed_data["X"][self.fourier_columns],
                    dims=("date", "fourier_mode"),
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
                    dims=("date", "fourier_mode"),
                )

                mu_var += fourier_contribution.sum(axis=-1)

            mu = pm.Deterministic(
                name="mu", var=mu_var, dims=model_config["mu"]["dims"]
            )

            pm.Normal(
                name="likelihood",
                mu=mu,
                sigma=sigma,
                observed=target_,
                dims=model_config["likelihood"]["dims"],
            )

    @property
    def default_model_config(self) -> Dict:
        model_config: Dict = {
            "intercept": {"mu": 0, "sigma": 2},
            "beta_channel": {"sigma": 2, "dims": ("channel",)},
            "alpha": {"alpha": 1, "beta": 3, "dims": ("channel",)},
            "lam": {"alpha": 3, "beta": 1, "dims": ("channel",)},
            "sigma": {"sigma": 2},
            "gamma_control": {
                "mu": 0,
                "sigma": 2,
                "dims": ("control",),
            },
            "mu": {"dims": ("date",)},
            "likelihood": {"dims": ("date",)},
            "gamma_fourier": {"mu": 0, "b": 1, "dims": "fourier_mode"},
        }
        return model_config

    def _get_fourier_models_data(self, X) -> pd.DataFrame:
        """Generates fourier modes to model seasonality.

        References
        ----------
        https://www.pymc.io/projects/examples/en/latest/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html
        """
        if self.yearly_seasonality is None:
            raise ValueError("yearly_seasonality must be specified.")
        date_data: pd.Series = pd.to_datetime(
            arg=X[self.date_column], format="%Y-%m-%d"
        )
        periods: npt.NDArray[np.float_] = date_data.dt.dayofyear.to_numpy() / 365.25
        return generate_fourier_modes(
            periods=periods,
            n_order=self.yearly_seasonality,
        )

    @property
    def _serializable_model_config(self) -> Dict[str, Any]:
        serializable_config = self.model_config.copy()
        return serializable_config

    def _data_setter(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series] = None,
    ) -> None:
        """
        Sets new data in the model.

        This function accepts data in various formats and sets them into the
        model using the PyMC's `set_data` method. The data corresponds to the
        channel data and the target.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Data for the channel. It can be a numpy array or pandas DataFrame.
            If it's a DataFrame, it should contain a column "channel_data".
        y : Union[np.ndarray, pd.Series], optional
            Target data. It can be a numpy array or a pandas Series.
            If it's a Series, its values are used. If it's an ndarray, it's used
            directly. The default is None.

        Raises
        ------
        RuntimeError
            If the data for the channel is not provided in `X`.
        TypeError
            If `y` is not a pandas Series or a numpy array.

        Returns
        -------
        None
        """
        new_channel_data = None
        if isinstance(X, pd.DataFrame):
            try:
                new_channel_data = X[self.channel_columns]
            except KeyError as e:
                raise RuntimeError("New data must contain channel_data!", e)
        elif isinstance(X, np.ndarray):
            new_channel_data = (
                X  # Adjust as necessary depending on the structure of your ndarray
            )
        else:
            raise TypeError("X must be either a pandas DataFrame or a numpy array")

        target = None
        if isinstance(y, pd.Series):
            target = y.values
        elif isinstance(y, np.ndarray):
            target = y
        else:
            raise TypeError("y must be either a pandas Series or a numpy array")

        with self.model:
            pm.set_data(
                {
                    "channel_data": new_channel_data,
                    "target": target,
                }
            )



    @property
    def _serializable_model_config(self) -> Dict[str, Any]:
        serializable_config = self.model_config.copy()
        return serializable_config

    def _data_setter(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """
        Sets new data in the model.

        This function accepts data in various formats and sets them into the
        model using the PyMC's `set_data` method. The data corresponds to the
        channel data and the target.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Data for the channel. It can be a numpy array or pandas DataFrame.
            If it's a DataFrame, the columns corresponding to self.channel_columns
            are used. If it's an ndarray, it's used directly.
        y : Union[np.ndarray, pd.Series], optional
            Target data. It can be a numpy array or a pandas Series.
            If it's a Series, its values are used. If it's an ndarray, it's used
            directly. The default is None.

        Raises
        ------
        RuntimeError
            If the data for the channel is not provided in `X`.
        TypeError
            If `X` is not a pandas DataFrame or a numpy array, or
            if `y` is not a pandas Series or a numpy array and is not None.

        Returns
        -------
        None
        """

        new_channel_data = None
        if isinstance(X, pd.DataFrame):
            try:
                new_channel_data = X[self.channel_columns].to_numpy()
            except KeyError as e:
                raise RuntimeError("New data must contain channel_data!", e)
        elif isinstance(X, np.ndarray):
            new_channel_data = X  # type: ignore
        else:
            raise TypeError("X must be either a pandas DataFrame or a numpy array")

        target = None
        if y is not None:
            if isinstance(y, pd.Series):
                target = y.values
            elif isinstance(y, np.ndarray):
                target = y
            else:
                raise TypeError("y must be either a pandas Series or a numpy array")

        with self.model:
            pm.set_data(
                {
                    "channel_data": new_channel_data,
                    "target": target,
                }
            )

class DelayedSaturatedMMM(
    MaxAbsScaleTarget,
    MaxAbsScaleChannels,
    ValidateControlColumns,
    BaseDelayedSaturatedMMM,
):
    ...