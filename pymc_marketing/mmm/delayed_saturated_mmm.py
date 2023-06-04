from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
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
        target_column: str,
        date_column: str,
        channel_columns: List[str],
        data: Optional[pd.DataFrame] = None,
        model_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
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
        target_column : str
            Column name of the target variable.
        date_column : str
            Column name of the date variable.
        channel_columns : List[str]
            Column names of the media channel variables.
        data : pd.DataFrame
            Training data.
        model_config : Dictionary, optional
            dictionary of parameters that initialise model configuration. Class-default defined by the user default_model_config method.
        sampler_config : Dictionary, optional
            dictionary of parameters that initialise sampler configuration. Class-default defined by the user default_sampler_config method.
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
        self.date_column = date_column
        self.data = data

        super().__init__(
            date_column=date_column,
            data=self.data,
            target_column=target_column,
            channel_columns=channel_columns,
            model_config=model_config,
            sampler_config=sampler_config,
            channel_prior=channel_prior,
            validate_data=validate_data,
            adstock_max_lag=adstock_max_lag,
        )

    @property
    def default_sampler_config(self) -> Dict:
        return {"progressbar": True, "random_seed": 1234}

    def generate_model_data(self, data=None, validate: bool = True) -> pd.DataFrame:
        if data is None:
            seed: int = sum(map(ord, "pymc_marketing"))
            rng: np.random.Generator = np.random.default_rng(seed=seed)
            date_data: pd.DatetimeIndex = pd.date_range(
                start="2019-06-01", end="2021-12-31", freq="W-MON"
            )
            n: int = date_data.size
            self.target_column = "y"
            self.date_column = "date"
            self.channel_columns = ["channel_1", "channel_2"]
            self.control_columns = ["control_1", "control_2"]
            self.data = pd.DataFrame(
                data={
                    "date": date_data,
                    "y": rng.integers(low=0, high=100, size=n),
                    "channel_1": rng.integers(low=0, high=400, size=n),
                    "channel_2": rng.integers(low=0, high=50, size=n),
                    "control_1": rng.gamma(shape=1000, scale=500, size=n),
                    "control_2": rng.gamma(shape=100, scale=5, size=n),
                    "other_column_1": rng.integers(low=0, high=100, size=n),
                    "other_column_2": rng.normal(loc=0, scale=1, size=n),
                }
            )
            data = self.data
        date_data = data[self.date_column]
        target_data = data[self.target_column]
        channel_data = data[self.channel_columns]
        coords: Dict[str, Any] = {
            "date": date_data,
            "channel": self.channel_columns,
        }
        control_data: Optional[pd.DataFrame] = None
        if self.control_columns is not None:
            control_data = data[self.control_columns]
            coords["control"] = self.control_columns
        fourier_features: Optional[pd.DataFrame] = None
        if self.yearly_seasonality is not None:
            fourier_features = self._get_fourier_models_data()
            coords["fourier_mode"] = fourier_features.columns.to_numpy()
        model_data_dict = {
            "channel_data_": {
                "type": "MutableData",
                "value": channel_data,
                "dims": ("date", "channel"),
            },
            self.target_column: {
                "type": "MutableData",
                "value": target_data,
                "dims": "date",
            },
            "date": {"value": date_data},
        }
        self.model_coords = coords

        if control_data is not None:
            model_data_dict["control_data"] = {
                "value": control_data,
                "dims": ("date", "control"),
            }

        if fourier_features is not None:
            model_data_dict["fourier_features"] = {
                "value": fourier_features,
                "dims": ("date", "fourier_mode"),
            }
        model_data = pd.DataFrame.from_dict(
            model_data_dict
        )  # change how DataFrame is created
        self.preprocessed_data = self.preprocess(model_data.copy())
        return model_data

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
        data: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        model_config: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        self.output_var = "target"
        if data is not None:
            self.data = self.generate_model_data(data=data)
        else:
            self.data = self.generate_model_data(data=self.data)

        if not model_config:
            model_config = self.default_model_config
        with pm.Model(coords=self.model_coords) as self.model:
            channel_data_ = pm.MutableData(
                name="channel_data",
                value=self.data.channel_data_.value,
                dims=self.data.channel_data_.dims,
            )

            target_ = pm.MutableData(
                name="target",
                value=self.data[self.target_column]["value"],
                dims=self.data[self.target_column]["dims"],
            )

            intercept = pm.Normal(
                name="intercept",
                mu=model_config["intercept"]["mu"],
                sigma=model_config["intercept"]["sigma"],
            )

            """beta_channel = pm.HalfNormal(
                name="beta_channel",
                sigma=model_config["beta_channel"]["sigma"],
                dims=model_config["beta_channel"]["dims"],
            )  # ? Allow prior depend on channel costs?"""
            channel_prior = self._preprocess_channel_prior()
            beta_channel = self.model.register_rv(
                rv_var=channel_prior, name="beta_channel", dims="channel"
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
                    l_max=kwargs["adstock_max_lag"]
                    if "adstock_max_lag" in kwargs
                    else 4,
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
            if "control_data" in self.data.columns:
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

            if (
                "fourier_features" in self.data.keys()
                and self.data["fourier_features"] is not None
            ):
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

    def _get_fourier_models_data(self) -> pd.DataFrame:
        """Generates fourier modes to model seasonality.

        References
        ----------
        https://www.pymc.io/projects/examples/en/latest/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html
        """
        if self.yearly_seasonality is None:
            raise ValueError("yearly_seasonality must be specified.")
        if self.data is not None:
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
                new_channel_data = X[self.channel_columns]
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
