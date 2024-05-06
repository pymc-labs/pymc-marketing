#   Copyright 2024 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Media Mix Model with delayed adstock and logistic saturation class."""

import json
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import seaborn as sns
from pytensor.tensor import TensorVariable
from xarray import DataArray, Dataset

from pymc_marketing.constants import DAYS_IN_YEAR
from pymc_marketing.mmm.base import MMM
from pymc_marketing.mmm.lift_test import (
    add_logistic_empirical_lift_measurements_to_likelihood,
    scale_lift_measurements,
)
from pymc_marketing.mmm.preprocessing import MaxAbsScaleChannels, MaxAbsScaleTarget
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from pymc_marketing.mmm.tvp import create_time_varying_intercept, infer_time_index
from pymc_marketing.mmm.utils import (
    apply_sklearn_transformer_across_dim,
    create_new_spend_data,
    generate_fourier_modes,
)
from pymc_marketing.mmm.validating import ValidateControlColumns

__all__ = ["DelayedSaturatedMMM"]


class BaseDelayedSaturatedMMM(MMM):
    """Base class for a media mix model with delayed adstock and logistic saturation class (see [1]_).

    References
    ----------
    .. [1] Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).
    """

    _model_type = "DelayedSaturatedMMM"
    version = "0.0.2"

    def __init__(
        self,
        date_column: str,
        channel_columns: list[str],
        adstock_max_lag: int,
        time_varying_intercept: bool = False,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        validate_data: bool = True,
        control_columns: list[str] | None = None,
        yearly_seasonality: int | None = None,
        **kwargs,
    ) -> None:
        """Constructor method.

        Parameters
        ----------
        date_column : str
            Column name of the date variable.
        channel_columns : List[str]
            Column names of the media channel variables.
        adstock_max_lag : int
            Number of lags to consider in the adstock transformation.
        time_varying_intercept : bool, optional
            Whether to consider time-varying intercept, by default False.
        model_config : Dictionary, optional
            dictionary of parameters that initialise model configuration.
            Class-default defined by the user default_model_config method.
        sampler_config : Dictionary, optional
            dictionary of parameters that initialise sampler configuration.
            Class-default defined by the user default_sampler_config method.
        validate_data : bool, optional
            Whether to validate the data before fitting to model, by default True.
        control_columns : Optional[List[str]], optional
            Column names of control variables to be added as additional regressors, by default None
        adstock_max_lag : int, optional
            Number of lags to consider in the adstock transformation, by default 4
        yearly_seasonality : Optional[int], optional
            Number of Fourier modes to model yearly seasonality, by default None.
        """
        self.control_columns = control_columns
        self.adstock_max_lag = adstock_max_lag
        self.time_varying_intercept = time_varying_intercept
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
    def default_sampler_config(self) -> dict:
        return {}

    @property
    def output_var(self):
        """Defines target variable for the model"""
        return "y"

    def _generate_and_preprocess_model_data(  # type: ignore
        self, X: pd.DataFrame | pd.Series, y: pd.Series | np.ndarray
    ) -> None:
        """Applies preprocessing to the data before fitting the model.

        If validate is True, it will check if the data is valid for the model.
        sets self.model_coords based on provided dataset

        Parameters
        ----------
        X : Union[pd.DataFrame, pd.Series], shape (n_obs, n_features)
        y : Union[pd.Series, np.ndarray], shape (n_obs,)

        Sets
        ----
        preprocessed_data : Dict[str, Union[pd.DataFrame, pd.Series]]
            Preprocessed data for the model.
        X : pd.DataFrame
            A filtered version of the input `X`, such that it is guaranteed that
            it contains only the `date_column`, the columns that are specified
            in the `channel_columns` and `control_columns`, and fourier features
            if `yearly_seasonality=True`.
        y : Union[pd.Series, np.ndarray]
            The target variable for the model (as provided).
        _time_index : np.ndarray
            The index of the date column. Used by TVP
        _time_index_mid : int
            The middle index of the date index. Used by TVP.
        _time_resolution: int
            The time resolution of the date index. Used by TVP.
        """
        date_data = X[self.date_column]
        channel_data = X[self.channel_columns]

        self.coords_mutable: dict[str, Any] = {
            "date": date_data,
        }
        coords: dict[str, Any] = {
            "channel": self.channel_columns,
        }

        new_X_dict = {
            self.date_column: date_data,
        }
        X_data = pd.DataFrame.from_dict(new_X_dict)
        X_data = pd.concat([X_data, channel_data], axis=1)
        control_data: pd.DataFrame | pd.Series | None = None
        if self.control_columns is not None:
            control_data = X[self.control_columns]
            coords["control"] = self.control_columns
            X_data = pd.concat([X_data, control_data], axis=1)

        fourier_features: pd.DataFrame | None = None
        if self.yearly_seasonality is not None:
            fourier_features = self._get_fourier_models_data(X=X)
            self.fourier_columns = fourier_features.columns
            coords["fourier_mode"] = fourier_features.columns.to_numpy()
            X_data = pd.concat([X_data, fourier_features], axis=1)

        self.model_coords = coords
        if self.validate_data:
            self.validate("X", X_data)
            self.validate("y", y)
        self.preprocessed_data: dict[str, pd.DataFrame | pd.Series] = {
            "X": self.preprocess("X", X_data),  # type: ignore
            "y": self.preprocess("y", y),  # type: ignore
        }
        self.X: pd.DataFrame = X_data
        self.y: pd.Series | np.ndarray = y

        if self.time_varying_intercept:
            self._time_index = np.arange(0, X.shape[0])
            self._time_index_mid = X.shape[0] // 2
            self._time_resolution = (
                self.X[self.date_column].iloc[1] - self.X[self.date_column].iloc[0]
            ).days

    def _save_input_params(self, idata) -> None:
        """Saves input parameters to the attrs of idata."""
        idata.attrs["date_column"] = json.dumps(self.date_column)
        idata.attrs["control_columns"] = json.dumps(self.control_columns)
        idata.attrs["channel_columns"] = json.dumps(self.channel_columns)
        idata.attrs["adstock_max_lag"] = json.dumps(self.adstock_max_lag)
        idata.attrs["validate_data"] = json.dumps(self.validate_data)
        idata.attrs["yearly_seasonality"] = json.dumps(self.yearly_seasonality)

    def _create_likelihood_distribution(
        self,
        dist: dict,
        mu: TensorVariable,
        observed: np.ndarray | pd.Series,
        dims: str,
    ) -> TensorVariable:
        """
        Create and return a likelihood distribution for the model.

        This method prepares the distribution and its parameters as specified in the
        configuration dictionary, validates them, and constructs the likelihood
        distribution using PyMC.

        Parameters
        ----------
        dist : Dict
            A configuration dictionary that must contain a 'dist' key with the name of
            the distribution and a 'kwargs' key with parameters for the distribution.
        observed : Union[np.ndarray, pd.Series]
            The observed data to which the likelihood distribution will be fitted.
        dims : str
            The dimensions of the data.

        Returns
        -------
        TensorVariable
            The likelihood distribution constructed with PyMC.

        Raises
        ------
        ValueError
            If 'kwargs' key is missing in `dist`, or the parameter configuration does
            not contain 'dist' and 'kwargs' keys, or if 'mu' is present in the nested
            'kwargs'
        """
        allowed_distributions = [
            "Normal",
            "StudentT",
            "Laplace",
            "Logistic",
            "LogNormal",
            "Wald",
            "TruncatedNormal",
            "Gamma",
            "AsymmetricLaplace",
            "VonMises",
        ]

        if dist["dist"] not in allowed_distributions:
            raise ValueError(
                f"""
                The distribution used for the likelihood is not allowed.
                Please, use one of the following distributions: {allowed_distributions}.
                """
            )

        # Validate that 'kwargs' is present and is a dictionary
        if "kwargs" not in dist or not isinstance(dist["kwargs"], dict):
            raise ValueError(
                "The 'kwargs' key must be present in the 'dist' dictionary and be a dictionary itself."
            )

        if "mu" in dist["kwargs"]:
            raise ValueError(
                "The 'mu' key is not allowed directly within 'kwargs' of the main distribution as it is reserved."
            )

        parameter_distributions = {}
        for param, param_config in dist["kwargs"].items():
            # Check if param_config is a dictionary with a 'dist' key
            if isinstance(param_config, dict) and "dist" in param_config:
                # Prepare nested distribution
                if "kwargs" not in param_config:
                    raise ValueError(
                        f"The parameter configuration for '{param}' must contain 'kwargs'."
                    )

                parameter_distributions[param] = self._get_distribution(
                    dist=param_config
                )(**param_config["kwargs"], name=f"likelihood_{param}")
            elif isinstance(param_config, int | float):
                # Use the value directly
                parameter_distributions[param] = param_config
            else:
                raise ValueError(
                    f"""
                    Invalid parameter configuration for '{param}'.
                    It must be either a dictionary with a 'dist' key or a numeric value.
                    """
                )

        # Extract the likelihood distribution name and instantiate it
        likelihood_dist_name = dist["dist"]
        likelihood_dist = self._get_distribution(dist={"dist": likelihood_dist_name})

        return likelihood_dist(
            name=self.output_var,
            mu=mu,
            observed=observed,
            dims=dims,
            **parameter_distributions,
        )

    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        **kwargs,
    ) -> None:
        """
        Builds a probabilistic model using PyMC for marketing mix modeling.

        The model incorporates channels, control variables, and Fourier components, applying
        adstock and saturation transformations to the channel data. The final model is
        constructed with multiple factors contributing to the response variable.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for the model, which should include columns for channels,
            control variables (if applicable), and Fourier components (if applicable).

        y : Union[pd.Series, np.ndarray]
            The target/response variable for the modeling.

        **kwargs : dict
            Additional keyword arguments that might be required by underlying methods or utilities.

        Attributes Set
        ---------------
        model : pm.Model
            The PyMC model object containing all the defined stochastic and deterministic variables.

        Examples
        --------
        custom_config = {
            'intercept': {'dist': 'Normal', 'kwargs': {'mu': 0, 'sigma': 2}},
            'beta_channel': {'dist': 'LogNormal', 'kwargs': {'mu': 1, 'sigma': 3}},
            'alpha': {'dist': 'Beta', 'kwargs': {'alpha': 1, 'beta': 3}},
            'lam': {'dist': 'Gamma', 'kwargs': {'alpha': 3, 'beta': 1}},
            'likelihood': {'dist': 'Normal',
                'kwargs': {'sigma': {'dist': 'HalfNormal', 'kwargs': {'sigma': 2}}}
            },
            'gamma_control': {'dist': 'Normal', 'kwargs': {'mu': 0, 'sigma': 2}},
            'gamma_fourier': {'dist': 'Laplace', 'kwargs': {'mu': 0, 'b': 1}}
        }

        model = DelayedSaturatedMMM(
                    date_column="date_week",
                    channel_columns=["x1", "x2"],
                    control_columns=[
                        "event_1",
                        "event_2",
                        "t",
                    ],
                    adstock_max_lag=8,
                    yearly_seasonality=2,
                    model_config=custom_config,
                )
        """

        self.intercept_dist = self._get_distribution(
            dist=self.model_config["intercept"]
        )
        self.beta_channel_dist = self._get_distribution(
            dist=self.model_config["beta_channel"]
        )
        self.lam_dist = self._get_distribution(dist=self.model_config["lam"])
        self.alpha_dist = self._get_distribution(dist=self.model_config["alpha"])
        self.gamma_control_dist = self._get_distribution(
            dist=self.model_config["gamma_control"]
        )
        self.gamma_fourier_dist = self._get_distribution(
            dist=self.model_config["gamma_fourier"]
        )

        self._generate_and_preprocess_model_data(X, y)
        with pm.Model(
            coords=self.model_coords,
            coords_mutable=self.coords_mutable,
        ) as self.model:
            channel_data_ = pm.MutableData(
                name="channel_data",
                value=self.preprocessed_data["X"][self.channel_columns],
                dims=("date", "channel"),
            )

            target_ = pm.MutableData(
                name="target",
                value=self.preprocessed_data["y"],
                dims="date",
            )

            if self.time_varying_intercept:
                time_index = pm.Data(
                    "time_index",
                    self._time_index,
                    dims="date",
                )
                intercept = create_time_varying_intercept(
                    time_index,
                    self._time_index_mid,
                    self._time_resolution,
                    self.intercept_dist,
                    self.model_config,
                )
            else:
                intercept = self.intercept_dist(
                    name="intercept", **self.model_config["intercept"]["kwargs"]
                )

            beta_channel = self.beta_channel_dist(
                name="beta_channel",
                **self.model_config["beta_channel"]["kwargs"],
                dims=("channel",),
            )
            alpha = self.alpha_dist(
                name="alpha",
                dims="channel",
                **self.model_config["alpha"]["kwargs"],
            )
            lam = self.lam_dist(
                name="lam",
                dims="channel",
                **self.model_config["lam"]["kwargs"],
            )

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

            channel_contributions_var = channel_adstock_saturated * beta_channel
            channel_contributions = pm.Deterministic(
                name="channel_contributions",
                var=channel_contributions_var,
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
                gamma_control = self.gamma_control_dist(
                    name="gamma_control",
                    dims="control",
                    **self.model_config["gamma_control"]["kwargs"],
                )

                control_data_ = pm.MutableData(
                    name="control_data",
                    value=self.preprocessed_data["X"][self.control_columns],
                    dims=("date", "control"),
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

                gamma_fourier = self.gamma_fourier_dist(
                    name="gamma_fourier",
                    dims="fourier_mode",
                    **self.model_config["gamma_fourier"]["kwargs"],
                )

                fourier_contribution = pm.Deterministic(
                    name="fourier_contributions",
                    var=fourier_data_ * gamma_fourier,
                    dims=("date", "fourier_mode"),
                )

                mu_var += fourier_contribution.sum(axis=-1)

            mu = pm.Deterministic(name="mu", var=mu_var, dims="date")

            self._create_likelihood_distribution(
                dist=self.model_config["likelihood"],
                mu=mu,
                observed=target_,
                dims="date",
            )

    @property
    def default_model_config(self) -> dict:
        return {
            "intercept": {
                "dist": "Normal",
                "kwargs": {"mu": 0, "sigma": 2},
            },
            "beta_channel": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
            "alpha": {"dist": "Beta", "kwargs": {"alpha": 1, "beta": 3}},
            "lam": {"dist": "Gamma", "kwargs": {"alpha": 3, "beta": 1}},
            "likelihood": {
                "dist": "Normal",
                "kwargs": {
                    "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
                },
            },
            "gamma_control": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}},
            "gamma_fourier": {"dist": "Laplace", "kwargs": {"mu": 0, "b": 1}},
            "intercept_tvp_kwargs": {
                "m": 200,
                "L": None,
                "eta_lam": 1,
                "ls_mu": None,
                "ls_sigma": 10,
                "cov_func": None,
            },
        }

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
        periods: npt.NDArray[np.float_] = (
            date_data.dt.dayofyear.to_numpy() / DAYS_IN_YEAR
        )
        return generate_fourier_modes(
            periods=periods,
            n_order=self.yearly_seasonality,
        )

    def channel_contributions_forward_pass(
        self, channel_data: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """Evaluate the channel contribution for a given channel data and a fitted model, ie. the forward pass.

        Parameters
        ----------
        channel_data : array-like
            Input channel data. Result of all the preprocessing steps.

        Returns
        -------
        array-like
            Transformed channel data.
        """
        alpha_posterior = self.fit_result["alpha"].to_numpy()

        lam_posterior = self.fit_result["lam"].to_numpy()
        lam_posterior_expanded = np.expand_dims(a=lam_posterior, axis=2)

        beta_channel_posterior = self.fit_result["beta_channel"].to_numpy()
        beta_channel_posterior_expanded = np.expand_dims(
            a=beta_channel_posterior, axis=2
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

    @property
    def _serializable_model_config(self) -> dict[str, Any]:
        def ndarray_to_list(d: dict) -> dict:
            new_d = d.copy()  # Copy the dictionary to avoid mutating the original one
            for key, value in new_d.items():
                if isinstance(value, np.ndarray):
                    new_d[key] = value.tolist()
                elif isinstance(value, dict):
                    new_d[key] = ndarray_to_list(value)
            return new_d

        serializable_config = self.model_config.copy()
        return ndarray_to_list(serializable_config)

    @classmethod
    def load(cls, fname: str):
        """
        Creates a DelayedSaturatedMMM instance from a file,
        instantiating the model with the saved original input parameters.
        Loads inference data for the model.

        Parameters
        ----------
        fname : string
            This denotes the name with path from where idata should be loaded from.

        Returns
        -------
        Returns an instance of DelayedSaturatedMMM.

        Raises
        ------
        ValueError
            If the inference data that is loaded doesn't match with the model.
        """

        filepath = Path(str(fname))
        idata = az.from_netcdf(filepath)
        model_config = cls._model_config_formatting(
            json.loads(idata.attrs["model_config"])
        )
        model = cls(
            date_column=json.loads(idata.attrs["date_column"]),
            control_columns=json.loads(idata.attrs["control_columns"]),
            channel_columns=json.loads(idata.attrs["channel_columns"]),
            adstock_max_lag=json.loads(idata.attrs["adstock_max_lag"]),
            validate_data=json.loads(idata.attrs["validate_data"]),
            yearly_seasonality=json.loads(idata.attrs["yearly_seasonality"]),
            model_config=model_config,
            sampler_config=json.loads(idata.attrs["sampler_config"]),
        )
        model.idata = idata
        dataset = idata.fit_data.to_dataframe()
        X = dataset.drop(columns=[model.output_var])
        y = dataset[model.output_var].values
        model.build_model(X, y)
        # All previously used data is in idata.
        if model.id != idata.attrs["id"]:
            error_msg = f"""The file '{fname}' does not contain an inference data of the same model
        or configuration as '{cls._model_type}'"""
            raise ValueError(error_msg)

        return model

    def _data_setter(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
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
        if not isinstance(X, pd.DataFrame):
            msg = "X must be a pandas DataFrame in order to access the columns"
            raise TypeError(msg)

        new_channel_data: np.ndarray | None = None
        coords = {"date": X[self.date_column].to_numpy()}

        try:
            new_channel_data = X[self.channel_columns].to_numpy()
        except KeyError as e:
            raise RuntimeError("New data must contain channel_data!") from e

        def identity(x):
            return x

        channel_transformation = (
            identity
            if not hasattr(self, "channel_transformer")
            else self.channel_transformer.transform
        )

        data: dict[str, np.ndarray | Any] = {
            "channel_data": channel_transformation(new_channel_data)
        }
        if self.control_columns is not None:
            control_data = X[self.control_columns].to_numpy()
            control_transformation = (
                identity
                if not hasattr(self, "control_transformer")
                else self.control_transformer.transform
            )
            data["control_data"] = control_transformation(control_data)

        if hasattr(self, "fourier_columns"):
            data["fourier_data"] = self._get_fourier_models_data(X)

        if self.time_varying_intercept:
            data["time_index"] = infer_time_index(
                X[self.date_column], self.X[self.date_column], self._time_resolution
            )

        if y is not None:
            if isinstance(y, pd.Series):
                data["target"] = (
                    y.to_numpy()
                )  # convert Series to numpy array explicitly
            elif isinstance(y, np.ndarray):
                data["target"] = y
            else:
                raise TypeError("y must be either a pandas Series or a numpy array")
        else:
            dtype = self.preprocessed_data["y"].dtype  # type: ignore
            data["target"] = np.zeros(X.shape[0], dtype=dtype)  # type: ignore

        with self.model:
            pm.set_data(data, coords=coords)

    @classmethod
    def _model_config_formatting(cls, model_config: dict) -> dict:
        """
        Because of json serialization, model_config values that were originally tuples
        or numpy are being encoded as lists. This function converts them back to tuples
        and numpy arrays to ensure correct id encoding.
        """

        def format_nested_dict(d: dict) -> dict:
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = format_nested_dict(value)
                elif isinstance(value, list):
                    # Check if the key is "dims" to convert it to tuple
                    if key == "dims":
                        d[key] = tuple(value)
                    # Convert all other lists to numpy arrays
                    else:
                        d[key] = np.array(value)
            return d

        return format_nested_dict(model_config.copy())


class DelayedSaturatedMMM(
    MaxAbsScaleTarget,
    MaxAbsScaleChannels,
    ValidateControlColumns,
    BaseDelayedSaturatedMMM,
):
    """Media Mix Model with delayed adstock and logistic saturation class (see [1]_).

    Given a time series target variable :math:`y_{t}` (e.g. sales on conversions), media variables
    :math:`x_{m, t}` (e.g. impressions, clicks or costs) and a set of control covariates :math:`z_{c, t}` (e.g. holidays, special events)
    we consider a Bayesian linear model of the form:

    .. math::
        y_{t} = \\alpha + \\sum_{m=1}^{M}\\beta_{m}f(x_{m, t}) +  \\sum_{c=1}^{C}\\gamma_{c}z_{c, t} + \\varepsilon_{t},

    where :math:`\\alpha` is the intercept, :math:`f` is a media transformation function and :math:`\\varepsilon_{t}` is the error therm
    which we assume is normally distributed. The function :math:`f` encodes the contribution of media on the target variable.
    Typically we consider two types of transformation: adstock (carry-over) and saturation effects.

    Notes
    -----
    Here are some important notes about the model:

    1. Before fitting the model, we scale the target variable and the media channels using the maximum absolute value of each variable.
    This enable us to have a more stable model and better convergence. If control variables are present, we do not scale them!
    If needed please do it before passing the data to the model.

    2. We allow to add yearly seasonality controls as Fourier modes.
    You can use the `yearly_seasonality` parameter to specify the number of Fourier modes to include.

    3. This class also allow us to calibrate the model using:

        * Custom priors for the parameters via the `model_config` parameter. You can also set the likelihood distribution.

        * Adding lift tests to the likelihood function via the :meth:`add_lift_test_measurements <pymc_marketing.mmm.delayed_saturated_mmm.DelayedSaturatedMMM.add_lift_test_measurements>` method.

    For details on a vanilla implementation in PyMC, see [2]_.

    Examples
    --------
    Here is an example of how to instantiate the model with the default configuration:

    .. code-block:: python

        import numpy as np
        import pandas as pd

        from pymc_marketing.mmm import DelayedSaturatedMMM

        data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/mmm_example.csv"
        data = pd.read_csv(data_url, parse_dates=["date_week"])

        mmm = DelayedSaturatedMMM(
            date_column="date_week",
            channel_columns=["x1", "x2"],
            control_columns=[
                "event_1",
                "event_2",
                "t",
            ],
            adstock_max_lag=8,
            yearly_seasonality=2,
        )

    Now we can fit the model with the data:

    .. code-block:: python

        # Set features and target
        X = data.drop("y", axis=1)
        y = data["y"]

        # Fit the model
        idata = mmm.fit(X, y)

    We can also define custom priors for the model:

    .. code-block:: python

        my_model_config = {
            "beta_channel": {
                "dist": "LogNormal",
                "kwargs": {"mu": np.array([2, 1]), "sigma": 1},
            },
            "likelihood": {
                "dist": "Normal",
                "kwargs": {"sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 2}}},
            },
        }

        mmm = DelayedSaturatedMMM(
            model_config=my_model_config,
            date_column="date_week",
            channel_columns=["x1", "x2"],
            control_columns=[
                "event_1",
                "event_2",
                "t",
            ],
            adstock_max_lag=8,
            yearly_seasonality=2,
        )

    As you can see, we can configure all prior and likelihood distributions via the `model_config`.

    The `fit` method accepts keyword arguments that are passed to the PyMC sampling method.
    For example, to change the number of samples and chains, and using a JAX implementation of NUTS we can do:

    .. code-block:: python

        sampler_kwargs = {
            "draws": 2_000,
            "target_accept": 0.9,
            "chains": 5,
            "random_seed": 42,
        }

        idata = mmm.fit(X, y, nuts_sampler="numpyro", **sampler_kwargs)

    References
    ----------
    .. [1] Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).
    .. [2] Orduz, J. `"Media Effect Estimation with PyMC: Adstock, Saturation & Diminishing Returns" <https://juanitorduz.github.io/pymc_mmm/>`_.
    """  # noqa: E501

    def channel_contributions_forward_pass(
        self, channel_data: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """Evaluate the channel contribution for a given channel data and a fitted model, ie. the forward pass.
        We return the contribution in the original scale of the target variable.

        Parameters
        ----------
        channel_data : array-like
            Input channel data. Result of all the preprocessing steps.
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
    ) -> DataArray:
        """Generate a grid of scaled channel contributions for a given grid of shared values.

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
        DataArray
            Grid of channel contributions.
        """
        if start < 0:
            raise ValueError("start must be greater than or equal to 0.")

        share_grid = np.linspace(start=start, stop=stop, num=num)

        channel_contributions = []
        for delta in share_grid:
            channel_data = (
                delta * self.preprocessed_data["X"][self.channel_columns].to_numpy()
            )
            channel_contribution_forward_pass = self.channel_contributions_forward_pass(
                channel_data=channel_data
            )
            channel_contributions.append(channel_contribution_forward_pass)
        return DataArray(
            data=np.array(channel_contributions),
            dims=("delta", "chain", "draw", "date", "channel"),
            coords={
                "delta": share_grid,
                "date": self.X[self.date_column],
                "channel": self.channel_columns,
            },
        )

    def plot_channel_contributions_grid(
        self,
        start: float,
        stop: float,
        num: int,
        absolute_xrange: bool = False,
        **plt_kwargs: Any,
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
        absolute_xrange : bool, optional
            If True, the x-axis is in absolute values (input units), otherwise it is in
            relative percentage values, by default False.

        Returns
        -------
        plt.Figure
            Plot of grid of channel contributions.
        """
        share_grid = np.linspace(start=start, stop=stop, num=num)
        contributions = self.get_channel_contributions_forward_pass_grid(
            start=start, stop=stop, num=num
        )

        fig, ax = plt.subplots(**plt_kwargs)

        for i, channel in enumerate(self.channel_columns):
            channel_contribution_total = contributions.sel(channel=channel).sum(
                dim="date"
            )

            hdi_contribution = az.hdi(ary=channel_contribution_total).x

            total_channel_input = self.X[channel].sum()
            x_range = (
                total_channel_input * share_grid if absolute_xrange else share_grid
            )

            ax.fill_between(
                x=x_range,
                y1=hdi_contribution[:, 0],
                y2=hdi_contribution[:, 1],
                color=f"C{i}",
                label=f"{channel} $94\%$ HDI contribution",  # noqa: W605
                alpha=0.4,
            )

            sns.lineplot(
                x=x_range,
                y=channel_contribution_total.mean(dim=("chain", "draw")),
                color=f"C{i}",
                marker="o",
                label=f"{channel} contribution mean",
                ax=ax,
            )
            if absolute_xrange:
                ax.axvline(
                    x=total_channel_input,
                    color=f"C{i}",
                    linestyle="--",
                    label=f"{channel} current total input",
                )

        if not absolute_xrange:
            ax.axvline(x=1, color="black", linestyle="--", label=r"$\delta = 1$")

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        x_label = "input" if absolute_xrange else r"$\delta$"
        ax.set(
            title="Channel contribution as a function of cost share",
            xlabel=x_label,
            ylabel="contribution",
        )
        return fig

    def new_spend_contributions(
        self,
        spend: np.ndarray | None = None,
        one_time: bool = True,
        spend_leading_up: np.ndarray | None = None,
        prior: bool = False,
        original_scale: bool = True,
        **sample_posterior_predictive_kwargs,
    ) -> DataArray:
        """Return the upcoming contributions for a given spend.

        The spend can be one time or constant over the period. The spend leading up to the
        period can also be specified in order account for the lagged effect of the spend.

        Parameters
        ----------
        spend : np.ndarray, optional
            Array of spend for each channel. If None, the average spend for each channel is used, by default None.
        one_time : bool, optional
            Whether the spends for each channel are only at the start of the period.
            If True, all spends after the initial spend are zero.
            If False, all spends after the initial spend are the same as the initial spend.
            By default True.
        spend_leading_up : np.array, optional
            Array of spend for each channel leading up to the spend, by default None or 0 for each channel.
            Use this parameter to account for the lagged effect of the spend.
        prior : bool, optional
            Whether to use the prior or posterior, by default False (posterior)
        **sample_posterior_predictive_kwargs
            Additional keyword arguments passed to pm.sample_posterior_predictive

        Returns
        -------
        DataArray
            Upcoming contributions for each channel

        Examples
        --------
        Channel contributions from 1 unit on each channel only once.

        .. code-block:: python

            n_channels = len(model.channel_columns)
            spend = np.ones(n_channels)
            new_spend_contributions = model.new_spend_contributions(spend=spend)

        Channel contributions from continuously spending 1 unit on each channel.

        .. code-block:: python

            n_channels = len(model.channel_columns)
            spend = np.ones(n_channels)
            new_spend_contributions = model.new_spend_contributions(spend=spend, one_time=False)

        Channel contributions from 1 unit on each channel only once but with 1 unit leading up to the spend.

        .. code-block:: python

            n_channels = len(model.channel_columns)
            spend = np.ones(n_channels)
            spend_leading_up = np.ones(n_channels)
            new_spend_contributions = model.new_spend_contributions(spend=spend, spend_leading_up=spend_leading_up)
        """
        if spend is None:
            spend = self.X.loc[:, self.channel_columns].mean().to_numpy()  # type: ignore

        n_channels = len(self.channel_columns)
        if len(spend) != n_channels:
            raise ValueError("spend must be the same length as the number of channels")

        new_data = create_new_spend_data(
            spend=spend,
            adstock_max_lag=self.adstock_max_lag,
            one_time=one_time,
            spend_leading_up=spend_leading_up,
        )

        new_data = (
            self.channel_transformer.transform(new_data) if not prior else new_data
        )

        idata: Dataset = self.fit_result if not prior else self.prior

        coords = {
            "time_since_spend": np.arange(
                -self.adstock_max_lag, self.adstock_max_lag + 1
            ),
            "channel": self.channel_columns,
        }
        with pm.Model(coords=coords):
            alpha = pm.Uniform("alpha", lower=0, upper=1, dims=("channel",))
            lam = pm.HalfFlat("lam", dims=("channel",))
            beta_channel = pm.HalfFlat("beta_channel", dims=("channel",))

            # Same as the forward pass of the model
            channel_adstock = geometric_adstock(
                x=new_data,
                alpha=alpha,
                l_max=self.adstock_max_lag,
                normalize=True,
                axis=0,
            )
            channel_adstock_saturated = logistic_saturation(x=channel_adstock, lam=lam)
            pm.Deterministic(
                name="channel_contributions",
                var=channel_adstock_saturated * beta_channel,
                dims=("time_since_spend", "channel"),
            )

            samples = pm.sample_posterior_predictive(
                idata,
                var_names=["channel_contributions"],
                **sample_posterior_predictive_kwargs,
            )

        channel_contributions = samples.posterior_predictive["channel_contributions"]

        if original_scale:
            channel_contributions = apply_sklearn_transformer_across_dim(
                data=channel_contributions,
                func=self.get_target_transformer().inverse_transform,
                dim_name="time_since_spend",
                combined=False,
            )

        return channel_contributions

    def plot_new_spend_contributions(
        self,
        spend_amount: float,
        one_time: bool = True,
        lower: float = 0.025,
        upper: float = 0.975,
        ylabel: str = "Sales",
        idx: slice | None = None,
        channels: list[str] | None = None,
        prior: bool = False,
        original_scale: bool = True,
        ax: plt.Axes | None = None,
        **sample_posterior_predictive_kwargs,
    ) -> plt.Axes:
        """Plot the upcoming sales for a given spend amount.

        Calls the new_spend_contributions method and plots the results. For more
        control over the plot, use new_spend_contributions directly.

        Parameters
        ----------
        spend_amount : float
            The amount of spend for each channel
        one_time : bool, optional
            Whether the spend are one time (at start of period) or constant (over period), by default True (one time)
        lower : float, optional
            The lower quantile for the confidence interval, by default 0.025
        upper : float, optional
            The upper quantile for the confidence interval, by default 0.975
        ylabel : str, optional
            The label for the y-axis, by default "Sales"
        idx : slice, optional
            The index slice of days to plot, by default None or only the positive days.
            More specifically, slice(0, None, None)
        channels : List[str], optional
            The channels to plot, by default None or all channels
        prior : bool, optional
            Whether to use the prior or posterior, by default False (posterior)
        original_scale : bool, optional
            Whether to plot in the original scale of the target variable, by default True
        ax : plt.Axes, optional
            The axes to plot on, by default None or current axes
        **sample_posterior_predictive_kwargs
            Additional keyword arguments passed to pm.sample_posterior_predictive

        Returns
        -------
        plt.Axes
            The plot of upcoming sales for the given spend amount

        """
        for value in [lower, upper]:
            if value < 0 or value > 1:
                raise ValueError("lower and upper must be between 0 and 1")
        if lower > upper:
            raise ValueError("lower must be less than or equal to upper")

        ax = ax or plt.gca()
        total_channels = len(self.channel_columns)
        contributions = self.new_spend_contributions(
            np.ones(total_channels) * spend_amount,
            one_time=one_time,
            spend_leading_up=np.ones(total_channels) * spend_amount,
            prior=prior,
            original_scale=original_scale,
            **sample_posterior_predictive_kwargs,
        )

        contributions_groupby = contributions.to_series().groupby(
            level=["time_since_spend", "channel"]
        )

        idx = idx or pd.IndexSlice[0:]

        conf = (
            contributions_groupby.quantile([lower, upper])
            .unstack("channel")
            .unstack()
            .loc[idx]
        )

        channels = channels or self.channel_columns  # type: ignore
        for channel in channels:  # type: ignore
            ax.fill_between(
                conf.index,
                conf[channel][lower],
                conf[channel][upper],
                label=f"{channel} {100 * (upper - lower):.0f}% CI",
                alpha=0.5,
            )
        mean = contributions_groupby.mean().unstack("channel").loc[idx, channels]
        color = [f"C{i}" for i in range(len(channels))]  # type: ignore
        mean.add_suffix(" mean").plot(ax=ax, color=color, alpha=0.75)
        ax.legend().set_title("Channel")
        ax.set(
            xlabel="Time since spend",
            ylabel=ylabel,
            title=f"Upcoming sales for {spend_amount:.02f} spend",
        )
        return ax

    def _validate_data(self, X, y=None):
        return X

    def sample_posterior_predictive(
        self,
        X_pred,
        extend_idata: bool = True,
        combined: bool = True,
        include_last_observations: bool = False,
        original_scale: bool = True,
        **sample_posterior_predictive_kwargs,
    ):
        """
        Sample from the model's posterior predictive distribution.

        Parameters
        ---------
        X_pred : array, shape (n_pred, n_features)
            The input data used for prediction.
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        include_last_observations: Boolean determining whether to include the last observations of the training
            data in order to carry over costs with the adstock transformation.
            Assumes that X_pred are the next predictions following the training data.
            Defaults to False.
        original_scale: Boolean determining whether to return the predictions in the original scale
            of the target variable. Defaults to True.
        **sample_posterior_predictive_kwargs: Additional arguments to pass to pymc.sample_posterior_predictive

        Returns
        -------
        posterior_predictive_samples : DataArray, shape (n_pred, samples)
            Posterior predictive samples for each input X_pred
        """
        if include_last_observations:
            X_pred = pd.concat(
                [self.X.iloc[-self.adstock_max_lag :, :], X_pred], axis=0
            ).sort_values(by=self.date_column)

        self._data_setter(X_pred)

        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(
                self.idata, **sample_posterior_predictive_kwargs
            )
            if extend_idata:
                self.idata.extend(post_pred, join="right")  # type: ignore

        posterior_predictive_samples = az.extract(
            post_pred, "posterior_predictive", combined=combined
        )

        if include_last_observations:
            posterior_predictive_samples = posterior_predictive_samples.isel(
                date=slice(self.adstock_max_lag, None)
            )

        if original_scale:
            posterior_predictive_samples = apply_sklearn_transformer_across_dim(
                data=posterior_predictive_samples,
                func=self.get_target_transformer().inverse_transform,
                dim_name="date",
                combined=combined,
            )

        return posterior_predictive_samples

    def add_lift_test_measurements(
        self,
        df_lift_test: pd.DataFrame,
        dist: pm.Distribution = pm.Gamma,
        name: str = "lift_measurements",
    ) -> None:
        """Add lift tests to the model.

        The model difference of a channel's saturation curve is created
        from `x` and `x + delta_x` for each channel. This random variable is
        then conditioned using the empirical lift, `delta_y`, and `sigma` of the lift test
        with the specified distribution `dist`.

        The pseudo-code for the lift test is as follows:

        .. code-block:: python

            model_estimated_lift = (
                saturation_curve(x + delta_x)
                - saturation_curve(x)
            )
            empirical_lift = delta_y
            dist(abs(model_estimated_lift), sigma=sigma, observed=abs(empirical_lift))


        The model has to be built before adding the lift tests.

        Parameters
        ----------
        df_lift_test : pd.DataFrame
            DataFrame with lift test results with at least the following columns:
                * `channel`: channel name. Must be present in `channel_columns`.
                * `x`: x axis value of the lift test.
                * `delta_x`: change in x axis value of the lift test.
                * `delta_y`: change in y axis value of the lift test.
                * `sigma`: standard deviation of the lift test.
        dist : pm.Distribution, optional
            The distribution to use for the likelihood, by default pm.Gamma
        name : str, optional
            The name of the likelihood of the lift test contribution(s),
            by default "lift_measurements". Name change required if calling
            this method multiple times.

        Raises
        ------
        RuntimeError
            If the model has not been built yet.
        KeyError
            If the 'channel' column is not present in df_lift_test.

        Examples
        --------
        Build the model first then add lift test measurements.

        .. code-block:: python

            model = DelayedSaturatedMMM(
                date_column="date_week",
                channel_columns=["x1", "x2"],
                control_columns=[
                    "event_1",
                    "event_2",
                ],
                adstock_max_lag=8,
                yearly_seasonality=2,
            )

            X: pd.DataFrame = ...
            y: np.ndarray = ...

            model.build_model(X, y)

            df_lift_test = pd.DataFrame({
                "channel": ["x1", "x1"],
                "x": [1, 1],
                "delta_x": [0.1, 0.2],
                "delta_y": [0.1, 0.1],
                "sigma": [0.1, 0.1],
            })

            model.add_lift_test_measurements(df_lift_test)

        """
        if self.model is None:
            raise RuntimeError(
                "The model has not been built yet. Please, build the model first."
            )

        if "channel" not in df_lift_test.columns:
            raise KeyError(
                "The 'channel' column is required to map the lift measurements to the model."
            )

        df_lift_test_scaled = scale_lift_measurements(
            df_lift_test=df_lift_test,
            channel_col="channel",
            channel_columns=self.channel_columns,  # type: ignore
            channel_transform=self.channel_transformer.transform,
            target_transform=self.target_transformer.transform,
        )
        with self.model:
            add_logistic_empirical_lift_measurements_to_likelihood(
                df_lift_test=df_lift_test_scaled,
                # Based on the model
                lam_name="lam",
                beta_name="beta_channel",
                dist=dist,
                name=name,
            )
