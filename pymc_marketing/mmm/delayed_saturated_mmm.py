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
"""Media Mix Model class."""

import json
import warnings
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
from xarray import DataArray, Dataset

from pymc_marketing.constants import DAYS_IN_YEAR
from pymc_marketing.mmm.base import BaseValidateMMM
from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer
from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
    _get_adstock_function,
)
from pymc_marketing.mmm.components.saturation import (
    SaturationTransformation,
    _get_saturation_function,
)
from pymc_marketing.mmm.lift_test import (
    add_lift_measurements_to_likelihood_from_saturation,
    scale_lift_measurements,
)
from pymc_marketing.mmm.preprocessing import MaxAbsScaleChannels, MaxAbsScaleTarget
from pymc_marketing.mmm.tvp import create_time_varying_gp_multiplier, infer_time_index
from pymc_marketing.mmm.utils import (
    apply_sklearn_transformer_across_dim,
    create_new_spend_data,
    generate_fourier_modes,
)
from pymc_marketing.mmm.validating import ValidateControlColumns
from pymc_marketing.model_config import (
    create_distribution_from_config,
    create_likelihood_distribution,
    get_distribution,
)

__all__ = ["BaseMMM", "MMM", "DelayedSaturatedMMM"]


class BaseMMM(BaseValidateMMM):
    """
    Base class for a media mix model using Delayed Adstock and Logistic Saturation (see [1]_).

    References
    ----------
    .. [1] Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).
    """

    _model_name: str = "BaseMMM"
    _model_type: str = "BaseValidateMMM"
    version: str = "0.0.3"

    def __init__(
        self,
        date_column: str,
        channel_columns: list[str],
        adstock_max_lag: int,
        adstock: str | AdstockTransformation,
        saturation: str | SaturationTransformation,
        time_varying_intercept: bool = False,
        time_varying_media: bool = False,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        validate_data: bool = True,
        control_columns: list[str] | None = None,
        yearly_seasonality: int | None = None,
        adstock_first: bool = True,
        **kwargs,
    ) -> None:
        """Constructor method.

        Parameters
        ----------
        date_column : str
            Column name of the date variable.
        channel_columns : List[str]
            Column names of the media channel variables.
        adstock_max_lag : int, optional
            Number of lags to consider in the adstock transformation, by default 4
        adstock : str | AdstockTransformation
            Type of adstock transformation to apply.
        saturation : str | SaturationTransformation
            Type of saturation transformation to apply.
        time_varying_intercept : bool, optional
            Whether to consider time-varying intercept, by default False.
            Because the `time-varying` variable is centered around 1 and acts as a multiplier,
            the variable `baseline_intercept` now represents the mean of the time-varying intercept.
        time_varying_media : bool, optional
            Whether to consider time-varying media contributions, by default False.
            The `time-varying-media` creates a time media variable centered around 1,
            this variable acts as a global multiplier (scaling factor) for all channels,
            meaning all media channels share the same latent fluctiation.
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
        yearly_seasonality : Optional[int], optional
            Number of Fourier modes to model yearly seasonality, by default None.
        """
        self.control_columns = control_columns
        self.adstock_max_lag = adstock_max_lag
        self.time_varying_intercept = time_varying_intercept
        self.time_varying_media = time_varying_media
        self.yearly_seasonality = yearly_seasonality
        self.date_column = date_column
        self.validate_data = validate_data

        self.adstock_first = adstock_first
        self.adstock = _get_adstock_function(function=adstock, l_max=adstock_max_lag)
        self.saturation = _get_saturation_function(function=saturation)

        if model_config is not None:
            self.adstock.update_priors({**self.default_model_config, **model_config})
            self.saturation.update_priors({**self.default_model_config, **model_config})

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

        if self.time_varying_intercept | self.time_varying_media:
            self._time_index = np.arange(0, X.shape[0])
            self._time_index_mid = X.shape[0] // 2
            self._time_resolution = (
                self.X[self.date_column].iloc[1] - self.X[self.date_column].iloc[0]
            ).days

    def _save_input_params(self, idata) -> None:
        """Saves input parameters to the attrs of idata."""
        idata.attrs["date_column"] = json.dumps(self.date_column)
        idata.attrs["adstock"] = json.dumps(self.adstock.lookup_name)
        idata.attrs["saturation"] = json.dumps(self.saturation.lookup_name)
        idata.attrs["adstock_first"] = json.dumps(self.adstock_first)
        idata.attrs["control_columns"] = json.dumps(self.control_columns)
        idata.attrs["channel_columns"] = json.dumps(self.channel_columns)
        idata.attrs["adstock_max_lag"] = json.dumps(self.adstock_max_lag)
        idata.attrs["validate_data"] = json.dumps(self.validate_data)
        idata.attrs["yearly_seasonality"] = json.dumps(self.yearly_seasonality)

    def forward_pass(
        self, x: pt.TensorVariable | npt.NDArray[np.float64]
    ) -> pt.TensorVariable:
        """Transforms channel input into target contributions of each channel.

        This method handles the ordering of the adstock and saturation
        transformations.

        This method must be called from without a pm.Model context but not
        necessarily in the instance's model. A dim named "channel" is required
        associated with the number of columns of `x`.

        Parameters
        ------------
        x : pt.TensorVariable | npt.NDArray[np.float64]
            The channel input which could be spends or impressions

        Returns
        --------
        The contributions associated with the channel input
        """
        first, second = (
            (self.adstock, self.saturation)
            if self.adstock_first
            else (self.saturation, self.adstock)
        )

        return second.apply(x=first.apply(x=x, dims="channel"), dims="channel")

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
            'saturation_beta': {'dist': 'Gamma', 'kwargs': {'mu': 1, 'sigma': 3}},
            'saturation_lambda': {'dist': 'Beta', 'kwargs': {'alpha': 3, 'beta': 1}},
            'adstock_alpha': {'dist': 'Beta', 'kwargs': {'alpha': 1, 'beta': 3}},
            'likelihood': {'dist': 'Normal',
                'kwargs': {'sigma': {'dist': 'HalfNormal', 'kwargs': {'sigma': 2}}}
            },
            'gamma_control': {'dist': 'Normal', 'kwargs': {'mu': 0, 'sigma': 2}},
            'gamma_fourier': {'dist': 'Laplace', 'kwargs': {'mu': 0, 'b': 1}}
        }

        model = MMM(
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

        self._generate_and_preprocess_model_data(X, y)
        with pm.Model(
            coords=self.model_coords,
            coords_mutable=self.coords_mutable,
        ) as self.model:
            channel_data_ = pm.Data(
                name="channel_data",
                value=self.preprocessed_data["X"][self.channel_columns],
                dims=("date", "channel"),
                mutable=True,
            )

            target_ = pm.Data(
                name="target",
                value=self.preprocessed_data["y"],
                dims="date",
                mutable=True,
            )
            if self.time_varying_intercept | self.time_varying_media:
                time_index = pm.Data(
                    "time_index",
                    self._time_index,
                    dims="date",
                )

            if self.time_varying_intercept:
                intercept_distribution = get_distribution(
                    name=self.model_config["intercept"]["dist"]
                )
                baseline_intercept = intercept_distribution(
                    name="baseline_intercept",
                    **self.model_config["intercept"]["kwargs"],
                )

                intercept_latent_process = create_time_varying_gp_multiplier(
                    name="intercept",
                    dims="date",
                    time_index=time_index,
                    time_index_mid=self._time_index_mid,
                    time_resolution=self._time_resolution,
                    model_config=self.model_config,
                )
                intercept = pm.Deterministic(
                    name="intercept",
                    var=baseline_intercept * intercept_latent_process,
                    dims="date",
                )
            else:
                intercept = create_distribution_from_config(
                    name="intercept", config=self.model_config
                )

            if self.time_varying_media:
                baseline_channel_contributions = pm.Deterministic(
                    name="baseline_channel_contributions",
                    var=self.forward_pass(x=channel_data_),
                    dims=("date", "channel"),
                )

                media_latent_process = create_time_varying_gp_multiplier(
                    name="media",
                    dims="date",
                    time_index=time_index,
                    time_index_mid=self._time_index_mid,
                    time_resolution=self._time_resolution,
                    model_config=self.model_config,
                )
                channel_contributions = pm.Deterministic(
                    name="channel_contributions",
                    var=baseline_channel_contributions * media_latent_process[:, None],
                    dims=("date", "channel"),
                )

            else:
                channel_contributions = pm.Deterministic(
                    name="channel_contributions",
                    var=self.forward_pass(x=channel_data_),
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
                if self.model_config["gamma_control"].get("dims") != "control":
                    self.model_config["gamma_control"]["dims"] = "control"

                gamma_control = create_distribution_from_config(
                    name="gamma_control",
                    config=self.model_config,
                )

                control_data_ = pm.Data(
                    name="control_data",
                    value=self.preprocessed_data["X"][self.control_columns],
                    dims=("date", "control"),
                    mutable=True,
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
                fourier_data_ = pm.Data(
                    name="fourier_data",
                    value=self.preprocessed_data["X"][self.fourier_columns],
                    dims=("date", "fourier_mode"),
                    mutable=True,
                )

                if self.model_config["gamma_fourier"].get("dims") != "fourier_mode":
                    self.model_config["gamma_fourier"]["dims"] = "fourier_mode"

                gamma_fourier = create_distribution_from_config(
                    name="gamma_fourier",
                    config=self.model_config,
                )

                fourier_contribution = pm.Deterministic(
                    name="fourier_contributions",
                    var=fourier_data_ * gamma_fourier,
                    dims=("date", "fourier_mode"),
                )

                yearly_seasonality_contribution = pm.Deterministic(
                    name="yearly_seasonality_contribution",
                    var=fourier_contribution.sum(axis=-1),
                    dims=("date"),
                )

                mu_var += yearly_seasonality_contribution

            mu = pm.Deterministic(name="mu", var=mu_var, dims="date")

            create_likelihood_distribution(
                name=self.output_var,
                param_config=self.model_config["likelihood"],
                mu=mu,
                observed=target_,
                dims="date",
            )

    @property
    def default_model_config(self) -> dict:
        base_config: dict[str, Any] = {
            "intercept": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}},
            "likelihood": {
                "dist": "Normal",
                "kwargs": {
                    "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
                },
            },
            "gamma_control": {
                "dist": "Normal",
                "kwargs": {"mu": 0, "sigma": 2},
                "dims": "control",
            },
            "gamma_fourier": {
                "dist": "Laplace",
                "kwargs": {"mu": 0, "b": 1},
                "dims": "fourier_mode",
            },
        }

        if self.time_varying_intercept:
            base_config["intercept_tvp_config"] = {
                "m": 200,
                "L": None,
                "eta_lam": 1,
                "ls_mu": None,
                "ls_sigma": 10,
                "cov_func": None,
            }
        if self.time_varying_media:
            base_config["media_tvp_config"] = {
                "m": 200,
                "L": None,
                "eta_lam": 1,
                "ls_mu": None,
                "ls_sigma": 10,
                "cov_func": None,
            }

        for media_transform in [self.adstock, self.saturation]:
            for config in media_transform.function_priors.values():
                if "dims" not in config:
                    config["dims"] = "channel"

        return {
            **base_config,
            **self.adstock.model_config,
            **self.saturation.model_config,
        }

    def _get_fourier_models_data(self, X) -> pd.DataFrame:
        """Generates fourier modes to model seasonality.

        Parameters
        ----------
        X : Union[pd.DataFrame, pd.Series], shape (n_obs, n_features)
            Input data for the model. To generate the Fourier modes, it must contain a date column.

        Returns
        -------
        pd.DataFrame
            Fourier modes (sin and cos with different frequencies) as columns in a dataframe.

        References
        ----------
        https://www.pymc.io/projects/examples/en/latest/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html
        """
        if self.yearly_seasonality is None:
            raise ValueError("yearly_seasonality must be specified.")
        date_data: pd.Series = pd.to_datetime(
            arg=X[self.date_column], format="%Y-%m-%d"
        )
        periods: npt.NDArray[np.float64] = (
            date_data.dt.dayofyear.to_numpy() / DAYS_IN_YEAR
        )
        return generate_fourier_modes(
            periods=periods,
            n_order=self.yearly_seasonality,
        )

    def channel_contributions_forward_pass(
        self, channel_data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
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
        coords = {
            **self.model_coords,
            **self.coords_mutable,
        }
        with pm.Model(coords=coords):
            pm.Deterministic(
                "channel_contributions",
                self.forward_pass(x=channel_data),
                dims=("date", "channel"),
            )

            idata = pm.sample_posterior_predictive(
                self.fit_result,
                var_names=["channel_contributions"],
                progressbar=False,
            )

        return idata.posterior_predictive.channel_contributions.to_numpy()

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
        Creates a MMM instance from a file,
        instantiating the model with the saved original input parameters.
        Loads inference data for the model.

        Parameters
        ----------
        fname : string
            This denotes the name with path from where idata should be loaded from.

        Returns
        -------
        Returns an instance of MMM.

        Raises
        ------
        ValueError
            If the inference data that is loaded doesn't match with the model.
        """

        filepath = Path(fname)
        idata = az.from_netcdf(filepath)
        model_config = cls._model_config_formatting(
            json.loads(idata.attrs["model_config"])
        )
        model = cls(
            date_column=json.loads(idata.attrs["date_column"]),
            control_columns=json.loads(idata.attrs["control_columns"]),
            channel_columns=json.loads(idata.attrs["channel_columns"]),
            adstock_max_lag=json.loads(idata.attrs["adstock_max_lag"]),
            adstock=json.loads(idata.attrs.get("adstock", "geometric")),
            saturation=json.loads(idata.attrs.get("saturation", "logistic")),
            adstock_first=json.loads(idata.attrs.get("adstock_first", True)),
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

        if self.time_varying_intercept | self.time_varying_media:
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


class MMM(
    MaxAbsScaleTarget,
    MaxAbsScaleChannels,
    ValidateControlColumns,
    BaseMMM,
):
    """
    Media Mix Model class, Delayed Adstock and logistic saturation as default initialization (see [1]_).

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

        from pymc_marketing.mmm import MMM

        data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/mmm_example.csv"
        data = pd.read_csv(data_url, parse_dates=["date_week"])

        mmm = MMM(
            date_column="date_week",
            adstock="geometric",
            saturation="logistic",
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

        mmm = MMM(
            adstock="geometric",
            saturation="logistic",
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

    _model_type = "MMM"
    version = "0.0.1"

    def channel_contributions_forward_pass(
        self, channel_data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
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

    def plot_channel_parameter(self, param_name: str, **plt_kwargs: Any) -> plt.Figure:
        """
        Plot the posterior distribution of a specific parameter for each channel.

        Parameters:
        ----------
        param_name : str
            The name of the parameter to plot.
        **plt_kwargs : Any
            Additional keyword arguments to pass to the `plt.subplots` function.

        Returns:
        -------
        plt.Figure
            The matplotlib Figure object containing the plot.

        Raises:
        ------
        ValueError
            If the specified parameter name is invalid or not found in the model
            saturation or adstock function.
        """
        saturation: SaturationTransformation = self.saturation
        adstock: AdstockTransformation = self.adstock

        parameters_to_check = list(saturation.variable_mapping.values()) + list(
            adstock.variable_mapping.values()
        )
        if param_name not in parameters_to_check:
            raise ValueError(
                f"Invalid parameter name: {param_name}. Choose from {parameters_to_check}"
            )

        param_samples_df = pd.DataFrame(
            data=az.extract(data=self.fit_result, var_names=[param_name]).T,
            columns=self.channel_columns,
        )

        fig, ax = plt.subplots(**plt_kwargs)
        sns.violinplot(data=param_samples_df, orient="h", ax=ax)
        ax.set(
            title=f"Posterior Distribution: {param_name} Parameter",
            xlabel=param_name,
            ylabel="channel",
        )
        return fig

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
        **plt_kwargs
            Keyword arguments to pass to `plt.subplots()`

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
                label=f"{channel} $94\\%$ HDI contribution",
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
            adstock_max_lag=self.adstock.l_max,
            one_time=one_time,
            spend_leading_up=spend_leading_up,
        )

        new_data = (
            self.channel_transformer.transform(new_data) if not prior else new_data
        )

        idata: Dataset = self.fit_result if not prior else self.prior

        coords = {
            "time_since_spend": np.arange(-self.adstock.l_max, self.adstock.l_max + 1),
            "channel": self.channel_columns,
        }
        with pm.Model(coords=coords):
            pm.Deterministic(
                "channel_contributions",
                self.forward_pass(x=new_data),
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

    def _channel_map_scales(self) -> dict:
        return dict(
            zip(
                self.channel_columns,
                self.channel_transformer["scaler"].scale_,
                strict=False,
            )
        )

    def format_recovered_transformation_parameters(
        self, quantile: float = 0.5
    ) -> dict[str, dict[str, dict[str, float]]]:
        """
        Format the recovered transformation parameters for each channel.

        This function retrieves the quantile of the parameters for each channel and formats them into a dictionary
        containing the channel name, the saturation parameters, and the adstock parameters.

        Parameters
        ----------
        quantile : float, optional
            The quantile to retrieve from the posterior distribution of the parameters. Default is 0.5.

        Returns
        -------
        dict
            A dictionary containing the channel names as keys and the corresponding saturation and adstock parameters
            as values.

        Example
        -------
        >>> self.format_recovered_transformation_parameters(quantile=.5)
        >>> Output:
        {
            'x1': {
                'saturation_params': {
                    'lam': 2.4761893929757077,
                    'beta': 0.360226791880304
                },
            'adstock_params': {
                'alpha': 0.39910387900504796
                }
            },
            'x2': {
                'saturation_params': {
                    'lam': 2.6485978655163436,
                    'beta': 0.2399381337197204
                },
            'adstock_params': {
                'alpha': 0.18859423763437405
                }
            }
        }
        """
        # Retrieve channel names
        channels = self.fit_result.channel.values

        # Initialize the dictionary to store channel information
        channels_info = {}

        # Define the parameter groups for consolidation
        param_groups = {
            "saturation_params": self.saturation.model_config.keys(),
            "adstock_params": self.adstock.model_config.keys(),
        }

        # Iterate through each channel to fetch and store parameters
        for channel in channels:
            channel_info = {}

            # Process each group of parameters (saturation and adstock)
            for group_name, params in param_groups.items():
                # Build dictionary for the current group of parameters
                param_dict = {
                    param.replace(group_name.split("_")[0] + "_", ""): self.fit_result[
                        param
                    ]
                    .quantile(quantile, dim=["chain", "draw"])
                    .to_pandas()
                    .to_dict()[channel]
                    for param in params
                    if param in self.fit_result
                }
                channel_info[group_name] = param_dict

            channels_info[channel] = channel_info

        return channels_info

    def _plot_response_curve_fit(
        self,
        ax: plt.Axes,
        channel: str,
        color_index: int,
        xlim_max: int | None,
        label: str = "Fit Curve",
        quantile_lower: float = 0.05,
        quantile_upper: float = 0.95,
    ) -> None:
        """
        Plot the curve fit for the given channel based on the estimation of the parameters by the model.

        Parameters
        ----------
        ax : plt.Axes
            The matplotlib axes object where the plot should be drawn.
        channel : str
            The name of the channel for which the curve fit is being plotted.
        color_index : int
            An index used for color selection to ensure distinct colors for multiple plots.
        xlim_max: int
            The maximum value to be plot on the X-axis
        label: str
            The label for the curve being plotted, default is "Fit Curve".
        quantile_lower: float
            The lower quantile for parameter estimation, default is 0.05.
        quantile_upper: float
            The upper quantile for parameter estimation, default is 0.95.

        Returns
        -------
        None
            The function modifies the given axes object in-place and doesn't return any object.
        """

        if self.X is not None:
            x_mean = np.max(self.X[channel])

        # Set x_limit based on the method or xlim_max
        if xlim_max is not None:
            x_limit = xlim_max
        else:
            x_limit = x_mean

        # Generate x_fit and y_fit
        x_fit = np.linspace(0, x_limit, 1000)
        upper_params = self.format_recovered_transformation_parameters(
            quantile=quantile_upper
        )
        lower_params = self.format_recovered_transformation_parameters(
            quantile=quantile_lower
        )
        mid_params = self.format_recovered_transformation_parameters(quantile=0.5)
        y_fit = self.saturation.function(
            x=x_fit, **mid_params[channel]["saturation_params"]
        ).eval()

        y_fit_lower = self.saturation.function(
            x=x_fit, **lower_params[channel]["saturation_params"]
        ).eval()
        y_fit_upper = self.saturation.function(
            x=x_fit, **upper_params[channel]["saturation_params"]
        ).eval()

        # scale all y fit values to the original scale using
        # `mmm.target_transformer.named_steps["scaler"].scale_.item()`
        y_fit = (
            self.get_target_transformer()
            .inverse_transform(y_fit.reshape(-1, 1))
            .flatten()
        )
        y_fit_lower = (
            self.get_target_transformer()
            .inverse_transform(y_fit_lower.reshape(-1, 1))
            .flatten()
        )
        y_fit_upper = (
            self.get_target_transformer()
            .inverse_transform(y_fit_upper.reshape(-1, 1))
            .flatten()
        )

        # scale x fit values
        x_fit = self._channel_map_scales()[channel] * x_fit

        ax.fill_between(
            x_fit, y_fit_lower, y_fit_upper, color=f"C{color_index}", alpha=0.25
        )
        ax.plot(x_fit, y_fit, color=f"C{color_index}", label=label, alpha=0.6)

        ax.set(xlabel="Spent", ylabel="Contribution")
        ax.legend()

    def plot_direct_contribution_curves(
        self,
        show_fit: bool = False,
        same_axes: bool = False,
        xlim_max: int | None = None,
        channels: list[str] | None = None,
        quantile_lower: float = 0.05,
        quantile_upper: float = 0.95,
    ) -> plt.Figure:
        """
        Plots the direct contribution curves for each marketing channel. The term "direct" refers to the fact
        we plot costs vs immediate returns and we do not take into account the lagged
        effects of the channels e.g. adstock transformations.

        Parameters
        ----------
        show_fit : bool, optional
            If True, the function will also plot the curve fit based on the specified method. Defaults to False.
        xlim_max : int, optional
            The maximum value to be plot on the X-axis. If not provided, the maximum value in the data will be used.
        channels : List[str], optional
            A list of channels to plot. If not provided, all channels will be plotted.
        same_axes : bool, optional
            If True, all channels will be plotted on the same axes. Defaults to False.

        Returns
        -------
        plt.Figure
            A matplotlib Figure object with the direct contribution curves.
        """
        channels_to_plot = self.channel_columns if channels is None else channels

        if not all(channel in self.channel_columns for channel in channels_to_plot):
            unknown_channels = set(channels_to_plot) - set(self.channel_columns)
            raise ValueError(
                f"The provided channels must be a subset of the available channels. Got {unknown_channels}"
            )

        if len(channels_to_plot) != len(set(channels_to_plot)):
            raise ValueError("The provided channels must be unique.")

        channel_contributions = self.compute_channel_contribution_original_scale().mean(
            ["chain", "draw"]
        )

        if same_axes:
            nrows = 1
            figsize = (12, 4)

            def label_func(channel):
                return f"{channel} Data Points"

            def legend_title_func(channel):
                return "Legend"

        else:
            nrows = len(channels_to_plot)
            figsize = (12, 4 * len(channels_to_plot))

            def label_func(channel):
                return "Data Points"

            def legend_title_func(channel):
                return f"{channel} Legend"

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex=False,
            sharey=False,
            figsize=figsize,
            layout="constrained",
        )

        if same_axes:
            axes_channels: list[tuple[Any, str]] | Any = [
                (axes, channel) for channel in channels_to_plot
            ]
        else:
            axes_channels = zip(np.ravel(axes), channels_to_plot, strict=False)

        for i, (ax, channel) in enumerate(axes_channels):
            if self.X is not None:
                x = self.X[channel].to_numpy()
                y = channel_contributions.sel(channel=channel).to_numpy()

                label = label_func(channel)
                ax.scatter(x, y, label=label, color=f"C{i}")

                if show_fit:
                    label = f"{channel} Fit Curve" if same_axes else "Fit Curve"
                    self._plot_response_curve_fit(
                        ax=ax,
                        channel=channel,
                        color_index=i,
                        xlim_max=xlim_max,
                        label=label,
                        quantile_lower=quantile_lower,
                        quantile_upper=quantile_upper,
                    )

                title = legend_title_func(channel)
                ax.legend(
                    loc="upper left",
                    facecolor="white",
                    title=title,
                    fontsize="small",
                )

                ax.set(xlabel="Spent", ylabel="Contribution")

        fig.suptitle("Direct response curves", fontsize=16)
        return fig

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
                [self.X.iloc[-self.adstock.l_max :, :], X_pred], axis=0
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
                date=slice(self.adstock.l_max, None)
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
        dist: type[pm.Distribution] = pm.Gamma,
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

            model = MMM(
                adstock="geometric",
                saturation="logistic",
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
        if not hasattr(self, "model"):
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
        # This is coupled with the name of the
        # latent process Deterministic
        time_varying_var_name = (
            "media_temporal_latent_multiplier" if self.time_varying_media else None
        )
        add_lift_measurements_to_likelihood_from_saturation(
            df_lift_test=df_lift_test_scaled,
            saturation=self.saturation,
            time_varying_var_name=time_varying_var_name,
            model=self.model,
            dist=dist,
            name=name,
        )

    def _create_synth_dataset(
        self,
        df: pd.DataFrame,
        date_column: str,
        allocation_strategy: dict[str, float],
        channels: list[str] | tuple[str],
        controls: list[str] | None,
        target_col: str,
        time_granularity: str,
        time_length: int,
        lag: int,
    ) -> pd.DataFrame:
        """
        Create a synthetic dataset based on the given allocation strategy (Budget) and time granularity.

        Parameters
        ----------
        df : pd.DataFrame
            The original dataset.
        date_column : str
            The name of the date column in the dataset.
        allocation_strategy : dict[str, float]
            A dictionary mapping channel names to their corresponding allocation values.
        channels : list[str] | tuple[str]
            A list or tuple of channel names.
        controls : list[str] | None
            A list of control column names or None if no controls are present.
        target_col : str
            The name of the target column.
        time_granularity : str
            The time granularity of the synthetic dataset: 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'.
        time_length : int
            The length of the synthetic dataset in terms of the time granularity.
        lag : int
            The lag value (not used in this function).

        Returns
        -------
        pd.DataFrame
            A synthetic dataset with the specified allocation strategy and time granularity.

        Raises
        ------
        ValueError
            If the time granularity is not supported.
        """
        time_offsets = {
            "daily": {"days": 1},
            "weekly": {"weeks": 1},
            "monthly": {"months": 1},
            "quarterly": {"months": 3},
            "yearly": {"years": 1},
        }

        if time_granularity not in time_offsets:
            raise ValueError(
                "Unsupported time granularity. Choose from 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'."
            )

        if controls is not None:
            _controls: list[str] = controls
        else:
            controls = []

        last_date = pd.to_datetime(df[date_column]).max()
        new_dates = []
        for i in range(1, time_length + 1):
            if time_granularity == "daily":
                new_date = last_date + pd.DateOffset(days=i)
            elif time_granularity == "weekly":
                new_date = last_date + pd.DateOffset(weeks=i)
            elif time_granularity == "monthly":
                new_date = last_date + pd.DateOffset(months=i)
            elif time_granularity == "quarterly":
                new_date = last_date + pd.DateOffset(months=3 * i)
            elif time_granularity == "yearly":
                new_date = last_date + pd.DateOffset(years=i)
            new_dates.append(new_date)

        new_rows = [
            {
                self.date_column: pd.to_datetime(new_date),
                **{
                    channel: allocation_strategy.get(channel, 0)
                    + np.random.normal(0, 0.1 * allocation_strategy.get(channel, 0))
                    for channel in channels
                },
                **{control: 0 for control in _controls},
                target_col: 0,
            }
            for new_date in new_dates
        ]

        return pd.DataFrame(new_rows)

    def allocate_budget_to_maximize_response(
        self,
        budget: float | int,
        time_granularity: str,
        num_days: int,
        budget_bounds: dict[str, list[Any]] | None = None,
        custom_constraints: dict[str, float] | None = None,
        quantile: float = 0.5,
    ) -> az.InferenceData:
        """
        Allocate the given budget to maximize the response over a specified time period.

        This function optimizes the allocation of a given budget across different channels
        to maximize the response, considering adstock and saturation effects. It scales the
        budget and budget bounds, performs the optimization, and generates a synthetic dataset
        for posterior predictive sampling.

        The function first scales the budget and budget bounds using the maximum scale
        of the channel transformer. It then uses the `BudgetOptimizer` to allocate the
        budget, and creates a synthetic dataset based on the optimal allocation. Finally,
        it performs posterior predictive sampling on the synthetic dataset.

        Parameters
        ----------
        budget : float or int
            The total budget to be allocated.
        time_granularity : str
            The granularity of the time periods (e.g., 'daily', 'weekly', 'monthly').
        num_days : int
            The number of days over which the budget is to be allocated.
        budget_bounds : dict[str, list[Any]], optional
            A dictionary specifying the lower and upper bounds for the budget allocation
            for each channel. If None, no bounds are applied.
        custom_constraints : dict[str, float], optional
            Custom constraints for the optimization. If None, no custom constraints are applied.
        quantile : float, optional
            The quantile to use for recovering transformation parameters. Default is 0.5.

        Returns
        -------
        az.InferenceData
            The posterior predictive samples generated from the synthetic dataset.

        Raises
        ------
        ValueError
            If the time granularity is not supported.
        """
        parameters_mid = self.format_recovered_transformation_parameters(
            quantile=quantile
        )

        scale_budget = budget / self.channel_transformer["scaler"].scale_.max()

        if isinstance(budget_bounds, dict):
            scale_budget_bounds: dict[str, tuple[float, float]] | None = {
                k: (
                    v[0] / self.channel_transformer["scaler"].scale_.max(),
                    v[1] / self.channel_transformer["scaler"].scale_.max(),
                )
                for k, v in budget_bounds.items()
            }
        else:
            scale_budget_bounds = None

        allocator = BudgetOptimizer(
            adstock=self.adstock,
            saturation=self.saturation,
            parameters=parameters_mid,
            adstock_first=self.adstock_first,
            num_days=num_days,
        )

        self.optimal_allocation_dict, _ = allocator.allocate_budget(
            total_budget=scale_budget,
            budget_bounds=scale_budget_bounds,
            custom_constraints=custom_constraints,
        )

        inverse_scaled_channel_spend = self.channel_transformer.inverse_transform(
            np.array([list(self.optimal_allocation_dict.values())])
        )
        original_scale_allocation_dict = dict(
            zip(
                self.optimal_allocation_dict.keys(),
                inverse_scaled_channel_spend[0],
                strict=False,
            )
        )

        synth_dataset = self._create_synth_dataset(
            df=self.X,
            date_column=self.date_column,
            allocation_strategy=original_scale_allocation_dict,
            channels=self.channel_columns,
            controls=self.control_columns,
            target_col=self.output_var,
            time_granularity=time_granularity,
            time_length=num_days,
            lag=self.adstock.l_max,
        )

        return self.sample_posterior_predictive(
            X_pred=synth_dataset,
            extend_idata=False,
            include_last_observations=True,
            original_scale=False,
            var_names=["y", "channel_contributions"],
            progressbar=False,
        )

    def plot_budget_allocation(
        self,
        samples: az.InferenceData,
        figsize: tuple[float, float] = (12, 6),
        ax: plt.Axes | None = None,
        original_scale: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the budget allocation and channel contributions.

        Parameters
        ----------
        samples : az.InferenceData
            The inference data containing the channel contributions.
        figsize : tuple[float, float], optional
            The size of the figure to be created, by default (12, 6).
        ax : plt.Axes, optional
            The axis to plot on. If None, a new figure and axis will be created.
        original_scale : bool, optional
            A boolean flag to determine if the values should be plotted in their original scale,
            by default True.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The matplotlib figure object and axis containing the plot.
        """

        if original_scale:
            channel_contributions = (
                samples["channel_contributions"]
                .mean(dim=["sample"])
                .mean(dim=["date"])
                .values
                * self.get_target_transformer()["scaler"].scale_
            )

            allocate_spend = (
                np.array(list(self.optimal_allocation_dict.values()))
                * self.channel_transformer["scaler"].scale_
            )

        else:
            channel_contributions = (
                samples["channel_contributions"]
                .mean(dim=["sample"])
                .mean(dim=["date"])
                .values
            )
            allocate_spend = np.array(list(self.optimal_allocation_dict.values()))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        bar_width = 0.35
        opacity = 0.7

        index = np.arange(len(self.channel_columns))

        bars1 = ax.bar(
            index,
            allocate_spend,
            bar_width,
            color="b",
            alpha=opacity,
            label="Allocate Spend",
        )

        ax2 = ax.twinx()

        bars2 = ax2.bar(
            index + bar_width,
            channel_contributions,
            bar_width,
            color="r",
            alpha=opacity,
            label="Channel Contributions",
        )

        ax.set_xlabel("Channels")
        ax.set_ylabel("Allocate Spend", color="b")
        ax.tick_params(axis="x", rotation=90)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(self.channel_columns)

        ax.set_ylabel("Allocate Spend", color="b", labelpad=10)
        ax2.set_ylabel("Channel Contributions", color="r", labelpad=10)

        ax.grid(False)
        ax2.grid(False)

        bars = [bars1[0], bars2[0]]
        labels = [bar.get_label() for bar in bars]
        ax.legend(bars, labels)

        return fig, ax

    def plot_allocated_contribution_by_channel(
        self,
        samples: az.InferenceData,
        lower_quantile: float = 0.025,
        upper_quantile: float = 0.975,
        original_scale: bool = True,
    ) -> plt.Figure:
        """
        Plot the allocated contribution by channel with uncertainty intervals.

        This function visualizes the mean allocated contributions by channel along with
        the uncertainty intervals defined by the lower and upper quantiles. The contributions
        can be plotted on the original scale or the transformed scale.

        Parameters
        ----------
        samples : az.InferenceData
            The inference data containing the samples of channel contributions.
        lower_quantile : float, optional
            The lower quantile for the uncertainty interval. Default is 0.025.
        upper_quantile : float, optional
            The upper quantile for the uncertainty interval. Default is 0.975.
        original_scale : bool, optional
            If True, the contributions are plotted on the original scale. Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the plot.
        """
        if original_scale:
            channel_contributions = (
                samples["channel_contributions"]
                * self.get_target_transformer()["scaler"].scale_
            )
        else:
            channel_contributions = samples["channel_contributions"]

        fig, ax = plt.subplots()
        channel_contributions.mean(dim="sample").plot(hue="channel", ax=ax)

        for channel in self.model_coords["channel"]:
            ax.fill_between(
                x=channel_contributions.date.values,
                y1=channel_contributions.sel(channel=channel).quantile(
                    lower_quantile, dim="sample"
                ),
                y2=channel_contributions.sel(channel=channel).quantile(
                    upper_quantile, dim="sample"
                ),
                alpha=0.1,
            )
        return fig


class DelayedSaturatedMMM(MMM):
    _model_type = "MMM"
    _model_name = "DelayedSaturatedMMM"
    version = "0.0.3"

    def __init__(
        self,
        date_column: str,
        channel_columns: list[str],
        adstock_max_lag: int,
        time_varying_intercept: bool = False,
        time_varying_media: bool = False,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        validate_data: bool = True,
        control_columns: list[str] | None = None,
        yearly_seasonality: int | None = None,
        adstock_first: bool = True,
        **kwargs,
    ) -> None:
        """
        Wrapper function for DelayedSaturatedMMM class initializer.

        Warns that MMM class should be used instead and returns an instance of MMM with
        geometric adstock and logistic saturation.
        """
        warnings.warn(
            "The DelayedSaturatedMMM class is deprecated. Please use the MMM class instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(
            date_column=date_column,
            channel_columns=channel_columns,
            adstock_max_lag=adstock_max_lag,
            time_varying_intercept=time_varying_intercept,
            time_varying_media=time_varying_media,
            model_config=model_config,
            sampler_config=sampler_config,
            validate_data=validate_data,
            control_columns=control_columns,
            yearly_seasonality=yearly_seasonality,
            adstock="geometric",
            saturation="logistic",
            adstock_first=adstock_first,
            **kwargs,
        )
