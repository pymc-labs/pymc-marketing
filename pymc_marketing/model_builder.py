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


import hashlib
import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc.util import RandomState

# If scikit-learn is available, use its data validator
try:
    from sklearn.utils.validation import check_array, check_X_y
# If scikit-learn is not available, return the data unchanged
except ImportError:

    def check_X_y(X, y, **kwargs):
        return X, y

    def check_array(X, **kwargs):
        return X


class ModelBuilder(ABC):
    """
    ModelBuilder can be used to provide an easy-to-use API (similar to scikit-learn) for models
    and help with deployment.
    """

    _model_type = "BaseClass"
    version = "None"

    X: pd.DataFrame | None = None
    y: pd.Series | np.ndarray | None = None

    def __init__(
        self,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        """
        Initializes model configuration and sampler configuration for the model

        Parameters
        ----------
        data : Dictionary, optional
            It is the data we need to train the model on.
        model_config : Dictionary, optional
            dictionary of parameters that initialise model configuration.
            Class-default defined by the user default_model_config method.
        sampler_config : Dictionary, optional
            dictionary of parameters that initialise sampler configuration.
            Class-default defined by the user default_sampler_config method.
        Examples
        --------
        >>> class MyModel(ModelBuilder):
        >>>     ...
        >>> model = MyModel(model_config, sampler_config)
        """
        if sampler_config is None:
            sampler_config = {}
        if model_config is None:
            model_config = {}
        self.sampler_config = (
            self.default_sampler_config | sampler_config
        )  # Parameters for fit sampling
        self.model_config = (
            self.default_model_config | model_config
        )  # parameters for priors etc.
        self.model: pm.Model | None = None  # Set by build_model
        self.idata: az.InferenceData | None = None  # idata is generated during fitting
        self.is_fitted_ = False

    def _validate_data(self, X, y=None):
        if y is not None:
            return check_X_y(
                X, y, accept_sparse=False, y_numeric=True, multi_output=False
            )
        else:
            return check_array(X, accept_sparse=False)

    @abstractmethod
    def _data_setter(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> None:
        """
        Sets new data in the model.

        Parameters
        ----------
        X : array, shape (n_obs, n_features)
            The training input samples.
        y : array, shape (n_obs,)
            The target values (real numbers).

        Returns:
        ----------
        None

        Examples
        --------
        >>> def _data_setter(self, data : pd.DataFrame):
        >>>     with self.model:
        >>>         pm.set_data({'x': X['x'].values})
        >>>         try: # if y values in new data
        >>>             pm.set_data({'y_data': y.values})
        >>>         except: # dummies otherwise
        >>>             pm.set_data({'y_data': np.zeros(len(data))})

        """

    @property
    @abstractmethod
    def output_var(self):
        """
        Returns the name of the output variable of the model.

        Returns
        -------
        output_var : str
            Name of the output variable of the model.
        """

    @property
    @abstractmethod
    def default_model_config(self) -> dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization
        Useful for understanding structure of required model_config to allow its customization by users
        Examples
        --------
        >>>     @classmethod
        >>>     def default_model_config(self):
        >>>         Return {
        >>>             'a' : {
        >>>                 'loc': 7,
        >>>                 'scale' : 3
        >>>             },
        >>>             'b' : {
        >>>                 'loc': 3,
        >>>                 'scale': 5
        >>>             }
        >>>              'obs_error': 2
        >>>         }

        Returns
        -------
        model_config : dict
            A set of default parameters for predictor distributions that allow to save and recreate the model.
        """

    @property
    @abstractmethod
    def default_sampler_config(self) -> dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization
        Useful for understanding structure of required sampler_config to allow its customization by users
        Examples
        --------
        >>>     @classmethod
        >>>     def default_sampler_config(self):
        >>>         Return {
        >>>             'draws': 1_000,
        >>>             'tune': 1_000,
        >>>             'chains': 1,
        >>>             'target_accept': 0.95,
        >>>         }

        Returns
        -------
        sampler_config : dict
            A set of default settings for used by model in fit process.
        """

    @abstractmethod
    def _generate_and_preprocess_model_data(
        self, X: pd.DataFrame | pd.Series, y: np.ndarray
    ) -> None:
        """
        Applies preprocessing to the data before fitting the model.
        if validate is True, it will check if the data is valid for the model.
        sets self.model_coords based on provided dataset

        In case of optional parameters being passed into the model, this method should implement the conditional
        logic responsible for correct handling of the optional parameters, and including them into the dataset.

        Parameters:
        X : array, shape (n_obs, n_features)
        y : array, shape (n_obs,)

        Examples
        --------
        >>>     @classmethod
        >>>     def _generate_and_preprocess_model_data(self, X, y):
                    coords = {
                        'x_dim': X.dim_variable,
                    } #only include if applicable for your model
        >>>         self.X = X
        >>>         self.y = y

        Returns
        -------
        None

        """

    @abstractmethod
    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        **kwargs,
    ) -> None:
        """
        Creates an instance of pm.Model based on provided data and model_config, and
        attaches it to self.

        Parameters
        ----------
        X : pd.DataFrame
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : Union[pd.Series, np.ndarray]
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.

        See Also
        --------
        default_model_config : returns default model config

        Returns
        -------
        None
        """

    def set_idata_attrs(self, idata=None):
        """
        Set attributes on an InferenceData object.

        Parameters
        ----------
        idata : arviz.InferenceData, optional
            The InferenceData object to set attributes on.

        Raises
        ------
        RuntimeError
            If no InferenceData object is provided.

        Returns
        -------
        None

        Examples
        --------
        >>> model = MyModel(ModelBuilder)
        >>> idata = az.InferenceData(your_dataset)
        >>> model.set_idata_attrs(idata=idata)
        """
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        idata.attrs["id"] = self.id
        idata.attrs["model_type"] = self._model_type
        idata.attrs["version"] = self.version
        idata.attrs["sampler_config"] = json.dumps(self.sampler_config)
        idata.attrs["model_config"] = json.dumps(self._serializable_model_config)
        # Only classes with non-dataset parameters will implement save_input_params
        if hasattr(self, "_save_input_params"):
            self._save_input_params(idata)
        return idata

    def save(self, fname: str) -> None:
        """
        Save the model's inference data to a file.

        Parameters
        ----------
        fname : str
            The name and path of the file to save the inference data with model parameters.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model hasn't been fit yet (no inference data available).

        Examples
        --------
        This method is meant to be overridden and implemented by subclasses.
        It should not be called directly on the base abstract class or its instances.

        >>> class MyModel(ModelBuilder):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>> model = MyModel()
        >>> model.fit(X,y)
        >>> model.save('model_results.nc')  # This will call the overridden method in MyModel
        """
        if self.idata is not None and "posterior" in self.idata:
            file = Path(str(fname))
            self.idata.to_netcdf(str(file))
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")

    @classmethod
    def _model_config_formatting(cls, model_config: dict) -> dict:
        """
        Because of json serialization, model_config values that were originally tuples
        or numpy are being encoded as lists. This function converts them back to tuples
        and numpy arrays to ensure correct id encoding.
        """
        for key in model_config:
            if isinstance(model_config[key], dict):
                for sub_key in model_config[key]:
                    if isinstance(model_config[key][sub_key], list):
                        # Check if "dims" key to convert it to tuple
                        if sub_key == "dims":
                            model_config[key][sub_key] = tuple(
                                model_config[key][sub_key]
                            )
                        # Convert all other lists to numpy arrays
                        else:
                            model_config[key][sub_key] = np.array(
                                model_config[key][sub_key]
                            )
        return model_config

    @classmethod
    def load(cls, fname: str):
        """
        Creates a ModelBuilder instance from a file,
        Loads inference data for the model.

        Parameters
        ----------
        fname : string
            This denotes the name with path from where idata should be loaded from.

        Returns
        -------
        Returns an instance of ModelBuilder.

        Raises
        ------
        ValueError
            If the inference data that is loaded doesn't match with the model.
        Examples
        --------
        >>> class MyModel(ModelBuilder):
        >>>     ...
        >>> name = './mymodel.nc'
        >>> imported_model = MyModel.load(name)
        """
        filepath = Path(str(fname))
        idata = az.from_netcdf(filepath)
        # needs to be converted, because json.loads was changing tuple to list
        model_config = cls._model_config_formatting(
            json.loads(idata.attrs["model_config"])
        )
        model = cls(
            model_config=model_config,
            sampler_config=json.loads(idata.attrs["sampler_config"]),
        )
        model.idata = idata
        dataset = idata.fit_data.to_dataframe()
        X = dataset.drop(columns=[model.output_var])
        y = dataset[model.output_var]
        model.build_model(X, y)
        # All previously used data is in idata.

        if model.id != idata.attrs["id"]:
            error_msg = f"""The file '{fname}' does not contain an inference data of the same model
            or configuration as '{cls._model_type}'"""
            raise ValueError(error_msg)

        return model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        progressbar: bool = True,
        predictor_names: list[str] | None = None,
        random_seed: RandomState | None = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Fit a model using the data passed as a parameter.
        Sets attrs to inference data of the model.


        Parameters
        ----------
        X : array-like if sklearn is available, otherwise array, shape (n_obs, n_features)
            The training input samples.
        y : array-like if sklearn is available, otherwise array, shape (n_obs,)
            The target values (real numbers).
        progressbar : bool
            Specifies whether the fit progressbar should be displayed
        predictor_names: Optional[List[str]] = None,
            Allows for custom naming of predictors given in a form of 2dArray
            Allows for naming of predictors when given in a form of np.ndarray, if not provided
            the predictors will be named like predictor1, predictor2...
        random_seed : Optional[RandomState]
            Provides sampler with initial random seed for obtaining reproducible samples
        **kwargs : Any
            Custom sampler settings can be provided in form of keyword arguments.

        Returns
        -------
        self : az.InferenceData
            returns inference data of the fitted model.
        Examples
        --------
        >>> model = MyModel()
        >>> idata = model.fit(X,y)
        Auto-assigning NUTS sampler...
        Initializing NUTS using jitter+adapt_diag...
        """
        if predictor_names is None:
            predictor_names = []
        if y is None:
            y = np.zeros(X.shape[0])
        y_df = pd.DataFrame({self.output_var: y})
        self._generate_and_preprocess_model_data(X, y_df.values.flatten())
        if self.X is None or self.y is None:
            raise ValueError("X and y must be set before calling build_model!")

        if self.model is None:
            self.build_model(self.X, self.y)

        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)

        sampler_config.update(**kwargs)
        if self.model is not None:
            with self.model:
                sampler_args = {**self.sampler_config, **kwargs}
                self.idata = pm.sample(**sampler_args)

        X_df = pd.DataFrame(X, columns=X.columns)
        combined_data = pd.concat([X_df, y_df], axis=1)
        if not all(combined_data.columns):
            raise ValueError("All columns must have non-empty names")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            self.idata.add_groups(fit_data=combined_data.to_xarray())  # type: ignore
        self.set_idata_attrs(self.idata)
        return self.idata  # type: ignore

    def predict(
        self,
        X_pred: np.ndarray | pd.DataFrame | pd.Series,
        extend_idata: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Uses model to predict on unseen data and return point prediction of all the samples. The point prediction
        for each input row is the expected output value, computed as the mean of MCMC samples.

        Parameters
        ---------
        X_pred : array-like if sklearn is available, otherwise array, shape (n_pred, n_features)
            The input data used for prediction.
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to True.
        **kwargs: Additional arguments to pass to sample_posterior_predictive method

        Returns
        -------
        y_pred : ndarray, shape (n_pred,)
            Predicted output corresponding to input X_pred.

        Examples
        --------
        >>> model = MyModel()
        >>> idata = model.fit(X,y)
        >>> x_pred = []
        >>> prediction_data = pd.DataFrame({'input':x_pred})
        >>> pred_mean = model.predict(prediction_data)
        """

        posterior_predictive_samples = self.sample_posterior_predictive(
            X_pred, extend_idata, combined=False, **kwargs
        )

        if self.output_var not in posterior_predictive_samples:
            raise KeyError(
                f"Output variable {self.output_var} not found in posterior predictive samples."
            )

        posterior_means = posterior_predictive_samples[self.output_var].mean(
            dim=["chain", "draw"], keep_attrs=True
        )
        return posterior_means.data

    def sample_prior_predictive(
        self,
        X_pred,
        y_pred=None,
        samples: int | None = None,
        extend_idata: bool = False,
        combined: bool = True,
        **kwargs,
    ):
        """
        Sample from the model's prior predictive distribution.

        Parameters
        ---------
        X_pred : array, shape (n_pred, n_features)
            The input data used for prediction using prior distribution.
        samples : int
            Number of samples from the prior parameter distributions to generate.
            If not set, uses sampler_config['draws'] if that is available, otherwise defaults to 500.
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to False.
        combined: Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        **kwargs: Additional arguments to pass to pymc.sample_prior_predictive

        Returns
        -------
        prior_predictive_samples : DataArray, shape (n_pred, samples)
            Prior predictive samples for each input X_pred
        """
        if y_pred is None:
            y_pred = np.zeros(len(X_pred))
        if samples is None:
            samples = self.sampler_config.get("draws", 500)

        if self.model is None:
            self.build_model(X_pred, y_pred)

        self._data_setter(X_pred, y_pred)
        if self.model is not None:
            with self.model:  # sample with new input data
                prior_pred: az.InferenceData = pm.sample_prior_predictive(
                    samples, **kwargs
                )
                self.set_idata_attrs(prior_pred)
                if extend_idata:
                    if self.idata is not None:
                        self.idata.extend(prior_pred, join="right")
                    else:
                        self.idata = prior_pred

        prior_predictive_samples = az.extract(
            prior_pred, "prior_predictive", combined=combined
        )

        return prior_predictive_samples

    def sample_posterior_predictive(
        self, X_pred, extend_idata: bool = True, combined: bool = True, **kwargs
    ):
        """
        Sample from the model's posterior predictive distribution.

        Parameters
        ---------
        X_pred : array, shape (n_pred, n_features)
            The input data used for prediction using prior distribution..
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        **kwargs: Additional arguments to pass to pymc.sample_posterior_predictive

        Returns
        -------
        posterior_predictive_samples : DataArray, shape (n_pred, samples)
            Posterior predictive samples for each input X_pred
        """
        self._data_setter(X_pred)

        with self.model:  # type: ignore
            post_pred = pm.sample_posterior_predictive(self.idata, **kwargs)
            if extend_idata:
                self.idata.extend(post_pred, join="right")  # type: ignore

        posterior_predictive_samples = az.extract(
            post_pred, "posterior_predictive", combined=combined
        )

        return posterior_predictive_samples

    def get_params(self, deep=True):
        """
        Get all the model parameters needed to instantiate a copy of the model, not including training data.
        """
        return {
            "model_config": self.model_config,
            "sampler_config": self.sampler_config,
        }

    def set_params(self, **params):
        """
        Set all the model parameters needed to instantiate the model, not including training data.
        """
        self.model_config = params["model_config"]
        self.sampler_config = params["sampler_config"]

    @property
    @abstractmethod
    def _serializable_model_config(self) -> dict[str, int | float | dict]:
        """
        Converts non-serializable values from model_config to their serializable reversable equivalent.
        Data types like pandas DataFrame, Series or datetime aren't JSON serializable,
        so in order to save the model they need to be formatted.

        Returns
        -------
        model_config: dict
        """

    def predict_proba(
        self,
        X_pred: np.ndarray | pd.DataFrame | pd.Series,
        extend_idata: bool = True,
        combined: bool = False,
        **kwargs,
    ) -> xr.DataArray:
        """Alias for `predict_posterior`, for consistency with scikit-learn probabilistic estimators."""
        return self.predict_posterior(X_pred, extend_idata, combined, **kwargs)

    def predict_posterior(
        self,
        X_pred: np.ndarray | pd.DataFrame | pd.Series,
        extend_idata: bool = True,
        combined: bool = True,
        **kwargs,
    ) -> xr.DataArray:
        """
        Generate posterior predictive samples on unseen data.

        Parameters
        ---------
        X_pred : array-like if sklearn is available, otherwise array, shape (n_pred, n_features)
            The input data used for prediction.
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        **kwargs: Additional arguments to pass to sample_posterior_predictive method

        Returns
        -------
        y_pred : DataArray, shape (n_pred, chains * draws) if combined is True, otherwise (chains, draws, n_pred)
            Posterior predictive samples for each input X_pred
        """

        X_pred = self._validate_data(X_pred)
        posterior_predictive_samples = self.sample_posterior_predictive(
            X_pred, extend_idata, combined, **kwargs
        )

        if self.output_var not in posterior_predictive_samples:
            raise KeyError(
                f"Output variable {self.output_var} not found in posterior predictive samples."
            )

        return posterior_predictive_samples[self.output_var]

    @property
    def id(self) -> str:
        """
        Generate a unique hash value for the model.

        The hash value is created using the last 16 characters of the SHA256 hash encoding,
        based on the model configuration, version, and model type.

        Returns
        -------
        str
            A string of length 16 characters containing a unique hash of the model.

        Examples
        --------
        >>> model = MyModel()
        >>> model.id
        '0123456789abcdef'
        """
        hasher = hashlib.sha256()
        hasher.update(str(self.model_config.values()).encode())
        hasher.update(self.version.encode())
        hasher.update(self._model_type.encode())
        return hasher.hexdigest()[:16]
