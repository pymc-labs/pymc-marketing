#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""Base class responsible of the high level API for model building, fitting saving and loading."""

import hashlib
import json
import warnings
from abc import ABC, abstractmethod
from inspect import signature
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc.util import RandomState

from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.utils import from_netcdf
from pymc_marketing.version import __version__

# If scikit-learn is available, use its data validator
try:
    from sklearn.utils.validation import check_array, check_X_y
# If scikit-learn is not available, return the data unchanged
except ImportError:

    def check_X_y(X, y, **kwargs):
        """Check if the input data is valid for the model."""
        return X, y

    def check_array(X, **kwargs):
        """Check if the input data is valid for the model."""
        return X


def _handle_deprecate_pred_argument(
    value,
    name: str,
    kwargs: dict,
    none_allowed: bool = False,
):
    name_pred = f"{name}_pred"
    if name_pred in kwargs and value is not None:
        raise ValueError(f"Both {name} and {name_pred} cannot be provided.")

    if name_pred not in kwargs and value is None and none_allowed:
        return value

    if name_pred not in kwargs and value is None:
        raise ValueError(f"Please provide {name}.")

    if name_pred in kwargs:
        warnings.warn(
            f"{name_pred} is deprecated, use {name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return kwargs.pop(name_pred)

    return value


def create_idata_accessor(value: str, message: str):
    """Create a property accessor for an InferenceData object.

    Underlying object must have an InferenceData object attribute named 'idata'.

    Parameters
    ----------
    value : str
        The value to access in the InferenceData object.
    message : str
        The error message to raise if the value is not found in the InferenceData object.

    Returns
    -------
    property
        The property accessor for the InferenceData object.

    """

    def accessor(self) -> xr.Dataset:
        __doc__ = f"""Access the '{value}' attribute of the InferenceData object."""  # noqa: F841
        if self.idata is None or value not in self.idata:
            raise RuntimeError(message)

        return self.idata[value]

    return property(accessor)


def create_sample_kwargs(
    sampler_config: dict[str, Any],
    progressbar: bool | None,
    random_seed,
    **kwargs,
) -> dict[str, Any]:
    """Create the dictionary of keyword arguments for `pm.sample`.

    Parameters
    ----------
    sampler_config : dict
        The configuration dictionary for the sampler.
    progressbar : bool, optional
        Whether to show the progress bar during sampling. Defaults to True.
    random_seed : RandomState
        The random seed for the sampler.
    **kwargs : Any
        Additional keyword arguments to pass to the sampler.

    Returns
    -------
    dict
        The dictionary of keyword arguments for `pm.sample`.

    """
    sampler_config = sampler_config.copy()

    if progressbar is not None:
        sampler_config["progressbar"] = progressbar
    else:
        sampler_config["progressbar"] = sampler_config.get("progressbar", True)

    if random_seed is not None:
        sampler_config["random_seed"] = random_seed

    sampler_config.update(**kwargs)

    return sampler_config


class DifferentModelError(Exception):
    """Error raised when a model loaded is different than one saved."""


class ModelBuilder(ABC):
    """Base class for building models with PyMC-Marketing.

    It provides an easy-to-use API (similar to scikit-learn) for models
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
        """Initialize model configuration and sampler configuration for the model.

        Parameters
        ----------
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
        self.model: pm.Model
        self.idata: az.InferenceData | None = None  # idata is generated during fitting
        self.is_fitted_ = False

    def _validate_data(self, X, y=None):
        if y is not None:
            return check_X_y(
                X, y, accept_sparse=False, y_numeric=True, multi_output=False
            )
        else:
            return check_array(X, accept_sparse=False)

    def _data_setter(
        self,
        X: np.ndarray | pd.DataFrame | xr.Dataset | xr.DataArray,
        y: np.ndarray | pd.Series | xr.DataArray | None = None,
    ) -> None:
        """Set new data in the model.

        Parameters
        ----------
        X : array, shape (n_obs, n_features)
            The training input samples.
        y : array, shape (n_obs,)
            The target values (real numbers).

        Examples
        --------
        Example logic of data_setter method

        .. code-block:: python

            def _data_setter(self, X, y=None):
                data = {"X": X}
                if y is None:
                    y = np.zeros(len(X))
                data["y"] = y

                with self.model:
                    pm.set_data(data)

        """
        msg = "This model doesn't support setting new data, posterior_predictive, or out of sample methods."
        raise NotImplementedError(msg)

    @property
    @abstractmethod
    def output_var(self) -> str:
        """Returns the name of the output variable of the model.

        Returns
        -------
        output_var : str
            Name of the output variable of the model.

        """

    @property
    @abstractmethod
    def default_model_config(self) -> dict:
        """Return a class default configuration dictionary.

        For model builder if no model_config is provided on class initialization
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
        """Return a class default sampler configuration dictionary.

        For model builder if no sampler_config is provided on class initialization
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
    def build_model(
        self,
        X: pd.DataFrame | xr.Dataset | xr.DataArray,
        y: pd.Series | np.ndarray | xr.DataArray,
        **kwargs,
    ) -> None:
        """Create an instance of `pm.Model` based on provided data and model_config.

        It attaches the model to self.model.

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

    def create_idata_attrs(self) -> dict[str, str]:
        """Create attributes for the inference data.

        Returns
        -------
        dict[str, str]
            A dictionary of attributes for the inference data.
        """

        def default(x):
            if hasattr(x, "to_dict"):
                return x.to_dict()
            elif isinstance(x, HSGPKwargs):
                return x.model_dump(mode="json")
            return x.__dict__

        attrs: dict[str, str] = {}

        attrs["id"] = self.id
        attrs["model_type"] = self._model_type
        attrs["version"] = self.version
        attrs["sampler_config"] = json.dumps(self.sampler_config)
        attrs["model_config"] = json.dumps(
            self._serializable_model_config,
            default=default,
        )

        return attrs

    def set_idata_attrs(
        self, idata: az.InferenceData | None = None
    ) -> az.InferenceData:
        """Set attributes on an InferenceData object.

        Parameters
        ----------
        idata : arviz.InferenceData, optional
            The InferenceData object to set attributes on.

        Raises
        ------
        ValueError
            If the attrs are missing for a property initialization of the class
        RuntimeError
            If no InferenceData object is provided.

        Returns
        -------
        InferenceData
            The InferenceData instance with the attrs set

        Examples
        --------
        Set the attrs for an InferenceData object manually.

        .. code-block:: python

            idata: az.InferenceData = ...
            model.set_idata_attrs(idata=idata)

        """
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")

        attrs = self.create_idata_attrs()
        attrs_keys = set(attrs.keys())
        required_keys = {
            "id",
            "model_type",
            "version",
            "sampler_config",
            "model_config",
        }
        if missing_keys := required_keys - attrs_keys:
            msg = (
                f"Missing required keys in attrs: {missing_keys}. "
                "Call super().create_idata_attrs()."
            )
            raise ValueError(msg)

        init_parameters: set[str] = set(signature(self.__init__).parameters.keys())  # type: ignore
        # Remove since this will be stored in the fit_data group of InferenceData
        init_parameters -= {"data"}

        if missing_keys := init_parameters - attrs_keys:
            msg = (
                f"__init__ has parameters that are not in the attrs: {missing_keys}. "
                "The save and load functionality will not work correctly."
            )
            raise ValueError(msg)

        idata.attrs = attrs
        return idata

    def save(self, fname: str) -> None:
        """Save the model's inference data to a file.

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
        >>> model.fit(X, y)
        >>> model.save(
        ...     "model_results.nc"
        ... )  # This will call the overridden method in MyModel

        """
        if self.idata is not None and "posterior" in self.idata:
            file = Path(str(fname))
            self.idata.to_netcdf(str(file))
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")

    @classmethod
    def _model_config_formatting(cls, model_config: dict) -> dict:
        """Format the model configuration.

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
    def attrs_to_init_kwargs(cls, attrs) -> dict[str, Any]:
        """Convert the model configuration and sampler configuration from the attributes to keyword arguments."""
        return {
            "model_config": cls._model_config_formatting(
                json.loads(attrs["model_config"])
            ),
            "sampler_config": json.loads(attrs["sampler_config"]),
        }

    def build_from_idata(self, idata: az.InferenceData) -> None:
        """Build model from the InferenceData object.

        This is part of the :func:`load` method. See :func:`load` for more larger context.

        Usually a wrapper around the :func:`build_model` method unless the model
        has some additional steps to be built.

        Parameters
        ----------
        idata : az.InferenceData
            The InferenceData object to build the model from.

        """
        dataset = idata.fit_data.to_dataframe()  # type: ignore
        X = dataset.drop(columns=[self.output_var])
        y = dataset[self.output_var]

        self.build_model(X, y)

    @classmethod
    def load_from_idata(cls, idata: az.InferenceData) -> "ModelBuilder":
        """Create a ModelBuilder instance from an InferenceData object.

        This class method has a few steps:

        - Construct a new instance of the model using the InferenceData attrs
        - Build the model from the InferenceData
        - Check if the model id matches the id in the InferenceData loaded.

        Parameters
        ----------
        idata : az.InferenceData
            The InferenceData object to load the model from.

        Returns
        -------
        ModelBuilder
            An instance of the ModelBuilder class.

        Raises
        ------
        DifferentModelError
            If the model id in the InferenceData does not match the model id built.

        """
        # needs to be converted, because json.loads was changing tuple to list
        init_kwargs = cls.attrs_to_init_kwargs(idata.attrs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            model = cls(**init_kwargs)

        model.idata = idata
        model.build_from_idata(idata)
        model.post_sample_model_transformation()

        if (model_version := model.version) != (
            loaded_version := idata.attrs["version"]
        ):
            msg = (
                f"The model version ({loaded_version}) in the InferenceData does not "
                f"match the model version ({model_version}). "
                "There was no error loading the inference data, but the model structure "
                "is different. "
            )
            raise DifferentModelError(msg)
        elif model.id != idata.attrs["id"]:
            msg = (
                "The model id in the InferenceData does not match the model id. "
                "There was no error loading the inference data, but the model may "
                "be different. "
                "Investigate if the model structure or configuration has changed."
            )
            raise DifferentModelError(msg)

        return model

    @classmethod
    def load(cls, fname: str):
        """Create a ModelBuilder instance from a file.

        Loads inference data for the model.

        This class method has a few steps:

        - Load the InferenceData from the file.
        - Construct a new instance of the model using the InferenceData attrs
        - Build the model from the InferenceData
        - Check if the model id matches the id in the InferenceData loaded.

        Parameters
        ----------
        fname : string
            This denotes the name with path from where idata should be loaded from.

        Returns
        -------
        Returns an instance of ModelBuilder.

        Raises
        ------
        DifferentModelError
            If the inference data that is loaded doesn't match with the model.

        Examples
        --------
        Load a model from a file

        .. code-block:: python

            file_name: str = "./mymodel.nc"
            model = MyModel.load(file_name)

        """
        filepath = Path(str(fname))
        idata = from_netcdf(filepath)

        try:
            return cls.load_from_idata(idata)
        except DifferentModelError as e:
            error_msg = (
                f"The file '{fname}' does not contain "
                "an InferenceData of the same model "
                f"or configuration as '{cls._model_type}'"
            )
            raise DifferentModelError(error_msg) from e

    def create_fit_data(
        self,
        X: pd.DataFrame | xr.Dataset | xr.DataArray,
        y: np.ndarray | pd.Series | xr.DataArray,
    ) -> xr.Dataset:
        """Create the fit_data group based on the input data."""
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index, name=self.output_var)

        y.name = self.output_var

        if isinstance(X, pd.DataFrame):
            X = X.to_xarray()

        if isinstance(y, pd.Series):
            y = y.to_xarray()

        return xr.merge([X, y])

    def post_sample_model_transformation(self) -> None:
        """Perform transformation on the model after sampling."""
        return

    def fit(
        self,
        X: pd.DataFrame | xr.Dataset | xr.DataArray,
        y: pd.Series | xr.DataArray | np.ndarray | None = None,
        progressbar: bool | None = None,
        random_seed: RandomState | None = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Fit a model using the data passed as a parameter.

        Sets attrs to inference data of the model.

        Parameters
        ----------
        X : array-like | array, shape (n_obs, n_features)
            The training input samples. If scikit-learn is available, array-like, otherwise array.
        y : array-like | array, shape (n_obs,)
            The target values (real numbers). If scikit-learn is available, array-like, otherwise array.
        progressbar : bool, optional
            Specifies whether the fit progress bar should be displayed. Defaults to True.
        random_seed : Optional[RandomState]
            Provides sampler with initial random seed for obtaining reproducible samples.
        **kwargs : Any
            Custom sampler settings can be provided in form of keyword arguments.

        Returns
        -------
        self : az.InferenceData
            Returns inference data of the fitted model.

        Examples
        --------
        >>> model = MyModel()
        >>> idata = model.fit(X, y)
        Auto-assigning NUTS sampler...
        Initializing NUTS using jitter+adapt_diag...

        """
        if (
            isinstance(y, pd.Series)
            and isinstance(X, pd.DataFrame)
            and not X.index.equals(y.index)
        ):
            raise ValueError("Index of X and y must match.")

        if y is None:
            y = np.zeros(X.shape[0])

        if self.output_var in X:
            raise ValueError(
                f"X includes a column named '{self.output_var}', which conflicts with the target variable."
            )

        if not hasattr(self, "model"):
            self.build_model(X, y)

        sampler_kwargs = create_sample_kwargs(
            self.sampler_config,
            progressbar,
            random_seed,
            **kwargs,
        )

        # Sample without deterministics first
        var_names = [var.name for var in self.model.free_RVs]
        with self.model:
            idata = pm.sample(var_names=var_names, **sampler_kwargs)

        # Compute deterministics after sampling
        with self.model:
            idata.posterior = pm.compute_deterministics(
                idata.posterior, merge_dataset=True
            )

        self.post_sample_model_transformation()

        if self.idata:
            self.idata = self.idata.copy()
            self.idata.extend(idata, join="right")
        else:
            self.idata = idata

        self.idata["posterior"].attrs["pymc_marketing_version"] = __version__

        if "fit_data" in self.idata:
            del self.idata.fit_data

        fit_data = self.create_fit_data(X, y)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            self.idata.add_groups(fit_data=fit_data)
        self.set_idata_attrs(self.idata)
        return self.idata  # type: ignore

    @property
    def fit_result(self) -> xr.Dataset:
        """Get the posterior fit_result.

        Returns
        -------
        InferenceData object.

        """
        return create_idata_accessor(
            "posterior", "The model hasn't been fit yet, call .fit() first"
        ).__get__(self)

    @fit_result.setter
    def fit_result(self, res: az.InferenceData) -> None:
        """Create a setter method to overwrite the pre-existing fit_result.

        Parameters
        ----------
        res : az.InferenceData
            The inferencedata object to be set

        Returns
        -------
        property
            The property setter for the InferenceData object.

        """
        if self.idata is None:
            self.idata = res
        elif "posterior" in self.idata:
            warnings.warn("Overriding pre-existing fit_result", stacklevel=1)
            self.idata.posterior = res
        else:
            self.idata.posterior = res

    def predict(
        self,
        X: np.ndarray | pd.DataFrame | pd.Series | None = None,
        extend_idata: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Use a model to predict on unseen data and return point prediction of all the samples.

        The point prediction for each input row is the expected output value, computed as the mean of MCMC samples.

        Parameters
        ----------
        X : array-like | array, shape (n_pred, n_features)
            The input data used for prediction. If scikit-learn is available, array-like, otherwise array.
        extend_idata : Boolean
            Determine whether the predictions should be added to inference data object.
            Defaults to True.
        **kwargs: Additional arguments to pass to sample_posterior_predictive method

        Returns
        -------
        ndarray, shape (n_pred,)
            Predicted output corresponding to input X.

        Examples
        --------
        >>> model = MyModel()
        >>> idata = model.fit(X, y)
        >>> x_pred = []
        >>> prediction_data = pd.DataFrame({"input": x_pred})
        >>> pred_mean = model.predict(prediction_data)

        """
        posterior_predictive_samples = self.sample_posterior_predictive(
            X,
            extend_idata=extend_idata,
            combined=False,
            **kwargs,
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
        X=None,
        y=None,
        samples: int | None = None,
        extend_idata: bool = True,
        combined: bool = True,
        **kwargs,
    ):
        """Sample from the model's prior predictive distribution.

        Parameters
        ----------
        X : array, shape (n_pred, n_features)
            The input data used for prediction using prior distribution.
        y : array, shape (n_pred,), optional
            The target values (real numbers) used for prediction using prior distribution.
            If not set, defaults to an array of zeros.
        samples : int
            Number of samples from the prior parameter distributions to generate.
            If not set, uses sampler_config['draws'] if that is available, otherwise defaults to 500.
        extend_idata : Boolean
            Determine whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Boolean
            Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        **kwargs: Additional arguments to pass to pymc.sample_prior_predictive

        Returns
        -------
        prior_predictive_samples : DataArray, shape (n_pred, samples)
            Prior predictive samples for each input X

        """
        X = _handle_deprecate_pred_argument(X, "X", kwargs)
        y = _handle_deprecate_pred_argument(y, "y", kwargs, none_allowed=True)

        if y is None:
            y = np.zeros(len(X))
        if samples is None:
            samples = self.sampler_config.get("draws", 500)

        if not hasattr(self, "model"):
            self.build_model(X, y)

        with self.model:  # sample with new input data
            prior_pred: az.InferenceData = pm.sample_prior_predictive(samples, **kwargs)
            prior_pred["prior"].attrs["pymc_marketing_version"] = __version__
            prior_pred["prior_predictive"].attrs["pymc_marketing_version"] = __version__
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
        self,
        X=None,
        extend_idata: bool = True,
        combined: bool = True,
        **sample_posterior_predictive_kwargs,
    ):
        """Sample from the model's posterior predictive distribution.

        Parameters
        ----------
        X : array, shape (n_pred, n_features)
            The input data used for prediction using prior distribution..
        extend_idata : Boolean
            Determine whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Boolean
            Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        **sample_posterior_predictive_kwargs: Additional arguments to pass to pymc.sample_posterior_predictive

        Returns
        -------
        posterior_predictive_samples : DataArray, shape (n_pred, samples)
            Posterior predictive samples for each input X

        """
        X = _handle_deprecate_pred_argument(X, "X", sample_posterior_predictive_kwargs)

        self._data_setter(X)

        with self.model:
            post_pred = pm.sample_posterior_predictive(
                self.idata, **sample_posterior_predictive_kwargs
            )

        if extend_idata:
            self.idata.extend(post_pred, join="right")  # type: ignore

        variable_name = (
            "predictions"
            if sample_posterior_predictive_kwargs.get("predictions")
            else "posterior_predictive"
        )

        return az.extract(post_pred, variable_name, combined=combined)

    @property
    @abstractmethod
    def _serializable_model_config(self) -> dict[str, int | float | dict]:
        """Converts non-serializable values from model_config to their serializable reversable equivalent.

        Data types like pandas DataFrame, Series or datetime aren't JSON serializable,
        so in order to save the model they need to be formatted.

        Returns
        -------
        model_config: dict

        """

    def predict_proba(
        self,
        X: np.ndarray | pd.DataFrame | pd.Series | None = None,
        extend_idata: bool = True,
        combined: bool = False,
        **kwargs,
    ) -> xr.DataArray:
        """Alias for `predict_posterior`, for consistency with scikit-learn probabilistic estimators."""
        return self.predict_posterior(X, extend_idata, combined, **kwargs)

    def predict_posterior(
        self,
        X: np.ndarray | pd.DataFrame | pd.Series | None = None,
        extend_idata: bool = True,
        combined: bool = True,
        **kwargs,
    ) -> xr.DataArray:
        """Generate posterior predictive samples on unseen data.

        Parameters
        ----------
        X : array-like | array, shape (n_pred, n_features)
            The input data used for prediction. If scikit-learn is available, array-like, otherwise array.
        extend_idata : Boolean
            Determine whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Boolean
            Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        **kwargs: Additional arguments to pass to sample_posterior_predictive method

        Returns
        -------
        y_pred : DataArray
            Posterior predictive samples for each input X.
            Shape is (n_pred, chains * draws) if combined is True, otherwise (chains, draws, n_pred).

        """
        X = _handle_deprecate_pred_argument(X, "X", kwargs)
        X = self._validate_data(X)
        posterior_predictive_samples = self.sample_posterior_predictive(
            X, extend_idata, combined, **kwargs
        )

        if self.output_var not in posterior_predictive_samples:
            raise KeyError(
                f"Output variable {self.output_var} not found in posterior predictive samples."
            )

        return posterior_predictive_samples[self.output_var]

    @property
    def id(self) -> str:
        """Generate a unique hash value for the model.

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

    def graphviz(self, **kwargs):
        """Get the graphviz representation of the model.

        Parameters
        ----------
        **kwargs
            Keyword arguments for the `pm.model_to_graphviz` function

        Returns
        -------
        graphviz.Digraph

        """
        return pm.model_to_graphviz(self.model, **kwargs)

    prior = create_idata_accessor(
        "prior",
        "The model hasn't been sampled yet, call .sample_prior_predictive() first",
    )
    prior_predictive = create_idata_accessor(
        "prior_predictive",
        "The model hasn't been sampled yet, call .sample_prior_predictive() first",
    )
    posterior = create_idata_accessor(
        "posterior", "The model hasn't been fit yet, call .fit() first"
    )

    posterior_predictive = create_idata_accessor(
        "posterior_predictive",
        "The model hasn't been fit yet, call .sample_posterior_predictive() first",
    )
    predictions = create_idata_accessor(
        "predictions",
        "Call the 'sample_posterior_predictive' method with predictions=True first.",
    )
