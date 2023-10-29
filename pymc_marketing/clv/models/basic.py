import json
import types
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from pymc import str_for_dist
from pymc.backends import NDArray
from pymc.backends.base import MultiTrace
from pytensor.tensor import TensorVariable
from xarray import Dataset

from pymc_marketing.model_builder import ModelBuilder


class CLVModel(ModelBuilder):
    _model_type = ""

    def __init__(
        self,
        model_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
    ):
        super().__init__(model_config, sampler_config)

    def __repr__(self):
        return f"{self._model_type}\n{self.model.str_repr()}"

    def fit(  # type: ignore
        self,
        fit_method: str = "mcmc",
        **kwargs,
    ) -> az.InferenceData:
        """Infer model posterior

        Parameters
        ----------
        fit_method: str
            Method used to fit the model. Options are:
            - "mcmc": Samples from the posterior via `pymc.sample` (default)
            - "map": Finds maximum a posteriori via `pymc.find_MAP`
        kwargs:
            Other keyword arguments passed to the underlying PyMC routines
        """

        self.build_model()  # type: ignore

        if fit_method == "mcmc":
            self._fit_mcmc(**kwargs)
        elif fit_method == "map":
            self._fit_MAP(**kwargs)
        else:
            raise ValueError(
                f"Fit method options are ['mcmc', 'map'], got: {fit_method}"
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            self.idata.add_groups(fit_data=self.data.to_xarray())  # type: ignore

        return self.idata

    def _fit_mcmc(self, **kwargs) -> az.InferenceData:
        """
        Fit a model using the data passed as a parameter.
        Sets attrs to inference data of the model.


        Parameters
        ----------
        X : array-like if sklearn is available, otherwise array, shape (n_obs, n_features)
            The training input samples.
        y : array-like if sklearn is available, otherwise array, shape (n_obs,)
            The target values (real numbers).
        **kwargs : Any
            Custom sampler settings can be provided in form of keyword arguments.

        Returns
        -------
        self : az.InferenceData
            returns inference data of the fitted model.
        """
        sampler_config = {}
        if self.sampler_config is not None:
            sampler_config = self.sampler_config.copy()
        sampler_config.update(**kwargs)
        self.idata = self.sample_model(**sampler_config)
        return self.idata

    def sample_model(self, **kwargs):
        """
        Sample from the PyMC model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the PyMC sampler.

        Returns
        -------
        xarray.Dataset
            The PyMC samples dataset.

        Raises
        ------
        RuntimeError
            If the PyMC model hasn't been built yet.

        """
        if self.model is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first or call .fit() instead."
            )

        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}
            idata = pm.sample(**sampler_args)

        self.set_idata_attrs(idata)
        return idata

    def _fit_MAP(self, **kwargs):
        """Find model maximum a posteriori using scipy optimizer"""
        model = self.model
        map_res = pm.find_MAP(model=model, **kwargs)
        # Filter non-value variables
        value_vars_names = set(v.name for v in model.value_vars)
        map_res = {k: v for k, v in map_res.items() if k in value_vars_names}
        # Convert map result to InferenceData
        map_strace = NDArray(model=model)
        map_strace.setup(draws=1, chain=0)
        map_strace.record(map_res)
        map_strace.close()
        trace = MultiTrace([map_strace])
        idata = pm.to_inference_data(trace, model=model)
        self.set_idata_attrs(idata)
        self.idata = idata
        return self.idata

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
        dataset = idata.fit_data.to_dataframe()

        model = cls(
            dataset,
            model_config=json.loads(idata.attrs["model_config"]),  # type: ignore
            sampler_config=json.loads(idata.attrs["sampler_config"]),
        )
        model.idata = idata

        model.build_model()  # type: ignore

        if model.id != idata.attrs["id"]:
            raise ValueError(
                f"The file '{fname}' does not contain an inference data of the same model or configuration as '{cls._model_type}'"
            )
        # All previously used data is in idata.

        return model

    @staticmethod
    def _check_prior_ndim(prior, ndim: int = 0):
        if prior.ndim != ndim:
            raise ValueError(
                f"Prior variable {prior} must be have {ndim} ndims, but it has {prior.ndim} ndims."
            )

    @staticmethod
    def _create_distribution(dist: Dict, ndim: int = 0) -> TensorVariable:
        try:
            prior_distribution = getattr(pm, dist["dist"]).dist(**dist["kwargs"])
            CLVModel._check_prior_ndim(prior_distribution, ndim)
        except AttributeError:
            raise ValueError(f"Distribution {dist['dist']} does not exist in PyMC")
        return prior_distribution

    @staticmethod
    def _process_priors(
        *priors: TensorVariable, check_ndim: bool = True
    ) -> Tuple[TensorVariable, ...]:
        """Check that each prior variable is unique and attach `str_repr` method."""
        if len(priors) != len(set(priors)):
            raise ValueError("Prior variables must be unique")
        # Related to https://github.com/pymc-devs/pymc/issues/6311
        for prior in priors:
            prior.str_repr = types.MethodType(str_for_dist, prior)  # type: ignore
        return priors

    @property
    def default_sampler_config(self) -> Dict:
        return {}

    @property
    def _serializable_model_config(self) -> Dict:
        return self.model_config

    def sample_prior_predictive(  # type: ignore
        self,
        samples: int = 1000,
        extend_idata: bool = True,
        combined: bool = True,
        **kwargs,
    ):
        if self.model is not None:
            with self.model:  # sample with new input data
                prior_pred: az.InferenceData = pm.sample_prior_predictive(
                    samples, **kwargs
                )
                self.set_idata_attrs(prior_pred)
                if extend_idata:
                    if self.idata is not None:
                        self.idata.extend(prior_pred)
                    else:
                        self.idata = prior_pred

        prior_predictive_samples = az.extract(
            prior_pred, "prior_predictive", combined=combined
        )

        return prior_predictive_samples

    @property
    def prior_predictive(self) -> az.InferenceData:
        if self.idata is None or "prior_predictive" not in self.idata:
            raise RuntimeError(
                "No prior predictive samples available, call sample_prior_predictive() first"
            )
        return self.idata["prior_predictive"]

    @property
    def fit_result(self) -> Dataset:
        if self.idata is None or "posterior" not in self.idata:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
        return self.idata["posterior"]

    @fit_result.setter
    def fit_result(self, res: az.InferenceData) -> None:
        if self.idata is None:
            self.idata = res
        elif "posterior" in self.idata:
            warnings.warn("Overriding pre-existing fit_result")
            self.idata.posterior = res
        else:
            self.idata.posterior = res

    def fit_summary(self, **kwargs):
        res = self.fit_result
        # Map fitting only gives one value, so we return it. We use arviz
        # just to get it nicely into a DataFrame
        if res.chain.size == 1 and res.draw.size == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = az.summary(self.fit_result, **kwargs, kind="stats")
            return res["mean"].rename("value")
        else:
            return az.summary(self.fit_result, **kwargs)

    @property
    def output_var(self):
        pass

    def _generate_and_preprocess_model_data(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.Series, np.ndarray[Any, Any]],
    ) -> None:
        pass

    def _data_setter(self):
        pass
