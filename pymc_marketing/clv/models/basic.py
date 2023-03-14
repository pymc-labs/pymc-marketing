import types
import warnings
from typing import Tuple

import arviz as az
import pymc as pm
from pymc import str_for_dist
from pymc.backends import NDArray
from pymc.backends.base import MultiTrace
from pytensor.tensor import TensorVariable


class CLVModel:
    _model_name = ""

    def __init__(self):
        self._fit_result = None

        with pm.Model() as self.model:
            pass

    @staticmethod
    def _check_prior_ndim(prior, ndim=0):
        if prior.ndim != ndim:
            raise ValueError(
                f"Prior variable {prior} must be have {ndim} ndims, but it has {prior.ndim} ndims."
            )

    @staticmethod
    def _process_priors(*priors: TensorVariable) -> Tuple[TensorVariable]:
        """Check that each prior variable is unique and attach `str_repr` method."""
        if len(priors) != len(set(priors)):
            raise ValueError("Prior variables must be unique")

        # Related to https://github.com/pymc-devs/pymc/issues/6311
        for prior in priors:
            prior.str_repr = types.MethodType(str_for_dist, prior)
        return priors

    def fit(self, fit_method="mcmc", **kwargs):
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
        if fit_method == "mcmc":
            res = self._fit_mcmc(**kwargs)
        elif fit_method == "map":
            res = self._fit_MAP(**kwargs)
        else:
            raise ValueError(
                f"Fit method options are ['mcmc', 'map'], got: {fit_method}"
            )
        self.fit_result = res
        return res

    def _fit_mcmc(self, **kwargs):
        """Draw samples from model posterior using MCMC sampling"""
        with self.model:
            return pm.sample(**kwargs)

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
        return pm.to_inference_data(trace, model=model)

    @property
    def fit_result(self):
        if self._fit_result is None:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
        return self._fit_result

    @fit_result.setter
    def fit_result(self, res):
        if self._fit_result is not None:
            warnings.warn("Overriding pre-existing fit_result")
        self._fit_result = res

    def fit_summary(self, **kwargs):
        res = self.fit_result
        # Map fitting only gives one value, so we return it. We use arviz
        # just to get it nicely into a DataFrame
        if res.posterior.chain.size == 1 and res.posterior.draw.size == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = az.summary(self.fit_result, **kwargs, kind="stats")
            return res["mean"].rename("value")
        else:
            return az.summary(self.fit_result, **kwargs)

    def __repr__(self):
        return f"{self._model_name}\n{self.model.str_repr()}"
