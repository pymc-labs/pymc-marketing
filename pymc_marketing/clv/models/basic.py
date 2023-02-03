import arviz as az
import pymc as pm
from pymc.backends import NDArray
from pymc.backends.base import MultiTrace


class CLVModel:
    _model_name = ""

    def __init__(self):
        self._fit_result = None

        with pm.Model() as self.model:
            pass

    def fit(self, fitting_method="mcmc", **kwargs):
        """Infer model posterior

        Parameters
        ----------
        fitting_method: str
            Method used to fit the model. Options are:
            - "mcmc": Samples from the posterior via `pymc.sample` (default)
            - "map": Finds maximum a posteriori via `pymc.find_MAP`
        kwargs:
            Other keyword arguments passed to the underlying PyMC routines
        """
        if fitting_method == "mcmc":
            return self._fit_mcmc(**kwargs)
        elif fitting_method == "map":
            return self._fit_MAP(**kwargs)
        else:
            raise ValueError(
                f"Fitting method options are ['mcmc', 'map'], got: {fitting_method}"
            )

    def _fit_mcmc(self, **kwargs):
        """Draw samples from model posterior using MCMC sampling"""
        with self.model:
            self._fit_result = pm.sample(**kwargs)
        return self._fit_result

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
        self._fit_result = pm.to_inference_data(trace, model=model)
        return self._fit_result

    @property
    def fit_result(self):
        if self._fit_result is None:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
        return self._fit_result

    def fit_summary(self, **kwargs):
        return az.summary(self.fit_result, **kwargs)

    def __repr__(self):
        return f"{self._model_name}\n{self.model.str_repr()}"
