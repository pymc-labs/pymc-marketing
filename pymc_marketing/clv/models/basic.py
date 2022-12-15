import arviz as az
import pymc as pm


class CLVModel:
    _model_name = ""

    def __init__(self):
        self._fit_result = None

        with pm.Model() as self.model:
            pass

    def fit(self, *args, **kwargs):
        with self.model:
            self._fit_result = pm.sample(*args, **kwargs)
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
