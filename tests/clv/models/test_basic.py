from unittest.mock import patch

import pymc as pm
import pytest
from arviz import InferenceData

from pymc_marketing.clv.models.basic import CLVModel


class CLVModelTest(CLVModel):
    _model_name = "CLVModelTest"

    def __init__(self):
        super().__init__()
        with pm.Model() as self.model:
            x = pm.Normal("x", 100)
            pm.Normal("y", x, observed=[1, 3, 3, 3, 5])


class TestCLVModel:
    def test_fit(self):
        model = CLVModelTest()

        with pytest.raises(RuntimeError, match="The model hasn't been fit yet"):
            model.fit_result

        idata = model.fit(
            tune=5,
            chains=2,
            draws=10,
            compute_convergence_checks=False,
        )
        assert isinstance(idata, InferenceData)
        assert len(idata.posterior.chain) == 2
        assert len(idata.posterior.draw) == 10
        assert model.fit_result is idata

    @patch("arviz.summary", return_value="fake_summary")
    def test_fit_summary(self, dummy_summary):
        model = CLVModelTest()
        model._fit_result = "fake_fit"
        assert model.fit_summary(opt_kwarg="opt_kwarg") == "fake_summary"
        dummy_summary.assert_called_once_with("fake_fit", opt_kwarg="opt_kwarg")

    def test_repr(self):
        model = CLVModelTest()
        assert model.__repr__() == "CLVModelTest\nx ~ N(100, 1)\ny ~ N(x, 1)"
