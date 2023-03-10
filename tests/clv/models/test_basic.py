import warnings
from unittest.mock import patch

import pymc as pm
import pytest
from arviz import InferenceData, from_dict
from pandas import Series

from pymc_marketing.clv.models.basic import CLVModel


class CLVModelTest(CLVModel):
    _model_name = "CLVModelTest"

    def __init__(self):
        super().__init__()
        with pm.Model(coords={"test_x_dim": [99]}) as self.model:
            x = pm.HalfNormal("x", 100, dims=("test_x_dim",))
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

    def test_fit_MAP(self):
        model = CLVModelTest()

        idata = model.fit(fit_method="map")
        assert isinstance(idata, InferenceData)
        assert len(idata.posterior.chain) == 1
        assert len(idata.posterior.draw) == 1
        assert idata.posterior["x"].dims == ("chain", "draw", "test_x_dim")
        assert model.fit_result is idata

        # Check that summary only includes single value
        summ = model.fit_summary()
        assert isinstance(summ, Series)
        assert summ.name == "value"

    @patch("arviz.summary", return_value="fake_summary")
    def test_fit_summary(self, dummy_summary):
        model = CLVModelTest()
        model._fit_result = "fake_fit"
        fake_mcmc_fit = from_dict({"x": [[0, 0]]})
        model._fit_result = fake_mcmc_fit
        assert model.fit_summary(opt_kwarg="opt_kwarg") == "fake_summary"
        dummy_summary.assert_called_once_with(fake_mcmc_fit, opt_kwarg="opt_kwarg")

    def test_repr(self):
        model = CLVModelTest()
        assert model.__repr__() == "CLVModelTest\nx ~ N**+(0, 100)\ny ~ N(x, 1)"

    def test_fit_override(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            model = CLVModelTest()
            model.fit_result = 1
            with pytest.warns(UserWarning, match="Overriding pre-existing fit_result"):
                model.fit_result = 2

    def test_check_prior_ndim(self):
        prior = pm.Normal.dist(shape=(5,))  # ndim = 1
        with pytest.raises(
            ValueError, match="must be have 0 ndims, but it has 1 ndims"
        ):
            # Default ndim=0
            CLVModel._check_prior_ndim(prior)
        CLVModel._check_prior_ndim(prior, ndim=1)
        with pytest.raises(
            ValueError, match="must be have 2 ndims, but it has 1 ndims"
        ):
            CLVModel._check_prior_ndim(prior, ndim=2)

    def test_process_priors(self):
        prior1 = pm.Normal.dist()
        prior2 = pm.HalfNormal.dist()

        ret_prior1, ret_prior2 = CLVModel._process_priors(prior1, prior2)

        assert ret_prior1 is prior1
        assert ret_prior2 is prior2
        assert ret_prior1.str_repr() == "N(0, 1)"
        assert ret_prior2.str_repr() == "N**+(0, 1)"

        with pytest.raises(ValueError, match="Prior variables must be unique"):
            CLVModel._process_priors(prior1, prior2, prior1)
