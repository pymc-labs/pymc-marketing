import os

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from arviz import InferenceData

from pymc_marketing.clv.models.basic import CLVModel


class CLVModelTest(CLVModel):
    _model_type = "CLVModelTest"

    def __init__(self, dataset=None, model_config=None, sampler_config=None):
        super().__init__()
        self.data = pd.DataFrame({"y": np.random.randn(100)})
        self.a = self._create_distribution(self.model_config["a"])
        self._process_priors(self.a)

    @property
    def default_model_config(self):
        return {
            "a": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}},
            "b": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}},
        }

    def build_model(self):
        with pm.Model() as self.model:
            self.a = pm.Normal("a", mu=0, sigma=1)
            self.b = pm.Normal("b", mu=0, sigma=1)
            self.y = pm.Normal(
                "y", mu=self.a + self.b, sigma=1, observed=self.data["y"]
            )


class TestCLVModel:
    def test_repr(self):
        model = CLVModelTest()
        model.build_model()
        assert (
            model.__repr__()
            == "CLVModelTest\na ~ Normal(0, 1)\nb ~ Normal(0, 1)\ny ~ Normal(f(a, b), 1)"
        )

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
        assert ret_prior1.str_repr() == "Normal(0, 1)"
        assert ret_prior2.str_repr() == "HalfNormal(0, 1)"

        with pytest.raises(ValueError, match="Prior variables must be unique"):
            CLVModel._process_priors(prior1, prior2, prior1)

    def test_create_distribution_from_wrong_prior(self):
        model = CLVModelTest()
        with pytest.raises(
            ValueError,
            match="Distribution definately_not_PyMC_dist does not exist in PyMC",
        ):
            model._create_distribution(
                {"dist": "definately_not_PyMC_dist", "kwargs": {"alpha": 1, "beta": 1}}
            )

    def test_fit_mcmc(self):
        model = CLVModelTest()
        model.build_model()
        idata = model.fit(
            tune=5,
            chains=2,
            draws=10,
            compute_convergence_checks=False,
        )
        assert isinstance(idata, InferenceData)
        assert len(idata.posterior.chain) == 2
        assert len(idata.posterior.draw) == 10
        assert model.fit_result is idata.posterior

    def test_fit_map(self):
        model = CLVModelTest()
        model.build_model()
        idata = model.fit(fit_method="map")
        assert isinstance(idata, InferenceData)
        assert len(idata.posterior.chain) == 1
        assert len(idata.posterior.draw) == 1
        assert model.fit_result is idata.posterior
        # Check that summary only includes single value
        summ = model.fit_summary()
        assert isinstance(summ, pd.Series)
        assert summ.name == "value"

    def test_wrong_fit_method(self):
        model = CLVModelTest()
        with pytest.raises(
            ValueError,
            match=r"Fit method options are \['mcmc', 'map'\], got: wrong_method",
        ):
            model.fit(fit_method="wrong_method")

    def test_fit_result_error(self):
        model = CLVModelTest()
        with pytest.raises(RuntimeError, match="The model hasn't been fit yet"):
            model.fit_result

    def test_load(self):
        model = CLVModelTest()
        model.build_model()
        model.fit(draws=100, chains=2, random_seed=1234)
        model.save("test_model")
        try:
            model2 = model.load("test_model")
            assert model2.fit_result is not None
            assert model2.model is not None
        finally:
            os.remove("test_model")

    def test_default_sampler_config(self):
        model = CLVModelTest()
        assert model.sampler_config == {}

    def test_set_fit_result(self):
        model = CLVModelTest()
        model.build_model()
        model.idata = None
        fake_fit = pm.sample_prior_predictive(
            samples=50, model=model.model, random_seed=1234
        )
        fake_fit.add_groups(dict(posterior=fake_fit.prior))
        model.fit_result = fake_fit
        with pytest.warns(UserWarning, match="Overriding pre-existing fit_result"):
            model.fit_result = fake_fit
        model.idata = None
        model.sample_prior_predictive(samples=50, extend_idata=True)
        model.fit_result = fake_fit

    def test_fit_summary_for_mcmc(self):
        model = CLVModelTest()
        model.build_model()
        model.fit()
        summ = model.fit_summary()
        assert isinstance(summ, pd.DataFrame)

    def test_serializable_model_config(self):
        model = CLVModelTest()
        serializable_config = model._serializable_model_config
        assert isinstance(serializable_config, dict)
        assert serializable_config == model.model_config

    def test_fail_id_after_load(self, monkeypatch):
        # This is the new behavior for the property
        def mock_property(self):
            return "for sure not correct id"

        # Now create an instance of MyClass
        mock_basic = CLVModelTest()

        # Check that the property returns the new value
        mock_basic.fit()
        mock_basic.save("test_model")
        try:
            # Apply the monkeypatch for the property
            monkeypatch.setattr(CLVModelTest, "id", property(mock_property))
            with pytest.raises(
                ValueError,
                match="The file 'test_model' does not contain an inference data of the same model or configuration as 'CLVModelTest'",
            ):
                CLVModelTest.load("test_model")
        finally:
            os.remove("test_model")
