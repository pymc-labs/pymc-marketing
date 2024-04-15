import os

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from arviz import InferenceData, from_dict

from pymc_marketing.clv.models.basic import CLVModel
from tests.conftest import set_model_fit


class CLVModelTest(CLVModel):
    _model_type = "CLVModelTest"

    def __init__(self, data=None, **kwargs):
        if data is None:
            data = pd.DataFrame({"y": np.random.randn(10)})
        super().__init__(data=data, **kwargs)

    @property
    def default_model_config(self):
        return {
            "x": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}},
        }

    def build_model(self):
        x_prior = self._create_distribution(self.model_config["x"])
        with pm.Model() as self.model:
            x = self.model.register_rv(x_prior, name="x")
            pm.Normal("y", mu=x, sigma=1, observed=self.data["y"])


@pytest.fixture(scope="module")
def posterior():
    # Create a random numpy array for posterior samples
    posterior_samples = np.random.randn(
        4, 100, 2
    )  # shape convention: (chain, draw, *shape)

    # Create a dictionary for posterior
    posterior_dict = {"theta": posterior_samples}
    return from_dict(posterior=posterior_dict)


class TestCLVModel:
    def test_repr(self):
        model = CLVModelTest()
        assert model.__repr__() == "CLVModelTest"

        model.build_model()
        assert model.__repr__() == "CLVModelTest\nx ~ Normal(0, 1)\ny ~ Normal(x, 1)"

    def test_create_distribution_from_wrong_prior(self):
        model = CLVModelTest()
        with pytest.raises(
            ValueError,
            match="Distribution definitely_not_PyMC_dist does not exist in PyMC",
        ):
            model._create_distribution(
                {"dist": "definitely_not_PyMC_dist", "kwargs": {"alpha": 1, "beta": 1}}
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
        model.fit(tune=0, chains=2, draws=5)
        model.save("test_model")
        model2 = model.load("test_model")
        assert model2.fit_result is not None
        assert model2.model is not None
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
        model.fit_result = fake_fit

    def test_fit_summary_for_mcmc(self):
        model = CLVModelTest()
        model.build_model()
        model.fit(tune=0, chains=2, draws=5)
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
        mock_basic.fit(tune=0, chains=2, draws=5)
        mock_basic.save("test_model")

        # Apply the monkeypatch for the property
        monkeypatch.setattr(CLVModelTest, "id", property(mock_property))
        with pytest.raises(
            ValueError,
            match="Inference data not compatible with CLVModelTest",
        ):
            CLVModelTest.load("test_model")
        os.remove("test_model")

    def test_thin_fit_result(self):
        data = pd.DataFrame(dict(y=[-3, -2, -1]))
        model = CLVModelTest(data=data)
        model.build_model()
        fake_idata = from_dict(dict(x=np.random.normal(size=(4, 1000))))
        set_model_fit(model, fake_idata)

        thin_model = model.thin_fit_result(keep_every=20)
        assert thin_model is not model
        assert thin_model.idata is not model.idata
        assert len(thin_model.idata.posterior["x"].chain) == 4
        assert len(thin_model.idata.posterior["x"].draw) == 50
        assert thin_model.data is not model.data
        assert np.all(thin_model.data == model.data)
