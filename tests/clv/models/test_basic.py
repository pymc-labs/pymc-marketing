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
import os

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from arviz import InferenceData, from_dict

from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.prior import Prior
from tests.conftest import mock_fit_MAP, mock_sample, set_model_fit


class CLVModelTest(CLVModel):
    _model_type = "CLVModelTest"

    def __init__(
        self,
        data=None,
        model_config=None,
        sampler_config: dict | None = None,
    ):
        if data is None:
            data = pd.DataFrame({"y": np.random.randn(10)})

        super().__init__(
            data=data,
            model_config=model_config,
            sampler_config=sampler_config,
            non_distributions=[],
        )

    @property
    def default_model_config(self):
        return {
            "x": Prior("Normal", mu=0, sigma=1),
        }

    def build_model(self):
        with pm.Model() as self.model:
            x = self.model_config["x"].create_variable("x")
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

    def test_fit_mcmc(self, mocker):
        model = CLVModelTest()

        mocker.patch("pymc.sample", mock_sample)

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

    def test_fit_map(self, mocker):
        model = CLVModelTest()

        mocker.patch("pymc_marketing.clv.models.basic.CLVModel._fit_MAP", mock_fit_MAP)
        idata = model.fit(fit_method="map")

        assert isinstance(idata, InferenceData)
        assert len(idata.posterior.chain) == 1
        assert len(idata.posterior.draw) == 1
        assert model.fit_result is idata.posterior
        # Check that summary only includes single value
        summ = model.fit_summary()
        assert isinstance(summ, pd.Series)
        assert summ.name == "value"

    def test_fit_demz(self, mocker):
        model = CLVModelTest()

        mocker.patch("pymc.sample", mock_sample)

        idata = model.fit(
            fit_method="demz",
            tune=5,
            chains=2,
            draws=10,
            compute_convergence_checks=False,
        )

        assert isinstance(idata, InferenceData)
        assert len(idata.posterior.chain) == 2
        assert len(idata.posterior.draw) == 10
        assert model.fit_result is idata.posterior

    def test_wrong_fit_method(self):
        model = CLVModelTest()
        with pytest.raises(
            ValueError,
            match=r"Fit method options are \['mcmc', 'map', 'demz'\], got: wrong_method",
        ):
            model.fit(fit_method="wrong_method")

    def test_load(self, mocker):
        model = CLVModelTest()

        mocker.patch("pymc.sample", mock_sample)

        model.fit(tune=0, chains=2, draws=5)
        model.save("test_model")
        model2 = model.load("test_model")
        assert model2.fit_result is not None
        assert model2.model is not None
        os.remove("test_model")

    def test_default_sampler_config(self):
        model = CLVModelTest()
        assert model.sampler_config == {}

    def test_fit_summary_for_mcmc(self, mocker):
        model = CLVModelTest()

        mocker.patch("pymc.sample", mock_sample)
        model.fit(tune=0, chains=2, draws=5)
        summ = model.fit_summary()
        assert isinstance(summ, pd.DataFrame)

    def test_serializable_model_config(self):
        model = CLVModelTest()
        serializable_config = model._serializable_model_config
        assert isinstance(serializable_config, dict)
        assert serializable_config == model.model_config

    def test_fail_id_after_load(self, mocker, monkeypatch):
        # This is the new behavior for the property
        def mock_property(self):
            return "for sure not correct id"

        # Now create an instance of MyClass
        mock_basic = CLVModelTest()
        mocker.patch("pymc.sample", mock_sample)
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

    def test_model_config_warns(self) -> None:
        model_config = {
            "x": {"dist": "StudentT", "kwargs": {"mu": 0, "sigma": 5, "nu": 15}},
        }
        with pytest.warns(DeprecationWarning, match="x is automatically"):
            model = CLVModelTest(model_config=model_config)

        assert model.model_config == {
            "x": Prior("StudentT", mu=0, sigma=5, nu=15),
        }
