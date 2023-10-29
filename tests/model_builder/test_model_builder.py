#   Copyright 2023 The PyMC Developers
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

import hashlib
import json
import sys
import tempfile
from typing import Dict

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pymc_marketing.model_builder import ModelBuilder


@pytest.fixture(scope="module")
def toy_X():
    x = np.linspace(start=0, stop=1, num=100)
    X = pd.DataFrame({"input": x})
    return X


@pytest.fixture(scope="module")
def toy_y(toy_X):
    y = 5 * toy_X["input"] + 3
    y = y + np.random.normal(0, 1, size=len(toy_X))
    y = pd.Series(y, name="output")
    return y


@pytest.fixture(scope="module")
def fitted_model_instance(toy_X, toy_y):
    sampler_config = {
        "draws": 100,
        "tune": 100,
        "chains": 2,
        "target_accept": 0.95,
    }
    model_config = {
        "a": {"loc": 0, "scale": 10, "dims": ("numbers",)},
        "b": {"loc": 0, "scale": 10},
        "obs_error": 2,
    }
    model = test_ModelBuilder(
        model_config=model_config,
        sampler_config=sampler_config,
        test_parameter="test_paramter",
    )
    model.fit(
        toy_X,
        chains=1,
        draws=100,
        tune=100,
    )
    return model


@pytest.fixture(scope="module")
def not_fitted_model_instance(toy_X, toy_y):
    sampler_config = {"draws": 100, "tune": 100, "chains": 2, "target_accept": 0.95}
    model_config = {
        "a": {"loc": 0, "scale": 10, "dims": ("numbers",)},
        "b": {"loc": 0, "scale": 10},
        "obs_error": 2,
    }
    model = test_ModelBuilder(
        model_config=model_config,
        sampler_config=sampler_config,
        test_parameter="test_paramter",
    )
    return model


class test_ModelBuilder(ModelBuilder):
    def __init__(self, model_config=None, sampler_config=None, test_parameter=None):
        self.test_parameter = test_parameter
        super().__init__(model_config=model_config, sampler_config=sampler_config)

    _model_type = "test_model"
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, model_config=None):
        coords = {"numbers": np.arange(len(X))}
        self._generate_and_preprocess_model_data(X, y)
        with pm.Model(coords=coords) as self.model:
            if model_config is None:
                model_config = self.default_model_config
            x = pm.MutableData("x", self.X["input"].values)
            y_data = pm.MutableData("y_data", self.y)

            # prior parameters
            a_loc = model_config["a"]["loc"]
            a_scale = model_config["a"]["scale"]
            b_loc = model_config["b"]["loc"]
            b_scale = model_config["b"]["scale"]
            obs_error = model_config["obs_error"]

            # priors
            a = pm.Normal("a", a_loc, sigma=a_scale, dims=model_config["a"]["dims"])
            b = pm.Normal("b", b_loc, sigma=b_scale)
            obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

            # observed data
            pm.Normal("output", a + b * x, obs_error, shape=x.shape, observed=y_data)

    def _save_input_params(self, idata):
        idata.attrs["test_paramter"] = json.dumps(self.test_parameter)

    @property
    def output_var(self):
        return "output"

    def _data_setter(self, x: pd.Series, y: pd.Series = None):
        with self.model:
            pm.set_data({"x": x.values})
            if y is not None:
                pm.set_data({"y_data": y.values})

    @property
    def _serializable_model_config(self):
        return self.model_config

    def _generate_and_preprocess_model_data(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    @property
    def default_model_config(self) -> Dict:
        return {
            "a": {"loc": 0, "scale": 10, "dims": ("numbers",)},
            "b": {"loc": 0, "scale": 10},
            "obs_error": 2,
        }

    @property
    def default_sampler_config(self) -> Dict:
        return {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 3,
            "target_accept": 0.95,
        }


def test_save_input_params(fitted_model_instance):
    assert fitted_model_instance.idata.attrs["test_paramter"] == '"test_paramter"'


def test_save_load(fitted_model_instance):
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    fitted_model_instance.save(temp.name)
    test_builder2 = test_ModelBuilder.load(temp.name)
    assert fitted_model_instance.idata.groups() == test_builder2.idata.groups()
    assert fitted_model_instance.id == test_builder2.id
    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred1 = fitted_model_instance.predict(prediction_data["input"])
    pred2 = test_builder2.predict(prediction_data["input"])
    assert pred1.shape == pred2.shape
    temp.close()


def test_initial_build_and_fit(fitted_model_instance, check_idata=True) -> ModelBuilder:
    if check_idata:
        assert fitted_model_instance.idata is not None
        assert "posterior" in fitted_model_instance.idata.groups()


def test_save_without_fit_raises_runtime_error():
    model_builder = test_ModelBuilder()
    with pytest.raises(RuntimeError):
        model_builder.save("saved_model")


def test_empty_sampler_config_fit(toy_X, toy_y):
    sampler_config = {}
    model_builder = test_ModelBuilder(sampler_config=sampler_config)
    model_builder.idata = model_builder.fit(
        X=toy_X, y=toy_y, chains=1, draws=100, tune=100
    )
    assert model_builder.idata is not None
    assert "posterior" in model_builder.idata.groups()


def test_fit(fitted_model_instance):
    assert fitted_model_instance.idata is not None
    assert "posterior" in fitted_model_instance.idata.groups()
    assert fitted_model_instance.idata.posterior.dims["draw"] == 100

    prediction_data = pd.DataFrame(
        {"input": np.random.uniform(low=0, high=1, size=100)}
    )
    fitted_model_instance.predict(prediction_data["input"])
    post_pred = fitted_model_instance.sample_posterior_predictive(
        prediction_data["input"], extend_idata=True, combined=True
    )
    post_pred[fitted_model_instance.output_var].shape[0] == prediction_data.input.shape


def test_fit_no_y(toy_X):
    model_builder = test_ModelBuilder()
    model_builder.idata = model_builder.fit(X=toy_X, chains=1, draws=100, tune=100)
    assert model_builder.model is not None
    assert model_builder.idata is not None
    assert "posterior" in model_builder.idata.groups()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Permissions for temp files not granted on windows CI.",
)
def test_predict(fitted_model_instance):
    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = fitted_model_instance.predict(prediction_data["input"])
    # Perform elementwise comparison using numpy
    assert type(pred) == np.ndarray
    assert len(pred) > 0


@pytest.mark.parametrize("combined", [True, False])
def test_sample_posterior_predictive(fitted_model_instance, combined):
    n_pred = 100
    x_pred = np.random.uniform(low=0, high=1, size=n_pred)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = fitted_model_instance.sample_posterior_predictive(
        prediction_data["input"], combined=combined, extend_idata=True
    )
    chains = fitted_model_instance.idata.sample_stats.dims["chain"]
    draws = fitted_model_instance.idata.sample_stats.dims["draw"]
    expected_shape = (n_pred, chains * draws) if combined else (chains, draws, n_pred)
    assert pred[fitted_model_instance.output_var].shape == expected_shape
    assert np.issubdtype(pred[fitted_model_instance.output_var].dtype, np.floating)


def test_model_config_formatting():
    model_config = {
        "a": {
            "loc": [0, 0],
            "scale": 10,
            "dims": [
                "x",
            ],
        },
    }
    model_builder = test_ModelBuilder()
    converted_model_config = model_builder._model_config_formatting(model_config)
    np.testing.assert_equal(converted_model_config["a"]["dims"], ("x",))
    np.testing.assert_equal(converted_model_config["a"]["loc"], np.array([0, 0]))


def test_id():
    model_builder = test_ModelBuilder()
    expected_id = hashlib.sha256(
        str(model_builder.model_config.values()).encode()
        + model_builder.version.encode()
        + model_builder._model_type.encode()
    ).hexdigest()[:16]

    assert model_builder.id == expected_id
