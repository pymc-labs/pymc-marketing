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

import arviz as az
import graphviz
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from pymc_marketing.model_builder import ModelBuilder, create_sample_kwargs


@pytest.fixture(scope="module")
def toy_X():
    x = np.linspace(start=0, stop=1, num=100)
    return pd.DataFrame({"input": x})


@pytest.fixture(scope="module")
def toy_y(toy_X):
    rng = np.random.default_rng(42)
    y = 5 * toy_X["input"] + 3
    y = y + rng.normal(0, 1, size=len(toy_X))
    y = pd.Series(y, name="output")
    return y


@pytest.fixture(scope="module")
def fitted_model_instance(toy_X):
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
    model = ModelBuilderTest(
        model_config=model_config,
        sampler_config=sampler_config,
        test_parameter="test_parameter",
    )
    model.fit(
        toy_X,
        chains=1,
        draws=100,
        tune=100,
    )
    return model


@pytest.fixture(scope="module")
def not_fitted_model_instance():
    sampler_config = {"draws": 100, "tune": 100, "chains": 2, "target_accept": 0.95}
    model_config = {
        "a": {"loc": 0, "scale": 10, "dims": ("numbers",)},
        "b": {"loc": 0, "scale": 10},
        "obs_error": 2,
    }
    return ModelBuilderTest(
        model_config=model_config,
        sampler_config=sampler_config,
        test_parameter="test_paramter",
    )


class ModelBuilderTest(ModelBuilder):
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
            x = pm.Data("x", self.X["input"].values)
            y_data = pm.Data("y_data", self.y)

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

    def create_idata_attrs(self):
        attrs = super().create_idata_attrs()
        attrs["test_parameter"] = json.dumps(self.test_parameter)

        return attrs

    @property
    def output_var(self):
        return "output"

    def _data_setter(self, X: pd.DataFrame, y: pd.Series = None):
        with self.model:
            pm.set_data({"x": X["input"].values})
            if y is not None:
                y = y.values if isinstance(y, pd.Series) else y
                pm.set_data({"y_data": y})

    @property
    def _serializable_model_config(self):
        return self.model_config

    def _generate_and_preprocess_model_data(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    @property
    def default_model_config(self) -> dict:
        return {
            "a": {"loc": 0, "scale": 10, "dims": ("numbers",)},
            "b": {"loc": 0, "scale": 10},
            "obs_error": 2,
        }

    @property
    def default_sampler_config(self) -> dict:
        return {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 3,
            "target_accept": 0.95,
        }


def test_model_and_sampler_config():
    default = ModelBuilderTest()
    assert default.model_config == default.default_model_config
    assert default.sampler_config == default.default_sampler_config

    nondefault = ModelBuilderTest(
        model_config={"obs_error": 3}, sampler_config={"draws": 42}
    )
    assert nondefault.model_config != nondefault.default_model_config
    assert nondefault.sampler_config != nondefault.default_sampler_config
    assert nondefault.model_config == default.model_config | {"obs_error": 3}
    assert nondefault.sampler_config == default.sampler_config | {"draws": 42}


def test_save_input_params(fitted_model_instance):
    assert fitted_model_instance.idata.attrs["test_parameter"] == '"test_parameter"'


def test_save_load(fitted_model_instance):
    rng = np.random.default_rng(42)
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    fitted_model_instance.save(temp.name)
    test_builder2 = ModelBuilderTest.load(temp.name)

    assert fitted_model_instance.idata.groups() == test_builder2.idata.groups()
    assert fitted_model_instance.id == test_builder2.id
    assert fitted_model_instance.model_config == test_builder2.model_config
    assert fitted_model_instance.sampler_config == test_builder2.sampler_config

    x_pred = rng.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred1 = fitted_model_instance.predict(prediction_data)
    pred2 = test_builder2.predict(prediction_data)
    assert pred1.shape == pred2.shape
    temp.close()


def test_initial_build_and_fit(fitted_model_instance, check_idata=True) -> ModelBuilder:
    if check_idata:
        assert fitted_model_instance.idata is not None
        assert "posterior" in fitted_model_instance.idata.groups()


def test_save_without_fit_raises_runtime_error():
    model_builder = ModelBuilderTest()
    match = "The model hasn't been fit yet"
    with pytest.raises(RuntimeError, match=match):
        model_builder.save("saved_model")


def test_empty_sampler_config_fit(toy_X, toy_y):
    sampler_config = {}
    model_builder = ModelBuilderTest(sampler_config=sampler_config)
    model_builder.idata = model_builder.fit(
        X=toy_X, y=toy_y, chains=1, draws=100, tune=100
    )
    assert model_builder.idata is not None
    assert "posterior" in model_builder.idata.groups()


def test_fit(fitted_model_instance):
    rng = np.random.default_rng(42)
    assert fitted_model_instance.idata is not None
    assert "posterior" in fitted_model_instance.idata.groups()
    assert fitted_model_instance.idata.posterior.sizes["draw"] == 100

    prediction_data = pd.DataFrame({"input": rng.uniform(low=0, high=1, size=100)})
    fitted_model_instance.predict(prediction_data)
    post_pred = fitted_model_instance.sample_posterior_predictive(
        prediction_data, extend_idata=True, combined=True
    )
    assert (
        post_pred[fitted_model_instance.output_var].shape[0]
        == prediction_data.input.shape[0]
    )


def test_fit_no_t(toy_X):
    model_builder = ModelBuilderTest()
    model_builder.idata = model_builder.fit(X=toy_X, chains=1, draws=100, tune=100)
    assert model_builder.model is not None
    assert model_builder.idata is not None
    assert "posterior" in model_builder.idata.groups()


def test_fit_dup_Y(toy_X, toy_y):
    toy_X = pd.concat((toy_X, toy_y), axis=1)
    model_builder = ModelBuilderTest()

    with pytest.raises(
        ValueError,
        match="X includes a column named 'output', which conflicts with the target variable.",
    ):
        model_builder.fit(X=toy_X, chains=1, draws=100, tune=100)


def test_fit_result_error():
    model = ModelBuilderTest()
    with pytest.raises(RuntimeError, match="The model hasn't been fit yet"):
        model.fit_result


def test_set_fit_result(toy_X, toy_y):
    model = ModelBuilderTest()
    model.build_model(X=toy_X, y=toy_y)
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


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Permissions for temp files not granted on windows CI.",
)
def test_predict(fitted_model_instance):
    rng = np.random.default_rng(42)
    x_pred = rng.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = fitted_model_instance.predict(prediction_data)
    # Perform elementwise comparison using numpy
    assert isinstance(pred, np.ndarray)
    assert len(pred) > 0


@pytest.mark.parametrize("combined", [True, False])
def test_sample_posterior_predictive(fitted_model_instance, combined):
    rng = np.random.default_rng(42)
    n_pred = 100
    x_pred = rng.uniform(low=0, high=1, size=n_pred)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = fitted_model_instance.sample_posterior_predictive(
        prediction_data, combined=combined, extend_idata=True
    )
    chains = fitted_model_instance.idata.sample_stats.sizes["chain"]
    draws = fitted_model_instance.idata.sample_stats.sizes["draw"]
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
    model_builder = ModelBuilderTest()
    converted_model_config = model_builder._model_config_formatting(model_config)
    np.testing.assert_equal(converted_model_config["a"]["dims"], ("x",))
    np.testing.assert_equal(converted_model_config["a"]["loc"], np.array([0, 0]))


def test_id():
    model_builder = ModelBuilderTest()
    expected_id = hashlib.sha256(
        str(model_builder.model_config.values()).encode()
        + model_builder.version.encode()
        + model_builder._model_type.encode()
    ).hexdigest()[:16]

    assert model_builder.id == expected_id


@pytest.mark.parametrize("name", ["prior_predictive", "posterior_predictive"])
def test_sample_xxx_predictive_keeps_second(
    fitted_model_instance, toy_X, name: str
) -> None:
    rng = np.random.default_rng(42)
    method_name = f"sample_{name}"
    method = getattr(fitted_model_instance, method_name)

    X_pred = toy_X

    kwargs = {
        "X_pred": X_pred,
        "combined": False,
        "extend_idata": True,
        "random_seed": rng,
    }
    first_sample = method(**kwargs)
    second_sample = method(**kwargs)

    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(first_sample, second_sample)

    sample = getattr(fitted_model_instance.idata, name)
    xr.testing.assert_allclose(sample, second_sample)


def test_prediction_kwarg(fitted_model_instance, toy_X):
    result = fitted_model_instance.sample_posterior_predictive(
        toy_X,
        extend_idata=True,
        predictions=True,
    )
    assert "predictions" in fitted_model_instance.idata
    assert "predictions_constant_data" in fitted_model_instance.idata

    assert isinstance(result, xr.Dataset)


def test_fit_after_prior_keeps_prior(toy_X, toy_y):
    model = ModelBuilderTest()
    model.sample_prior_predictive(toy_X)
    assert "prior" in model.idata
    assert "prior_predictive" in model.idata

    model.fit(X=toy_X, y=toy_y, chains=1, draws=100, tune=100)
    assert "prior" in model.idata
    assert "prior_predictive" in model.idata


def test_second_fit(toy_X, toy_y):
    model = ModelBuilderTest()

    model.fit(X=toy_X, y=toy_y, chains=1, draws=100, tune=100)
    assert "posterior" in model.idata
    id_before = id(model.idata)
    assert "fit_data" in model.idata

    model.fit(X=toy_X, y=toy_y, chains=1, draws=100, tune=100)
    id_after = id(model.idata)

    assert id_before != id_after


class InsufficientModel(ModelBuilder):
    def __init__(
        self, model_config=None, sampler_config=None, new_parameter=None
    ) -> None:
        super().__init__(model_config=model_config, sampler_config=sampler_config)
        self.new_parameter = new_parameter

    def _data_setter(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        pass

    def build_model(self, X: pd.DataFrame, y: pd.Series, model_config=None) -> None:
        with pm.Model() as self.model:
            intercept = pm.Normal("intercept")
            sigma = pm.HalfNormal("sigma")

            pm.Normal("output", mu=intercept, sigma=sigma, observed=y)

    @property
    def output_var(self) -> str:
        return "output"

    @property
    def default_model_config(self) -> dict:
        return {}

    @property
    def default_sampler_config(self) -> dict:
        return {}

    def _generate_and_preprocess_model_data(
        self,
        X,
        y,
    ) -> None:
        pass

    def _serializable_model_config(self) -> dict[str, int | float | dict]:
        return {}


def test_insufficient_attrs() -> None:
    model = InsufficientModel()

    X_pred = [1, 2, 3]

    match = "__init__ has parameters that are not in the attrs"
    with pytest.raises(ValueError, match=match):
        model.sample_prior_predictive(X_pred=X_pred)


def test_incorrect_set_idata_attrs_override() -> None:
    class IncorrectSetAttrs(InsufficientModel):
        def create_idata_attrs(self) -> dict:
            return {"new_parameter": self.new_parameter}

    model = IncorrectSetAttrs()

    X_pred = [1, 2, 3]

    match = "Missing required keys in attrs"
    with pytest.raises(ValueError, match=match):
        model.sample_prior_predictive(X_pred=X_pred)


@pytest.mark.parametrize(
    "sampler_config, fit_kwargs, expected",
    [
        (
            {},
            {
                "progressbar": None,
                "random_seed": None,
            },
            {
                "progressbar": True,
            },
        ),
        (
            {
                "random_seed": 52,
                "progressbar": False,
            },
            {
                "progressbar": None,
                "random_seed": None,
            },
            {
                "progressbar": False,
                "random_seed": 52,
            },
        ),
        (
            {
                "random_seed": 52,
                "progressbar": True,
            },
            {
                "progressbar": False,
                "random_seed": 42,
            },
            {
                "progressbar": False,
                "random_seed": 42,
            },
        ),
    ],
    ids=[
        "no_sampler_config/defaults",
        "use_sampler_config",
        "override_sampler_config",
    ],
)
def test_create_sample_kwargs(sampler_config, fit_kwargs, expected) -> None:
    sampler_config_before = sampler_config.copy()
    assert create_sample_kwargs(sampler_config, **fit_kwargs) == expected

    # Doesn't override
    assert sampler_config_before == sampler_config


def create_int_seed():
    return 42


def create_rng_seed():
    return np.random.default_rng(42)


@pytest.mark.parametrize(
    "create_random_seed",
    [
        create_int_seed,
        create_rng_seed,
    ],
    ids=["int", "rng"],
)
def test_fit_random_seed_reproducibility(toy_X, toy_y, create_random_seed) -> None:
    sampler_config = {
        "chains": 1,
        "draws": 10,
        "tune": 5,
    }
    model = ModelBuilderTest(sampler_config=sampler_config)

    idata = model.fit(toy_X, toy_y, random_seed=create_random_seed())
    idata2 = model.fit(toy_X, toy_y, random_seed=create_random_seed())

    assert idata.posterior.equals(idata2.posterior)

    sizes = idata.posterior.sizes
    assert sizes["chain"] == 1
    assert sizes["draw"] == 10


def test_fit_sampler_config_seed_reproducibility(toy_X, toy_y) -> None:
    sampler_config = {
        "chains": 1,
        "draws": 10,
        "tune": 5,
        "random_seed": 42,
    }
    model = ModelBuilderTest(sampler_config=sampler_config)

    idata = model.fit(toy_X, toy_y)
    idata2 = model.fit(toy_X, toy_y)

    assert idata.posterior.equals(idata2.posterior)


def test_fit_sampler_config_with_rng_fails(mocker, toy_X, toy_y) -> None:
    def mock_sample(*args, **kwargs):
        idata = pm.sample_prior_predictive(10)
        return az.InferenceData(posterior=idata.prior)

    mocker.patch("pymc.sample", mock_sample)
    sampler_config = {
        "chains": 1,
        "draws": 10,
        "tune": 5,
        "random_seed": np.random.default_rng(42),
    }
    model = ModelBuilderTest(sampler_config=sampler_config)

    match = "Object of type Generator is not JSON serializable"
    with pytest.raises(TypeError, match=match):
        model.fit(toy_X, toy_y)


def test_unmatched_index(toy_X, toy_y) -> None:
    model = ModelBuilderTest()
    toy_X = toy_X.copy()
    toy_X.index = toy_X.index + 1
    match = "Index of X and y must match"
    with pytest.raises(ValueError, match=match):
        model.fit(toy_X, toy_y)


def test_graphviz(toy_X, toy_y):
    """Test pymc.graphviz utility on model before and after being built"""
    model = ModelBuilderTest()

    with pytest.raises(
        AttributeError, match="'ModelBuilderTest' object has no attribute 'model'"
    ):
        model.graphviz()

    model.build_model(X=toy_X, y=toy_y)
    assert isinstance(model.graphviz(), graphviz.graphs.Digraph)
