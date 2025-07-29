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
import hashlib
import json
import re
import sys
import tempfile

import arviz as az
import graphviz
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from rich.table import Table

from pymc_marketing.model_builder import (
    BaseModelBuilder,
    DifferentModelError,
    ModelBuilder,
    RegressionModelBuilder,
    _handle_deprecate_pred_argument,
    create_sample_kwargs,
)


@pytest.fixture(scope="module")
def toy_X():
    x = np.linspace(start=0, stop=1, num=100)
    return pd.DataFrame({"input": x})


@pytest.fixture(scope="module")
def toy_y(toy_X):
    rng = np.random.default_rng(42)
    y = 5 * toy_X["input"] + 3
    y = y + rng.normal(0, 1, size=len(toy_X))
    y = pd.Series(y, name="different name than output")
    return y


@pytest.fixture(scope="module")
def fitted_model_instance(toy_X, toy_y, mock_pymc_sample):
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
        toy_y,
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


@pytest.fixture(scope="module")
def toy_data(toy_X, toy_y):
    """Create a combined dataset for DataModelBuilderTest."""
    data = toy_X.copy()
    data["output"] = toy_y
    return data


@pytest.fixture(scope="module")
def fitted_data_model_instance(toy_data, mock_pymc_sample):
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
    model = DataModelBuilderTest(
        data=toy_data,
        model_config=model_config,
        sampler_config=sampler_config,
        test_parameter="test_parameter",
    )
    model.fit(
        chains=1,
        draws=100,
        tune=100,
    )
    return model


@pytest.fixture(scope="module")
def fitted_base_model_instance(mock_pymc_sample):
    sampler_config = {
        "draws": 100,
        "tune": 100,
        "chains": 2,
        "target_accept": 0.95,
    }
    model_config = {
        "mu_loc": 0,
        "mu_scale": 1,
        "sigma_scale": 1,
    }
    model = BaseModelBuilderTest(
        model_config=model_config,
        sampler_config=sampler_config,
        test_parameter="test_parameter",
    )
    model.fit(
        chains=1,
        draws=100,
        tune=100,
    )
    return model


class ModelBuilderTest(RegressionModelBuilder):
    def __init__(self, model_config=None, sampler_config=None, test_parameter=None):
        self.test_parameter = test_parameter
        super().__init__(model_config=model_config, sampler_config=sampler_config)

    _model_type = "test_model"
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series):
        coords = {"numbers": np.arange(len(X))}

        y = y if isinstance(y, np.ndarray) else y.values

        with pm.Model(coords=coords) as self.model:
            x = pm.Data("x", X["input"].values)
            y_data = pm.Data("y_data", y)

            # prior parameters
            a_loc = self.model_config["a"]["loc"]
            a_scale = self.model_config["a"]["scale"]
            b_loc = self.model_config["b"]["loc"]
            b_scale = self.model_config["b"]["scale"]
            obs_error = self.model_config["obs_error"]

            # priors
            a = pm.Normal(
                "a",
                a_loc,
                sigma=a_scale,
                dims=self.model_config["a"]["dims"],
            )
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

    def _data_setter(self, X: pd.DataFrame, y: pd.Series | None = None):
        with self.model:
            pm.set_data({"x": X["input"].values})
            if y is not None:
                y = y.values if isinstance(y, pd.Series) else y
                pm.set_data({"y_data": y})

    @property
    def _serializable_model_config(self):
        return self.model_config

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


class DataModelBuilderTest(ModelBuilder):
    """Test class for ModelBuilder that takes data at initialization."""

    def __init__(
        self, data, model_config=None, sampler_config=None, test_parameter=None
    ):
        self.test_parameter = test_parameter
        super().__init__(
            data=data, model_config=model_config, sampler_config=sampler_config
        )

    _model_type = "data_test_model"
    version = "0.1"

    def build_model(self, **kwargs):
        # Simplified model to avoid compilation issues
        with pm.Model() as self.model:
            # Very simple model to avoid compilation issues
            pm.Normal("test", 0, 1)

    def create_idata_attrs(self):
        attrs = super().create_idata_attrs()
        attrs["test_parameter"] = json.dumps(self.test_parameter)
        return attrs

    @property
    def _serializable_model_config(self):
        return self.model_config

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


class BaseModelBuilderTest(BaseModelBuilder):
    """Test class for BaseModelBuilder."""

    def __init__(self, model_config=None, sampler_config=None, test_parameter=None):
        self.test_parameter = test_parameter
        super().__init__(model_config=model_config, sampler_config=sampler_config)

    _model_type = "base_test_model"
    version = "0.1"

    def build_model(self, **kwargs):
        # This is a simple model for testing BaseModelBuilder
        with pm.Model() as self.model:
            # Very simple model to avoid compilation issues
            pm.Normal("test", 0, 1)

    def build_from_idata(self, idata: az.InferenceData) -> None:
        self.build_model()

    def create_idata_attrs(self):
        attrs = super().create_idata_attrs()
        attrs["test_parameter"] = json.dumps(self.test_parameter)
        return attrs

    @property
    def _serializable_model_config(self):
        return self.model_config

    @property
    def default_model_config(self) -> dict:
        return {"mu_loc": 0, "mu_scale": 1, "sigma_scale": 1}

    @property
    def default_sampler_config(self) -> dict:
        return {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 3,
            "target_accept": 0.95,
        }

    def fit(self, **kwargs):
        """Override fit method for BaseModelBuilderTest."""
        if not hasattr(self, "model"):
            self.build_model()

        sampler_kwargs = create_sample_kwargs(
            self.sampler_config,
            kwargs.get("progressbar"),
            kwargs.get("random_seed"),
            **kwargs,
        )
        with self.model:
            idata = pm.sample(**sampler_kwargs)

        if self.idata:
            self.idata = self.idata.copy()
            self.idata.extend(idata, join="right")
        else:
            self.idata = idata

        self.set_idata_attrs(self.idata)
        return self.idata


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


def test_data_model_builder_config():
    """Test DataModelBuilderTest configuration."""
    toy_data = pd.DataFrame({"input": [1, 2, 3], "output": [2, 4, 6]})
    default = DataModelBuilderTest(data=toy_data)
    assert default.model_config == default.default_model_config
    assert default.sampler_config == default.default_sampler_config
    assert default.data.equals(toy_data)

    nondefault = DataModelBuilderTest(
        data=toy_data, model_config={"obs_error": 3}, sampler_config={"draws": 42}
    )
    assert nondefault.model_config != nondefault.default_model_config
    assert nondefault.sampler_config != default.default_sampler_config
    assert nondefault.model_config == default.model_config | {"obs_error": 3}
    assert nondefault.sampler_config == default.sampler_config | {"draws": 42}


def test_base_model_builder_config():
    """Test BaseModelBuilderTest configuration."""
    default = BaseModelBuilderTest()
    assert default.model_config == default.default_model_config
    assert default.sampler_config == default.default_sampler_config

    nondefault = BaseModelBuilderTest(
        model_config={"mu_loc": 5}, sampler_config={"draws": 42}
    )
    assert nondefault.model_config != nondefault.default_model_config
    assert nondefault.sampler_config != default.default_sampler_config
    assert nondefault.model_config == default.model_config | {"mu_loc": 5}
    assert nondefault.sampler_config == default.sampler_config | {"draws": 42}


def test_handle_deprecate_pred_argument():
    """Test the _handle_deprecate_pred_argument utility function."""
    kwargs = {}

    # Test normal case
    result = _handle_deprecate_pred_argument("test_value", "test", kwargs)
    assert result == "test_value"

    # Test deprecated argument
    kwargs = {"test_pred": "deprecated_value"}
    with pytest.warns(DeprecationWarning, match="test_pred is deprecated"):
        result = _handle_deprecate_pred_argument(None, "test", kwargs)
    assert result == "deprecated_value"
    assert "test_pred" not in kwargs  # Should be removed

    # Test both arguments provided
    kwargs = {"test_pred": "deprecated_value"}
    with pytest.raises(ValueError, match="Both test and test_pred cannot be provided"):
        _handle_deprecate_pred_argument("test_value", "test", kwargs)

    # Test none allowed (without deprecated argument)
    kwargs = {}
    result = _handle_deprecate_pred_argument(None, "test", kwargs, none_allowed=True)
    assert result is None

    # Test none not allowed
    with pytest.raises(ValueError, match="Please provide test"):
        _handle_deprecate_pred_argument(None, "test", kwargs, none_allowed=False)


def test_save_input_params(fitted_model_instance):
    assert fitted_model_instance.idata.attrs["test_parameter"] == '"test_parameter"'


def test_has_pymc_marketing_version(fitted_model_instance):
    assert "pymc_marketing_version" in fitted_model_instance.posterior.attrs


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


def test_data_model_builder_save_load(fitted_data_model_instance):
    """Test save/load functionality for DataModelBuilderTest."""
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    fitted_data_model_instance.save(temp.name)

    test_builder2 = DataModelBuilderTest.load(temp.name)

    assert fitted_data_model_instance.idata.groups() == test_builder2.idata.groups()
    assert fitted_data_model_instance.id == test_builder2.id
    assert fitted_data_model_instance.model_config == test_builder2.model_config
    assert fitted_data_model_instance.sampler_config == test_builder2.sampler_config

    # Handle data comparison - it might be converted to xarray during save/load
    original_data = fitted_data_model_instance.data
    loaded_data = test_builder2.data

    # Convert both to xarray for comparison
    if hasattr(original_data, "to_xarray"):
        original_xarray = original_data.to_xarray()
    else:
        original_xarray = original_data

    if hasattr(loaded_data, "to_xarray"):
        loaded_xarray = loaded_data.to_xarray()
    else:
        loaded_xarray = loaded_data

    assert original_xarray.identical(loaded_xarray)

    temp.close()


def test_base_model_builder_save_load(fitted_base_model_instance):
    """Test save/load functionality for BaseModelBuilderTest."""
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    fitted_base_model_instance.save(temp.name)

    test_builder2 = BaseModelBuilderTest.load(temp.name)

    assert fitted_base_model_instance.idata.groups() == test_builder2.idata.groups()
    assert fitted_base_model_instance.id == test_builder2.id
    assert fitted_base_model_instance.model_config == test_builder2.model_config
    assert fitted_base_model_instance.sampler_config == test_builder2.sampler_config
    temp.close()


def test_initial_build_and_fit(
    fitted_model_instance, check_idata=True
) -> RegressionModelBuilder:
    if check_idata:
        assert fitted_model_instance.idata is not None
        assert "posterior" in fitted_model_instance.idata.groups()


def test_save_without_fit_raises_runtime_error():
    model_builder = ModelBuilderTest()
    match = "The model hasn't been fit yet"
    with pytest.raises(RuntimeError, match=match):
        model_builder.save("saved_model")


def test_empty_sampler_config_fit(toy_X, toy_y, mock_pymc_sample):
    sampler_config = {}
    model_builder = ModelBuilderTest(sampler_config=sampler_config)
    model_builder.idata = model_builder.fit(
        X=toy_X, y=toy_y, chains=1, draws=100, tune=100
    )
    assert model_builder.idata is not None
    assert "posterior" in model_builder.idata.groups()


def test_data_model_builder_fit(toy_data, mock_pymc_sample):
    """Test fitting for DataModelBuilderTest."""
    sampler_config = {}
    model_builder = DataModelBuilderTest(data=toy_data, sampler_config=sampler_config)
    model_builder.idata = model_builder.fit(chains=1, draws=100, tune=100)
    assert model_builder.idata is not None
    assert "posterior" in model_builder.idata.groups()
    assert "fit_data" in model_builder.idata.groups()


def test_base_model_builder_fit(mock_pymc_sample):
    """Test fitting for BaseModelBuilderTest."""
    sampler_config = {}
    model_builder = BaseModelBuilderTest(sampler_config=sampler_config)
    model_builder.idata = model_builder.fit(chains=1, draws=100, tune=100)
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


def test_fit_no_t(toy_X, mock_pymc_sample):
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
    fake_fit = pm.sample_prior_predictive(draws=50, model=model.model, random_seed=1234)
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
    chains = fitted_model_instance.idata.posterior.sizes["chain"]
    draws = fitted_model_instance.idata.posterior.sizes["draw"]
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


def test_model_io_functionality():
    """Test ModelIO mixin functionality."""
    # Test with different model types
    regression_model = ModelBuilderTest()
    data_model = DataModelBuilderTest(data=pd.DataFrame({"input": [1], "output": [2]}))
    base_model = BaseModelBuilderTest()

    # Test that all have unique IDs
    ids = [regression_model.id, data_model.id, base_model.id]
    assert len(set(ids)) == 3

    # Test that all have proper model types
    assert regression_model._model_type == "test_model"
    assert data_model._model_type == "data_test_model"
    assert base_model._model_type == "base_test_model"

    # Test that all have proper versions
    assert regression_model.version == "0.1"
    assert data_model.version == "0.1"
    assert base_model.version == "0.1"


def test_model_io_attrs_creation():
    """Test ModelIO attrs creation functionality."""
    model = ModelBuilderTest(test_parameter="test_parameter")
    attrs = model.create_idata_attrs()

    required_keys = {"id", "model_type", "version", "sampler_config", "model_config"}
    assert all(key in attrs for key in required_keys)
    assert attrs["model_type"] == "test_model"
    assert attrs["version"] == "0.1"
    assert attrs["test_parameter"] == '"test_parameter"'


def test_model_io_set_idata_attrs():
    """Test ModelIO set_idata_attrs functionality."""
    model = ModelBuilderTest(test_parameter="test_parameter")

    # Create a simple model without complex operations
    with pm.Model() as simple_model:
        pm.Normal("test", 0, 1)

    # Create fake idata
    fake_idata = pm.sample_prior_predictive(
        draws=10, model=simple_model, random_seed=1234
    )
    fake_idata.add_groups(dict(posterior=fake_idata.prior))

    # Test setting attrs
    result_idata = model.set_idata_attrs(fake_idata)
    assert result_idata.attrs["id"] == model.id
    assert result_idata.attrs["model_type"] == model._model_type
    assert result_idata.attrs["version"] == model.version

    # Test error when no idata provided
    with pytest.raises(RuntimeError, match="No idata provided to set attrs on"):
        model.set_idata_attrs(None)


@pytest.mark.parametrize("name", ["prior_predictive", "posterior_predictive"])
def test_sample_xxx_predictive_keeps_second(
    fitted_model_instance, toy_X, name: str
) -> None:
    rng = np.random.default_rng(42)
    method_name = f"sample_{name}"
    method = getattr(fitted_model_instance, method_name)

    X_pred = toy_X

    kwargs = {
        "X": X_pred,
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


@pytest.fixture(scope="module")
def model_with_prior_predictive(toy_X) -> ModelBuilderTest:
    model = ModelBuilderTest()
    model.sample_prior_predictive(toy_X)
    return model


def test_sample_prior_predictive_groups(model_with_prior_predictive):
    assert "prior" in model_with_prior_predictive.idata
    assert "prior_predictive" in model_with_prior_predictive.idata


def test_sample_prior_predictive_has_pymc_marketing_version(
    model_with_prior_predictive,
):
    assert "pymc_marketing_version" in model_with_prior_predictive.prior.attrs
    assert (
        "pymc_marketing_version" in model_with_prior_predictive.prior_predictive.attrs
    )


def test_fit_after_prior_keeps_prior(
    model_with_prior_predictive,
    toy_X,
    toy_y,
    mock_pymc_sample,
):
    model_with_prior_predictive.fit(X=toy_X, y=toy_y, chains=1, draws=100, tune=100)
    assert "prior" in model_with_prior_predictive.idata
    assert "prior_predictive" in model_with_prior_predictive.idata


def test_second_fit(toy_X, toy_y, mock_pymc_sample):
    model = ModelBuilderTest()

    model.fit(X=toy_X, y=toy_y, chains=1, draws=100, tune=100)
    assert "posterior" in model.idata
    id_before = id(model.idata)
    assert "fit_data" in model.idata

    model.fit(X=toy_X, y=toy_y, chains=1, draws=100, tune=100)
    id_after = id(model.idata)

    assert id_before != id_after


class InsufficientModel(RegressionModelBuilder):
    def __init__(
        self, model_config=None, sampler_config=None, new_parameter=None
    ) -> None:
        super().__init__(model_config=model_config, sampler_config=sampler_config)
        self.new_parameter = new_parameter

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
        model.sample_prior_predictive(X=X_pred)


def test_abstract_methods():
    """Test that abstract methods are properly enforced."""
    # Test that we can't instantiate BaseModelBuilder directly
    with pytest.raises(TypeError):
        BaseModelBuilder()

    # Test that we can't instantiate ModelBuilder directly
    with pytest.raises(TypeError):
        ModelBuilder(data=None)

    # Test that we can't instantiate RegressionModelBuilder directly
    with pytest.raises(TypeError):
        RegressionModelBuilder()


def test_requires_model_decorator():
    """Test the requires_model decorator."""
    model = ModelBuilderTest()

    # match = "The model hasn't been built yet"
    # with pytest.raises(RuntimeError, match=match):
    #     model.graphviz()

    # Test that graphviz fails before model is built
    with pytest.raises(RuntimeError, match="The model hasn't been built yet"):
        model.graphviz()

    # Test that it works after model is built
    model.build_model(pd.DataFrame({"input": [1, 2, 3]}), pd.Series([1, 2, 3]))
    assert isinstance(model.graphviz(), graphviz.graphs.Digraph)


def test_incorrect_set_idata_attrs_override() -> None:
    class IncorrectSetAttrs(InsufficientModel):
        def create_idata_attrs(self) -> dict:
            return {"new_parameter": self.new_parameter}

    model = IncorrectSetAttrs()

    X_pred = [1, 2, 3]

    match = "Missing required keys in attrs"
    with pytest.raises(ValueError, match=match):
        model.sample_prior_predictive(X=X_pred)


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


def test_fit_sampler_config_with_rng_fails(toy_X, toy_y, mock_pymc_sample) -> None:
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


def test_regression_model_builder_data_validation():
    """Test data validation in RegressionModelBuilder."""
    model = ModelBuilderTest()

    # Test _validate_data method
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])

    # Test with X and y
    X_valid, y_valid = model._validate_data(X, y)
    assert isinstance(X_valid, np.ndarray)
    assert isinstance(y_valid, np.ndarray)

    # Test with only X
    X_valid_only = model._validate_data(X)
    assert isinstance(X_valid_only, np.ndarray)

    # Test with pandas DataFrame and Series
    X_df = pd.DataFrame(X, columns=["a", "b"])
    y_series = pd.Series(y)
    X_valid_df, y_valid_series = model._validate_data(X_df, y_series)
    assert isinstance(X_valid_df, np.ndarray)
    assert isinstance(y_valid_series, np.ndarray)


def test_regression_model_builder_output_var_conflict(toy_X, toy_y):
    """Test that RegressionModelBuilder detects output variable conflicts."""
    model = ModelBuilderTest()

    # Create X with conflicting output variable name
    X_with_output = toy_X.copy()
    X_with_output["output"] = toy_y

    with pytest.raises(ValueError, match="X includes a column named 'output'"):
        model.fit(X_with_output, toy_y)


def test_graphviz(toy_X, toy_y):
    """Test pymc.graphviz utility on model before and after being built"""
    model = ModelBuilderTest()

    with pytest.raises(
        AttributeError, match="'ModelBuilderTest' object has no attribute 'model'"
    ):
        model.graphviz()

    model.build_model(X=toy_X, y=toy_y)
    assert isinstance(model.graphviz(), graphviz.graphs.Digraph)


@pytest.mark.parametrize(
    "method_name",
    [
        "sample_posterior_predictive",
        "predict",
    ],
)
def test_X_pred_posterior_deprecation(
    method_name,
    fitted_model_instance,
    toy_X,
) -> None:
    if "posterior_predictive" in fitted_model_instance.idata:
        del fitted_model_instance.idata.posterior_predictive

    with pytest.warns(DeprecationWarning, match="X_pred is deprecated"):
        method = getattr(fitted_model_instance, method_name)
        method(X_pred=toy_X)

    assert isinstance(fitted_model_instance.posterior_predictive, xr.Dataset)


def test_X_pred_prior_deprecation(fitted_model_instance, toy_X, toy_y) -> None:
    if "prior" in fitted_model_instance.idata:
        del fitted_model_instance.idata.prior
    if "prior_predictive" in fitted_model_instance.idata:
        del fitted_model_instance.idata.prior_predictive

    with pytest.warns(DeprecationWarning, match="X_pred is deprecated"):
        fitted_model_instance.sample_prior_predictive(X_pred=toy_X)

    with pytest.warns(DeprecationWarning, match="y_pred is deprecated"):
        fitted_model_instance.sample_prior_predictive(toy_X, y_pred=toy_y)

    assert isinstance(fitted_model_instance.prior, xr.Dataset)
    assert isinstance(fitted_model_instance.prior_predictive, xr.Dataset)


class XarrayModel(RegressionModelBuilder):
    """Multivariate Regression model."""

    def build_model(self, X, y, **kwargs):
        if isinstance(X, xr.Dataset):
            X = X["x"]

        coords = {
            "country": ["A", "B"],
            "date": [0, 1],
        }
        with pm.Model(coords=coords) as self.model:
            x = pm.Data("X", X.values, dims=("country", "date"))
            y = pm.Data("y", y.values, dims=("country", "date"))

            alpha = pm.Normal("alpha", 0, 1, dims=("country",))
            beta = pm.Normal("beta", 0, 1, dims=("country",))

            mu = alpha + beta * x

            sigma = pm.HalfNormal("sigma")

            pm.Normal("output", mu=mu, sigma=sigma, observed=y)

    def _data_setter(self, X, y=None):
        pass

    @property
    def _serializable_model_config(self):
        return {}

    @property
    def output_var(self):
        return "output"

    @property
    def default_model_config(self):
        return {}

    @property
    def default_sampler_config(self):
        return {}


@pytest.fixture
def xarray_X() -> xr.Dataset:
    return (
        pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "date": [0, 1, 0, 1],
                "country": ["A", "A", "B", "B"],
            }
        )
        .set_index(["country", "date"])
        .to_xarray()
    )


@pytest.fixture
def xarray_y(xarray_X) -> xr.DataArray:
    alpha = xr.DataArray(
        [1, 2],
        dims=["country"],
        coords={"country": ["A", "B"]},
    )
    beta = xr.DataArray([1, 2], dims=["country"], coords={"country": ["A", "B"]})

    return (alpha + beta * xarray_X["x"]).rename("name other than output")


@pytest.mark.parametrize("X_is_array", [False, True], ids=["DataArray", "Dataset"])
def test_xarray_model_builder(X_is_array, xarray_X, xarray_y, mock_pymc_sample) -> None:
    model = XarrayModel()

    X = xarray_X if X_is_array else xarray_X["x"]

    model.fit(X, xarray_y)

    xr.testing.assert_equal(
        model.idata.fit_data,  # type: ignore
        pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "output": [2, 3, 8, 10],
            },
            index=pd.MultiIndex.from_tuples(
                [("A", 0), ("A", 1), ("B", 0), ("B", 1)], names=["country", "date"]
            ),
        ).to_xarray(),
    )


@pytest.fixture(scope="module")
def stale_idata(fitted_model_instance) -> az.InferenceData:
    idata = fitted_model_instance.idata.copy()
    idata.attrs["version"] = "0.0.1"

    return idata


@pytest.fixture(scope="module")
def different_configuration_idata(fitted_model_instance) -> az.InferenceData:
    idata = fitted_model_instance.idata.copy()
    model_config = json.loads(idata.attrs["model_config"])
    model_config["a"] = {"loc": 1, "scale": 15, "dims": ("numbers",)}
    idata.attrs["model_config"] = json.dumps(model_config)

    return idata


@pytest.mark.parametrize(
    "fixture_name, match",
    [
        pytest.param(
            "stale_idata",
            re.escape("The model version (0.0.1)"),
            id="different version",
        ),
        pytest.param(
            "different_configuration_idata", "The model id", id="different id"
        ),
    ],
)
def test_load_from_idata_errors(request, fixture_name, match) -> None:
    idata = request.getfixturevalue(fixture_name)
    with pytest.raises(DifferentModelError, match=match):
        ModelBuilderTest.load_from_idata(idata)


def test_table() -> None:
    model = ModelBuilderTest()
    match = "The model hasn't been built yet"
    with pytest.raises(RuntimeError, match=match):
        model.table()

    model.build_model(pd.DataFrame({"input": [1, 2, 3]}), pd.Series([1, 2, 3]))
    assert isinstance(model.table(), Table)


def test_idata_accessors():
    """Test the idata accessor properties."""
    model = ModelBuilderTest()

    # Test that accessors fail when no idata is available
    with pytest.raises(RuntimeError, match="The model hasn't been fit yet"):
        model.posterior

    with pytest.raises(RuntimeError, match="The model hasn't been sampled yet"):
        model.prior

    with pytest.raises(RuntimeError, match="The model hasn't been sampled yet"):
        model.prior_predictive

    with pytest.raises(RuntimeError, match="The model hasn't been fit yet"):
        model.posterior_predictive

    with pytest.raises(
        RuntimeError, match="Call the 'sample_posterior_predictive' method"
    ):
        model.predictions

    # Test fit_result accessor
    with pytest.raises(RuntimeError, match="The model hasn't been fit yet"):
        model.fit_result


def test_model_config_formatting_edge_cases():
    """Test edge cases in model config formatting."""
    model = ModelBuilderTest()

    # Test with empty config
    empty_config = {}
    formatted = model._model_config_formatting(empty_config)
    assert formatted == {}

    # Test with nested dicts but no lists
    simple_config = {"a": {"b": "c"}}
    formatted = model._model_config_formatting(simple_config)
    assert formatted == simple_config

    # Test with mixed types
    mixed_config = {"a": {"dims": ["x", "y"], "loc": [1, 2], "scale": 10}}
    formatted = model._model_config_formatting(mixed_config)
    assert formatted["a"]["dims"] == ("x", "y")
    assert isinstance(formatted["a"]["loc"], np.ndarray)
    assert formatted["a"]["scale"] == 10
