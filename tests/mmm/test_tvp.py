import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc_marketing.mmm.tvp import infer_time_index, time_varying_prior


@pytest.fixture
def coords():
    return {"time": pd.Series(pd.date_range("2024-01-01", periods=5))}


def test_calling_without_model():
    with pytest.raises(TypeError, match="No model on context stack."):
        time_varying_prior(
            name="test", X=np.array([1, 2, 3]), X_mid=2, dims="time", m=5, L=10
        )


def test_output_verification(coords):
    with pm.Model(coords=coords):
        X = np.array([0, 1, 2, 3, 4])
        prior = time_varying_prior(name="test", X=X, X_mid=2, dims="time", m=3, L=10)
        assert isinstance(prior, pt.TensorVariable)


def test_dependency_checks(coords):
    with pm.Model(coords=coords) as model:
        X = np.array([0, 1, 2, 3, 4])
        _ = time_varying_prior(name="test", X=X, X_mid=2, dims="time", m=3, L=10)
        # Assert default cov_func is used when none is provided
        assert "eta_test" in model.named_vars
        assert "ls_test" in model.named_vars
        assert "_hsgp_coefs_test" in model.named_vars


def test_integration_with_model(coords):
    with pm.Model(coords=coords):
        X = np.array([0, 1, 2, 3, 4])
        prior = time_varying_prior(name="test", X=X, X_mid=2, dims="time", m=3, L=10)
        pm.Normal("obs", mu=prior, sigma=1, observed=np.random.randn(5))
        # This should compile the model without errors, indicating successful integration
        pm.sample(50, tune=50, chains=1)


@pytest.mark.parametrize("freq, time_resolution", [("D", 1), ("W", 7)])
def test_infer_time_index_in_sample(freq, time_resolution):
    date_series = pd.Series(pd.date_range(start="1/1/2022", periods=5, freq=freq))
    date_series_new = date_series
    expected = np.arange(0, 5)
    result = infer_time_index(date_series_new, date_series, time_resolution)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("freq, time_resolution", [("D", 1), ("W", 7)])
def test_infer_time_index_oos_forward(freq, time_resolution):
    date_series = pd.Series(pd.date_range(start="1/1/2022", periods=5, freq=freq))
    date_series_new = date_series + pd.Timedelta(5, unit=freq)
    expected = np.arange(5, 10)
    result = infer_time_index(date_series_new, date_series, time_resolution)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("freq, time_resolution", [("D", 1), ("W", 7)])
def test_infer_time_index_oos_backward(freq, time_resolution):
    date_series = pd.Series(pd.date_range(start="1/1/2022", periods=5, freq=freq))
    date_series_new = date_series - pd.Timedelta(5, unit=freq)
    expected = np.arange(-5, 0)
    result = infer_time_index(date_series_new, date_series, time_resolution)
    np.testing.assert_array_equal(result, expected)
