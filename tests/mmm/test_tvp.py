import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc_marketing.mmm.tvp import (
    create_time_varying_intercept,
    infer_time_index,
    time_varying_prior,
)


@pytest.fixture
def coords():
    return {
        "date": pd.Series(pd.date_range("2024-01-01", periods=5)),
        "channel": pd.Series(["a", "b", "c"]),
    }


@pytest.fixture
def model_config():
    return {
        "intercept_tvp_kwargs": {
            "m": 200,
            "eta_lam": 1,
            "ls_mu": None,
            "ls_sigma": 5,
            "L": None,
        },
        "intercept": {
            "dist": pm.Normal,
            "kwargs": {"mu": 0, "sigma": 1},
        },
    }


def test_time_varying_prior(coords):
    with pm.Model(coords=coords) as model:
        X = pm.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        prior = time_varying_prior(name="test", X=X, X_mid=2, dims="date", m=3, L=10)

        # Assert output verification
        assert isinstance(prior, pt.TensorVariable)

        # Assert internal parameters are created correctly
        assert model.test_hsgp_coefs.eval().shape == (3,)

        # Assert default cov_func is used when none is provided
        assert "test_eta" in model.named_vars
        assert "test_ls" in model.named_vars
        assert "test_hsgp_coefs" in model.named_vars

        # Test that model can compile and sample
        pm.Normal("obs", mu=prior, sigma=1, observed=np.random.randn(5))
        pm.sample(50, tune=50, chains=1)


def test_calling_without_default_args(coords):
    with pm.Model(coords=coords):
        X = pm.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        time_varying_prior(name="test", X=X, dims="date")


def test_multidimensional(coords):
    with pm.Model(coords=coords) as model:
        X = pm.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        prior = time_varying_prior(
            name="test", X=X, X_mid=2, dims=("date", "channel"), m=7
        )

        # Assert internal parameters are created correctly
        assert model.test_hsgp_coefs.eval().shape == (3, 7)

        # Test that model can compile and sample
        pm.Normal("obs", mu=prior, sigma=1, observed=np.random.randn(5, 3))
        pm.sample(50, tune=50, chains=1)


def test_calling_without_model():
    with pytest.raises(TypeError, match="No model on context stack."):
        X = pm.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        time_varying_prior(name="test", X=X, X_mid=2, dims="date", m=5, L=10)


def test_create_time_varying_intercept(coords, model_config):
    time_index_mid = 2
    time_resolution = 1
    intercept_dist = model_config["intercept"]["dist"]
    with pm.Model(coords=coords):
        time_index = pm.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        result = create_time_varying_intercept(
            time_index, time_index_mid, time_resolution, intercept_dist, model_config
        )
        assert isinstance(result, pt.TensorVariable)


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