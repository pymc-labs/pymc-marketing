#   Copyright 2022 - 2026 The PyMC Labs Developers
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
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm.hsgp import SoftPlusHSGP
from pymc_marketing.mmm.tvp import (
    create_hsgp_from_config,
    create_time_varying_gp_multiplier,
    infer_time_index,
    is_hsgp_kwargs_format,
    time_varying_prior,
)


@pytest.fixture
def coords():
    return {
        "date": pd.Series(pd.date_range("2024-01-01", periods=5)),
        "channel": pd.Series(["a", "b", "c"]),
    }


@pytest.fixture
def model_config() -> dict[str, HSGPKwargs]:
    return {
        "intercept_tvp_config": HSGPKwargs(
            m=200,
            L=None,
            eta_lam=1,
            ls_mu=5,
            ls_sigma=5,
        )
    }


def test_time_varying_prior(coords):
    with pm.Model(coords=coords) as model:
        X = pm.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        hsgp_kwargs = HSGPKwargs(m=3, L=10, eta_lam=1, ls_sigma=5)
        f = time_varying_prior(
            name="test",
            X=X,
            X_mid=2,
            dims="date",
            hsgp_kwargs=hsgp_kwargs,
        )

        # Assert output verification
        assert isinstance(f, pt.TensorVariable)

        # Assert internal parameters are created correctly
        assert model.test_raw_hsgp_coefs.eval().shape == (3,)

        # Assert default cov_func is used when none is provided
        assert "test_raw_eta" in model.named_vars
        assert "test_raw_ls" in model.named_vars
        assert "test_raw_hsgp_coefs" in model.named_vars

        # Test that model can compile and sample
        pm.Normal("obs", mu=f, sigma=1, observed=np.random.randn(5))
        try:
            pm.sample(50, tune=50, chains=1)
        except pm.SamplingError:
            pytest.fail("Time varying parameter didn't sample")


def test_calling_without_default_args(coords):
    with pm.Model(coords=coords) as model:
        X = pm.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        f = time_varying_prior(name="test", X=X, dims="date")

        # Assert output verification
        assert isinstance(f, pt.TensorVariable)

        # Assert internal parameters are created correctly
        assert model.test_raw_hsgp_coefs.eval().shape == (200,)

        # Assert default cov_func is used when none is provided
        assert "test_raw_eta" in model.named_vars
        assert "test_raw_ls" in model.named_vars
        assert "test_raw_hsgp_coefs" in model.named_vars


def test_multidimensional(coords):
    with pm.Model(coords=coords) as model:
        X = pm.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        m = 7
        hsgp_kwargs = HSGPKwargs(m=m)
        prior = time_varying_prior(
            name="test",
            X=X,
            X_mid=2,
            dims=("date", "channel"),
            hsgp_kwargs=hsgp_kwargs,
        )

        # Assert internal parameters are created correctly
        assert model.test_raw_hsgp_coefs.eval().shape == (3, m)

        # Test that model can compile and sample
        pm.Normal(
            "obs",
            mu=prior,
            sigma=1,
            observed=np.random.randn(5, 3),
            dims=("date", "channel"),
        )
        try:
            pm.sample(50, tune=50, chains=1)
        except pm.SamplingError:
            pytest.fail("Time varying parameter didn't sample")


def test_calling_without_model():
    with pytest.raises(TypeError, match=r"No model on context stack."):
        X = pm.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        hsgp_kwargs = HSGPKwargs(m=5, L=10)
        time_varying_prior(
            name="test", X=X, X_mid=2, dims="date", hsgp_kwargs=hsgp_kwargs
        )


def test_create_time_varying_intercept(coords, model_config):
    time_index_mid = 2
    time_resolution = 1
    with pm.Model(coords=coords):
        time_index = pm.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        result = create_time_varying_gp_multiplier(
            name="intercept",
            dims="date",
            time_index=time_index,
            time_index_mid=time_index_mid,
            time_resolution=time_resolution,
            hsgp_kwargs=model_config["intercept_tvp_config"],
        )
        assert isinstance(result, pt.TensorVariable)


@pytest.mark.parametrize(
    "freq, time_resolution",
    [
        pytest.param("D", 1, id="daily"),
        pytest.param("W", 7, id="weekly"),
    ],
)
@pytest.mark.parametrize(
    "index",
    [np.arange(5), np.arange(5) + 10],
    ids=["zero-start", "non-zero-start"],
)
@pytest.mark.parametrize(
    "offset, expected",
    [(0, np.arange(0, 5)), (5, np.arange(5, 10)), (-5, np.arange(-5, 0))],
    ids=["in-sample", "oos_forward", "oos_backward"],
)
def test_infer_time_index(freq, time_resolution, index, offset, expected):
    dates = pd.date_range(start="1/1/2022", periods=5, freq=freq)
    date_series = pd.Series(dates, index=index)
    date_series_new = date_series + pd.Timedelta(offset, unit=freq)
    result = infer_time_index(date_series_new, date_series, time_resolution)
    np.testing.assert_array_equal(result, expected)


class TestIsHsgpKwargsFormat:
    """Tests for is_hsgp_kwargs_format function."""

    def test_hsgp_kwargs_format_with_eta_lam(self) -> None:
        """Test detection with eta_lam key."""
        config: dict[str, float] = {"m": 200, "eta_lam": 1.0}
        assert is_hsgp_kwargs_format(config) is True

    def test_hsgp_kwargs_format_with_ls_mu(self) -> None:
        """Test detection with ls_mu key."""
        config: dict[str, float] = {"m": 200, "ls_mu": 5.0}
        assert is_hsgp_kwargs_format(config) is True

    def test_hsgp_kwargs_format_with_ls_sigma(self) -> None:
        """Test detection with ls_sigma key."""
        config: dict[str, float] = {"ls_sigma": 10.0}
        assert is_hsgp_kwargs_format(config) is True

    def test_hsgp_kwargs_format_full_config(self) -> None:
        """Test detection with full HSGPKwargs config."""
        config: dict[str, float | None] = {
            "m": 200,
            "L": 100,
            "eta_lam": 1.0,
            "ls_mu": 5.0,
            "ls_sigma": 10.0,
            "cov_func": None,
        }
        assert is_hsgp_kwargs_format(config) is True

    def test_parameterize_from_data_format(self) -> None:
        """Test that parameterize_from_data format returns False."""
        config: dict[str, float] = {"ls_lower": 0.3, "ls_upper": 2.0}
        assert is_hsgp_kwargs_format(config) is False

    def test_empty_dict(self) -> None:
        """Test empty dict returns False."""
        assert is_hsgp_kwargs_format({}) is False

    def test_unrelated_keys(self) -> None:
        """Test dict with unrelated keys returns False."""
        config: dict[str, str | int] = {"foo": "bar", "m": 200}
        assert is_hsgp_kwargs_format(config) is False


class TestCreateHsgpFromConfig:
    """Tests for create_hsgp_from_config function."""

    @pytest.fixture
    def time_index(self) -> npt.NDArray[np.int_]:
        """Create sample time index."""
        return np.arange(52)

    def test_with_hsgp_kwargs_instance(self, time_index: npt.NDArray[np.int_]) -> None:
        """Test with HSGPKwargs instance."""
        config = HSGPKwargs(m=200, eta_lam=1.0, ls_mu=5.0, ls_sigma=10.0)
        hsgp = create_hsgp_from_config(X=time_index, dims="date", config=config)
        assert isinstance(hsgp, SoftPlusHSGP)
        assert hsgp.m == 200

    def test_with_hsgp_kwargs_dict(self, time_index: npt.NDArray[np.int_]) -> None:
        """Test with dict in HSGPKwargs format."""
        config: dict[str, float | None] = {
            "m": 200,
            "L": 100.0,
            "eta_lam": 1.0,
            "ls_mu": 5.0,
            "ls_sigma": 10.0,
            "cov_func": None,
        }
        hsgp = create_hsgp_from_config(X=time_index, dims="date", config=config)
        assert isinstance(hsgp, SoftPlusHSGP)
        assert hsgp.m == 200

    def test_with_parameterize_from_data_dict(
        self, time_index: npt.NDArray[np.int_]
    ) -> None:
        """Test with dict in parameterize_from_data format."""
        config: dict[str, float] = {"ls_lower": 0.3, "ls_upper": 2.0}
        hsgp = create_hsgp_from_config(X=time_index, dims="date", config=config)
        assert isinstance(hsgp, SoftPlusHSGP)

    def test_with_x_mid_provided(self, time_index: npt.NDArray[np.int_]) -> None:
        """Test with explicit X_mid parameter."""
        config = HSGPKwargs(m=200, eta_lam=1.0, ls_mu=5.0, ls_sigma=10.0)
        hsgp = create_hsgp_from_config(
            X=time_index, dims="date", config=config, X_mid=26.0
        )
        assert hsgp.X_mid == 26.0

    def test_with_tuple_dims(self, time_index: npt.NDArray[np.int_]) -> None:
        """Test with tuple dimensions."""
        config: dict[str, float] = {"ls_lower": 0.3, "ls_upper": 2.0}
        hsgp = create_hsgp_from_config(
            X=time_index, dims=("date", "channel"), config=config
        )
        assert hsgp.dims == ("date", "channel")

    def test_invalid_config_type_raises(self, time_index: npt.NDArray[np.int_]) -> None:
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="config must be HSGPKwargs or dict"):
            create_hsgp_from_config(
                X=time_index,
                dims="date",
                config="invalid",  # type: ignore
            )
