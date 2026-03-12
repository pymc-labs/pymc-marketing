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
import pymc.dims as pmd
import pytest
from pytensor.xtensor.type import XTensorVariable
from xarray import DataArray

from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm.hsgp import CovFunc, SoftPlusHSGP
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


@pytest.fixture(
    params=[None, CovFunc.Matern52], ids=["cov_func_none", "cov_func_matern52"]
)
def model_config(request) -> dict[str, HSGPKwargs]:
    return {
        "intercept_tvp_config": HSGPKwargs(
            m=200,
            L=None,
            eta_lam=1,
            ls_mu=5,
            ls_sigma=5,
            cov_func=request.param,
        )
    }


def test_time_varying_prior(coords, mock_pymc_sample):
    with pm.Model(coords=coords) as model:
        X = pmd.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        hsgp_kwargs = HSGPKwargs(m=3, L=10, eta_lam=1, ls_sigma=5)
        f = time_varying_prior(
            name="test",
            X=X,
            X_mid=2,
            dims="date",
            hsgp_kwargs=hsgp_kwargs,
        )

        # Assert output verification
        assert isinstance(f, XTensorVariable)

        # Assert internal parameters are created correctly
        assert model.test_raw_hsgp_coefs.eval().shape == (3,)

        # Assert default cov_func is used when none is provided
        assert "test_raw_eta" in model.named_vars
        assert "test_raw_ls" in model.named_vars
        assert "test_raw_hsgp_coefs" in model.named_vars

        # Test that model can compile and sample
        pm.Normal("obs", mu=f.values, sigma=1, observed=np.random.randn(5))
        idata = pm.sample(50, tune=50, chains=1)
        assert idata is not None


def test_calling_without_default_args(coords):
    with pm.Model(coords=coords) as model:
        X = pmd.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        f = time_varying_prior(name="test", X=X, dims="date")

        # Assert output verification
        assert isinstance(f, XTensorVariable)

        # Assert internal parameters are created correctly
        assert model.test_raw_hsgp_coefs.eval().shape == (200,)

        # Assert default cov_func is used when none is provided
        assert "test_raw_eta" in model.named_vars
        assert "test_raw_ls" in model.named_vars
        assert "test_raw_hsgp_coefs" in model.named_vars


def test_multidimensional(coords, mock_pymc_sample):
    with pm.Model(coords=coords) as model:
        X = pmd.Data("X", np.array([0, 1, 2, 3, 4.0]), dims="date")
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
            mu=prior.values,
            sigma=1,
            observed=np.random.randn(5, 3),
            dims=("date", "channel"),
        )
        idata = pm.sample(50, tune=50, chains=1)
        assert idata is not None


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
        time_index = pmd.Data("X", np.array([0, 1, 2, 3, 4]), dims="date")
        result = create_time_varying_gp_multiplier(
            name="intercept",
            dims="date",
            time_index=time_index,
            time_index_mid=time_index_mid,
            time_resolution=time_resolution,
            hsgp_kwargs=model_config["intercept_tvp_config"],
        )
        assert isinstance(result, XTensorVariable)


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
        config: dict[str, str | int] = {"foo": "bar"}
        assert is_hsgp_kwargs_format(config) is False

    def test_hsgp_kwargs_format_with_m_only(self) -> None:
        """Test detection with only m key (HSGPKwargs-only param)."""
        config: dict[str, int] = {"m": 200}
        assert is_hsgp_kwargs_format(config) is True

    def test_hsgp_kwargs_format_with_L_only(self) -> None:
        """Test detection with only L key (HSGPKwargs-only param)."""
        config: dict[str, float] = {"L": 100.0}
        assert is_hsgp_kwargs_format(config) is True

    def test_hsgp_kwargs_format_with_m_and_L(self) -> None:
        """Test detection with m and L keys."""
        config: dict[str, int | float] = {"m": 200, "L": 100.0}
        assert is_hsgp_kwargs_format(config) is True


class TestCreateHsgpFromConfig:
    """Tests for create_hsgp_from_config function."""

    @pytest.fixture
    def time_index(self) -> npt.NDArray[np.int_]:
        """Create sample time index."""
        return np.arange(52)

    def test_with_hsgp_kwargs_instance(self, time_index: npt.NDArray[np.int_]) -> None:
        """Test with HSGPKwargs instance."""
        config = HSGPKwargs(m=200, eta_lam=1.0, ls_mu=5.0, ls_sigma=10.0)
        hsgp = create_hsgp_from_config(
            X=DataArray(time_index, dims=("date",)), dims="date", config=config
        )
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
        hsgp = create_hsgp_from_config(
            X=DataArray(time_index, dims=("date",)), dims="date", config=config
        )
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
            X=DataArray(time_index, dims=("date",)),
            dims="date",
            config=config,
            X_mid=26.0,
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

    def test_with_m_only_dict(self, time_index: npt.NDArray[np.int_]) -> None:
        """Test with dict containing only m key (customizing basis functions).

        This tests the bug fix where {"m": 200} was incorrectly falling through
        to parameterize_from_data which doesn't accept m parameter.
        """
        config: dict[str, int] = {"m": 200}
        hsgp = create_hsgp_from_config(
            X=DataArray(time_index, dims=("date",)), dims="date", config=config
        )
        assert isinstance(hsgp, SoftPlusHSGP)
        assert hsgp.m == 200

    def test_with_L_only_dict(self, time_index: npt.NDArray[np.int_]) -> None:
        """Test with dict containing only L key (customizing basis extent).

        This tests the bug fix where {"L": 100.0} was incorrectly falling through
        to parameterize_from_data which doesn't accept L parameter.
        """
        config: dict[str, float] = {"L": 100.0}
        hsgp = create_hsgp_from_config(
            X=DataArray(time_index, dims=("date",)), dims="date", config=config
        )
        assert isinstance(hsgp, SoftPlusHSGP)
        assert hsgp.L == 100.0

    def test_with_m_and_L_dict(self, time_index: npt.NDArray[np.int_]) -> None:
        """Test with dict containing m and L keys."""
        config: dict[str, int | float] = {"m": 150, "L": 80.0}
        hsgp = create_hsgp_from_config(
            X=DataArray(time_index, dims=("date",)), dims="date", config=config
        )
        assert isinstance(hsgp, SoftPlusHSGP)
        assert hsgp.m == 150
        assert hsgp.L == 80.0

    def test_config_with_X_key_does_not_raise(
        self, time_index: npt.NDArray[np.int_]
    ) -> None:
        """Test that config dict with X key doesn't cause multiple values error.

        The explicitly passed X argument should take precedence over config["X"].
        """
        different_X = np.arange(10, 62)  # Different from time_index
        config: dict[str, float | npt.NDArray[np.int_]] = {
            "ls_lower": 0.3,
            "ls_upper": 2.0,
            "X": different_X,
        }
        # Should not raise TypeError: got multiple values for keyword argument 'X'
        hsgp = create_hsgp_from_config(X=time_index, dims="date", config=config)
        assert isinstance(hsgp, SoftPlusHSGP)
        # The explicitly passed X should be used, not the one in config
        np.testing.assert_array_equal(hsgp.X.eval(), time_index)

    def test_config_with_dims_key_does_not_raise(
        self, time_index: npt.NDArray[np.int_]
    ) -> None:
        """Test that config dict with dims key doesn't cause multiple values error.

        The explicitly passed dims argument should take precedence over config["dims"].
        """
        config: dict[str, float | str] = {
            "ls_lower": 0.3,
            "ls_upper": 2.0,
            "dims": "other_dim",
        }
        # Should not raise TypeError: got multiple values for keyword argument 'dims'
        hsgp = create_hsgp_from_config(X=time_index, dims="date", config=config)
        assert isinstance(hsgp, SoftPlusHSGP)
        # The explicitly passed dims should be used, not the one in config
        assert hsgp.dims == ("date",)

    def test_config_with_X_mid_key_does_not_raise(
        self, time_index: npt.NDArray[np.int_]
    ) -> None:
        """Test that config dict with X_mid key doesn't cause multiple values error.

        The explicitly passed X_mid argument should take precedence over config["X_mid"].
        """
        config: dict[str, float] = {
            "ls_lower": 0.3,
            "ls_upper": 2.0,
            "X_mid": 100.0,  # Different from what we'll pass explicitly
        }
        # Should not raise TypeError: got multiple values for keyword argument 'X_mid'
        hsgp = create_hsgp_from_config(
            X=time_index, dims="date", config=config, X_mid=26.0
        )
        assert isinstance(hsgp, SoftPlusHSGP)
        # The explicitly passed X_mid should be used, not the one in config
        assert hsgp.X_mid == 26.0

    def test_config_with_all_reserved_keys_does_not_raise(
        self, time_index: npt.NDArray[np.int_]
    ) -> None:
        """Test that config dict with all reserved keys doesn't cause errors.

        Reserved keys (X, dims, X_mid) in config should be filtered out.
        """
        different_X = np.arange(100, 152)
        config: dict[str, float | str | npt.NDArray[np.int_]] = {
            "ls_lower": 0.3,
            "ls_upper": 2.0,
            "X": different_X,
            "dims": "wrong_dim",
            "X_mid": 999.0,
        }
        # Should not raise TypeError for any of the reserved keys
        hsgp = create_hsgp_from_config(
            X=time_index, dims=("date", "channel"), config=config, X_mid=26.0
        )
        assert isinstance(hsgp, SoftPlusHSGP)
        # All explicitly passed values should take precedence
        np.testing.assert_array_equal(hsgp.X.eval(), time_index)
        assert hsgp.dims == ("date", "channel")
        assert hsgp.X_mid == 26.0
