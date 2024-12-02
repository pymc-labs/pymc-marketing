#   Copyright 2024 The PyMC Labs Developers
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
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import ValidationError

from pymc_marketing.hsgp_kwargs import (
    HSGP,
    approx_hsgp_hyperparams,
    create_complexity_penalizing_prior,
)
from pymc_marketing.prior import Prior


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(alpha=2),
        dict(lower=-1),
    ],
    ids=["invalid_alpha", "invalid_lower"],
)
def test_create_complexity_penalizing_prior_raises(kwargs) -> None:
    with pytest.raises(ValidationError):
        create_complexity_penalizing_prior(**kwargs)


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return np.arange(10)


@pytest.fixture(scope="module")
def default_hsgp(data) -> HSGP:
    return HSGP.parameterize_from_data(data, dims="time")


def test_hspg_class_method_assigns_data(data, default_hsgp) -> None:
    np.testing.assert_array_equal(default_hsgp.X.data, data)
    assert default_hsgp.X_mid == 4.5


def test_hspg_class_method_default_distributions(default_hsgp) -> None:
    assert isinstance(default_hsgp.ls, Prior)
    assert default_hsgp.ls.distribution == "Weibull"
    assert isinstance(default_hsgp.eta, Prior)
    assert default_hsgp.eta.distribution == "Exponential"


def test_hsgp_class_method_ls_upper_changes_distribution(data) -> None:
    hsgp = HSGP.parameterize_from_data(data, dims="time", ls_upper=10)
    assert isinstance(hsgp.ls, Prior)
    assert hsgp.ls.distribution == "InverseGamma"
    assert isinstance(hsgp.eta, Prior)
    assert hsgp.eta.distribution == "Exponential"


def test_hspg_makes_dims_tuple(default_hsgp) -> None:
    assert default_hsgp.dims == ("time",)


def test_unordered_bounds_raises() -> None:
    match = "The boundaries are out of order"
    with pytest.raises(ValueError, match=match):
        approx_hsgp_hyperparams(
            x=None,
            x_center=None,
            lengthscale_range=(1, 0),
            cov_func=None,
        )


def test_unsupported_cov_func_raises() -> None:
    x = np.arange(10)
    x_center = 4.5
    match = "Unsupported covariance function"
    with pytest.raises(ValueError, match=match):
        approx_hsgp_hyperparams(
            x=x,
            x_center=x_center,
            lengthscale_range=(1, 2),
            cov_func="unsupported",
        )


def test_curve_workflow(default_hsgp, data) -> None:
    coords = {default_hsgp.dims[0]: data}
    prior = default_hsgp.sample_prior(coords=coords, samples=25)
    assert isinstance(prior, xr.Dataset)
    curve = prior["f"]
    fig, axes = default_hsgp.plot_curve(curve)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert len(axes) == 1
    assert isinstance(axes[0], Axes)
    plt.close()
