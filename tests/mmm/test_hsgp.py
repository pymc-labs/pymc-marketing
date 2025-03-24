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
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import ValidationError

from pymc_marketing.mmm.hsgp import (
    HSGP,
    CovFunc,
    HSGPPeriodic,
    PeriodicCovFunc,
    SoftPlusHSGP,
    approx_hsgp_hyperparams,
    create_complexity_penalizing_prior,
)
from pymc_marketing.model_graph import deterministics_to_flat
from pymc_marketing.prior import Prior, UnknownTransformError


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


@pytest.mark.parametrize(
    argnames="cov_func",
    argvalues=["expquad", "matern32", "matern52"],
    ids=["exp_quad", "matern32", "matern52"],
)
def test_supported_cov_func(cov_func) -> None:
    x = np.arange(10)
    x_center = 4.5
    _ = approx_hsgp_hyperparams(
        x=x,
        x_center=x_center,
        lengthscale_range=(1, 2),
        cov_func=cov_func,
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


def test_X_at_init_stores_as_tensor_variable() -> None:
    X = np.arange(10)
    hsgp = HSGP(X=X, dims="time", m=200, L=5, eta=1, ls=1)
    assert isinstance(hsgp.X, pt.TensorVariable)


@pytest.fixture(scope="module")
def periodic_hsgp(data) -> HSGP:
    scale = 1
    ls = 1
    hsgp = HSGPPeriodic(scale=scale, ls=ls, m=20, period=60, dims="time")
    hsgp.register_data(data)
    return hsgp


@pytest.mark.parametrize(
    "hsgp_fixture_name",
    [
        "default_hsgp",
        "periodic_hsgp",
    ],
    ids=["HSGP", "HSGPPeriodic"],
)
def test_curve_workflow(request, hsgp_fixture_name, data) -> None:
    hsgp = request.getfixturevalue(hsgp_fixture_name)
    coords = {hsgp.dims[0]: data}
    prior = hsgp.sample_prior(coords=coords, draws=25)
    assert isinstance(prior, xr.Dataset)
    curve = prior["f"]
    fig, axes = hsgp.plot_curve(curve)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert len(axes) == 1
    assert isinstance(axes[0], Axes)
    plt.close()


def test_hsgp_to_dict() -> None:
    eta = Prior("Exponential", lam=pt.as_tensor_variable(1))
    ls = Prior(
        "Weibull",
        alpha=1,
        beta=pt.as_tensor_variable(1),
        transform="reciprocal",
    )
    X = np.arange(10)
    hsgp = HSGP(
        eta=eta,
        ls=ls,
        dims="time",
        m=20,
        L=30,
        X=X,
    )
    data = hsgp.to_dict()

    assert data == {
        "L": 30.0,
        "m": 20,
        "ls": {
            "dist": "Weibull",
            "kwargs": {"alpha": 1, "beta": 1},
            "transform": "reciprocal",
        },
        "eta": {
            "dist": "Exponential",
            "kwargs": {"lam": 1},
        },
        "X_mid": None,
        "centered": False,
        "dims": ("time",),
        "drop_first": True,
        "cov_func": CovFunc.ExpQuad,
        "transform": None,
        "demeaned_basis": False,
    }


def test_hsgp_periodic_to_dict() -> None:
    scale = Prior("Exponential", lam=pt.as_tensor_variable(1))
    ls = Prior(
        "Weibull",
        alpha=1,
        beta=pt.as_tensor_variable(1),
        transform="reciprocal",
    )
    X = np.arange(10)
    hsgp = HSGPPeriodic(
        ls=ls,
        scale=scale,
        period=60,
        dims="time",
        m=20,
        X=X,
    )

    data = hsgp.to_dict()

    assert data == {
        "m": 20,
        "period": 60.0,
        "cov_func": PeriodicCovFunc.Periodic,
        "ls": {
            "dist": "Weibull",
            "kwargs": {"alpha": 1, "beta": 1},
            "transform": "reciprocal",
        },
        "scale": {
            "dist": "Exponential",
            "kwargs": {"lam": 1},
        },
        "X_mid": None,
        "dims": ("time",),
        "transform": None,
        "demeaned_basis": False,
    }


def test_non_prior_parameters_still_serialize() -> None:
    hsgp = HSGP(m=10, dims="time", ls=1, eta=1, L=5)

    data = hsgp.to_dict()

    assert data == {
        "L": 5,
        "m": 10,
        "ls": 1,
        "eta": 1,
        "X_mid": None,
        "centered": False,
        "dims": ("time",),
        "drop_first": True,
        "cov_func": CovFunc.ExpQuad,
        "transform": None,
        "demeaned_basis": False,
    }


def test_higher_dimension_hsgp(data) -> None:
    hsgp = HSGP.parameterize_from_data(data, dims=("time", "channel", "product"))
    coords = {
        "time": np.arange(10),
        "channel": np.arange(5),
        "product": np.arange(3),
    }
    prior = hsgp.sample_prior(draws=25, coords=coords)
    assert isinstance(prior, xr.Dataset)
    curve = prior["f"]
    assert curve.shape == (1, 25, 10, 5, 3)


def test_from_dict_with_non_dictionary_distributions_hsgp() -> None:
    data = {
        "L": 5,
        "m": 10,
        "ls": 1,
        "eta": 1,
        "X_mid": None,
        "centered": False,
        "dims": ("time",),
        "drop_first": True,
        "cov_func": CovFunc.ExpQuad,
    }

    hsgp = HSGP.from_dict(data)
    assert hsgp.L == 5
    assert hsgp.m == 10
    assert hsgp.ls == 1
    assert hsgp.eta == 1
    assert hsgp.X_mid is None
    assert hsgp.centered is False
    assert hsgp.dims == ("time",)
    assert hsgp.drop_first is True
    assert hsgp.cov_func == CovFunc.ExpQuad
    assert hsgp.transform is None


def test_from_dict_with_non_dictionary_distribution_hspg_periodic() -> None:
    data = {
        "m": 20,
        "period": 60.0,
        "cov_func": PeriodicCovFunc.Periodic,
        "ls": 1,
        "scale": 1,
        "X_mid": None,
        "dims": ("time",),
    }

    hsgp = HSGPPeriodic.from_dict(data)
    assert hsgp.m == 20
    assert hsgp.period == 60.0
    assert hsgp.cov_func == PeriodicCovFunc.Periodic
    assert hsgp.ls == 1
    assert hsgp.scale == 1
    assert hsgp.X_mid is None
    assert hsgp.dims == ("time",)
    assert hsgp.transform is None


def test_hsgp_with_shared_data():
    """
    Test that HSGP works with a shared variable (pm.MutableData / pm.Data) and that
    the computed graph properly includes and depends on the shared data.
    """
    n_points = 10
    X = np.arange(n_points, dtype=float)
    coords = {"time": X}

    # Create a model and a shared data variable using pm.MutableData
    with pm.Model(coords=coords) as model:
        # Create a shared data variable with the name "X_shared"
        X_shared = pm.Data("X_shared", X, dims="time")
        # Parameterize the HSGP using the shared data
        hsgp = HSGP.parameterize_from_data(X_shared, dims="time")
        # Create the deterministic variable "f" from the HSGP configuration
        f = hsgp.create_variable("f")

        # Check that "f" is added to the model variables
        assert "f" in model.named_vars

        # Ensure that the stored X is a shared tensor variable
        assert isinstance(hsgp.X, pt.TensorVariable)

        # Verify that f depends on X_shared in the computational graph
        assert any(
            var.name == "X_shared"
            for var in pytensor.graph.basic.ancestors([f])
            if hasattr(var, "name")
        ), "f is not connected to X_shared in the computational graph"

        # Sample from prior to get initial values
        prior = pm.sample_prior_predictive(draws=1)

        # prior should have a "f" variable
        assert "f" in prior.prior


def test_soft_plus_hsgp_is_centered_around_1() -> None:
    seed = sum(map(ord, "Is centered around 1"))
    rng = np.random.default_rng(seed)
    hsgp = SoftPlusHSGP(
        m=10,
        L=5,
        dims="date",
        ls=Prior("Exponential", lam=1),
        eta=Prior("Exponential", lam=1),
    )

    n_points = 100
    data = np.linspace(0, 10, n_points)

    n_out_of_sample = 1
    insample = data[: n_points - n_out_of_sample]

    prior_samples = 50

    coords = {"date": insample}
    with pm.Model(coords=coords):
        X = pm.Data("X", insample, dims="date")
        hsgp.register_data(X).create_variable("f")

        idata = pm.sample_prior_predictive(prior_samples, random_seed=rng)

    f_mean = idata.prior["f"].mean("date")

    np.testing.assert_allclose(f_mean, 1.0)


def test_soft_plus_hsgp_continous_with_new_data() -> None:
    seed = sum(map(ord, "No jump from in-sample to out-of-sample"))
    rng = np.random.default_rng(seed)
    hsgp = SoftPlusHSGP(
        m=10,
        L=5,
        dims="date",
        ls=Prior("Exponential", lam=1),
        eta=Prior("Exponential", lam=1),
    )

    n_points = 100
    data = np.linspace(0, 10, n_points)

    n_out_of_sample = 1
    insample = data[: n_points - n_out_of_sample]
    outsample = data[n_points - n_out_of_sample :]

    prior_samples = 50

    coords = {"date": insample}
    with pm.Model(coords=coords) as model:
        X = pm.Data("X", insample, dims="date")
        hsgp.register_data(X).create_variable("f")

        idata = pm.sample_prior_predictive(prior_samples, random_seed=rng)

    # set posterior as prior for out of sample
    idata["posterior"] = idata.prior

    with deterministics_to_flat(model, names=hsgp.deterministics_to_replace("f")):
        pm.set_data({"X": outsample}, coords={"date": outsample})

        idata.extend(
            pm.sample_posterior_predictive(
                idata,
                var_names=["f"],
                random_seed=rng,
            )
        )

    jump = idata.posterior_predictive["f"].isel(date=0) - idata.prior["f"].isel(date=-1)
    diffs = idata.prior["f"].diff(dim="date")

    q = 0.95
    threshold = abs(diffs).quantile(q, dim="date")
    stat = abs(jump) < threshold

    # Approx 95% of the differences should be below the threshold
    assert stat.mean().item() >= (q - 0.05)


def test_hsgp_with_unknown_transform_errors() -> None:
    X = np.arange(10)
    match = "Neither pytensor.tensor nor pymc.math"
    with pytest.raises(UnknownTransformError, match=match):
        HSGP.parameterize_from_data(X, dims="time", transform="unknown")


def test_hsgp_with_transform() -> None:
    X = np.arange(10)
    hsgp = HSGP.parameterize_from_data(X, dims="time", transform="sigmoid")

    coords = {"time": X}
    prior = hsgp.sample_prior(draws=25, coords=coords)
    assert "f_raw" in prior
    assert "f" in prior

    assert ((prior["f"] >= 0) & (prior["f"] <= 1)).all()


def test_hsgp_periodic_with_transform() -> None:
    X = np.arange(10)

    hsgp = HSGPPeriodic(
        m=20,
        dims="time",
        ls=Prior("Exponential", lam=1),
        scale=Prior("Exponential", lam=1),
        period=60,
        transform="sigmoid",
    ).register_data(X)

    coords = {"time": X}
    prior = hsgp.sample_prior(draws=25, coords=coords)
    assert "f_raw" in prior
    assert "f" in prior

    assert ((prior["f"] >= 0) & (prior["f"] <= 1)).all()
