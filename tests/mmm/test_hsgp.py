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
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import ValidationError
from pymc_extras.prior import Prior, UnknownTransformError
from pytensor.xtensor.type import XTensorVariable

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
    match = r"The boundaries are out of order"
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
    match = r"Unsupported covariance function"
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
    assert isinstance(hsgp.X, XTensorVariable)


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
        "__type__": "pymc_marketing.mmm.hsgp.HSGP",
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
        "__type__": "pymc_marketing.mmm.hsgp.HSGPPeriodic",
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
        "__type__": "pymc_marketing.mmm.hsgp.HSGP",
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


class TestDimsNormalization:
    """Tests for dims list-to-tuple normalization in the validator."""

    @pytest.mark.parametrize(
        "dims, expected",
        [
            (["time"], ("time",)),
            (["time", "channel"], ("time", "channel")),
            ("time", ("time",)),
            (("time",), ("time",)),
        ],
    )
    @pytest.mark.parametrize(
        "cls, extra_kwargs",
        [
            (HSGP, {"ls": 1, "eta": 1, "L": 5}),
            (HSGPPeriodic, {"ls": 1, "scale": 1, "period": 52}),
        ],
    )
    def test_dims_normalized_to_tuple(self, cls, extra_kwargs, dims, expected):
        obj = cls(m=10, dims=dims, **extra_kwargs)
        assert obj.dims == expected
        assert isinstance(obj.dims, tuple)

    @pytest.mark.parametrize("dims", [[], ()])
    def test_empty_dims_raises(self, dims):
        with pytest.raises(ValueError, match="At least one dimension is required"):
            HSGP(m=10, dims=dims, ls=1, eta=1, L=5)

    @pytest.mark.parametrize(
        "cls, base_data, dims",
        [
            (HSGP, {"L": 5, "m": 10, "ls": 1, "eta": 1}, ["time"]),
            (HSGP, {"L": 5, "m": 10, "ls": 1, "eta": 1}, ["time", "channel"]),
            (HSGPPeriodic, {"m": 20, "period": 60.0, "ls": 1, "scale": 1}, ["time"]),
        ],
    )
    def test_from_dict_with_list_dims(self, cls, base_data, dims):
        data = {**base_data, "dims": dims}
        obj = cls.from_dict(data)
        assert obj.dims == tuple(dims)
        assert isinstance(obj.dims, tuple)


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
        X_shared = pmd.Data("X_shared", X, dims="time")
        # Parameterize the HSGP using the shared data
        hsgp = HSGP.parameterize_from_data(X_shared, dims="time")
        # Create the deterministic variable "f" from the HSGP configuration
        f = hsgp.create_variable("f", xdist=True)

        # Check that "f" is added to the model variables
        assert "f" in model.named_vars

        # Ensure that the stored X is a shared tensor variable
        assert isinstance(hsgp.X, XTensorVariable)

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
        hsgp.register_data(X).create_variable("f", xdist=True)

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
        hsgp.register_data(X).create_variable("f", xdist=True)

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
    match = r"not present in pytensor.tensor or pymc.math"
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


@pytest.mark.parametrize(
    "cov_func",
    ["expquad", "matern32", "matern52"],
    ids=["expquad", "matern32", "matern52"],
)
def test_hsgp_sample_prior_with_covariance(cov_func) -> None:
    """Ensure HSGP can sample prior for supported covariance functions using PyMC directly."""
    X = np.arange(10)
    hsgp = HSGP.parameterize_from_data(X, dims="time", cov_func=cov_func)

    coords = {"time": X}
    prior = hsgp.sample_prior(draws=25, coords=coords)

    assert isinstance(prior, xr.Dataset)
    assert "f" in prior
    assert prior["f"].dims == ("chain", "draw", "time")


def test_softplus_hsgp_intercept_is_non_negative() -> None:
    """Pure prior test: intercept * SoftPlusHSGP multiplier stays non-negative.

    Construct an intercept baseline with HalfNormal(5) and a time-varying multiplier
    from SoftPlusHSGP. The SoftPlusHSGP maps to positive values and normalizes by the
    time-mean, ensuring positivity and mean 1. We verify multiplier > 0 everywhere
    and the intercept contribution is non-negative.
    """
    seed = sum(map(ord, "SoftPlus intercept can be negative"))
    rng = np.random.default_rng(seed)

    # Choose fixed hyperparameters to induce sufficient variability deterministically
    hsgp = SoftPlusHSGP(
        m=20,
        L=10,
        dims="date",
        ls=1.0,
        eta=3.0,
    )

    n_points = 100
    data = np.linspace(0, 10, n_points)

    coords = {"date": data}
    with pm.Model(coords=coords):
        X = pm.Data("X", data, dims="date")
        multiplier = hsgp.register_data(X).create_variable("multiplier", xdist=True)

        intercept_baseline = pm.HalfNormal("intercept_baseline", sigma=5.0)
        pm.Deterministic(
            "intercept",
            intercept_baseline * multiplier,
            dims="date",
        )

        idata = pm.sample_prior_predictive(draws=200, random_seed=rng)

    # Multiplier strictly positive
    assert (idata.prior["multiplier"] > 0).all()

    # Multiplier time-mean should be ~1 for every chain/draw and remaining dims
    multiplier_mean = idata.prior["multiplier"].mean("date").values
    np.testing.assert_allclose(multiplier_mean, 1.0, rtol=1e-6, atol=1e-6)

    # Baseline is HalfNormal -> never negative
    assert (idata.prior["intercept_baseline"] >= 0).all()

    # Verify intercept contribution never negative
    assert (idata.prior["intercept"] >= 0).all()


# VariableFactory Test Classes
class ScalarVariableFactory:
    """A VariableFactory with scalar dimensions."""

    dims = ()  # Scalar: no dimensions

    def create_variable(self, name: str, xdist: bool = False):
        """Create a scalar variable."""
        if xdist:
            return pmd.HalfNormal(name, sigma=1.0)
        else:
            return pm.HalfNormal(name, sigma=1.0)


class NonScalarVariableFactory:
    """A VariableFactory with non-scalar dimensions."""

    dims = ("time",)  # Non-scalar: has dimensions

    def create_variable(self, name: str, xdist: bool = False):
        """Create a non-scalar variable."""
        if xdist:
            return pmd.HalfNormal(name, sigma=1.0, dims=self.dims)
        else:
            return pm.HalfNormal(name, sigma=1.0, shape=(10,))


class SerializableVariableFactory:
    """A VariableFactory with serialization support."""

    dims = ()

    def create_variable(self, name: str, xdist: bool = False):
        """Create a variable."""
        if xdist:
            return pmd.HalfNormal(name, sigma=1.0)
        else:
            return pm.HalfNormal(name, sigma=1.0)

    def to_dict(self):
        """Serialization method."""
        return {"type": "SerializableVariableFactory"}


class TestVariableFactoryHSGP:
    """Test HSGP class with VariableFactory support."""

    def test_hsgp_accepts_scalar_variable_factory(self):
        """Test that HSGP accepts scalar VariableFactory instances."""
        data = np.arange(10)
        scalar_factory = ScalarVariableFactory()

        # Should not raise
        hsgp = HSGP(
            X=data,
            dims="time",
            m=200,
            L=5,
            eta=scalar_factory,
            ls=scalar_factory,
        )

        assert hsgp.ls is scalar_factory
        assert hsgp.eta is scalar_factory

    def test_hsgp_rejects_non_scalar_ls_variable_factory(self):
        """Test that HSGP rejects non-scalar VariableFactory for ls."""
        data = np.arange(10)
        non_scalar_factory = NonScalarVariableFactory()
        scalar_factory = ScalarVariableFactory()

        with pytest.raises(ValueError, match="lengthscale prior must be scalar"):
            HSGP(
                X=data,
                dims="time",
                m=200,
                L=5,
                eta=scalar_factory,
                ls=non_scalar_factory,  # Non-scalar should fail
            )

    def test_hsgp_rejects_non_scalar_eta_variable_factory(self):
        """Test that HSGP rejects non-scalar VariableFactory for eta."""
        data = np.arange(10)
        non_scalar_factory = NonScalarVariableFactory()
        scalar_factory = ScalarVariableFactory()

        with pytest.raises(ValueError, match="eta prior must be scalar"):
            HSGP(
                X=data,
                dims="time",
                m=200,
                L=5,
                eta=non_scalar_factory,  # Non-scalar should fail
                ls=scalar_factory,
            )

    def test_hsgp_backwards_compatibility_with_prior(self):
        """Test that HSGP still works with Prior objects (backwards compatibility)."""
        data = np.arange(10)
        ls_prior = Prior("HalfNormal", sigma=1.0)
        eta_prior = Prior("HalfNormal", sigma=1.0)

        # Should not raise - Prior objects implement VariableFactory
        hsgp = HSGP(
            X=data,
            dims="time",
            m=200,
            L=5,
            eta=eta_prior,
            ls=ls_prior,
        )

        assert hsgp.ls is ls_prior
        assert hsgp.eta is eta_prior


class TestVariableFactoryHSGPPeriodic:
    """Test HSGPPeriodic class with VariableFactory support."""

    def test_hsgp_periodic_accepts_scalar_variable_factory(self):
        """Test that HSGPPeriodic accepts scalar VariableFactory instances."""
        scalar_factory = ScalarVariableFactory()

        # Should not raise
        hsgp = HSGPPeriodic(
            scale=scalar_factory,
            ls=scalar_factory,
            m=20,
            period=60,
            dims="time",
        )

        assert hsgp.ls is scalar_factory
        assert hsgp.scale is scalar_factory

    def test_hsgp_periodic_rejects_non_scalar_ls_variable_factory(self):
        """Test that HSGPPeriodic rejects non-scalar VariableFactory for ls."""
        non_scalar_factory = NonScalarVariableFactory()
        scalar_factory = ScalarVariableFactory()

        with pytest.raises(ValueError, match="lengthscale prior must be scalar"):
            HSGPPeriodic(
                scale=scalar_factory,
                ls=non_scalar_factory,  # Non-scalar should fail
                m=20,
                period=60,
                dims="time",
            )

    def test_hsgp_periodic_rejects_non_scalar_scale_variable_factory(self):
        """Test that HSGPPeriodic rejects non-scalar VariableFactory for scale."""
        non_scalar_factory = NonScalarVariableFactory()
        scalar_factory = ScalarVariableFactory()

        with pytest.raises(ValueError, match="scale prior must be scalar"):
            HSGPPeriodic(
                scale=non_scalar_factory,  # Non-scalar should fail
                ls=scalar_factory,
                m=20,
                period=60,
                dims="time",
            )

    def test_hsgp_periodic_backwards_compatibility_with_prior(self):
        """Test that HSGPPeriodic still works with Prior objects (backwards compatibility)."""
        ls_prior = Prior("HalfNormal", sigma=1.0)
        scale_prior = Prior("HalfNormal", sigma=1.0)

        # Should not raise - Prior objects implement VariableFactory
        hsgp = HSGPPeriodic(
            scale=scale_prior,
            ls=ls_prior,
            m=20,
            period=60,
            dims="time",
        )

        assert hsgp.ls is ls_prior
        assert hsgp.scale is scale_prior


class TestVariableFactorySerialization:
    """Test serialization behavior with VariableFactory objects."""

    def test_serializable_variable_factory(self):
        """Test VariableFactory with to_dict method is serializable."""
        serializable_factory = SerializableVariableFactory()

        hsgp = HSGP(
            X=np.arange(10),
            dims="time",
            m=200,
            L=5,
            eta=serializable_factory,
            ls=serializable_factory,
        )

        # Should not raise during to_dict - object has to_dict method
        result = hsgp.to_dict()
        assert isinstance(result, dict)

    def test_non_serializable_variable_factory_fallback(self):
        """Test VariableFactory without to_dict method falls back gracefully."""
        non_serializable_factory = ScalarVariableFactory()  # No to_dict method

        hsgp = HSGP(
            X=np.arange(10),
            dims="time",
            m=200,
            L=5,
            eta=non_serializable_factory,
            ls=non_serializable_factory,
        )

        # Should not raise - serialization should handle missing to_dict gracefully
        result = hsgp.to_dict()
        assert isinstance(result, dict)


class TestHSGPTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of HSGP classes."""

    @pytest.mark.parametrize(
        "hsgp_cls", [HSGP, HSGPPeriodic, SoftPlusHSGP], ids=lambda c: c.__name__
    )
    def test_to_dict_includes_type_key(self, hsgp_cls):
        if hsgp_cls is HSGPPeriodic:
            obj = hsgp_cls(m=10, scale=1.0, ls=1.0, period=365.25, dims="time")
        else:
            obj = hsgp_cls(m=10, L=1.5, eta=1.0, ls=1.0, dims="time")
        data = obj.to_dict()
        assert "__type__" in data
        expected = f"{hsgp_cls.__module__}.{hsgp_cls.__qualname__}"
        assert data["__type__"] == expected

    @pytest.mark.parametrize(
        "hsgp_cls", [HSGP, HSGPPeriodic, SoftPlusHSGP], ids=lambda c: c.__name__
    )
    def test_registered_in_type_registry(self, hsgp_cls):
        from pymc_marketing.serialization import registry

        type_key = f"{hsgp_cls.__module__}.{hsgp_cls.__qualname__}"
        assert type_key in registry._registry, f"{hsgp_cls.__name__} not registered"

    def test_hsgp_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.hsgp_kwargs import CovFunc
        from pymc_marketing.serialization import registry

        original = HSGP(
            m=15,
            L=2.5,
            eta=Prior("Exponential", lam=2.0),
            ls=Prior("InverseGamma", alpha=3.0, beta=2.0),
            dims=("time", "geo"),
            centered=True,
            drop_first=False,
            cov_func=CovFunc.Matern52,
            demeaned_basis=True,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is HSGP
        assert restored.m == 15
        assert restored.L == 2.5
        assert isinstance(restored.dims, tuple)
        assert restored.dims == ("time", "geo")
        assert restored.centered is True
        assert restored.drop_first is False
        assert restored.cov_func == CovFunc.Matern52
        assert restored.demeaned_basis is True
        assert restored == original

    def test_hsgp_periodic_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry

        original = HSGPPeriodic(
            m=15,
            scale=Prior("Exponential", lam=1.5),
            ls=Prior("InverseGamma", alpha=2.0, beta=1.0),
            period=7.0,
            dims=("time", "geo"),
            demeaned_basis=True,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is HSGPPeriodic
        assert restored.m == 15
        assert restored.period == 7.0
        assert isinstance(restored.dims, tuple)
        assert restored.dims == ("time", "geo")
        assert restored.demeaned_basis is True
        assert restored == original

    def test_softplus_hsgp_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry

        original = SoftPlusHSGP(
            m=20,
            L=3.0,
            eta=Prior("Exponential", lam=1.0),
            ls=2.0,
            dims="time",
            centered=True,
            drop_first=False,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is SoftPlusHSGP
        assert restored.m == 20
        assert restored.L == 3.0
        assert restored.centered is True
        assert restored.drop_first is False
        assert restored == original


class TestDeferredFactoryInHSGP:
    def test_deferred_factory_in_eta(self):
        from pymc_marketing.serialization import DeferredFactory

        deferred = DeferredFactory(
            factory="pymc_marketing.mmm.hsgp.create_eta_prior",
            kwargs={"upper": 5.0, "mass": 0.95},
        )
        hsgp = HSGP(m=10, L=1.5, eta=deferred, ls=1.0, dims="time")
        data = hsgp.to_dict()
        assert data["eta"]["__deferred__"] is True
        assert data["eta"]["factory"] == "pymc_marketing.mmm.hsgp.create_eta_prior"

    def test_deferred_roundtrip_all_parameters(self):
        from pymc_marketing.serialization import DeferredFactory, registry

        deferred_eta = DeferredFactory(
            factory="pymc_marketing.mmm.hsgp.create_eta_prior",
            kwargs={"upper": 5.0, "mass": 0.95},
        )
        deferred_ls = DeferredFactory(
            factory="pymc_marketing.mmm.hsgp.create_constrained_inverse_gamma_prior",
            kwargs={"upper": 30.0, "lower": 1.0, "mass": 0.9},
        )
        original = HSGP(
            m=12,
            L=2.0,
            eta=deferred_eta,
            ls=deferred_ls,
            dims=("time", "geo"),
            centered=True,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is HSGP
        assert isinstance(restored.eta, DeferredFactory)
        assert isinstance(restored.ls, DeferredFactory)
        assert restored.eta.factory == deferred_eta.factory
        assert restored.eta.kwargs == deferred_eta.kwargs
        assert restored.ls.factory == deferred_ls.factory
        assert restored.m == 12
        assert restored.L == 2.0
        assert restored.dims == ("time", "geo")
        assert restored.centered is True
        assert restored.eta.resolve() is not None
