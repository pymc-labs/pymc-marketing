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
from copy import deepcopy

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from graphviz.graphs import Digraph
from preliz.distributions.distributions import Distribution
from pymc.model_graph import fast_eval

from pymc_marketing.prior import (
    MuAlreadyExistsError,
    Prior,
    UnknownTransformError,
    UnsupportedDistributionError,
    UnsupportedParameterizationError,
    UnsupportedShapeError,
    get_transform,
)


def test_missing_transform() -> None:
    with pytest.raises(UnknownTransformError):
        get_transform("foo_bar")


def test_get_item() -> None:
    var = Prior("Normal", mu=0, sigma=1)

    assert var["mu"] == 0
    assert var["sigma"] == 1


def test_noncentered_with_scalars() -> None:
    with pytest.raises(ValueError):
        Prior(
            "Normal",
            mu=2,
            sigma=10,
            centered=False,
        )


def test_noncentered_needs_params() -> None:
    with pytest.raises(ValueError):
        Prior(
            "Normal",
            centered=False,
        )


def test_different_than_pymc_params() -> None:
    with pytest.raises(ValueError):
        Prior("Normal", mu=0, b=1)


def test_non_unique_dims() -> None:
    with pytest.raises(ValueError):
        Prior("Normal", mu=0, sigma=1, dims=("channel", "channel"))


def test_doesnt_check_validity_parameterization() -> None:
    try:
        Prior("Normal", mu=0, sigma=1, tau=1)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_doesnt_check_validity_values() -> None:
    try:
        Prior("Normal", mu=0, sigma=-1)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_preliz() -> None:
    var = Prior("Normal", mu=0, sigma=1)
    dist = var.preliz
    assert isinstance(dist, Distribution)


@pytest.mark.parametrize(
    "var, expected",
    [
        (Prior("Normal", mu=0, sigma=1), 'Prior("Normal", mu=0, sigma=1)'),
        (
            Prior("Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal")),
            'Prior("Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal"))',
        ),
        (Prior("Normal", dims="channel"), 'Prior("Normal", dims="channel")'),
        (
            Prior("Normal", mu=0, sigma=1, transform="sigmoid"),
            'Prior("Normal", mu=0, sigma=1, transform="sigmoid")',
        ),
    ],
)
def test_str(var, expected) -> None:
    assert str(var) == expected


@pytest.mark.parametrize(
    "var",
    [
        Prior("Normal", mu=0, sigma=1),
        Prior("Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal"), dims="channel"),
        Prior("Normal", dims=("geo", "channel")),
    ],
)
def test_repr(var) -> None:
    assert eval(repr(var)) == var  # noqa: S307


def test_invalid_distribution() -> None:
    with pytest.raises(UnsupportedDistributionError):
        Prior("Invalid")


def test_broadcast_doesnt_work():
    with pytest.raises(UnsupportedShapeError):
        Prior(
            "Normal",
            mu=0,
            sigma=Prior("HalfNormal", sigma=1, dims="x"),
            dims="y",
        )


def test_dim_workaround_flaw() -> None:
    distribution = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="y",
    )

    try:
        distribution["mu"].dims = "x"
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

    with pytest.raises(UnsupportedShapeError):
        distribution._param_dims_work()


def test_noncentered_error() -> None:
    with pytest.raises(UnsupportedParameterizationError):
        Prior(
            "Gamma",
            mu=0,
            sigma=1,
            dims="x",
            centered=False,
        )


def test_create_variable_multiple_times() -> None:
    mu = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="channel",
        centered=False,
    )

    coords = {
        "channel": ["a", "b", "c"],
    }
    with pm.Model(coords=coords) as model:
        mu.create_variable("mu")
        mu.create_variable("mu_2")

    suffixes = [
        "",
        "_offset",
        "_mu",
        "_sigma",
    ]
    dims = [(3,), (3,), (), ()]

    for prefix in ["mu", "mu_2"]:
        for suffix, dim in zip(suffixes, dims, strict=False):
            assert fast_eval(model[f"{prefix}{suffix}"]).shape == dim


@pytest.fixture
def large_var() -> Prior:
    mu = Prior(
        "Normal",
        mu=Prior("Normal", mu=1),
        sigma=Prior("HalfNormal"),
        dims="channel",
        centered=False,
    )
    sigma = Prior("HalfNormal", sigma=Prior("HalfNormal"), dims="geo")

    return Prior("Normal", mu=mu, sigma=sigma, dims=("geo", "channel"))


def test_create_variable(large_var) -> None:
    coords = {
        "channel": ["a", "b", "c"],
        "geo": ["x", "y"],
    }
    with pm.Model(coords=coords) as model:
        large_var.create_variable("var")

    var_names = [
        "var",
        "var_mu",
        "var_sigma",
        "var_mu_offset",
        "var_mu_mu",
        "var_mu_sigma",
        "var_sigma_sigma",
    ]
    assert set(var.name for var in model.unobserved_RVs) == set(var_names)
    dims = [
        (2, 3),
        (3,),
        (2,),
        (3,),
        (),
        (),
        (),
    ]
    for var_name, dim in zip(var_names, dims, strict=False):
        assert fast_eval(model[var_name]).shape == dim


def test_transform() -> None:
    var = Prior("Normal", mu=0, sigma=1, transform="sigmoid")

    with pm.Model() as model:
        var.create_variable("var")

    var_names = [
        "var",
        "var_raw",
    ]
    dims = [
        (),
        (),
    ]
    for var_name, dim in zip(var_names, dims, strict=False):
        assert fast_eval(model[var_name]).shape == dim


def test_to_json(large_var) -> None:
    data = large_var.to_json()

    assert data == {
        "dist": "Normal",
        "kwargs": {
            "mu": {
                "dist": "Normal",
                "kwargs": {
                    "mu": {
                        "dist": "Normal",
                        "kwargs": {
                            "mu": 1,
                        },
                    },
                    "sigma": {
                        "dist": "HalfNormal",
                    },
                },
                "centered": False,
                "dims": ("channel",),
            },
            "sigma": {
                "dist": "HalfNormal",
                "kwargs": {
                    "sigma": {
                        "dist": "HalfNormal",
                    },
                },
                "dims": ("geo",),
            },
        },
        "dims": ("geo", "channel"),
    }


def test_to_json_numpy() -> None:
    var = Prior("Normal", mu=np.array([0, 10, 20]), dims="channel")
    assert var.to_json() == {
        "dist": "Normal",
        "kwargs": {
            "mu": [0, 10, 20],
        },
        "dims": ("channel",),
    }


def test_json_round_trip(large_var) -> None:
    assert Prior.from_json(large_var.to_json()) == large_var


def test_constrain_with_transform_error() -> None:
    var = Prior("Normal", mu=0, sigma=1, transform="sigmoid")

    with pytest.raises(ValueError):
        var.constrain(lower=0, upper=1)


def test_constrain(mocker) -> None:
    var = Prior("Normal", mu=0, sigma=1)

    mocker.patch(
        "pymc.find_constrained_prior",
        return_value={
            "mu": 5,
            "sigma": 2,
        },
    )

    new_var = var.constrain(lower=0, upper=1)
    assert new_var == Prior("Normal", mu=5, sigma=2)


def test_dims_change() -> None:
    var = Prior("Normal", mu=0, sigma=1)
    var.dims = "channel"

    assert var.dims == ("channel",)


def test_dims_change_error() -> None:
    mu = Prior("Normal", dims="channel")
    var = Prior("Normal", mu=mu, dims="channel")

    with pytest.raises(UnsupportedShapeError):
        var.dims = "geo"


def test_deepcopy() -> None:
    priors = {
        "alpha": Prior("Beta", alpha=1, beta=1),
        "gamma": Prior("Normal", mu=0, sigma=1),
    }

    new_priors = deepcopy(priors)
    priors["alpha"].dims = "channel"

    assert new_priors["alpha"].dims == ()


@pytest.fixture
def mmm_default_model_config():
    return {
        "intercept": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}},
        "likelihood": {
            "dist": "Normal",
            "kwargs": {
                "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
            },
        },
        "gamma_control": {
            "dist": "Normal",
            "kwargs": {"mu": 0, "sigma": 2},
            "dims": "control",
        },
        "gamma_fourier": {
            "dist": "Laplace",
            "kwargs": {"mu": 0, "b": 1},
            "dims": "fourier_mode",
        },
    }


def test_backwards_compat(mmm_default_model_config) -> None:
    result = {
        param: Prior.from_json(value)
        for param, value in mmm_default_model_config.items()
    }
    assert result == {
        "intercept": Prior("Normal", mu=0, sigma=2),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
        "gamma_control": Prior("Normal", mu=0, sigma=2, dims="control"),
        "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
    }


def test_sample_prior() -> None:
    var = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="channel",
        transform="sigmoid",
    )

    coords = {"channel": ["A", "B", "C"]}
    prior = var.sample_prior(coords=coords, samples=25)

    assert isinstance(prior, xr.Dataset)
    assert prior.sizes == {"chain": 1, "draw": 25, "channel": 3}


def test_graph() -> None:
    hierarchical_distribution = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="channel",
    )

    G = hierarchical_distribution.graph()
    assert isinstance(G, Digraph)


def test_from_json_list() -> None:
    data = {
        "dist": "Normal",
        "kwargs": {
            "mu": [0, 1, 2],
            "sigma": 1,
        },
        "dims": "channel",
    }

    var = Prior.from_json(data)
    assert var.dims == ("channel",)
    assert isinstance(var["mu"], np.ndarray)
    np.testing.assert_array_equal(var["mu"], [0, 1, 2])


def test_from_json_list_dims() -> None:
    data = {
        "dist": "Normal",
        "kwargs": {
            "mu": 0,
            "sigma": 1,
        },
        "dims": ["channel", "geo"],
    }

    var = Prior.from_json(data)
    assert var.dims == ("channel", "geo")


def test_to_json_transform() -> None:
    dist = Prior("Normal", transform="sigmoid")

    data = dist.to_json()
    assert data == {
        "dist": "Normal",
        "transform": "sigmoid",
    }


def test_equality_non_prior() -> None:
    dist = Prior("Normal")

    assert dist != 1


def test_deepcopy_memo() -> None:
    memo = {}
    dist = Prior("Normal")
    deepcopy(dist, memo)
    assert len(memo) == 2
    deepcopy(dist, memo)

    assert len(memo) == 2


def test_create_likelihood_variable() -> None:
    distribution = Prior("Normal", sigma=Prior("HalfNormal"))

    with pm.Model() as model:
        mu = pm.Normal("mu")

        data = distribution.create_likelihood_variable("data", mu=mu, observed=10)

    assert model.observed_RVs == [data]
    assert "data_sigma" in model


def test_create_likelihood_variable_already_has_mu() -> None:
    distribution = Prior("Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal"))

    with pm.Model():
        mu = pm.Normal("mu")

        with pytest.raises(MuAlreadyExistsError):
            distribution.create_likelihood_variable("data", mu=mu, observed=10)


def test_create_likelihood_non_mu_parameterized_distribution() -> None:
    distribution = Prior("Cauchy")

    with pm.Model():
        mu = pm.Normal("mu")
        with pytest.raises(UnsupportedDistributionError):
            distribution.create_likelihood_variable("data", mu=mu, observed=10)
