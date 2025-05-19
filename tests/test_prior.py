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
from copy import deepcopy
from typing import NamedTuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from graphviz.graphs import Digraph
from preliz.distributions.distributions import Distribution
from pydantic import ValidationError
from pymc.model_graph import fast_eval

from pymc_marketing.deserialize import (
    DESERIALIZERS,
    deserialize,
    register_deserialization,
)
from pymc_marketing.prior import (
    Censored,
    MuAlreadyExistsError,
    Prior,
    Scaled,
    UnknownTransformError,
    UnsupportedDistributionError,
    UnsupportedParameterizationError,
    UnsupportedShapeError,
    VariableFactory,
    handle_dims,
    register_tensor_transform,
    sample_prior,
)


@pytest.mark.parametrize(
    "x, dims, desired_dims, expected_fn",
    [
        (np.arange(3), "channel", "channel", lambda x: x),
        (np.arange(3), "channel", ("geo", "channel"), lambda x: x),
        (np.arange(3), "channel", ("channel", "geo"), lambda x: x[:, None]),
        (np.arange(3), "channel", ("x", "y", "channel", "geo"), lambda x: x[:, None]),
        (
            np.arange(3 * 2).reshape(3, 2),
            ("channel", "geo"),
            ("geo", "x", "y", "channel"),
            lambda x: x.T[:, None, None, :],
        ),
        (
            np.arange(4 * 2 * 3).reshape(4, 2, 3),
            ("channel", "geo", "store"),
            ("geo", "x", "store", "channel"),
            lambda x: x.swapaxes(0, 2).swapaxes(0, 1)[:, None, :, :],
        ),
    ],
    ids=[
        "same_dims",
        "different_dims",
        "dim_padding",
        "just_enough_dims",
        "transpose_and_padding",
        "swaps_and_padding",
    ],
)
def test_handle_dims(x, dims, desired_dims, expected_fn) -> None:
    result = handle_dims(x, dims, desired_dims)
    if isinstance(result, pt.TensorVariable):
        result = fast_eval(result)

    np.testing.assert_array_equal(result, expected_fn(x))


@pytest.mark.parametrize(
    "x, dims, desired_dims",
    [
        (np.ones(3), "channel", "something_else"),
        (np.ones((3, 2)), ("a", "b"), ("a", "B")),
    ],
    ids=["no_incommon", "some_incommon"],
)
def test_handle_dims_with_impossible_dims(x, dims, desired_dims) -> None:
    match = " are not a subset of the desired dims "
    with pytest.raises(UnsupportedShapeError, match=match):
        handle_dims(x, dims, desired_dims)


def test_missing_transform() -> None:
    match = "Neither pytensor.tensor nor pymc.math have the function 'foo_bar'"
    with pytest.raises(UnknownTransformError, match=match):
        Prior("Normal", transform="foo_bar")


def test_get_item() -> None:
    var = Prior("Normal", mu=0, sigma=1)

    assert var["mu"] == 0
    assert var["sigma"] == 1


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


def test_to_dict(large_var) -> None:
    data = large_var.to_dict()

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


def test_to_dict_numpy() -> None:
    var = Prior("Normal", mu=np.array([0, 10, 20]), dims="channel")
    assert var.to_dict() == {
        "dist": "Normal",
        "kwargs": {
            "mu": [0, 10, 20],
        },
        "dims": ("channel",),
    }


def test_dict_round_trip(large_var) -> None:
    assert Prior.from_dict(large_var.to_dict()) == large_var


def test_constrain_with_transform_error() -> None:
    var = Prior("Normal", transform="sigmoid")

    with pytest.raises(ValueError):
        var.constrain(lower=0, upper=1)


def test_constrain(mocker) -> None:
    var = Prior("Normal")

    mocker.patch(
        "preliz.maxent",
        return_value=mocker.Mock(params_dict={"mu": 5, "sigma": 2}),
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
        param: Prior.from_dict(value)
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


def test_sample_prior_missing_coords() -> None:
    dist = Prior("Normal", dims="channel")

    with pytest.raises(KeyError, match="Coords"):
        dist.sample_prior()


def test_to_graph() -> None:
    hierarchical_distribution = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="channel",
    )

    G = hierarchical_distribution.to_graph()
    assert isinstance(G, Digraph)


def test_from_dict_list() -> None:
    data = {
        "dist": "Normal",
        "kwargs": {
            "mu": [0, 1, 2],
            "sigma": 1,
        },
        "dims": "channel",
    }

    var = Prior.from_dict(data)
    assert var.dims == ("channel",)
    assert isinstance(var["mu"], np.ndarray)
    np.testing.assert_array_equal(var["mu"], [0, 1, 2])


def test_from_dict_list_dims() -> None:
    data = {
        "dist": "Normal",
        "kwargs": {
            "mu": 0,
            "sigma": 1,
        },
        "dims": ["channel", "geo"],
    }

    var = Prior.from_dict(data)
    assert var.dims == ("channel", "geo")


def test_to_dict_transform() -> None:
    dist = Prior("Normal", transform="sigmoid")

    data = dist.to_dict()
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
    memo[id(dist)] = dist
    deepcopy(dist, memo)
    assert len(memo) == 1
    deepcopy(dist, memo)

    assert len(memo) == 1


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


def test_non_centered_student_t() -> None:
    try:
        Prior(
            "StudentT",
            mu=Prior("Normal"),
            sigma=Prior("HalfNormal"),
            nu=Prior("HalfNormal"),
            dims="channel",
            centered=False,
        )
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_cant_reset_distribution() -> None:
    dist = Prior("Normal")
    with pytest.raises(AttributeError, match="Can't change the distribution"):
        dist.distribution = "Cauchy"


def test_nonstring_distribution() -> None:
    with pytest.raises(ValidationError, match=".*Input should be a valid string.*"):
        Prior(pm.Normal)


def test_change_the_transform() -> None:
    dist = Prior("Normal")
    dist.transform = "logit"
    assert dist.transform == "logit"


def test_nonstring_transform() -> None:
    with pytest.raises(ValidationError, match=".*Input should be a valid string.*"):
        Prior("Normal", transform=pm.math.log)


def test_checks_param_value_types() -> None:
    with pytest.raises(
        ValueError, match="Parameters must be one of the following types"
    ):
        Prior("Normal", mu="str", sigma="str")


def test_check_equality_with_numpy() -> None:
    dist = Prior("Normal", mu=np.array([1, 2, 3]), sigma=1)
    assert dist == dist.deepcopy()


def clear_custom_transforms() -> None:
    global CUSTOM_TRANSFORMS
    CUSTOM_TRANSFORMS = {}


def test_custom_transform() -> None:
    new_transform_name = "foo_bar"
    with pytest.raises(UnknownTransformError):
        Prior("Normal", transform=new_transform_name)

    register_tensor_transform(new_transform_name, lambda x: x**2)

    dist = Prior("Normal", transform=new_transform_name)
    prior = dist.sample_prior(samples=10)
    df_prior = prior.to_dataframe()

    np.testing.assert_array_equal(
        df_prior["var"].to_numpy(), df_prior["var_raw"].to_numpy() ** 2
    )


def test_custom_transform_comes_first() -> None:
    # function in pytensor.tensor
    register_tensor_transform("square", lambda x: 2 * x)

    dist = Prior("Normal", transform="square")
    prior = dist.sample_prior(samples=10)
    df_prior = prior.to_dataframe()

    np.testing.assert_array_equal(
        df_prior["var"].to_numpy(), 2 * df_prior["var_raw"].to_numpy()
    )

    clear_custom_transforms()


def test_serialize_with_pytensor() -> None:
    sigma = pt.arange(1, 4)
    dist = Prior("Normal", mu=0, sigma=sigma)

    assert dist.to_dict() == {
        "dist": "Normal",
        "kwargs": {
            "mu": 0,
            "sigma": [1, 2, 3],
        },
    }


def test_zsn_non_centered() -> None:
    try:
        Prior("ZeroSumNormal", sigma=1, centered=False)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


class Arbitrary:
    def __init__(self, dims: str | tuple[str, ...]) -> None:
        self.dims = dims

    def create_variable(self, name: str):
        return pm.Normal(name, dims=self.dims)


class ArbitraryWithoutName:
    def __init__(self, dims: str | tuple[str, ...]) -> None:
        self.dims = dims

    def create_variable(self, name: str):
        with pm.Model(name=name):
            location = pm.Normal("location", dims=self.dims)
            scale = pm.HalfNormal("scale", dims=self.dims)

            return pm.Normal("standard_normal") * scale + location


def test_sample_prior_arbitrary() -> None:
    var = Arbitrary(dims="channel")

    prior = sample_prior(var, coords={"channel": ["A", "B", "C"]}, draws=25)

    assert isinstance(prior, xr.Dataset)


def test_sample_prior_arbitrary_no_name() -> None:
    var = ArbitraryWithoutName(dims="channel")

    prior = sample_prior(var, coords={"channel": ["A", "B", "C"]}, draws=25)

    assert isinstance(prior, xr.Dataset)
    assert "var" not in prior

    prior_with = sample_prior(
        var,
        coords={"channel": ["A", "B", "C"]},
        draws=25,
        wrap=True,
    )

    assert isinstance(prior_with, xr.Dataset)
    assert "var" in prior_with


def test_create_prior_with_arbitrary() -> None:
    dist = Prior(
        "Normal",
        mu=Arbitrary(dims=("channel",)),
        sigma=1,
        dims=("channel", "geo"),
    )

    coords = {
        "channel": ["C1", "C2", "C3"],
        "geo": ["G1", "G2"],
    }
    with pm.Model(coords=coords) as model:
        dist.create_variable("var")

    assert "var_mu" in model
    var_mu = model["var_mu"]

    assert fast_eval(var_mu).shape == (len(coords["channel"]),)


def test_censored_is_variable_factory() -> None:
    normal = Prior("Normal")
    censored_normal = Censored(normal, lower=0)

    assert isinstance(censored_normal, VariableFactory)


@pytest.mark.parametrize(
    "dims, expected_dims",
    [
        ("channel", ("channel",)),
        (("channel", "geo"), ("channel", "geo")),
    ],
    ids=["string", "tuple"],
)
def test_censored_dims_from_distribution(dims, expected_dims) -> None:
    normal = Prior("Normal", dims=dims)
    censored_normal = Censored(normal, lower=0)

    assert censored_normal.dims == expected_dims


def test_censored_variables_created() -> None:
    normal = Prior("Normal", mu=Prior("Normal"), dims="dim")
    censored_normal = Censored(normal, lower=0)

    coords = {"dim": range(3)}
    with pm.Model(coords=coords) as model:
        censored_normal.create_variable("var")

    var_names = ["var", "var_mu"]
    assert set(var.name for var in model.unobserved_RVs) == set(var_names)
    dims = [(3,), ()]
    for var_name, dim in zip(var_names, dims, strict=False):
        assert fast_eval(model[var_name]).shape == dim


def test_censored_sample_prior() -> None:
    normal = Prior("Normal", dims="channel")
    censored_normal = Censored(normal, lower=0)

    coords = {"channel": ["A", "B", "C"]}
    prior = censored_normal.sample_prior(coords=coords, samples=25)

    assert isinstance(prior, xr.Dataset)
    assert prior.sizes == {"chain": 1, "draw": 25, "channel": 3}


def test_censored_to_graph() -> None:
    normal = Prior("Normal", dims="channel")
    censored_normal = Censored(normal, lower=0)

    G = censored_normal.to_graph()
    assert isinstance(G, Digraph)


def test_censored_likelihood_variable() -> None:
    normal = Prior("Normal", sigma=Prior("HalfNormal"), dims="channel")
    censored_normal = Censored(normal, lower=0)

    coords = {"channel": range(3)}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu")
        variable = censored_normal.create_likelihood_variable(
            name="likelihood",
            mu=mu,
            observed=[1, 2, 3],
        )

    assert isinstance(variable, pt.TensorVariable)
    assert model.observed_RVs == [variable]
    assert "likelihood_sigma" in model


def test_censored_likelihood_unsupported_distribution() -> None:
    cauchy = Prior("Cauchy")
    censored_cauchy = Censored(cauchy, lower=0)

    with pm.Model():
        mu = pm.Normal("mu")
        with pytest.raises(UnsupportedDistributionError):
            censored_cauchy.create_likelihood_variable(
                name="likelihood",
                mu=mu,
                observed=1,
            )


def test_censored_likelihood_already_has_mu() -> None:
    normal = Prior("Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal"))
    censored_normal = Censored(normal, lower=0)

    with pm.Model():
        mu = pm.Normal("mu")
        with pytest.raises(MuAlreadyExistsError):
            censored_normal.create_likelihood_variable(
                name="likelihood",
                mu=mu,
                observed=1,
            )


def test_censored_to_dict() -> None:
    normal = Prior("Normal", mu=0, sigma=1, dims="channel")
    censored_normal = Censored(normal, lower=0)

    data = censored_normal.to_dict()
    assert data == {
        "class": "Censored",
        "data": {"dist": normal.to_dict(), "lower": 0, "upper": float("inf")},
    }


def test_deserialize_censored() -> None:
    data = {
        "class": "Censored",
        "data": {
            "dist": {
                "dist": "Normal",
            },
            "lower": 0,
            "upper": float("inf"),
        },
    }

    instance = deserialize(data)
    assert isinstance(instance, Censored)
    assert isinstance(instance.distribution, Prior)
    assert instance.lower == 0
    assert instance.upper == float("inf")


class ArbitrarySerializable(Arbitrary):
    def to_dict(self):
        return {"dims": self.dims}


@pytest.fixture
def arbitrary_serialized_data() -> dict:
    return {"dims": ("channel",)}


def test_create_prior_with_arbitrary_serializable(arbitrary_serialized_data) -> None:
    dist = Prior(
        "Normal",
        mu=ArbitrarySerializable(dims=("channel",)),
        sigma=1,
        dims=("channel", "geo"),
    )

    assert dist.to_dict() == {
        "dist": "Normal",
        "kwargs": {
            "mu": arbitrary_serialized_data,
            "sigma": 1,
        },
        "dims": ("channel", "geo"),
    }


@pytest.fixture
def register_arbitrary_deserialization():
    register_deserialization(
        lambda data: isinstance(data, dict) and data.keys() == {"dims"},
        lambda data: ArbitrarySerializable(**data),
    )

    yield

    DESERIALIZERS.pop()


def test_deserialize_arbitrary_within_prior(
    arbitrary_serialized_data,
    register_arbitrary_deserialization,
) -> None:
    data = {
        "dist": "Normal",
        "kwargs": {
            "mu": arbitrary_serialized_data,
            "sigma": 1,
        },
        "dims": ("channel", "geo"),
    }

    dist = deserialize(data)
    assert isinstance(dist["mu"], ArbitrarySerializable)
    assert dist["mu"].dims == ("channel",)


def test_censored_with_tensor_variable() -> None:
    normal = Prior("Normal", dims="channel")
    lower = pt.as_tensor_variable([0, 1, 2])
    censored_normal = Censored(normal, lower=lower)

    assert censored_normal.to_dict() == {
        "class": "Censored",
        "data": {
            "dist": normal.to_dict(),
            "lower": [0, 1, 2],
            "upper": float("inf"),
        },
    }


def test_censored_dims_setter() -> None:
    normal = Prior("Normal", dims="channel")
    censored_normal = Censored(normal, lower=0)
    censored_normal.dims = "date"
    assert normal.dims == ("date",)


class ModelData(NamedTuple):
    mu: float
    observed: list[float]


@pytest.fixture(scope="session")
def model_data() -> ModelData:
    return ModelData(mu=0, observed=[0, 1, 2, 3, 4])


@pytest.fixture(scope="session")
def normal_model_with_censored_API(model_data) -> pm.Model:
    coords = {"idx": range(len(model_data.observed))}
    with pm.Model(coords=coords) as model:
        sigma = Prior("HalfNormal")
        normal = Prior("Normal", sigma=sigma, dims="idx")
        Censored(normal, lower=0).create_likelihood_variable(
            "censored_normal",
            mu=model_data.mu,
            observed=model_data.observed,
        )

    return model


@pytest.fixture(scope="session")
def normal_model_with_censored_logp(normal_model_with_censored_API):
    return normal_model_with_censored_API.compile_logp()


@pytest.fixture(scope="session")
def expected_normal_model(model_data) -> pm.Model:
    n_points = len(model_data.observed)
    with pm.Model() as expected_model:
        sigma = pm.HalfNormal("censored_normal_sigma")
        normal = pm.Normal.dist(mu=model_data.mu, sigma=sigma, shape=n_points)
        pm.Censored(
            "censored_normal",
            normal,
            lower=0,
            upper=np.inf,
            observed=model_data.observed,
        )

    return expected_model


@pytest.fixture(scope="session")
def expected_normal_model_logp(expected_normal_model):
    return expected_normal_model.compile_logp()


@pytest.mark.parametrize("sigma_log__", [-10, -5, -2.5, 0, 2.5, 5, 10])
def test_censored_normal_logp(
    sigma_log__,
    normal_model_with_censored_logp,
    expected_normal_model_logp,
) -> None:
    points = {"censored_normal_sigma_log__": sigma_log__}
    normal_model_logp = normal_model_with_censored_logp(points)
    expected_model_logp = expected_normal_model_logp(points)
    np.testing.assert_allclose(normal_model_logp, expected_model_logp)


@pytest.mark.parametrize(
    "mu",
    [
        0,
        np.arange(10),
    ],
    ids=["scalar", "vector"],
)
def test_censored_logp(mu) -> None:
    n_points = 10
    observed = np.zeros(n_points)
    coords = {"idx": range(n_points)}
    with pm.Model(coords=coords) as model:
        normal = Prior("Normal", dims="idx")
        Censored(normal, lower=0).create_likelihood_variable(
            "censored_normal",
            observed=observed,
            mu=mu,
        )
    logp = model.compile_logp()

    with pm.Model() as expected_model:
        pm.Censored(
            "censored_normal",
            pm.Normal.dist(mu=mu, sigma=1, shape=n_points),
            lower=0,
            upper=np.inf,
            observed=observed,
        )
    expected_logp = expected_model.compile_logp()

    point = {}
    np.testing.assert_allclose(logp(point), expected_logp(point))


def test_scaled_initializes_correctly() -> None:
    """Test that the Scaled class initializes correctly."""
    normal = Prior("Normal", mu=0, sigma=1)
    scaled = Scaled(normal, factor=2.0)

    assert scaled.dist == normal
    assert scaled.factor == 2.0


def test_scaled_dims_property() -> None:
    """Test that the dims property returns the dimensions of the underlying distribution."""
    normal = Prior("Normal", mu=0, sigma=1, dims="channel")
    scaled = Scaled(normal, factor=2.0)

    assert scaled.dims == ("channel",)

    # Test with multiple dimensions
    normal.dims = ("channel", "geo")
    assert scaled.dims == ("channel", "geo")


def test_scaled_create_variable() -> None:
    """Test that the create_variable method properly scales the variable."""
    normal = Prior("Normal", mu=0, sigma=1)
    scaled = Scaled(normal, factor=2.0)

    with pm.Model() as model:
        scaled_var = scaled.create_variable("scaled_var")

    # Check that both the scaled and unscaled variables exist
    assert "scaled_var" in model
    assert "scaled_var_unscaled" in model

    # The deterministic node should be the scaled variable
    assert model["scaled_var"] == scaled_var


def test_scaled_creates_correct_dimensions() -> None:
    """Test that the scaled variable has the correct dimensions."""
    normal = Prior("Normal", dims="channel")
    scaled = Scaled(normal, factor=2.0)

    coords = {"channel": ["A", "B", "C"]}
    with pm.Model(coords=coords):
        scaled_var = scaled.create_variable("scaled_var")

    # Check that the scaled variable has the correct dimensions
    assert fast_eval(scaled_var).shape == (3,)


def test_scaled_applies_factor() -> None:
    """Test that the scaling factor is correctly applied."""
    normal = Prior("Normal", mu=0, sigma=1)
    factor = 3.5
    scaled = Scaled(normal, factor=factor)

    # Sample from prior to verify scaling
    prior = sample_prior(scaled, samples=10, name="scaled_var")
    df_prior = prior.to_dataframe()

    # Check that scaled values are original values times the factor
    unscaled_values = df_prior["scaled_var_unscaled"].to_numpy()
    scaled_values = df_prior["scaled_var"].to_numpy()
    np.testing.assert_allclose(scaled_values, unscaled_values * factor)


def test_scaled_with_tensor_factor() -> None:
    """Test that the Scaled class works with a tensor factor."""
    normal = Prior("Normal", mu=0, sigma=1)
    factor = pt.as_tensor_variable(2.5)
    scaled = Scaled(normal, factor=factor)

    # Sample from prior to verify tensor scaling
    prior = sample_prior(scaled, samples=10, name="scaled_var")
    df_prior = prior.to_dataframe()

    # Check that scaled values are original values times the factor
    unscaled_values = df_prior["scaled_var_unscaled"].to_numpy()
    scaled_values = df_prior["scaled_var"].to_numpy()
    np.testing.assert_allclose(scaled_values, unscaled_values * 2.5)


def test_scaled_with_hierarchical_prior() -> None:
    """Test that the Scaled class works with hierarchical priors."""
    normal = Prior(
        "Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal"), dims="channel"
    )
    scaled = Scaled(normal, factor=2.0)

    coords = {"channel": ["A", "B", "C"]}
    with pm.Model(coords=coords) as model:
        scaled.create_variable("scaled_var")

    # Check that all necessary variables were created
    assert "scaled_var" in model
    assert "scaled_var_unscaled" in model
    assert "scaled_var_unscaled_mu" in model
    assert "scaled_var_unscaled_sigma" in model


def test_scaled_sample_prior() -> None:
    """Test that sample_prior works with the Scaled class."""
    normal = Prior("Normal", dims="channel")
    scaled = Scaled(normal, factor=2.0)

    coords = {"channel": ["A", "B", "C"]}
    prior = sample_prior(scaled, coords=coords, draws=25, name="scaled_var")

    assert isinstance(prior, xr.Dataset)
    assert prior.sizes == {"chain": 1, "draw": 25, "channel": 3}
    assert "scaled_var" in prior
    assert "scaled_var_unscaled" in prior


def test_prior_list_dims() -> None:
    dist = Prior("Normal", dims=["channel", "geo"])
    assert dist.dims == ("channel", "geo")


@pytest.mark.parametrize(
    "data, expected",
    [
        pytest.param(
            {
                "distribution": "Laplace",
                "mu": 1,
                "b": 2,
                "dims": ("x", "y"),
                "transform": "sigmoid",
            },
            Prior("Laplace", mu=1, b=2, dims=("x", "y"), transform="sigmoid"),
            id="Prior",
        ),
        pytest.param(
            {"distribution": "Normal", "mu": {"distribution": "Normal"}},
            Prior("Normal", mu=Prior("Normal")),
            id="Prior with nested distribution",
        ),
        pytest.param(
            {
                "class": "Censored",
                "data": {
                    "dist": {"distribution": "Normal"},
                    "lower": 0,
                    "upper": 10,
                },
            },
            Censored(Prior("Normal"), lower=0, upper=10),
            id="Censored with alternative",
        ),
    ],
)
def test_alternative_prior_deserialize(data, expected) -> None:
    assert deserialize(data) == expected
