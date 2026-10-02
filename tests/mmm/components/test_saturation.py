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
import inspect
from inspect import signature

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from pydantic import ValidationError
from pymc_extras.prior import Prior
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

import pymc_marketing.mmm.components.saturation as saturation_module
from pymc_marketing.mmm.components.saturation import (
    LogisticSaturation,
    MichaelisMentenSaturation,
    RootSaturation,
    SaturationTransformation,
    TanhSaturationBaselined,
)
from pymc_marketing.serialization import serialization

ALL_SATURATION_CLASSES: list[type[SaturationTransformation]] = [
    cls
    for _, cls in inspect.getmembers(saturation_module, inspect.isclass)
    if issubclass(cls, SaturationTransformation) and cls is not SaturationTransformation
]


@pytest.fixture
def model() -> pm.Model:
    coords = {"channel": ["a", "b", "c"]}
    return pm.Model(coords=coords)


def saturation_functions():
    return [
        pytest.param(saturation_cls(), id=saturation_cls.__name__)
        for saturation_cls in ALL_SATURATION_CLASSES
    ]


@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
@pytest.mark.parametrize(
    "x, dims",
    [
        pytest.param(np.linspace(0, 1, 100), ("time",), id="vector"),
        pytest.param(np.ones((100, 3)), ("time", "channel"), id="matrix"),
    ],
)
def test_apply_method(
    model,
    saturation: SaturationTransformation,
    x,
    dims,
) -> None:
    x = as_xtensor(x, dims=dims)

    with model:
        y = saturation.apply(x)

    assert isinstance(y, XTensorVariable)
    assert y.eval().shape == x.type.shape


@pytest.mark.parametrize("saturation_cls", ALL_SATURATION_CLASSES)
def test_parameters_broadcast_against_x_by_dim_name(
    saturation_cls: type[SaturationTransformation],
) -> None:
    """Check that per-channel parameters pair with ``x`` by dim name, not position.

    The result on a channel-first and on a channel-last input must both match
    applying the function to each channel on its own. The date and channel
    lengths differ, so a positional broadcast either fails or pairs a
    parameter with the wrong axis.
    """
    saturation = saturation_cls()
    x_date_channel = np.array(
        [[0.0, 1.0, 2.0], [0.5, 3.0, 0.0], [2.0, 0.25, 1.0], [4.0, 2.0, 0.75]]
    )
    per_channel = np.array([0.3, 0.5, 0.7])
    params = {
        name: as_xtensor(per_channel, dims=("channel",))
        for name in saturation.default_priors
    }

    expected = np.stack(
        [
            saturation.function(
                as_xtensor(x_date_channel[:, i], dims=("date",)),
                **{name: per_channel[i] for name in saturation.default_priors},
            ).eval()
            for i in range(per_channel.size)
        ],
        axis=1,
    )
    for dims, x in [
        (("date", "channel"), x_date_channel),
        (("channel", "date"), x_date_channel.T),
    ]:
        y = saturation.function(as_xtensor(x, dims=dims), **params)
        np.testing.assert_allclose(
            y.transpose("date", "channel").eval(), expected, err_msg=str(dims)
        )


def test_root_saturation_logp_is_differentiable_at_zero_input() -> None:
    """RootSaturation gradients must stay finite at exactly-zero input.

    In an MMM the saturation input is a function of random variables (e.g.
    adstocked spend), and channels routinely have zero-spend periods. The
    derivative ``d/dx (x ** alpha) = alpha * x ** (alpha - 1)`` is infinite at
    ``x == 0`` for ``alpha < 1``, so the resulting NaN propagated into the
    log-probability gradient of every upstream parameter and broke NUTS. The
    transformation evaluates the power on a safe input (``ptx.where`` in
    xtensor) so that ``f(0) = 0`` exactly and the derivative is finite everywhere.
    """
    x = np.linspace(0.0, 1.0, 30)
    x[:5] = 0.0  # exact zero-spend periods
    rng = np.random.default_rng(0)
    y_obs = rng.normal(size=x.shape[0])
    with pm.Model(coords={"time": range(x.shape[0])}) as model:
        # The input depends on a free RV, as it does after adstock in an MMM.
        scale = pm.HalfNormal("scale", 1)
        x_tensor = as_xtensor(x, dims=("time",)) * scale
        mu = RootSaturation().apply(x_tensor)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("obs", mu=mu.values, sigma=sigma, observed=y_obs, dims=("time",))

    dlogp = model.compile_dlogp()
    grad = dlogp(model.initial_point())
    assert np.all(np.isfinite(grad))


def test_tanh_saturation_baselined_default_priors_have_a_finite_logp() -> None:
    """The default priors must not put a pole at the sampler's starting point.

    PyMC initialises each parameter at its distribution's moment, so a default
    prior for a parameter that feeds a pole of the transformation can land
    exactly on it. The overspend fraction ``r`` was the offending case: the
    pre-fix default ``r ~ HalfNormal(1)`` started at ``r = 1``, where
    ``arctanh(r)`` is infinite, so the logp and its gradient were NaN for the
    zero-spend periods that occur in real MMM data (#3047).
    """
    x = np.linspace(0.0, 1.0, 10)
    x[:2] = 0.0  # exact zero-spend periods
    with pm.Model(coords={"time": range(x.shape[0])}) as model:
        x_tensor = as_xtensor(x, dims=("time",))
        mu = TanhSaturationBaselined().apply(x_tensor)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal(
            "obs", mu=mu.values, sigma=sigma, observed=np.zeros_like(x), dims=("time",)
        )

    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    assert np.all(np.isfinite(model.compile_dlogp()(point)))


@pytest.mark.parametrize(
    "saturation_cls", ALL_SATURATION_CLASSES, ids=lambda c: c.__name__
)
def test_all_saturation_default_priors_have_a_finite_logp_at_the_initial_point(
    saturation_cls: type[SaturationTransformation],
) -> None:
    """Every saturation's default priors must give a finite logp and gradient.

    PyMC starts each parameter at its distribution's support point.
    If it lands on a pole of the transformation, the logp or gradient is non-finite at ``initial_point()``.
    ``TanhSaturationBaselined`` was the reported case (#3047).
    Its old ``r ~ HalfNormal(1)`` started at ``r = 1``, where ``arctanh(r)`` diverges.
    On positive input that breaks only the gradient, which is what this test catches.
    The zero-spend logp case is covered by ``test_tanh_saturation_baselined_default_priors_have_a_finite_logp``.
    Covering every class catches a future singular default anywhere.
    """
    x = np.linspace(0.1, 1.0, 10)
    saturation = saturation_cls()
    with pm.Model(coords={"time": range(x.shape[0])}) as model:
        x_tensor = as_xtensor(x, dims=("time",))
        mu = saturation.apply(x_tensor)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal(
            "obs", mu=mu.values, sigma=sigma, observed=np.zeros_like(x), dims=("time",)
        )

    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    assert np.all(np.isfinite(model.compile_dlogp()(point)))


def test_tanh_saturation_baselined_default_r_prior() -> None:
    """The default prior for ``r`` is ``Beta(2, 3)``."""
    assert TanhSaturationBaselined().default_priors["r"] == Prior(
        "Beta", alpha=2, beta=3
    )


def test_tanh_saturation_baselined_default_r_is_in_unit_interval() -> None:
    """All default ``r`` draws must be strictly inside (0, 1).

    The transformation evaluates ``arctanh(r)``, so any mass at ``r >= 1`` is
    invalid. This rejects a fix such as ``HalfNormal(sigma < 1)``, which would
    start the sampler at a finite point but still allow ``r > 1`` (#3047).
    """
    prior = TanhSaturationBaselined().sample_prior(draws=5000, random_seed=0)

    r = prior["saturation_r"].values
    assert np.all((r > 0) & (r < 1))


def test_tanh_saturation_baselined_saved_priors_are_preserved() -> None:
    """A saved payload's priors win over the class defaults.

    Saved models store ``function_priors`` in full (see
    :meth:`pymc_marketing.mmm.components.base.Transformation.to_dict`), so a
    model saved before the default changed carries ``r ~ HalfNormal(1)``. It
    must load with that prior instead of adopting the new default (#3047).
    """
    payload = {
        "__type__": (
            "pymc_marketing.mmm.components.saturation.TanhSaturationBaselined"
        ),
        "prefix": "saturation",
        "priors": {
            "x0": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
            "gain": {"dist": "HalfNormal", "kwargs": {"sigma": 3}},
            "r": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
            "beta": {"dist": "HalfNormal", "kwargs": {"sigma": 4}},
        },
    }

    restored = serialization.deserialize(payload)

    assert restored.function_priors["r"] == Prior("HalfNormal", sigma=1)
    assert restored.function_priors["x0"] == Prior("HalfNormal", sigma=2)
    assert restored.function_priors["gain"] == Prior("HalfNormal", sigma=3)
    assert restored.function_priors["beta"] == Prior("HalfNormal", sigma=4)


@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
def test_default_prefix(saturation: SaturationTransformation) -> None:
    assert saturation.prefix == "saturation"
    for value in saturation.variable_mapping.values():
        assert value.startswith("saturation_")


@pytest.mark.parametrize(
    "saturation",
    saturation_functions(),
)
def test_support_for_lift_test_integrations(
    saturation: SaturationTransformation,
) -> None:
    function_parameters = signature(saturation.function).parameters

    for key in saturation.variable_mapping.keys():
        assert isinstance(key, str)
        assert key in function_parameters

    assert len(saturation.variable_mapping) == len(function_parameters) - 2


@pytest.mark.parametrize("saturation", saturation_functions())
def test_sample_curve(saturation: SaturationTransformation) -> None:
    prior = saturation.sample_prior()
    assert isinstance(prior, xr.Dataset)
    curve = saturation.sample_curve(prior)
    assert isinstance(curve, xr.DataArray)
    assert curve.name == "saturation"
    assert curve.shape == (1, 500, 100)


@pytest.mark.parametrize("saturation", saturation_functions())
@pytest.mark.parametrize("num_points", [50, 200, 1000])
def test_sample_curve_num_points(
    saturation: SaturationTransformation,
    num_points,
) -> None:
    """Test that num_points parameter controls the number of points in the curve."""
    prior = saturation.sample_prior()
    curve = saturation.sample_curve(prior, num_points=num_points)
    assert isinstance(curve, xr.DataArray)
    assert curve.name == "saturation"
    assert curve.shape == (1, 500, num_points)


@pytest.mark.parametrize(
    argnames="num_points", argvalues=[0, -1], ids=["zero", "negative"]
)
def test_sample_curve_with_bad_num_points(num_points) -> None:
    """Test that invalid num_points raises ValidationError."""
    saturation = LogisticSaturation()
    prior = saturation.sample_prior()

    with pytest.raises(ValidationError):
        saturation.sample_curve(prior, num_points=num_points)


def create_mock_parameters(
    coords: dict[str, list],
    variable_dim_mapping: dict[str, tuple[str]],
) -> xr.Dataset:
    dim_sizes = {coord: len(values) for coord, values in coords.items()}
    return xr.Dataset(
        {
            name: xr.DataArray(
                np.ones(tuple(dim_sizes[coord] for coord in dims)),
                dims=dims,
                coords={coord: coords[coord] for coord in dims},
            )
            for name, dims in variable_dim_mapping.items()
        }
    )


@pytest.fixture
def mock_menten_parameters() -> xr.Dataset:
    coords = {
        "chain": np.arange(1),
        "draw": np.arange(500),
    }

    variable_dim_mapping = {
        "saturation_alpha": ("chain", "draw"),
        "saturation_lam": ("chain", "draw"),
        "another_random_variable": ("chain", "draw"),
    }

    return create_mock_parameters(coords, variable_dim_mapping)


def test_sample_curve_additional_dataset_variables(mock_menten_parameters) -> None:
    """Case when the parameter dataset has additional variables."""
    saturation = MichaelisMentenSaturation()

    try:
        curve = saturation.sample_curve(parameters=mock_menten_parameters)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

    assert isinstance(curve, xr.DataArray)
    assert curve.name == "saturation"


@pytest.fixture
def mock_menten_parameters_with_additional_dim() -> xr.Dataset:
    coords = {
        "chain": np.arange(1),
        "draw": np.arange(500),
        "channel": ["C1", "C2", "C3"],
        "random_dim": ["R1", "R2"],
    }
    variable_dim_mapping = {
        "saturation_alpha": ("chain", "draw", "channel"),
        "saturation_lam": ("chain", "draw", "channel"),
        "another_random_variable": ("chain", "draw", "channel", "random_dim"),
    }

    return create_mock_parameters(coords, variable_dim_mapping)


def test_sample_curve_with_additional_dims(
    mock_menten_parameters_with_additional_dim,
) -> None:
    dummy_distribution = Prior("HalfNormal", dims="channel")
    priors = {
        "alpha": dummy_distribution,
        "lam": dummy_distribution,
    }
    saturation = MichaelisMentenSaturation(priors=priors)

    curve = saturation.sample_curve(
        parameters=mock_menten_parameters_with_additional_dim
    )

    assert curve.coords["channel"].to_numpy().tolist() == ["C1", "C2", "C3"]
    assert "random_dim" not in curve.coords


@pytest.mark.parametrize(
    argnames="max_value", argvalues=[0, -1], ids=["zero", "negative"]
)
def test_sample_curve_with_bad_max_value(max_value) -> None:
    dummy_distribution = Prior("HalfNormal", dims="channel")
    priors = {
        "alpha": dummy_distribution,
        "lam": dummy_distribution,
    }
    saturation = MichaelisMentenSaturation(priors=priors)

    with pytest.raises(ValidationError):
        saturation.sample_curve(
            parameters=mock_menten_parameters_with_additional_dim, max_value=max_value
        )


class TestSaturationRoundtrips:
    """Every SaturationTransformation subclass round-trips with all params."""

    @pytest.mark.parametrize(
        "sat_cls", ALL_SATURATION_CLASSES, ids=lambda c: c.__name__
    )
    def test_roundtrip_all_parameters(self, sat_cls):
        custom_priors = {
            name: Prior("HalfNormal", sigma=0.5) for name in sat_cls.default_priors
        }
        kwargs: dict = {
            "prefix": "custom_sat",
            "priors": custom_priors,
        }

        original = sat_cls(**kwargs)
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is sat_cls
        assert restored.prefix == "custom_sat"
        for prior_name, prior in custom_priors.items():
            assert restored.function_priors[prior_name] == prior
        assert restored == original

    @pytest.mark.parametrize("lam", [2.0, [2.0, 3.0]], ids=["float", "list"])
    def test_roundtrip_constant_prior(self, lam) -> None:
        """A constant parameter survives serialization next to a Prior (#1613)."""
        original = LogisticSaturation(
            priors={"lam": lam, "beta": Prior("HalfNormal", sigma=1)}
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert restored == original
        assert restored.function_priors["beta"] == Prior("HalfNormal", sigma=1)
        np.testing.assert_allclose(restored.function_priors["lam"], lam)


@pytest.mark.parametrize(
    "type_key",
    [
        "pymc_marketing.mmm.components.saturation.LogisticSaturation",
        "pymc_marketing.mmm.components.saturation.TanhSaturation",
        "pymc_marketing.mmm.components.saturation.TanhSaturationBaselined",
        "pymc_marketing.mmm.components.saturation.HillSaturation",
        "pymc_marketing.mmm.components.saturation.HillSaturationSigmoid",
        "pymc_marketing.mmm.components.saturation.MichaelisMentenSaturation",
        "pymc_marketing.mmm.components.saturation.RootSaturation",
        "pymc_marketing.mmm.components.saturation.InverseScaledLogisticSaturation",
        "pymc_marketing.mmm.components.saturation.LogSaturation",
        "pymc_marketing.mmm.components.saturation.NoSaturation",
    ],
    ids=lambda s: s.rsplit(".", 1)[-1],
)
def test_type_registered(type_key):
    assert type_key in serialization._registry, f"{type_key} not registered"
