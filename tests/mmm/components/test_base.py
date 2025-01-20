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
import pytensor.tensor as pt
import pytest
import xarray as xr
from pytensor.tensor import TensorVariable

from pymc_marketing.mmm.components.base import (
    DuplicatedTransformationError,
    MissingDataParameter,
    ParameterPriorException,
    Transformation,
    create_registration_meta,
)
from pymc_marketing.mmm.components.saturation import TanhSaturation
from pymc_marketing.prior import Prior, VariableFactory


def test_new_transformation_missing_prefix() -> None:
    class NewTransformation(Transformation):
        pass

    with pytest.raises(NotImplementedError, match="prefix must be implemented"):
        NewTransformation()


def test_new_transformation_missing_default_priors() -> None:
    class NewTransformation(Transformation):
        prefix = "new"

    with pytest.raises(NotImplementedError, match="default_priors must be implemented"):
        NewTransformation()


def test_new_transformation_missing_function() -> None:
    class NewTransformation(Transformation):
        prefix = "new"
        default_priors = {}

    with pytest.raises(NotImplementedError, match="function must be implemented"):
        NewTransformation()


def test_new_transformation_missing_lookup_name() -> None:
    class NewTransformation(Transformation):
        prefix = "new"
        default_priors = {}
        function = lambda x: x  # noqa: E731

    with pytest.raises(NotImplementedError, match="lookup_name must be implemented"):
        NewTransformation()


lambda_function = lambda x, a: x + a  # noqa: E731


def function(x, a):
    return x + a


def method_like_function(self, x, a):
    return x + a


@pytest.mark.parametrize(
    "function",
    [
        lambda_function,
        function,
        method_like_function,
        staticmethod(function),
    ],
)
def test_new_transformation_function_works_on_instances(function) -> None:
    class NewTransformation(Transformation):
        lookup_name: str = "new_transformation"
        prefix = "new"
        default_priors = {"a": "dummy"}
        function = function

    try:
        new_transformation = NewTransformation()
    except Exception as e:
        pytest.fail(f"Error: {e}")

    x = np.array([1, 2, 3])
    expected = np.array([2, 3, 4])
    np.testing.assert_allclose(new_transformation.function(x, 1), expected)


def test_new_transformation_missing_data_parameter() -> None:
    def function_without_reserved_parameter(spends, a):
        return spends + a

    class NewTransformation(Transformation):
        prefix = "new"
        lookup_name: str = "new_transformation"
        function = function_without_reserved_parameter
        default_priors = {"a": "dummy"}

    with pytest.raises(MissingDataParameter):
        NewTransformation()


def test_new_transformation_missing_priors() -> None:
    def method_like_function(self, x, a, b):
        return x + a + b

    class NewTransformation(Transformation):
        prefix = "new"
        lookup_name: str = "new_transformation"
        function = method_like_function
        default_priors = {"a": "dummy"}

    with pytest.raises(ParameterPriorException, match="Missing default prior: {'b'}"):
        NewTransformation()


def test_new_transformation_missing_parameter() -> None:
    class NewTransformation(Transformation):
        prefix = "new"
        lookup_name: str = "new_transformation"
        function = method_like_function
        default_priors = {"b": "dummy"}

    with pytest.raises(
        ParameterPriorException, match="Missing function parameter: {'b'}"
    ):
        NewTransformation()


@pytest.fixture
def new_transformation_class() -> type[Transformation]:
    class NewTransformation(Transformation):
        prefix = "new"
        lookup_name: str = "new_transformation"

        def function(self, x, a, b):
            return a * b * x

        default_priors = {
            "a": Prior("HalfNormal", sigma=1),
            "b": Prior("HalfNormal", sigma=1),
        }

    return NewTransformation


@pytest.fixture
def new_transformation(new_transformation_class) -> Transformation:
    return new_transformation_class()


def test_new_transformation_function_priors(new_transformation) -> None:
    assert new_transformation.function_priors == {
        "a": Prior("HalfNormal", sigma=1),
        "b": Prior("HalfNormal", sigma=1),
    }


def test_new_transformation_priors_at_init(new_transformation_class) -> None:
    new_prior = {"a": {"dist": "HalfNormal", "kwargs": {"sigma": 2}}}
    with pytest.warns(DeprecationWarning, match="a is automatically converted"):
        new_transformation = new_transformation_class(priors=new_prior)
    assert new_transformation.function_priors == {
        "a": Prior("HalfNormal", sigma=2),
        "b": Prior("HalfNormal", sigma=1),
    }


def test_new_transformation_variable_mapping(new_transformation) -> None:
    assert new_transformation.variable_mapping == {"a": "new_a", "b": "new_b"}


def test_apply(new_transformation):
    x = np.array([1, 2, 3])
    expected = np.array([6, 12, 18])
    with pm.Model() as generative_model:
        pm.Deterministic("y", new_transformation.apply(x))

    fixed_model = pm.do(generative_model, {"new_a": 2, "new_b": 3})
    np.testing.assert_allclose(fixed_model["y"].eval(), expected)


def test_new_transformation_access_function(new_transformation) -> None:
    x = np.array([1, 2, 3])
    expected = np.array([6, 12, 18])
    np.testing.assert_allclose(new_transformation.function(x, 2, 3), expected)


def test_new_transformation_apply_outside_model(new_transformation) -> None:
    with pytest.raises(TypeError, match="on context stack"):
        new_transformation.apply(1)


def test_model_config(new_transformation) -> None:
    assert new_transformation.model_config == {
        "new_a": Prior("HalfNormal", sigma=1),
        "new_b": Prior("HalfNormal", sigma=1),
    }


def test_new_transform_update_priors(new_transformation) -> None:
    new_transformation.update_priors(
        {"new_a": Prior("HalfNormal", sigma=2)},
    )

    assert new_transformation.function_priors == {
        "a": Prior("HalfNormal", sigma=2),
        "b": Prior("HalfNormal", sigma=1),
    }


def test_new_transformation_warning_no_priors_updated(new_transformation) -> None:
    with pytest.warns(UserWarning, match="No priors were updated"):
        new_transformation.update_priors({"new_c": Prior("HalfNormal")})


def test_new_transformation_sample_prior(new_transformation) -> None:
    prior = new_transformation.sample_prior()

    assert isinstance(prior, xr.Dataset)
    assert dict(prior.coords.sizes) == {
        "chain": 1,
        "draw": 500,
    }

    assert set(prior.keys()) == {"new_a", "new_b"}


def create_curve(coords) -> xr.DataArray:
    size = [len(values) for values in coords.values()]
    dims = list(coords.keys())
    data = np.ones(size)
    return xr.DataArray(
        data,
        dims=dims,
        coords=coords,
        name="data",
    )


@pytest.mark.parametrize(
    "coords, expected_size",
    [
        ({"chain": np.arange(1), "draw": np.arange(250), "x": np.arange(10)}, 1),
        (
            {
                "chain": np.arange(1),
                "draw": np.arange(250),
                "x": np.arange(10),
                "channel": ["A", "B", "C"],
            },
            3,
        ),
    ],
)
def test_new_transformation_plot_curve(
    new_transformation, coords, expected_size
) -> None:
    curve = create_curve(coords)
    fig, axes = new_transformation.plot_curve(curve)

    assert isinstance(fig, plt.Figure)
    assert len(axes) == expected_size

    plt.close()


def test_change_instance_function_priors_has_no_impact_new_instance(
    new_transformation_class,
) -> None:
    """What happens in the MMM logic."""
    instance = new_transformation_class()

    for _, config in instance.function_priors.items():
        config.dims = "channel"

    new_instance = new_transformation_class()

    assert new_instance.function_priors == {
        "a": Prior("HalfNormal", sigma=1),
        "b": Prior("HalfNormal", sigma=1),
    }


def test_change_instance_function_priors_has_no_impact_on_class(
    new_transformation_class,
) -> None:
    instance = new_transformation_class()

    for _, config in instance.function_priors.items():
        config.dims = "channel"

    assert new_transformation_class.default_priors == {
        "a": Prior("HalfNormal", sigma=1),
        "b": Prior("HalfNormal", sigma=1),
    }


def test_equality() -> None:
    class NewClass(Transformation):
        prefix = "new"
        lookup_name: str = "new_transformation"
        default_priors = {
            "a": Prior("HalfNormal", sigma=1),
            "b": Prior("HalfNormal", sigma=1),
        }

        def function(self, x, a, b):
            return a * b * x

    class AnotherNewClass(Transformation):
        prefix = "new"
        lookup_name: str = "new_transformation"
        default_priors = {
            "a": Prior("HalfNormal", sigma=1),
            "b": Prior("HalfNormal", sigma=1),
        }

        def function(self, x, a, b):
            return a * b * x

    assert NewClass() == NewClass()
    assert NewClass() != AnotherNewClass()


def test_repr(new_transformation) -> None:
    assert repr(new_transformation) == (
        "NewTransformation("
        "prefix='new', "
        "priors={'a': Prior(\"HalfNormal\", sigma=1), 'b': Prior(\"HalfNormal\", sigma=1)}"
        ")"
    )


def test_support_for_non_prior(new_transformation_class) -> None:
    instance = new_transformation_class(
        priors={"a": 1, "b": 2},
    )

    assert instance.to_dict() == {
        "lookup_name": "new_transformation",
        "prefix": "new",
        "priors": {"a": 1, "b": 2},
    }


class StandardNormal:
    def __init__(self, dims: tuple[str, ...]) -> None:
        self.dims = dims

    def to_dict(self) -> dict:
        return {
            "type": "StandardNormal",
            "dims": self.dims,
        }

    def create_variable(self, name: str) -> TensorVariable:
        return pm.Normal(name, 0, 1, dims=self.dims)


@pytest.fixture
def new_transformation_with_custom(new_transformation_class):
    return new_transformation_class(
        priors={"a": StandardNormal(("channel",))},
    )


def test_support_customer_serialization(
    new_transformation_with_custom,
) -> None:
    assert new_transformation_with_custom.to_dict() == {
        "lookup_name": "new_transformation",
        "prefix": "new",
        "priors": {
            "a": {"type": "StandardNormal", "dims": ("channel",)},
            "b": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        },
    }


def test_support_customer_samples(
    new_transformation_with_custom,
) -> None:
    coords = {"channel": ["C1", "C2", "C3"]}
    priors = new_transformation_with_custom.sample_prior(coords=coords)

    assert priors.data_vars.keys() == {"new_a", "new_b"}
    assert priors["new_a"].sizes == {"chain": 1, "draw": 500, "channel": 3}


def test_serialization(new_transformation_class) -> None:
    instance = new_transformation_class(
        priors={
            "a": pt.as_tensor_variable([1, 2, 3]),
            "b": np.array([1, 2, 3]),
        }
    )

    assert instance.to_dict() == {
        "lookup_name": "new_transformation",
        "prefix": "new",
        "priors": {
            "a": [1, 2, 3],
            "b": [1, 2, 3],
        },
    }


def test_automatic_registration() -> None:
    subclasses = {}

    RegistrationMeta = create_registration_meta(subclasses)

    class BaseTransform:
        pass

    class Transform(BaseTransform, metaclass=RegistrationMeta):
        pass

    class NewTransform(Transform):
        lookup_name = "new"

    assert subclasses == {"new": NewTransform}

    class AnotherTransform(Transform):
        lookup_name = "another"

    assert subclasses == {"new": NewTransform, "another": AnotherTransform}

    with pytest.raises(DuplicatedTransformationError) as e:

        class _(Transform):
            lookup_name = "new"

    exception = e.value

    assert exception.lookup_name == "new"
    assert exception.name == "Transform"


def test_transform_sample_curve_with_variable_factory():
    class Example(VariableFactory):
        dims = ("dim_a",)

        def create_variable(self, name: str):
            with pm.Model(name=name):
                beta = pm.Normal("beta", dims="dim_b")
                c = pm.Normal("c", dims=("dim_a", "dim_b"))
                return pt.dot(c, beta)

    saturation = TanhSaturation(
        priors={
            "b": Example(),
            "c": Example(),
        },
        prefix="outlet_saturation",
    )

    with pm.Model(coords={"dim_a": range(5), "dim_b": range(3), "obs": range(10)}):
        prior = saturation.sample_prior()

    curve = saturation.sample_curve(prior, 10)
    assert curve.dims == ("chain", "draw", "x", "dim_a")
