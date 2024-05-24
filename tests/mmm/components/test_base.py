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
import numpy as np
import pytest

from pymc_marketing.mmm.components.base import (
    MissingDataParameter,
    ParameterPriorException,
    Transformation,
)


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
        function = function_without_reserved_parameter
        default_priors = {"a": "dummy"}

    with pytest.raises(MissingDataParameter):
        NewTransformation()


def test_new_transformation_missing_priors() -> None:
    def method_like_function(self, x, a, b):
        return x + a + b

    class NewTransformation(Transformation):
        prefix = "new"
        function = method_like_function
        default_priors = {"a": "dummy"}

    with pytest.raises(ParameterPriorException, match="Missing default prior: {'b'}"):
        NewTransformation()


def test_new_transformation_missing_parameter() -> None:
    class NewTransformation(Transformation):
        prefix = "new"
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

        def function(self, x, a, b):
            return a * b * x

        default_priors = {
            "a": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
            "b": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        }

    return NewTransformation


@pytest.fixture
def new_transformation(new_transformation_class) -> Transformation:
    return new_transformation_class()


def test_new_transformation_function_priors(new_transformation) -> None:
    assert new_transformation.function_priors == {
        "a": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "b": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


def test_new_transformation_update_priors(new_transformation_class) -> None:
    new_prior = {"a": {"dist": "HalfNormal", "kwargs": {"sigma": 2}}}
    new_transformation = new_transformation_class(priors=new_prior)
    assert new_transformation.function_priors == {
        "a": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
        "b": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


def test_new_transformation_variable_mapping(new_transformation) -> None:
    assert new_transformation.variable_mapping == {"a": "new_a", "b": "new_b"}


def test_new_transformation_access_function(new_transformation) -> None:
    x = np.array([1, 2, 3])
    expected = np.array([6, 12, 18])
    np.testing.assert_allclose(new_transformation.function(x, 2, 3), expected)


def test_new_transformation_apply_outside_model(new_transformation) -> None:
    with pytest.raises(TypeError, match="on context stack"):
        new_transformation.apply(1)
