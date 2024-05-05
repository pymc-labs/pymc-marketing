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
        NewTransformation()
    except Exception as e:
        pytest.fail(f"Error: {e}")


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
