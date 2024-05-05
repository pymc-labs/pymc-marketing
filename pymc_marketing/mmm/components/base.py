import warnings
from inspect import signature
from typing import Any

from pytensor import tensor as pt

from pymc_marketing.mmm.utils import _get_distribution_from_dict


class ParameterPriorException(Exception):
    """Error when the functions and specified priors don't match up."""

    def __init__(self, priors: set[str], parameters: set[str]) -> None:
        self.priors = priors
        self.parameters = parameters

        msg = "The function parameters and priors don't line up."

        if self.priors:
            msg = f"{msg} Missing default prior: {self.priors}."

        if self.parameters:
            msg = f"{msg} Missing function parameter: {self.parameters}."

        super().__init__(msg)


RESERVED_DATA_PARAMETER_NAMES = {"x", "data"}


class MissingDataParameter(Exception):
    """Error if the function doesn't have a data parameter."""

    def __init__(self) -> None:
        msg = (
            f"The function must have a data parameter."
            " The first parameter is assumed to be the data"
            f" with name being one of: {RESERVED_DATA_PARAMETER_NAMES}"
        )

        super().__init__(msg)


class Transformation:
    """Base class for adstock and saturation functions.

    The subclasses will need to implement the following attributes:

    - function: The function that will be applied to the data.
    - prefix: The prefix for the variables that will be created.
    - default_priors: The default priors for the parameters of the function.

    In order to make a new saturation or adstock function, this class would not be used but rather
    **SaturationTransformation** and **AdstockTransformation** not be used. Instead, the subclasses would be used.

    """

    prefix: str
    default_priors: dict[str, Any]
    function: Any

    def __init__(self, priors: dict | None = None) -> None:
        self._checks()

        priors = priors or {}
        self.function_priors = {**self.default_priors, **priors}

    def update_priors(self, priors: dict[str, Any]) -> None:
        """

        model_config = {
            "saturation_lam": {"dist": "Gamma", "kwargs": {"alpha": 3, "beta": 1}},
            "lam": {"dist": "Gamma", "kwargs": {"alpha": 3, "beta": 1}},
        }

        class MMM:
            def __init__(self, model_config):
                adstock.update_priors(model_config)

        """
        new_priors = {
            parameter_name: priors[variable_name]
            for parameter_name, variable_name in self.variable_mapping.items()
        }
        if not new_priors:
            available_priors = list(self.variable_mapping.values())
            warnings.warn(
                f"No priors were updated. Available parameters are {available_priors}",
                UserWarning,
                stacklevel=2,
            )

        self.function_priors.update(new_priors)

    @property
    def model_config(self) -> dict[str, Any]:
        return {
            variable_name: self.function_priors[parameter_name]
            for parameter_name, variable_name in self.variable_mapping.items()
        }

    def _checks(self) -> None:
        self._has_all_attributes()
        self._function_works_on_instances()
        self._has_defaults_for_all_arguments()

    def _has_all_attributes(self) -> None:
        if not hasattr(self, "prefix"):
            raise NotImplementedError("prefix must be implemented in the subclass")

        if not hasattr(self, "default_priors"):
            raise NotImplementedError(
                "default_priors must be implemented in the subclass"
            )

        if not hasattr(self, "function"):
            raise NotImplementedError("function must be implemented in the subclass")

    def _has_defaults_for_all_arguments(self) -> None:
        function_signature = signature(self.function)

        # Remove the first one as assumed to be the data
        parameters_that_need_priors = set(
            list(function_signature.parameters.keys())[1:]
        )
        parameters_with_priors = set(self.default_priors.keys())

        missing_priors = parameters_that_need_priors - parameters_with_priors
        missing_parameters = parameters_with_priors - parameters_that_need_priors

        if missing_priors or missing_parameters:
            raise ParameterPriorException(missing_priors, missing_parameters)

    def _function_works_on_instances(self) -> None:
        class_function = self.__class__.function
        function_parameters = list(signature(class_function).parameters)

        is_method = function_parameters[0] == "self"
        data_parameter_idx = 1 if is_method else 0

        has_data_parameter = (
            function_parameters[data_parameter_idx] in RESERVED_DATA_PARAMETER_NAMES
        )
        if not has_data_parameter:
            raise MissingDataParameter()

        if is_method:
            return

        self.function = class_function

    @property
    def variable_mapping(self) -> dict[str, str]:
        return {
            parameter: f"{self.prefix}_{parameter}"
            for parameter in self.default_priors.keys()
        }

    def _create_distributions(self, dim_name: str) -> dict[str, pt.TensorVariable]:
        distributions: dict[str, pt.TensorVariable] = {}
        for parameter_name, variable_name in self.variable_mapping.items():
            parameter_prior = self.function_priors[parameter_name]

            distribution = _get_distribution_from_dict(
                dist=parameter_prior,
            )

            distributions[parameter_name] = distribution(
                name=variable_name,
                dims=dim_name,
                **parameter_prior["kwargs"],
            )

        return distributions

    def apply(self, x):
        """Called within a model context."""
        kwargs = self._create_distributions(dim_name="channel")
        return self.function(x, **kwargs)
