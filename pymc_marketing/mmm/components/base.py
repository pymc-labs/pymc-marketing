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
"""Base class for adstock and saturation functions used in MMM.

Use the subclasses directly for custom transformations:

- Adstock Transformations: :class:`pymc_marketing.mmm.components.adstock.AdstockTransformation`
- Saturation Transformations: :class:`pymc_marketing.mmm.components.saturation.SaturationTransformation`

"""

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

    In order to make a new saturation or adstock function, use the specific subclasses:

    - :class:`pymc_marketing.mmm.components.saturation.SaturationTransformation`
    - :class:`pymc_marketing.mmm.components.adstock.AdstockTransformation`

    View the documentation for those classes for more information.

    Parameters
    ----------
    priors : dict, optional
        Dictionary with the priors for the parameters of the function. The keys should be the
        parameter names and the values should be dictionaries with the distribution and kwargs.
    prefix : str, optional
        The prefix for the variables that will be created. If not provided, it will use the prefix
        from the subclass.

    """

    prefix: str
    default_priors: dict[str, Any]
    function: Any

    def __init__(self, priors: dict | None = None, prefix: str | None = None) -> None:
        self._checks()
        priors = priors or {}
        self.function_priors = {**self.default_priors, **priors}
        self.prefix = prefix or self.prefix

    def update_priors(self, priors: dict[str, Any]) -> None:
        """Helper to update the priors for a function after initialization.

        Uses {prefix}_{parameter_name} as the key for the priors instead of the parameter name
        in order to be used in the larger MMM.

        Parameters
        ----------
        priors : dict
            Dictionary with the new priors for the parameters of the function.

        Examples
        --------
        Update the priors for a transformation after initialization.

        .. code-block:: python

            class MyTransformation(Transformation):
                prefix: str = "transformation"
                function = lambda x, lam: x * lam
                default_priors = {"lam": {"dist": "Gamma", "kwargs": {"alpha": 3, "beta": 1}}}

            transformation = MyTransformation()
            transformation.update_priors(
                {"transformation_lam": {"dist": "HalfNormal", "kwargs": {"sigma": 1}}}
            )

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
        """Mapping from variable name to prior for the model."""
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
        """Mapping from parameter name to variable name in the model."""
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

    def apply(self, x: pt.TensorLike, dim_name: str = "channel") -> pt.TensorVariable:
        """Called within a model context.

        Used internally of the MMM to apply the transformation to the data.

        Parameters
        ----------
        x : pt.TensorLike
            The data to be transformed.
        dim_name : str, optional
            The name of the dimension associated with the columns of the data.
            Defaults to "channel".

        Returns
        -------
        pt.TensorVariable
            The transformed data.


        Examples
        --------
        Call the function for custom use-case

        .. code-block:: python

            transformation = ...

            coords = {"channel": ["TV", "Radio", "Digital"]}
            with pm.Model(coords=coords):
                transformed_data = transformation.apply(data, dim_name="channel")

        """
        kwargs = self._create_distributions(dim_name=dim_name)
        return self.function(x, **kwargs)
