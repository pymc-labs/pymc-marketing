from inspect import signature
from typing import Dict, Optional

import pymc as pm
from pytensor import tensor as pt

from pymc_marketing.mmm.transformers import (
    hill_saturation,
    logistic_saturation,
    michaelis_menten,
)
from pymc_marketing.mmm.utils import _get_distribution


def function_parameters(func) -> set[str]:
    return set(signature(func).parameters.keys())


class MissingSaturationParameters(Exception):
    pass


### SATURATION FUNCTIONS
class BaseSaturationFunction:
    """Core logic for checks for all saturation functions.

    Subclasses need to define three items:

    - saturation_function: Method representation of the saturation function
    - variable_mapping: The mapping from saturation function parameters to model variable names
    - default_saturation_config: The mapping from model variable names to prior distributions

    """

    def __init__(
        self, model: Optional[pm.Model] = None, model_config: Optional[Dict] = None
    ):
        self.model = model
        self.model_config = self._initialize_model_config(model_config)

    def _initialize_model_config(self, model_config: Optional[Dict]) -> Dict:
        REQUIRED_KEYS = list(self.default_saturation_config.keys())

        if model_config is not None and not all(
            key in model_config for key in REQUIRED_KEYS
        ):
            update_model_config = {**model_config, **self.default_saturation_config}
        elif model_config is None:
            update_model_config = self.default_saturation_config
        else:
            update_model_config = model_config
        return update_model_config

    @property
    def default_saturation_config(self) -> Dict:
        msg = (
            "Subclasses should implement this method."
            " This contains the model variable name and their distributions."
        )
        raise NotImplementedError(msg)

    @property
    def variable_mapping(self) -> dict[str, str]:
        """Mapping between the saturation function args and model variables."""
        msg = (
            "Subclasses should implement this mapping from parameters"
            " to model variable names"
        )
        raise NotImplementedError(msg)

    def saturation_function(self, x, **kwargs):
        """Saturation function to be used..."""

    def _checks(self) -> None:
        self._check_variable_mapping_has_all_saturation_function_parameters()

    def _check_variable_mapping_has_all_saturation_function_parameters(self) -> None:
        saturation_parameters = function_parameters(self.saturation_function)
        variable_mapping_keys = set(self.variable_mapping.keys())

        missing_parameters = (
            saturation_parameters - set(["self", "x"]) - variable_mapping_keys
        )
        if missing_parameters:
            msg = f"The saturation function has missing parameters {list(missing_parameters)}"
            raise MissingSaturationParameters(msg)

    def _create_distributions(self) -> dict[str, pt.TensorVariable]:
        distributions: dict[str, pt.TensorVariable] = {}
        for parameter_name, variable_name in self.variable_mapping.items():
            distribution = _get_distribution(
                dist=self.model_config[variable_name],
            )
            distributions[parameter_name] = distribution(
                name=variable_name,
                dims=("channel",),
                **self.model_config[variable_name]["kwargs"],
            )

        return distributions

    def apply(self, data):
        kwargs = self._create_distributions()
        return self.saturation_function(data, **kwargs)


class HillSaturationComponent(BaseSaturationFunction):
    """
    A class representing the Hill Saturation component of a marketing mix model.

    Parameters:
    -----------
    model : PyMC.Model
        The PyMC model object.
    model_config : dict
        A dictionary containing the configuration parameters for the model.

    Methods:
    --------
    apply(data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        Apply the Hill Saturation component to the given data.

    """

    @property
    def default_saturation_config(self) -> Dict:
        return {
            "saturation_sigma": {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            },
            "saturation_lambda": {
                "dist": "Gamma",
                "kwargs": {"mu": 1, "sigma": 2},
            },
            "saturation_beta": {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            },
        }

    @property
    def variable_mapping(self) -> Dict[str, str]:
        """The mapping from saturation function parameters to variables in a model."""
        return {
            "lam": "saturation_lambda",
            "beta": "saturation_beta",
            "sigma": "saturation_sigma",
        }

    def saturation_function(self, x, sigma, beta, lam):
        return hill_saturation(
            x=x,
            sigma=sigma,
            beta=beta,
            lam=lam,
        )


class MentenSaturationComponent(BaseSaturationFunction):
    """
    A class representing the Menten Saturation component of a marketing mix model.

    Parameters:
    -----------
    model : PyMC.Model
        The PyMC model object.
    model_config : dict
        A dictionary containing the configuration parameters for the model.

    Methods:
    --------
    apply(data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        Apply the Menten Saturation component to the given data.

    """

    @property
    def default_saturation_config(self) -> Dict:
        return {
            "saturation_alpha": {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            },
            "saturation_lambda": {
                "dist": "Gamma",
                "kwargs": {"mu": 1, "sigma": 2},
            },
        }

    @property
    def variable_mapping(self) -> Dict[str, str]:
        """The mapping from saturation function parameters to variables in a model."""
        return {
            "lam": "saturation_lambda",
            "alpha": "saturation_alpha",
        }

    def saturation_function(self, x, alpha, lam):
        return michaelis_menten(
            x=x,
            alpha=alpha,
            lam=lam,
        )


class LogisticSaturationComponent(BaseSaturationFunction):
    """
    A class representing the Logistic Saturation component of a marketing mix model.

    Parameters:
    -----------
    model : PyMC.Model
        The PyMC model object.
    model_config : dict
        A dictionary containing the configuration parameters for the model.

    Methods:
    --------
    apply(data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        Apply the Logistic Saturation component to the given data.

    """

    @property
    def default_saturation_config(self) -> Dict:
        return {
            "saturation_beta": {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            },
            "saturation_lambda": {
                "dist": "Gamma",
                "kwargs": {"alpha": 3, "beta": 1},
            },
        }

    @property
    def variable_mapping(self) -> Dict[str, str]:
        """The mapping from saturation function parameters to variables in a model."""
        return {
            "lam": "saturation_lambda",
            "beta": "saturation_beta",
        }

    def saturation_function(self, x, beta, lam):
        return beta * logistic_saturation(x=x, lam=lam)


def _get_saturation_function(
    name: str, model: Optional[pm.Model] = None, model_config: Optional[Dict] = None
):
    """
    Get the saturation function based on the given name.

    Parameters:
    -----------
    name : str
        The name of the saturation function.
    model : pm.Model
        The PyMC model object.
    model_config : Optional[Dict], optional
        A dictionary containing the configuration parameters for the model, by default None.

    Returns:
    --------
    Union[HillSaturationComponent, MentenSaturationComponent, LogisticSaturationComponent]:
        The saturation function object based on the given name.

    Raises:
    -------
    ValueError:
        If the saturation function name is not recognized.
    """
    saturation_functions = {
        "hill": HillSaturationComponent,
        "michaelis_menten": MentenSaturationComponent,
        "logistic": LogisticSaturationComponent,
        # Add other lagging function classes here as needed
    }

    if name in saturation_functions:
        return saturation_functions[name](model=model, model_config=model_config)
    else:
        raise ValueError(f"Saturation function {name} not recognized.")
