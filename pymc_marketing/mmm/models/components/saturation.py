from typing import Dict, List, Optional, Union

import pymc as pm

from pymc_marketing.mmm.transformers import (
    hill_saturation,
    logistic_saturation,
    michaelis_menten,
)
from pymc_marketing.mmm.utils import _get_distribution


### SATURATION FUNCTIONS
class BaseFunction:
    REQUIRED_KEYS: List[str] = []  # Default empty; subclasses should override this

    def __init__(
        self, model: Optional[pm.Model] = None, model_config: Optional[Dict] = None
    ):
        self.model = model
        self.model_config = self.initialize_model_config(model_config)

    def initialize_model_config(self, model_config: Optional[Dict]) -> Dict:
        if model_config is not None and not all(
            key in model_config for key in self.REQUIRED_KEYS
        ):
            update_model_config = {**model_config, **self._default_saturation_config}
        elif model_config is None:
            update_model_config = self._default_saturation_config
        else:
            update_model_config = model_config
        return update_model_config

    @property
    def _default_saturation_config(self) -> Dict:
        raise NotImplementedError("Subclasses should implement this method.")

    def apply(self, data):
        raise NotImplementedError("Subclasses must implement this method.")


class HillSaturationComponent(BaseFunction):
    """
    A class representing the Hill Saturation component of a marketing mix model.

    Parameters:
    -----------
    model : PyMC.Model
        The PyMC model object.
    model_config : dict
        A dictionary containing the configuration parameters for the model.

    Attributes:
    -----------
    REQUIRED_KEYS : list
        A list of required keys in the model_config dictionary.

    Methods:
    --------
    apply(data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        Apply the Hill Saturation component to the given data.

    """

    REQUIRED_KEYS = [
        "saturation_sigma",
        "saturation_lambda",
        "saturation_beta",
    ]

    def __init__(
        self, model: Optional[pm.Model] = None, model_config: Optional[Dict] = None
    ):
        super().__init__(model, model_config)

    @property
    def _default_saturation_config(self) -> Dict:
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

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        """
        Apply the Hill Saturation component to the given data.

        Parameters:
        -----------
        data : Union[pm.Data, pm.MutableData]
            The input data for the model.

        Returns:
        --------
        pm.Deterministic
            The deterministic variable representing the channel contributions.

        """
        self.saturation_sigma_dist = _get_distribution(
            dist=self.model_config["saturation_sigma"]
        )

        self.saturation_lambda_dist = _get_distribution(
            dist=self.model_config["saturation_lambda"]
        )

        self.saturation_beta_dist = _get_distribution(
            dist=self.model_config["saturation_beta"]
        )

        with pm.modelcontext(self.model):
            saturation_sigma = self.saturation_sigma_dist(
                "saturation_sigma",
                **self.model_config["saturation_sigma"]["kwargs"],
                dims=("channel"),
            )

            # lam
            saturation_lambda = self.saturation_lambda_dist(
                "saturation_lambda",
                **self.model_config["saturation_lambda"]["kwargs"],
                dims=("channel"),
            )

            # beta
            saturation_beta = self.saturation_beta_dist(
                "saturation_beta",
                **self.model_config["saturation_beta"]["kwargs"],
            )

            return hill_saturation(
                x=data,
                sigma=saturation_sigma,
                beta=saturation_beta,
                lam=saturation_lambda,
            )


class MentenSaturationComponent(BaseFunction):
    """
    A class representing the Menten Saturation component of a marketing mix model.

    Parameters:
    -----------
    model : PyMC.Model
        The PyMC model object.
    model_config : dict
        A dictionary containing the configuration parameters for the model.

    Attributes:
    -----------
    REQUIRED_KEYS : list
        A list of required keys in the model_config dictionary.

    Methods:
    --------
    apply(data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        Apply the Menten Saturation component to the given data.

    """

    REQUIRED_KEYS = [
        "saturation_alpha",
        "saturation_lambda",
    ]

    def __init__(
        self, model: Optional[pm.Model] = None, model_config: Optional[Dict] = None
    ):
        """
        Initialize the MentenSaturationComponent.

        Parameters:
        -----------
        model : PyMC.Model
            The PyMC model object.
        model_config : dict
            A dictionary containing the configuration parameters for the model.
        """
        super().__init__(model, model_config)

    @property
    def _default_saturation_config(self) -> Dict:
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

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        """
        Apply the Menten Saturation component to the given data.

        Parameters:
        -----------
        data : Union[pm.Data, pm.MutableData]
            The data to which the Menten Saturation component will be applied.

        Returns:
        --------
        pm.Deterministic:
            The result of applying the Menten Saturation component to the data.
        """
        self.saturation_alpha_dist = _get_distribution(
            dist=self.model_config["saturation_alpha"]
        )

        self.saturation_lambda_dist = _get_distribution(
            dist=self.model_config["saturation_lambda"]
        )

        with pm.modelcontext(self.model):
            # lam
            saturation_lambda = self.saturation_lambda_dist(
                "saturation_lambda",
                **self.model_config["saturation_lambda"]["kwargs"],
                dims=("channel"),
            )

            # alpha
            saturation_alpha = self.saturation_alpha_dist(
                "saturation_alpha",
                **self.model_config["saturation_alpha"]["kwargs"],
                dims=("channel"),
            )

            return michaelis_menten(
                x=data, alpha=saturation_alpha, lam=saturation_lambda
            )


class LogisticSaturationComponent(BaseFunction):
    """
    A class representing the Logistic Saturation component of a marketing mix model.

    Parameters:
    -----------
    model : PyMC.Model
        The PyMC model object.
    model_config : dict
        A dictionary containing the configuration parameters for the model.

    Attributes:
    -----------
    REQUIRED_KEYS : list
        A list of required keys in the model_config dictionary.

    Methods:
    --------
    apply(data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        Apply the Logistic Saturation component to the given data.

    """

    REQUIRED_KEYS = [
        "saturation_beta",
        "saturation_lambda",
    ]

    def __init__(
        self, model: Optional[pm.Model] = None, model_config: Optional[Dict] = None
    ):
        """
        Initialize the MentenSaturationComponent.

        Parameters:
        -----------
        model : PyMC.Model
            The PyMC model object.
        model_config : dict
            A dictionary containing the configuration parameters for the model.
        """
        super().__init__(model, model_config)

    @property
    def _default_saturation_config(self) -> Dict:
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

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        """
        Apply the Logistic Saturation component to the given data.

        Parameters:
        -----------
        data : Union[pm.Data, pm.MutableData]
            The data to which the Logistic Saturation component will be applied.

        Returns:
        --------
        pm.Deterministic:
            The result of applying the Logistic Saturation component to the data.
        """
        self.saturation_beta_dist = _get_distribution(
            dist=self.model_config["saturation_beta"]
        )

        self.saturation_lambda_dist = _get_distribution(
            dist=self.model_config["saturation_lambda"]
        )

        with pm.modelcontext(self.model):
            # lam
            saturation_lambda = self.saturation_lambda_dist(
                "saturation_lambda",
                **self.model_config["saturation_lambda"]["kwargs"],
                dims=("channel"),
            )

            # beta
            saturation_beta = self.saturation_beta_dist(
                "saturation_beta",
                **self.model_config["saturation_beta"]["kwargs"],
                dims=("channel"),
            )

            kwargs = {
                "lam": saturation_lambda,
                "beta": saturation_beta,
            }

            return self.saturation_function(x=data, **kwargs)


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
