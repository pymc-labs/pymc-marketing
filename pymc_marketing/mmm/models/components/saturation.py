from typing import Dict, Optional, Union

import pymc as pm

from pymc_marketing.mmm.transformers import (
    hill_saturation,
    logistic_saturation,
    michaelis_menten,
)
from pymc_marketing.mmm.utils import _get_distribution, _validate_model_config


### SATURATION FUNCTIONS
class BaseFunction:
    def __init__(self, max_lagging, model):
        self.max_lagging = max_lagging
        self.model = model

    def apply(self, data):
        raise NotImplementedError("Subclasses must implement this method.")


class HillSaturationComponent(BaseFunction):
    REQUIRED_KEYS = [
        "saturation_sigma",
        "saturation_lambda",
        "saturation_beta",
    ]

    def __init__(self, model, model_config):
        self.model = model
        self.model_config = model_config
        _validate_model_config(
            required_keys=self.REQUIRED_KEYS, model_config=self.model_config
        )

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
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

            return pm.Deterministic(
                "channel_contributions",
                var=hill_saturation(
                    x=data,
                    sigma=saturation_sigma,
                    beta=saturation_beta,
                    lam=saturation_lambda,
                ),
                dims=("date", "channel"),
            )


class MentenSaturationComponent(BaseFunction):
    REQUIRED_KEYS = [
        "saturation_alpha",
        "saturation_lambda",
    ]

    def __init__(self, model, model_config):
        self.model = model
        self.model_config = model_config
        _validate_model_config(
            required_keys=self.REQUIRED_KEYS, model_config=self.model_config
        )

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
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

            return pm.Deterministic(
                "channel_contributions",
                var=michaelis_menten(
                    x=data, alpha=saturation_alpha, lam=saturation_lambda
                ),
                dims=("date", "channel"),
            )


class LogisticSaturationComponent(BaseFunction):
    REQUIRED_KEYS = [
        "saturation_beta",
        "saturation_lambda",
    ]

    def __init__(self, model, model_config):
        self.model = model
        self.model_config = model_config
        _validate_model_config(
            required_keys=self.REQUIRED_KEYS, model_config=self.model_config
        )

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
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

            return pm.Deterministic(
                name="channel_contributions",
                var=logistic_saturation(x=data, lam=saturation_lambda)
                * saturation_beta,
                dims=("date", "channel"),
            )


def _get_saturation_function(
    name: str, model: pm.Model, model_config: Optional[Dict] = None
):
    saturation_functions = {
        "hill": HillSaturationComponent,
        "michaelis_menten": MentenSaturationComponent,
        "logistic": LogisticSaturationComponent,
        # Add other lagging function classes here as needed
    }

    if name in saturation_functions:
        return saturation_functions[name](model, model_config)
    else:
        raise ValueError(f"Saturation function {name} not recognized.")
