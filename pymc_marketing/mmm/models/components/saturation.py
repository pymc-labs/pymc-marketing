import pymc as pm
from typing import Union, Optional, Dict
from pymc_marketing.mmm.transformers import hill_saturation, michaelis_menten
from pymc_marketing.mmm.utils import (
    _get_distribution, 
    _validate_model_config
)

### SATURATION FUNCTIONS
class BaseFunction:
    def __init__(self, max_lagging, model):
        self.max_lagging = max_lagging
        self.model = model

    def apply(self, data):
        raise NotImplementedError("Subclasses must implement this method.")


class HillSaturationComponent(BaseFunction):
    REQUIRED_KEYS = [
        "saturation_sigma_offset",
        "saturation_sigma_mu",
        "saturation_sigma_sigma",
        "saturation_lambda_offset",
        "saturation_lambda_mu",
        "saturation_lambda_sigma",
        "saturation_beta_offset",
        "saturation_beta_mu",
        "saturation_beta_sigma",
    ]

    def __init__(self, model, model_config):
        self.model = model
        self.model_config = model_config
        _validate_model_config(
            required_keys=self.REQUIRED_KEYS,
            model_config=self.model_config
        )

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        self.saturation_sigma_offset_dist = _get_distribution(
            dist=self.model_config["saturation_sigma_offset"]
        )
        self.saturation_sigma_mu_dist = _get_distribution(
            dist=self.model_config["saturation_sigma_mu"]
        )
        self.saturation_sigma_sigma_dist = _get_distribution(
            dist=self.model_config["saturation_sigma_sigma"]
        )


        self.saturation_lambda_offset_dist = _get_distribution(
            dist=self.model_config["saturation_lambda_offset"]
        )
        self.saturation_lambda_mu_dist = _get_distribution(
            dist=self.model_config["saturation_lambda_mu"]
        )
        self.saturation_lambda_sigma_dist = _get_distribution(
            dist=self.model_config["saturation_lambda_sigma"]
        )

        self.saturation_beta_offset_dist = _get_distribution(
            dist=self.model_config["saturation_beta_offset"]
        )
        self.saturation_beta_mu_dist = _get_distribution(
            dist=self.model_config["saturation_beta_mu"]
        )
        self.saturation_beta_sigma_dist = _get_distribution(
            dist=self.model_config["saturation_beta_sigma"]
        )


        with pm.modelcontext(self.model):
            saturation_sigma_offset_params = pm.find_constrained_prior(
                self.saturation_sigma_offset_dist, 
                lower=0.01, upper=0.9, 
                init_guess={"mu": 0.1, "sigma": 0.5}
            )
            saturation_sigma_offset = self.saturation_sigma_offset_dist(
                'saturation_sigma_offset',
                **saturation_sigma_offset_params,
                dims=("channel", "hierarchy")
            )

            saturation_sigma_mu = self.saturation_sigma_mu_dist(
                name="saturation_sigma_mu", 
                **self.model_config["saturation_sigma_mu"]["kwargs"]
            )
            saturation_sigma_sigma = self.saturation_sigma_sigma_dist(
                name="saturation_sigma_sigma", 
                **self.model_config["saturation_sigma_sigma"]["kwargs"]
            )

            saturation_sigma = pm.Deterministic(
                "saturation_sigma",
                var=saturation_sigma_mu + saturation_sigma_offset * saturation_sigma_sigma,
                dims=("channel", "hierarchy")
            )

            # lam
            saturation_lambda_offset_params = pm.find_constrained_prior(
                self.saturation_lambda_offset_dist, 
                lower=0.1, upper=2, 
                init_guess={"mu": 0.5, "sigma": 1}
            )
            saturation_lambda_offset = self.saturation_lambda_offset_dist(
                'saturation_lambda_offset',
                **saturation_lambda_offset_params,
                dims=("channel", "hierarchy")
            )
            saturation_lambda_mu = self.saturation_lambda_mu_dist(
                name="saturation_lambda_mu", 
                **self.model_config["saturation_lambda_mu"]["kwargs"]
            )
            saturation_lambda_sigma = self.saturation_lambda_sigma_dist(
                name="saturation_lambda_sigma", 
                **self.model_config["saturation_lambda_sigma"]["kwargs"]
            )

            saturation_lambda = pm.Deterministic(
                "saturation_lambda",
                var=saturation_lambda_mu + saturation_lambda_offset * saturation_lambda_sigma,
                dims=("channel", "hierarchy")
            )

            #beta
            saturation_beta_offset_params = pm.find_constrained_prior(
                self.saturation_beta_offset_dist, 
                lower=0.1, upper=2, 
                init_guess={"mu": 0.5, "sigma": 1}
            )
            saturation_beta_offset = self.saturation_beta_offset_dist(
                'saturation_beta_offset',
                **saturation_beta_offset_params,
                dims=("channel", "hierarchy")
            )
            saturation_beta_mu = self.saturation_beta_mu_dist(
                name="saturation_beta_mu", 
                **self.model_config["saturation_beta_mu"]["kwargs"]
            )
            saturation_beta_sigma = self.saturation_beta_sigma_dist(
                name="saturation_beta_sigma", 
                **self.model_config["saturation_beta_sigma"]["kwargs"]
            )

            saturation_beta = pm.Deterministic(
                "saturation_beta",
                var=saturation_beta_mu + saturation_beta_offset * saturation_beta_sigma,
                dims=("channel", "hierarchy")
            )

            return pm.Deterministic(
                'contribution',
                var=hill_saturation(
                    x=data,
                    sigma=saturation_sigma,
                    beta=saturation_beta,
                    lam=saturation_lambda
                ),
                dims=("date", "channel", "hierarchy")
            )
        
class MentenSaturationComponent(BaseFunction):
    REQUIRED_KEYS = [
        "saturation_alpha_offset",
        "saturation_alpha_mu",
        "saturation_alpha_sigma",
        "saturation_lambda_offset",
        "saturation_lambda_mu",
        "saturation_lambda_sigma",
    ]

    def __init__(self, model, model_config):
        self.model = model
        self.model_config = model_config
        _validate_model_config(
            required_keys=self.REQUIRED_KEYS,
            model_config=self.model_config
        )

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        self.saturation_alpha_offset_dist = _get_distribution(
            dist=self.model_config["saturation_alpha_offset"]
        )
        self.saturation_alpha_mu_dist = _get_distribution(
            dist=self.model_config["saturation_alpha_mu"]
        )
        self.saturation_alpha_sigma_dist = _get_distribution(
            dist=self.model_config["saturation_alpha_sigma"]
        )

        self.saturation_lambda_offset_dist = _get_distribution(
            dist=self.model_config["saturation_lambda_offset"]
        )
        self.saturation_lambda_mu_dist = _get_distribution(
            dist=self.model_config["saturation_lambda_mu"]
        )
        self.saturation_lambda_sigma_dist = _get_distribution(
            dist=self.model_config["saturation_lambda_sigma"]
        )


        with pm.modelcontext(self.model):
            saturation_alpha_offset_params = pm.find_constrained_prior(
                self.saturation_alpha_offset_dist, 
                lower=0.01, upper=0.9, 
                init_guess={"mu": 0.1, "sigma": 0.5}
            )
            saturation_alpha_offset = self.saturation_alpha_offset_dist(
                'saturation_alpha_offset',
                **saturation_alpha_offset_params,
                dims=("channel", "hierarchy")
            )

            saturation_alpha_mu = self.saturation_alpha_mu_dist(
                name="saturation_alpha_mu", 
                **self.model_config["saturation_alpha_mu"]["kwargs"]
            )
            saturation_alpha_sigma = self.saturation_alpha_sigma_dist(
                name="saturation_alpha_sigma", 
                **self.model_config["saturation_alpha_sigma"]["kwargs"]
            )

            saturation_alpha = pm.Deterministic(
                "saturation_alpha",
                var=saturation_alpha_mu + saturation_alpha_offset * saturation_alpha_sigma,
                dims=("channel", "hierarchy")
            )

            # lam
            saturation_lambda_offset_params = pm.find_constrained_prior(
                self.saturation_lambda_offset_dist, 
                lower=0.1, upper=2, 
                init_guess={"mu": 0.5, "sigma": 1}
            )
            saturation_lambda_offset = self.saturation_lambda_offset_dist(
                'saturation_lambda_offset',
                **saturation_lambda_offset_params,
                dims=("hierarchy")
            )
            saturation_lambda_mu = self.saturation_lambda_mu_dist(
                name="saturation_lambda_mu", 
                **self.model_config["saturation_lambda_mu"]["kwargs"]
            )
            saturation_lambda_sigma = self.saturation_lambda_sigma_dist(
                name="saturation_lambda_sigma", 
                **self.model_config["saturation_lambda_sigma"]["kwargs"]
            )

            saturation_lambda = pm.Deterministic(
                "saturation_lambda",
                var=saturation_lambda_mu + saturation_lambda_offset * saturation_lambda_sigma,
                dims=("hierarchy")
            )

            return pm.Deterministic(
                'contribution',
                var=michaelis_menten(
                    x=data,
                    alpha=saturation_alpha,
                    lam=saturation_lambda
                ),
                dims=("date", "channel", "hierarchy")
            )

def _get_saturation_function(
        name: str, 
        model: pm.Model, 
        model_config: Optional[Dict] = None
    ):
    saturation_functions = {
        "hill": HillSaturationComponent,
        "michaelis_menten": MentenSaturationComponent
        # Add other lagging function classes here as needed
    }

    if name in saturation_functions:
        return saturation_functions[name](model, model_config)
    else:
        raise ValueError(f"Saturation function {name} not recognized.")