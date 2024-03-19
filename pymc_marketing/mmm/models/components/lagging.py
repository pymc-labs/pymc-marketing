import pymc as pm
from typing import Union, Optional, Dict
from pymc_marketing.mmm.utils import _validate_model_config, _get_distribution
from pymc_marketing.mmm.transformers import geometric_adstock


class BaseFunction:
    def __init__(self, max_lagging, model):
        self.max_lagging = max_lagging
        self.model = model

    def apply(self, data):
        raise NotImplementedError("Subclasses must implement this method.")

class GeometricAdstockComponent(BaseFunction):
    REQUIRED_KEYS = [
        "adstock_offset",
        "adstock_mu",
        "adstock_sigma",
    ]

    def __init__(self, max_lagging, model, model_config):
        super().__init__(max_lagging, model)
        self.model_config = model_config
        _validate_model_config(
            required_keys=self.REQUIRED_KEYS,
            model_config=self.model_config
        )

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        self.adstock_offset_dist = _get_distribution(
            dist=self.model_config["adstock_offset"]
        )
        self.adstock_mu_dist = _get_distribution(
            dist=self.model_config["adstock_mu"]
        )
        self.adstock_sigma_dist = _get_distribution(
            dist=self.model_config["adstock_sigma"]
        )

        with pm.modelcontext(self.model):
            adstock_offset_params = pm.find_constrained_prior(
                self.adstock_offset_dist, 
                lower=0.1, 
                upper=0.8, 
                init_guess={"mu": .5, "sigma": .5}
            )
            adstock_mu = self.adstock_mu_dist(name="adstock_mu", **self.model_config["adstock_mu"]["kwargs"])
            adstock_sigma = self.adstock_sigma_dist(name="adstock_sigma", **self.model_config["adstock_sigma"]["kwargs"])

            adstock_alpha_offset = self.adstock_offset_dist(
                'adstock_offset', 
                **adstock_offset_params,
                dims=("channel", "hierarchy")
            )

            adstock_alpha = pm.Deterministic("adstock_alpha", 
                                            var=adstock_mu + adstock_alpha_offset * adstock_sigma, 
                                            dims=("channel", "hierarchy")
                                            )

            return pm.Deterministic(
                'adstock_contribution',
                var = geometric_adstock(
                    x=data, 
                    alpha=adstock_alpha, 
                    l_max=self.max_lagging,
                    normalize=True),
                dims=("date", "channel", "hierarchy")
            )

def _get_lagging_function(
        name: str, 
        max_lagging: int, 
        model: pm.Model, 
        model_config: Optional[Dict] = None
    ):
    lagging_functions = {
        "geometric": GeometricAdstockComponent,
        # Add other lagging function classes here as needed
    }

    if name in lagging_functions:
        return lagging_functions[name](max_lagging, model, model_config)
    else:
        raise ValueError(f"Lagging function {name} not recognized.")