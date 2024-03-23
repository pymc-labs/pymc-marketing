from typing import Dict, Optional, Union

import pymc as pm

from pymc_marketing.mmm.transformers import geometric_adstock, weibull_adstock
from pymc_marketing.mmm.utils import _get_distribution, _validate_model_config


class BaseFunction:
    def __init__(self, max_lagging, model):
        self.max_lagging = max_lagging
        self.model = model

    def apply(self, data):
        raise NotImplementedError("Subclasses must implement this method.")


class GeometricAdstockComponent(BaseFunction):
    REQUIRED_KEYS = [
        "adstock_alpha",
    ]

    def __init__(self, max_lagging, model, model_config):
        super().__init__(max_lagging, model)
        self.model_config = model_config
        _validate_model_config(
            required_keys=self.REQUIRED_KEYS, model_config=self.model_config
        )

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        self.adstock_alpha_dist = _get_distribution(
            dist=self.model_config["adstock_alpha"]
        )

        with pm.modelcontext(self.model):
            adstock_alpha = self.adstock_alpha_dist(
                name="adstock_alpha",
                dims="channel",
                **self.model_config["adstock_alpha"]["kwargs"],
            )

            return pm.Deterministic(
                name="channel_adstock",
                var=geometric_adstock(
                    x=data,
                    alpha=adstock_alpha,
                    l_max=self.max_lagging,
                    normalize=True,
                    axis=0,
                ),
                dims=("date", "channel"),
            )


class WeibullPDFAdstockComponent(BaseFunction):
    REQUIRED_KEYS = ["adstock_lambda", "adstock_shape"]

    def __init__(self, max_lagging, model, model_config):
        super().__init__(max_lagging, model)
        self.model_config = model_config
        _validate_model_config(
            required_keys=self.REQUIRED_KEYS, model_config=self.model_config
        )

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        self.adstock_lambda_dist = _get_distribution(
            dist=self.model_config["adstock_lambda"]
        )

        self.adstock_shape_dist = _get_distribution(
            dist=self.model_config["adstock_shape"]
        )

        with pm.modelcontext(self.model):
            adstock_lambda = self.adstock_lambda_dist(
                name="adstock_lambda",
                dims="channel",
                **self.model_config["adstock_lambda"]["kwargs"],
            )

            adstock_shape = self.adstock_shape_dist(
                name="adstock_shape",
                dims="channel",
                **self.model_config["adstock_shape"]["kwargs"],
            )

            return pm.Deterministic(
                name="channel_adstock",
                var=weibull_adstock(
                    x=data,
                    lam=adstock_lambda,
                    k=adstock_shape,
                    type="PDF",
                    l_max=self.max_lagging,
                ),
                dims=("date", "channel"),
            )


class WeibullCDFAdstockComponent(BaseFunction):
    REQUIRED_KEYS = ["adstock_lambda", "adstock_shape"]

    def __init__(self, max_lagging, model, model_config):
        super().__init__(max_lagging, model)
        self.model_config = model_config
        _validate_model_config(
            required_keys=self.REQUIRED_KEYS, model_config=self.model_config
        )

    def apply(self, data: Union[pm.Data, pm.MutableData]) -> pm.Deterministic:
        self.adstock_lambda_dist = _get_distribution(
            dist=self.model_config["adstock_lambda"]
        )

        self.adstock_shape_dist = _get_distribution(
            dist=self.model_config["adstock_shape"]
        )

        with pm.modelcontext(self.model):
            adstock_lambda = self.adstock_lambda_dist(
                name="adstock_lambda",
                dims="channel",
                **self.model_config["adstock_lambda"]["kwargs"],
            )

            adstock_shape = self.adstock_shape_dist(
                name="adstock_shape",
                dims="channel",
                **self.model_config["adstock_shape"]["kwargs"],
            )

            return pm.Deterministic(
                name="channel_adstock",
                var=weibull_adstock(
                    x=data,
                    lam=adstock_lambda,
                    k=adstock_shape,
                    type="CDF",
                    l_max=self.max_lagging,
                ),
                dims=("date", "channel"),
            )


def _get_lagging_function(
    name: str, max_lagging: int, model: pm.Model, model_config: Optional[Dict] = None
):
    lagging_functions = {
        "geometric": GeometricAdstockComponent,
        "weibull_pdf": WeibullPDFAdstockComponent,
        "weibull_cdf": WeibullCDFAdstockComponent,
        # Add other lagging function classes here as needed
    }

    if name in lagging_functions:
        return lagging_functions[name](max_lagging, model, model_config)
    else:
        raise ValueError(f"Lagging function {name} not recognized.")
