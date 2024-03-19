from pymc_marketing.model_builder import ModelBuilder
from typing import Dict, Optional

import pandas as pd
import pymc as pm

class BaseMMM(ModelBuilder):
    _model_type = "BaseMMM"

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        model_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
    ):
        super().__init__(model_config, sampler_config)
        self.data = data

    def fit(self, sample_params=None, posterior_predictive_params=None):
        """
        Infer model posterior
        
        Parameters:
        ----------
        sample_params : dict, optional
            Parameters for the pm.sample() function. (default: None)
        posterior_predictive_params : dict, optional
            Parameters for the pm.sample_posterior_predictive() function. (default: None)

        Returns:
        -------
        self : object
            Returns the instance itself.
        """
        if isinstance(sample_params, type(None)):
            sample_params = {}
        if isinstance(posterior_predictive_params, type(None)):
            posterior_predictive_params = {}

        with self.model:
            self.idata = pm.sample(**sample_params)
            self.posterior_predictive = pm.sample_posterior_predictive(
                self.idata, **posterior_predictive_params
            )

        return self
