#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""Multinomial Logit for Product Preference Analysis."""

import json
from typing import Any, cast

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib.axes import Axes
from typing_extensions import Self
from xarray import DataArray
import patsy

import pytensor.tensor as pt

from pymc_marketing.model_builder import ModelBuilder, create_idata_accessor
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.prior import Prior

HDI_ALPHA = 0.5

class MNLogit(ModelBuilder):
    """
    Multinomial Logit class.

    Class to perform a multinomial logit analysis with the
    specific intent of determining the product attribute
    effects on consumer preference.

    Parameters
    ----------
    choice_df : pd.DataFrame
        A wide DataFrame where each row is a choice scenario. Product-specific
        attributes are stored in columns, and the dependent variable identifies
        the chosen product.

    utility_equations : list of formula strings
        A list of formulas specifying how to model the utility of
        each product alternative. The formulas should be in Wilkinson
        style notation and allow the target product to be specified as
        as a function of the alternative specific attributes and the individual
        specific attributes: 
        target_product ~ target_attribute1 + target_attribute2 | individual_attribute

    depvar : str
        The name of the dependent variable in the choice_df.

    covariates : list of str
        Covariate names (e.g., ['X1', 'X2']).

    model_config : dict, optional
        Model configuration. If None, the default config is used.

    sampler_config : dict, optional
        Sampler configuration. If None, the default config is used.

    Notes
    -----

    Example:
    -------
    The format of `choice_df`:

        +------------+------------+------------+------------+------------+
        | Depvar     | alt_1_X1   | alt_1_X2   | alt_2_X1   | alt_2_X2   |
        +============+============+============+============+============+
        | alt_1      | 2.4        | 4.5        | 5.4        | 6.7        |
        +------------+------------+------------+------------+------------+
        | alt_2      | 3.5        | 6.7        | 2.3        | 8.9        |
        +------------+------------+------------+------------+------------+

    Example `utility_equations` list:

    >>> utility_equations = [
    ...     'alt_1 ~ X1_alt1 + X2_alt1 | income',
    ...     'alt_2 ~ X1_alt2 + X2_alt2 | income',
    ...     'alt_3 ~ X1_alt3 + X2_alt3 | income'
    ... ]

    """

    _model_type = "Multinomial Logit Model"
    version = "0.1.0"

    def __init__(
        self,
        choice_df: pd.DataFrame,
        utility_equations: list[str],
        depvar: str,
        covariates: list[str],
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        self.choice_df = choice_df
        self.utility_equations = utility_equations
        self.depvar = depvar
        self.covariates = covariates

        model_config = model_config or {}
        model_config = parse_model_config(model_config)

        super().__init__(model_config=model_config, sampler_config=sampler_config)

    @property
    def default_model_config(self) -> dict:
        """Default model configuration.

        This is a Categorical likelihood, Normal intercept,
        and a Normal vector of beta coefficients

        Returns
        -------
        dict
            The default model configuration.

        """
        alphas = Prior("Normal", mu=0, sigma=5, dims='alts')
        betas = Prior("Normal", mu=0, sigma=1, dims='alt_covariates')

        return {
            "alphas_": alphas,
            "betas": betas,
            "likelihood": Prior(
                "Categorical",
                p=0,
                dims="obs",
            )
        }
    
    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration."""
        return {}

    @property
    def output_var(self) -> str:
        """The output variable of the model."""
        return "y"

    @property
    def _serializable_model_config(self) -> dict[str, int | float | dict]:  # type: ignore
        result: dict[str, int | float | dict] = {
            "alphas_": self.model_config["alphas_"].to_dict(),
            "likelihood": self.model_config["likelihood"].to_dict(),
            "betas": self.model_config["betas"].to_dict(),
        }

        return result
    
    def parse_formula(self, df, formula, depvar):
        """ Helper function to parse the three part
            structure of the formula specification
        """
        target, covariates = formula.split('~')
        target = target.strip()
        alt_covariates, fixed_covariates = covariates.split('|')
        alt_covariates = alt_covariates.strip()
        fixed_covariates = fixed_covariates.strip()
        assert target in df[depvar].unique()
        for f in fixed_covariates.split('+'): 
            assert f.strip() in df.columns
        for a in alt_covariates.split('+'):
            assert a.strip() in df.columns
        return target, alt_covariates, fixed_covariates
    
    def prepare_X_matrix(self, df, utility_formulas, depvar):
        """ Helper function to prepare a X matrix for utility equations

                alt1 ~ alt1_X1 + alt1_X2 | income 
            
            The Dimensions of the X matrix should return a tensor
            with N observations x Alt x Covariates. Assumes utility
            formulas have an equal number of covariates per alternative. 
            These can be zero values if one alternative lacks some attribute.

            The utility formulas should express the driver relationship
            between the choice value in the dependent variable and the attributes
            of the alternative that would incentivise that choice. 
            The LHS of each formula needs to relate to a value of the dependent choice 
            variable and the RHS needs to express an additive relation of the available
            covariates

            We also allow for the incorporation of fixed covariates which do not vary
            across the alternatives. For the fixed covariates we need an alternative specific
            parameter to identify the contribution to utility varies.
        """
        n_obs = len(df)
        n_alts = len(utility_formulas)
        n_covariates = len(utility_formulas[0].split('|')[0].split('+'))

        alts = []
        alt_covariates = []
        fixed_covariates = []
        for f in utility_formulas:
            target, alt_covar, fixed_covar = self.parse_formula(df, f, depvar)
            f = '0 + ' + alt_covar
            alt_covariates.append(np.asarray(patsy.dmatrix(f, df)).T)
            alts.append(target)
            if fixed_covar:
                fixed_covariates.append(fixed_covar)

        if fixed_covariates:  
            F = np.unique(fixed_covariates)[0]
            F = '0 + ' + F
            F = np.asarray(patsy.dmatrix(F, df))
        else:
            F = []
        
        X = np.stack(alt_covariates, axis=1).T
        assert X.shape == (n_obs, n_alts, n_covariates)
        return X, F, alts, np.unique(fixed_covariates)
    
    @staticmethod
    def _prepare_y_outcome(df, alternatives, depvar):
        """ Helper function to categorically encode the outcome variable for
            use in the modelling. 

            The order of the alterntives should map to the order of the
            utility formulas. 
        """
        mode_mapping = dict(zip(alternatives, range(len(alternatives))))
        df['mode_encoded'] = df[depvar].map(mode_mapping)
        y = df['mode_encoded'].values
        return y
    
    @staticmethod
    def _prepare_coords(df, alternatives, covariates, f_covariates):
        coords = {
            "alts": alternatives,
            "alts_probs": alternatives[:-1],
            "alt_covariates": covariates,
            'fixed_covariates': [s.strip() for s in [s.split('+') for s in f_covariates][0]],
            "obs": range(len(df)),
    }
        return coords
    
    def preprocess_model_data(
        self,
    ) -> None:

        self.X, self.F, self.alternatives, self.fixed_covar = self.prepare_X_matrix(self.choice_df, 
                                                           self.utility_equations, 
                                                           self.depvar)
        self.y = self._prepare_y_outcome(self.choice_df, self.alternatives, self.depvar)

        # note: type hints for coords required for mypy to not get confused
        self.coords: dict[str, list[str]] = self._prepare_coords(self.choice_df, 
                                                                 self.alternatives, 
                                                                 self.covariates, 
                                                                 self.fixed_covar)
        
    def build_model(
        self
        ) -> None:
        self.preprocess_model_data()  # type: ignore
        
        with pm.Model(coords=self.coords) as model:
            # Intercept Parameters
            alphas = self.model_config["alphas_"].create_variable(name="alphas_")
            # Covariate Weight Parameters
            betas = self.model_config["betas"].create_variable("betas")

            # Instantiate covariate data for each Utility function
            X_data = pm.Data('X', self.X, dims=('obs', 'alts', 'alt_covariates'))
            # Instantiate outcome data
            observed = pm.Data('y', self.y, dims='obs')
            if self.F is not None:
                betas_fixed_ = pm.Normal('betas_fixed_', 0, 1, dims=('alts','fixed_covariates'))
                betas_fixed = pm.Deterministic('betas_fixed', pt.set_subtensor(betas_fixed_[-1, :], 0), 
                dims=('alts','fixed_covariates'))
                F_data = pm.Data('F', self.F)
                F = pm.Deterministic('F_interaction', pm.math.dot(F_data, betas_fixed.T))
            else: 
                F = pt.zeros(observed.shape[0])

            # Compute utility as a dot product
            U = pm.math.dot(X_data, betas)  # (N, alts)
            # Zero out reference alternative intercept
            alphas = pm.Deterministic('alphas', pt.set_subtensor(alphas[-1], 0), 
            dims='alts')
            U = pm.Deterministic("U", F + U + alphas, dims=("obs", "alts"))
            ## Apply Softmax Transform
            p_ = pm.Deterministic("p", pm.math.softmax(U, axis=1), 
            dims=("obs", "alts"))

            # likelihood
            choice_obs = pm.Categorical("likelihood", p=p_, 
                observed=observed, dims="obs")
        
        self.model = model

    def _data_setter(
            self,
            X: np.ndarray | pd.DataFrame,
            y: np.ndarray | pd.Series | None = None,
        ) -> None:
            """Set the data.

            Required from the parent class

            """

    def create_idata_attrs(self) -> dict[str, str]:
        """Create the attributes for the InferenceData object.

        Returns
        -------
        dict[str, str]
            The attributes for the InferenceData object.

        """
        attrs = super().create_idata_attrs()
        attrs["covariates"] = json.dumps(self.covariates)
        attrs["depvar"] = json.dumps(self.depvar)
        attrs["choice_df"] = json.dumps('Placeholder for DF')
        attrs["utility_equations"] = json.dumps(self.utility_equations)

        return attrs
    
    def sample_prior_predictive(self, extend_idata, kwargs):
        with self.model:  # sample with new input data
            prior_pred: az.InferenceData = pm.sample_prior_predictive(500, **kwargs)
            self.set_idata_attrs(prior_pred)

        if extend_idata:
            if self.idata is not None:
                self.idata.extend(prior_pred)
            else:
                self.idata = prior_pred

    def fit(self, extend_idata, kwargs):
        if extend_idata:
            with self.model:
                self.idata.extend(pm.sample(**kwargs))
        else: 
            with self.model:
                self.idata = pm.sample(**kwargs)

    def sample_posterior_predictive(self, extend_idata, kwargs):
        if extend_idata:
            with self.model:
                self.idata.extend(pm.sample_posterior_predictive(self.idata, 
                                                        var_names=['likelihood', 'p'],
                                                         **kwargs))
        else: 
            with self.model:
                self.post_pred = pm.sample_posterior_predictive(self.idata, 
                                                        var_names=['likelihood', 'p'],
                                                         **kwargs)
    
    

    def sample(
        self,
        sample_prior_predictive_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        sample_posterior_predictive_kwargs: dict | None = None,
    ) -> Self:
        """Sample all the things.

        Run all of the sample methods in the sequence:

        - :meth:`sample_prior_predictive`
        - :meth:`fit`
        - :meth:`sample_posterior_predictive`

        Parameters
        ----------
        sample_prior_predictive_kwargs : dict, optional
            The keyword arguments for the sample_prior_predictive method.
        fit_kwargs : dict, optional
            The keyword arguments for the fit method.
        sample_posterior_predictive_kwargs : dict, optional
            The keyword arguments for the sample_posterior_predictive method.

        Returns
        -------
        Self
            The model instance.

        """
        sample_prior_predictive_kwargs = sample_prior_predictive_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        sample_posterior_predictive_kwargs = sample_posterior_predictive_kwargs or {}
        self.build_model()

        self.sample_prior_predictive(extend_idata=True, kwargs=sample_prior_predictive_kwargs)
        self.fit(extend_idata=True, kwargs=fit_kwargs)
        self.sample_posterior_predictive(extend_idata=True, kwargs=sample_posterior_predictive_kwargs)
        return self


        
    