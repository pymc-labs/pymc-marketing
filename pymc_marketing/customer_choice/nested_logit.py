#   Copyright 2022 - 2026 The PyMC Labs Developers
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
"""Nested Logit for Product Preference Analysis."""

import json
import warnings
from typing import Self

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc.util import RandomState
from pymc_extras.prior import Prior

from pymc_marketing.model_builder import ModelBuilder, create_sample_kwargs
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.version import __version__

HDI_ALPHA = 0.5


class NestedLogit(ModelBuilder):
    """
    Nested Logit class.

    Class to perform a nested logit analysis with the
    specific intent of determining the product attribute
    effects on consumer preference. The implementation here
    is drawn from a discussion in Kenneth Train's book
    "Discrete Choice Methods with Simulation" Second
    Edition from 2009. Useful discussion of the model
    can also be found in Paez & Boisjoly's book
    "Discrete Choice Analysis with R" from 2022.

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
        Covariate names (e.g., ['X1', 'X2'])

    nested_structure: dict
        Dictionary to specify how to nest the choices between products.
        Single-layer nesting only. For more complex substitution patterns,
        consider using MixedLogit instead.

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

    .. code-block:: python

        utility_equations = [
            "alt_1 ~ X1_alt1 + X2_alt1 | income",
            "alt_2 ~ X1_alt2 + X2_alt2 | income",
            "alt_3 ~ X1_alt3 + X2_alt3 | income",
        ]

    Example nesting structure (single-layer only):

    .. code-block:: python

        nesting_structure = {
            "Land": ["alt_1", "alt_2"],
            "Air": ["alt_3"],
        }

    """

    _model_type = "Nested Logit Model"
    version = "0.2.0"

    def __init__(
        self,
        choice_df: pd.DataFrame,
        utility_equations: list[str],
        depvar: str,
        covariates: list[str],
        nesting_structure: dict,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        alphas_nests: bool = False,
    ):
        self.choice_df = choice_df
        self.utility_equations = utility_equations
        self.depvar = depvar
        self.covariates = covariates
        self.nesting_structure = nesting_structure
        self.alphas_nests = alphas_nests  # whether to estimate nest-specific intercepts

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
        alphas = Prior("Normal", mu=0, sigma=5, dims="alts")
        betas = Prior("Normal", mu=0, sigma=1, dims="alt_covariates")
        betas_fixed = Prior("Normal", mu=0, sigma=1, dims=("alts", "fixed_covariates"))
        lambdas_nests = Prior("Beta", alpha=1, beta=1, dims="nests")
        alphas_nests = Prior("Normal", mu=0, sigma=1, dims="nests")

        return {
            "alphas_": alphas,
            "betas": betas,
            "betas_fixed_": betas_fixed,
            "lambdas_nests": lambdas_nests,
            "alphas_nests": alphas_nests,
            "likelihood": Prior(
                "Categorical",
                p=0,
                dims="obs",
            ),
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
            "lambdas_nests": self.model_config["lambdas_nests"].to_dict(),
            "betas": self.model_config["betas"].to_dict(),
            "betas_fixed_": self.model_config["betas_fixed_"].to_dict(),
            "alphas_nests": self.model_config["alphas_nests"].to_dict(),
        }

        return result

    @staticmethod
    def _check_columns(
        df: pd.DataFrame, fixed_covariates: str, alt_covariates: str
    ) -> None:
        for f in fixed_covariates.split("+"):
            if f.strip() and f.strip() not in df.columns:
                raise ValueError(
                    f"Fixed covariate '{f.strip()}' not found in dataframe columns."
                )

        for a in alt_covariates.split("+"):
            if a.strip() and a.strip() not in df.columns:
                raise ValueError(
                    f"Alternative covariate '{a.strip()}' not found in dataframe columns."
                )

    @staticmethod
    def _check_dependent_variable(target: str, df: pd.DataFrame, depvar: str) -> None:
        if target not in df[depvar].unique():
            raise ValueError(
                f"Target '{target}' not found in dependent variable '{depvar}'."
            )

    def parse_formula(
        self, df: pd.DataFrame, formula: str, depvar: str
    ) -> tuple[str, str, str]:
        """Parse the three-part structure of a formula specification.

        Splits the formula into target, alternative-specific covariates, and
        fixed covariates. Ensures that the target variable appears in the
        dependent variable column and that all specified covariates exist
        in the input dataframe.
        """
        target, covariates = formula.split("~")
        target = target.strip()

        if "|" in covariates:
            alt_covariates, fixed_covariates = covariates.split("|")
        else:
            alt_covariates, fixed_covariates = covariates, ""
        alt_covariates = alt_covariates.strip()
        fixed_covariates = fixed_covariates.strip()

        self._check_columns(df, fixed_covariates, alt_covariates)
        self._check_dependent_variable(target, df, depvar)

        return target, alt_covariates, fixed_covariates

    def prepare_X_matrix(
        self, df: pd.DataFrame, utility_formulas: list[str], depvar: str
    ) -> tuple[np.ndarray, np.ndarray | None, list[str], np.ndarray]:
        """Prepare the X matrix for the utility equations.

        The X matrix is a tensor with dimensions:
        (N observations) x (Alternatives) x (Covariates).
        Assumes that the utility formulas specify an equal number of covariates
        per alternative; these can be zero-valued if an alternative lacks a
        specific attribute.

        Utility formulas should express the relationship between the choice
        outcome (dependent variable) and the attributes of each alternative
        that incentivize that choice. The left-hand side (LHS) of each formula
        must correspond to a value of the dependent variable, while the
        right-hand side (RHS) should define an additive combination of the
        available covariates.

        We also allow the incorporation of fixed covariates that do not vary
        across alternatives. For these, an alternative-specific parameter is
        used to allow the contribution to utility to vary by alternative.

        Returns
        -------
        Tuple containing:
        - X : ndarray of shape (n_obs, n_alts, n_covariates)
            The design matrix for the utility model.
        - F : Optional ndarray of shape (n_obs, n_fixed_covariates)
            The matrix for fixed covariates (None if no fixed covariates used).
        - alts : list of str
            Names of the alternatives (LHS of the utility equations).
        - unique_fixed_covariates : np.ndarray
            Array of unique fixed covariate names.
        """
        n_obs = len(df)
        n_alts = len(utility_formulas)
        n_covariates = len(utility_formulas[0].split("|")[0].split("+"))

        alts = []
        alt_covariates = []
        fixed_covariates = []
        for f in utility_formulas:
            target, alt_covar, fixed_covar = self.parse_formula(df, f, depvar)
            f = "0 + " + alt_covar
            alt_covariates.append(np.asarray(patsy.dmatrix(f, df)).T)
            alts.append(target)
            if fixed_covar:
                fixed_covariates.append(fixed_covar)

        for alt in df[depvar].unique():
            if alt not in alts:
                raise ValueError(f"""There is an alternative {alt} in the data
                                 with no matching equation""")
        if fixed_covariates:
            F = np.unique(fixed_covariates)[0]
            F = "0 + " + F
            F = np.asarray(patsy.dmatrix(F, df))
        else:
            F = None

        X = np.stack(alt_covariates, axis=1).T
        if X.shape != (n_obs, n_alts, n_covariates):
            raise ValueError(
                f"X has shape {X.shape}, but expected {(n_obs, n_alts, n_covariates)}."
            )
        return X, F, alts, np.unique(fixed_covariates)

    @staticmethod
    def _prepare_y_outcome(
        df: pd.DataFrame, alternatives: list[str], depvar: str
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Encode the outcome category variable for use in the modelling.

        The order of the alterntives should map to the order of the
        utility formulas.
        """
        prod_mapping = dict(zip(alternatives, range(len(alternatives)), strict=False))
        df["mode_encoded"] = df[depvar].map(prod_mapping)
        y = np.asarray(df["mode_encoded"].values)
        return y, prod_mapping

    @staticmethod
    def _parse_nesting(
        nest_dict: dict[str, list[str]],
        product_indices: dict[str, int],
    ) -> dict[str, np.ndarray]:
        """Parse single-layer nesting structure.

        Parameters
        ----------
        nest_dict : dict[str, list[str]]
            Mapping of nest names to lists of alternative names
        product_indices : dict[str, int]
            Mapping of alternative names to indices

        Returns
        -------
        nest_indices : dict[str, np.ndarray]
            Mapping of nest names to arrays of alternative indices

        Examples
        --------
        >>> nest_dict = {"Land": ["Car", "Bus"], "Air": ["Plane"]}
        >>> product_indices = {"Car": 0, "Bus": 1, "Plane": 2}
        >>> _parse_nesting(nest_dict, product_indices)
        {"Land": array([0, 1]), "Air": array([2])}
        """
        if not nest_dict:
            raise ValueError("Nesting structure must not be empty.")

        nest_indices = {}
        for nest_name, alternatives in nest_dict.items():
            if not isinstance(alternatives, list):
                raise ValueError(
                    f"Nest '{nest_name}' must map to a list of alternatives. "
                    f"Two-layer nesting is not supported. "
                    f"Consider using MixedLogit for complex substitution patterns."
                )
            indices = [product_indices[alt] for alt in alternatives]
            nest_indices[nest_name] = np.sort(indices)

        return nest_indices

    @staticmethod
    def _prepare_coords(
        df, alternatives, covariates, f_covariates, nests, nest_indices
    ):
        """Prepare coordinates for PyMC nested logit model.

        Parameters
        ----------
        df : pd.DataFrame
            The choice dataframe
        alternatives : list[str]
            List of all alternatives
        covariates : list[str]
            List of covariate names
        f_covariates : np.ndarray or list
            Fixed covariate names
        nests : list[str]
            List of nest names
        nest_indices : dict[str, np.ndarray]
            Mapping of nest names to alternative indices

        Returns
        -------
        coords : dict
            Coordinate dictionary for PyMC model
        """
        if isinstance(f_covariates, np.ndarray) & (f_covariates.size > 0):
            f_cov = [s.strip() for s in f_covariates[0].split("+")]
        else:
            f_cov = []

        coords = {
            "alts": alternatives,
            "alts_probs": alternatives[:-1],
            "alt_covariates": covariates,
            "fixed_covariates": f_cov,
            "nests": nests,
            "obs": range(len(df)),
        }

        # Add nest-specific alternative dimensions
        for nest_name, indices in nest_indices.items():
            nest_alts = [alternatives[i] for i in indices]
            coords[f"{nest_name}_alts"] = nest_alts

        return coords

    def preprocess_model_data(
        self, choice_df, utility_equations
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Pre-process the model initiation inputs into a format that can be used by the PyMC model.

        This method prepares the 3D design matrix `X`, fixed covariate matrix `F` (if applicable),
        and the encoded response vector `y`, while also extracting and storing relevant metadata
        such as alternatives, fixed covariate names, product index mappings, and nesting structures.

        Parameters
        ----------
        choice_df : pd.DataFrame
            A pandas DataFrame containing the observed choices and covariates for each alternative.
            Each row represents an individual choice observation.

        utility_equations : list[str]
            A list of model formulas, one per alternative. Each formula should be of the form:
            `"alt_name ~ alt_covariates | fixed_covariates"`.
            The left-hand side identifies the alternative name; the right-hand side specifies
            the covariates used to explain utility for that alternative.

        Returns
        -------
        X : np.ndarray
            A 3D numpy array of shape (n_observations, n_alternatives, n_covariates), representing
            the covariate tensor for alternative-specific attributes.

        F : np.ndarray | None
            A 2D numpy array (n_observations, n_fixed_covariates) for covariates shared across
            alternatives, or None if no such covariates are used.

        y : np.ndarray
            A 1D numpy array of encoded target labels (integers), where each entry represents
            the chosen alternative for an observation.

        Notes
        -----
        - Updates internal state: assigns `X_data`, `F`, `alternatives`, `fixed_covar`, `y`,
        `prod_indices`, `nest_indices`, `all_nests`, `lambda_lkup`, and `coords`.
        - Handles single-layer nesting structures only.
        - Assumes the existence of instance attributes `depvar`, `covariates`, and `nesting_structure`.

        """
        X, F, alternatives, fixed_covar = self.prepare_X_matrix(
            choice_df, utility_equations, self.depvar
        )
        self.X_data = X
        self.F = F
        self.alternatives = alternatives
        self.fixed_covar = fixed_covar

        y, prod_mapping = self._prepare_y_outcome(
            choice_df, self.alternatives, self.depvar
        )
        self.y = y
        self.prod_indices = prod_mapping

        # Parse nesting structure (single-layer only)
        nest_indices = self._parse_nesting(self.nesting_structure, self.prod_indices)
        self.nest_indices = {"top": nest_indices}
        self.all_nests = list(nest_indices.keys())
        self.lambda_lkup = dict(
            zip(self.all_nests, range(len(self.all_nests)), strict=False)
        )

        # note: type hints for coords required for mypy to not get confused
        # Pass nest_indices to _prepare_coords so it can create nest-specific dims
        self.coords: dict[str, list[str]] = self._prepare_coords(
            choice_df,
            self.alternatives,
            self.covariates,
            self.fixed_covar,
            self.all_nests,
            nest_indices,  # Add this parameter
        )

        return X, F, y

    def build_model(self, **kwargs) -> None:
        """
        Build model using stored choice_df and utility_equations.

        This is the abstract method from ModelBuilder. For discrete choice,
        we don't pass data as arguments - we use the stored data from __init__.
        """
        X, F, y = self.preprocess_model_data(self.choice_df, self.utility_equations)
        self.model = self.make_model(X, F, y)

    def make_intercepts(self) -> pt.TensorVariable:
        """Create alternative-specific intercepts with reference alternative set to zero.

        Returns
        -------
        alphas : TensorVariable
            Alternative-specific intercepts with last alternative = 0
        """
        alphas_ = self.model_config["alphas_"].create_variable(name="alphas_")
        alphas = pm.Deterministic(
            "alphas", pt.set_subtensor(alphas_[-1], 0), dims="alts"
        )
        return alphas

    def make_alt_coefs(self) -> pt.TensorVariable:
        """Create coefficients for alternative-specific covariates.

        Returns
        -------
        betas : TensorVariable
            Coefficients for alternative-specific covariates
        """
        betas = self.model_config["betas"].create_variable("betas")
        return betas

    def make_fixed_coefs(
        self, X_fixed: np.ndarray | None, n_obs: int, n_alts: int
    ) -> pt.TensorVariable:
        """Create alternative-varying coefficients for fixed (non-varying) covariates.

        Each fixed covariate gets a separate coefficient for each alternative, allowing
        the effect of individual characteristics (e.g., income, age) to vary by choice.
        The reference alternative (last) has all coefficients constrained to zero for
        identification.

        Parameters
        ----------
        X_fixed : np.ndarray or None
            Fixed covariates matrix of shape (n_obs, n_fixed_covariates)
        n_obs : int
            Number of observations
        n_alts : int
            Number of alternatives

        Returns
        -------
        W_contrib : TensorVariable
            Contribution to utility from fixed covariates, shape (n_obs, n_alts)
        """
        if X_fixed is None or len(X_fixed) == 0:
            W_contrib = pt.zeros((n_obs, n_alts))
        else:
            W_data = pm.Data("W", X_fixed, dims=("obs", "fixed_covariates"))
            betas_fixed_ = self.model_config["betas_fixed_"].create_variable(
                "betas_fixed_"
            )
            # Set reference alternative coefficients to zero for identification
            # This creates a (n_alts, n_fixed_covariates) matrix where each
            # alternative has its own response to each fixed covariate
            betas_fixed = pm.Deterministic(
                "betas_fixed",
                pt.set_subtensor(betas_fixed_[-1, :], 0),
                dims=("alts", "fixed_covariates"),
            )
            W_contrib = pm.Deterministic(
                "w_nest", pm.math.dot(W_data, betas_fixed.T), dims=("obs", "alts")
            )
        return W_contrib

    def make_lambdas(self) -> pt.TensorVariable:
        """Create nest-specific lambda (scale) parameters.

        Returns
        -------
        lambdas_nests : TensorVariable
            Lambda parameters for each nest, controlling within-nest correlation
        """
        lambdas_nests = self.model_config["lambdas_nests"].create_variable(
            "lambdas_nests"
        )
        lambdas_nests = pm.math.clip(lambdas_nests, 0.05, 1.0)
        return lambdas_nests

    def calc_conditional_prob(
        self,
        U: pt.TensorVariable,
        lambdas: pt.TensorVariable,
        nest_idx: int,
        nest_name: str,
        nest_indices: dict[str, np.ndarray],
        alphas_nest: pt.TensorVariable,
    ) -> tuple[pt.TensorVariable, pt.TensorVariable]:
        """Calculate conditional probability within a nest.

        This implements the scaled softmax probability within a nest:
        P(y_i = j | j ∈ nest) = exp(U_ij / λ) / Σ_{k ∈ nest} exp(U_ik / λ)

        Parameters
        ----------
        U : TensorVariable
            Systematic utility, shape (n_obs, n_alts)
        lambdas : TensorVariable
            Nest-specific lambda parameters
        nest_idx : int
            Index of current nest in lambda array
        nest_name : str
            Name of the nest
        nest_indices : dict
            Mapping of nest names to alternative indices

        Returns
        -------
        exp_W_nest : TensorVariable
            Exponentiated inclusive value for the nest, shape (n_obs,)
        P_y_given_nest : TensorVariable
            Conditional probabilities within the nest, shape (n_obs, n_alts_in_nest)
        """
        # Extract utilities for alternatives in this nest
        alt_indices = nest_indices[nest_name]
        u_nest = U[:, alt_indices]

        # Store utilities in deterministic for inspection
        y_nest = pm.Deterministic(
            f"y_{nest_name}", u_nest, dims=("obs", f"{nest_name}_alts")
        )

        # Numerical stability: subtract max across alternatives for each observation
        # This prevents overflow in exp() calculations
        max_y_nest = pm.math.max(y_nest, axis=1, keepdims=True)

        # Conditional probability within nest (scaled softmax)
        # Using softmax directly is more stable than manual exp/sum
        P_y_given_nest = pm.Deterministic(
            f"p_y_given_{nest_name}",
            pm.math.softmax((y_nest - max_y_nest) / lambdas[nest_idx], axis=1),
            dims=("obs", f"{nest_name}_alts"),
        )

        # Inclusive value (log-sum-exp) for each observation
        # I_nest[i] = λ * log(Σ_j exp(U_ij / λ))
        # The log-sum-exp trick: log(Σ exp(x)) = max(x) + log(Σ exp(x - max(x)))
        lsexp = pm.math.logsumexp((y_nest - max_y_nest) / lambdas[nest_idx], axis=1)
        I_nest = pm.Deterministic(f"I_{nest_name}", lambdas[nest_idx] * lsexp)

        # Exponentiated inclusive value for nest probability calculation
        exp_W_nest = pm.math.exp(alphas_nest[nest_idx] + I_nest)

        return exp_W_nest, P_y_given_nest

    def make_nest_probs(
        self,
        U: pt.TensorVariable,
        lambdas: pt.TensorVariable,
        nest_indices: dict[str, np.ndarray],
        alphas_nest: pt.TensorVariable | None = None,
    ) -> tuple[dict[str, pt.TensorVariable], dict[str, pt.TensorVariable]]:
        """Calculate nest selection probabilities and conditional probabilities.

        This computes:
        1. P(nest) = exp(I_nest) / Σ exp(I_nest')  [nest selection probability]
        2. P(y|nest) = exp(U_y/λ) / Σ_{j∈nest} exp(U_j/λ)  [within-nest choice]

        Parameters
        ----------
        U : TensorVariable
            Systematic utility, shape (n_obs, n_alts)
        lambdas : TensorVariable
            Nest-specific lambda parameters
        nest_indices : dict
            Mapping of nest names to alternative indices

        Returns
        -------
        nest_probs : dict
            Nest selection probabilities
        conditional_probs : dict
            Within-nest conditional probabilities
        """
        exp_inclusive_values = []
        conditional_probs = {}

        # Calculate conditional probs and inclusive values for each nest
        for i, nest_name in enumerate(nest_indices.keys()):
            exp_W, P_cond = self.calc_conditional_prob(
                U, lambdas, i, nest_name, nest_indices, alphas_nest=alphas_nest
            )
            exp_inclusive_values.append(exp_W)
            conditional_probs[nest_name] = P_cond

        # Normalize to get nest selection probabilities
        total_inclusive = pm.math.sum(exp_inclusive_values, axis=0)

        nest_probs = {}
        for i, nest_name in enumerate(nest_indices.keys()):
            nest_probs[nest_name] = pm.Deterministic(
                f"P_{nest_name}", exp_inclusive_values[i] / total_inclusive, dims="obs"
            )

        return nest_probs, conditional_probs

    def make_model(self, X, W, y) -> pm.Model:
        """Build nested logit model.

        Parameters
        ----------
        X : np.ndarray
            Alternative-specific covariates, shape (n_obs, n_alts, n_covariates)
        W : np.ndarray or None
            Fixed covariates, shape (n_obs, n_fixed_covariates)
        y : np.ndarray
            Observed choices, shape (n_obs,)
        alphas_nests : bool, optional
            Whether to include nest-specific intercepts (default is False)

        Returns
        -------
        model : pm.Model
            PyMC model
        """
        n_obs, n_alts = X.shape[0], X.shape[1]
        nest_indices = self.nest_indices["top"]  # Only single-layer nesting

        with pm.Model(coords=self.coords) as model:
            # Create parameters
            alphas = self.make_intercepts()
            betas = self.make_alt_coefs()
            lambdas = self.make_lambdas()
            if self.alphas_nests:
                alphas_nests_ = self.model_config["alphas_nests"].create_variable(
                    name="alphas_nests"
                )
            else:
                alphas_nests_ = pt.zeros(len(nest_indices.keys()))

            # Data containers
            X_data = pm.Data("X", X, dims=("obs", "alts", "alt_covariates"))
            y_data = pm.Data("y", y, dims="obs")

            # Fixed covariate contribution
            W_contrib = self.make_fixed_coefs(W, n_obs, n_alts)

            # Systematic utility
            U = pm.Deterministic(
                "U",
                alphas + pm.math.dot(X_data, betas) + W_contrib,
                dims=("obs", "alts"),
            )

            # Nest probabilities and conditional probabilities
            nest_probs, conditional_probs = self.make_nest_probs(
                U, lambdas, nest_indices, alphas_nest=alphas_nests_
            )

            # Combine to get final choice probabilities
            # P(y) = P(y|nest) * P(nest)
            p = pt.zeros((n_obs, n_alts))

            for nest_name, indices in nest_indices.items():
                p = pt.set_subtensor(
                    p[:, indices],
                    conditional_probs[nest_name] * nest_probs[nest_name][:, None],
                )
            p = pm.Deterministic("p", p, dims=("obs", "alts"))
            # Likelihood
            _ = pm.Categorical("likelihood", p=p, observed=y_data, dims="obs")

        self.model = model
        return model

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
        attrs["choice_df"] = json.dumps("Placeholder for DF")
        attrs["nesting_structure"] = json.dumps(self.nesting_structure)
        attrs["utility_equations"] = json.dumps(self.utility_equations)
        attrs["alphas_nests"] = json.dumps(self.alphas_nests)

        return attrs

    def sample_prior_predictive(
        self,
        choice_df: pd.DataFrame | None = None,
        utility_equations: list[str] | None = None,
        samples: int = 500,
        extend_idata: bool = True,
        **kwargs,
    ) -> az.InferenceData:
        """
        Sample from prior predictive distribution.

        Parameters
        ----------
        choice_df : pd.DataFrame, optional
            New choice data. If None, uses data from initialization.
        utility_equations : list[str], optional
            New utility equations. If None, uses from initialization.
        samples : int, optional
            Number of prior samples
        extend_idata : bool, optional
            Whether to add to self.idata
        **kwargs
            Additional arguments for pm.sample_prior_predictive

        Returns
        -------
        az.InferenceData
            Prior predictive samples
        """
        if choice_df is not None:
            self.choice_df = choice_df
        if utility_equations is not None:
            self.utility_equations = utility_equations

        if not hasattr(self, "model"):
            self.build_model()

        with self.model:
            prior_pred = pm.sample_prior_predictive(samples, **kwargs)
            prior_pred["prior"].attrs["pymc_marketing_version"] = __version__
            prior_pred["prior_predictive"].attrs["pymc_marketing_version"] = __version__
            self.set_idata_attrs(prior_pred)

        if extend_idata:
            if self.idata is not None:
                self.idata.extend(prior_pred, join="right")
            else:
                self.idata = prior_pred

        return prior_pred

    def _create_fit_data(self) -> xr.Dataset:
        """
        Create xarray Dataset for storing choice_df in InferenceData.

        This allows the model to be reconstructed when loading from file.

        Returns
        -------
        xr.Dataset
            Choice data as xarray Dataset with 'obs' dimension
        """
        df_xr = self.choice_df.to_xarray()
        df_xr = df_xr.rename({"index": "obs"})
        return df_xr

    def fit(
        self,
        choice_df: pd.DataFrame | None = None,
        utility_equations: list[str] | None = None,
        progressbar: bool | None = None,
        random_seed: RandomState | None = None,
        **kwargs,
    ) -> az.InferenceData:
        """
        Fit the discrete choice model.

        Parameters
        ----------
        choice_df : pd.DataFrame, optional
            New choice data. If None, uses data from initialization.
        utility_equations : list[str], optional
            New utility equations. If None, uses equations from initialization.
        progressbar : bool, optional
            Show progress bar during sampling
        random_seed : RandomState, optional
            Random seed for reproducibility
        **kwargs
            Additional arguments passed to pm.sample()

        Returns
        -------
        az.InferenceData
            Fitted model with posterior samples
        """
        # Allow updating data at fit time
        if choice_df is not None:
            self.choice_df = choice_df
        if utility_equations is not None:
            self.utility_equations = utility_equations

        # Build model if not already built
        if not hasattr(self, "model"):
            self.build_model()

        # Configure sampler
        sampler_kwargs = create_sample_kwargs(
            self.sampler_config,
            progressbar,
            random_seed,
            **kwargs,
        )

        # Sample
        with self.model:
            idata = pm.sample(**sampler_kwargs)

        # Store and extend results
        if self.idata:
            self.idata = self.idata.copy()
            self.idata.extend(idata, join="right")
        else:
            self.idata = idata

        # Add version metadata
        self.idata["posterior"].attrs["pymc_marketing_version"] = __version__

        # Add fit_data group
        if "fit_data" in self.idata:
            del self.idata.fit_data

        fit_data = self._create_fit_data()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            self.idata.add_groups(fit_data=fit_data)

        # Set attributes for save/load
        self.set_idata_attrs(self.idata)

        return self.idata

    def build_from_idata(self, idata: az.InferenceData) -> None:
        """
        Build model from loaded InferenceData.

        This is called by load() after the model is initialized.

        Parameters
        ----------
        idata : az.InferenceData
            Loaded inference data
        """
        self.choice_df = idata["fit_data"].to_dataframe()
        if not hasattr(self, "model"):
            self.build_model()

    def sample_posterior_predictive(
        self,
        choice_df: pd.DataFrame | None = None,
        extend_idata: bool = True,
        **kwargs,
    ) -> az.InferenceData:
        """
        Sample from posterior predictive distribution.

        Parameters
        ----------
        choice_df : pd.DataFrame, optional
            New choice data for prediction. If None, uses training data.
        extend_idata : bool, optional
            Whether to add to self.idata
        **kwargs
            Additional arguments for pm.sample_posterior_predictive

        Returns
        -------
        az.InferenceData
            Posterior predictive samples
        """
        if choice_df is not None:
            # Update data in existing model
            new_X, new_F, new_y = self.preprocess_model_data(
                choice_df, self.utility_equations
            )
            with self.model:
                data_dict = {"X": new_X, "y": new_y}
                if new_F is not None and len(new_F) > 0:
                    data_dict["W"] = new_F
                pm.set_data(data_dict)

        if self.idata is None:
            raise RuntimeError("self.idata must be initialized before extending")
        with self.model:
            post_pred = pm.sample_posterior_predictive(
                self.idata, var_names=["likelihood", "p"], **kwargs
            )

        if extend_idata:
            self.idata.extend(post_pred, join="right")

        return post_pred

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
        fit_kwargs = fit_kwargs or {
            "nuts_sampler": "numpyro",
            "idata_kwargs": {"log_likelihood": True},
        }
        sample_posterior_predictive_kwargs = sample_posterior_predictive_kwargs or {}
        if not hasattr(self, "model"):
            X, F, y = self.preprocess_model_data(self.choice_df, self.utility_equations)  # type: ignore
            model = self.make_model(X, F, y)
            self.model = model

        self.sample_prior_predictive(
            extend_idata=True, **sample_prior_predictive_kwargs
        )
        self.fit(extend_idata=True, **fit_kwargs)
        self.sample_posterior_predictive(
            extend_idata=True, **sample_posterior_predictive_kwargs
        )
        return self

    def apply_intervention(
        self,
        new_choice_df: pd.DataFrame,
        new_utility_equations: list[str] | None = None,
        fit_kwargs: dict | None = None,
    ) -> az.InferenceData:
        r"""Apply one of two types of intervention.

        This method supports two intervention strategies:

        1. **Observable attribute changes**: Uses the existing fitted model and modifies observable
        inputs (e.g., prices, features) to simulate how market shares change. This method uses
        posterior predictive sampling.

        2. **Product removal**: Allows a product to be entirely removed from the choice set and
        simulates how demand redistributes among the remaining alternatives. This re-specifies
        and re-estimates the model before posterior prediction.

        Parameters
        ----------
        new_choice_df : pd.DataFrame
            The new dataset reflecting changes to observable attributes or product availability.

        new_utility_equations : list[str] | None, optional
            An updated list of utility specifications for each alternative, if different from
            the original model. If `None`, the original equations are reused and only the data is changed.

        fit_kwargs : dict | None, optional
            Keyword arguments for sampling if refitting the model. Default uses high target_accept
            and extended tuning.

        Returns
        -------
        az.InferenceData
            The posterior or full predictive distribution under the intervention, including
            predicted probabilities (`"p"`) and likelihood draws (`"likelihood"`).

        """
        if fit_kwargs is None:
            fit_kwargs = {
                "target_accept": 0.97,
                "tune": 2000,
                "idata_kwargs": {"log_likelihood": True},
            }

        if not hasattr(self, "model"):
            self.sample()
        if new_utility_equations is None:
            new_X, new_F, new_y = self.preprocess_model_data(
                new_choice_df, self.utility_equations
            )
            with self.model:
                if new_F is None:
                    pm.set_data({"X": new_X, "y": new_y})
                else:
                    pm.set_data({"X": new_X, "W": new_F, "y": new_y})
                # use the updated values and predict outcomes and probabilities:
                idata_new_policy = pm.sample_posterior_predictive(
                    self.idata,
                    var_names=["p", "likelihood"],
                    return_inferencedata=True,
                    extend_inferencedata=False,
                    random_seed=100,
                )

            self.intervention_idata = idata_new_policy
        else:
            new_X, new_F, new_y = self.preprocess_model_data(
                new_choice_df, new_utility_equations
            )
            new_model = self.make_model(new_X, new_F, new_y)
            with new_model:
                idata_new_policy = pm.sample_prior_predictive()
                idata_new_policy.extend(pm.sample(**fit_kwargs))
                idata_new_policy.extend(
                    pm.sample_posterior_predictive(
                        idata_new_policy, var_names=["p", "likelihood"]
                    )
                )

            self.intervention_idata = idata_new_policy

        return idata_new_policy

    @staticmethod
    def calculate_share_change(
        idata: az.InferenceData, new_idata: az.InferenceData
    ) -> pd.DataFrame:
        """Calculate difference in market share due to market intervention.

        Parameters
        ----------
        idata : az.InferenceData
            Posterior predictive samples under the baseline (pre-intervention) policy.
            Must contain a "posterior_predictive" group with a "p" variable representing
            predicted market shares.

        new_idata : az.InferenceData
            Posterior predictive samples under the new (post-intervention) policy.
            Structure should match `idata`, with a "posterior_predictive" group containing "p".

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by product, containing:
            - 'policy_share': mean predicted share under the baseline policy
            - 'new_policy_share': mean predicted share under the new policy
            - 'relative_change': relative change in share due to the intervention
        """
        expected = idata["posterior_predictive"].mean(dim=("chain", "draw", "obs"))["p"]
        expected_new = new_idata["posterior_predictive"].mean(
            dim=("chain", "draw", "obs")
        )["p"]
        shares_df = pd.DataFrame(
            {"product": expected["alts"], "policy_share": expected}
        )
        shares_df_new = pd.DataFrame(
            {"product": expected_new["alts"], "new_policy_share": expected_new}
        )
        shares_df = shares_df.merge(
            shares_df_new, left_on="product", right_on="product", how="left"
        )
        shares_df.fillna(0, inplace=True)

        shares_df["relative_change"] = (
            shares_df["new_policy_share"] - shares_df["policy_share"]
        ) / shares_df["policy_share"]

        shares_df.set_index("product", inplace=True)
        return shares_df

    @staticmethod
    def plot_change(
        change_df: pd.DataFrame,
        title: str = "Change in Proportion due to Intervention",
        figsize: tuple = (8, 4),
    ) -> plt.Figure:
        """Plot change induced by a market intervention.

        Parameters
        ----------
        change_df : pd.DataFrame
            A DataFrame indexed by product (or mode) with the following columns:
            - 'policy_share': share before the intervention
            - 'new_policy_share': share after the intervention

        title : str, optional
            Title of the plot. Default is "Change in Proportion due to Intervention".

        figsize : tuple, optional
            Size of the plot in inches, as (width, height). Default is (8, 4).

        Returns
        -------
        object
            A matplotlib Figure object showing before/after market share comparisons,
            with colored lines indicating gain (green) or loss (red) for each product.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.axvline(x=1, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(-0.2, 1.2)

        upperbound = change_df[["policy_share", "new_policy_share"]].max().max() + 0.05
        ax.text(
            -0.05, upperbound, "BEFORE", fontsize=12, color="black", fontweight="bold"
        )
        ax.text(
            0.95, upperbound, "AFTER", fontsize=12, color="black", fontweight="bold"
        )

        for mode in change_df.index:
            # Color depending on the evolution
            value_before = change_df[change_df.index == mode]["policy_share"].item()
            value_after = change_df[change_df.index == mode]["new_policy_share"].item()

            # Red if the value has decreased, green otherwise
            if value_before > value_after:
                color = "red"
            else:
                color = "green"

            # Add the line to the plot
            ax.plot(
                [0, 1],
                change_df.loc[mode][["policy_share", "new_policy_share"]],
                marker="o",
                label=mode,
                color=color,
            )

        for mode in change_df.index:
            for metric in ["policy_share", "new_policy_share"]:
                y_position = np.round(change_df.loc[mode][metric], 2)
                if metric == "policy_share":
                    x_position = 0 - 0.12
                else:
                    x_position = 1 + 0.02
                ax.text(
                    x_position,
                    y_position,
                    f"{mode}, {y_position}",
                    fontsize=8,  # Text size
                    color="black",  # Text color
                )
        ax.set_xticks([])
        ax.set_ylabel("Share of Market %")
        ax.set_xlabel("Before/After")
        ax.set_title(title)
        return fig
