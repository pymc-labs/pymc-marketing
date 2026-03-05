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
"""Mixed Logit for Product Preference Analysis with Random Coefficients."""

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
from pytensor.tensor.variable import TensorVariable

from pymc_marketing.model_builder import ModelBuilder, create_sample_kwargs
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.version import __version__

HDI_ALPHA = 0.5


class MixedLogit(ModelBuilder):
    """
    Mixed Logit (Random Parameters Logit) class.

    Class to perform a mixed logit analysis with random coefficients
    to capture heterogeneity in consumer preferences. Random coefficients
    can vary across individuals (or groups in panel data).

    Parameters
    ----------
    choice_df : pd.DataFrame
        A wide DataFrame where each row is a choice scenario. Product-specific
        attributes are stored in columns, and the dependent variable identifies
        the chosen product.

    utility_equations : list of formula strings
        A list of formulas specifying how to model the utility of
        each product alternative. The formulas should be in Wilkinson
        style notation with three parts separated by |:
        target_product ~ alt_specific_covariates | fixed_covariates | random_covariates

    depvar : str
        The name of the dependent variable in the choice_df.

    covariates : list of str
        Covariate names (e.g., ['price', 'time', 'comfort'])

    model_config : dict, optional
        Model configuration. If None, the default config is used.

    sampler_config : dict, optional
        Sampler configuration. If None, the default config is used.

    group_id : str, optional
        Column name for group identifier (for panel data). If None,
        each observation is treated as a unique individual.

    instrumental_vars : dict, optional
        Dictionary specifying instrumental variables for endogenous price:
        {'X_instruments': np.ndarray, 'y_price': np.ndarray, 'diagonal': bool}
        If None, no control function is used.

    non_centered : bool, optional
        Whether to use non-centered parameterization for random coefficients.

    Notes
    -----
    Example:
    -------
    The format of `choice_df`:

        +------------+------------+------------+------------+------------+
        | choice     | bus_price  | bus_time   | car_price  | car_time   |
        +============+============+============+============+============+
        | bus        | 2.4        | 45         | 5.4        | 30         |
        +------------+------------+------------+------------+------------+
        | car        | 3.5        | 50         | 2.3        | 25         |
        +------------+------------+------------+------------+------------+

    Example `utility_equations` list:

    .. code-block:: python

        utility_equations = [
            "bus ~ bus_price + bus_time | income | bus_price",
            "car ~ car_price + car_time | income | car_price",
            "train ~ train_price + train_time | income | train_price",
        ]

    This specifies:
    - Alternative-specific: price and time for each mode
    - Fixed across alternatives: income (with alternative-specific coefficients)
    - Random coefficients: price varies across individuals

    """

    _model_type = "Mixed Logit Model"
    version = "0.1.0"

    def __init__(
        self,
        choice_df: pd.DataFrame,
        utility_equations: list[str],
        depvar: str,
        covariates: list[str],
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        group_id: str | None = None,
        instrumental_vars: dict | None = None,
        non_centered: bool = True,
    ):
        self.choice_df = choice_df
        self.utility_equations = utility_equations
        self.depvar = depvar
        self.covariates = covariates
        self.group_id = group_id
        self.instrumental_vars = instrumental_vars
        self.non_centered = non_centered

        model_config = model_config or {}
        model_config = parse_model_config(model_config)

        super().__init__(model_config=model_config, sampler_config=sampler_config)

    @property
    def default_model_config(self) -> dict:
        """Default model configuration.

        This includes priors for:
        - Alternative-specific constants (alphas)
        - Fixed coefficients (betas_fixed)
        - Non-random alternative-specific coefficients (betas_non_random)
        - Random coefficient means (mu_random)
        - Random coefficient standard deviations (sigma_random)
        - Control function parameters (if using instrumental variables)

        Returns
        -------
        dict
            The default model configuration.

        """
        alphas = Prior("Normal", mu=0, sigma=5, dims="alts")
        betas_fixed = Prior("Normal", mu=0, sigma=1, dims=("alts", "fixed_covariates"))
        betas_non_random = Prior("Normal", mu=0, sigma=1, dims="normal_covariates")

        # Random coefficient priors (population level)
        mu_random = Prior("Normal", mu=0, sigma=0.5, dims="random_covariates")
        sigma_random = Prior("Exponential", lam=1, dims="random_covariates")

        # Individual-level random coefficients (hierarchical)
        betas_random_individual = Prior(
            "Normal",
            mu=0,  # Will be set to mu_random in make_random_coefs
            sigma=1,  # Will be set to sigma_random in make_random_coefs
            dims=("individuals", "random_covariates"),
        )

        # Control function parameters (for price endogeneity)
        gamma = Prior("Normal", mu=0, sigma=5, dims=("instruments", "alts"))
        lambda_cf = Prior("Normal", mu=0, sigma=1, dims="alts")
        sigma_eta = Prior("HalfNormal", sigma=1)

        return {
            "alphas_": alphas,
            "betas_fixed_": betas_fixed,
            "betas_non_random": betas_non_random,
            "mu_random": mu_random,
            "sigma_random": sigma_random,
            "betas_random_individual": betas_random_individual,
            "gamma": gamma,
            "lambda_cf": lambda_cf,
            "sigma_eta": sigma_eta,
            "likelihood": Prior(
                "Categorical",
                p=0,
                dims="obs",
            ),
        }

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration."""
        return {
            "nuts_sampler": "numpyro",
            "idata_kwargs": {"log_likelihood": True},
        }

    @property
    def output_var(self) -> str:
        """The output variable of the model."""
        return "y"

    @property
    def _serializable_model_config(self) -> dict[str, int | float | dict]:
        result: dict[str, int | float | dict] = {
            "alphas_": self.model_config["alphas_"].to_dict(),
            "betas_fixed_": self.model_config["betas_fixed_"].to_dict(),
            "betas_non_random": self.model_config["betas_non_random"].to_dict(),
            "mu_random": self.model_config["mu_random"].to_dict(),
            "sigma_random": self.model_config["sigma_random"].to_dict(),
            "betas_random_individual": self.model_config[
                "betas_random_individual"
            ].to_dict(),
            "likelihood": self.model_config["likelihood"].to_dict(),
        }

        if self.instrumental_vars is not None:
            result["gamma"] = self.model_config["gamma"].to_dict()
            result["lambda_cf"] = self.model_config["lambda_cf"].to_dict()
            result["sigma_eta"] = self.model_config["sigma_eta"].to_dict()

        return result

    @staticmethod
    def _check_columns(
        df: pd.DataFrame,
        fixed_covariates: str,
        alt_covariates: str,
        random_covariates: str,
    ) -> None:
        """Validate that all specified covariates exist in the dataframe."""
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

        for r in random_covariates.split("+"):
            if r.strip() and r.strip() not in df.columns:
                raise ValueError(
                    f"Random covariate '{r.strip()}' not found in dataframe columns."
                )

    @staticmethod
    def _check_dependent_variable(target: str, df: pd.DataFrame, depvar: str) -> None:
        """Validate that the target alternative exists in the dependent variable."""
        if target not in df[depvar].unique():
            raise ValueError(
                f"Target '{target}' not found in dependent variable '{depvar}'."
            )

    def parse_formula(
        self, df: pd.DataFrame, formula: str, depvar: str
    ) -> tuple[str, str, str, str]:
        """Parse the three-part structure of a mixed logit formula specification.

        Splits the formula into target, alternative-specific covariates,
        fixed covariates, and random covariates. Ensures that the target
        variable appears in the dependent variable column and that all
        specified covariates exist in the input dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        formula : str
            Formula string in the format:
            "target ~ alt_covariates | fixed_covariates | random_covariates"
        depvar : str
            Name of dependent variable column

        Returns
        -------
        tuple[str, str, str, str]
            (target, alt_covariates, fixed_covariates, random_covariates)
        """
        target, covariates = formula.split("~")
        target = target.strip()

        parts = covariates.split("|")

        if len(parts) == 1:
            # No fixed or random covariates
            alt_covariates = parts[0].strip()
            fixed_covariates = ""
            random_covariates = ""
        elif len(parts) == 2:
            # Has fixed covariates, no random
            alt_covariates = parts[0].strip()
            fixed_covariates = parts[1].strip()
            random_covariates = ""
        elif len(parts) == 3:
            # Has both fixed and random covariates
            alt_covariates = parts[0].strip()
            fixed_covariates = parts[1].strip()
            random_covariates = parts[2].strip()
        else:
            raise ValueError(
                f"Formula '{formula}' has too many '|' separators. "
                "Expected format: target ~ alt_covs | fixed_covs | random_covs"
            )

        self._check_columns(df, fixed_covariates, alt_covariates, random_covariates)
        self._check_dependent_variable(target, df, depvar)

        return target, alt_covariates, fixed_covariates, random_covariates

    def prepare_X_matrix(
        self, df: pd.DataFrame, utility_formulas: list[str], depvar: str
    ) -> tuple[np.ndarray, np.ndarray | None, list[str], np.ndarray, list[str]]:
        """Prepare the design matrices for the utility equations.

        Creates:
        - X: Alternative-specific covariates (N x J x K)
        - F: Fixed covariates (N x K_fixed) or None
        - List of alternative names
        - Array of unique fixed covariate names
        - List of random covariate names

        Returns
        -------
        tuple
            (X, F, alternatives, fixed_covar_names, random_covar_names)
        """
        n_obs = len(df)
        n_alts = len(utility_formulas)
        n_covariates = len(utility_formulas[0].split("|")[0].split("+"))

        alts = []
        alt_covariates = []
        fixed_covariates = []
        random_covariates_set = set()

        for f in utility_formulas:
            target, alt_covar, fixed_covar, random_covar = self.parse_formula(
                df, f, depvar
            )
            f_patsy = "0 + " + alt_covar
            alt_covariates.append(np.asarray(patsy.dmatrix(f_patsy, df)).T)
            alts.append(target)

            if fixed_covar:
                fixed_covariates.append(fixed_covar)

            if random_covar:
                # Extract individual covariate names from the random specification
                for cov_name in random_covar.split("+"):
                    cov_name = cov_name.strip()
                    if cov_name:
                        # Extract the base covariate name (e.g., "price" from "bus_price")
                        # by removing the alternative prefix
                        for base_cov in self.covariates:
                            if base_cov in cov_name:
                                random_covariates_set.add(base_cov)
                                break

        # Check all alternatives in data have equations
        for alt in df[depvar].unique():
            if alt not in alts:
                raise ValueError(
                    f"Alternative '{alt}' appears in data but has no utility equation"
                )

        # Process fixed covariates
        if fixed_covariates:
            F = np.unique(fixed_covariates)[0]
            F = "0 + " + F
            F = np.asarray(patsy.dmatrix(F, df))
        else:
            F = None

        # Convert set to list preserving order from self.covariates
        random_covar_names = [c for c in self.covariates if c in random_covariates_set]

        X = np.stack(alt_covariates, axis=1).T
        if X.shape != (n_obs, n_alts, n_covariates):
            raise ValueError(
                f"X has shape {X.shape}, but expected {(n_obs, n_alts, n_covariates)}."
            )

        return X, F, alts, np.unique(fixed_covariates), random_covar_names

    @staticmethod
    def _prepare_y_outcome(
        df: pd.DataFrame, alternatives: list[str], depvar: str
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Encode the outcome category variable for use in modelling.

        The order of the alternatives should map to the order of the
        utility formulas.
        """
        alt_mapping = dict(zip(alternatives, range(len(alternatives)), strict=False))
        df["alt_encoded"] = df[depvar].map(alt_mapping)
        y = np.asarray(df["alt_encoded"].values)
        return y, alt_mapping

    def _prepare_group_index(
        self, df: pd.DataFrame, group_id: str | None
    ) -> tuple[np.ndarray | None, int]:
        """Prepare group index array for panel data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        group_id : str or None
            Column name for group identifier

        Returns
        -------
        tuple[np.ndarray | None, int]
            (group_index_array, n_groups)
        """
        if group_id is None:
            # Each observation is unique individual
            return None, len(df)

        if group_id not in df.columns:
            raise ValueError(f"Group ID column '{group_id}' not found in dataframe")

        # Create mapping from group IDs to indices
        unique_groups = df[group_id].unique()
        group_mapping = dict(
            zip(unique_groups, range(len(unique_groups)), strict=False)
        )
        grp_idx = np.asarray(df[group_id].map(group_mapping).values)

        return grp_idx, len(unique_groups)

    def _prepare_coords(
        self,
        df: pd.DataFrame,
        alternatives: list[str],
        covariates: list[str],
        f_covariates: np.ndarray,
        random_covar_names: list[str],
        n_individuals: int,
    ) -> dict:
        """Prepare coordinates for PyMC mixed logit model.

        Parameters
        ----------
        df : pd.DataFrame
            Choice dataframe
        alternatives : list[str]
            List of all alternatives
        covariates : list[str]
            List of covariate names
        f_covariates : np.ndarray
            Fixed covariate names
        random_covar_names : list[str]
            Names of covariates with random coefficients
        n_individuals : int
            Number of unique individuals (or groups)

        Returns
        -------
        dict
            Coordinate dictionary for PyMC model
        """
        if isinstance(f_covariates, np.ndarray) and f_covariates.size > 0:
            f_cov = [s.strip() for s in f_covariates[0].split("+")]
        else:
            f_cov = []

        # Identify normal (non-random) covariates
        normal_covar_names = [c for c in covariates if c not in random_covar_names]

        coords = {
            "alts": alternatives,
            "alts_probs": alternatives[:-1],
            "covariates": covariates,
            "normal_covariates": normal_covar_names,
            "random_covariates": random_covar_names,
            "fixed_covariates": f_cov,
            "obs": range(len(df)),
            "individuals": range(n_individuals),
        }

        if self.instrumental_vars is not None:
            instruments = self.instrumental_vars["X_instruments"].shape[1]
            n_inst_per_alt = instruments // len(alternatives)
            coords["instruments"] = [f"inst_{i}" for i in range(n_inst_per_alt)]

        return coords

    def preprocess_model_data(
        self, choice_df: pd.DataFrame, utility_equations: list[str]
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Pre-process the model initiation inputs into PyMC-ready format.

        This method prepares:
        - X: 3D design matrix (n_obs, n_alts, n_covariates)
        - F: Fixed covariate matrix (n_obs, n_fixed_covariates) or None
        - y: Encoded response vector (n_obs,)
        - group_idx: Group membership array (n_obs,) or None

        Also extracts and stores metadata including alternatives, covariate names,
        and coordinate dimensions.

        Parameters
        ----------
        choice_df : pd.DataFrame
            DataFrame containing choice observations and covariates
        utility_equations : list[str]
            List of utility formulas, one per alternative

        Returns
        -------
        tuple
            (X, F, y) ready for model building
        """
        X, F, alternatives, fixed_covar, random_covar_names = self.prepare_X_matrix(
            choice_df, utility_equations, self.depvar
        )

        self.X = X
        self.F = F
        self.alternatives = alternatives
        self.fixed_covar = fixed_covar
        self.random_covar_names = random_covar_names

        # Identify indices of random covariates
        self.random_covariate_idx = [
            i for i, cov in enumerate(self.covariates) if cov in random_covar_names
        ]

        y, alt_mapping = self._prepare_y_outcome(
            choice_df, self.alternatives, self.depvar
        )
        self.y = y
        self.alt_mapping = alt_mapping

        # Handle group structure for panel data
        grp_idx, n_individuals = self._prepare_group_index(choice_df, self.group_id)
        self.grp_idx = grp_idx
        self.n_individuals = n_individuals

        # Prepare coordinates
        self.coords: dict[str, list[str]] = self._prepare_coords(
            choice_df,
            self.alternatives,
            self.covariates,
            self.fixed_covar,
            self.random_covar_names,
            self.n_individuals,
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

    def make_intercepts(self) -> TensorVariable:
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

    def make_fixed_coefs(
        self, X_fixed: np.ndarray | None, n_obs: int, n_alts: int
    ) -> TensorVariable:
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
            betas_fixed = pm.Deterministic(
                "betas_fixed",
                pt.set_subtensor(betas_fixed_[-1, :], 0),
                dims=("alts", "fixed_covariates"),
            )
            W_contrib = pm.Deterministic(
                "W_interaction",
                pm.math.dot(W_data, betas_fixed.T),
                dims=("obs", "alts"),
            )
        return W_contrib

    def make_non_random_coefs(self) -> TensorVariable | None:
        """Create coefficients for non-random alternative-specific covariates.

        These are standard fixed coefficients that don't vary across individuals.

        Returns
        -------
        betas_non_random : TensorVariable or None
            Coefficients for non-random covariates, or None if all are random
        """
        if len(self.coords["normal_covariates"]) == 0:
            return None

        betas_non_random = self.model_config["betas_non_random"].create_variable(
            "betas_non_random"
        )
        return betas_non_random

    def make_random_coefs(
        self, n_obs: int, grp_idx: np.ndarray | None = None, non_centered: bool = True
    ) -> tuple[TensorVariable | None, list[str]]:
        """Create random coefficients that vary across individuals.

        For each covariate specified as random, this creates:
        1. Population-level mean (mu_random)
        2. Population-level standard deviation (sigma_random)
        3. Individual/group-level deviations (betas_random_individual)

        Parameters
        ----------
        n_obs : int
            Number of observations
        grp_idx : np.ndarray or None
            Group index array for panel data (maps observations to groups)

        Returns
        -------
        tuple
            (betas_random_expanded, random_param_names)
            - betas_random_expanded: shape (n_obs, n_random_covariates) or None
            - random_param_names: list of random covariate names
        """
        if len(self.random_covariate_idx) == 0:
            return None, []

        # Population-level parameters
        mu_random = self.model_config["mu_random"].create_variable("mu_random")
        sigma_random = self.model_config["sigma_random"].create_variable("sigma_random")

        # ---------- PARAMETERIZATION SWITCH ----------
        if non_centered:
            # -------- NON-CENTRED --------
            if grp_idx is None:
                z = pm.Normal(
                    "z_random",
                    0.0,
                    1.0,
                    dims=("obs", "random_covariates"),
                )

                betas_random = pm.Deterministic(
                    "betas_random_individual",
                    mu_random + z * sigma_random,
                    dims=("obs", "random_covariates"),
                )

            else:
                z_grp = pm.Normal(
                    "z_random_group",
                    0.0,
                    1.0,
                    dims=("individuals", "random_covariates"),
                )

                betas_random_grp = pm.Deterministic(
                    "betas_random_group",
                    mu_random + z_grp * sigma_random,
                    dims=("individuals", "random_covariates"),
                )

                betas_random = pm.Deterministic(
                    "betas_random_individual",
                    betas_random_grp[grp_idx],
                    dims=("obs", "random_covariates"),
                )

        else:
            # -------- CENTRED --------
            if grp_idx is None:
                betas_random = pm.Normal(
                    "betas_random_individual",
                    mu=mu_random,
                    sigma=sigma_random,
                    dims=("obs", "random_covariates"),
                )

            else:
                betas_random_grp = pm.Normal(
                    "betas_random_group",
                    mu=mu_random,
                    sigma=sigma_random,
                    dims=("individuals", "random_covariates"),
                )

                betas_random = pm.Deterministic(
                    "betas_random_individual",
                    betas_random_grp[grp_idx],
                    dims=("obs", "random_covariates"),
                )

        return betas_random, self.random_covar_names

    def make_beta_matrix(
        self,
        betas_non_random: TensorVariable | None,
        betas_random: TensorVariable | None,
        n_obs: int,
    ) -> TensorVariable:
        """Combine random and non-random coefficients into full coefficient matrix.

        Creates a (n_obs, n_covariates) matrix where:
        - Random coefficients vary across observations
        - Non-random coefficients are constant across observations

        Parameters
        ----------
        betas_non_random : TensorVariable or None
            Non-random coefficients, shape (n_normal_covariates,)
        betas_random : TensorVariable or None
            Random coefficients, shape (n_obs, n_random_covariates)
        n_obs : int
            Number of observations

        Returns
        -------
        B_full : TensorVariable
            Full coefficient matrix, shape (n_obs, n_covariates)
        """
        n_covariates = len(self.covariates)
        B_full = pt.zeros((n_obs, n_covariates))

        # Fill in random coefficients
        if betas_random is not None:
            for i, cov_idx in enumerate(self.random_covariate_idx):
                B_full = pt.set_subtensor(B_full[:, cov_idx], betas_random[:, i])

        # Fill in non-random coefficients
        if betas_non_random is not None:
            non_random_idx = [
                i for i in range(n_covariates) if i not in self.random_covariate_idx
            ]
            # Broadcast non-random coefficients across all observations
            B_full = pt.set_subtensor(B_full[:, non_random_idx], betas_non_random)

        B_full = pm.Deterministic(
            "betas_individuals", B_full, dims=("obs", "covariates")
        )

        return B_full

    def make_control_function(self, n_obs: int, n_alts: int) -> TensorVariable:
        """Create control function for price endogeneity correction.

        Implements a control function approach where:
        1. Price is modeled as a function of instruments
        2. Price errors are computed
        3. Price errors are included in utility with correlation parameter

        Parameters
        ----------
        n_obs : int
            Number of observations
        n_alts : int
            Number of alternatives

        Returns
        -------
        price_error_contrib : TensorVariable
            Contribution to utility from price endogeneity, shape (n_obs, n_alts)
        """
        if self.instrumental_vars is None:
            return pt.zeros((n_obs, n_alts))

        X_instruments = self.instrumental_vars["X_instruments"]
        y_price = self.instrumental_vars["y_price"]
        diagonal = self.instrumental_vars.get("diagonal", True)

        X_inst_data = pm.Data("X_instruments", X_instruments)
        y_price_data = pm.Data("y_price", y_price)
        n_inst_per_alt = X_instruments.shape[1] // n_alts  # 8 / 4 = 2

        sigma_eta = self.model_config["sigma_eta"].create_variable("sigma_eta")

        # First stage: model price as function of instruments
        if diagonal:
            gamma = self.model_config["gamma"].create_variable("gamma")
            gamma_0 = pm.Normal("gamma_0", 0, 10, shape=n_alts)  # Price intercepts
            gamma_0 = pt.set_subtensor(gamma_0[-1], 0)  # Reference alt intercept = 0
            X_inst = X_inst_data.reshape((n_obs, n_alts, n_inst_per_alt))  # (N, J, K)
            mu_P = gamma_0 + pt.sum(X_inst * gamma.T, axis=2)  # (N, J)
        else:
            gamma = self.model_config["gamma"].create_variable("gamma")
            gamma_0 = pm.Normal("gamma_0", 0, 10, shape=n_alts)  # Price intercepts
            gamma_0 = pt.set_subtensor(gamma_0[-1], 0)  # Reference alt intercept = 0
            mu_P = gamma_0 + pt.dot(X_inst_data, gamma)

        # Price likelihood (first stage)
        _ = pm.Normal("P_obs", mu_P, sigma_eta, observed=y_price_data)
        raw_residual = y_price_data - mu_P
        # Compute price errors (residuals)
        price_error = pm.Deterministic(
            "price_error",
            raw_residual - raw_residual.mean(axis=0),
            dims=("obs", "alts"),
        )

        # Control function: include correlated errors in utility
        lambda_cf = self.model_config["lambda_cf"].create_variable("lambda_cf")
        price_error_contrib = lambda_cf * price_error
        price_error_contrib = pm.Deterministic(
            "price_error_contrib", price_error_contrib, dims=("obs", "alts")
        )

        return price_error_contrib

    def make_utility(
        self,
        X_data: TensorVariable,
        B_full: TensorVariable,
        alphas: TensorVariable,
        W_contrib: TensorVariable,
        price_error_contrib: TensorVariable,
    ) -> TensorVariable:
        """Compute total systematic utility for each alternative.

        Combines contributions from:
        - Alternative-specific covariates with individual-specific coefficients
        - Fixed covariates with alternative-specific effects
        - Alternative-specific constants
        - Price endogeneity correction (if applicable)

        Parameters
        ----------
        X_data : TensorVariable
            Alternative-specific covariates, shape (n_obs, n_alts, n_covariates)
        B_full : TensorVariable
            Individual coefficient matrix, shape (n_obs, n_covariates)
        alphas : TensorVariable
            Alternative-specific constants
        W_contrib : TensorVariable
            Fixed covariate contribution, shape (n_obs, n_alts)
        price_error_contrib : TensorVariable
            Control function contribution, shape (n_obs, n_alts)

        Returns
        -------
        U : TensorVariable
            Systematic utility, shape (n_obs, n_alts)
        """
        # X_data: (n_obs, n_alts, n_covariates)
        # B_full: (n_obs, n_covariates)
        # Expand B_full to (n_obs, 1, n_covariates) for broadcasting
        # Result: (n_obs, n_alts)
        U_cov = pt.sum(X_data * B_full[:, None, :], axis=2)

        # Combine all utility components
        U = pm.Deterministic(
            "U", U_cov + W_contrib + alphas + price_error_contrib, dims=("obs", "alts")
        )
        return U

    def make_choice_prob(self, U: TensorVariable) -> TensorVariable:
        """Compute choice probabilities via softmax transformation.

        Parameters
        ----------
        U : TensorVariable
            Systematic utility, shape (n_obs, n_alts)

        Returns
        -------
        p : TensorVariable
            Choice probabilities, shape (n_obs, n_alts)
        """
        U_centered = U - U.max(axis=1, keepdims=True)
        p = pm.Deterministic(
            "p", pm.math.softmax(U_centered, axis=1), dims=("obs", "alts")
        )
        return p

    def make_model(
        self, X: np.ndarray, F: np.ndarray | None, y: np.ndarray, observed: bool = True
    ) -> pm.Model:
        """Build mixed logit model with random coefficients.

        Parameters
        ----------
        X : np.ndarray
            Alternative-specific covariates, shape (n_obs, n_alts, n_covariates)
        F : np.ndarray or None
            Fixed covariates, shape (n_obs, n_fixed_covariates)
        y : np.ndarray
            Observed choices, shape (n_obs,)
        observed: bool
            Whether to include observed data in the model

        Returns
        -------
        model : pm.Model
            PyMC model
        """
        with pm.Model(coords=self.coords) as model:
            # Instantiate data
            X_data = pm.Data("X", X, dims=("obs", "alts", "covariates"))
            observed_data = pm.Data("y", y, dims="obs")
            if self.grp_idx is not None:
                grp_idx_data = pm.Data("grp_idx", self.grp_idx, dims="obs")
            else:
                grp_idx_data = self.grp_idx
            n_obs, n_alts = X_data.shape[0], X_data.shape[1]

            # Create parameters
            alphas = self.make_intercepts()
            betas_non_random = self.make_non_random_coefs()
            betas_random, _ = self.make_random_coefs(
                n_obs, grp_idx_data, self.non_centered
            )

            # Build coefficient matrix (combines random and non-random)
            B_full = self.make_beta_matrix(betas_non_random, betas_random, n_obs)

            # Handle fixed covariates
            W_contrib = self.make_fixed_coefs(F, n_obs, n_alts)

            # Control function for price endogeneity (if applicable)
            price_error_contrib = self.make_control_function(n_obs, n_alts)

            # Compute utility and probabilities
            U = self.make_utility(
                X_data, B_full, alphas, W_contrib, price_error_contrib
            )
            p = self.make_choice_prob(U)

            # Likelihood
            if observed:
                _ = pm.Categorical(
                    "likelihood", p=p, observed=observed_data, dims="obs"
                )
            else:
                _ = pm.Categorical("likelihood", p=p, dims="obs")

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
        attrs["random_covariates"] = json.dumps(self.random_covar_names)
        attrs["depvar"] = json.dumps(self.depvar)
        attrs["choice_df"] = json.dumps("Placeholder for DF")
        attrs["utility_equations"] = json.dumps(self.utility_equations)
        attrs["group_id"] = json.dumps(self.group_id)
        attrs["instrumental_vars"] = json.dumps(
            "Placeholder for instrumental_vars" if self.instrumental_vars else None
        )
        attrs["non_centered"] = json.dumps(self.non_centered)

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
        fit_kwargs = fit_kwargs or self.default_sampler_config
        sample_posterior_predictive_kwargs = sample_posterior_predictive_kwargs or {}

        if not hasattr(self, "model"):
            X, F, y = self.preprocess_model_data(self.choice_df, self.utility_equations)
            model = self.make_model(X, F, y)
            self.model = model

        self.sample_prior_predictive(
            extend_idata=True, **sample_prior_predictive_kwargs
        )
        self.fit(extend_idata=True, kwargs=fit_kwargs)
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
        """Apply intervention by changing observable attributes or market structure.

        This method supports two intervention strategies:

        1. **Observable attribute changes**: Uses the existing fitted model and modifies
        observable inputs (e.g., prices, features) to simulate how choices change.
        Uses posterior predictive sampling with the existing random coefficient draws.

        2. **Product removal/addition**: Allows alternatives to be added or removed from
        the choice set. This re-specifies and re-estimates the model.

        Parameters
        ----------
        new_choice_df : pd.DataFrame
            The new dataset reflecting changes to observable attributes or product
            availability.

        new_utility_equations : list[str] | None, optional
            An updated list of utility specifications for each alternative, if different
            from the original model. If None, the original equations are reused and only
            the data is changed.

        fit_kwargs : dict | None, optional
            Keyword arguments for sampling if refitting the model.

        Returns
        -------
        az.InferenceData
            The posterior or full predictive distribution under the intervention.
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
            # Intervention type 1: Change observable attributes only
            new_X, new_F, new_y = self.preprocess_model_data(
                new_choice_df, self.utility_equations
            )

            with self.model:
                data_dict = {"X": new_X, "y": new_y}
                if new_F is not None:
                    data_dict["W"] = new_F
                pm.set_data(data_dict)

                # Use existing random coefficient draws for counterfactual
                idata_new_policy = pm.sample_posterior_predictive(
                    self.idata,
                    var_names=["p", "likelihood"],
                    return_inferencedata=True,
                    extend_inferencedata=False,
                    random_seed=100,
                )

            self.intervention_idata = idata_new_policy
        else:
            # Intervention type 2: Change market structure (refit model)
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
        """Calculate difference in market share due to intervention.

        Parameters
        ----------
        idata : az.InferenceData
            Posterior predictive samples under baseline policy.
            Must contain a "posterior_predictive" group with "p" variable.

        new_idata : az.InferenceData
            Posterior predictive samples under new policy.
            Structure should match `idata`.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by alternative, containing:
            - 'policy_share': mean predicted share under baseline
            - 'new_policy_share': mean predicted share under new policy
            - 'relative_change': relative change in share
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
        title: str = "Change in Market Share due to Intervention",
        figsize: tuple = (8, 4),
    ) -> plt.Figure:
        """Plot change induced by a market intervention.

        Parameters
        ----------
        change_df : pd.DataFrame
            DataFrame indexed by product with columns:
            - 'policy_share': share before intervention
            - 'new_policy_share': share after intervention

        title : str, optional
            Title of the plot.

        figsize : tuple, optional
            Size of the plot in inches, as (width, height).

        Returns
        -------
        matplotlib.figure.Figure
            Figure showing before/after market share comparisons.
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

        for product in change_df.index:
            value_before = change_df[change_df.index == product]["policy_share"].item()
            value_after = change_df[change_df.index == product][
                "new_policy_share"
            ].item()

            # Red if decreased, green if increased
            color = "red" if value_before > value_after else "green"

            ax.plot(
                [0, 1],
                change_df.loc[product][["policy_share", "new_policy_share"]],
                marker="o",
                label=product,
                color=color,
            )

        for product in change_df.index:
            for metric in ["policy_share", "new_policy_share"]:
                y_position = np.round(change_df.loc[product][metric], 2)
                x_position = 0 - 0.12 if metric == "policy_share" else 1 + 0.02
                ax.text(
                    x_position,
                    y_position,
                    f"{product}, {y_position}",
                    fontsize=8,
                    color="black",
                )

        ax.set_xticks([])
        ax.set_ylabel("Share of Market %")
        ax.set_xlabel("Before/After")
        ax.set_title(title)
        return fig
