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
"""Nested Logit for Product Preference Analysis."""

import json
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
from typing_extensions import Self

from pymc_marketing.model_builder import ModelBuilder
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.prior import Prior

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
        Dictionary to specify how to nest the choices between products

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
    ...     "alt_1 ~ X1_alt1 + X2_alt1 | income",
    ...     "alt_2 ~ X1_alt2 + X2_alt2 | income",
    ...     "alt_3 ~ X1_alt3 + X2_alt3 | income",
    ... ]

    Example nesting structure:

    >>> nesting_structure = {
    ...     "Nest1": ["alt1"],
    ...     "Nest2": {"Nest2_1": ["alt_2", "alt_3"], "Nest_2_2": ["alt_4", "alt_5"]},
    ... }

    """

    _model_type = "Nested Logit Model"
    version = "0.1.0"

    def __init__(
        self,
        choice_df: pd.DataFrame,
        utility_equations: list[str],
        depvar: str,
        covariates: list[str],
        nesting_structure: dict,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        self.choice_df = choice_df
        self.utility_equations = utility_equations
        self.depvar = depvar
        self.covariates = covariates
        self.nesting_structure = nesting_structure

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
        betas_fixed = Prior("Normal", mu=0, sigma=1, dims="fixed_covariates")
        lambdas_nests = Prior("Beta", alpha=1, beta=1, dims="nests")

        return {
            "alphas_": alphas,
            "betas": betas,
            "betas_fixed_": betas_fixed,
            "lambdas_nests": lambdas_nests,
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
        nest_dict: dict[str, dict[str, list[str]] | list[str]],
        product_indices: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
        if not nest_dict:
            raise ValueError("Nesting structure must not be empty.")

        top_level: dict[str, np.ndarray] = {}
        mid_level: dict[str, list[int]] = {}

        for k in nest_dict.keys():
            value = nest_dict[k]
            if isinstance(value, dict):
                collected_idxs: list[int] = []
                for j in value:
                    inner_value = value[j]
                    if isinstance(inner_value, dict):
                        raise ValueError("Cannot have more than 2 layers of Nesting")
                    else:
                        inner_list: list[str] = inner_value  # for Mypy
                        idxs = [product_indices[i] for i in inner_list]
                        mid_level[k + "_" + j] = idxs
                        collected_idxs += idxs
                top_level[k] = np.sort(collected_idxs)
            else:
                alt_list: list[str] = value  # for Mypy
                top_level[k] = np.sort([product_indices[i] for i in alt_list])

        mid_level_result: dict[str, np.ndarray] | None = None
        if mid_level:
            mid_level_result = {k: np.array(v) for k, v in mid_level.items()}

        return top_level, mid_level_result

    @staticmethod
    def _prepare_coords(df, alternatives, covariates, f_covariates, nests):
        """Prepare coordinates for PyMC nested logit model."""
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
        - Handles multi-level nesting structures if provided in `self.nesting_structure`.
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
        top_level, mid_level = self._parse_nesting(
            self.nesting_structure, self.prod_indices
        )
        if mid_level:
            all_nests = list(top_level.keys()) + list(mid_level.keys())
            self.nest_indices = {"top": top_level, "mid": mid_level}
        else:
            all_nests = list(top_level.keys())
            self.nest_indices = {"top": top_level}

        self.all_nests = all_nests
        self.lambda_lkup = dict(zip(all_nests, range(len(all_nests)), strict=False))

        # note: type hints for coords required for mypy to not get confused
        self.coords: dict[str, list[str]] = self._prepare_coords(
            choice_df,
            self.alternatives,
            self.covariates,
            self.fixed_covar,
            self.all_nests,
        )

        return X, F, y

    def build_model(self, X, y, **kwargs):
        """Do not use, required by parent class. Prefer make_model()."""
        return super().build_model(X, y, **kwargs)

    def make_exp_nest(
        self,
        U: TensorVariable,
        W: TensorVariable | None,
        betas_fixed: TensorVariable,
        lambdas_nests: TensorVariable,
        nest: str,
        level: str = "top",
    ) -> tuple[TensorVariable, TensorVariable]:
        r"""
        Calculate within-nest probabilities for nested logit models.

        This function recursively computes the utility aggregates used to build a
        nested logit model within a PyMC probabilistic framework. Specifically, it calculates:

        1. **Conditional choice probabilities within a nest**:

        $$
        P(y_i = j \mid j \in \text{nest}) =
        \frac{\exp\left( \frac{U_{ij}}{\lambda} \right)}
                {\sum_{j \in \text{nest}} \exp\left( \frac{U_{ij}}{\lambda} \right)}
        $$

        This is a softmax probability scaled by the nest-specific temperature
        (scale) parameter \\( \lambda \\).

        2. **Inclusive value (or log-sum utility)**:

        $$
        I_{\text{nest}}(i) = \lambda \cdot \log \left(
        \sum_{j \in \text{nest}} \exp \left( \frac{U_{ij}}{\lambda} \right) \right)
        $$

        This quantity represents the “meta-utility” of a nest, passed up the
        hierarchy in nested logit models.

        3. **Exponentiated meta-utility**:

        An exponentiated term combining inclusive value and fixed covariate
        contributions, used when computing choice probabilities in the parent nest.

        Parameters
        ----------
        U : TensorVariable
            Tensor of shape (N, J), where N is the number of observations and J is the
            number of alternatives. Represents latent utilities.

        W : TensorVariable or None
            Optional tensor of shape (N, K), where K is the number of fixed covariates.
            Represents covariate contributions that do not vary across alternatives.

        betas_fixed : TensorVariable
            Tensor of shape (J, K), with one coefficient vector per alternative for
            fixed (non-alternative-varying) covariates.

        lambdas_nests : TensorVariable
            A tensor containing the nest-specific scale parameters \\( \lambda \\),
            typically modeled with a Beta distribution.

        nest : str
            Name of the current nest to process (e.g., `"Land"` or `"Land_Car"`).
            Determines which subset of alternatives belongs to the nest.

        level : str, default="top"
            Either `"top"` or `"mid"`, indicating the level of the nest in the
            hierarchical structure. Used to select the correct index mapping.

        Returns
        -------
        exp_W_nest : TensorVariable
            Exponentiated meta-utility for the current nest, used in the parent nest’s
            softmax normalization.

        P_y_given_nest : TensorVariable
            Conditional probability of choosing each alternative within the current nest.

        Notes
        -----
        This function supports two-level nested logit models, where alternatives are
        grouped into mutually exclusive nests. The scale parameter \\( \lambda \\)
        controls the degree of substitutability within each nest.

        Currently, deeper nesting levels (more than two) are not supported, to simplify
        both modeling and computation.
        """
        nest_indices = self.nest_indices
        lambda_lkup = self.lambda_lkup
        N = U.shape[0]
        if "_" in nest:
            parent, child = nest.split("_")
        else:
            parent = None
        y_nest = U[:, nest_indices[level][nest]]
        if W is None:
            w_nest = pm.math.zeros((N, len(self.alternatives)))
        else:
            betas_fixed_temp = betas_fixed[nest_indices[level][nest], :]
            betas_fixed_temp = pt.set_subtensor(betas_fixed_temp[-1], 0)
            w_nest = pm.math.dot(W, betas_fixed_temp.T)

        if len(nest_indices[level][nest]) > 1:
            max_y_nest = pm.math.max(y_nest, axis=0)
            P_y_given_nest = pm.Deterministic(
                f"p_y_given_{nest}",
                pm.math.softmax(y_nest / lambdas_nests[lambda_lkup[nest]], axis=1),
            )
        else:
            max_y_nest = pm.math.max(y_nest)
            ones = pm.math.ones((N, 1))
            P_y_given_nest = pm.Deterministic(f"p_y_given_{nest}", ones)
        if parent is None:
            lambda_ = lambdas_nests[lambda_lkup[nest]]
            I_nest = pm.Deterministic(
                f"I_{nest}", pm.math.logsumexp((y_nest - max_y_nest) / lambda_)
            )
            W_nest = w_nest + I_nest * lambda_
        else:
            l1 = lambdas_nests[lambda_lkup[nest]]
            l2 = lambdas_nests[lambda_lkup[parent]]
            lambdas_ = l1 * l2
            I_nest = pm.Deterministic(
                f"I_{nest}", pm.math.logsumexp((y_nest - max_y_nest) / lambdas_)
            )
            W_nest = w_nest + I_nest * (lambdas_)

        exp_W_nest = pm.math.exp(W_nest)
        return exp_W_nest, P_y_given_nest

    def make_P_nest(
        self,
        U: TensorVariable,
        W: TensorVariable | None,
        betas_fixed: TensorVariable,
        lambdas_nests: TensorVariable,
        level: str,
    ) -> tuple[
        dict[str, dict[str, TensorVariable]],
        dict[str, TensorVariable],
    ]:
        """Calculate the probability of choosing a nest.

        This function collates the exponentiated inclusive value (`exp_W_nest`)
        for each alternative group (nest), sums them, and then normalizes across
        nests to obtain the probability of selecting a nest. The within-nest
        conditional probabilities (`P_y_given`) are computed in the `make_exp_nest`
        method.

        This is used within the PyMC model to construct the **tree-based aggregation**
        of utilities, where lower-level nodes (alternative-specific utilities) are
        passed upward in the tree structure to compute nest-level and top-level
        choice probabilities.

        Parameters
        ----------
        U : TensorVariable
            Tensor of systematic utilities with shape `(n_obs, n_alternatives)`.
        W : TensorVariable | None
            Fixed covariates design matrix (if used), else `None`.
        betas_fixed : TensorVariable
            Alternative-specific coefficients for the fixed covariates.
        lambdas_nests : TensorVariable
            A Beta random variable for each of the nests
        level : str
            Which nesting level to compute ("top" or "mid"), indicating whether
            to compute probabilities for top-level or mid-level nests.

        Returns
        -------
        conditional_probs : dict[str, dict[str, TensorVariable]]
            Dictionary for each nest containing:
            - "exp": the exponentiated inclusive value of the nest.
            - "P_y_given": conditional choice probabilities within the nest.
        nest_probs : dict[str, TensorVariable]
            Dictionary mapping each nest to its overall selection probability.

        Raises
        ------
        ValueError
            If the nesting structure is invalid or the nest name is not found.
        """
        nest_indices = self.nest_indices
        conditional_probs = {}
        ## Collect All Exp Inclusive Value terms per nest
        for n in nest_indices[level].keys():
            exp_W_nest, P_y_given_nest = self.make_exp_nest(
                U, W, betas_fixed, lambdas_nests, n, level
            )
            exp_W_nest = pm.math.sum(exp_W_nest, axis=1)
            conditional_probs[n] = {"exp": exp_W_nest, "P_y_given": P_y_given_nest}

        ## Sum the exp inclusive value terms as normalising constanT
        denom = pm.Deterministic(
            f"denom_{level}",
            pm.math.sum(
                [conditional_probs[n]["exp"] for n in nest_indices[level].keys()],
                axis=0,
            ),
        )
        ## Calculate the nest probability
        nest_probs = {}
        for n in nest_indices[level].keys():
            P_nest = pm.Deterministic(
                f"P_{n}", (conditional_probs[n]["exp"] / denom), dims="obs"
            )
            nest_probs[n] = P_nest
        return conditional_probs, nest_probs

    def make_model(self, X, W, y) -> pm.Model:
        """Build Model."""
        nest_indices = self.nest_indices
        coords = self.coords

        with pm.Model(coords=coords) as model:
            # alternative specific intercepts
            alphas = self.model_config["alphas_"].create_variable(name="alphas_")
            # Covariate Weight Parameters
            betas = self.model_config["betas"].create_variable("betas")
            lambdas_nests = self.model_config["lambdas_nests"].create_variable(
                "lambdas_nests"
            )
            alphas = pm.Deterministic(
                "alphas", pt.set_subtensor(alphas[-1], 0), dims="alts"
            )

            if W is None:
                betas_fixed = None
            else:
                W_data = pm.Data("W", W, dims=("obs", "fixed_covariates"))
                betas_fixed_ = self.model_config["betas_fixed_"].create_variable(
                    "betas_fixed_"
                )
                betas_fixed = pm.Deterministic(
                    "betas_fixed",
                    pt.outer(alphas, betas_fixed_),
                    dims=("alts", "fixed_covariates"),
                )
                _ = pm.Deterministic("w_nest", pm.math.dot(W_data, betas_fixed.T))
            X_data = pm.Data("X", X, dims=("obs", "alts", "alt_covariates"))
            y_data = pm.Data("y", y, dims="obs")

            # Compute utility as a dot product
            u = alphas + pm.math.dot(X_data, betas)
            U = pm.Deterministic("U", u, dims=("obs", "alts"))

            ## Mid Level
            if "mid" in nest_indices.keys():
                cond_prob_m, nest_prob_m = self.make_P_nest(
                    U, W, betas_fixed, lambdas_nests, "mid"
                )

                ## Construct Paths Bottom -> Up
                child_nests = {}
                path_prods_m: dict[str, list] = {}
                ordered = [
                    (key, min(vals)) for key, vals in nest_indices["mid"].items()
                ]
                middle_nests = [x[0] for x in sorted(ordered, key=lambda x: x[1])]

                for idx, n in enumerate(middle_nests):
                    is_last = idx == len(middle_nests) - 1
                    parent, child = n.split("_")
                    P_nest = nest_prob_m[n]
                    P_y_given_nest = cond_prob_m[n]["P_y_given"]
                    prod = pm.Deterministic(
                        f"prod_{n}_m", (P_nest[:, pt.newaxis] * P_y_given_nest)
                    )
                    if parent in path_prods_m:
                        path_prods_m[parent].append(prod)
                    else:
                        path_prods_m[parent] = []
                        path_prods_m[parent].append(prod)
                    if is_last:
                        P_ = pm.Deterministic(
                            f"P_{parent}_children",
                            pm.math.concatenate(path_prods_m[parent], axis=1),
                        )
                        child_nests[parent] = P_
            else:
                child_nests = {}

            ## Top Level
            cond_prob_t, nest_prob_t = self.make_P_nest(
                U, W, betas_fixed, lambdas_nests, "top"
            )

            path_prods_t = []
            ordered = [(key, min(vals)) for key, vals in nest_indices["top"].items()]
            top_nests = [x[0] for x in sorted(ordered, key=lambda x: x[1])]
            for _idx, n in enumerate(top_nests):
                P_nest = nest_prob_t[n]
                P_y_given_nest = cond_prob_t[n]["P_y_given"]
                if n in child_nests:
                    P_y_given_nest_mid = pm.Deterministic(
                        f"P_y_given_nest_mid_{n}", child_nests[n]
                    )
                    prod = pm.Deterministic(
                        f"prod_{n}_t", (P_nest[:, pt.newaxis] * (P_y_given_nest_mid))
                    )
                else:
                    prod = pm.Deterministic(
                        f"prod_{n}_t", (P_nest[:, pt.newaxis] * (P_y_given_nest))
                    )
                path_prods_t.append(prod)
            P_ = pm.Deterministic(
                "p", pm.math.concatenate(path_prods_t, axis=1), dims=("obs", "alts")
            )

            _ = pm.Categorical("likelihood", p=P_, observed=y_data, dims="obs")

        self.model = model
        return model

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
        attrs["choice_df"] = json.dumps("Placeholder for DF")
        attrs["nesting_structure"] = json.dumps(self.nesting_structure)
        attrs["utility_equations"] = json.dumps(self.utility_equations)

        return attrs

    def sample_prior_predictive(  # type: ignore[override]
        self, extend_idata: bool, kwargs: dict[str, Any]
    ) -> None:  # type: ignore[override]
        """Sample Prior Predictive Distribution."""
        with self.model:  # sample with new input data
            prior_pred: az.InferenceData = pm.sample_prior_predictive(500, **kwargs)
            self.set_idata_attrs(prior_pred)

        if extend_idata:
            if self.idata is not None:
                self.idata.extend(prior_pred)
            else:
                self.idata = prior_pred

    def fit(self, extend_idata: bool, kwargs: dict[str, Any]) -> None:  # type: ignore[override]
        """Fit Nested Logit Model."""
        if extend_idata:
            with self.model:
                if self.idata is not None:
                    self.idata.extend(pm.sample(**kwargs))
        else:
            with self.model:
                self.idata = pm.sample(**kwargs)

    def sample_posterior_predictive(  # type: ignore[override]
        self, extend_idata: bool, kwargs: dict[str, Any]
    ) -> None:
        """Sample Posterior Predictive Distribution."""
        if self.idata is not None:
            with self.model:
                self.post_pred = pm.sample_posterior_predictive(
                    self.idata, var_names=["likelihood", "p"], **kwargs
                )
            if extend_idata:
                self.idata.extend(self.post_pred)
        else:
            raise ValueError("Cannot extend `idata` because it is None.")

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
            extend_idata=True, kwargs=sample_prior_predictive_kwargs
        )
        self.fit(extend_idata=True, kwargs=fit_kwargs)
        self.sample_posterior_predictive(
            extend_idata=True, kwargs=sample_posterior_predictive_kwargs
        )
        return self

    def apply_intervention(
        self,
        new_choice_df: pd.DataFrame,
        new_utility_equations: list[str] | None = None,
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

        Returns
        -------
        az.InferenceData
            The posterior or full predictive distribution under the intervention, including
            predicted probabilities (`"p"`) and likelihood draws (`"likelihood"`).

        """
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
                idata_new_policy.extend(
                    pm.sample(
                        target_accept=0.99,
                        tune=2000,
                        idata_kwargs={"log_likelihood": True},
                        random_seed=101,
                    )
                )
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
    ):
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
