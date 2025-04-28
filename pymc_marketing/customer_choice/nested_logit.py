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

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import pymc as pm
import pytensor.tensor as pt
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
    ...     'alt_1 ~ X1_alt1 + X2_alt1 | income',
    ...     'alt_2 ~ X1_alt2 + X2_alt2 | income',
    ...     'alt_3 ~ X1_alt3 + X2_alt3 | income'
    ... ]

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
        betas_fixed = Prior("Normal", mu=0, sigma=1, dims=("alts", "fixed_covariates"))

        return {
            "alphas_": alphas,
            "betas": betas,
            "betas_fixed_": betas_fixed,
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
            "betas": self.model_config["betas"].to_dict(),
            "betas_fixed": self.model_config["betas_fixed_"].to_dict(),
        }

        return result

    def parse_formula(self, df, formula, depvar):
        """Parse the three-part structure of a formula specification.

        Splits the formula into target, alternative-specific covariates, and
        fixed covariates. Ensures that the target variable appears in the
        dependent variable column and that all specified covariates exist
        in the input dataframe.
        """
        target, covariates = formula.split("~")
        target = target.strip()

        alt_covariates, fixed_covariates = covariates.split("|")
        alt_covariates = alt_covariates.strip()
        fixed_covariates = fixed_covariates.strip()

        if target not in df[depvar].unique():
            raise ValueError(
                f"Target '{target}' not found in dependent variable '{depvar}'."
            )

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

        return target, alt_covariates, fixed_covariates

    def prepare_X_matrix(self, df, utility_formulas, depvar):
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
    def _prepare_y_outcome(df, alternatives, depvar):
        """Encode the outcome category variable for use in the modelling.

        The order of the alterntives should map to the order of the
        utility formulas.
        """
        prod_mapping = dict(zip(alternatives, range(len(alternatives)), strict=False))
        df["mode_encoded"] = df[depvar].map(prod_mapping)
        y = df["mode_encoded"].values
        return y, prod_mapping

    @staticmethod
    def _parse_nesting(nest_dict, product_indices):
        """Encode the relevant nesting structure specified when initialising the model.

        We build two dictionaries for the top and middle logit
        models in the hierarchical nesting structure as required.
        """
        if not nest_dict:
            raise ValueError("Nesting structure must not be empty.")
        top_level = {}
        mid_level = {}
        for k in nest_dict.keys():
            if isinstance(nest_dict[k], dict):
                collected_idxs = []
                for j in nest_dict[k]:
                    if isinstance(nest_dict[k][j], dict):
                        raise ValueError("Cannot have more than 2 layers of Nesting")
                    else:
                        idxs = [product_indices[i] for i in nest_dict[k][j]]
                        mid_level[k + "_" + j] = idxs
                        collected_idxs = collected_idxs + idxs
                top_level[k] = np.sort(collected_idxs)
            else:
                top_level[k] = np.sort([product_indices[i] for i in nest_dict[k]])

        if not mid_level:
            mid_level = None
        return top_level, mid_level

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pre-process the model initiation inputs into a format that can be used by the PyMC model."""
        X, F, alternatives, fixed_covar = self.prepare_X_matrix(
            choice_df, utility_equations, self.depvar
        )
        self.X, self.F, self.alternatives, self.fixed_covar = (
            X,
            F,
            alternatives,
            fixed_covar,
        )
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

    def make_exp_nest(self, U, W, betas_fixed_, lambdas_nests, nest, level="top"):
        """Calculate within nest probabilities.

        Allows mixing between levels of the nesting structure within the PyMC model.
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
            betas_fixed_temp = betas_fixed_[nest_indices[level][nest], :]
            betas_fixed_temp = pm.Deterministic(
                f"beta_fixed_{level}_{nest}",
                pt.set_subtensor(betas_fixed_temp[-1, :], 0),
            )
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

    def make_P_nest(self, U, W, betas_fixed_, lambdas_nests, level):
        """Calculate the probability of choosing a nest."""
        nest_indices = self.nest_indices
        conditional_probs = {}
        for n in nest_indices[level].keys():
            exp_W_nest, P_y_given_nest = self.make_exp_nest(
                U, W, betas_fixed_, lambdas_nests, n, level
            )
            if W is None:
                exp_W_nest = pm.math.sum(exp_W_nest, axis=1)
                conditional_probs[n] = {"exp": exp_W_nest, "P_y_given": P_y_given_nest}
            else:
                exp_W_nest = pm.math.sum(exp_W_nest, axis=1)
                conditional_probs[n] = {"exp": exp_W_nest, "P_y_given": P_y_given_nest}

        denom = pm.Deterministic(
            f"denom_{level}",
            pm.math.sum(
                [conditional_probs[n]["exp"] for n in nest_indices[level].keys()],
                axis=0,
            ),
        )
        nest_probs = {}
        for n in nest_indices[level].keys():
            P_nest = pm.Deterministic(f"P_{n}", (conditional_probs[n]["exp"] / denom))
            nest_probs[n] = P_nest
        return conditional_probs, nest_probs

    def make_model(self, X, W, y) -> None:
        """Build Model."""
        nest_indices = self.nest_indices
        coords = self.coords

        with pm.Model(coords=coords) as model:
            alphas = pm.Normal("alphas", 0, 1, dims="alts")
            betas = pm.Normal("betas", 0, 1, dims=("alt_covariates"))
            lambdas_nests = pm.Beta("lambda_nests", 2, 2, dims="nests")

            if W is None:
                w_nest = pm.math.zeros((len(coords["obs"]), len(coords["alts"])))
                betas_fixed_ = None
            else:
                W_data = pm.Data("W", W, dims=("obs", "fixed_covariates"))
                betas_fixed_ = pm.Normal(
                    "betas_fixed_", 0, 1, dims=("alts", "fixed_covariates")
                )
                betas_fixed = pm.Deterministic(
                    "betas_fixed",
                    pt.set_subtensor(betas_fixed_[-1, :], 0),
                    dims=("alts", "fixed_covariates"),
                )
                w_nest = pm.math.dot(W_data, betas_fixed.T)
            X_data = pm.Data("X", X, dims=("obs", "alts", "alt_covariates"))
            y_data = pm.Data("y", y, dims="obs")

            # Compute utility as a dot product
            alphas = pt.set_subtensor(alphas[-1], 0)
            u = alphas + pm.math.dot(X_data, betas)
            U = pm.Deterministic("U", w_nest + u, dims=("obs", "alts"))

            ## Mid Level
            if "mid" in nest_indices.keys():
                cond_prob_m, nest_prob_m = self.make_P_nest(
                    U, W, betas_fixed_, lambdas_nests, "mid"
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
                U, W, betas_fixed_, lambdas_nests, "top"
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

    def sample_prior_predictive(self, extend_idata, kwargs):
        """Sample Prior Predictive Distribution."""
        with self.model:  # sample with new input data
            prior_pred: az.InferenceData = pm.sample_prior_predictive(500, **kwargs)
            self.set_idata_attrs(prior_pred)

        if extend_idata:
            if self.idata is not None:
                self.idata.extend(prior_pred)
            else:
                self.idata = prior_pred

    def fit(self, extend_idata, kwargs):
        """Fit Nested Logit Model."""
        if extend_idata:
            with self.model:
                self.idata.extend(pm.sample(**kwargs))
        else:
            with self.model:
                self.idata = pm.sample(**kwargs)

    def sample_posterior_predictive(self, extend_idata, kwargs):
        """Sample Posterior Predictive Distribution."""
        if extend_idata:
            with self.model:
                self.idata.extend(
                    pm.sample_posterior_predictive(
                        self.idata, var_names=["likelihood", "p"], **kwargs
                    )
                )
        else:
            with self.model:
                self.post_pred = pm.sample_posterior_predictive(
                    self.idata, var_names=["likelihood", "p"], **kwargs
                )

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

    def apply_intervention(self, new_choice_df, new_utility_equations=None):
        """Apply one of two types of intervention.

        The first type of intervention assumes we have a fitted model and
        just aims to sample from the posterior predictive distribution after
        adjusting one of more of the models observable attributes and passing
        in the new_choice_df. The second type of intervention allows that we remove a
        product entirely from the market place and model the market share which
        accrues to each product in the adjusted market.
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
    def calculate_share_change(idata, new_idata):
        """Calculate difference in market share due to market intervention."""
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
    def make_change_plot(change_df, title="Price Intervention", figsize=(8, 4)):
        """Plot change induced by a market intervention."""
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
        ax.set_title(
            f" Multinomial Logit implied Market Shares \n Before and After {title} \n \n\n",
        )
        return fig
