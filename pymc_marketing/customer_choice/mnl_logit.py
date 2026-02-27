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
"""Multinomial Logit for Product Preference Analysis."""

import json
import warnings
from collections.abc import Sequence
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

    .. code-block:: python

        utility_equations = [
            "alt_1 ~ X1_alt1 + X2_alt1 | income",
            "alt_2 ~ X1_alt2 + X2_alt2 | income",
            "alt_3 ~ X1_alt3 + X2_alt3 | income",
        ]

    """

    _model_type = "Multinomial Logit Model"
    version = "0.2.0"

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
        mode_mapping = dict(zip(alternatives, range(len(alternatives)), strict=False))
        df["mode_encoded"] = df[depvar].map(mode_mapping)
        y = df["mode_encoded"].values
        return y

    @staticmethod
    def _prepare_coords(
        df: pd.DataFrame,
        alternatives: list[str],
        covariates: list[str],
        f_covariates: np.ndarray,
    ) -> dict[str, Sequence[str] | Sequence[int]]:
        """Prepare coordinates for PyMC model."""
        if isinstance(f_covariates, np.ndarray) & (f_covariates.size > 0):
            f_cov = [s.strip() for s in f_covariates[0].split("+")]
        else:
            f_cov = []
        coords = {
            "alts": alternatives,
            "alts_probs": alternatives[:-1],
            "alt_covariates": covariates,
            "fixed_covariates": f_cov,
            "obs": range(len(df)),
        }
        return coords

    def preprocess_model_data(
        self, choice_df: pd.DataFrame, utility_equations: list[str]
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Pre-process the model initiation inputs into a format that can be used by the PyMC model."""
        X, F, alternatives, fixed_covar = self.prepare_X_matrix(
            choice_df, utility_equations, self.depvar
        )
        self.X = X
        self.F: np.ndarray | None = F
        self.alternatives = alternatives
        self.fixed_covar = fixed_covar
        y = self._prepare_y_outcome(choice_df, self.alternatives, self.depvar)
        self.y = y

        # note: type hints for coords required for mypy to not get confused
        self.coords: dict[str, Sequence[str] | Sequence[int]] = self._prepare_coords(
            choice_df, self.alternatives, self.covariates, self.fixed_covar
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
        F : TensorVariable
            Contribution to utility from fixed covariates, shape (n_obs, n_alts)
        """
        if X_fixed is None or len(X_fixed) == 0:
            F = pt.zeros((n_obs, n_alts))
        else:
            X_fixed_data = pm.Data("F", X_fixed)
            betas_fixed_ = self.model_config["betas_fixed_"].create_variable(
                name="betas_fixed_"
            )
            betas_fixed = pm.Deterministic(
                "betas_fixed",
                pt.set_subtensor(betas_fixed_[-1, :], 0),
                dims=("alts", "fixed_covariates"),
            )
            F = pm.Deterministic(
                "F_interaction", pm.math.dot(X_fixed_data, betas_fixed.T)
            )
        return F

    def make_utility(
        self,
        X_data: pt.TensorVariable,
        alphas: pt.TensorVariable,
        betas: pt.TensorVariable,
        F: pt.TensorVariable,
    ) -> pt.TensorVariable:
        """Compute systematic utility for each alternative.

        Parameters
        ----------
        X_data : TensorVariable
            Alternative-specific covariates, shape (n_obs, n_alts, n_covariates)
        alphas : TensorVariable
            Alternative-specific intercepts
        betas : TensorVariable
            Coefficients for alternative-specific covariates
        F : TensorVariable
            Contribution from fixed covariates

        Returns
        -------
        U : TensorVariable
            Systematic utility, shape (n_obs, n_alts)
        """
        U = pm.math.dot(X_data, betas)  # (n_obs, n_alts)
        U = pm.Deterministic("U", F + U + alphas, dims=("obs", "alts"))
        return U

    def make_choice_prob(self, U: pt.TensorVariable) -> pt.TensorVariable:
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
        p = pm.Deterministic("p", pm.math.softmax(U, axis=1), dims=("obs", "alts"))
        return p

    def make_model(
        self, X: np.ndarray, F: np.ndarray | None, y: np.ndarray
    ) -> pm.Model:
        """Build Model.

        Parameters
        ----------
        X : np.ndarray
            Alternative-specific covariates, shape (n_obs, n_alts, n_covariates)
        F : np.ndarray or None
            Fixed covariates, shape (n_obs, n_fixed_covariates)
        y : np.ndarray
            Observed choices, shape (n_obs,)

        Returns
        -------
        model : pm.Model
            PyMC model
        """
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=self.coords) as model:
            # Create parameters
            alphas = self.make_intercepts()
            betas = self.make_alt_coefs()

            # Instantiate data
            X_data = pm.Data("X", X, dims=("obs", "alts", "alt_covariates"))
            observed = pm.Data("y", y, dims="obs")

            # Handle fixed covariates
            F_contrib = self.make_fixed_coefs(F, n_obs, n_alts)

            # Compute utility and probabilities
            U = self.make_utility(X_data, alphas, betas, F_contrib)
            p = self.make_choice_prob(U)

            # Likelihood
            _ = pm.Categorical("likelihood", p=p, observed=observed, dims="obs")

        return model

    def create_idata_attrs(self) -> dict[str, str]:
        """Create attributes for InferenceData."""
        attrs = super().create_idata_attrs()
        attrs["covariates"] = json.dumps(self.covariates)
        attrs["depvar"] = json.dumps(self.depvar)
        attrs["choice_df"] = json.dumps("Placeholder for DF")
        attrs["utility_equations"] = json.dumps(self.utility_equations)
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
                    data_dict["F"] = new_F
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
        fit_kwargs = fit_kwargs or {}
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
        self, new_choice_df, new_utility_equations=None
    ) -> az.InferenceData:
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
                pm.set_data({"X": new_X, "F": new_F, "y": new_y})
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
    def calculate_share_change(idata, new_idata) -> pd.DataFrame:
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
    def plot_change(
        change_df, title="Change due to Intervention", figsize=(8, 4)
    ) -> plt.Figure:
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
        ax.set_title(title)
        return fig
