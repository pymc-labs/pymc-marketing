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
"""Consideration Set Mixed Logit with optional random consideration intercepts."""

import json

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc_extras.prior import Prior
from pytensor.tensor.variable import TensorVariable

from pymc_marketing.customer_choice.mixed_logit import MixedLogit


class ConsiderationSetMixedLogit(MixedLogit):
    """
    Consideration Set Mixed Logit with optional random consideration intercepts.

    Extends :class:`MixedLogit` with a two-stage structure that separates
    *who gets considered* from *who is preferred conditional on consideration*:

    Stage 1 (Consideration):
        pi_nj = sigmoid([gamma_0j +] sum_k gamma_zjk * z_tilde_njk  [+ eta_n])

    Stage 2 (Choice):
        P(j | n) = softmax(log(pi_nj) + V_nj)

    where V_nj is the standard mixed logit utility from the parent class.

    The consideration instruments Z must satisfy an exclusion restriction:
    they must be structurally separate from the utility covariates X. This
    is the discrete choice analogue of K != V in transformer attention.

    Parameters
    ----------
    choice_df : pd.DataFrame
        Wide DataFrame where each row is a choice scenario.
    utility_equations : list of str
        Utility formulas in Wilkinson notation (see MixedLogit).
    depvar : str
        Name of the dependent variable column.
    covariates : list of str
        Base covariate names (e.g., ['price', 'time']).
    consideration_instruments : dict
        Must contain 'Z_tilde': np.ndarray of shape (N, J) for a single
        instrument per alternative, or (N, J, K_z) for multiple
        instruments per alternative. Mean-centring is the caller's
        responsibility. At z_tilde = 0, pi = sigmoid(0) = 0.5.
        Optionally contains 'z_instrument_names': list of str of
        length K_z for labelling the instrument dimensions.
    consideration_intercept : bool, optional
        If True, adds alternative-specific intercepts gamma_0j to the
        consideration stage. Default False. When True, gamma_0j and
        alpha_j (utility intercept) are not jointly identifiable
        without further constraints; use informative priors or fix
        one set of intercepts.
    random_consideration : bool, optional
        If True, adds a random intercept eta_n to the consideration
        stage, capturing unobserved individual-level variation in
        "visibility" across all alternatives. Default False.
    model_config : dict, optional
        Model configuration with Prior specifications.
    sampler_config : dict, optional
        Sampler configuration for pm.sample().
    group_id : str, optional
        Column name for group identifier (panel data).
    instrumental_vars : dict, optional
        Instrumental variables for price endogeneity (inherited).
    non_centered : bool, optional
        Non-centered parameterization for random coefficients.

    Notes
    -----
    Identification design:

    - By default (consideration_intercept=False), there is no
      consideration intercept: gamma_0j is aliased with alpha_j.
      The utility intercept alpha_j absorbs both baseline preference
      and baseline consideration for each alternative.

    - When consideration_intercept=True, gamma_0j is estimated
      separately. This requires care: gamma_0j and alpha_j compete
      to explain baseline alternative-specific effects. Use
      informative priors or constrain one set to zero.

    - Z_tilde is mean-centred: at average screening (z_tilde = 0),
      pi = sigmoid(0) = 0.5. Only the deviation from population-mean
      screening drives consideration. This is the exclusion restriction.

    - When random_consideration=True, the random intercept eta_n has
      zero mean (not identifiable separately from alpha_j). Only
      sigma_consider is identified, capturing between-individual
      dispersion in consideration probability.
    """

    _model_type = "Consideration Set Mixed Logit"
    version = "0.1.0"

    _multi_instrument: bool
    _n_z_instruments: int
    _z_instrument_names: list[str] | None

    def __init__(
        self,
        choice_df: pd.DataFrame,
        utility_equations: list[str],
        depvar: str,
        covariates: list[str],
        consideration_instruments: dict,
        consideration_intercept: bool = False,
        random_consideration: bool = False,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        group_id: str | None = None,
        instrumental_vars: dict | None = None,
        non_centered: bool = True,
    ):
        self.consideration_intercept = consideration_intercept
        self.random_consideration = random_consideration

        # Validate Z_tilde
        if "Z_tilde" not in consideration_instruments:
            raise ValueError(
                "consideration_instruments must contain 'Z_tilde' key "
                "with a mean-centred (N, J) or (N, J, K_z) array."
            )

        self._update_consideration_instruments(consideration_instruments)

        super().__init__(
            choice_df=choice_df,
            utility_equations=utility_equations,
            depvar=depvar,
            covariates=covariates,
            model_config=model_config,
            sampler_config=sampler_config,
            group_id=group_id,
            instrumental_vars=instrumental_vars,
            non_centered=non_centered,
        )

    @property
    def default_model_config(self) -> dict:
        """Default model configuration including consideration stage priors."""
        config = super().default_model_config

        # Consideration instrument slopes
        if self._multi_instrument:
            # gamma_z: (alts, z_instruments) — each mode has its own
            # sensitivity to each instrument
            config["gamma_z"] = Prior(
                "Normal", mu=0, sigma=2, dims=("alts", "z_instruments")
            )
        else:
            # gamma_z: (alts,) — one slope per alternative
            config["gamma_z"] = Prior("Normal", mu=0, sigma=2, dims="alts")

        # Optional consideration intercepts (one per alternative)
        if self.consideration_intercept:
            config["gamma_0"] = Prior("Normal", mu=0, sigma=2, dims="alts")

        # Random consideration intercept priors
        if self.random_consideration:
            config["sigma_consider_eta"] = Prior("HalfNormal", sigma=2)
            config["z_consider_eta"] = Prior(
                "Normal", mu=0, sigma=1, dims="individuals"
            )

        return config

    @property
    def _serializable_model_config(self) -> dict[str, int | float | dict]:
        result = super()._serializable_model_config
        result["gamma_z"] = self.model_config["gamma_z"].to_dict()
        if self.consideration_intercept:
            result["gamma_0"] = self.model_config["gamma_0"].to_dict()
        if self.random_consideration:
            result["sigma_consider_eta"] = self.model_config[
                "sigma_consider_eta"
            ].to_dict()
            result["z_consider_eta"] = self.model_config["z_consider_eta"].to_dict()
        return result

    def make_consideration_probs(
        self,
        Z_data: TensorVariable,
        n_obs: int,
        n_alts: int,
        grp_idx: np.ndarray | TensorVariable | None = None,
    ) -> tuple[TensorVariable, TensorVariable]:
        """Compute consideration probabilities via independent sigmoid per alternative.

        Parameters
        ----------
        Z_data : TensorVariable
            Mean-centred consideration instruments, shape (n_obs, n_alts) for
            single instrument or (n_obs, n_alts, n_z_instruments) for multiple.
        n_obs : int
            Number of observations.
        n_alts : int
            Number of alternatives.
        grp_idx : array-like or None
            Group index mapping observations to individuals (panel data).

        Returns
        -------
        pi : TensorVariable
            Consideration probabilities, shape (n_obs, n_alts).
        log_pi : TensorVariable
            Log-consideration probabilities via the numerically stable
            identity log(sigmoid(x)) = x − softplus(x), shape (n_obs, n_alts).
        """
        gamma_z = self.model_config["gamma_z"].create_variable("gamma_z")

        # Core consideration log-odds
        if self._multi_instrument:
            # gamma_z: (J, K_z), Z_data: (N, J, K_z)
            # log_odds: (N, J) = sum_k gamma_z[j,k] * Z[n,j,k]
            log_odds = pt.sum(gamma_z[None, :, :] * Z_data, axis=2)
        else:
            # gamma_z: (J,), Z_data: (N, J)
            log_odds = gamma_z[None, :] * Z_data  # (N, J)

        # Optional: alternative-specific consideration intercepts
        if self.consideration_intercept:
            gamma_0 = self.model_config["gamma_0"].create_variable("gamma_0")
            log_odds = gamma_0[None, :] + log_odds  # (N, J)

        # Optional: random consideration intercept per individual
        if self.random_consideration:
            sigma_eta = self.model_config["sigma_consider_eta"].create_variable(
                "sigma_consider_eta"
            )

            if self.non_centered:
                z_eta = self.model_config["z_consider_eta"].create_variable(
                    "z_consider_eta"
                )
                eta_individual = pm.Deterministic(
                    "eta_consider", z_eta * sigma_eta, dims="individuals"
                )
            else:
                eta_individual = pm.Normal(
                    "eta_consider",
                    mu=0,
                    sigma=sigma_eta,
                    dims="individuals",
                )

            # Map individuals to observations
            if grp_idx is not None:
                eta = eta_individual[grp_idx]  # (n_obs,)
            else:
                eta = eta_individual  # (n_obs,) — one individual per obs

            log_odds = log_odds + eta[:, None]  # (N, J)

        pi = pm.Deterministic("pi", pm.math.sigmoid(log_odds), dims=("obs", "alts"))
        # Use log-sigmoid identity: log(sigmoid(x)) = x - softplus(x)
        # This is numerically superior to log(sigmoid(x) + eps)
        log_pi = log_odds - pt.softplus(log_odds)

        return pi, log_pi

    def make_model(
        self,
        X: np.ndarray,
        F: np.ndarray | None,
        y: np.ndarray,
        observed: bool = True,
    ) -> pm.Model:
        """Build consideration set mixed logit model.

        Reuses all utility machinery from the parent ``MixedLogit``, then
        adds the consideration stage and computes
        log-consideration-adjusted choice probabilities via the bridge
        formula: ``P(j|n) = softmax(log(pi_nj) + V_nj)``.

        Parameters are identical to :meth:`MixedLogit.make_model`.
        """
        Z_tilde = self.consideration_instruments["Z_tilde"]

        # Ensure z_instruments coord is present for multi-instrument models
        if self._multi_instrument and self._z_instrument_names is not None:
            self.coords["z_instruments"] = self._z_instrument_names

        with pm.Model(coords=self.coords) as model:
            # --- Data ---
            X_data = pm.Data("X", X, dims=("obs", "alts", "covariates"))
            if self._multi_instrument:
                Z_data = pm.Data("Z", Z_tilde, dims=("obs", "alts", "z_instruments"))
            else:
                Z_data = pm.Data("Z", Z_tilde, dims=("obs", "alts"))
            observed_data = pm.Data("y", y, dims="obs")

            if self.grp_idx is not None:
                grp_idx_data = pm.Data("grp_idx", self.grp_idx, dims="obs")
            else:
                grp_idx_data = self.grp_idx

            n_obs, n_alts = X_data.shape[0], X_data.shape[1]

            # --- Utility stage (reuse all parent methods) ---
            alphas = self.make_intercepts()
            betas_non_random = self.make_non_random_coefs()
            betas_random, _ = self.make_random_coefs(
                n_obs, grp_idx_data, self.non_centered
            )
            B_full = self.make_beta_matrix(betas_non_random, betas_random, n_obs)
            W_contrib = self.make_fixed_coefs(F, n_obs, n_alts)
            price_error_contrib = self.make_control_function(n_obs, n_alts)
            U = self.make_utility(
                X_data, B_full, alphas, W_contrib, price_error_contrib
            )

            # --- Consideration stage ---
            _pi, log_pi = self.make_consideration_probs(
                Z_data, n_obs, n_alts, grp_idx_data
            )

            # --- Bridge: log-consideration-adjusted choice probability ---
            U_avail = U + log_pi
            U_c = U_avail - U_avail.max(axis=1, keepdims=True)
            p = pm.Deterministic(
                "p", pm.math.softmax(U_c, axis=1), dims=("obs", "alts")
            )

            # --- Likelihood ---
            if observed:
                _ = pm.Categorical(
                    "likelihood", p=p, observed=observed_data, dims="obs"
                )
            else:
                _ = pm.Categorical("likelihood", p=p, dims="obs")

        return model

    def _update_consideration_instruments(
        self, consideration_instruments: dict
    ) -> None:
        """Update consideration instruments and refresh internal state.

        Parameters
        ----------
        consideration_instruments : dict
            Must contain 'Z_tilde' key. May contain 'z_instrument_names'.
        """
        self.consideration_instruments = consideration_instruments
        Z = consideration_instruments["Z_tilde"]
        self._multi_instrument = Z.ndim == 3
        if self._multi_instrument:
            self._n_z_instruments = Z.shape[2]
            self._z_instrument_names = consideration_instruments.get(
                "z_instrument_names",
                [f"z_{k}" for k in range(self._n_z_instruments)],
            )
        else:
            self._n_z_instruments = 1
            self._z_instrument_names = None

    def sample_posterior_predictive(  # type: ignore[override]  # consideration_instruments param extends parent signature
        self,
        choice_df: pd.DataFrame | None = None,
        consideration_instruments: dict | None = None,
        extend_idata: bool = True,
        **kwargs,
    ) -> az.InferenceData:
        """Sample from posterior predictive, updating consideration instruments if needed.

        Parameters
        ----------
        choice_df : pd.DataFrame, optional
            New choice data for prediction. If None, uses training data.
            When provided, ``consideration_instruments`` must also be
            supplied with a Z_tilde whose first dimension matches the
            new DataFrame.
        consideration_instruments : dict, optional
            New consideration instruments. If None, uses training instruments.
        extend_idata : bool, optional
            Whether to add to self.idata.
        **kwargs
            Additional arguments for pm.sample_posterior_predictive.

        Returns
        -------
        az.InferenceData
            Posterior predictive samples.
        """
        if consideration_instruments is not None:
            self._update_consideration_instruments(consideration_instruments)

        # Build the data dict for pm.set_data — we always update Z when
        # either the choice_df or the instruments change.
        needs_data_update = (
            choice_df is not None or consideration_instruments is not None
        )

        if needs_data_update:
            if choice_df is not None:
                new_X, new_F, new_y = self.preprocess_model_data(
                    choice_df, self.utility_equations
                )
            else:
                new_X, new_F, new_y = None, None, None

            with self.model:
                data_dict = {}
                if new_X is not None and new_y is not None:
                    data_dict["X"] = new_X
                    data_dict["y"] = new_y
                    if new_F is not None and len(new_F) > 0:
                        data_dict["W"] = new_F
                data_dict["Z"] = self.consideration_instruments["Z_tilde"]
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

    def apply_intervention(  # type: ignore[override]  # new_consideration_instruments param extends parent signature
        self,
        new_choice_df: pd.DataFrame,
        new_utility_equations: list[str] | None = None,
        new_consideration_instruments: dict | None = None,
        fit_kwargs: dict | None = None,
    ) -> az.InferenceData:
        """Apply intervention, optionally updating consideration instruments.

        Parameters
        ----------
        new_choice_df : pd.DataFrame
            New dataset reflecting changes.
        new_utility_equations : list[str] or None
            Updated utility specifications (triggers refit if provided).
        new_consideration_instruments : dict or None
            Updated consideration instruments. If None, reuses current.
            When ``new_choice_df`` has a different number of rows from
            the training data, this must be provided with matching shape.
        fit_kwargs : dict or None
            Keyword arguments for sampling if refitting.

        Returns
        -------
        az.InferenceData
            Posterior or predictive distribution under intervention.
        """
        if new_consideration_instruments is not None:
            self._update_consideration_instruments(new_consideration_instruments)

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
                data_dict = {"X": new_X, "y": new_y}
                if new_F is not None:
                    data_dict["W"] = new_F
                data_dict["Z"] = self.consideration_instruments["Z_tilde"]
                pm.set_data(data_dict)

                idata_new_policy = pm.sample_posterior_predictive(
                    self.idata,
                    var_names=["p", "likelihood"],
                    return_inferencedata=True,
                    extend_inferencedata=False,
                    random_seed=100,  # TODO: inherit from sampler_config or accept as param
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

    def create_idata_attrs(self) -> dict[str, str]:
        """Create attributes for the InferenceData object.

        Z_tilde is not serialized into the attrs dict because it is a
        large numeric array. When loading a saved model, callers must
        re-supply ``consideration_instruments`` before prediction.
        Shape metadata is stored so loaders can validate the new array.
        """
        attrs = super().create_idata_attrs()
        attrs["consideration_intercept"] = json.dumps(self.consideration_intercept)
        attrs["random_consideration"] = json.dumps(self.random_consideration)
        attrs["consideration_instruments"] = json.dumps(
            {
                "note": "Z_tilde array not serialized; re-supply on load",
                "Z_tilde_shape": list(self.consideration_instruments["Z_tilde"].shape),
                "multi_instrument": self._multi_instrument,
                "z_instrument_names": self._z_instrument_names,
            }
        )
        return attrs
