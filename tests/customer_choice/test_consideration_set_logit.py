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


import numpy as np
import pandas as pd
import pytest

from pymc_marketing.customer_choice.consideration_set_logit import (
    ConsiderationSetMixedLogit,
)

seed = sum(map(ord, "ConsiderationSet"))
rng = np.random.default_rng(seed)


def _generate_hiring_data(n=50, random_consideration=False, seed=42):
    """Generate synthetic hiring data with consideration instruments."""
    local_rng = np.random.default_rng(seed)

    TRUE_ALPHA = np.array([0.6, 1.1, 0.0])
    TRUE_B_AGE = -0.9
    TRUE_B_EXP = 0.4
    TRUE_GAMMA_Z = np.array([2.5, 3.5, 3.0])

    Z_GRAD_MEAN = 0.40
    Z_WHITE_MEAN = 0.65
    Z_YOUNG_MEAN = 0.55
    Z_MEANS = np.array([Z_GRAD_MEAN, Z_WHITE_MEAN, Z_YOUNG_MEAN])

    TRUE_SIGMA_CONSIDER = 1.5 if random_consideration else 0.0
    eta = local_rng.normal(0, max(TRUE_SIGMA_CONSIDER, 1e-10), size=n)

    rows = []
    for i in range(n):
        age_A = local_rng.uniform(22, 60)
        age_B = local_rng.uniform(22, 60)
        age_C = local_rng.uniform(22, 60)
        exp_A = local_rng.uniform(0, 30)
        exp_B = local_rng.uniform(0, 30)
        exp_C = local_rng.uniform(0, 30)

        is_college_grad = local_rng.beta(2, 3)
        is_white = float(local_rng.binomial(1, 0.65))
        is_under_40 = float(local_rng.binomial(1, 0.55))
        z_raw = np.array([is_college_grad, is_white, is_under_40])
        z_tilde = z_raw - Z_MEANS

        log_odds_c = TRUE_GAMMA_Z * z_tilde
        if random_consideration:
            log_odds_c += eta[i]
        pi = 1.0 / (1.0 + np.exp(-log_odds_c))

        V = np.array(
            [
                TRUE_ALPHA[0] + TRUE_B_AGE * age_A + TRUE_B_EXP * exp_A,
                TRUE_ALPHA[1] + TRUE_B_AGE * age_B + TRUE_B_EXP * exp_B,
                TRUE_ALPHA[2] + TRUE_B_AGE * age_C + TRUE_B_EXP * exp_C,
            ]
        )

        U_adj = np.log(pi + 1e-8) + V
        U_adj = U_adj - U_adj.max()
        exp_u = np.exp(U_adj)
        p = exp_u / exp_u.sum()
        choice = local_rng.choice(["Firm A", "Firm B", "Firm C"], p=p)

        rows.append(
            {
                "choice": choice,
                "age_A": age_A,
                "age_B": age_B,
                "age_C": age_C,
                "exp_A": exp_A,
                "exp_B": exp_B,
                "exp_C": exp_C,
                "is_college_grad": is_college_grad,
                "is_white": is_white,
                "is_under_40": is_under_40,
            }
        )

    df = pd.DataFrame(rows)
    Z_raw = np.column_stack([df["is_college_grad"], df["is_white"], df["is_under_40"]])
    Z_tilde = Z_raw - Z_MEANS

    return df, Z_tilde


@pytest.fixture
def hiring_df():
    df, _ = _generate_hiring_data(n=50)
    return df


@pytest.fixture
def Z_tilde():
    _, Z = _generate_hiring_data(n=50)
    return Z


@pytest.fixture
def utility_equations():
    return [
        "Firm A ~ age_A + exp_A",
        "Firm B ~ age_B + exp_B",
        "Firm C ~ age_C + exp_C",
    ]


class TestConsiderationSetMixedLogitBuild:
    def test_model_builds(self, hiring_df, Z_tilde, utility_equations):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        model.build_model()
        assert model.model is not None

        # Check that consideration-stage variables exist
        var_names = [v.name for v in model.model.free_RVs]
        assert "gamma_z" in var_names
        assert "sigma_consider_eta" not in var_names  # not random

    def test_model_builds_with_random_consideration(
        self, hiring_df, Z_tilde, utility_equations
    ):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
            random_consideration=True,
        )
        model.build_model()

        var_names = [v.name for v in model.model.free_RVs]
        assert "gamma_z" in var_names
        assert "sigma_consider_eta" in var_names
        assert "z_consider_eta" in var_names

    def test_model_has_pi_deterministic(self, hiring_df, Z_tilde, utility_equations):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        model.build_model()

        det_names = [v.name for v in model.model.deterministics]
        assert "pi" in det_names
        assert "p" in det_names

    def test_model_builds_with_consideration_intercept(
        self, hiring_df, Z_tilde, utility_equations
    ):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
            consideration_intercept=True,
        )
        model.build_model()

        var_names = [v.name for v in model.model.free_RVs]
        assert "gamma_0" in var_names
        assert "gamma_z" in var_names

    def test_no_gamma_0_by_default(self, hiring_df, Z_tilde, utility_equations):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        model.build_model()

        var_names = [v.name for v in model.model.free_RVs]
        assert "gamma_0" not in var_names

    def test_missing_Z_tilde_raises(self, hiring_df, utility_equations):
        with pytest.raises(ValueError, match="Z_tilde"):
            ConsiderationSetMixedLogit(
                choice_df=hiring_df,
                utility_equations=utility_equations,
                depvar="choice",
                covariates=["age", "exp"],
                consideration_instruments={"wrong_key": np.zeros((50, 3))},
            )


class TestConsiderationSetMixedLogitPrior:
    def test_prior_predictive(self, hiring_df, Z_tilde, utility_equations):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        prior = model.sample_prior_predictive(samples=10, random_seed=42)
        assert "prior_predictive" in prior.groups()
        assert "likelihood" in prior["prior_predictive"]

    def test_prior_predictive_random(self, hiring_df, Z_tilde, utility_equations):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
            random_consideration=True,
        )
        prior = model.sample_prior_predictive(samples=10, random_seed=42)
        assert "prior_predictive" in prior.groups()


class TestConsiderationSetMixedLogitConfig:
    def test_default_config_has_gamma_z(self, hiring_df, Z_tilde, utility_equations):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        assert "gamma_z" in model.model_config

    def test_default_config_random_has_sigma(
        self, hiring_df, Z_tilde, utility_equations
    ):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
            random_consideration=True,
        )
        assert "sigma_consider_eta" in model.model_config
        assert "z_consider_eta" in model.model_config

    def test_serializable_config(self, hiring_df, Z_tilde, utility_equations):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
            random_consideration=True,
        )
        model.build_model()
        config = model._serializable_model_config
        assert "gamma_z" in config
        assert "sigma_consider_eta" in config

    def test_model_type(self, hiring_df, Z_tilde, utility_equations):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        assert model._model_type == "Consideration Set Mixed Logit"

    def test_config_with_consideration_intercept(
        self, hiring_df, Z_tilde, utility_equations
    ):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
            consideration_intercept=True,
        )
        assert "gamma_0" in model.model_config
        model.build_model()
        config = model._serializable_model_config
        assert "gamma_0" in config


class TestConsiderationSetMixedLogitInference:
    """Tests for sample_posterior_predictive and apply_intervention."""

    def test_sample_posterior_predictive(self, hiring_df, Z_tilde, utility_equations):
        """Fit model then sample posterior predictive."""
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        idata = model.fit(
            random_seed=42,
            fit_kwargs={
                "draws": 50,
                "tune": 50,
                "chains": 1,
                "target_accept": 0.8,
            },
        )
        assert idata is not None

        post_pred = model.sample_posterior_predictive(random_seed=42)
        assert "posterior_predictive" in post_pred.groups()
        assert "p" in post_pred["posterior_predictive"]
        assert "likelihood" in post_pred["posterior_predictive"]

    def test_sample_posterior_predictive_with_new_instruments(
        self, hiring_df, Z_tilde, utility_equations
    ):
        """Posterior predictive with new consideration instruments."""
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        model.fit(
            random_seed=42,
            fit_kwargs={
                "draws": 50,
                "tune": 50,
                "chains": 1,
                "target_accept": 0.8,
            },
        )

        # Counterfactual: shift Z_tilde
        Z_cf = Z_tilde.copy()
        Z_cf[:, 0] = 1.0  # increase consideration for first alt

        post_pred = model.sample_posterior_predictive(
            consideration_instruments={"Z_tilde": Z_cf},
            extend_idata=False,
            random_seed=42,
        )
        assert "posterior_predictive" in post_pred.groups()
        assert "p" in post_pred["posterior_predictive"]

    def test_apply_intervention(self, hiring_df, Z_tilde, utility_equations):
        """Apply intervention with new consideration instruments."""
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        model.fit(
            random_seed=42,
            fit_kwargs={
                "draws": 50,
                "tune": 50,
                "chains": 1,
                "target_accept": 0.8,
            },
        )

        # Counterfactual: everyone gets high consideration for alt 0
        Z_cf = Z_tilde.copy()
        Z_cf[:, 0] = 2.0

        idata_cf = model.apply_intervention(
            new_choice_df=hiring_df,
            new_consideration_instruments={"Z_tilde": Z_cf},
        )
        assert "posterior_predictive" in idata_cf.groups()
        assert "p" in idata_cf["posterior_predictive"]
        assert hasattr(model, "intervention_idata")

    def test_sample_posterior_predictive_with_new_choice_df(
        self, hiring_df, Z_tilde, utility_equations
    ):
        """Posterior predictive with a new choice_df and matching instruments."""
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        model.fit(
            random_seed=42,
            fit_kwargs={
                "draws": 50,
                "tune": 50,
                "chains": 1,
                "target_accept": 0.8,
            },
        )

        # Use the same-sized data but with shifted values + new Z
        new_df = hiring_df.copy()
        new_df["age_A"] = new_df["age_A"] + 5  # everyone is older
        Z_cf = Z_tilde.copy()
        Z_cf[:, 0] = 1.0  # shift consideration for first alt

        post_pred = model.sample_posterior_predictive(
            choice_df=new_df,
            consideration_instruments={"Z_tilde": Z_cf},
            extend_idata=False,
            random_seed=42,
        )
        assert "posterior_predictive" in post_pred.groups()
        assert "p" in post_pred["posterior_predictive"]

    def test_apply_intervention_multi_instrument(self, hiring_df, utility_equations):
        """Apply intervention with multi-instrument Z_tilde."""
        n = len(hiring_df)
        Z_multi = rng.standard_normal((n, 3, 2))
        z_names = ["inst_A", "inst_B"]

        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={
                "Z_tilde": Z_multi,
                "z_instrument_names": z_names,
            },
        )
        model.fit(
            random_seed=42,
            fit_kwargs={
                "draws": 50,
                "tune": 50,
                "chains": 1,
                "target_accept": 0.8,
            },
        )

        # Counterfactual: shift first instrument
        Z_cf = Z_multi.copy()
        Z_cf[:, :, 0] = 2.0

        idata_cf = model.apply_intervention(
            new_choice_df=hiring_df,
            new_consideration_instruments={
                "Z_tilde": Z_cf,
                "z_instrument_names": z_names,
            },
        )
        assert "posterior_predictive" in idata_cf.groups()
        assert "p" in idata_cf["posterior_predictive"]

    def test_apply_intervention_with_new_equations(
        self, hiring_df, Z_tilde, utility_equations
    ):
        """Apply intervention with new utility equations (triggers refit).

        The refit branch builds a fresh model and runs full MCMC, so we
        pass equations with the same covariates but a different formula
        structure (e.g., swapped covariate assignment) to exercise the
        ``new_utility_equations is not None`` code path.
        """
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        model.fit(
            random_seed=42,
            fit_kwargs={
                "draws": 50,
                "tune": 50,
                "chains": 1,
                "target_accept": 0.8,
            },
        )

        # Same structure but swap age and exp (different utility spec)
        new_equations = [
            "Firm A ~ exp_A + age_A",
            "Firm B ~ exp_B + age_B",
            "Firm C ~ exp_C + age_C",
        ]

        idata_cf = model.apply_intervention(
            new_choice_df=hiring_df,
            new_utility_equations=new_equations,
            new_consideration_instruments={"Z_tilde": Z_tilde},
            fit_kwargs={
                "draws": 50,
                "tune": 50,
                "chains": 1,
                "target_accept": 0.8,
            },
        )
        assert "posterior_predictive" in idata_cf.groups()
        assert "posterior" in idata_cf.groups()
        assert "p" in idata_cf["posterior_predictive"]
        assert hasattr(model, "intervention_idata")


class TestConsiderationSetMixedLogitStability:
    def test_extreme_z_tilde_no_nan(self, hiring_df, utility_equations):
        """Large negative Z_tilde (unavailable alternatives) should not cause NaN."""
        n = len(hiring_df)
        Z_extreme = np.zeros((n, 3))
        Z_extreme[:, 2] = -10.0  # Firm C effectively unavailable

        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_extreme},
        )
        prior = model.sample_prior_predictive(samples=10, random_seed=42)
        assert "prior_predictive" in prior.groups()
        # Check no NaN in choice probabilities
        p_vals = prior["prior"]["p"].values
        assert not np.any(np.isnan(p_vals))

    def test_idata_attrs(self, hiring_df, Z_tilde, utility_equations):
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
            consideration_intercept=True,
            random_consideration=True,
        )
        model.build_model()
        attrs = model.create_idata_attrs()
        assert "random_consideration" in attrs
        assert "consideration_intercept" in attrs
        assert "consideration_instruments" in attrs

    def test_multi_instrument_z_tilde(self, hiring_df, utility_equations):
        """Test Z_tilde with shape (N, J, K_z) for multiple instruments."""
        n = len(hiring_df)
        # 3 alternatives, 2 instruments each
        Z_multi = rng.standard_normal((n, 3, 2))

        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={
                "Z_tilde": Z_multi,
                "z_instrument_names": ["inst_A", "inst_B"],
            },
        )
        model.build_model()
        assert model.model is not None
        assert model._multi_instrument is True

        # gamma_z should have shape (alts, z_instruments)
        var_names = [v.name for v in model.model.free_RVs]
        assert "gamma_z" in var_names

        # Prior predictive should work
        prior = model.sample_prior_predictive(samples=10, random_seed=42)
        assert "prior_predictive" in prior.groups()
        p_vals = prior["prior"]["p"].values
        assert not np.any(np.isnan(p_vals))


class TestConsiderationSetMixedLogitValidation:
    """Negative tests for input validation."""

    def test_z_tilde_wrong_rank_1d_raises(self, hiring_df, utility_equations):
        with pytest.raises(ValueError, match=r"2-D.*3-D"):
            ConsiderationSetMixedLogit(
                choice_df=hiring_df,
                utility_equations=utility_equations,
                depvar="choice",
                covariates=["age", "exp"],
                consideration_instruments={"Z_tilde": np.zeros(50)},
            )

    def test_z_tilde_wrong_rank_4d_raises(self, hiring_df, utility_equations):
        with pytest.raises(ValueError, match=r"2-D.*3-D"):
            ConsiderationSetMixedLogit(
                choice_df=hiring_df,
                utility_equations=utility_equations,
                depvar="choice",
                covariates=["age", "exp"],
                consideration_instruments={"Z_tilde": np.zeros((50, 3, 2, 1))},
            )

    def test_z_instrument_names_length_mismatch_raises(
        self, hiring_df, utility_equations
    ):
        n = len(hiring_df)
        Z_multi = rng.standard_normal((n, 3, 2))
        with pytest.raises(ValueError, match="z_instrument_names"):
            ConsiderationSetMixedLogit(
                choice_df=hiring_df,
                utility_equations=utility_equations,
                depvar="choice",
                covariates=["age", "exp"],
                consideration_instruments={
                    "Z_tilde": Z_multi,
                    "z_instrument_names": ["only_one"],  # K_z=2, names=1
                },
            )

    def test_missing_Z_tilde_key_raises(self, hiring_df, utility_equations):
        with pytest.raises(ValueError, match="Z_tilde"):
            ConsiderationSetMixedLogit(
                choice_df=hiring_df,
                utility_equations=utility_equations,
                depvar="choice",
                covariates=["age", "exp"],
                consideration_instruments={"wrong_key": np.zeros((50, 3))},
            )

    def test_dim_switch_2d_to_3d_after_build_raises(
        self, hiring_df, Z_tilde, utility_equations
    ):
        """Switching from 2-D to 3-D Z_tilde on an already-built model must fail."""
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={"Z_tilde": Z_tilde},
        )
        model.build_model()

        n = len(hiring_df)
        Z_3d = rng.standard_normal((n, 3, 2))
        with pytest.raises(ValueError, match="dimensionality"):
            model.sample_posterior_predictive(
                consideration_instruments={"Z_tilde": Z_3d}
            )

    def test_dim_switch_3d_to_2d_after_build_raises(
        self, hiring_df, Z_tilde, utility_equations
    ):
        """Switching from 3-D to 2-D Z_tilde on an already-built model must fail."""
        n = len(hiring_df)
        Z_3d = rng.standard_normal((n, 3, 2))
        model = ConsiderationSetMixedLogit(
            choice_df=hiring_df,
            utility_equations=utility_equations,
            depvar="choice",
            covariates=["age", "exp"],
            consideration_instruments={
                "Z_tilde": Z_3d,
                "z_instrument_names": ["inst_A", "inst_B"],
            },
        )
        model.build_model()

        with pytest.raises(ValueError, match="dimensionality"):
            model.sample_posterior_predictive(
                consideration_instruments={"Z_tilde": Z_tilde}
            )
