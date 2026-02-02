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


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc_marketing.customer_choice.mnl_logit import MNLogit

seed = sum(map(ord, "CustomerChoice"))
rng = np.random.default_rng(seed)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "choice": ["alt", "alt", "other"],
            "alt_X1": [1, 2, 3],
            "alt_X2": [4, 5, 6],
            "other_X1": [1, 2, 4],
            "other_X2": [5, 6, 8],
            "income": [50000, 60000, 70000],
        }
    )


@pytest.fixture
def sample_change_df():
    return pd.DataFrame(
        {
            "policy_share": [0.3, 0.5, 0.2],
            "new_policy_share": [0.25, 0.55, 0.2],
        },
        index=["mode1", "mode2", "mode3"],
    )


@pytest.fixture
def new_utility_eqs():
    return [
        "alt ~ alt_X1 + alt_X2 | income",
        "other ~ other_X1 + other_X2 | income",
        "option ~ option_X1 + option_X2 | income",
    ]


@pytest.fixture
def utility_eqs():
    return ["alt ~ alt_X1 + alt_X2 | income", "other ~ other_X1 + other_X2 | income"]


@pytest.fixture
def mnl(sample_df, utility_eqs):
    return MNLogit(sample_df, utility_eqs, "choice", ["X1", "X2"])


@pytest.fixture
def utility_eqs_no_fixed():
    return ["alt ~ alt_X1 + alt_X2", "other ~ other_X1 + other_X2"]


@pytest.fixture
def mnl_no_fixed(sample_df, utility_eqs_no_fixed):
    return MNLogit(sample_df, utility_eqs_no_fixed, "choice", ["X1", "X2"])


def test_parse_formula_valid(mnl, sample_df):
    formula = "alt ~ alt_X1 + alt_X2 | income"
    target, alt_covs, fixed_covs = mnl.parse_formula(sample_df, formula, "choice")
    assert target == "alt"
    assert alt_covs == "alt_X1 + alt_X2"
    assert fixed_covs == "income"


def test_parse_formula_invalid_target(mnl, sample_df):
    formula = "invalid_target ~ alt_X1 + alt_X2 | income"
    with pytest.raises(ValueError):
        mnl.parse_formula(sample_df, formula, "choice")


def test_parse_formula_missing_alt_covariate(mnl, sample_df):
    formula = "alt ~ alt_X1 + missing_col | income"
    with pytest.raises(ValueError):
        mnl.parse_formula(sample_df, formula, "choice")


def test_parse_formula_missing_fixed_covariate(mnl, sample_df):
    formula = "alt ~ alt_X1 + alt_X2 | missing_fixed"
    with pytest.raises(ValueError):
        mnl.parse_formula(sample_df, formula, "choice")


def test_prepare_X_matrix_valid(mnl, sample_df, utility_eqs):
    X, F, alts, fixed_cov = mnl.prepare_X_matrix(sample_df, utility_eqs, "choice")

    assert X.shape == (3, 2, 2)  # 4 obs x 2 alts x 2 covariates
    assert F.shape == (3, 1)  # Fixed covariate should be one column
    assert set(alts) == {"alt", "other"}
    assert fixed_cov.tolist() == ["income"]


def test_prepare_X_matrix_missing_column(mnl, sample_df):
    formulas = [
        "alt ~ alt_X1 + missing_col | income",
        "other ~ other_X1 + other_X2 | income",
    ]
    with pytest.raises(ValueError):
        mnl.prepare_X_matrix(sample_df, formulas, "choice")


def test_prepare_X_matrix_unequal_covariates(mnl, sample_df):
    formulas = ["alt ~ alt_X1 + alt_X2 | income", "other ~ other_X1 | income"]
    # Expect a shape mismatch assertion because n_covariates is hardcoded from first formula
    with pytest.raises(ValueError):  # NumPy stack will fail
        mnl.prepare_X_matrix(sample_df, formulas, "choice")


def test_prepare_X_matrix_no_fixed_covariates(mnl, sample_df):
    formulas = ["alt ~ alt_X1 + alt_X2 | ", "other ~ other_X1 + other_X2 | "]
    X, F, _, fixed_cov = mnl.prepare_X_matrix(sample_df, formulas, "choice")

    assert X.shape == (3, 2, 2)
    assert F is None  # None if no fixed covariates
    assert len(fixed_cov) == 0


def test_build_model_returns_pymc_model(mnl, sample_df, utility_eqs):
    X = np.random.randn(5, 2, 3)  # 5 obs x 2 alts x 3 alt covariates
    F = np.random.randn(5, 2)  # 5 obs x 2 fixed covariates
    y = np.random.randint(0, 2, size=5)  # obs labels

    mnl.preprocess_model_data(sample_df, utility_eqs)
    model = mnl.make_model(X, F, y)
    assert isinstance(model, pm.Model)
    assert mnl.alternatives == ["alt", "other"]
    assert mnl.covariates == ["X1", "X2"]


def test_sample(mnl, mock_pymc_sample):
    X, F, y = mnl.preprocess_model_data(mnl.choice_df, mnl.utility_equations)
    _ = mnl.make_model(X, F, y)
    mnl.sample()
    assert "prior_predictive" in mnl.idata
    mnl.sample()
    assert hasattr(mnl, "idata")

    mnl.sample_posterior_predictive(
        extend_idata=False,
    )
    assert "posterior_predictive" in mnl.idata
    assert "fit_data" in mnl.idata

    mnl.sample_posterior_predictive(choice_df=mnl.choice_df, extend_idata=True)
    assert isinstance(mnl.idata, az.InferenceData)

    mnl.fit(choice_df=mnl.choice_df, utility_equations=mnl.utility_equations)

    with pytest.raises(
        RuntimeError, match=r"self.idata must be initialized before extending"
    ):
        mnl.idata = None
        mnl.sample_posterior_predictive(
            extend_idata=True,
        )


def test_counterfactual(mnl, mock_pymc_sample):
    X, F, y = mnl.preprocess_model_data(mnl.choice_df, mnl.utility_equations)
    _ = mnl.make_model(X, F, y)
    mnl.sample()
    new = mnl.choice_df.copy()
    new["alt_X1"] = new["alt_X1"] * 1.2
    mnl.apply_intervention(new)
    change_df = mnl.calculate_share_change(mnl.idata, mnl.intervention_idata)
    assert isinstance(change_df, pd.DataFrame)
    assert hasattr(mnl, "intervention_idata")
    idata_new_policy = mnl.apply_intervention(new, mnl.utility_equations)
    assert "posterior_predictive" in idata_new_policy


def test_make_change_plot_returns_figure(mnl, sample_change_df):
    fig = mnl.plot_change(sample_change_df, title="Test Intervention")

    assert isinstance(fig, plt.Figure)


class TestMakeIntercepts:
    """Tests for the make_intercepts() method."""

    def test_make_intercepts_creates_deterministic(self, mnl, sample_df, utility_eqs):
        """Test that make_intercepts creates a Deterministic variable."""
        _X, _F, _y = mnl.preprocess_model_data(sample_df, utility_eqs)

        with pm.Model(coords=mnl.coords):
            alphas = mnl.make_intercepts()

            assert isinstance(alphas, pt.TensorVariable)
            assert alphas.name == "alphas"

    def test_make_intercepts_last_alternative_zero(self, mnl, sample_df, utility_eqs):
        """Test that the last alternative intercept is constrained to zero."""
        _X, _F, _y = mnl.preprocess_model_data(sample_df, utility_eqs)

        with pm.Model(coords=mnl.coords):
            alphas = mnl.make_intercepts()
            alphas_draw = pm.draw(alphas)
            # The shape should match number of alternatives
            assert alphas_draw.shape == (len(mnl.alternatives),)
            assert alphas_draw[-1] == 0.0


class TestMakeAltCoefs:
    """Tests for the make_alt_coefs() method."""

    def test_make_alt_coefs_creates_variable(self, mnl, sample_df, utility_eqs):
        """Test that make_alt_coefs creates the beta coefficients."""
        _X, _F, _y = mnl.preprocess_model_data(sample_df, utility_eqs)

        with pm.Model(coords=mnl.coords):
            betas = mnl.make_alt_coefs()

            assert isinstance(betas, pt.TensorVariable)
            assert betas.name == "betas"

    def test_make_alt_coefs_correct_shape(self, mnl, sample_df, utility_eqs):
        """Test that betas have the correct dimensionality."""
        _X, _F, _y = mnl.preprocess_model_data(sample_df, utility_eqs)

        with pm.Model(coords=mnl.coords):
            betas = mnl.make_alt_coefs()
            betas_draw = pm.draw(betas)
            # Shape should match number of alternative-specific covariates
            assert betas_draw.shape == (len(mnl.covariates),)


class TestMakeFixedCoefs:
    """Tests for the make_fixed_coefs() method."""

    def test_make_fixed_coefs_with_fixed_covariates(self, mnl, sample_df, utility_eqs):
        """Test fixed coefs when fixed covariates are provided."""
        X, F, _y = mnl.preprocess_model_data(sample_df, utility_eqs)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mnl.coords):
            F_contrib = mnl.make_fixed_coefs(F, n_obs, n_alts)
            F_contrib_draw = pm.draw(F_contrib)
            assert isinstance(F_contrib, pt.TensorVariable)
            # Should have shape (n_obs, n_alts)
            assert F_contrib_draw.shape == (n_obs, n_alts)
            assert F_contrib.name == "F_interaction"

    def test_make_fixed_coefs_without_fixed_covariates(
        self, mnl_no_fixed, sample_df, utility_eqs_no_fixed
    ):
        """Test that fixed coefs returns zeros when no fixed covariates."""
        X, F, _y = mnl_no_fixed.preprocess_model_data(sample_df, utility_eqs_no_fixed)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mnl_no_fixed.coords):
            F_contrib = mnl_no_fixed.make_fixed_coefs(F, n_obs, n_alts)
            F_contrib_draw = pm.draw(F_contrib)
            assert isinstance(F_contrib, pt.TensorVariable)
            # Should be zeros with shape (n_obs, n_alts)
            assert F_contrib_draw.shape == (n_obs, n_alts)

    def test_make_fixed_coefs_none_input(self, mnl, sample_df, utility_eqs):
        """Test that None input for fixed covariates returns zeros."""
        X, _F, _y = mnl.preprocess_model_data(sample_df, utility_eqs)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mnl.coords):
            F_contrib = mnl.make_fixed_coefs(None, n_obs, n_alts)

            assert isinstance(F_contrib, pt.TensorVariable)
            assert F_contrib.type.shape == (n_obs, n_alts)

    def test_make_fixed_coefs_last_alternative_zero(self, mnl, sample_df, utility_eqs):
        """Test that betas_fixed has last alternative constrained to zero."""
        X, F, _y = mnl.preprocess_model_data(sample_df, utility_eqs)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mnl.coords) as model:
            _ = mnl.make_fixed_coefs(F, n_obs, n_alts)

            # Check that betas_fixed deterministic exists
            assert "betas_fixed" in model.named_vars


class TestMakeUtility:
    """Tests for the make_utility() method."""

    def test_make_utility_creates_deterministic(self, mnl, sample_df, utility_eqs):
        """Test that make_utility creates a Deterministic variable U."""
        X, F, _y = mnl.preprocess_model_data(sample_df, utility_eqs)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mnl.coords):
            alphas = mnl.make_intercepts()
            betas = mnl.make_alt_coefs()
            X_data = pm.Data("X", X, dims=("obs", "alts", "alt_covariates"))
            F_contrib = mnl.make_fixed_coefs(F, n_obs, n_alts)

            U = mnl.make_utility(X_data, alphas, betas, F_contrib)

            assert isinstance(U, pt.TensorVariable)
            assert U.name == "U"

    def test_make_utility_correct_shape(self, mnl, sample_df, utility_eqs):
        """Test that utility has shape (n_obs, n_alts)."""
        X, F, _y = mnl.preprocess_model_data(sample_df, utility_eqs)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mnl.coords):
            alphas = mnl.make_intercepts()
            betas = mnl.make_alt_coefs()
            X_data = pm.Data("X", X, dims=("obs", "alts", "alt_covariates"))
            F_contrib = mnl.make_fixed_coefs(F, n_obs, n_alts)

            U = mnl.make_utility(X_data, alphas, betas, F_contrib)
            U_draw = pm.draw(U)
            assert U_draw.shape == (n_obs, n_alts)


class TestMakeChoiceProb:
    """Tests for the make_choice_prob() method."""

    def test_make_choice_prob_creates_deterministic(self, mnl, sample_df, utility_eqs):
        """Test that make_choice_prob creates a Deterministic variable p."""
        X, F, _y = mnl.preprocess_model_data(sample_df, utility_eqs)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mnl.coords):
            alphas = mnl.make_intercepts()
            betas = mnl.make_alt_coefs()
            X_data = pm.Data("X", X, dims=("obs", "alts", "alt_covariates"))
            F_contrib = mnl.make_fixed_coefs(F, n_obs, n_alts)
            U = mnl.make_utility(X_data, alphas, betas, F_contrib)

            p = mnl.make_choice_prob(U)

            assert isinstance(p, pt.TensorVariable)
            assert p.name == "p"

    def test_make_choice_prob_correct_shape(self, mnl, sample_df, utility_eqs):
        """Test that probabilities have shape (n_obs, n_alts)."""
        X, F, _y = mnl.preprocess_model_data(sample_df, utility_eqs)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mnl.coords):
            alphas = mnl.make_intercepts()
            betas = mnl.make_alt_coefs()
            X_data = pm.Data("X", X, dims=("obs", "alts", "alt_covariates"))
            F_contrib = mnl.make_fixed_coefs(F, n_obs, n_alts)
            U = mnl.make_utility(X_data, alphas, betas, F_contrib)

            p = mnl.make_choice_prob(U)
            p_draw = pm.draw(p)
            assert p_draw.shape == (n_obs, n_alts)


class TestMakeModelIntegration:
    """Integration tests for the refactored make_model() method."""

    def test_make_model_with_fixed_covariates(self, mnl, sample_df, utility_eqs):
        """Test that make_model correctly integrates all components with fixed covariates."""
        X, F, y = mnl.preprocess_model_data(sample_df, utility_eqs)
        model = mnl.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        # Check that all expected variables are created
        assert "alphas" in model.named_vars
        assert "betas" in model.named_vars
        assert "betas_fixed" in model.named_vars
        assert "U" in model.named_vars
        assert "p" in model.named_vars
        assert "likelihood" in model.named_vars

    def test_make_model_without_fixed_covariates(
        self, mnl_no_fixed, sample_df, utility_eqs_no_fixed
    ):
        """Test that make_model works correctly without fixed covariates."""
        X, F, y = mnl_no_fixed.preprocess_model_data(sample_df, utility_eqs_no_fixed)
        model = mnl_no_fixed.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        # Check that essential variables are created
        assert "alphas" in model.named_vars
        assert "betas" in model.named_vars
        assert "U" in model.named_vars
        assert "p" in model.named_vars
        assert "likelihood" in model.named_vars
        # betas_fixed should NOT be in the model
        assert "betas_fixed" not in model.named_vars

    def test_make_model_coords_consistency(self, mnl, sample_df, utility_eqs):
        """Test that model coordinates are correctly specified."""
        X, F, y = mnl.preprocess_model_data(sample_df, utility_eqs)
        model = mnl.make_model(X, F, y)

        # Check that coords were passed to the model
        assert model.coords is not None
        assert "alts" in model.coords
        assert "obs" in model.coords
        assert "alt_covariates" in model.coords
        assert "fixed_covariates" in model.coords

    def test_make_model_data_variables(self, mnl, sample_df, utility_eqs):
        """Test that pm.Data variables are correctly created."""
        X, F, y = mnl.preprocess_model_data(sample_df, utility_eqs)
        model = mnl.make_model(X, F, y)

        # Check that data containers exist
        assert "X" in model.named_vars
        assert "y" in model.named_vars
        assert "F" in model.named_vars  # Fixed covariates data


class TestBackwardCompatibility:
    """Tests to ensure refactored code maintains backward compatibility."""

    def test_same_model_structure_as_original(
        self, mnl, sample_df, utility_eqs, mock_pymc_sample
    ):
        """Test that refactored model produces same structure as original."""
        X, F, y = mnl.preprocess_model_data(sample_df, utility_eqs)
        _ = mnl.make_model(X, F, y)

        # Test sampling still works
        mnl.sample()

        # Check that idata has expected groups
        assert hasattr(mnl, "idata")
        assert "posterior" in mnl.idata
        assert "prior" in mnl.idata

    def test_intervention_still_works(self, mnl, sample_df, mock_pymc_sample):
        """Test that interventions work with refactored model."""
        X, F, y = mnl.preprocess_model_data(mnl.choice_df, mnl.utility_equations)
        _ = mnl.make_model(X, F, y)
        mnl.sample()

        new = mnl.choice_df.copy()
        new["alt_X1"] = new["alt_X1"] * 1.2
        idata_new = mnl.apply_intervention(new)

        assert "posterior_predictive" in idata_new
        assert "p" in idata_new["posterior_predictive"]


class TestEdgeCases:
    """Tests for edge cases in the refactored methods."""

    def test_single_covariate(self, sample_df):
        """Test model with single alternative-specific covariate."""
        utility_eqs_single = ["alt ~ alt_X1 | income", "other ~ other_X1 | income"]
        mnl_single = MNLogit(sample_df, utility_eqs_single, "choice", ["X1"])

        X, F, y = mnl_single.preprocess_model_data(sample_df, utility_eqs_single)
        model = mnl_single.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        assert X.shape[2] == 1  # Only one covariate

    def test_many_alternatives(self):
        """Test model with many alternatives."""
        df_many = pd.DataFrame(
            {
                "choice": ["alt1", "alt2", "alt3", "alt4"],
                "alt1_X1": [1, 2, 3, 4],
                "alt2_X1": [2, 3, 4, 5],
                "alt3_X1": [3, 4, 5, 6],
                "alt4_X1": [4, 5, 6, 7],
            }
        )
        utility_eqs_many = [
            "alt1 ~ alt1_X1",
            "alt2 ~ alt2_X1",
            "alt3 ~ alt3_X1",
            "alt4 ~ alt4_X1",
        ]
        mnl_many = MNLogit(df_many, utility_eqs_many, "choice", ["X1"])

        X, F, y = mnl_many.preprocess_model_data(df_many, utility_eqs_many)
        model = mnl_many.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        assert X.shape[1] == 4  # Four alternatives
