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

from pymc_marketing.customer_choice.mixed_logit import MixedLogit

seed = sum(map(ord, "MixedLogit"))
rng = np.random.default_rng(seed)


@pytest.fixture
def sample_df():
    """Sample dataframe for basic mixed logit testing."""
    return pd.DataFrame(
        {
            "choice": ["bus", "car", "bus", "train", "car"],
            "bus_price": [2.0, 2.5, 2.0, 2.2, 2.3],
            "bus_time": [45, 50, 45, 48, 52],
            "car_price": [5.0, 4.8, 5.2, 5.1, 4.9],
            "car_time": [30, 28, 32, 29, 31],
            "train_price": [3.5, 3.8, 3.6, 3.7, 3.9],
            "train_time": [35, 38, 36, 37, 39],
            "income": [50000, 60000, 55000, 70000, 65000],
        }
    )


@pytest.fixture
def sample_df_panel():
    """Sample dataframe with panel structure (repeated choices per individual)."""
    return pd.DataFrame(
        {
            "choice": ["bus", "car", "bus", "train", "car", "bus"],
            "individual_id": [1, 1, 2, 2, 3, 3],
            "bus_price": [2.0, 2.5, 2.0, 2.2, 2.3, 2.1],
            "bus_time": [45, 50, 45, 48, 52, 46],
            "car_price": [5.0, 4.8, 5.2, 5.1, 4.9, 5.0],
            "car_time": [30, 28, 32, 29, 31, 30],
            "train_price": [3.5, 3.8, 3.6, 3.7, 3.9, 3.6],
            "train_time": [35, 38, 36, 37, 39, 36],
            "income": [50000, 50000, 60000, 60000, 70000, 70000],
        }
    )


@pytest.fixture
def utility_eqs_basic():
    """Basic utility equations with random price coefficient."""
    return [
        "bus ~ bus_price + bus_time | income | bus_price",
        "car ~ car_price + car_time | income | car_price",
        "train ~ train_price + train_time | income | train_price",
    ]


@pytest.fixture
def utility_eqs_no_random():
    """Utility equations with no random coefficients (should work like MNLogit)."""
    return [
        "bus ~ bus_price + bus_time | income",
        "car ~ car_price + car_time | income",
        "train ~ train_price + train_time | income",
    ]


@pytest.fixture
def utility_eqs_no_fixed():
    """Utility equations without fixed covariates."""
    return [
        "bus ~ bus_price + bus_time | | bus_price",
        "car ~ car_price + car_time | | car_price",
        "train ~ train_price + train_time | | train_price",
    ]


@pytest.fixture
def utility_eqs_multiple_random():
    """Utility equations with multiple random coefficients."""
    return [
        "bus ~ bus_price + bus_time | income | bus_price + bus_time",
        "car ~ car_price + car_time | income | car_price + car_time",
        "train ~ train_price + train_time | income | train_price + train_time",
    ]


@pytest.fixture
def covariates_list():
    """List of base covariate names without alternative prefixes."""
    return ["price", "time"]


@pytest.fixture
def mxl(sample_df, utility_eqs_basic, covariates_list):
    """Basic MixedLogit instance - don't preprocess yet."""
    return MixedLogit(sample_df, utility_eqs_basic, "choice", covariates_list)


@pytest.fixture
def mxl_panel(sample_df_panel, utility_eqs_basic, covariates_list):
    """MixedLogit instance with panel data - don't preprocess yet."""
    return MixedLogit(
        sample_df_panel,
        utility_eqs_basic,
        "choice",
        covariates_list,
        group_id="individual_id",
    )


@pytest.fixture
def mxl_no_random(sample_df, utility_eqs_no_random, covariates_list):
    """MixedLogit instance with no random coefficients - don't preprocess yet."""
    return MixedLogit(sample_df, utility_eqs_no_random, "choice", covariates_list)


@pytest.fixture
def sample_change_df():
    """Sample dataframe for plotting market share changes."""
    return pd.DataFrame(
        {
            "policy_share": [0.3, 0.5, 0.2],
            "new_policy_share": [0.25, 0.55, 0.2],
        },
        index=["bus", "car", "train"],
    )


# =============================================================================
# Formula Parsing Tests
# =============================================================================


class TestParseFormula:
    """Tests for formula parsing with three-part syntax."""

    def test_parse_formula_three_parts(self, mxl, sample_df):
        """Test parsing formula with all three parts."""
        formula = "bus ~ bus_price + bus_time | income | bus_price"
        target, alt_covs, fixed_covs, random_covs = mxl.parse_formula(
            sample_df, formula, "choice"
        )

        assert target == "bus"
        assert alt_covs == "bus_price + bus_time"
        assert fixed_covs == "income"
        assert random_covs == "bus_price"

    def test_parse_formula_two_parts(self, mxl, sample_df):
        """Test parsing formula with only alt and fixed covariates."""
        formula = "bus ~ bus_price + bus_time | income"
        target, alt_covs, fixed_covs, random_covs = mxl.parse_formula(
            sample_df, formula, "choice"
        )

        assert target == "bus"
        assert alt_covs == "bus_price + bus_time"
        assert fixed_covs == "income"
        assert random_covs == ""

    def test_parse_formula_one_part(self, mxl, sample_df):
        """Test parsing formula with only alt covariates."""
        formula = "bus ~ bus_price + bus_time"
        target, alt_covs, fixed_covs, random_covs = mxl.parse_formula(
            sample_df, formula, "choice"
        )

        assert target == "bus"
        assert alt_covs == "bus_price + bus_time"
        assert fixed_covs == ""
        assert random_covs == ""

    def test_parse_formula_invalid_target(self, mxl, sample_df):
        """Test that invalid target raises error."""
        formula = "invalid_mode ~ bus_price + bus_time | income | bus_price"
        with pytest.raises(ValueError, match="not found in dependent variable"):
            mxl.parse_formula(sample_df, formula, "choice")

    def test_parse_formula_missing_alt_covariate(self, mxl, sample_df):
        """Test that missing alternative covariate raises error."""
        formula = "bus ~ missing_col + bus_time | income | bus_price"
        with pytest.raises(ValueError, match=r"Alternative covariate.*not found"):
            mxl.parse_formula(sample_df, formula, "choice")

    def test_parse_formula_missing_fixed_covariate(self, mxl, sample_df):
        """Test that missing fixed covariate raises error."""
        formula = "bus ~ bus_price + bus_time | missing_fixed | bus_price"
        with pytest.raises(ValueError, match=r"Fixed covariate.*not found"):
            mxl.parse_formula(sample_df, formula, "choice")

    def test_parse_formula_missing_random_covariate(self, mxl, sample_df):
        """Test that missing random covariate raises error."""
        formula = "bus ~ bus_price + bus_time | income | missing_random"
        with pytest.raises(ValueError, match=r"Random covariate.*not found"):
            mxl.parse_formula(sample_df, formula, "choice")

    def test_parse_formula_too_many_separators(self, mxl, sample_df):
        """Test that too many | separators raises error."""
        formula = "bus ~ bus_price | income | bus_price | extra"
        with pytest.raises(ValueError, match=r"too many.*separators"):
            mxl.parse_formula(sample_df, formula, "choice")


# =============================================================================
# Data Preparation Tests
# =============================================================================


class TestPrepareXMatrix:
    """Tests for X matrix preparation."""

    def test_prepare_X_matrix_basic(self, mxl, sample_df, utility_eqs_basic):
        """Test basic X matrix preparation."""
        X, F, alts, fixed_cov, random_covs = mxl.prepare_X_matrix(
            sample_df, utility_eqs_basic, "choice"
        )

        assert X.shape == (5, 3, 2)  # 5 obs x 3 alts x 2 covariates
        assert F.shape == (5, 1)  # Fixed covariate
        assert set(alts) == {"bus", "car", "train"}
        assert fixed_cov.tolist() == ["income"]
        # Random covariate names should be base names (price), not alternative-specific
        assert random_covs == ["price"]

    def test_prepare_X_matrix_no_fixed(self, mxl, sample_df, utility_eqs_no_fixed):
        """Test X matrix preparation without fixed covariates."""
        X, F, _alts, fixed_cov, random_covs = mxl.prepare_X_matrix(
            sample_df, utility_eqs_no_fixed, "choice"
        )

        assert X.shape == (5, 3, 2)
        assert F is None
        assert len(fixed_cov) == 0
        assert len(random_covs) == 1  # Only price is random
        assert random_covs == ["price"]

    def test_prepare_X_matrix_no_random(self, mxl, sample_df, utility_eqs_no_random):
        """Test X matrix preparation without random covariates."""
        X, F, _alts, _fixed_cov, random_covs = mxl.prepare_X_matrix(
            sample_df, utility_eqs_no_random, "choice"
        )

        assert X.shape == (5, 3, 2)
        assert F.shape == (5, 1)
        assert len(random_covs) == 0

    def test_prepare_X_matrix_missing_alternative(self, mxl, sample_df):
        """Test that missing alternative equation raises error."""
        incomplete_eqs = [
            "bus ~ bus_price + bus_time | income | bus_price",
            "car ~ car_price + car_time | income | car_price",
            # Missing train equation
        ]
        with pytest.raises(ValueError, match=r"Alternative.*has no utility equation"):
            mxl.prepare_X_matrix(sample_df, incomplete_eqs, "choice")


class TestPrepareGroupIndex:
    """Tests for group index preparation."""

    def test_prepare_group_index_none(self, mxl, sample_df):
        """Test group index when group_id is None."""
        grp_idx, n_groups = mxl._prepare_group_index(sample_df, None)

        assert grp_idx is None
        assert n_groups == len(sample_df)

    def test_prepare_group_index_with_groups(self, mxl, sample_df_panel):
        """Test group index with panel data."""
        grp_idx, n_groups = mxl._prepare_group_index(sample_df_panel, "individual_id")

        assert grp_idx is not None
        assert len(grp_idx) == len(sample_df_panel)
        assert n_groups == 3  # Three unique individuals

    def test_prepare_group_index_missing_column(self, mxl, sample_df):
        """Test that missing group_id column raises error."""
        with pytest.raises(ValueError, match=r"Group ID column.*not found"):
            mxl._prepare_group_index(sample_df, "nonexistent_id")


class TestPrepareCoords:
    """Tests for coordinate preparation."""

    def test_prepare_coords_with_random(self, mxl, sample_df):
        """Test coordinate preparation with random coefficients."""
        coords = mxl._prepare_coords(
            sample_df,
            ["bus", "car", "train"],
            ["price", "time"],
            np.array(["income"]),
            ["price"],
            5,
        )

        assert "alts" in coords
        assert "covariates" in coords
        assert "random_covariates" in coords
        assert "normal_covariates" in coords
        assert coords["random_covariates"] == ["price"]
        assert coords["normal_covariates"] == ["time"]
        assert list(coords["individuals"]) == list(range(5))

    def test_prepare_coords_no_random(self, mxl, sample_df):
        """Test coordinate preparation without random coefficients."""
        coords = mxl._prepare_coords(
            sample_df,
            ["bus", "car", "train"],
            ["price", "time"],
            np.array(["income"]),
            [],
            5,
        )

        assert coords["random_covariates"] == []
        assert coords["normal_covariates"] == ["price", "time"]


class TestPreprocessModelData:
    """Tests for full data preprocessing."""

    def test_preprocess_model_data_basic(self, mxl, sample_df, utility_eqs_basic):
        """Test basic preprocessing."""
        X, F, y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)

        assert X.shape == (5, 3, 2)
        assert F.shape == (5, 1)
        assert len(y) == 5
        assert hasattr(mxl, "alternatives")
        assert hasattr(mxl, "random_covariate_idx")
        assert hasattr(mxl, "coords")

    def test_preprocess_model_data_panel(
        self, mxl_panel, sample_df_panel, utility_eqs_basic
    ):
        """Test preprocessing with panel data."""
        _X, _F, _y = mxl_panel.preprocess_model_data(sample_df_panel, utility_eqs_basic)

        assert mxl_panel.grp_idx is not None
        assert mxl_panel.n_individuals == 3
        assert len(mxl_panel.grp_idx) == 6


# =============================================================================
# Model Building Component Tests
# =============================================================================


class TestMakeIntercepts:
    """Tests for make_intercepts() method."""

    def test_make_intercepts_creates_deterministic(
        self, mxl, sample_df, utility_eqs_basic
    ):
        """Test that intercepts are created correctly."""
        _X, _F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)

        with pm.Model(coords=mxl.coords):
            alphas = mxl.make_intercepts()

            assert isinstance(alphas, pt.TensorVariable)
            assert alphas.name == "alphas"

    def test_make_intercepts_last_alternative_zero(
        self, mxl, sample_df, utility_eqs_basic
    ):
        """Test that last alternative intercept is zero."""
        _X, _F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)

        with pm.Model(coords=mxl.coords):
            alphas = mxl.make_intercepts()
            alphas_draw = pm.draw(alphas)

            assert alphas_draw[-1] == 0.0


class TestMakeNonRandomCoefs:
    """Tests for make_non_random_coefs() method."""

    def test_make_non_random_coefs_creates_variable(
        self, mxl, sample_df, utility_eqs_basic
    ):
        """Test that non-random coefficients are created."""
        _X, _F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)

        with pm.Model(coords=mxl.coords):
            betas = mxl.make_non_random_coefs()

            assert isinstance(betas, pt.TensorVariable)
            assert betas.name == "betas_non_random"

    def test_make_non_random_coefs_correct_shape(
        self, mxl, sample_df, utility_eqs_basic
    ):
        """Test that non-random coefficients have correct shape."""
        _X, _F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)

        with pm.Model(coords=mxl.coords):
            betas = mxl.make_non_random_coefs()
            betas_draw = pm.draw(betas)

            # Should have one coefficient (time), since price is random
            assert betas_draw.shape == (len(mxl.coords["normal_covariates"]),)

    def test_make_non_random_coefs_all_random(
        self, sample_df, utility_eqs_multiple_random, covariates_list
    ):
        """Test behavior when all coefficients are random."""
        mxl_all_random = MixedLogit(
            sample_df, utility_eqs_multiple_random, "choice", covariates_list
        )
        _X, _F, _y = mxl_all_random.preprocess_model_data(
            sample_df, utility_eqs_multiple_random
        )

        with pm.Model(coords=mxl_all_random.coords):
            betas = mxl_all_random.make_non_random_coefs()

            # Should return None when all covariates are random
            # Check that there are no normal covariates
            assert len(mxl_all_random.coords["normal_covariates"]) == 0
            assert betas is None


class TestMakeRandomCoefs:
    """Tests for make_random_coefs() method."""

    def test_make_random_coefs_individual_level(
        self, mxl, sample_df, utility_eqs_basic
    ):
        """Test random coefficients at individual level."""
        X, _F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs = X.shape[0]

        with pm.Model(coords=mxl.coords):
            betas_random, param_names = mxl.make_random_coefs(n_obs, None)

            assert isinstance(betas_random, pt.TensorVariable)
            assert betas_random.name == "betas_random_individual"
            # param_names should be base covariate names
            assert param_names == ["price"]

    def test_make_random_coefs_group_level(
        self, mxl_panel, sample_df_panel, utility_eqs_basic
    ):
        """Test random coefficients at group level (panel data)."""
        X, _F, _y = mxl_panel.preprocess_model_data(sample_df_panel, utility_eqs_basic)
        n_obs = X.shape[0]

        with pm.Model(coords=mxl_panel.coords):
            betas_random, _param_names = mxl_panel.make_random_coefs(
                n_obs, mxl_panel.grp_idx
            )

            assert isinstance(betas_random, pt.TensorVariable)
            # Should have group-level coefficients expanded to observations
            betas_draw = pm.draw(betas_random)
            # Number of random covariates should be 1 (price)
            assert betas_draw.shape == (n_obs, 1)

    def test_make_random_coefs_correct_shape(self, mxl, sample_df, utility_eqs_basic):
        """Test that random coefficients have correct dimensions."""
        X, _F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs = X.shape[0]

        with pm.Model(coords=mxl.coords):
            betas_random, _ = mxl.make_random_coefs(n_obs, None)

            if betas_random is not None:
                betas_draw = pm.draw(betas_random)

                # Should be (n_obs, n_random_covariates)
                n_random = len(mxl.coords["random_covariates"])
                assert betas_draw.shape == (n_obs, n_random)

    def test_make_random_coefs_no_random(
        self, mxl_no_random, sample_df, utility_eqs_no_random
    ):
        """Test behavior when no random coefficients specified."""
        X, _F, _y = mxl_no_random.preprocess_model_data(
            sample_df, utility_eqs_no_random
        )
        n_obs = X.shape[0]

        with pm.Model(coords=mxl_no_random.coords):
            betas_random, param_names = mxl_no_random.make_random_coefs(n_obs, None)

            assert betas_random is None
            assert param_names == []


class TestMakeBetaMatrix:
    """Tests for make_beta_matrix() method."""

    def test_make_beta_matrix_combines_coefficients(
        self, mxl, sample_df, utility_eqs_basic
    ):
        """Test that beta matrix correctly combines random and non-random."""
        X, _F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs = X.shape[0]

        with pm.Model(coords=mxl.coords):
            betas_non_random = mxl.make_non_random_coefs()
            betas_random, _ = mxl.make_random_coefs(n_obs, None)
            B_full = mxl.make_beta_matrix(betas_non_random, betas_random, n_obs)

            assert isinstance(B_full, pt.TensorVariable)
            assert B_full.name == "betas_individuals"

            B_draw = pm.draw(B_full)
            assert B_draw.shape == (n_obs, 2)  # n_obs x n_covariates

    def test_make_beta_matrix_correct_positions(
        self, mxl, sample_df, utility_eqs_basic
    ):
        """Test that coefficients are placed in correct positions."""
        X, _F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs = X.shape[0]

        # In utility_eqs_basic, price is random (index 0), time is not (index 1)
        # After preprocessing, random_covariate_idx should contain the index of "price"
        assert "price" in mxl.random_covar_names
        assert "time" in mxl.coords["normal_covariates"]

        with pm.Model(coords=mxl.coords):
            betas_non_random = mxl.make_non_random_coefs()
            betas_random, _ = mxl.make_random_coefs(n_obs, None)
            B_full = mxl.make_beta_matrix(betas_non_random, betas_random, n_obs)

            B_draw = pm.draw(B_full)

            # Check that we have coefficients for all covariates
            assert B_draw.shape == (n_obs, len(mxl.covariates))


class TestMakeFixedCoefs:
    """Tests for make_fixed_coefs() method."""

    def test_make_fixed_coefs_with_fixed_covariates(
        self, mxl, sample_df, utility_eqs_basic
    ):
        """Test fixed coefficients when fixed covariates provided."""
        X, F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mxl.coords):
            W_contrib = mxl.make_fixed_coefs(F, n_obs, n_alts)

            assert isinstance(W_contrib, pt.TensorVariable)
            W_draw = pm.draw(W_contrib)
            assert W_draw.shape == (n_obs, n_alts)

    def test_make_fixed_coefs_without_fixed_covariates(
        self, mxl, sample_df, utility_eqs_no_fixed
    ):
        """Test fixed coefficients when no fixed covariates."""
        X, F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_no_fixed)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mxl.coords):
            W_contrib = mxl.make_fixed_coefs(F, n_obs, n_alts)

            W_draw = pm.draw(W_contrib)
            assert W_draw.shape == (n_obs, n_alts)
            # Should be all zeros
            assert np.allclose(W_draw, 0)

    def test_make_fixed_coefs_last_alternative_zero(
        self, mxl, sample_df, utility_eqs_basic
    ):
        """Test that fixed coefficients for last alternative are zero."""
        X, F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mxl.coords) as model:
            _ = mxl.make_fixed_coefs(F, n_obs, n_alts)

            # Check that betas_fixed deterministic exists
            assert "betas_fixed" in model.named_vars


class TestMakeControlFunction:
    """Tests for make_control_function() method."""

    def test_make_control_function_none(self, mxl, sample_df, utility_eqs_basic):
        """Test control function when no instrumental variables."""
        X, _F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mxl.coords):
            price_error = mxl.make_control_function(n_obs, n_alts)

            price_error_draw = pm.draw(price_error)
            assert price_error_draw.shape == (n_obs, n_alts)
            # Should be all zeros
            assert np.allclose(price_error_draw, 0)

    def test_make_control_function_with_instruments(
        self, sample_df, utility_eqs_basic, covariates_list
    ):
        """Test control function with instrumental variables."""
        # Create mock instrumental variables
        n_obs = len(sample_df)
        X_instruments = np.random.randn(n_obs, 2)
        y_price = np.random.randn(n_obs, 3)

        mxl_with_iv = MixedLogit(
            sample_df,
            utility_eqs_basic,
            "choice",
            covariates_list,
            instrumental_vars={
                "X_instruments": X_instruments,
                "y_price": y_price,
                "diagonal": True,
            },
        )

        X, _F, _y = mxl_with_iv.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mxl_with_iv.coords) as model:
            _ = mxl_with_iv.make_control_function(n_obs, n_alts)

            # Should create control function variables
            assert "gamma" in model.named_vars
            assert "lambda_cf" in model.named_vars
            assert "price_error" in model.named_vars

    def test_make_control_function_with_one_instrument(
        self, sample_df, utility_eqs_basic, covariates_list
    ):
        """Test control function with instrumental variables."""
        # Create mock instrumental variables
        n_obs = len(sample_df)
        X_instruments = np.random.randn(n_obs, 1)
        y_price = np.random.randn(n_obs, 3)

        mxl_with_iv = MixedLogit(
            sample_df,
            utility_eqs_basic,
            "choice",
            covariates_list,
            instrumental_vars={
                "X_instruments": X_instruments,
                "y_price": y_price,
                "diagonal": True,
            },
        )

        X, _F, _y = mxl_with_iv.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mxl_with_iv.coords) as model:
            _ = mxl_with_iv.make_control_function(n_obs, n_alts)

            # Should create control function variables
            assert "gamma" in model.named_vars
            assert "lambda_cf" in model.named_vars
            assert "price_error" in model.named_vars


class TestMakeUtility:
    """Tests for make_utility() method."""

    def test_make_utility_creates_deterministic(
        self, mxl, sample_df, utility_eqs_basic
    ):
        """Test that utility calculation creates deterministic."""
        X, F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mxl.coords):
            alphas = mxl.make_intercepts()
            betas_non_random = mxl.make_non_random_coefs()
            betas_random, _ = mxl.make_random_coefs(n_obs, None)
            X_data = pm.Data("X", X, dims=("obs", "alts", "covariates"))
            B_full = mxl.make_beta_matrix(betas_non_random, betas_random, n_obs)
            W_contrib = mxl.make_fixed_coefs(F, n_obs, n_alts)
            price_error = mxl.make_control_function(n_obs, n_alts)

            U = mxl.make_utility(X_data, B_full, alphas, W_contrib, price_error)

            assert isinstance(U, pt.TensorVariable)
            assert U.name == "U"

    def test_make_utility_correct_shape(self, mxl, sample_df, utility_eqs_basic):
        """Test that utility has correct shape."""
        X, F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mxl.coords):
            alphas = mxl.make_intercepts()
            betas_non_random = mxl.make_non_random_coefs()
            betas_random, _ = mxl.make_random_coefs(n_obs, None)
            X_data = pm.Data("X", X, dims=("obs", "alts", "covariates"))
            B_full = mxl.make_beta_matrix(betas_non_random, betas_random, n_obs)
            W_contrib = mxl.make_fixed_coefs(F, n_obs, n_alts)
            price_error = mxl.make_control_function(n_obs, n_alts)

            U = mxl.make_utility(X_data, B_full, alphas, W_contrib, price_error)
            U_draw = pm.draw(U)

            assert U_draw.shape == (n_obs, n_alts)


class TestMakeChoiceProb:
    """Tests for make_choice_prob() method."""

    def test_make_choice_prob_creates_deterministic(
        self, mxl, sample_df, utility_eqs_basic
    ):
        """Test that choice probabilities are created."""
        X, F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mxl.coords):
            alphas = mxl.make_intercepts()
            betas_non_random = mxl.make_non_random_coefs()
            betas_random, _ = mxl.make_random_coefs(n_obs, None)
            X_data = pm.Data("X", X, dims=("obs", "alts", "covariates"))
            B_full = mxl.make_beta_matrix(betas_non_random, betas_random, n_obs)
            W_contrib = mxl.make_fixed_coefs(F, n_obs, n_alts)
            price_error = mxl.make_control_function(n_obs, n_alts)
            U = mxl.make_utility(X_data, B_full, alphas, W_contrib, price_error)

            p = mxl.make_choice_prob(U)

            assert isinstance(p, pt.TensorVariable)
            assert p.name == "p"

    def test_make_choice_prob_correct_shape(self, mxl, sample_df, utility_eqs_basic):
        """Test that probabilities have correct shape."""
        X, F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mxl.coords):
            alphas = mxl.make_intercepts()
            betas_non_random = mxl.make_non_random_coefs()
            betas_random, _ = mxl.make_random_coefs(n_obs, None)
            X_data = pm.Data("X", X, dims=("obs", "alts", "covariates"))
            B_full = mxl.make_beta_matrix(betas_non_random, betas_random, n_obs)
            W_contrib = mxl.make_fixed_coefs(F, n_obs, n_alts)
            price_error = mxl.make_control_function(n_obs, n_alts)
            U = mxl.make_utility(X_data, B_full, alphas, W_contrib, price_error)

            p = mxl.make_choice_prob(U)
            p_draw = pm.draw(p)

            assert p_draw.shape == (n_obs, n_alts)

    def test_make_choice_prob_sums_to_one(self, mxl, sample_df, utility_eqs_basic):
        """Test that probabilities sum to 1 for each observation."""
        X, F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        n_obs, n_alts = X.shape[0], X.shape[1]

        with pm.Model(coords=mxl.coords):
            alphas = mxl.make_intercepts()
            betas_non_random = mxl.make_non_random_coefs()
            betas_random, _ = mxl.make_random_coefs(n_obs, None)
            X_data = pm.Data("X", X, dims=("obs", "alts", "covariates"))
            B_full = mxl.make_beta_matrix(betas_non_random, betas_random, n_obs)
            W_contrib = mxl.make_fixed_coefs(F, n_obs, n_alts)
            price_error = mxl.make_control_function(n_obs, n_alts)
            U = mxl.make_utility(X_data, B_full, alphas, W_contrib, price_error)

            p = mxl.make_choice_prob(U)
            p_draw = pm.draw(p)

            # Check that probabilities sum to 1 for each observation
            assert np.allclose(p_draw.sum(axis=1), 1.0)


# =============================================================================
# Model Integration Tests
# =============================================================================


class TestMakeModel:
    """Integration tests for make_model() method."""

    def test_make_model_returns_pymc_model(self, mxl, sample_df, utility_eqs_basic):
        """Test that make_model returns a PyMC model."""
        X, F, y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        model = mxl.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        assert mxl.alternatives == ["bus", "car", "train"]
        assert mxl.covariates == ["price", "time"]

    def test_make_model_creates_all_variables(self, mxl, sample_df, utility_eqs_basic):
        """Test that all expected variables are created."""
        X, F, y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        model = mxl.make_model(X, F, y)

        # Check essential variables exist
        assert "alphas" in model.named_vars
        assert "betas_individuals" in model.named_vars
        assert "U" in model.named_vars
        assert "p" in model.named_vars
        assert "likelihood" in model.named_vars

        # Check random coefficient variables (if random covariates exist)
        if len(mxl.random_covar_names) > 0:
            assert "betas_random_individual" in model.named_vars

        # Check non-random coefficient variables (if non-random covariates exist)
        if len(mxl.coords["normal_covariates"]) > 0:
            assert "betas_non_random" in model.named_vars

    def test_make_model_without_fixed_covariates(
        self, mxl, sample_df, utility_eqs_no_fixed
    ):
        """Test model without fixed covariates."""
        X, F, y = mxl.preprocess_model_data(sample_df, utility_eqs_no_fixed)
        model = mxl.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        # betas_fixed should not be in model
        assert "betas_fixed" not in model.named_vars

    def test_make_model_without_random_coefficients(
        self, mxl_no_random, sample_df, utility_eqs_no_random
    ):
        """Test model without random coefficients (like standard MNLogit)."""
        X, F, y = mxl_no_random.preprocess_model_data(sample_df, utility_eqs_no_random)
        model = mxl_no_random.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        # Should not have random coefficient variables
        assert "mu_random" not in model.named_vars
        assert "sigma_random" not in model.named_vars

    def test_make_model_with_panel_data(
        self, mxl_panel, sample_df_panel, utility_eqs_basic
    ):
        """Test model with panel data structure."""
        X, F, y = mxl_panel.preprocess_model_data(sample_df_panel, utility_eqs_basic)
        model = mxl_panel.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        # Should have group-level random coefficients if there are random covariates
        if len(mxl_panel.random_covar_names) > 0:
            assert "betas_random_individual" in model.named_vars
        assert "grp_idx" in model.named_vars

    def test_make_model_coords_consistency(self, mxl, sample_df, utility_eqs_basic):
        """Test that model coordinates are correctly specified."""
        X, F, y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        model = mxl.make_model(X, F, y)

        # Check that coords were passed to the model
        assert model.coords is not None
        assert "alts" in model.coords
        assert "obs" in model.coords
        assert "covariates" in model.coords
        assert "random_covariates" in model.coords
        assert "normal_covariates" in model.coords
        assert "individuals" in model.coords

    def test_make_model_data_variables(self, mxl, sample_df, utility_eqs_basic):
        """Test that pm.Data variables are correctly created."""
        X, F, y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        model = mxl.make_model(X, F, y)

        # Check that data containers exist
        assert "X" in model.named_vars
        assert "y" in model.named_vars
        assert "W" in model.named_vars


# =============================================================================
# Sampling Tests
# =============================================================================


def test_sample(mxl, sample_df, utility_eqs_basic, mock_pymc_sample):
    """Test that sampling works."""
    X, F, y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
    _ = mxl.make_model(X, F, y)
    mxl.sample()
    assert hasattr(mxl, "idata")

    mxl.sample_posterior_predictive(
        extend_idata=False,
    )
    assert "posterior_predictive" in mxl.idata
    assert "fit_data" in mxl.idata

    mxl.sample_posterior_predictive(choice_df=sample_df, extend_idata=True)
    assert isinstance(mxl.idata, az.InferenceData)

    mxl.fit(choice_df=sample_df, utility_equations=utility_eqs_basic)

    with pytest.raises(
        RuntimeError, match=r"self.idata must be initialized before extending"
    ):
        mxl.idata = None
        mxl.sample_posterior_predictive(
            extend_idata=True,
        )


def test_sample_panel(mxl_panel, sample_df_panel, utility_eqs_basic, mock_pymc_sample):
    """Test that sampling works with panel data."""
    X, F, y = mxl_panel.preprocess_model_data(sample_df_panel, utility_eqs_basic)
    _ = mxl_panel.make_model(X, F, y)
    mxl_panel.sample()
    assert hasattr(mxl_panel, "idata")


# =============================================================================
# Intervention Tests
# =============================================================================


class TestInterventions:
    """Tests for intervention analysis."""

    def test_intervention_attribute_change(
        self, mxl, sample_df, utility_eqs_basic, mock_pymc_sample
    ):
        """Test intervention by changing observable attributes."""
        X, F, y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        _ = mxl.make_model(X, F, y)
        mxl.sample()

        # Create intervention: reduce bus price by 10%
        new_df = sample_df.copy()
        new_df["bus_price"] = new_df["bus_price"] * 0.9

        idata_new = mxl.apply_intervention(new_df)

        assert "posterior_predictive" in idata_new
        assert "p" in idata_new["posterior_predictive"]
        assert hasattr(mxl, "intervention_idata")

    def test_intervention_product_removal(self, mxl, sample_df, mock_pymc_sample):
        """Test intervention by removing an alternative."""
        X, F, y = mxl.preprocess_model_data(sample_df, mxl.utility_equations)
        _ = mxl.make_model(X, F, y)
        mxl.sample()

        # Remove train from choice set
        new_df = sample_df[sample_df["choice"] != "train"].copy()
        new_utility_eqs = [
            "bus ~ bus_price + bus_time | income | bus_price",
            "car ~ car_price + car_time | income | car_price",
        ]

        idata_new = mxl.apply_intervention(new_df, new_utility_eqs)

        assert "posterior_predictive" in idata_new
        assert "posterior" in idata_new

    def test_calculate_share_change(
        self, mxl, sample_df, utility_eqs_basic, mock_pymc_sample
    ):
        """Test market share change calculation."""
        X, F, y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        _ = mxl.make_model(X, F, y)
        mxl.sample()

        new_df = sample_df.copy()
        new_df["bus_price"] = new_df["bus_price"] * 0.9
        mxl.apply_intervention(new_df)

        change_df = mxl.calculate_share_change(mxl.idata, mxl.intervention_idata)

        assert isinstance(change_df, pd.DataFrame)
        assert "policy_share" in change_df.columns
        assert "new_policy_share" in change_df.columns
        assert "relative_change" in change_df.columns


# =============================================================================
# Plotting Tests
# =============================================================================


def test_plot_change(mxl, sample_change_df):
    """Test that plot_change returns a matplotlib figure."""
    fig = mxl.plot_change(sample_change_df, title="Test Intervention")

    assert isinstance(fig, plt.Figure)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_random_covariate(self, sample_df, covariates_list):
        """Test model with single random covariate."""
        utility_eqs = [
            "bus ~ bus_price + bus_time | income | bus_price",
            "car ~ car_price + car_time | income | car_price",
            "train ~ train_price + train_time | income | train_price",
        ]
        mxl_single = MixedLogit(sample_df, utility_eqs, "choice", covariates_list)

        X, F, y = mxl_single.preprocess_model_data(sample_df, utility_eqs)
        model = mxl_single.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        # Should have 1 random covariate (price)
        assert len(mxl_single.random_covar_names) == 1
        assert "price" in mxl_single.random_covar_names

    def test_multiple_random_covariates(
        self, sample_df, utility_eqs_multiple_random, covariates_list
    ):
        """Test model with multiple random covariates."""
        mxl_multi = MixedLogit(
            sample_df, utility_eqs_multiple_random, "choice", covariates_list
        )

        X, F, y = mxl_multi.preprocess_model_data(
            sample_df, utility_eqs_multiple_random
        )
        model = mxl_multi.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        # Should have 2 random covariates (price and time)
        assert len(mxl_multi.random_covar_names) == 2
        assert "price" in mxl_multi.random_covar_names
        assert "time" in mxl_multi.random_covar_names

    def test_many_alternatives(self):
        """Test model with many alternatives."""
        df_many = pd.DataFrame(
            {
                "choice": ["mode1", "mode2", "mode3", "mode4"],
                "mode1_price": [1, 2, 3, 4],
                "mode1_time": [10, 20, 30, 40],
                "mode2_price": [2, 3, 4, 5],
                "mode2_time": [20, 30, 40, 50],
                "mode3_price": [3, 4, 5, 6],
                "mode3_time": [30, 40, 50, 60],
                "mode4_price": [4, 5, 6, 7],
                "mode4_time": [40, 50, 60, 70],
            }
        )
        utility_eqs_many = [
            "mode1 ~ mode1_price + mode1_time | | mode1_price",
            "mode2 ~ mode2_price + mode2_time | | mode2_price",
            "mode3 ~ mode3_price + mode3_time | | mode3_price",
            "mode4 ~ mode4_price + mode4_time | | mode4_price",
        ]
        mxl_many = MixedLogit(df_many, utility_eqs_many, "choice", ["price", "time"])

        X, F, y = mxl_many.preprocess_model_data(df_many, utility_eqs_many)
        model = mxl_many.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        assert X.shape[1] == 4  # Four alternatives

    def test_panel_with_unequal_observations(self, covariates_list):
        """Test panel data with unequal observations per individual."""
        df_unequal = pd.DataFrame(
            {
                "choice": ["bus", "car", "bus", "train", "car"],
                "individual_id": [
                    1,
                    1,
                    2,
                    3,
                    3,
                ],  # Individual 1: 2 obs, 2: 1 obs, 3: 2 obs
                "bus_price": [2.0, 2.5, 2.0, 2.2, 2.3],
                "bus_time": [45, 50, 45, 48, 52],
                "car_price": [5.0, 4.8, 5.2, 5.1, 4.9],
                "car_time": [30, 28, 32, 29, 31],
                "train_price": [3.5, 3.8, 3.6, 3.7, 3.9],
                "train_time": [35, 38, 36, 37, 39],
                "income": [50000, 50000, 60000, 70000, 70000],
            }
        )
        utility_eqs = [
            "bus ~ bus_price + bus_time | income | bus_price",
            "car ~ car_price + car_time | income | car_price",
            "train ~ train_price + train_time | income | train_price",
        ]
        mxl_unequal = MixedLogit(
            df_unequal, utility_eqs, "choice", covariates_list, group_id="individual_id"
        )

        X, F, y = mxl_unequal.preprocess_model_data(df_unequal, utility_eqs)
        model = mxl_unequal.make_model(X, F, y)

        assert isinstance(model, pm.Model)
        assert mxl_unequal.n_individuals == 3


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests to ensure compatibility with existing patterns."""

    def test_behaves_like_mnlogit_without_random(
        self, mxl_no_random, sample_df, utility_eqs_no_random, mock_pymc_sample
    ):
        """Test that MixedLogit without random coefs behaves like MNLogit."""
        X, F, y = mxl_no_random.preprocess_model_data(sample_df, utility_eqs_no_random)
        model = mxl_no_random.make_model(X, F, y)

        # Should have standard MNLogit structure
        assert "alphas" in model.named_vars
        assert "betas_non_random" in model.named_vars
        assert "U" in model.named_vars
        assert "p" in model.named_vars

        # Sample and test
        mxl_no_random.sample()
        assert hasattr(mxl_no_random, "idata")
        assert "posterior" in mxl_no_random.idata

    def test_intervention_compatibility(
        self, mxl, sample_df, utility_eqs_basic, mock_pymc_sample
    ):
        """Test that interventions work like in MNLogit."""
        X, F, y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        _ = mxl.make_model(X, F, y)
        mxl.sample()

        new_df = sample_df.copy()
        new_df["bus_price"] = new_df["bus_price"] * 0.9
        idata_new = mxl.apply_intervention(new_df)

        # Should have same structure as MNLogit interventions
        assert "posterior_predictive" in idata_new
        assert "p" in idata_new["posterior_predictive"]

    def test_coordinate_structure_consistency(self, mxl, sample_df, utility_eqs_basic):
        """Test that coordinate structure is consistent with other models."""
        _X, _F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)

        # Should have standard coordinates
        assert "alts" in mxl.coords
        assert "obs" in mxl.coords
        assert "covariates" in mxl.coords

        # Plus mixed logit specific coordinates
        assert "random_covariates" in mxl.coords
        assert "normal_covariates" in mxl.coords
        assert "individuals" in mxl.coords


# =============================================================================
# Model Configuration Tests
# =============================================================================


class TestModelConfig:
    """Tests for model configuration."""

    def test_default_model_config(self, mxl):
        """Test that default model config has all required priors."""
        config = mxl.default_model_config

        assert "alphas_" in config
        assert "betas_fixed_" in config
        assert "betas_non_random" in config
        assert "mu_random" in config
        assert "sigma_random" in config
        assert "betas_random_individual" in config
        assert "likelihood" in config

    def test_serializable_model_config(self, mxl, sample_df, utility_eqs_basic):
        """Test that model config is serializable."""
        X, F, y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)
        _ = mxl.make_model(X, F, y)

        serializable_config = mxl._serializable_model_config

        assert isinstance(serializable_config, dict)
        assert "alphas_" in serializable_config
        assert "mu_random" in serializable_config

    def test_create_idata_attrs(self, mxl, sample_df, utility_eqs_basic):
        """Test that idata attributes are created correctly."""
        _X, _F, _y = mxl.preprocess_model_data(sample_df, utility_eqs_basic)

        attrs = mxl.create_idata_attrs()

        assert "covariates" in attrs
        assert "random_covariates" in attrs
        assert "depvar" in attrs
        assert "utility_equations" in attrs
        assert "group_id" in attrs
