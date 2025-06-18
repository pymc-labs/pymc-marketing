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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
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
    X, F, alts, fixed_cov = mnl.prepare_X_matrix(sample_df, formulas, "choice")

    assert X.shape == (3, 2, 2)
    assert F == []  # Empty list if no fixed covariates
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
    assert hasattr(mnl, "idata")


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
