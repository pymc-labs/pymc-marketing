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
import pytest

from pymc_marketing.customer_choice.nested_logit import NestedLogit

seed = sum(map(ord, "CustomerChoice"))
rng = np.random.default_rng(seed)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "choice": ["alt", "another", "other", "option"],
            "alt_X1": [1, 2, 3, 7],
            "alt_X2": [4, 5, 6, 5],
            "other_X1": [1, 2, 4, 7],
            "other_X2": [5, 6, 8, 0],
            "option_X1": [4, 6, 7, 1],
            "option_X2": [6, 6, 7, 4],
            "another_X1": [5, 3, 7, 8],
            "another_X2": [7, 4, 2, 8],
            "income": [50000, 60000, 70000, 90002],
        }
    )


@pytest.fixture
def utility_eqs():
    return [
        "alt ~ alt_X1 + alt_X2 | income",
        "other ~ other_X1 + other_X2 | income",
        "option ~ option_X1 + option_X2 | income",
        "another ~ another_X1 + another_X2 | income",
    ]


@pytest.fixture
def utility_eqs_non_fixed():
    return [
        "alt ~ alt_X1 + alt_X2 ",
        "other ~ other_X1 + other_X2 ",
        "option ~ option_X1 + option_X2 ",
        "another ~ another_X1 + another_X2",
    ]


@pytest.fixture
def new_utility_eqs():
    return [
        "alt ~ alt_X1 + alt_X2 ",
        "other ~ other_X1 + other_X2",
        "option ~ option_X1 + option_X2",
    ]


@pytest.fixture
def nesting_structure_1():
    """Single-layer nesting structure."""
    return {"nest1": ["alt", "other"], "nest2": ["option", "another"]}


@pytest.fixture
def nesting_structure_invalid_two_layer():
    """Two-layer nesting structure that should raise an error."""
    return {
        "nest1": ["alt"],
        "nest2": {"option": ["option"], "another": ["other", "another"]},
    }


@pytest.fixture
def nstL(sample_df, utility_eqs, nesting_structure_1):
    return NestedLogit(
        sample_df, utility_eqs, "choice", ["X1", "X2"], nesting_structure_1
    )


@pytest.fixture
def nstL2_invalid(sample_df, utility_eqs, nesting_structure_invalid_two_layer):
    """NestedLogit instance that will fail with two-layer nesting."""
    return NestedLogit(
        sample_df,
        utility_eqs,
        "choice",
        ["X1", "X2"],
        nesting_structure_invalid_two_layer,
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


def test_parse_nesting_standard(nstL):
    """Test parsing of single-layer nesting structure."""
    nesting_dict = {"nest1": ["alt", "other"], "nest2": ["option", "another"]}
    product_indices = {"option": 0, "another": 1, "alt": 2, "other": 3}

    nest_indices = nstL._parse_nesting(nesting_dict, product_indices)

    assert np.array_equal(nest_indices["nest1"], np.array([2, 3]))
    assert np.array_equal(nest_indices["nest2"], np.array([0, 1]))


def test_parse_nesting_no_midlevel(nstL):
    """Test that single nest works correctly."""
    nesting_dict = {"nest1": ["alt", "option", "another"]}
    product_indices = {"alt": 0, "option": 1, "another": 2}

    nest_indices = nstL._parse_nesting(nesting_dict, product_indices)

    assert np.array_equal(nest_indices["nest1"], np.array([0, 1, 2]))


def test_parse_nesting_two_layer_raises(nstL):
    """Test that two-layer nesting raises an error."""
    nesting_dict = {
        "nest1": ["alt"],
        "nest2": {"option": ["option"], "another": ["other", "another"]},
    }
    product_indices = {"option": 0, "another": 1, "alt": 2, "other": 3}

    with pytest.raises(ValueError, match=r"must map to a list of alternatives"):
        nstL._parse_nesting(nesting_dict, product_indices)


def test_parse_nesting_empty_dict_raises(nstL):
    """Test that empty nesting structure raises an error."""
    nesting_dict = {}
    product_indices = {"alt": 0, "option": 1, "another": 2}

    with pytest.raises(ValueError, match=r"Nesting structure must not be empty."):
        nstL._parse_nesting(nesting_dict, product_indices)


def test_preprocess_model_data_sets_attributes(nstL, sample_df, utility_eqs):
    _, _, _ = nstL.preprocess_model_data(sample_df, utility_eqs)

    # Check main attributes exist
    assert hasattr(nstL, "X_data")
    assert hasattr(nstL, "F")
    assert hasattr(nstL, "y")
    assert hasattr(nstL, "alternatives")
    assert hasattr(nstL, "prod_indices")
    assert hasattr(nstL, "nest_indices")
    assert hasattr(nstL, "all_nests")
    assert hasattr(nstL, "lambda_lkup")
    assert hasattr(nstL, "coords")

    # Check X, F, y shapes match the data
    assert nstL.X_data.shape[0] == sample_df.shape[0]
    assert nstL.y.shape[0] == sample_df.shape[0]

    # Check that nest_indices has 'top' (but not 'mid' anymore)
    assert "top" in nstL.nest_indices
    assert "mid" not in nstL.nest_indices

    # Check that all_nests contains the correct names
    expected_nests = ["nest1", "nest2"]
    assert set(nstL.all_nests) == set(expected_nests)

    # Check that lambda_lkup maps all nests to integers
    for nest in expected_nests:
        assert nest in nstL.lambda_lkup
        assert isinstance(nstL.lambda_lkup[nest], int)

    # Check that nest-specific coordinate dimensions exist
    assert "nest1_alts" in nstL.coords
    assert "nest2_alts" in nstL.coords


def test_build_model_returns_pymc_model(nstL, sample_df, utility_eqs):
    X, F, y = nstL.preprocess_model_data(sample_df, utility_eqs)
    model = nstL.make_model(X, F, y)
    assert isinstance(nstL.coords, dict)
    assert isinstance(model, pm.Model)
    assert nstL.alternatives == ["alt", "other", "option", "another"]
    assert nstL.covariates == ["X1", "X2"]


def test_build_model_2layer_raises_error(nstL2_invalid, sample_df, utility_eqs):
    """Test that attempting two-layer nesting raises an informative error."""
    with pytest.raises(ValueError, match=r"must map to a list of alternatives"):
        _, _, _ = nstL2_invalid.preprocess_model_data(sample_df, utility_eqs)


def test_sample(nstL, sample_df, utility_eqs, mock_pymc_sample):
    X, F, y = nstL.preprocess_model_data(sample_df, utility_eqs)
    _ = nstL.make_model(X, F, y)
    nstL.sample_prior_predictive(
        samples=5,
        extend_idata=True,
    )
    assert "prior_predictive" in nstL.idata
    nstL.sample()
    assert hasattr(nstL, "idata")

    nstL.sample_posterior_predictive(
        extend_idata=False,
    )
    assert "posterior_predictive" in nstL.idata
    assert "fit_data" in nstL.idata

    nstL.sample_posterior_predictive(choice_df=sample_df, extend_idata=True)
    assert isinstance(nstL.idata, az.InferenceData)

    nstL.fit(choice_df=sample_df, utility_equations=utility_eqs)

    with pytest.raises(
        RuntimeError, match=r"self.idata must be initialized before extending"
    ):
        nstL.idata = None
        nstL.sample_posterior_predictive(
            extend_idata=True,
        )


def test_counterfactual(
    nstL, sample_df, utility_eqs, new_utility_eqs, mock_pymc_sample
):
    X, F, y = nstL.preprocess_model_data(sample_df, utility_eqs)
    _ = nstL.make_model(X, F, y)
    nstL.sample()
    new = sample_df.copy()
    nstL.apply_intervention(new)
    change_df = nstL.calculate_share_change(nstL.idata, nstL.intervention_idata)
    assert isinstance(change_df, pd.DataFrame)
    assert hasattr(nstL, "intervention_idata")
    new = new[new["choice"] != "another"]
    nstL.nesting_structure = {"nest1": ["alt", "other"], "nest2": ["option"]}
    idata_new_policy = nstL.apply_intervention(new, new_utility_eqs)
    assert "posterior_predictive" in idata_new_policy


def test_counterfactual_removal(
    nstL, sample_df, utility_eqs_non_fixed, new_utility_eqs, mock_pymc_sample
):
    X, F, y = nstL.preprocess_model_data(sample_df, utility_eqs_non_fixed)
    _ = nstL.make_model(X, F, y)
    nstL.sample()
    new = sample_df[sample_df["choice"] != "another"]
    nstL.nesting_structure = {"nest1": ["alt", "other"], "nest2": ["option"]}
    idata_new_policy = nstL.apply_intervention(new, new_utility_eqs)
    assert "posterior_predictive" in idata_new_policy


def test_make_change_plot_returns_figure(nstL, sample_change_df):
    fig = nstL.plot_change(sample_change_df, title="Test Intervention")

    assert isinstance(fig, plt.Figure)
