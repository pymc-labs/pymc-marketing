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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

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
def nstL(utility_eqs, nesting_structure_1):
    return NestedLogit(
        utility_equations=utility_eqs,
        depvar="choice",
        covariates=["X1", "X2"],
        nesting_structure=nesting_structure_1,
    )


@pytest.fixture
def nstL2_invalid(utility_eqs, nesting_structure_invalid_two_layer):
    """NestedLogit instance that will fail with two-layer nesting."""
    return NestedLogit(
        utility_equations=utility_eqs,
        depvar="choice",
        covariates=["X1", "X2"],
        nesting_structure=nesting_structure_invalid_two_layer,
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


def test_build_model_returns_pymc_model(nstL, sample_df):
    nstL._set_choice_inputs(sample_df)
    nstL.build_model()
    X, F, y = nstL._training_arrays()
    model = nstL.make_model(X, F, y)
    assert isinstance(nstL.coords, dict)
    assert isinstance(model, pm.Model)
    assert nstL.alternatives == ["alt", "other", "option", "another"]
    assert nstL.covariates == ["X1", "X2"]


def test_make_model_nest_indices_override(nstL, sample_df, utility_eqs):
    bundle = nstL._preprocess(sample_df, utility_eqs)
    coords = {**bundle.coords, "single_alts": bundle.alternatives}

    model = nstL.make_model(
        bundle.X,
        bundle.F,
        bundle.y,
        coords=coords,
        nest_indices={"single": np.arange(len(bundle.alternatives))},
    )

    assert "P_single" in model.named_vars
    assert "P_nest1" not in model.named_vars
    assert "P_nest2" not in model.named_vars


def test_build_model_2layer_raises_error(nstL2_invalid, sample_df, utility_eqs):
    """Test that attempting two-layer nesting raises an informative error."""
    with pytest.raises(ValueError, match=r"must map to a list of alternatives"):
        nstL2_invalid._preprocess(sample_df, utility_eqs)


def test_sample(nstL, sample_df, utility_eqs, mock_pymc_sample):
    nstL.fit(sample_df)
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
    assert isinstance(nstL.idata, xr.DataTree)

    nstL.fit(choice_df=sample_df, utility_equations=utility_eqs)

    with pytest.raises(
        RuntimeError, match=r"self.idata must be initialized before extending"
    ):
        nstL.idata = None
        nstL.sample_posterior_predictive(
            extend_idata=True,
        )


def test_counterfactual(nstL, sample_df, new_utility_eqs, mock_pymc_sample):
    nstL.fit(sample_df)
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
    nstL.fit(sample_df, utility_equations=utility_eqs_non_fixed)
    nstL.sample()
    new = sample_df[sample_df["choice"] != "another"]
    nstL.nesting_structure = {"nest1": ["alt", "other"], "nest2": ["option"]}
    idata_new_policy = nstL.apply_intervention(new, new_utility_eqs)
    assert "posterior_predictive" in idata_new_policy


def test_make_change_plot_returns_figure(nstL, sample_change_df):
    fig = nstL.plot_change(sample_change_df, title="Test Intervention")

    assert isinstance(fig, plt.Figure)


@pytest.mark.parametrize("lam", [1.0, 0.5])
def test_nest_probs_match_closed_form(nstL, sample_df, lam):
    nstL._set_choice_inputs(sample_df)
    nstL.build_model()
    model = nstL.model
    fixed = pm.do(
        model,
        {
            "lambdas_nests": np.full(2, lam),
            "alphas_": np.zeros(model["alphas_"].shape.eval()),
            "betas": np.array([0.3, -0.2]),
            "betas_fixed_": np.zeros(model["betas_fixed_"].shape.eval()),
        },
    )
    p, U = pm.draw([fixed["p"], fixed["U"]], random_seed=0)

    # Closed form nested logit with nests [alt, other] and [option, another]
    nests = [[0, 1], [2, 3]]
    inclusive = np.stack(
        [lam * np.log(np.exp(U[:, idx] / lam).sum(axis=1)) for idx in nests], axis=1
    )
    p_nest = np.exp(inclusive) / np.exp(inclusive).sum(axis=1, keepdims=True)
    expected = np.empty_like(U)
    for k, idx in enumerate(nests):
        within = np.exp(U[:, idx] / lam)
        expected[:, idx] = p_nest[:, [k]] * within / within.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(p, expected, rtol=1e-6)
