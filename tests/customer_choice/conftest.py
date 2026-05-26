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
"""Shared fixtures for customer-choice tests."""

import warnings

import pytest

from pymc_marketing.customer_choice import BayesianBLP, generate_blp_panel


@pytest.fixture(scope="session")
def blp_panel_small():
    """Small single-region BLP panel for fast unit tests."""
    df, truth = generate_blp_panel(
        T=20,
        J=3,
        K=2,
        L=2,
        R_geo=1,
        true_alpha=-2.0,
        sigma_alpha=0.5,
        price_xi_corr=0.6,
        market_size=2_000,
        n_dgp_draws=2_000,
        random_seed=42,
        return_truth=True,
    )
    return df, truth


@pytest.fixture(scope="session")
def blp_panel_multi_region():
    """Three-region BLP panel for hierarchical-pooling tests."""
    df, truth = generate_blp_panel(
        T=15,
        J=3,
        K=2,
        L=2,
        R_geo=3,
        region_heterogeneity=0.5,
        true_alpha=-2.0,
        sigma_alpha=0.5,
        price_xi_corr=0.5,
        market_size=2_000,
        n_dgp_draws=2_000,
        random_seed=7,
        return_truth=True,
    )
    return df, truth


@pytest.fixture(scope="session")
def fitted_blp(blp_panel_small):
    """A small BayesianBLP fitted on ``blp_panel_small`` (1-D price RC).

    Session-scoped — the ~30s fit runs once for the entire test session.
    Promoted from ``test_bayesian_blp.py`` to ``conftest.py`` so other
    test modules (e.g. ``test_taste_profiles.py``) can reuse it.
    """
    df, truth = blp_panel_small
    model = BayesianBLP(
        market_data=df,
        characteristics=truth["characteristic_cols"],
        instruments=truth["instrument_cols"],
        random_coef_on=["price"],
        n_mc_draws=80,
        random_seed=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            nuts_sampler="numpyro",
            draws=200,
            tune=200,
            chains=2,
            progressbar=False,
            random_seed=0,
        )
    return model, truth


@pytest.fixture(scope="session")
def fitted_blp_multidim(blp_panel_small):
    """BayesianBLP fitted with random_coef_on=['price', 'x_0', 'x_1'].

    Used by tests that need to exercise multi-dimensional taste-profile paths
    (the dim>0 slice of brand_buyer_nu, the full (S, M, D) shape of
    buyer_nu_posterior). Session-scoped — the ~30s fit runs once.
    """
    df, truth = blp_panel_small
    model = BayesianBLP(
        market_data=df,
        characteristics=truth["characteristic_cols"],
        instruments=truth["instrument_cols"],
        random_coef_on=["price", *truth["characteristic_cols"]],
        n_mc_draws=120,
        random_seed=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            nuts_sampler="numpyro",
            draws=200,
            tune=200,
            chains=2,
            target_accept=0.95,
            progressbar=False,
            random_seed=0,
        )
    return model, truth
