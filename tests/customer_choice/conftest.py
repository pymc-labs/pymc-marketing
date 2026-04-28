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

import pytest

from pymc_marketing.customer_choice import generate_blp_panel


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
