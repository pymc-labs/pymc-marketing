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
"""Data generation functions for consumer choice models."""

import numpy as np
import pandas as pd


def generate_saturated_data(
    total_sales_mu: int,
    total_sales_sigma: float,
    treatment_time: int,
    n_observations: int,
    market_shares_before,
    market_shares_after,
    market_share_labels,
    random_seed: int | np.random.Generator | None = None,
):
    """Generate synthetic data for the MVITS model, assuming market is saturated.

    This function generates synthetic data for the MVITS model, assuming that the market is
    saturated. This makes the assumption that the total sales are normally distributed around
    some average level of sales, and that the market shares are constant over time.

    Parameters
    ----------
    total_sales_mu: int
        The average level of sales in the market.
    total_sales_sigma: float
        The standard deviation of sales in the market.
    treatment_time: int
        The time at which the new model is introduced.
    n_observations: int
        The number of observations to generate.
    market_shares_before: list[float]
        The market shares before the introduction of the new model.
    market_shares_after: list[float]
        The market shares after the introduction of the new model.
    market_share_labels: list[str]
        The labels for the market shares.
    random_seed: np.random.Generator | int, optional
        The random number generator to use.

    Returns
    -------
    data: pd.DataFrame
        The synthetic data generated.

    """
    rng: np.random.Generator = (
        random_seed
        if isinstance(random_seed, np.random.Generator)
        else np.random.default_rng(random_seed)
    )

    rates = np.array(
        treatment_time * market_shares_before
        + (n_observations - treatment_time) * market_shares_after
    )

    # Generate total demand (sales) as normally distributed around some average level of sales
    total = (
        rng.normal(loc=total_sales_mu, scale=total_sales_sigma, size=n_observations)
    ).astype(int)

    # Ensure total sales are never negative
    total[total < 0] = 0

    # Generate sales counts
    counts = rng.multinomial(total, rates)

    # Convert to DataFrame
    data = pd.DataFrame(counts)
    data.columns = market_share_labels
    data.columns.name = "product"
    data.index.name = "day"
    data["pre"] = data.index < treatment_time
    return data


def generate_unsaturated_data(
    total_sales_before: list[int],
    total_sales_after: list[int],
    total_sales_sigma: float,
    treatment_time: int,
    n_observations: int,
    market_shares_before: list[list[float]],
    market_shares_after: list[list[float]],
    market_share_labels: list[str],
    random_seed: np.random.Generator | int | None = None,
):
    """Generate synthetic data for the MVITS model.

    Notably, we can define different total sales levels before and after the
    introduction of the new model.

    This function generates synthetic data for the MVITS model, assuming that the market is
    unsaturated meaning that there are new sales to be made.

    This makes the assumption that the total sales are normally distributed around
    some average level of sales, and that the market shares are constant over time.

    Parameters
    ----------
    total_sales_mu: int
        The average level of sales in the market.
    total_sales_sigma: float
        The standard deviation of sales in the market.
    treatment_time: int
        The time at which the new model is introduced.
    n_observations: int
        The number of observations to generate.
    market_shares_before: list[float]
        The market shares before the introduction of the new model.
    market_shares_after: list[float]
        The market shares after the introduction of the new model.
    market_share_labels: list[str]
        The labels for the market shares.
    random_seed: np.random.Generator | int, optional
        The random number generator to use.

    Returns
    -------
    data: pd.DataFrame
        The synthetic data generated.

    """
    rng: np.random.Generator = (
        random_seed
        if isinstance(random_seed, np.random.Generator)
        else np.random.default_rng(random_seed)
    )

    rates = np.array(
        treatment_time * market_shares_before
        + (n_observations - treatment_time) * market_shares_after
    )

    total_sales_mu = np.array(
        treatment_time * total_sales_before
        + (n_observations - treatment_time) * total_sales_after
    )

    total = (
        rng.normal(loc=total_sales_mu, scale=total_sales_sigma, size=n_observations)
    ).astype(int)

    # Ensure total sales are never negative
    total[total < 0] = 0

    # Generate sales counts
    counts = rng.multinomial(total, rates)

    # Convert to DataFrame
    data = pd.DataFrame(counts)
    data.columns = pd.Index(market_share_labels)
    data.columns.name = "product"
    data.index.name = "day"
    data["pre"] = data.index < treatment_time
    return data
