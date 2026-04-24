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


    Examples
    --------
    Generate some synthetic data for the MVITS model:

    .. code-block:: python

        import numpy as np

        from pymc_marketing.customer_choice import generate_saturated_data

        seed = sum(map(ord, "Saturated Market Data"))
        rng = np.random.default_rng(seed)

        scenario = {
            "total_sales_mu": 1_000,
            "total_sales_sigma": 5,
            "treatment_time": 40,
            "n_observations": 100,
            "market_shares_before": [[0.7, 0.3, 0]],
            "market_shares_after": [[0.65, 0.25, 0.1]],
            "market_share_labels": ["competitor", "own", "new"],
            "random_seed": rng,
        }

        data = generate_saturated_data(**scenario)

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


def generate_maxdiff_data(
    n_respondents: int = 200,
    n_items: int = 20,
    n_tasks_per_resp: int = 12,
    subset_size: int = 4,
    true_utilities: np.ndarray | None = None,
    sigma_respondent: float = 0.6,
    items: list[str] | None = None,
    random_seed: np.random.Generator | int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generate synthetic MaxDiff (best-worst scaling) data.

    Simulates a MaxDiff survey where each respondent sees ``n_tasks_per_resp``
    tasks, each showing a random ``subset_size`` of items drawn uniformly from
    the full pool of ``n_items``. The respondent picks the best and worst items
    from the subset according to the Louviere sequential best-worst model.

    Parameters
    ----------
    n_respondents : int, default 200
        Number of respondents.
    n_items : int, default 20
        Full item pool size.
    n_tasks_per_resp : int, default 12
        Tasks shown per respondent.
    subset_size : int, default 4
        Items shown per task (must be ``<= n_items``).
    true_utilities : np.ndarray, optional
        Ground-truth item utilities of length ``n_items``. If None, drawn
        from ``Normal(0, 1)``. The last item's utility is shifted to 0 to
        match the default identification constraint.
    sigma_respondent : float, default 0.6
        Scale of per-respondent item-level deviations. Set to 0 for a
        homogeneous-preferences population.
    items : list[str], optional
        Item names (length ``n_items``). Defaults to ``["item_0", ...]``.
    random_seed : np.random.Generator or int, optional
        Random state for reproducibility.

    Returns
    -------
    task_df : pd.DataFrame
        Long-format data with columns ``respondent_id``, ``task_id``,
        ``item_id``, ``is_best``, ``is_worst``. One row per shown item per task.
    ground_truth : dict
        ``{"utilities": (n_items,), "respondent_utilities": (R, I),
        "sigma_respondent": float, "items": list[str]}``. ``utilities`` is
        the population-level ground truth (with last item shifted to 0);
        ``respondent_utilities`` holds the per-respondent values used to
        simulate each respondent's picks.

    Notes
    -----
    Subsets are drawn uniformly without replacement. Real MaxDiff studies
    use balanced designs (BIBD) for efficiency; this generator trades that
    for simplicity and is adequate for parameter-recovery testing.
    """
    rng: np.random.Generator = (
        random_seed
        if isinstance(random_seed, np.random.Generator)
        else np.random.default_rng(random_seed)
    )

    if subset_size > n_items:
        raise ValueError(
            f"subset_size ({subset_size}) cannot exceed n_items ({n_items})."
        )
    if subset_size < 2:
        raise ValueError(
            "subset_size must be at least 2 (need distinct best and worst)."
        )

    if items is None:
        items = [f"item_{i}" for i in range(n_items)]
    if len(items) != n_items:
        raise ValueError(f"items has length {len(items)} but n_items is {n_items}.")

    if true_utilities is None:
        true_utilities = rng.normal(0, 1, size=n_items)
    true_utilities = np.asarray(true_utilities, dtype=float)
    # Identification: shift so the reference (last) item is at 0.
    true_utilities = true_utilities - true_utilities[-1]

    respondent_utilities = true_utilities[None, :] + sigma_respondent * rng.normal(
        size=(n_respondents, n_items)
    )

    records = []
    for r in range(n_respondents):
        for task in range(n_tasks_per_resp):
            subset = rng.choice(n_items, size=subset_size, replace=False)
            u = respondent_utilities[r, subset]

            # Sequential best then worst from remaining.
            p_best = _softmax_1d(u)
            best_local = int(rng.choice(subset_size, p=p_best))
            remaining = np.ones(subset_size, dtype=bool)
            remaining[best_local] = False
            u_worst = -u
            u_worst[~remaining] = -np.inf
            p_worst = _softmax_1d(u_worst)
            worst_local = int(rng.choice(subset_size, p=p_worst))

            for local_pos, item_idx in enumerate(subset):
                records.append(
                    {
                        "respondent_id": f"r{r}",
                        "task_id": task,
                        "item_id": items[item_idx],
                        "is_best": int(local_pos == best_local),
                        "is_worst": int(local_pos == worst_local),
                    }
                )

    task_df = pd.DataFrame(records)
    ground_truth = {
        "utilities": true_utilities,
        "respondent_utilities": respondent_utilities,
        "sigma_respondent": sigma_respondent,
        "items": items,
    }
    return task_df, ground_truth


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    """Numerically stable 1-D softmax used by the synthetic MaxDiff generator."""
    x_shift = x - np.max(x)
    exp = np.exp(x_shift)
    return exp / exp.sum()


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
