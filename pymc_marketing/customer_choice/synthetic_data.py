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
import patsy


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


def generate_maxdiff_conjoint_data(
    n_respondents: int = 150,
    n_items: int = 12,
    item_attributes: pd.DataFrame | None = None,
    utility_formula: str = "~ 0 + C(brand) + price + quality",
    true_betas: dict[str, float] | None = None,
    n_tasks_per_resp: int = 12,
    subset_size: int = 4,
    random_attributes: list[str] | None = None,
    sigma_respondent: float = 0.4,
    items: list[str] | None = None,
    random_seed: np.random.Generator | int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    r"""Generate synthetic MaxDiff data with item-attribute utilities (part-worths).

    Simulates a MaxDiff survey where each item has a fixed attribute profile
    and utilities are computed as :math:`U_i = X_i^\top \beta + \text{noise}`.
    Respondents optionally carry heterogeneous part-worths on a subset of
    features (analogous to the random-coefficients formulation in
    :class:`~pymc_marketing.customer_choice.MaxDiffMixedLogit`).

    Parameters
    ----------
    n_respondents : int, default 150
        Number of respondents.
    n_items : int, default 12
        Full item pool size. Ignored if ``item_attributes`` is provided.
    item_attributes : pd.DataFrame, optional
        One row per item, index = item name, columns = attributes. If None,
        attributes are auto-generated: a 3-level ``brand`` categorical and two
        continuous features ``price ~ Uniform(0, 1)`` and
        ``quality ~ Normal(0, 1)``.
    utility_formula : str, default ``"~ 0 + C(brand) + price + quality"``
        Patsy formula used to expand ``item_attributes`` into a design matrix.
    true_betas : dict[str, float], optional
        Ground-truth part-worths keyed by patsy-expanded feature name. Missing
        keys are drawn from ``Normal(0, 1)``.
    n_tasks_per_resp : int, default 12
        Tasks per respondent.
    subset_size : int, default 4
        Items shown per task.
    random_attributes : list[str], optional
        Feature names whose part-worths vary across respondents. Defaults to
        all features.
    sigma_respondent : float, default 0.4
        Scale of per-respondent deviations on the random-feature subset.
    items : list[str], optional
        Item names. Defaults to ``["item_0", ...]`` when ``item_attributes``
        is None; otherwise taken from ``item_attributes.index``.
    random_seed : np.random.Generator or int, optional
        Random state.

    Returns
    -------
    task_df : pd.DataFrame
        Long-format data with columns ``respondent_id``, ``task_id``,
        ``item_id``, ``is_best``, ``is_worst``.
    item_attributes : pd.DataFrame
        The attribute table, indexed by item name. Aligned with ``items``.
    ground_truth : dict
        ``{"betas", "respondent_betas", "feature_names",
        "random_attributes", "sigma_respondent", "items", "X"}``.
        ``betas`` is the population part-worth vector;
        ``respondent_betas`` holds the per-respondent part-worth matrix
        actually used to simulate picks.

    Notes
    -----
    Real conjoint studies use balanced designs; this generator draws task
    subsets uniformly for simplicity, which is adequate for recovery tests.
    """
    rng: np.random.Generator = (
        random_seed
        if isinstance(random_seed, np.random.Generator)
        else np.random.default_rng(random_seed)
    )

    if subset_size < 2:
        raise ValueError("subset_size must be >= 2.")

    # Attribute table -------------------------------------------------------
    if item_attributes is None:
        if items is None:
            items = [f"item_{i}" for i in range(n_items)]
        if subset_size > len(items):
            raise ValueError(
                f"subset_size ({subset_size}) cannot exceed n_items ({len(items)})."
            )
        item_attributes = pd.DataFrame(
            {
                "brand": rng.choice(["A", "B", "C"], size=len(items)),
                "price": rng.uniform(0, 1, size=len(items)),
                "quality": rng.standard_normal(len(items)),
            },
            index=pd.Index(items, name="item_id"),
        )
    else:
        if items is None:
            items = list(item_attributes.index)
        item_attributes = item_attributes.loc[items].copy()
        if subset_size > len(items):
            raise ValueError(
                f"subset_size ({subset_size}) cannot exceed n_items ({len(items)})."
            )

    # Design matrix ---------------------------------------------------------
    design = patsy.dmatrix(utility_formula, item_attributes, return_type="matrix")
    X = np.asarray(design, dtype=float)
    feature_names = list(design.design_info.column_names)
    n_features = X.shape[1]

    # Part-worths ----------------------------------------------------------
    betas: np.ndarray = np.asarray(rng.normal(0, 1, size=n_features))
    if true_betas is not None:
        for k, v in true_betas.items():
            if k not in feature_names:
                raise ValueError(
                    f"true_betas key {k!r} not in expanded features {feature_names}."
                )
            betas[feature_names.index(k)] = float(v)

    rc_names = (
        list(random_attributes)
        if random_attributes is not None
        else list(feature_names)
    )
    unknown = [rc for rc in rc_names if rc not in feature_names]
    if unknown:
        raise ValueError(
            f"random_attributes {unknown} not in expanded features {feature_names}."
        )
    rc_idx = np.array([feature_names.index(rc) for rc in rc_names], dtype=np.int64)

    respondent_betas = np.broadcast_to(
        betas[None, :], (n_respondents, n_features)
    ).copy()
    if len(rc_idx) > 0:
        respondent_betas[:, rc_idx] += sigma_respondent * rng.standard_normal(
            size=(n_respondents, len(rc_idx))
        )

    # Per-respondent per-item utility: (R, I)
    U_ri = respondent_betas @ X.T

    # Simulate picks --------------------------------------------------------
    records = []
    n_items_total = len(items)
    for r in range(n_respondents):
        for task in range(n_tasks_per_resp):
            subset = rng.choice(n_items_total, size=subset_size, replace=False)
            u = U_ri[r, subset]

            p_best = _softmax_1d(u)
            best_local = int(rng.choice(subset_size, p=p_best))
            remaining = np.ones(subset_size, dtype=bool)
            remaining[best_local] = False
            u_worst = -u.copy()
            u_worst[~remaining] = -np.inf
            p_worst = _softmax_1d(u_worst)
            worst_local = int(rng.choice(subset_size, p=p_worst))

            for local_pos, item_pos in enumerate(subset):
                records.append(
                    {
                        "respondent_id": f"r{r}",
                        "task_id": task,
                        "item_id": items[item_pos],
                        "is_best": int(local_pos == best_local),
                        "is_worst": int(local_pos == worst_local),
                    }
                )

    task_df = pd.DataFrame(records)

    ground_truth = {
        "betas": betas,
        "respondent_betas": respondent_betas,
        "feature_names": feature_names,
        "random_attributes": rc_names,
        "sigma_respondent": sigma_respondent,
        "items": items,
        "X": X,
    }
    return task_df, item_attributes, ground_truth


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
