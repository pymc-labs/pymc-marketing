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

from pymc_marketing.customer_choice._choice_helpers import halton_draws


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


def generate_blp_panel(
    *,
    T: int = 50,
    J: int = 4,
    K: int = 2,
    L: int = 2,
    R_geo: int = 1,
    true_alpha: float = -2.0,
    true_beta: np.ndarray | None = None,
    sigma_alpha: float = 0.5,
    sigma_beta: np.ndarray | None = None,
    instrument_strength: float = 0.7,
    price_xi_corr: float = 0.6,
    xi_sigma: float = 0.3,
    xi_product_sigma: float = 0.5,
    sigma_eta: float = 1.0,
    market_size: int = 5000,
    n_dgp_draws: int = 5000,
    region_heterogeneity: float = 0.0,
    return_truth: bool = False,
    random_seed: np.random.Generator | int | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Generate a synthetic BLP-style aggregate-share panel.

    Produces a long-format DataFrame suitable for fitting
    :class:`pymc_marketing.customer_choice.BayesianBLP` (and for unit tests
    of the same), together with — when ``return_truth=True`` — the latent
    parameters that generated it. The data-generating process explicitly
    induces correlation between the price residual ``η_jt`` and the
    structural error ``ξ_jt`` (controlled by ``price_xi_corr``), so that no-IV
    fits exhibit the expected endogeneity bias on the price coefficient and
    IV fits can be shown to recover it.

    Parameters
    ----------
    T
        Number of periods per region.
    J
        Number of inside products. An outside good (row label ``"outside"``)
        is added on top.
    K
        Number of product characteristics ``x_jt``.
    L
        Number of instruments ``z_jt``.
    R_geo
        Number of regions. Defaults to 1; set ``>1`` together with
        ``region_heterogeneity > 0`` to test hierarchical pooling.
    true_alpha
        Population-level price coefficient (should be negative).
    true_beta
        Population-level characteristic coefficients, shape ``(K,)``. Defaults
        to a vector of ones.
    sigma_alpha
        Across-consumer SD of the price coefficient.
    sigma_beta
        Across-consumer SD of each characteristic coefficient, shape ``(K,)``.
        Zero entries indicate no heterogeneity on that characteristic.
        Defaults to all zeros (heterogeneity only on price).
    instrument_strength
        Magnitude of the first-stage instrument loading
        ``π_z = instrument_strength / sqrt(L)``. Set small (e.g. 0.1) to
        simulate weak instruments.
    price_xi_corr
        Correlation ``Cor(η_jt, ξ̃_jt)`` of the joint price-residual /
        structural-error draws. Drives the endogeneity bias.
    xi_sigma
        SD of the time-varying part ``ξ̃_jt``.
    xi_product_sigma
        SD of the product fixed effect ``ξ_j``.
    sigma_eta
        SD of the price first-stage residual ``η_jt``.
    market_size
        Total category volume per market. Used both to scale the
        Multinomial draws of observed shares and as the ``n`` column on the
        returned panel.
    n_dgp_draws
        Number of QMC draws used to compute the *true* mixed-logit shares.
        Should be much larger than the number of draws the model itself uses
        so that the DGP is essentially exact.
    region_heterogeneity
        Across-region SD applied to ``α_r`` and ``β_r``. Zero (default)
        produces homogeneous regions; positive values produce heterogeneous
        ones.
    return_truth
        If ``True``, return ``(df, truth_dict)``; otherwise return ``df``.
    random_seed
        Seed or ``np.random.Generator``.

    Returns
    -------
    df : pd.DataFrame
        Long-format panel with columns
        ``["region", "market", "period", "product", "share", "n", "price",
        "x_0", ..., "x_{K-1}", "z_0", ..., "z_{L-1}"]``. The outside good
        appears once per market with ``price`` and all characteristics /
        instruments set to zero. ``market`` is a global integer;
        ``period`` is the integer time index within a region (0..T-1);
        ``region`` is a string label. Pass ``time_col="period"`` to
        :class:`pymc_marketing.customer_choice.BayesianBLP` to make the
        time dimension first-class for counterfactuals.
    truth : dict, optional
        Returned only when ``return_truth=True``. Contains the population
        and per-cell parameters that generated the panel — useful for
        recovery tests. See source for the exact keys.

    Examples
    --------
    Generate a small panel with strong instruments and a bias-inducing
    price/structural-error correlation:

    .. code-block:: python

        from pymc_marketing.customer_choice import generate_blp_panel

        df, truth = generate_blp_panel(
            T=30,
            J=3,
            K=2,
            L=2,
            true_alpha=-2.0,
            price_xi_corr=0.6,
            random_seed=42,
            return_truth=True,
        )

    """
    rng: np.random.Generator = (
        random_seed
        if isinstance(random_seed, np.random.Generator)
        else np.random.default_rng(random_seed)
    )

    if true_beta is None:
        true_beta = np.full(K, 1.0)
    else:
        true_beta = np.asarray(true_beta, dtype=float)
        if true_beta.shape != (K,):
            raise ValueError(f"true_beta must have shape ({K},), got {true_beta.shape}")

    if sigma_beta is None:
        sigma_beta = np.zeros(K)
    else:
        sigma_beta = np.asarray(sigma_beta, dtype=float)
        if sigma_beta.shape != (K,):
            raise ValueError(
                f"sigma_beta must have shape ({K},), got {sigma_beta.shape}"
            )
        if np.any(sigma_beta < 0):
            raise ValueError("sigma_beta entries must be non-negative")

    if R_geo > 1 and region_heterogeneity > 0:
        alpha_r = true_alpha + region_heterogeneity * rng.standard_normal(R_geo)
        beta_r = true_beta + region_heterogeneity * rng.standard_normal((R_geo, K))
    else:
        alpha_r = np.full(R_geo, true_alpha)
        beta_r = np.tile(true_beta, (R_geo, 1))

    xi_j = rng.normal(0.0, xi_product_sigma, size=J)

    Sigma_corr = np.array([[1.0, price_xi_corr], [price_xi_corr, 1.0]])
    L_chol = np.linalg.cholesky(Sigma_corr)
    raw = rng.standard_normal((R_geo, T, J, 2))
    eta_xi = raw @ L_chol.T
    eta = eta_xi[..., 0] * sigma_eta
    xi_tilde = eta_xi[..., 1] * xi_sigma
    xi = xi_j[None, None, :] + xi_tilde

    x = np.empty((R_geo, T, J, K))
    if K >= 1:
        x[..., 0] = rng.binomial(1, 0.3, size=(R_geo, T, J)).astype(float)
    if K >= 2:
        x[..., 1] = rng.normal(0.0, 1.0, size=(R_geo, T, J))
    for k in range(2, K):
        x[..., k] = rng.normal(0.0, 1.0, size=(R_geo, T, J))

    z = rng.normal(0.0, 1.0, size=(R_geo, T, J, L))

    pi_0 = rng.normal(2.0, 0.3, size=J)
    pi_z = (instrument_strength / np.sqrt(L)) * np.ones(L)
    price = pi_0[None, None, :] + (z * pi_z).sum(axis=-1) + eta
    price = np.maximum(price, 0.1)

    n_random = 1 + int(np.sum(sigma_beta > 0))
    halton_seed = int(rng.integers(0, 2**31 - 1))
    nu = halton_draws(n_dgp_draws, n_random, seed=halton_seed)
    nu_alpha = nu[:, 0]
    nu_beta_idx = np.where(sigma_beta > 0)[0]
    nu_beta_cols = {k: nu[:, 1 + i] for i, k in enumerate(nu_beta_idx)}

    delta = alpha_r[:, None, None] * price + np.einsum("rk,rtjk->rtj", beta_r, x) + xi
    mu_dev = sigma_alpha * nu_alpha[None, None, None, :] * price[..., None]
    for k in nu_beta_idx:
        mu_dev = mu_dev + (
            sigma_beta[k] * nu_beta_cols[k][None, None, None, :] * x[..., k, None]
        )

    U = delta[..., None] + mu_dev
    U_max = np.maximum(U.max(axis=2, keepdims=True), 0.0)
    eU = np.exp(U - U_max)
    e0 = np.exp(-U_max[:, :, 0, :])
    denom = e0 + eU.sum(axis=2)
    s_inside = (eU / denom[:, :, None, :]).mean(axis=-1)
    s_outside = (e0 / denom).mean(axis=-1)
    true_shares = np.concatenate([s_outside[..., None], s_inside], axis=-1)

    obs_counts = np.empty((R_geo, T, J + 1), dtype=int)
    for r in range(R_geo):
        for t in range(T):
            obs_counts[r, t] = rng.multinomial(market_size, true_shares[r, t])
    obs_shares = obs_counts / market_size

    rows = []
    char_cols = [f"x_{k}" for k in range(K)]
    inst_cols = [f"z_{ell}" for ell in range(L)]
    market_idx = 0
    for r in range(R_geo):
        for t in range(T):
            for j_idx in range(J + 1):
                is_outside = j_idx == 0
                row = {
                    "region": f"r{r}",
                    "market": market_idx,
                    "period": int(t),
                    "product": "outside" if is_outside else f"prod_{j_idx - 1}",
                    "share": float(obs_shares[r, t, j_idx]),
                    "n": int(market_size),
                    "price": 0.0 if is_outside else float(price[r, t, j_idx - 1]),
                }
                for k, col in enumerate(char_cols):
                    row[col] = 0.0 if is_outside else float(x[r, t, j_idx - 1, k])
                for ell, col in enumerate(inst_cols):
                    row[col] = 0.0 if is_outside else float(z[r, t, j_idx - 1, ell])
                rows.append(row)
            market_idx += 1
    df = pd.DataFrame(rows)

    if not return_truth:
        return df

    truth = {
        "alpha": true_alpha,
        "alpha_r": alpha_r,
        "beta": true_beta,
        "beta_r": beta_r,
        "sigma_alpha": sigma_alpha,
        "sigma_beta": sigma_beta,
        "xi_j": xi_j,
        "xi_tilde": xi_tilde,
        "xi": xi,
        "pi_0": pi_0,
        "pi_z": pi_z,
        "price_xi_corr": price_xi_corr,
        "sigma_eta": sigma_eta,
        "true_shares": true_shares,
        "price_array": price,
        "characteristics_array": x,
        "instruments_array": z,
        "characteristic_cols": char_cols,
        "instrument_cols": inst_cols,
    }
    return df, truth


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
