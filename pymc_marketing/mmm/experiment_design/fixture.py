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

"""Fixture generation for ExperimentDesigner demos and tests.

Generates realistic posteriors by fitting an MMM to simulated data,
producing posterior structure (parameter correlations, degeneracies)
that hand-crafted distributions cannot replicate.
"""

from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import xarray as xr
from arviz import InferenceData


def _simulate_spend(
    n_weeks: int,
    n_channels: int,
    channel_names: list[str],
    correlation_matrix: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate correlated log-normal spend series.

    Parameters
    ----------
    n_weeks : int
        Number of weeks.
    n_channels : int
        Number of channels.
    channel_names : list[str]
        Channel names (for labelling).
    correlation_matrix : np.ndarray | None
        Desired correlation matrix. If None, uses moderate correlation.
    rng : np.random.Generator | None
        Random number generator.

    Returns
    -------
    np.ndarray
        Spend array of shape ``(n_weeks, n_channels)``.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if correlation_matrix is None:
        base = 0.4 * np.ones((n_channels, n_channels))
        np.fill_diagonal(base, 1.0)
        correlation_matrix = base

    L = np.linalg.cholesky(correlation_matrix)
    raw = rng.standard_normal((n_weeks, n_channels))
    correlated = raw @ L.T

    spend = np.exp(correlated * 0.3 + 1.0)
    spend = np.maximum(spend, 0.01)

    return spend


def _geometric_adstock_np(
    x: np.ndarray, alpha: float, l_max: int, normalize: bool = True
) -> np.ndarray:
    """Apply geometric adstock to a 1-D series (numpy)."""
    n = len(x)
    weights = alpha ** np.arange(l_max)
    if normalize:
        weights = weights / weights.sum()

    out = np.zeros(n)
    for t in range(n):
        for lag in range(min(l_max, t + 1)):
            out[t] += weights[lag] * x[t - lag]
    return out


def _logistic_saturation_np(x: np.ndarray, lam: float) -> np.ndarray:
    """Numpy logistic saturation (without beta scaling)."""
    return (1.0 - np.exp(-lam * x)) / (1.0 + np.exp(-lam * x))


def generate_experiment_fixture(
    channels: list[str] | None = None,
    saturation: str = "logistic",
    adstock: str = "geometric",
    true_params: dict[str, dict[str, float]] | None = None,
    n_weeks: int = 104,
    noise_std: float = 0.5,
    intercept: float = 10.0,
    l_max: int = 8,
    normalize: bool = True,
    seed: int = 42,
    fit_model: bool = True,
    n_chains: int = 2,
    n_draws: int = 2000,
    n_tune: int = 1000,
    target_accept: float = 0.9,
) -> InferenceData:
    """Generate a realistic InferenceData fixture for testing.

    Simulates spend data, applies a known DGP, and (optionally) fits a
    PyMC-Marketing MMM to produce realistic posterior samples.

    Parameters
    ----------
    channels : list[str] | None
        Channel names. Defaults to ``["tv", "search", "social"]``.
    saturation : str
        Saturation type. Only ``"logistic"`` is supported.
    adstock : str
        Adstock type. Only ``"geometric"`` is supported.
    true_params : dict | None
        Ground-truth parameters per channel. Each value is a dict with
        ``"lam"``, ``"beta"``, ``"alpha"`` keys. If None, uses defaults.
    n_weeks : int
        Number of weeks to simulate.
    noise_std : float
        Standard deviation of observation noise.
    intercept : float
        Intercept of the DGP.
    l_max : int
        Maximum adstock lag.
    normalize : bool
        Whether adstock weights are normalised.
    seed : int
        Random seed.
    fit_model : bool
        If True, fits an actual MMM (slow, 2-5 minutes). If False,
        creates synthetic posterior samples from the true parameters
        with added noise (fast, for testing).
    n_chains : int
        Number of MCMC chains (only used if ``fit_model=True``).
    n_draws : int
        Number of MCMC draws per chain (only used if ``fit_model=True``).
    n_tune : int
        Number of tuning steps (only used if ``fit_model=True``).
    target_accept : float
        Target acceptance rate (only used if ``fit_model=True``).

    Returns
    -------
    InferenceData
        An ArviZ InferenceData suitable for
        :meth:`ExperimentDesigner.from_idata`.
    """
    if channels is None:
        channels = ["tv", "search", "social"]
    n_channels = len(channels)

    if true_params is None:
        true_params = {
            "tv": {"lam": 0.5, "beta": 3.0, "alpha": 0.7},
            "search": {"lam": 2.0, "beta": 1.5, "alpha": 0.3},
            "social": {"lam": 1.0, "beta": 0.8, "alpha": 0.5},
        }

    rng = np.random.default_rng(seed)

    spend = _simulate_spend(n_weeks, n_channels, channels, rng=rng)

    y = np.full(n_weeks, intercept)
    for i, ch in enumerate(channels):
        p = true_params[ch]
        adstocked = _geometric_adstock_np(spend[:, i], p["alpha"], l_max, normalize)
        saturated = _logistic_saturation_np(adstocked, p["lam"])
        y += p["beta"] * saturated

    y += rng.normal(0, noise_std, n_weeks)

    spend_corr = np.corrcoef(spend.T)
    if spend_corr.ndim == 0:
        spend_corr = spend_corr.reshape(1, 1)

    max_spend = np.max(np.abs(spend), axis=0)
    spend_scaled = spend / max_spend
    max_y = np.max(np.abs(y))
    y_scaled = y / max_y

    current_spend = np.mean(spend_scaled[-8:], axis=0)

    y_pred_true = np.full(n_weeks, intercept / max_y)
    for i, ch in enumerate(channels):
        p = true_params[ch]
        adstocked = _geometric_adstock_np(
            spend_scaled[:, i], p["alpha"], l_max, normalize
        )
        saturated = _logistic_saturation_np(adstocked, p["lam"])
        y_pred_true += (p["beta"] / max_y) * saturated

    residuals = y_scaled - y_pred_true
    residual_std = float(np.std(residuals))
    if len(residuals) > 2:
        residual_autocorr = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    else:
        residual_autocorr = 0.0

    if fit_model:
        idata = _fit_mmm_for_fixture(
            spend,
            y,
            channels,
            l_max,
            normalize,
            seed,
            n_chains,
            n_draws,
            n_tune,
            target_accept,
        )
    else:
        idata = _create_synthetic_posterior(
            channels,
            true_params,
            rng,
            n_draws,
            n_chains,
        )

    constant_data = xr.Dataset(
        {
            "current_weekly_spend": xr.DataArray(
                current_spend,
                dims=["channel"],
                coords={"channel": channels},
            ),
            "residual_std": xr.DataArray(residual_std),
            "residual_autocorr": xr.DataArray(residual_autocorr),
            "l_max": xr.DataArray(l_max),
            "normalize": xr.DataArray(normalize),
            "spend_correlation": xr.DataArray(
                spend_corr,
                dims=["channel", "channel_bis"],
                coords={"channel": channels, "channel_bis": channels},
            ),
        },
        attrs={
            "saturation_type": saturation,
            "adstock_type": adstock,
            "n_weeks": n_weeks,
            "intercept": intercept,
            "noise_std": noise_std,
            "true_params": str(true_params),
        },
    )
    idata.add_groups(constant_data=constant_data)

    return idata


def _create_synthetic_posterior(
    channels: list[str],
    true_params: dict[str, dict[str, float]],
    rng: np.random.Generator,
    n_draws: int = 2000,
    n_chains: int = 2,
) -> InferenceData:
    """Create synthetic posterior samples around true parameter values.

    This is a fast alternative to fitting when realistic MCMC structure
    is not needed (e.g., for unit tests).
    """
    lam_arr = np.zeros((n_chains, n_draws, len(channels)))
    beta_arr = np.zeros((n_chains, n_draws, len(channels)))
    alpha_arr = np.zeros((n_chains, n_draws, len(channels)))

    for i, ch in enumerate(channels):
        p = true_params[ch]
        lam_arr[:, :, i] = np.maximum(
            p["lam"] + rng.normal(0, p["lam"] * 0.15, (n_chains, n_draws)),
            0.01,
        )
        beta_arr[:, :, i] = np.maximum(
            p["beta"] + rng.normal(0, p["beta"] * 0.15, (n_chains, n_draws)),
            0.01,
        )
        alpha_arr[:, :, i] = np.clip(
            p["alpha"] + rng.normal(0, 0.05, (n_chains, n_draws)),
            0.01,
            0.99,
        )

    posterior = xr.Dataset(
        {
            "saturation_lam": xr.DataArray(
                lam_arr,
                dims=["chain", "draw", "channel"],
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "channel": channels,
                },
            ),
            "saturation_beta": xr.DataArray(
                beta_arr,
                dims=["chain", "draw", "channel"],
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "channel": channels,
                },
            ),
            "adstock_alpha": xr.DataArray(
                alpha_arr,
                dims=["chain", "draw", "channel"],
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "channel": channels,
                },
            ),
        }
    )

    return az.InferenceData(posterior=posterior)


def _fit_mmm_for_fixture(
    spend: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    l_max: int,
    normalize: bool,
    seed: int,
    n_chains: int,
    n_draws: int,
    n_tune: int,
    target_accept: float,
) -> InferenceData:
    """Fit a PyMC-Marketing MMM on simulated data."""
    try:
        import pandas as pd

        from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
    except ImportError as e:
        raise ImportError(
            "PyMC-Marketing must be installed to fit an MMM fixture. "
            "Use fit_model=False for synthetic posteriors."
        ) from e

    dates = pd.date_range("2020-01-06", periods=len(y), freq="W-MON")
    df = pd.DataFrame(spend, columns=channels)
    df["date"] = dates
    df["y"] = y

    X = df[["date", *channels]]
    y_series = df["y"]

    mmm = MMM(
        date_column="date",
        channel_columns=channels,
        target_column="y",
        adstock=GeometricAdstock(l_max=l_max, normalize=normalize),
        saturation=LogisticSaturation(),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mmm.fit(
            X,
            y_series,
            random_seed=seed,
        )

    return mmm.idata
