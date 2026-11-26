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

"""Generate sample fits for the ExperimentDesigner tutorial and tests."""

from __future__ import annotations

import numpy as np
import xarray as xr
from pytensor.xtensor import as_xtensor

from pymc_marketing.mmm.experiment_design._statistics import residual_statistics
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from pymc_marketing.names.mmm import Dim, Var


def _simulate_spend(
    n_weeks: int,
    n_channels: int,
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


def _validate_transformations(saturation: str, adstock: str) -> None:
    if saturation != "logistic":
        raise NotImplementedError(
            f"Saturation {saturation!r} is unsupported; use 'logistic'."
        )
    if adstock != "geometric":
        raise NotImplementedError(
            f"Adstock {adstock!r} is unsupported; use 'geometric'."
        )


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
    n_chains: int | None = None,
    n_draws: int | None = None,
    *,
    spend_lookback: int = 8,
) -> xr.DataTree:
    """Generate a reproducible experiment-design example or test fixture.

    Simulates weekly national spend and applies the package's geometric adstock
    and logistic saturation. The fast mode supplies synthetic draws for the
    experiment-design tutorial and regression tests. The fitted mode runs an MMM
    on the simulated data to capture posterior dependence between parameters.
    See :ref:`mmm_data_generator` for an end-to-end example of parameter recovery
    with the MMM graph.

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
    n_chains : int | None
        Number of synthetic chains. Defaults to two in synthetic mode.
    n_draws : int | None
        Number of draws per synthetic chain. Defaults to 2,000 in synthetic
        mode. Leave both counts unset when fitting an MMM.
    spend_lookback : int
        Number of recent weeks used for the current spend in the fixture.

    Returns
    -------
    xr.DataTree
        An xarray DataTree suitable for
        :meth:`ExperimentDesigner.from_idata`.
    """
    _validate_transformations(saturation, adstock)
    if fit_model and (n_chains is not None or n_draws is not None):
        raise ValueError("n_chains and n_draws only apply to synthetic draws.")
    if spend_lookback < 1:
        raise ValueError("spend_lookback must be positive.")
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

    spend = _simulate_spend(n_weeks, n_channels, rng=rng)
    params = {
        key: np.array([true_params[ch][key] for ch in channels])
        for key in ("alpha", "lam", "beta")
    }
    adstocked = geometric_adstock(
        as_xtensor(spend, dims=(Dim.DATE, Dim.CHANNEL)),
        alpha=as_xtensor(params["alpha"], dims=(Dim.CHANNEL,)),
        l_max=l_max,
        normalize=normalize,
        dim=Dim.DATE,
    )
    response = (
        logistic_saturation(
            adstocked, lam=as_xtensor(params["lam"], dims=(Dim.CHANNEL,))
        )
        .transpose(Dim.DATE, Dim.CHANNEL)
        .values.eval()
    )
    noise = rng.normal(0, noise_std, n_weeks)
    y = intercept + response @ params["beta"] + noise

    spend_corr = np.atleast_2d(np.corrcoef(spend.T))
    if fit_model:
        idata = _fit_mmm_for_fixture(spend, y, channels, l_max, normalize, seed)
        spend_scale = idata.constant_data[Var.CHANNEL_SCALE].values
        target_scale = float(idata.constant_data[Var.TARGET_SCALE].item())
    else:
        spend_scale = np.max(np.abs(spend), axis=0)
        target_scale = float(np.max(np.abs(y))) or 1.0
        scaled_params = {
            ch: {
                "lam": true_params[ch]["lam"] * spend_scale[i],
                "beta": true_params[ch]["beta"] / target_scale,
                "alpha": true_params[ch]["alpha"],
            }
            for i, ch in enumerate(channels)
        }
        idata = _create_synthetic_posterior(
            channels,
            scaled_params,
            rng,
            n_draws=n_draws if n_draws is not None else 2000,
            n_chains=n_chains if n_chains is not None else 2,
        )
        idata["constant_data"] = xr.Dataset(
            {
                Var.CHANNEL_SCALE: ((Dim.CHANNEL,), spend_scale),
                Var.TARGET_SCALE: target_scale,
            },
            coords={Dim.CHANNEL: channels},
        )

    current_spend = (spend[-spend_lookback:] / spend_scale).mean(axis=0)
    residual_std, residual_autocorr = residual_statistics(noise / target_scale)

    constant_data = xr.Dataset(
        {
            "current_weekly_spend": xr.DataArray(
                current_spend,
                dims=[Dim.CHANNEL],
                coords={Dim.CHANNEL: channels},
            ),
            "residual_std": xr.DataArray(residual_std),
            "residual_autocorr": xr.DataArray(residual_autocorr),
            "l_max": xr.DataArray(l_max),
            "normalize": xr.DataArray(normalize),
            "spend_correlation": xr.DataArray(
                spend_corr,
                dims=[Dim.CHANNEL, "channel_bis"],
                coords={Dim.CHANNEL: channels, "channel_bis": channels},
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
    if "constant_data" in idata:
        constant_data = constant_data.merge(
            idata.constant_data.dataset, combine_attrs="no_conflicts"
        )
    idata["constant_data"] = constant_data

    return idata


def _create_synthetic_posterior(
    channels: list[str],
    true_params: dict[str, dict[str, float]],
    rng: np.random.Generator,
    n_draws: int = 2000,
    n_chains: int = 2,
) -> xr.DataTree:
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
            0.0,
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
                dims=["chain", "draw", Dim.CHANNEL],
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    Dim.CHANNEL: channels,
                },
            ),
            "saturation_beta": xr.DataArray(
                beta_arr,
                dims=["chain", "draw", Dim.CHANNEL],
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    Dim.CHANNEL: channels,
                },
            ),
            "adstock_alpha": xr.DataArray(
                alpha_arr,
                dims=["chain", "draw", Dim.CHANNEL],
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    Dim.CHANNEL: channels,
                },
            ),
        }
    )

    return xr.DataTree.from_dict({"posterior": posterior})


def _fit_mmm_for_fixture(
    spend: np.ndarray,
    y: np.ndarray,
    channels: list[str],
    l_max: int,
    normalize: bool,
    seed: int,
) -> xr.DataTree:
    """Fit a PyMC-Marketing MMM on simulated data."""
    import pandas as pd

    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    dates = pd.date_range("2020-01-06", periods=len(y), freq="W-MON")
    df = pd.DataFrame(spend, columns=channels)
    df[Dim.DATE] = dates
    df[Var.Y] = y

    X = df[[Dim.DATE, *channels]]
    y_series = df[Var.Y]

    mmm = MMM(
        date_column=Dim.DATE,
        channel_columns=channels,
        target_column=Var.Y,
        adstock=GeometricAdstock(l_max=l_max, normalize=normalize),
        saturation=LogisticSaturation(),
    )

    return mmm.fit(X, y_series, random_seed=seed)
