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
"""Placebo (negative-control) channels for MMM misattribution checks.

An MMM is selected and tuned against the same observed outcome it explains,
so some of the contribution it reports is attribution the model would hand
to *any* plausible-looking spend series. A placebo channel measures that
floor directly: add a channel that cannot have caused anything — the real
spend series with its time alignment destroyed — refit, and see how much
credit the model gives it. Channel contributions worth acting on should
clearly exceed what the model attributes to noise.

Workflow::

    X_placebo, name = add_placebo_channel(X, "TV", method="shift", seed=1)
    mmm.fit(X_placebo, y)  # placebo included as an ordinary channel
    contribution = ...  # the model's channel-contribution DataArray
    summary = summarize_placebo_contribution(contribution, name)

The functions here are deliberately model-agnostic: they operate on the
input dataframe and on any channel-contribution ``xarray.DataArray``, so
they work with every MMM variant in this package.
"""

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "add_placebo_channel",
    "summarize_placebo_contribution",
]


def add_placebo_channel(
    X: pd.DataFrame,
    channel_column: str,
    method: str = "shift",
    placebo_name: str | None = None,
    seed: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return a copy of ``X`` with a placebo copy of one spend channel.

    Parameters
    ----------
    X : pd.DataFrame
        Model input data containing the channel column.
    channel_column : str
        Real spend column to build the placebo from.
    method : str, default "shift"
        How the time alignment is destroyed:

        - ``"shift"``: circular shift by a random offset. Preserves the
          series' autocorrelation structure (flighting, seasonality of
          spend), making it the harder, more realistic placebo.
        - ``"permute"``: random permutation of rows. Preserves only the
          marginal distribution of spend.
    placebo_name : str, optional
        Name for the placebo column. Defaults to
        ``f"{channel_column}_placebo"``.
    seed : int, Generator, optional
        Seed for reproducible placebo construction.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The augmented dataframe and the placebo column's name. Include the
        returned name in ``channel_columns`` when fitting.

    Examples
    --------
    .. code-block:: python

        X_placebo, name = add_placebo_channel(X, "TV", method="shift", seed=1)
        mmm = MMM(channel_columns=[*channels, name], ...)
        mmm.fit(X_placebo, y)

    References
    ----------
    .. [1] Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. (2010).
       "Negative controls: a tool for detecting confounding and bias in
       observational studies." Epidemiology, 21(3), 383-388.
    """
    if channel_column not in X.columns:
        raise KeyError(f"channel_column {channel_column!r} not in X")
    name = placebo_name or f"{channel_column}_placebo"
    if name in X.columns:
        raise ValueError(f"placebo column {name!r} already exists in X")

    rng = np.random.default_rng(seed)
    values = X[channel_column].to_numpy()
    n = len(values)
    if n < 3:
        raise ValueError("need at least 3 rows to build a placebo channel")

    if method == "shift":
        # a shift of 0 (or n) would reproduce the real channel exactly
        offset = int(rng.integers(1, n))
        placebo = np.roll(values, offset)
    elif method == "permute":
        placebo = rng.permutation(values)
    else:
        raise ValueError(f"unknown method {method!r}; use 'shift' or 'permute'")

    X_out = X.copy()
    X_out[name] = placebo
    return X_out, name


def summarize_placebo_contribution(
    channel_contribution: xr.DataArray,
    placebo_name: str,
    channel_dim: str = "channel",
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Summarize how much credit the model gave the placebo channel.

    Parameters
    ----------
    channel_contribution : xr.DataArray
        Posterior channel contributions with a channel dimension and
        ``chain``/``draw`` sample dimensions (any other dimensions, e.g.
        date, are summed over before comparing channels).
    placebo_name : str
        Name of the placebo channel along ``channel_dim``.
    channel_dim : str, default "channel"
        Name of the channel dimension.
    hdi_prob : float, default 0.94
        Probability mass of the highest-density interval reported.

    Returns
    -------
    pd.DataFrame
        One row per channel: posterior mean share of total absolute
        contribution, HDI bounds of the share, and whether the channel's
        share distribution exceeds the placebo's (share of posterior
        draws where channel share > placebo share). The placebo row is
        the noise floor the other rows should clearly exceed.

    Examples
    --------
    .. code-block:: python

        contribution = mmm.compute_channel_contribution_original_scale()
        summary = summarize_placebo_contribution(contribution, "TV_placebo")
        summary[summary["prob_exceeds_placebo"] < 0.9]  # channels to distrust
    """
    import arviz as az

    if placebo_name not in channel_contribution[channel_dim].values:
        raise KeyError(
            f"placebo channel {placebo_name!r} not found on dimension {channel_dim!r}"
        )

    sample_dims = [d for d in ("chain", "draw") if d in channel_contribution.dims]
    extra_dims = [
        d for d in channel_contribution.dims if d not in (*sample_dims, channel_dim)
    ]
    total_by_channel = np.abs(channel_contribution).sum(dim=extra_dims)
    share = total_by_channel / total_by_channel.sum(dim=channel_dim)

    placebo_share = share.sel({channel_dim: placebo_name})
    rows = []
    for channel in share[channel_dim].values:
        channel_share = share.sel({channel_dim: channel})
        hdi_result = np.asarray(
            az.hdi(channel_share.values.flatten(), prob=hdi_prob)
        ).ravel()
        hdi_low, hdi_high = float(hdi_result[0]), float(hdi_result[1])
        exceeds = float(
            (channel_share > placebo_share).mean(dim=sample_dims)
            if channel != placebo_name
            else np.nan
        )
        rows.append(
            {
                "channel": channel,
                "share_mean": float(channel_share.mean()),
                "share_hdi_low": float(hdi_low),
                "share_hdi_high": float(hdi_high),
                "prob_exceeds_placebo": exceeds,
                "is_placebo": channel == placebo_name,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("share_mean", ascending=False)
        .reset_index(drop=True)
    )
