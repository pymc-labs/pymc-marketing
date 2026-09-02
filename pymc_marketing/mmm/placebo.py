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

An MMM is fitted against the same observed outcome it later explains, so part of
the contribution it reports is attribution the model would hand to *any*
plausible-looking spend series. A placebo channel measures that floor: add a
channel that cannot have caused anything, refit, and see how much credit it still
collects.

The measurement takes **two fits**, not one. Contribution shares are
compositional, so adding a third channel shrinks every share by arithmetic alone.
Without a baseline fit there is no way to separate "the placebo revealed that this
channel was overstated" from "three channels divide the same pie three ways", and
those have opposite consequences for a media plan::

    X_placebo, placebo = add_placebo_channel(X, "x1", group_column="geo", seed=1)

    with_placebo = fit(X_placebo, y, channels + [placebo])
    without_placebo = fit(X_placebo, y, channels)

    summary = summarize_placebo_contribution(
        with_placebo.posterior["channel_contribution_original_scale"],
        without_placebo.posterior["channel_contribution_original_scale"],
        placebo,
    )

Read ``lost`` against ``dilution_alone``. A channel that loses about the dilution
amount was not specifically overstated: the placebo took its cut from everyone. A
channel that loses appreciably more was carrying credit a shifted copy of a spend
series could take from it, and that is the finding worth acting on.

Run several placebos as **separate refits**, one placebo each. Five placebos in a
single model is not the same experiment: shares are compositional, so each takes
roughly a seventh of the total by arithmetic and the floor comes out artificially
low.

See the ``mmm_placebo_check`` notebook for a worked example.
"""

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "add_placebo_channel",
    "contribution_share",
    "placebo_correlation",
    "summarize_placebo_contribution",
]


def _as_generator(
    seed: int | np.random.Generator | None,
) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def placebo_correlation(
    X: pd.DataFrame, channel_column: str, placebo_name: str
) -> float:
    """Correlation between a real spend column and its placebo copy."""
    return float(X[channel_column].corr(X[placebo_name]))


def add_placebo_channel(
    X: pd.DataFrame,
    channel_column: str,
    *,
    group_column: str | None = None,
    method: str = "shift",
    placebo_name: str | None = None,
    seed: int | np.random.Generator | None = None,
    max_abs_corr: float = 0.3,
    max_tries: int = 50,
) -> tuple[pd.DataFrame, str]:
    """Return a copy of ``X`` with a placebo copy of one spend channel.

    Parameters
    ----------
    X : pd.DataFrame
        Model input data containing the channel column.
    channel_column : str
        Real spend column to build the placebo from.
    group_column : str, optional
        Shift within each group, for example a geo, so that the placebo stays a
        plausible series for that market.
    method : str, default "shift"
        ``"shift"`` circularly shifts, preserving flighting and autocorrelation,
        and is the harder placebo. ``"permute"`` preserves only the marginal
        distribution of spend and is the easier one.
    placebo_name : str, optional
        Column name for the placebo. Defaults to ``f"{channel_column}_placebo"``.
    seed : int or np.random.Generator, optional
        Seed or generator for the offset.
    max_abs_corr : float, default 0.3
        Reject a draw that leaves the placebo correlated with its source above
        this, and draw again.
    max_tries : int, default 50
        Give up after this many rejected draws and raise.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The input frame with the placebo column added, and that column's name.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or no draw satisfied ``max_abs_corr``.

    Notes
    -----
    The correlation guard is not cosmetic. A circular shift of one step, or of
    ``n - 1`` steps, is the original series barely moved and is still nearly
    aligned with the outcome, so it is not a placebo at all. On the 159-week
    series of ``mmm_multidimensional_example``, four of the 158 available offsets
    leave ``abs(corr) > 0.5`` and the worst is 0.76, so an unseeded caller draws a
    near-copy about 2.5 % of the time. Restricted to the draws this guard admits,
    the largest correlation available is 0.32.

    Examples
    --------
    Add a placebo built from one channel and shifted within each geo:

    .. code-block:: python

        from pymc_marketing.mmm import add_placebo_channel, placebo_correlation

        X_placebo, placebo = add_placebo_channel(X, "x1", group_column="geo", seed=1)
        placebo_correlation(X_placebo, "x1", placebo)
    """
    if method not in ("shift", "permute"):
        raise ValueError(f"unknown method {method!r}, expected 'shift' or 'permute'")

    rng = _as_generator(seed)
    name = placebo_name or f"{channel_column}_placebo"
    if group_column is None:
        groups = [X]
    else:
        groups = [group for _, group in X.groupby(group_column, sort=False)]

    for _ in range(max_tries):
        parts = []
        for group in groups:
            values = group[channel_column].to_numpy()
            if method == "shift":
                # an offset of 0 would hand back the real channel unchanged
                shifted = np.roll(values, int(rng.integers(1, len(values))))
            else:
                shifted = rng.permutation(values)
            parts.append(pd.Series(shifted, index=group.index))
        out = X.copy()
        out[name] = pd.concat(parts).sort_index()
        if abs(placebo_correlation(out, channel_column, name)) <= max_abs_corr:
            return out, name

    raise ValueError(
        f"no draw left abs(corr) <= {max_abs_corr} after {max_tries} tries; the "
        "series may be too short or too strongly periodic for a shift placebo, so "
        "try method='permute'"
    )


def contribution_share(contribution: xr.DataArray) -> xr.DataArray:
    """Posterior share of total absolute contribution, per channel.

    ``np.abs`` is a no-op wherever ``saturation_beta`` is strictly positive, which
    is the usual specification. Under one where beta can go negative it would
    score a large negative contribution as though it were a large positive one;
    there you want the signed sum, or to handle the sign separately.
    """
    extra_dims = [
        dim for dim in contribution.dims if dim not in ("chain", "draw", "channel")
    ]
    total = np.abs(contribution).sum(dim=extra_dims)
    return total / total.sum(dim="channel")


def summarize_placebo_contribution(
    with_placebo: xr.DataArray,
    without_placebo: xr.DataArray,
    placebo_name: str,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Summarise a placebo check across the two fits it requires.

    Parameters
    ----------
    with_placebo, without_placebo : xr.DataArray
        Channel-contribution posteriors from the fit that included the placebo
        and from the baseline fit that did not.
    placebo_name : str
        Name of the placebo channel. It must appear in ``with_placebo`` and must
        not appear in ``without_placebo``.
    hdi_prob : float, default 0.94
        Credible-interval mass.

    Returns
    -------
    pd.DataFrame
        Indexed by channel and sorted by share descending, with columns
        ``share``, ``hdi_low``, ``hdi_high``, ``prob_exceeds_placebo``,
        ``share_without_placebo``, ``lost`` and ``dilution_alone``. The placebo's
        own row carries NaN in the last four, which have no meaning for it.

    Raises
    ------
    ValueError
        If the placebo is missing from the first fit or present in the second.

    Examples
    --------
    Compare the two fits the check requires:

    .. code-block:: python

        from pymc_marketing.mmm import summarize_placebo_contribution

        var = "channel_contribution_original_scale"
        summarize_placebo_contribution(
            mmm.idata.posterior[var],
            mmm_baseline.idata.posterior[var],
            placebo,
        )

    ``lost`` close to ``dilution_alone`` means the placebo took a proportional
    cut from every channel and singled none of them out. A channel losing
    appreciably more than its ``dilution_alone`` is the one to look at.
    """
    channels_with = [str(c) for c in np.asarray(with_placebo.channel.values)]
    channels_without = [str(c) for c in np.asarray(without_placebo.channel.values)]
    if placebo_name not in channels_with:
        raise ValueError(f"{placebo_name!r} is not a channel of the with-placebo fit")
    if placebo_name in channels_without:
        raise ValueError(
            f"{placebo_name!r} appears in the baseline fit; the baseline must be "
            "the same model without the placebo channel"
        )

    share = contribution_share(with_placebo)
    baseline = contribution_share(without_placebo).mean(dim=("chain", "draw"))
    placebo_share = share.sel(channel=placebo_name)
    floor = float(placebo_share.mean())

    rows = []
    for channel in channels_with:
        draws = share.sel(channel=channel)
        low, high = np.asarray(az.hdi(draws.values.ravel(), prob=hdi_prob)).ravel()
        is_placebo = channel == placebo_name
        without = np.nan if is_placebo else float(baseline.sel(channel=channel))
        mean_share = float(draws.mean())
        rows.append(
            {
                "channel": channel,
                "share": mean_share,
                "hdi_low": float(low),
                "hdi_high": float(high),
                "prob_exceeds_placebo": np.nan
                if is_placebo
                else float((draws > placebo_share).mean()),
                "share_without_placebo": without,
                "lost": np.nan if is_placebo else without - mean_share,
                # what proportional dilution alone predicts: the placebo takes
                # `floor` out of the total and leaves the real channels' standing
                # relative to each other untouched
                "dilution_alone": np.nan if is_placebo else without * floor,
            }
        )

    return pd.DataFrame(rows).sort_values("share", ascending=False).set_index("channel")
