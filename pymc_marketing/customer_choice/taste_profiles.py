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
r"""Post-hoc taste-profile analysis for fitted :class:`BayesianBLP` models.

This module exposes the lenses the demo notebooks use to characterise *who*
buys in each market, given the posterior over preference parameters and the
Halton grid of consumer types attached to the model:

- :func:`buyer_nu_posterior`       posterior of average buyer :math:`\bar\nu_{m,d}`
- :func:`brand_buyer_nu`           brand-level buyer profile :math:`\bar\nu_{m,j,d}`
- :func:`demand_concentration_gini` Gini of inside-good demand across types
- :func:`taste_type_demand_share`  bucketed sensitive / modal / insensitive shares
- :func:`consumer_taste_grid`      the Halton grid as a labelled DataFrame

Plus four convenience plotters that wrap each computation in a standard
matplotlib figure with ``layout="constrained"`` so colorbar layouts compose
cleanly with the rest of pymc-marketing's notebook style.

All public functions take a fitted ``BayesianBLP`` model and raise
``RuntimeError`` when the model has no posterior. The single internal helper
:func:`_compute_inside_choice_probs` is the *only* place that reaches into the
model's private API; all public functions delegate through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from pymc_marketing.customer_choice.bayesian_blp import BayesianBLP


def _require_fitted(model: BayesianBLP, fn_name: str) -> None:
    """Raise ``RuntimeError`` if ``model`` has no posterior."""
    if model.idata is None or "posterior" not in model.idata:
        raise RuntimeError(
            f"Model has no posterior; call .fit(...) before {fn_name}()."
        )


def _require_random_coefs(model: BayesianBLP) -> None:
    """Raise ``RuntimeError`` if ``model`` has no random coefficients."""
    if model._n_random == 0:
        raise RuntimeError(
            "Taste-profile analyses require at least one random coefficient, "
            "but this model was constructed with random_coef_on=[] — there "
            "is no consumer-heterogeneity dimension to profile."
        )


def _price_dim_index(model: BayesianBLP, fn_name: str) -> int:
    """Index of the price dimension in the model's random-coefficient grid.

    Raises ``ValueError`` when the model has no random coefficient on price,
    since the calling function's sensitivity buckets / x-axis are defined on
    the price taste shock specifically.
    """
    try:
        return model._random_coef_names.index("price")
    except ValueError:
        raise ValueError(
            f"{fn_name}() profiles consumers along the price taste dimension, "
            "but this model has no random coefficient on price "
            f"(random_coef_on={model._random_coef_names!r}). Construct the "
            "model with 'price' in random_coef_on to use this function."
        ) from None


def _compute_inside_choice_probs(
    model: BayesianBLP, n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adapter that pulls inside-good probabilities for downstream summaries.

    Returns
    -------
    s_in_per_draw : np.ndarray
        Shape ``(n_samples, M, J, R)``. Per-draw probability that a consumer
        of type ``r`` buys inside product ``j`` in market ``m``.
    s_out_per_draw : np.ndarray
        Shape ``(n_samples, M, R)``. Per-draw outside-good probability.
    halton_grid : np.ndarray
        Shape ``(R, D)``. The Halton consumer-type grid the model integrates
        over; column ``d`` matches ``model._random_coef_names[d]``.

    Uses the public :meth:`BayesianBLP.iterate_posterior_samples` and
    :meth:`BayesianBLP.batch_shares` methods. All public functions in this
    module delegate here so any future change to the inside-choice contract
    only requires touching one place.
    """
    _require_random_coefs(model)
    alpha_M, beta_M, xi_M, sigma_M = model.iterate_posterior_samples(n_samples)
    s_in_per_draw, _, s_out_per_draw, _, _ = model.batch_shares(
        alpha_M, beta_M, xi_M, sigma_M, model._price
    )
    return s_in_per_draw, s_out_per_draw, model._halton[: model.n_mc_draws]


# --------------------------------------------------------------------------- #
# Compute functions
# --------------------------------------------------------------------------- #
def consumer_taste_grid(model: BayesianBLP) -> pd.DataFrame:
    """Halton grid as a labelled DataFrame.

    Parameters
    ----------
    model : BayesianBLP
        Constructed (need not be fitted) BayesianBLP.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_mc_draws, n_random)``. Column names are the random-coefficient
        names, in the same order as ``model._random_coef_names``. Each row is
        one consumer type; column ``d`` carries the standard-normal taste draw
        for that random-coefficient dimension.

    Raises
    ------
    RuntimeError
        If the model was constructed with ``random_coef_on=[]`` (no
        consumer-heterogeneity dimensions exist).
    """
    _require_random_coefs(model)
    return pd.DataFrame(
        model._halton[: model.n_mc_draws],
        columns=list(model._random_coef_names),
    )


def buyer_nu_posterior(model: BayesianBLP, n_samples: int = 300) -> np.ndarray:
    r"""Posterior of the average buyer's taste vector per market.

    For each posterior sample :math:`s`, market :math:`m`, and random-coef
    dimension :math:`d`,

    .. math::
        \bar\nu_{m,d}^{(s)} =
        \frac{\sum_r \nu_{r,d} \cdot s^{\mathrm{in}}_{m,r}(s)}
             {\sum_r s^{\mathrm{in}}_{m,r}(s)}

    where :math:`s^{\mathrm{in}}_{m,r}` sums the per-consumer-type inside-good
    probability across products.

    Parameters
    ----------
    model : BayesianBLP
        A fitted model.
    n_samples : int
        Number of posterior draws to use.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, M, D)``. The posterior of the average buyer's taste
        vector per market.
    """
    _require_fitted(model, "buyer_nu_posterior")
    s_in_per_draw, _, halton = _compute_inside_choice_probs(model, n_samples)
    s_in_total = s_in_per_draw.sum(axis=2)  # (S, M, R)
    weighted = np.einsum("smr,rd->smd", s_in_total, halton)
    total = s_in_total.sum(axis=2, keepdims=True)
    return weighted / np.maximum(total, 1e-30)


def brand_buyer_nu(
    model: BayesianBLP, n_samples: int = 200, dim: int = 0
) -> np.ndarray:
    r"""Posterior-mean buyer taste per brand and market, for one taste dimension.

    For each market :math:`m` and brand :math:`j`,

    .. math::
        \bar\nu_{m,j,d} = E[\nu_d \mid \mathrm{buys\ brand\ } j \mathrm{\ in\ } m]

    averaged across posterior draws.

    Parameters
    ----------
    model : BayesianBLP
        A fitted model.
    n_samples : int
        Number of posterior draws.
    dim : int
        Which random-coefficient dimension to slice (``0`` is typically price).

    Returns
    -------
    np.ndarray
        Shape ``(M, J)``. Posterior-mean buyer taste on the chosen dimension
        for each (market, brand).

    Raises
    ------
    IndexError
        If ``dim`` is outside ``[0, n_random)``.
    """
    _require_fitted(model, "brand_buyer_nu")
    D = model._halton.shape[1]
    if not 0 <= dim < D:
        raise IndexError(
            f"dim={dim} is out of range for a model with {D} random-coefficient "
            f"dimension(s); valid range is [0, {D})."
        )
    s_in_per_draw, _, halton = _compute_inside_choice_probs(model, n_samples)
    nu_d = halton[:, dim]
    weighted = (s_in_per_draw * nu_d).sum(axis=3)  # (S, M, J)
    total = s_in_per_draw.sum(axis=3)
    return (weighted / np.maximum(total, 1e-30)).mean(axis=0)


def demand_concentration_gini(model: BayesianBLP, n_samples: int = 300) -> np.ndarray:
    r"""Per-sample Gini of inside-good demand across consumer types, per market.

    For each (sample, market) the contributions are
    :math:`s^{\mathrm{in}}_{m,r}` summed across products. The Gini coefficient
    is computed via the sorted-and-weighted formula

    .. math::
        G = \frac{\sum_r (2r - R - 1)\, x_{(r)}}{R \sum_r x_{(r)}}

    where :math:`x_{(r)}` are the sorted contributions ascending. ``G=0``
    means demand is uniformly spread across consumer types; ``G \to 1``
    means one type carries almost all the demand.

    Parameters
    ----------
    model : BayesianBLP
        A fitted model.
    n_samples : int
        Number of posterior draws.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, M)``. Per-sample Gini per market.
    """
    _require_fitted(model, "demand_concentration_gini")
    s_in_per_draw, _, _ = _compute_inside_choice_probs(model, n_samples)
    s_in_total = s_in_per_draw.sum(axis=2)  # (S, M, R)
    R = s_in_total.shape[2]
    sorted_vals = np.sort(s_in_total, axis=2)
    weights = 2 * np.arange(1, R + 1) - R - 1
    return (sorted_vals * weights).sum(axis=2) / (
        R * np.maximum(sorted_vals.sum(axis=2), 1e-30)
    )


def taste_type_demand_share(
    model: BayesianBLP, n_samples: int = 200, threshold: float = 1.0
) -> pd.DataFrame:
    r"""Share of inside-good demand contributed by each price-taste bucket.

    Consumer types are split into three buckets by their price taste shock
    :math:`\nu_{\mathrm{price}}`:

    - ``sensitive``: :math:`\nu < -\mathrm{threshold}`
    - ``modal``: :math:`-\mathrm{threshold} \le \nu \le \mathrm{threshold}`
    - ``insensitive``: :math:`\nu > \mathrm{threshold}`

    With ``threshold=1.0`` the population baseline (under a flat logit) is
    ``{0.16, 0.68, 0.16}``.

    Parameters
    ----------
    model : BayesianBLP
        A fitted model.
    n_samples : int
        Number of posterior draws.
    threshold : float
        Boundary used to split the :math:`\nu` axis. Must be positive.

    Returns
    -------
    pd.DataFrame
        Columns: ``market, avg_price, sensitive_pct, modal_pct,
        insensitive_pct``. One row per market. The three ``*_pct`` columns
        sum to 1.0 per row.

    Raises
    ------
    ValueError
        If ``threshold`` is not strictly positive, or if the model has no
        random coefficient on price (the buckets are defined on the price
        taste shock).
    """
    _require_fitted(model, "taste_type_demand_share")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0; got {threshold}.")
    price_dim = _price_dim_index(model, "taste_type_demand_share")

    s_in_per_draw, _, halton = _compute_inside_choice_probs(model, n_samples)
    s_avg = s_in_per_draw.mean(axis=(0, 2))  # (M, R), averaged over draws and products
    nu_price = halton[:, price_dim]
    sensitive = nu_price < -threshold
    insensitive = nu_price > threshold
    modal = ~sensitive & ~insensitive

    rows = []
    for m in range(model._M):
        total = s_avg[m].sum()
        if total <= 0:
            sens = mod = ins = float("nan")
        else:
            sens = s_avg[m, sensitive].sum() / total
            mod = s_avg[m, modal].sum() / total
            ins = s_avg[m, insensitive].sum() / total
        rows.append(
            {
                "market": model._markets[m],
                "avg_price": float(model._price[m].mean()),
                "sensitive_pct": sens,
                "modal_pct": mod,
                "insensitive_pct": ins,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _default_price_span_markets(model: BayesianBLP, n: int = 4) -> list[int]:
    """Pick ``n`` markets spanning the price range, cheapest to dearest."""
    order = np.argsort(model._price.mean(axis=1))
    if n >= len(order):
        return [int(i) for i in order]
    fractions = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [int(order[i]) for i in fractions]


def plot_taste_profile_stacked(
    model: BayesianBLP,
    market_indices: list[int] | None = None,
    n_samples: int = 200,
    axes: list[Axes] | None = None,
) -> Figure:
    r"""Stacked area chart of consumer allocation across the :math:`\nu_{\mathrm{price}}` axis.

    The outside good sits at the bottom; inside products stack above. Reading
    left to right, as :math:`\nu` rises (less price-sensitive), the outside
    band shrinks and inside-product bands grow.

    Parameters
    ----------
    model : BayesianBLP
        A fitted model.
    market_indices : list of int, optional
        Markets to plot. Defaults to four markets spanning the price range
        (cheapest, 33rd / 67th percentile, dearest).
    n_samples : int
        Number of posterior draws.
    axes : list of matplotlib Axes, optional
        Pre-existing axes to draw into; must have length ``len(market_indices)``.
        When ``None``, a new figure with one axis per market is created.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the model has no random coefficient on price (the x-axis is the
        price taste shock).
    """
    _require_fitted(model, "plot_taste_profile_stacked")
    price_dim = _price_dim_index(model, "plot_taste_profile_stacked")
    if market_indices is None:
        market_indices = _default_price_span_markets(model, n=4)

    s_in_per_draw, s_out_per_draw, halton = _compute_inside_choice_probs(
        model, n_samples
    )
    s_avg = s_in_per_draw.mean(axis=0)  # (M, J, R)
    s_out_avg = s_out_per_draw.mean(axis=0)  # (M, R)

    nu = halton[:, price_dim]
    order = np.argsort(nu)
    nu_sorted = nu[order]

    n = len(market_indices)
    if axes is None:
        fig, axes = plt.subplots(
            n, 1, layout="constrained", figsize=(10, 3.2 * n), sharex=True
        )
        if n == 1:
            axes = [axes]
        else:
            axes = list(axes)
    else:
        if len(axes) != n:
            raise ValueError(
                f"axes has length {len(axes)} but market_indices has length {n}"
            )
        fig = axes[0].figure

    outside_color = "#d4c5a9"
    # tab20 colormap above 8 brands; default prop_cycle below for the synthetic case.
    if model._J > 8:
        cmap = plt.get_cmap("tab20", model._J)
        inside_colors = [cmap(j) for j in range(model._J)]
    else:
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        inside_colors = prop_cycle[: model._J]

    for ax, m_idx in zip(axes, market_indices, strict=True):
        stacks = [
            s_out_avg[m_idx, order],
            *[s_avg[m_idx, j, order] for j in range(model._J)],
        ]
        labels = ["outside", *model._inside_products]
        colors = [outside_color, *inside_colors]
        ax.stackplot(
            nu_sorted, *stacks, labels=labels, colors=colors, alpha=0.75, zorder=1
        )
        ax.axvline(0, color="black", linestyle=":", lw=1, alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Consumer allocation")
        avg_p = float(model._price[m_idx].mean())
        ax.set_title(f"Market {model._markets[m_idx]} (avg price = {avg_p:.2f})")
        if ax is axes[0]:
            # Legend only on the first panel — with many brands it dominates.
            if model._J > 8:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1),
                    fontsize=7,
                    ncol=1,
                )
            else:
                ax.legend(
                    loc="upper right",
                    ncol=model._J + 1,
                    fontsize=9,
                    framealpha=0.9,
                )

    axes[-1].set_xlabel(
        "nu_price (price taste shock, standard-normal scale)\n"
        "<- more price-sensitive   *   modal at 0   *   less price-sensitive ->"
    )
    return fig


def plot_buyer_profile_heatmap(
    model: BayesianBLP,
    n_samples: int = 300,
    ax: Axes | None = None,
) -> Figure:
    """``(market × dimension)`` diverging heatmap of the average buyer's taste vector.

    Each row is a market (sorted by avg price); each column is one random-
    coefficient dimension. Red cells mean the typical buyer scores
    above-modal on that dimension; blue means below-modal.

    Parameters
    ----------
    model : BayesianBLP
        A fitted model.
    n_samples : int
        Number of posterior draws.
    ax : matplotlib Axes, optional
        Pre-existing axis. When ``None`` a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_fitted(model, "plot_buyer_profile_heatmap")
    nu_bar = buyer_nu_posterior(model, n_samples=n_samples)  # (S, M, D)
    market_avg_price = model._price.mean(axis=1)
    order = np.argsort(market_avg_price)

    mean_profile = nu_bar.mean(axis=0)
    heatmap = mean_profile[order]
    D = model._halton.shape[1]

    if ax is None:
        fig, ax = plt.subplots(
            layout="constrained",
            figsize=(2 + 2.0 * D, 0.22 * model._M + 1),
        )
    else:
        fig = ax.figure

    vmax = float(np.abs(mean_profile).max())
    im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(D))
    ax.set_xticklabels([rf"$\bar\nu_{{{n}}}$" for n in model._random_coef_names])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [f"{model._markets[i]}  ($\\bar p$={market_avg_price[i]:.2f})" for i in order],
        fontsize=7,
    )
    ax.set_title("Average buyer taste profile per market")
    ax.set_xlabel("Taste dimension")
    fig.colorbar(im, ax=ax, label=r"$E[\nu_d \mid \mathrm{buys\ inside}]$")
    return fig


def plot_brand_buyer_heatmap(
    model: BayesianBLP,
    n_samples: int = 200,
    dim: int = 0,
    ax: Axes | None = None,
) -> Figure:
    """``(market × brand)`` diverging heatmap of brand-level buyer taste.

    Slices one taste dimension (``dim``, default = price). Brands and markets
    are sorted by their average buyer taste / avg price respectively to put
    similar profiles next to each other.

    Parameters
    ----------
    model : BayesianBLP
        A fitted model.
    n_samples : int
        Number of posterior draws.
    dim : int
        Which random-coefficient dimension to slice.
    ax : matplotlib Axes, optional
        Pre-existing axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_fitted(model, "plot_brand_buyer_heatmap")
    brand_nu = brand_buyer_nu(model, n_samples=n_samples, dim=dim)  # (M, J)

    brand_order = np.argsort(brand_nu.mean(axis=0))
    market_order = np.argsort(model._price.mean(axis=1))
    heatmap = brand_nu[market_order][:, brand_order]

    if ax is None:
        fig, ax = plt.subplots(layout="constrained", figsize=(11, 5.5))
    else:
        fig = ax.figure

    vmax = float(np.abs(brand_nu).max())
    im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(model._J))
    ax.set_xticklabels(
        [model._inside_products[j] for j in brand_order],
        rotation=75,
        fontsize=8,
    )
    ax.set_yticks(range(model._M))
    ax.set_yticklabels(
        [
            f"{model._markets[m]} ($\\bar p$={model._price[m].mean():.1f})"
            for m in market_order
        ],
        fontsize=8,
    )
    dim_name = model._random_coef_names[dim]
    fig.colorbar(
        im, ax=ax, label=rf"$E[\nu_{{{dim_name}}} \mid \mathrm{{buys\ brand}}]$"
    )
    ax.set_title(f"Brand-level buyer profile across markets (dim={dim_name})")
    ax.set_xlabel(f"Brand (sorted by avg buyer ν_{dim_name})")
    ax.set_ylabel("Market (sorted by avg price)")
    return fig


def plot_demand_concentration(
    model: BayesianBLP,
    n_samples: int = 300,
    ax: Axes | None = None,
) -> Figure:
    """Gini coefficient of inside-good demand vs. average market price.

    Each dot is one market; the error bar shows the 94% HDI of the Gini's
    posterior. The pattern is typically monotone: cheap markets serve all
    consumer types uniformly (low Gini); expensive markets concentrate
    demand on the insensitive tail (high Gini).

    Parameters
    ----------
    model : BayesianBLP
        A fitted model.
    n_samples : int
        Number of posterior draws.
    ax : matplotlib Axes, optional
        Pre-existing axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_fitted(model, "plot_demand_concentration")
    gini = demand_concentration_gini(model, n_samples=n_samples)
    g_mean = gini.mean(axis=0)
    g_lo, g_hi = np.percentile(gini, [3, 97], axis=0)
    prices = model._price.mean(axis=1)

    if ax is None:
        fig, ax = plt.subplots(layout="constrained", figsize=(9, 5))
    else:
        fig = ax.figure

    ax.errorbar(
        prices,
        g_mean,
        yerr=[g_mean - g_lo, g_hi - g_mean],
        fmt="o",
        color="darkred",
        alpha=0.6,
        capsize=3,
        markersize=5,
        label="posterior mean (94% HDI)",
    )
    ax.set_xlabel("Average market price")
    ax.set_ylabel("Gini coefficient of inside-good demand")
    ax.set_title("Demand concentrates on a narrow consumer slice as price rises")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


__all__ = [
    "brand_buyer_nu",
    "buyer_nu_posterior",
    "consumer_taste_grid",
    "demand_concentration_gini",
    "plot_brand_buyer_heatmap",
    "plot_buyer_profile_heatmap",
    "plot_demand_concentration",
    "plot_taste_profile_stacked",
    "taste_type_demand_share",
]
