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

"""Matplotlib renderers for experiment-design diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pymc_marketing.mmm.experiment_design.recommendation import (
    ExperimentRecommendation,
    ExperimentRecommendations,
)


def plot_channel_diagnostics(
    channels: Sequence[str],
    metrics: dict[str, dict[str, float]],
    l_max: int,
    colors: dict[str, str] | None,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, np.ndarray]:
    if colors is None:
        colors = {ch: f"C{i}" for i, ch in enumerate(channels)}
    if figsize is None:
        figsize = (10, 4)

    n_metrics = 4
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    x_pos = np.arange(len(channels))
    bar_colors = [colors[ch] for ch in channels]

    titles = [
        "Response HDI Width",
        "Mean |Spend Correlation|",
        "Response Gradient\n(at operating point)",
        f"Ramp @ {l_max}w\n(fraction of steady state)",
    ]
    value_keys = ["hdi_width", "mean_correlation", "gradient", "ramp_at_lmax"]

    for ax_i, title, key in zip(axes, titles, value_keys, strict=True):
        vals = [metrics[ch][key] for ch in channels]
        ax_i.bar(x_pos, vals, color=bar_colors)
        ax_i.set_xticks(x_pos)
        ax_i.set_xticklabels(channels, rotation=45, ha="right")
        ax_i.set_title(title)

    fig.tight_layout()
    return fig, axes


def plot_power_cost(
    channels: Sequence[str],
    recommendations: ExperimentRecommendations | Sequence[ExperimentRecommendation],
    colors: dict[str, str] | None,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, Axes]:
    if figsize is None:
        figsize = (10, 4)
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = {ch: f"C{i}" for i, ch in enumerate(channels)}
    markers = {
        "increase": "^",
        "decrease": "v",
        "go-dark": "X",
    }

    for rec in recommendations:
        if rec.spend_change_frac == -1.0:
            direction = "go-dark"
        elif rec.spend_change_frac > 0:
            direction = "increase"
        else:
            direction = "decrease"

        ax.scatter(
            abs(rec.net_cost),
            rec.assurance,
            c=colors.get(rec.channel, "C0"),
            marker=markers[direction],
            s=60,
            alpha=0.7,
        )

    for ch, color in colors.items():
        ax.scatter([], [], c=color, label=ch, s=60)
    for name, marker in markers.items():
        ax.scatter([], [], c="gray", marker=marker, label=name, s=60)

    ax.set_xlabel("Absolute net cost (spend units)")
    ax.set_ylabel("Assurance (posterior-predictive power)")
    ax.set_title("Experiment Assurance vs. Cost")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig, ax


def plot_lift_distributions(
    channel: str,
    current_spend: float,
    predict_lift: Callable[[str, float, int], np.ndarray],
    spend_changes: list[float] | None,
    durations: list[int] | None,
    color: str | None,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, np.ndarray]:
    if spend_changes is None:
        spend_changes = [0.2, 0.5, -0.5, -1.0]
    if durations is None:
        durations = [4, 6, 8, 12]
    if color is None:
        color = "C0"

    n_rows, n_cols = len(spend_changes), len(durations)
    if figsize is None:
        figsize = (10, 3 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for i, frac in enumerate(spend_changes):
        for j, T in enumerate(durations):
            ax_ij = axes[i, j]
            delta_x = frac * current_spend
            lift = predict_lift(channel, delta_x, T)
            hdi = az.hdi(lift, prob=0.94)

            ax_ij.hist(lift, bins=50, density=True, alpha=0.6, color=color)
            ax_ij.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax_ij.axvspan(hdi[0], hdi[1], alpha=0.15, color=color)
            ax_ij.set_title(f"\u0394={frac * 100:+.0f}%, T={T}w", fontsize=9)
            if j == 0:
                ax_ij.set_ylabel("Density")
            if i == n_rows - 1:
                ax_ij.set_xlabel("Total Lift")

    fig.suptitle(f"Lift Distributions \u2014 {channel}", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_saturation_curve(
    channel: str,
    channel_index: int,
    current_spend: float,
    baseline: np.ndarray,
    steady_responses: Callable[[np.ndarray], np.ndarray],
    spend_levels: list[float] | None,
    ax: Axes | None,
    color: str | None,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, Axes]:
    if color is None:
        color = "C0"
    if ax is None:
        if figsize is None:
            figsize = (10, 4)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x_max = current_spend * 2.5
    n_grid = 200
    x_grid = np.linspace(0, x_max, n_grid)

    y_mean, y_lo, y_hi = np.empty((3, len(x_grid)))
    # Reduce each batch before evaluating the next spend range.
    batch_size = 32
    for start in range(0, len(x_grid), batch_size):
        batch = slice(start, start + batch_size)
        levels = x_grid[batch]
        spend = np.repeat(baseline[None, :, :], len(levels), axis=0)
        spend[:, :, channel_index] = levels[:, None]
        response = steady_responses(spend)[:, :, channel_index]
        y_mean[batch] = response.mean(axis=0)
        y_lo[batch], y_hi[batch] = np.percentile(response, [3.0, 97.0], axis=0)

    ax.fill_between(x_grid, y_lo, y_hi, color=color, alpha=0.15)
    ax.plot(x_grid, y_mean, color=color, linewidth=2, label="Posterior mean")

    ax.axvline(
        current_spend,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Current spend ({current_spend:.2f})",
    )

    if spend_levels:
        for frac in spend_levels:
            x_new = current_spend * (1 + frac)
            ax.axvline(
                x_new,
                color="C1",
                linestyle=":",
                linewidth=1,
                label=f"+{frac * 100:.0f}%",
            )

    ax.set_xlabel("Weekly spend")
    ax.set_ylabel("Response (scaled target units)")
    ax.set_title(f"Saturation Curve \u2014 {channel}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, ax


def plot_adstock_ramp(
    channels: Sequence[str],
    current_spend: dict[str, float],
    l_max: int,
    lift_path: Callable[[int, float, int, dict[int, np.ndarray]], np.ndarray],
    max_weeks: int,
    colors: dict[str, str] | None,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, Axes]:
    if colors is None:
        colors = {ch: f"C{i}" for i, ch in enumerate(channels)}
    if figsize is None:
        figsize = (10, 4)
    fig, ax = plt.subplots(figsize=figsize)
    weeks = np.arange(1, max_weeks + 1)

    baselines: dict[int, np.ndarray] = {}
    for ch_idx, ch in enumerate(channels):
        c = colors[ch]
        x_current = current_spend[ch]
        small_delta = 0.01 * x_current
        if small_delta <= 0:
            continue

        duration = max(max_weeks, 4 * l_max)
        path = lift_path(ch_idx, small_delta, duration, baselines)
        ss_per_week = path[:, -1]
        diff = path[:, :max_weeks]
        cumulative_lift = np.cumsum(diff, axis=1)

        avg_per_week = cumulative_lift / weeks[None, :]
        keep = np.abs(ss_per_week) > 1e-12
        if not np.any(keep):
            continue
        ramp_fracs = np.clip(avg_per_week[keep] / ss_per_week[keep, None], 0, 1)

        mean_ramp = ramp_fracs.mean(axis=0)
        low = np.percentile(ramp_fracs, 5.5, axis=0)
        high = np.percentile(ramp_fracs, 94.5, axis=0)

        ax.plot(weeks, mean_ramp, color=c, label=ch)
        ax.fill_between(weeks, low, high, color=c, alpha=0.15)

    ax.set_xlabel("Experiment Duration (weeks)")
    ax.set_ylabel("Adstock Ramp Fraction")
    ax.set_title("Adstock Ramp-up by Channel")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    return fig, ax
