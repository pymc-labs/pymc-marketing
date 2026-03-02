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

"""ExperimentRecommendation dataclass and rationale template."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentRecommendation:
    """A recommended experiment design for a single channel.

    Each recommendation represents a candidate experiment: a specific channel,
    spend change, and duration, with posterior-derived metrics for the
    expected lift, power (assurance), and cost.

    Parameters
    ----------
    channel : str
        The marketing channel to test.
    spend_change_frac : float
        Fractional per-week spend change (e.g. 0.2 for +20%, -1.0 for go-dark).
    spend_change_abs : float
        Absolute weekly spend change in model-scale units.
    duration_weeks : int
        Active intervention period in weeks.
    expected_lift : float
        Posterior mean of total cumulative lift over the experiment.
    expected_lift_hdi : tuple[float, float]
        94% HDI of the total cumulative lift distribution.
    snr : float
        Signal-to-noise ratio (expected lift / measurement noise).
    assurance : float
        Posterior-predictive power (Bayesian assurance).
    adstock_ramp_fraction : float
        Mean fraction of steady-state effect captured over the experiment
        duration (posterior mean across draws).
    net_cost : float
        Direct additional spend: spend_change_abs * duration_weeks.
    score : float
        Weighted composite score used for ranking.
    rationale : str
        Auto-generated template-based explanation of the recommendation.
    """

    channel: str
    spend_change_frac: float
    spend_change_abs: float
    duration_weeks: int
    expected_lift: float
    expected_lift_hdi: tuple[float, float]
    snr: float
    assurance: float
    adstock_ramp_fraction: float
    net_cost: float
    score: float
    rationale: str


def _format_rationale(
    rec: ExperimentRecommendation,
    uncertainty_rank: int | None = None,
    correlation_info: str | None = None,
) -> str:
    """Generate a template-based rationale string for a recommendation.

    Parameters
    ----------
    rec : ExperimentRecommendation
        The recommendation to generate a rationale for.
    uncertainty_rank : int | None
        Rank of this channel by posterior uncertainty (1 = most uncertain).
    correlation_info : str | None
        Description of spend correlation (e.g. "high correlation with
        Digital (r = 0.91)"). None if unavailable.

    Returns
    -------
    str
        A three-sentence rationale.
    """
    direction = (
        "go-dark"
        if rec.spend_change_frac == -1.0
        else (
            f"+{rec.spend_change_frac * 100:.0f}%/wk"
            if rec.spend_change_frac > 0
            else f"{rec.spend_change_frac * 100:.0f}%/wk"
        )
    )

    uncertainty_str = (
        f" (uncertainty rank #{uncertainty_rank})" if uncertainty_rank else ""
    )
    correlation_str = f" and {correlation_info}" if correlation_info else ""

    sentence1 = (
        f"{rec.channel} is a high-priority test target{uncertainty_str}"
        f"{correlation_str}."
    )

    sentence2 = (
        f"A {direction} change for {rec.duration_weeks} weeks produces "
        f"an expected total lift of {rec.expected_lift:.0f} "
        f"(94% HDI: [{rec.expected_lift_hdi[0]:.0f}, "
        f"{rec.expected_lift_hdi[1]:.0f}]) "
        f"with assurance {rec.assurance:.2f}."
    )

    sentence3 = (
        f"Adstock ramp fraction {rec.adstock_ramp_fraction:.2f} — "
        f"net cost: {rec.net_cost:.0f} (model-scale units)."
    )

    return f"{sentence1} {sentence2} {sentence3}"
