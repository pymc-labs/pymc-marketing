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

"""ExperimentRecommendation dataclass, container, and rationale template."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import pandas


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


class ExperimentRecommendations(Sequence[ExperimentRecommendation]):
    """Ordered collection of experiment recommendations.

    Behaves like an immutable list of :class:`ExperimentRecommendation`
    objects (supports iteration, indexing, slicing, and ``len``).  In
    Jupyter notebooks the collection renders automatically as an HTML
    table via ``_repr_html_``.

    Parameters
    ----------
    recommendations : list[ExperimentRecommendation]
        Pre-sorted list of recommendations (highest score first).
    """

    def __init__(self, recommendations: list[ExperimentRecommendation]) -> None:
        self._recommendations = list(recommendations)

    # -- Sequence protocol ---------------------------------------------------

    @overload
    def __getitem__(self, index: int) -> ExperimentRecommendation: ...

    @overload
    def __getitem__(self, index: slice) -> ExperimentRecommendations: ...

    def __getitem__(
        self, index: int | slice
    ) -> ExperimentRecommendation | ExperimentRecommendations:
        """Return a recommendation by index, or a sub-collection by slice."""
        if isinstance(index, slice):
            return ExperimentRecommendations(self._recommendations[index])
        return self._recommendations[index]

    def __len__(self) -> int:
        """Return the number of recommendations."""
        return len(self._recommendations)

    def __iter__(self) -> Iterator[ExperimentRecommendation]:
        """Iterate over the recommendations."""
        return iter(self._recommendations)

    def __eq__(self, other: object) -> bool:
        """Check equality against another container or plain list."""
        if isinstance(other, ExperimentRecommendations):
            return self._recommendations == other._recommendations
        if isinstance(other, list):
            return self._recommendations == other
        return NotImplemented

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"ExperimentRecommendations({len(self)} recommendations)"

    # -- Jupyter rendering ---------------------------------------------------

    def _repr_html_(self) -> str:
        """Render as an HTML table in Jupyter notebooks."""
        if not self._recommendations:
            return "<p><em>No recommendations (all candidates filtered out).</em></p>"

        header = (
            "<tr>"
            "<th>Rank</th>"
            "<th>Channel</th>"
            "<th>&Delta; Spend</th>"
            "<th>Duration</th>"
            "<th>E[Lift]</th>"
            "<th>Lift 94% HDI</th>"
            "<th>SNR</th>"
            "<th>Assurance</th>"
            "<th>Ramp</th>"
            "<th>Score</th>"
            "</tr>"
        )

        rows: list[str] = []
        for i, rec in enumerate(self._recommendations):
            spend_str = (
                "go-dark"
                if rec.spend_change_frac == -1.0
                else f"{rec.spend_change_frac:+.0%}/wk"
            )
            rows.append(
                f"<tr>"
                f"<td>{i + 1}</td>"
                f"<td>{escape(rec.channel)}</td>"
                f"<td>{spend_str}</td>"
                f"<td>{rec.duration_weeks}w</td>"
                f"<td>{rec.expected_lift:.1f}</td>"
                f"<td>[{rec.expected_lift_hdi[0]:.0f}, "
                f"{rec.expected_lift_hdi[1]:.0f}]</td>"
                f"<td>{rec.snr:.1f}</td>"
                f"<td>{rec.assurance:.2f}</td>"
                f"<td>{rec.adstock_ramp_fraction:.2f}</td>"
                f"<td>{rec.score:.3f}</td>"
                f"</tr>"
            )

        return f"<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"

    def to_dataframe(self) -> pandas.DataFrame:
        """Convert to a :class:`pandas.DataFrame`.

        Returns
        -------
        pd.DataFrame
            One row per recommendation with formatted columns.
        """
        import pandas as pd

        records = [
            {
                "Rank": i + 1,
                "Channel": rec.channel,
                "Spend Change": (
                    "go-dark"
                    if rec.spend_change_frac == -1.0
                    else f"{rec.spend_change_frac:+.0%}/wk"
                ),
                "Duration": f"{rec.duration_weeks}w",
                "E[Lift]": rec.expected_lift,
                "Lift HDI Low": rec.expected_lift_hdi[0],
                "Lift HDI High": rec.expected_lift_hdi[1],
                "SNR": rec.snr,
                "Assurance": rec.assurance,
                "Ramp": rec.adstock_ramp_fraction,
                "Net Cost": rec.net_cost,
                "Score": rec.score,
            }
            for i, rec in enumerate(self._recommendations)
        ]
        return pd.DataFrame(records)


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
