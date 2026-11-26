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

"""Golden regression tests for ExperimentDesigner.

These tests pin the end-to-end numerical output of the public API against
known values from a deterministic fixture. They exist to catch numerical
regressions during internal refactoring — if the maths didn't change, these
tests must pass regardless of how private methods are reorganised.

Do NOT update the expected values here unless you have intentionally changed
the underlying mathematics (adstock ramp, saturation, assurance formula, or
scoring). If a refactoring breaks these tests, it means the refactoring
changed the numerical output and needs investigation.

Reference values were checked against the finite geometric sum, logistic
response, and the finite-horizon AR(1) covariance matrix using the fixture's
stored posterior draws and scales.
"""

from __future__ import annotations

import pytest

from pymc_marketing.mmm.experiment_design import (
    ExperimentDesigner,
    generate_experiment_fixture,
)

# Tolerances: rel=1e-4 catches real regressions while allowing
# floating-point variation across platforms and numpy versions.
REL_TOL = 1e-4
ABS_TOL_ASSURANCE = 0.005


@pytest.fixture(scope="module")
def three_channel_designer():
    """Deterministic 3-channel designer from a known fixture."""
    idata = generate_experiment_fixture(
        channels=["tv", "search", "social"],
        true_params={
            "tv": {"lam": 0.5, "beta": 3.0, "alpha": 0.7},
            "search": {"lam": 2.0, "beta": 1.5, "alpha": 0.3},
            "social": {"lam": 1.0, "beta": 0.8, "alpha": 0.5},
        },
        fit_model=False,
        seed=42,
    )
    return ExperimentDesigner.from_idata(idata)


@pytest.fixture(scope="module")
def boundary_designer():
    """Designer with one strong and one near-zero channel."""
    idata = generate_experiment_fixture(
        channels=["strong", "weak"],
        true_params={
            "strong": {"lam": 1.0, "beta": 5.0, "alpha": 0.3},
            "weak": {"lam": 1.0, "beta": 0.01, "alpha": 0.3},
        },
        fit_model=False,
        seed=42,
    )
    return ExperimentDesigner.from_idata(idata)


class TestGoldenRecommendRanking:
    """Pin the ranking and top-line metrics from recommend().

    Verifies that for a known fixture, the ranking order, channel
    assignments, and numerical scores are stable.
    """

    def test_total_recommendation_count(self, three_channel_designer):
        recs = three_channel_designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4, 8],
            min_snr=0.0,
        )
        assert len(recs) == 12

    def test_top_recommendation_is_search_go_dark_4w(self, three_channel_designer):
        recs = three_channel_designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4, 8],
            min_snr=0.0,
        )
        top = recs[0]
        assert top.channel == "search"
        assert top.spend_change_frac == -1.0
        assert top.duration_weeks == 4

    def test_top_recommendation_numerical_values(self, three_channel_designer):
        recs = three_channel_designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4, 8],
            min_snr=0.0,
        )
        top = recs[0]
        assert top.expected_lift == pytest.approx(-0.280297346, rel=REL_TOL)
        assert top.assurance == pytest.approx(0.960018048, abs=ABS_TOL_ASSURANCE)
        assert top.snr == pytest.approx(-4.090878004, rel=REL_TOL)
        assert top.score == pytest.approx(0.979170462, rel=REL_TOL)

    def test_ranking_order_top_5(self, three_channel_designer):
        """Verify the channel+config ordering of the top 5 results."""
        recs = three_channel_designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4, 8],
            min_snr=0.0,
        )
        expected_order = [
            ("search", -1.0, 4),
            ("tv", -1.0, 4),
            ("search", -1.0, 8),
            ("tv", -1.0, 8),
            ("social", -1.0, 8),
        ]
        actual_order = [
            (r.channel, r.spend_change_frac, r.duration_weeks) for r in recs[:5]
        ]
        assert actual_order == expected_order

    def test_scores_monotonically_decreasing(self, three_channel_designer):
        recs = three_channel_designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4, 8],
            min_snr=0.0,
        )
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)


class TestGoldenLiftPrediction:
    """Pin per-channel lift values for a single spend change and duration.

    This catches regressions in the adstock ramp, saturation evaluation,
    and cumulative lift summation.
    """

    @pytest.mark.parametrize(
        "channel, expected_lift, expected_assurance, expected_snr, expected_ramp",
        [
            ("search", 0.003089422, 0.050297495, 0.036565530, 0.963485474),
            ("tv", 0.094805949, 0.206146158, 1.122096677, 0.723001834),
            ("social", 0.017118089, 0.055193096, 0.202604909, 0.874426384),
        ],
    )
    def test_channel_lift_values(
        self,
        three_channel_designer,
        channel,
        expected_lift,
        expected_assurance,
        expected_snr,
        expected_ramp,
    ):
        recs = three_channel_designer.recommend(
            spend_changes=[0.3],
            durations=[6],
            min_snr=0.0,
        )
        rec = next(r for r in recs if r.channel == channel)

        assert rec.expected_lift == pytest.approx(expected_lift, rel=REL_TOL)
        assert rec.assurance == pytest.approx(expected_assurance, abs=ABS_TOL_ASSURANCE)
        assert rec.snr == pytest.approx(expected_snr, rel=REL_TOL)
        assert rec.adstock_ramp_fraction == pytest.approx(expected_ramp, rel=REL_TOL)

    def test_lift_hdi_contains_mean(self, three_channel_designer):
        """The 94% HDI should bracket the expected lift for all channels."""
        recs = three_channel_designer.recommend(
            spend_changes=[0.3],
            durations=[6],
            min_snr=0.0,
        )
        for rec in recs:
            lo, hi = rec.expected_lift_hdi
            assert lo < rec.expected_lift < hi


class TestGoldenAssuranceBoundaries:
    """Pin assurance behaviour at the extremes.

    A channel with beta=5.0 should have assurance ~1.0 for go-dark;
    a channel with beta=0.01 should have assurance ~alpha (0.05).
    These boundaries validate the Bayesian assurance formula end-to-end.
    """

    def test_strong_channel_detected(self, boundary_designer):
        recs = boundary_designer.recommend(
            spend_changes=[-1.0],
            durations=[8],
            min_snr=0.0,
        )
        strong = next(r for r in recs if r.channel == "strong")
        assert strong.assurance == pytest.approx(1.0, abs=0.005)

    def test_weak_channel_near_alpha(self, boundary_designer):
        recs = boundary_designer.recommend(
            spend_changes=[-1.0],
            durations=[8],
            min_snr=0.0,
        )
        weak = next(r for r in recs if r.channel == "weak")
        assert weak.assurance == pytest.approx(0.05, abs=0.02)

    def test_weak_channel_is_null_candidate(self, boundary_designer):
        recs = boundary_designer.recommend(
            spend_changes=[-1.0],
            durations=[8],
            min_snr=0.0,
        )
        assert "weak" in recs.null_confirmation_candidates

    def test_strong_channel_not_null_candidate(self, boundary_designer):
        recs = boundary_designer.recommend(
            spend_changes=[-1.0],
            durations=[8],
            min_snr=0.0,
        )
        assert "strong" not in recs.null_confirmation_candidates

    def test_assurance_gap(self, boundary_designer):
        """The assurance gap between strong and weak should be > 0.9."""
        recs = boundary_designer.recommend(
            spend_changes=[-1.0],
            durations=[8],
            min_snr=0.0,
        )
        strong = next(r for r in recs if r.channel == "strong")
        weak = next(r for r in recs if r.channel == "weak")
        assert strong.assurance - weak.assurance > 0.9
