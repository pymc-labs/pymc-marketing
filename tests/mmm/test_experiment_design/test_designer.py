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

"""Tests for ExperimentDesigner core functionality."""

from __future__ import annotations

import numpy as np
import pytest

from pymc_marketing.mmm.experiment_design import (
    ExperimentDesigner,
    ExperimentRecommendation,
    generate_experiment_fixture,
)


@pytest.fixture(scope="module")
def designer():
    """Create an ExperimentDesigner from a synthetic fixture."""
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


class TestExperimentDesignerInit:
    """Tests for ExperimentDesigner initialisation from idata."""

    def test_channel_columns(self, designer):
        assert designer.channel_columns == ["tv", "search", "social"]

    def test_n_draws(self, designer):
        assert designer.n_draws == 4000  # 2 chains * 2000 draws

    def test_l_max(self, designer):
        assert designer.l_max == 8

    def test_normalize(self, designer):
        assert designer.normalize is True

    def test_current_spend_positive(self, designer):
        for ch in designer.channel_columns:
            assert designer._current_spend[ch] > 0

    def test_residual_std_positive(self, designer):
        assert designer._residual_std > 0


class TestAdstockRamp:
    """Tests for adstock ramp computation."""

    def test_ramp_fraction_between_0_and_1(self, designer):
        for ch in designer.channel_columns:
            alpha = designer._posterior_samples[ch]["alpha"]
            frac = designer._compute_ramp_fraction(alpha, T_active=8)
            assert 0 < frac < 1

    def test_ramp_fraction_increases_with_duration(self, designer):
        alpha = designer._posterior_samples["tv"]["alpha"]
        frac_4 = designer._compute_ramp_fraction(alpha, T_active=4)
        frac_12 = designer._compute_ramp_fraction(alpha, T_active=12)
        assert frac_12 > frac_4

    def test_fast_channel_ramps_faster(self, designer):
        alpha_search = designer._posterior_samples["search"]["alpha"]
        alpha_tv = designer._posterior_samples["tv"]["alpha"]
        frac_search = designer._compute_ramp_fraction(alpha_search, T_active=4)
        frac_tv = designer._compute_ramp_fraction(alpha_tv, T_active=4)
        # search has alpha~0.3, tv has alpha~0.7 → search ramps faster
        assert frac_search > frac_tv

    @pytest.mark.parametrize("T_active", [1, 4, 8, 16])
    def test_ramp_shape_correct(self, designer, T_active):
        alpha = designer._posterior_samples["tv"]["alpha"]
        ramp = designer._compute_adstock_ramp(alpha, T_active)
        assert ramp.shape == (designer.n_draws, T_active)

    def test_known_ramp_geometric_series(self):
        """Verify ramp matches analytic geometric series for a fixed alpha."""
        alpha = np.array([0.5] * 100)
        l_max = 8
        T = 4

        idata = generate_experiment_fixture(
            channels=["ch1"],
            true_params={"ch1": {"lam": 1.0, "beta": 1.0, "alpha": 0.5}},
            fit_model=False,
            seed=1,
        )
        d = ExperimentDesigner.from_idata(idata)
        d._posterior_samples["ch1"]["alpha"] = alpha
        d.l_max = l_max
        d.normalize = True

        ramp = d._compute_adstock_ramp(alpha, T)
        S = (1 - 0.5**l_max) / (1 - 0.5)
        for t in range(T):
            expected = ((1 - 0.5 ** (t + 1)) / (1 - 0.5)) / S
            assert ramp[0, t] == pytest.approx(expected, rel=1e-6)


class TestPredictLift:
    """Tests for lift prediction."""

    def test_lift_shape(self, designer):
        lift = designer._predict_lift("tv", delta_x=0.1, T_active=6)
        assert lift.shape == (designer.n_draws,)

    def test_positive_spend_change_positive_lift(self, designer):
        lift = designer._predict_lift("tv", delta_x=0.1, T_active=6)
        assert np.mean(lift) > 0

    def test_negative_spend_change_negative_lift(self, designer):
        x_current = designer._current_spend["tv"]
        lift = designer._predict_lift("tv", delta_x=-x_current, T_active=6)
        assert np.mean(lift) < 0

    def test_longer_duration_larger_lift(self, designer):
        lift_4 = designer._predict_lift("tv", delta_x=0.1, T_active=4)
        lift_12 = designer._predict_lift("tv", delta_x=0.1, T_active=12)
        assert abs(np.mean(lift_12)) > abs(np.mean(lift_4))


class TestAssurance:
    """Tests for posterior-predictive power (Bayesian assurance)."""

    def test_assurance_between_0_and_1(self, designer):
        lift = designer._predict_lift("tv", delta_x=0.1, T_active=6)
        sigma_d = designer._residual_std * np.sqrt(6)
        assurance = designer._compute_assurance(lift, sigma_d)
        assert 0 <= assurance <= 1

    def test_large_effect_high_assurance(self, designer):
        lift = np.full(1000, 100.0)
        assurance = designer._compute_assurance(lift, sigma_d=1.0)
        assert assurance > 0.99

    def test_zero_effect_low_assurance(self, designer):
        lift = np.full(1000, 0.0)
        assurance = designer._compute_assurance(lift, sigma_d=1.0)
        assert assurance < 0.10

    def test_assurance_near_alpha_for_zero_effects(self, designer):
        lift = np.zeros(1000)
        assurance = designer._compute_assurance(
            lift,
            sigma_d=1.0,
            significance_level=0.05,
        )
        assert assurance == pytest.approx(0.05, abs=0.01)


class TestRecommend:
    """Tests for the recommend method."""

    def test_returns_list(self, designer):
        recs = designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4, 6],
        )
        assert isinstance(recs, list)

    def test_all_recommendations_are_dataclass(self, designer):
        recs = designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4],
        )
        for rec in recs:
            assert isinstance(rec, ExperimentRecommendation)

    def test_sorted_by_score_descending(self, designer):
        recs = designer.recommend(
            spend_changes=[0.2, 0.5, -1.0],
            durations=[4, 8],
        )
        if len(recs) > 1:
            scores = [r.score for r in recs]
            assert scores == sorted(scores, reverse=True)

    def test_min_snr_filtering(self, designer):
        recs_low = designer.recommend(
            spend_changes=[0.2],
            durations=[4],
            min_snr=0.0,
        )
        recs_high = designer.recommend(
            spend_changes=[0.2],
            durations=[4],
            min_snr=100.0,
        )
        assert len(recs_high) <= len(recs_low)

    def test_go_dark_has_frac_minus_1(self, designer):
        recs = designer.recommend(
            spend_changes=[-1.0],
            durations=[6],
            min_snr=0.0,
        )
        for rec in recs:
            assert rec.spend_change_frac == -1.0

    def test_net_cost_sign(self, designer):
        recs = designer.recommend(
            spend_changes=[0.2, -0.5],
            durations=[4],
            min_snr=0.0,
        )
        for rec in recs:
            if rec.spend_change_frac > 0:
                assert rec.net_cost > 0
            elif rec.spend_change_frac < 0:
                assert rec.net_cost < 0

    def test_rationale_nonempty(self, designer):
        recs = designer.recommend(
            spend_changes=[0.2],
            durations=[4],
            min_snr=0.0,
        )
        for rec in recs:
            assert len(rec.rationale) > 0

    def test_custom_score_weights(self, designer):
        recs = designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4],
            min_snr=0.0,
            score_weights={
                "assurance": 1.0,
                "uncertainty": 0.0,
                "correlation": 0.0,
                "gradient": 0.0,
                "cost_efficiency": 0.0,
            },
        )
        if len(recs) > 1:
            scores = [r.score for r in recs]
            assert scores == sorted(scores, reverse=True)

    def test_empty_result_when_nothing_passes_filter(self, designer):
        recs = designer.recommend(
            spend_changes=[0.001],
            durations=[1],
            min_snr=1000.0,
        )
        assert recs == []


class TestScoring:
    """Tests for scoring dimensions and normalisation."""

    def test_min_max_normalize_constant(self):
        values = np.array([5.0, 5.0, 5.0])
        result = ExperimentDesigner._min_max_normalize(values)
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

    def test_min_max_normalize_range(self):
        values = np.array([0.0, 5.0, 10.0])
        result = ExperimentDesigner._min_max_normalize(values)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])


class TestPlotting:
    """Smoke tests for plotting methods (verify they don't error)."""

    def test_plot_channel_diagnostics(self, designer):
        fig, _axes = designer.plot_channel_diagnostics()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_power_cost(self, designer):
        recs = designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4],
            min_snr=0.0,
        )
        if recs:
            fig, _ax = designer.plot_power_cost(recs)
            assert fig is not None
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_lift_distributions(self, designer):
        fig, _axes = designer.plot_lift_distributions(
            "tv",
            spend_changes=[0.2, -1.0],
            durations=[4, 6],
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_saturation_curve(self, designer):
        fig, _ax = designer.plot_saturation_curve(
            "tv",
            n_samples=10,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_adstock_ramp(self, designer):
        fig, _ax = designer.plot_adstock_ramp(max_weeks=8)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
