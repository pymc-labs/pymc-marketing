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
    ExperimentRecommendations,
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

    def test_returns_recommendations_container(self, designer):
        recs = designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4, 6],
        )
        assert isinstance(recs, ExperimentRecommendations)

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
        assert len(recs) == 0


class TestExperimentRecommendations:
    """Tests for the ExperimentRecommendations container."""

    def test_len(self, designer):
        recs = designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0.0)
        assert len(recs) > 0

    def test_indexing(self, designer):
        recs = designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0.0)
        assert isinstance(recs[0], ExperimentRecommendation)

    def test_slicing_returns_container(self, designer):
        recs = designer.recommend(spend_changes=[0.2, -1.0], durations=[4], min_snr=0.0)
        sliced = recs[:2]
        assert isinstance(sliced, ExperimentRecommendations)
        assert len(sliced) <= 2

    def test_iteration(self, designer):
        recs = designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0.0)
        items = list(recs)
        assert all(isinstance(r, ExperimentRecommendation) for r in items)

    def test_repr_html(self, designer):
        recs = designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0.0)
        html = recs._repr_html_()
        assert "<table>" in html
        assert "Channel" in html

    def test_repr_html_empty(self):
        recs = ExperimentRecommendations([])
        html = recs._repr_html_()
        assert "No recommendations" in html

    def test_to_dataframe(self, designer):
        recs = designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0.0)
        df = recs.to_dataframe()
        assert len(df) == len(recs)
        assert "Channel" in df.columns
        assert "Score" in df.columns

    def test_repr(self, designer):
        recs = designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0.0)
        assert "ExperimentRecommendations" in repr(recs)


class TestAutocorrelationCorrection:
    """Tests for AR(1) autocorrelation correction on sigma."""

    def test_effective_sigma_no_autocorr(self, designer):
        """With zero autocorrelation, effective sigma equals IID formula."""
        designer._residual_autocorr = 0.0
        sigma_iid = designer._residual_std * np.sqrt(8)
        assert designer._effective_sigma(8) == pytest.approx(sigma_iid)

    def test_effective_sigma_positive_autocorr_inflates(self, designer):
        """Positive autocorrelation inflates sigma above IID baseline."""
        designer._residual_autocorr = 0.0
        sigma_iid = designer._effective_sigma(8)
        designer._residual_autocorr = 0.4
        sigma_corr = designer._effective_sigma(8)
        assert sigma_corr > sigma_iid

    def test_effective_sigma_negative_autocorr_clamped(self, designer):
        """Negative autocorrelation is clamped to zero (no deflation)."""
        designer._residual_autocorr = -0.3
        sigma = designer._effective_sigma(8)
        designer._residual_autocorr = 0.0
        sigma_iid = designer._effective_sigma(8)
        assert sigma == pytest.approx(sigma_iid)

    def test_effective_sigma_known_correction(self, designer):
        """Verify the AR(1) correction factor for a known rho."""
        designer._residual_autocorr = 0.5
        sigma = designer._effective_sigma(8)
        expected = designer._residual_std * np.sqrt(8 * (1.5 / 0.5))
        assert sigma == pytest.approx(expected)

    def test_residual_autocorr_loaded_from_fixture(self, designer):
        """Fixture-based designer should have autocorrelation set."""
        assert hasattr(designer, "_residual_autocorr")
        assert isinstance(designer._residual_autocorr, float)


class TestNullConfirmation:
    """Tests for null-confirmation candidate detection."""

    def test_null_candidates_is_list(self, designer):
        recs = designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4],
            min_snr=0.0,
        )
        assert isinstance(recs.null_confirmation_candidates, list)

    def test_null_candidates_with_near_zero_beta(self):
        """A channel with beta near zero should be a null candidate."""
        idata = generate_experiment_fixture(
            channels=["strong", "null_channel"],
            true_params={
                "strong": {"lam": 1.0, "beta": 3.0, "alpha": 0.5},
                "null_channel": {"lam": 1.0, "beta": 0.001, "alpha": 0.5},
            },
            fit_model=False,
            seed=42,
        )
        d = ExperimentDesigner.from_idata(idata)
        recs = d.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4, 8],
            min_snr=0.0,
        )
        assert "null_channel" in recs.null_confirmation_candidates

    def test_null_candidates_preserved_on_slice(self, designer):
        recs = designer.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4],
            min_snr=0.0,
        )
        sliced = recs[:2]
        assert sliced.null_confirmation_candidates == recs.null_confirmation_candidates

    def test_null_candidates_in_repr_html(self):
        """Null candidates should appear in HTML output (empty recs)."""
        recs = ExperimentRecommendations(
            [], null_confirmation_candidates=["dead_channel"]
        )
        html = recs._repr_html_()
        assert "dead_channel" in html
        assert "near-zero effect" in html

    def test_null_candidates_in_repr_html_with_recs(self):
        """Null candidates appear in HTML when recommendations exist too."""
        rec = ExperimentRecommendation(
            channel="tv",
            spend_change_frac=0.2,
            spend_change_abs=0.1,
            duration_weeks=4,
            expected_lift=1.0,
            expected_lift_hdi=(0.5, 1.5),
            snr=3.0,
            assurance=0.8,
            adstock_ramp_fraction=0.9,
            net_cost=0.4,
            score=0.7,
            rationale="test",
        )
        recs = ExperimentRecommendations(
            [rec], null_confirmation_candidates=["weak_ch"]
        )
        html = recs._repr_html_()
        assert "<table>" in html
        assert "weak_ch" in html
        assert "near-zero effect" in html

    def test_null_candidates_in_repr(self):
        recs = ExperimentRecommendations(
            [], null_confirmation_candidates=["ch1", "ch2"]
        )
        assert "null-confirmation" in repr(recs)


class TestCorrelationWarning:
    """Tests for identification warnings on high-correlation channels."""

    def test_high_correlation_triggers_warning(self):
        """Rationale includes caution when correlation exceeds threshold."""
        from pymc_marketing.mmm.experiment_design.recommendation import (
            _format_rationale,
        )

        rec = ExperimentRecommendation(
            channel="tv",
            spend_change_frac=0.2,
            spend_change_abs=0.1,
            duration_weeks=8,
            expected_lift=100.0,
            expected_lift_hdi=(50.0, 150.0),
            snr=3.0,
            assurance=0.85,
            adstock_ramp_fraction=0.9,
            net_cost=0.8,
            score=0.7,
            rationale="",
        )
        rationale = _format_rationale(
            rec,
            correlation_info="high spend correlation with search (r = 0.85)",
            max_correlation=0.85,
        )
        assert "Caution" in rationale
        assert "identification" in rationale

    def test_low_correlation_no_warning(self):
        """Rationale does NOT include caution when correlation is low."""
        from pymc_marketing.mmm.experiment_design.recommendation import (
            _format_rationale,
        )

        rec = ExperimentRecommendation(
            channel="tv",
            spend_change_frac=0.2,
            spend_change_abs=0.1,
            duration_weeks=8,
            expected_lift=100.0,
            expected_lift_hdi=(50.0, 150.0),
            snr=3.0,
            assurance=0.85,
            adstock_ramp_fraction=0.9,
            net_cost=0.8,
            score=0.7,
            rationale="",
        )
        rationale = _format_rationale(
            rec,
            correlation_info="high spend correlation with search (r = 0.40)",
            max_correlation=0.40,
        )
        assert "Caution" not in rationale


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

    def test_weight_redistribution_without_correlation(self):
        """When spend_correlation is None, correlation weight is redistributed."""
        idata = generate_experiment_fixture(
            channels=["ch1", "ch2"],
            true_params={
                "ch1": {"lam": 1.0, "beta": 1.0, "alpha": 0.5},
                "ch2": {"lam": 0.5, "beta": 2.0, "alpha": 0.3},
            },
            fit_model=False,
            seed=99,
        )
        d = ExperimentDesigner.from_idata(idata)
        d._spend_correlation = None

        recs = d.recommend(
            spend_changes=[0.2, -1.0],
            durations=[4],
            min_snr=0.0,
            score_weights={
                "assurance": 0.4,
                "cost_efficiency": 0.3,
                "correlation": 0.3,
            },
        )
        for rec in recs:
            assert rec.score >= 0.0

    def test_uncertainty_rank_most_uncertain_first(self, designer):
        """The channel with widest posterior should be rank #1."""
        ranks = designer._get_uncertainty_ranks()
        assert set(ranks.values()) == {1, 2, 3}
        assert min(ranks.values()) == 1


class TestFromIdataEdgeCases:
    """Tests for from_idata edge cases and alternate branches."""

    def test_from_idata_without_residual_autocorr(self):
        """from_idata sets autocorrelation to 0.0 when field is absent."""
        import xarray as xr

        idata = generate_experiment_fixture(
            channels=["ch1"],
            true_params={"ch1": {"lam": 1.0, "beta": 1.0, "alpha": 0.5}},
            fit_model=False,
            seed=1,
        )
        cd = {
            k: v
            for k, v in idata.constant_data.data_vars.items()
            if k != "residual_autocorr"
        }
        del idata.constant_data
        idata.add_groups({"constant_data": xr.Dataset(cd)})
        d = ExperimentDesigner.from_idata(idata)
        assert d._residual_autocorr == 0.0

    def test_from_idata_without_spend_correlation(self):
        """from_idata sets correlation to None when field is absent."""
        import xarray as xr

        idata = generate_experiment_fixture(
            channels=["ch1"],
            true_params={"ch1": {"lam": 1.0, "beta": 1.0, "alpha": 0.5}},
            fit_model=False,
            seed=1,
        )
        cd = {
            k: v
            for k, v in idata.constant_data.data_vars.items()
            if k != "spend_correlation"
        }
        del idata.constant_data
        idata.add_groups({"constant_data": xr.Dataset(cd)})
        d = ExperimentDesigner.from_idata(idata)
        assert d._spend_correlation is None


class TestNormalizeFlag:
    """Tests for normalize=False branch."""

    def test_steady_state_unnormalized(self):
        idata = generate_experiment_fixture(
            channels=["ch1"],
            true_params={"ch1": {"lam": 1.0, "beta": 1.0, "alpha": 0.5}},
            fit_model=False,
            seed=1,
        )
        d = ExperimentDesigner.from_idata(idata)
        d.normalize = False
        alpha = d._posterior_samples["ch1"]["alpha"]
        result = d._compute_steady_state_spend(1.0, alpha)
        assert result.shape == alpha.shape
        assert np.all(result > 1.0)

    def test_adstock_ramp_unnormalized(self):
        idata = generate_experiment_fixture(
            channels=["ch1"],
            true_params={"ch1": {"lam": 1.0, "beta": 1.0, "alpha": 0.5}},
            fit_model=False,
            seed=1,
        )
        d = ExperimentDesigner.from_idata(idata)
        d.normalize = False
        alpha = d._posterior_samples["ch1"]["alpha"]
        ramp = d._compute_adstock_ramp(alpha, T_active=4)
        assert ramp.shape == (d.n_draws, 4)


class TestRecommendDefaults:
    """Test recommend() with default arguments."""

    def test_recommend_default_args(self, designer):
        recs = designer.recommend(min_snr=0.0)
        assert len(recs) > 0

    def test_recommend_includes_decrease_direction(self, designer):
        recs = designer.recommend(
            spend_changes=[-0.3],
            durations=[4],
            min_snr=0.0,
        )
        for rec in recs:
            assert rec.spend_change_frac < 0
            assert rec.spend_change_frac != -1.0


class TestSingleChannel:
    """Test behaviour with a single channel (no correlation possible)."""

    def test_single_channel_correlation_info_none(self):
        idata = generate_experiment_fixture(
            channels=["solo"],
            true_params={"solo": {"lam": 1.0, "beta": 2.0, "alpha": 0.5}},
            fit_model=False,
            seed=1,
        )
        d = ExperimentDesigner.from_idata(idata)
        result = d._get_correlation_info("solo")
        assert result is None


class TestRecommendationsEquality:
    """Tests for ExperimentRecommendations.__eq__."""

    def test_eq_with_list(self, designer):
        recs = designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0.0)
        rec_list = list(recs)
        assert recs == rec_list

    def test_eq_with_unknown_type(self, designer):
        recs = designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0.0)
        assert recs.__eq__("not a list") is NotImplemented

    def test_eq_with_container(self, designer):
        recs = designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0.0)
        recs2 = ExperimentRecommendations(list(recs))
        assert recs == recs2


class TestFixtureDefaults:
    """Tests for generate_experiment_fixture default parameters."""

    def test_default_channels_and_params(self):
        idata = generate_experiment_fixture(fit_model=False, seed=42)
        d = ExperimentDesigner.from_idata(idata)
        assert d.channel_columns == ["tv", "search", "social"]


class TestPlotting:
    """Smoke tests for plotting methods (verify they don't error)."""

    def test_plot_channel_diagnostics(self, designer):
        fig, _axes = designer.plot_channel_diagnostics()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_channel_diagnostics_no_correlation(self):
        """Diagnostics with spend_correlation=None."""
        import matplotlib.pyplot as plt

        idata = generate_experiment_fixture(
            channels=["a", "b"],
            true_params={
                "a": {"lam": 1.0, "beta": 1.0, "alpha": 0.5},
                "b": {"lam": 0.5, "beta": 2.0, "alpha": 0.3},
            },
            fit_model=False,
            seed=1,
        )
        d = ExperimentDesigner.from_idata(idata)
        d._spend_correlation = None
        fig, _axes = d.plot_channel_diagnostics()
        assert fig is not None
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

    def test_plot_power_cost_decrease_direction(self, designer):
        """Power-cost plot includes decrease markers."""
        import matplotlib.pyplot as plt

        recs = designer.recommend(
            spend_changes=[0.2, -0.3, -1.0],
            durations=[4],
            min_snr=0.0,
        )
        if recs:
            fig, _ax = designer.plot_power_cost(recs)
            assert fig is not None
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

    def test_plot_lift_distributions_defaults(self, designer):
        """Lift distributions with default spend_changes and durations."""
        import matplotlib.pyplot as plt

        fig, _axes = designer.plot_lift_distributions("tv")
        assert fig is not None
        plt.close(fig)

    def test_plot_lift_distributions_single_row(self, designer):
        """Lift distributions with a single spend change (triggers row newaxis)."""
        import matplotlib.pyplot as plt

        fig, _axes = designer.plot_lift_distributions(
            "tv",
            spend_changes=[0.2],
            durations=[4, 6],
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_lift_distributions_single_col(self, designer):
        """Lift distributions with a single duration (triggers col newaxis)."""
        import matplotlib.pyplot as plt

        fig, _axes = designer.plot_lift_distributions(
            "tv",
            spend_changes=[0.2, -1.0],
            durations=[4],
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_saturation_curve(self, designer):
        fig, _ax = designer.plot_saturation_curve(
            "tv",
            n_samples=10,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_saturation_curve_existing_ax(self, designer):
        """Saturation curve on a pre-existing axes."""
        import matplotlib.pyplot as plt

        _fig0, ax0 = plt.subplots()
        fig, ax = designer.plot_saturation_curve("tv", ax=ax0, n_samples=10)
        assert fig is not None
        assert ax is ax0
        plt.close(fig)

    def test_plot_saturation_curve_with_spend_levels(self, designer):
        """Saturation curve with spend-level markers."""
        import matplotlib.pyplot as plt

        fig, _ax = designer.plot_saturation_curve(
            "tv", n_samples=10, spend_levels=[0.2, 0.5]
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_adstock_ramp(self, designer):
        fig, _ax = designer.plot_adstock_ramp(max_weeks=8)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
