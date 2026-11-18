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

    def test_has_eval_fn(self, designer):
        assert callable(designer._eval_fn)


class TestInitFromMMM:
    """Tests for ExperimentDesigner.__init__(mmm) constructor path."""

    @pytest.fixture(scope="class")
    @classmethod
    def fitted_mmm(cls):
        import pandas as pd

        from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

        rng = np.random.default_rng(8)
        n_dates = 35
        spend = rng.uniform(1, 8, n_dates)
        x = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=n_dates, freq="W"),
                "tv": spend,
            }
        )
        response = 10_000 + 4_500 * np.tanh(spend / 5) + rng.normal(0, 50, n_dates)
        mmm = MMM(
            date_column="date",
            channel_columns=["tv"],
            target_column="response",
            adstock=GeometricAdstock(l_max=1),
            saturation=LogisticSaturation(),
        )
        mmm.fit(x, response, random_seed=8)
        return mmm

    def test_fitted_mmm_uses_same_units_without_mutating_idata(self, fitted_mmm):
        import xarray as xr

        idata = fitted_mmm.idata
        before = idata.copy(deep=True)
        designer = ExperimentDesigner(fitted_mmm, spend_lookback=5)
        recommendations = designer.recommend(
            spend_changes=[0.2], durations=[4], min_snr=0.5
        )
        xr.testing.assert_identical(idata, before)
        assert designer.channel_columns == ["tv"]
        raw_spend = idata.constant_data.channel_data.isel(date=slice(-5, None)).values
        scaled_predictions = idata.posterior.mu.mean(("chain", "draw")).values
        target_scale = idata.constant_data.target_scale.item()
        residual = np.asarray(fitted_mmm.y) / target_scale - scaled_predictions
        assert designer._residual_std == pytest.approx(residual.std())
        (recommendation,) = recommendations
        assert recommendation.spend_change_abs == pytest.approx(0.2 * raw_spend.mean())
        assert recommendation.net_cost == pytest.approx(
            4 * recommendation.spend_change_abs
        )
        assert 0.5 < recommendation.snr < 100

    def test_fixed_media_parameters_keep_the_draw_axis(self, simple_mmm_data):
        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import MMM, NoAdstock, NoSaturation
        from tests.mmm.conftest import mock_fit

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2", "channel_3"],
            target_column="target",
            adstock=NoAdstock(l_max=1),
            saturation=NoSaturation(priors={"beta": 0.4}),
        )
        mock_fit(mmm, **simple_mmm_data, draws=10, random_seed=42)
        designer = ExperimentDesigner(mmm)

        recommendations = designer.recommend(
            spend_changes=[0.2], durations=[4], min_snr=0
        )
        assert len(recommendations) == 3
        for rec in recommendations:
            assert rec.expected_lift_hdi == pytest.approx([rec.expected_lift] * 2)

        fig, ax = designer.plot_saturation_curve("channel_1")
        spend, response = ax.lines[0].get_data()
        assert response[-1] == pytest.approx(
            0.4 * spend[-1] / mmm.idata.constant_data.channel_scale[0].item()
        )
        plt.close(fig)

    def test_unfitted_mmm_raises(self):
        from types import SimpleNamespace

        with pytest.raises(ValueError, match="fitted"):
            ExperimentDesigner(SimpleNamespace(idata=None))

    def test_unsupported_mmm_raises(self, fitted_mmm, monkeypatch):
        monkeypatch.setattr(fitted_mmm, "adstock_first", False)
        with pytest.raises(NotImplementedError, match="adstock_first"):
            ExperimentDesigner(fitted_mmm)

    def test_time_varying_media_rejected_at_construction(self, fitted_mmm, monkeypatch):
        monkeypatch.setattr(fitted_mmm, "time_varying_media", True)
        with pytest.raises(NotImplementedError, match="time-invariant media"):
            ExperimentDesigner(fitted_mmm)

    def test_additive_spend_path_rejected_at_construction(self, simple_mmm_data):
        from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
        from tests.mmm.conftest import mock_fit
        from tests.mmm.test_budget_optimizer_mmm import _MediatedEffect

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            target_column="target",
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.add_mu_effect(_MediatedEffect())
        mock_fit(mmm, **simple_mmm_data, draws=10, random_seed=42)

        with pytest.raises(NotImplementedError, match="additional mu_effects"):
            ExperimentDesigner(mmm)

    def test_likelihood_with_mean_offset_rejected(self, truncated_normal_fitted_mmm):
        with pytest.raises(
            ValueError, match=r"conditional mean equal to mu.*TruncatedNormal"
        ):
            ExperimentDesigner(truncated_normal_fitted_mmm)

    def test_likelihood_with_positive_mean_domain_rejected(self, simple_mmm_data):
        from pymc_extras.prior import Prior

        from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
        from tests.mmm.conftest import mock_fit

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1"],
            target_column="target",
            adstock=GeometricAdstock(l_max=1, priors={"alpha": 0.5}),
            saturation=LogisticSaturation(priors={"lam": 2, "beta": 1.9}),
            model_config={
                "intercept": Prior("Normal", mu=-0.45, sigma=0.01),
                "likelihood": Prior("Gamma", sigma=0.1, dims="date"),
            },
        )
        simple_mmm_data["X"]["channel_1"] = np.linspace(
            60, 100, len(simple_mmm_data["X"])
        )
        mock_fit(mmm, **simple_mmm_data, draws=10, random_seed=42)
        assert bool((mmm.idata.posterior.mu > 0).all())
        assert bool((mmm.idata.posterior.intercept_contribution < 0).all())

        with pytest.raises(ValueError, match=r"conditional mean equal to mu.*Gamma"):
            ExperimentDesigner(mmm)

    @pytest.mark.parametrize("mode", ["Before", "Overlap"])
    def test_nontrailing_adstock_rejected(self, fitted_mmm, monkeypatch, mode):
        from pymc_marketing.mmm.transformers import ConvMode

        monkeypatch.setattr(fitted_mmm.adstock, "mode", ConvMode(mode))
        with pytest.raises(NotImplementedError, match="trailing adstock"):
            ExperimentDesigner(fitted_mmm)

    def test_daily_mmm_is_not_treated_as_weekly(self, simple_mmm_data):
        import pandas as pd

        from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
        from tests.mmm.conftest import mock_fit

        x = simple_mmm_data["X"]
        x["date"] = pd.date_range("2025-01-01", periods=len(x), freq="D")
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2", "channel_3"],
            target_column="target",
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mock_fit(mmm, **simple_mmm_data, draws=10, random_seed=42)
        with pytest.raises(ValueError, match="regular weekly observations"):
            ExperimentDesigner(mmm)

    def test_mismatched_observations_raise(self, fitted_mmm, monkeypatch):
        idata = fitted_mmm.idata.copy()
        idata["observed_data"] = idata.observed_data.dataset.isel(date=slice(1, None))
        monkeypatch.setattr(fitted_mmm, "idata", idata)
        with pytest.raises(ValueError, match="exact"):
            ExperimentDesigner(fitted_mmm)

    @pytest.mark.parametrize("posterior_change", ["channel_order", "missing_mu"])
    def test_loaded_fit_preserves_channel_lift_and_residuals(
        self, simple_mmm_data, posterior_change
    ):
        import pymc as pm
        import xarray as xr

        from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
        from pymc_marketing.names.mmm import Var
        from tests.mmm.conftest import mock_fit

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2", "channel_3"],
            target_column="target",
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mock_fit(mmm, **simple_mmm_data, draws=10, random_seed=42)
        expected = ExperimentDesigner(mmm)
        saved = mmm.idata.copy(deep=True)
        posterior = saved.posterior.dataset
        saved["posterior"] = (
            posterior.isel(channel=slice(None, None, -1))
            if posterior_change == "channel_order"
            else posterior.drop_vars("mu")
        )
        loaded = MMM.load_from_idata(saved)
        before = loaded.idata.copy(deep=True)
        changed_spend = np.zeros_like(loaded.model[Var.CHANNEL_DATA].get_value())
        pm.set_data({Var.CHANNEL_DATA: changed_spend}, model=loaded.model)

        actual = ExperimentDesigner(loaded)

        for channel in mmm.channel_columns:
            np.testing.assert_allclose(
                actual._predict_lift(channel, delta_x=10, T_active=4),
                expected._predict_lift(channel, delta_x=10, T_active=4),
            )
        assert actual._residual_std == pytest.approx(expected._residual_std)
        xr.testing.assert_identical(loaded.idata, before)
        np.testing.assert_array_equal(
            loaded.model[Var.CHANNEL_DATA].get_value(), changed_spend
        )

    def test_weekly_timezone_aware_fit_across_dst(self, simple_mmm_data):
        import pandas as pd

        from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
        from tests.mmm.conftest import mock_fit

        x = simple_mmm_data["X"]
        x["date"] = pd.date_range(
            "2025-03-01", periods=len(x), freq="W", tz="Europe/London"
        )
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2", "channel_3"],
            target_column="target",
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mock_fit(mmm, **simple_mmm_data, draws=10, random_seed=42)

        assert ExperimentDesigner(mmm)._residual_std > 0


class TestAdstockRamp:
    """Tests for graph-based ramp fraction computation."""

    def test_ramp_fraction_between_0_and_1(self, designer):
        for ch in designer.channel_columns:
            x_current = designer._current_spend[ch]
            delta_x = 0.1 * x_current
            ss = designer._steady_state_per_week_lift(ch, delta_x)
            lift = designer._predict_lift(ch, delta_x, T_active=8)
            ramp = designer._compute_graph_ramp_fraction(lift, 8, ss)
            assert 0 < ramp < 1

    def test_ramp_fraction_increases_with_duration(self, designer):
        x_current = designer._current_spend["tv"]
        delta_x = 0.1 * x_current
        ss = designer._steady_state_per_week_lift("tv", delta_x)
        lift_4 = designer._predict_lift("tv", delta_x, T_active=4)
        lift_12 = designer._predict_lift("tv", delta_x, T_active=12)
        frac_4 = designer._compute_graph_ramp_fraction(lift_4, 4, ss)
        frac_12 = designer._compute_graph_ramp_fraction(lift_12, 12, ss)
        assert frac_12 > frac_4

    def test_fast_channel_ramps_faster(self, designer):
        delta_search = 0.1 * designer._current_spend["search"]
        delta_tv = 0.1 * designer._current_spend["tv"]
        ss_search = designer._steady_state_per_week_lift("search", delta_search)
        ss_tv = designer._steady_state_per_week_lift("tv", delta_tv)
        lift_search = designer._predict_lift("search", delta_search, T_active=4)
        lift_tv = designer._predict_lift("tv", delta_tv, T_active=4)
        frac_search = designer._compute_graph_ramp_fraction(lift_search, 4, ss_search)
        frac_tv = designer._compute_graph_ramp_fraction(lift_tv, 4, ss_tv)
        # search has alpha~0.3, tv has alpha~0.7 → search ramps faster
        assert frac_search > frac_tv


class TestPredictLift:
    """Tests for lift prediction (graph-based)."""

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
        with pytest.warns(UserWarning, match="Best absolute SNR"):
            recs_high = designer.recommend(
                spend_changes=[0.2], durations=[4], min_snr=100.0
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
                "cost_efficiency": 0.0,
            },
        )
        if len(recs) > 1:
            scores = [r.score for r in recs]
            assert scores == sorted(scores, reverse=True)

    def test_partial_weights_rank_by_assurance_only(self, designer):
        scores = designer._compute_scores(
            np.array([0.4, 0.8, 0.6]), np.array([1.0, 20.0, 2.0]), {"assurance": 1}
        )
        np.testing.assert_allclose(scores, [0, 1, 0.5])

    def test_unknown_weight_raises(self, designer):
        with pytest.raises(ValueError, match=r"Unknown score weights.*uncertainty"):
            designer.recommend(score_weights={"uncertainty": 1})

    def test_invalid_significance_level_raises(self, designer):
        with pytest.raises(ValueError, match="significance_level"):
            designer.recommend(significance_level=0)

    @pytest.mark.parametrize("change", [-1.2, np.nan])
    def test_invalid_spend_change_raises(self, designer, change):
        with pytest.raises(
            ValueError, match="Spend changes must be finite and at least -1"
        ):
            designer.recommend(spend_changes=[change], durations=[4])

    def test_zero_current_spend_is_not_a_candidate(self, designer, monkeypatch):
        monkeypatch.setitem(designer._current_spend, "tv", 0.0)
        result = designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0)
        assert {rec.channel for rec in result} == {"search", "social"}

    def test_zero_spend_change_is_not_a_candidate(self, designer):
        assert not designer.recommend(spend_changes=[0], durations=[4], min_snr=0)

    def test_baselines_are_evaluated_once_per_horizon(self, designer, monkeypatch):
        calls = {}
        evaluate = designer._eval_fn
        baseline_row = designer._make_spend_array(1)[0]

        def tracked(spend):
            if len(spend) in (designer.l_max + 4, designer.l_max + 8) and np.all(
                spend == baseline_row
            ):
                calls[len(spend)] = calls.get(len(spend), 0) + 1
            return evaluate(spend)

        monkeypatch.setattr(designer, "_eval_fn", tracked)
        designer.recommend(spend_changes=[0.2, -0.2], durations=[4, 8], min_snr=0)
        assert calls == {designer.l_max + 4: 1, designer.l_max + 8: 1}

    def test_empty_result_when_nothing_passes_filter(self, designer):
        with pytest.warns(UserWarning, match=r"min_snr=1000.*Best absolute SNR"):
            recs = designer.recommend(
                spend_changes=[0.001], durations=[1], min_snr=1000.0
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
        expected = designer._residual_std * np.sqrt(
            sum(0.5 ** abs(i - j) for i in range(8) for j in range(8))
        )
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
            correlation=("search", 0.85),
        )
        assert "high spend correlation with search (|r| = 0.85)" in rationale
        assert "Caution" in rationale
        assert "identification" in rationale

    def test_low_correlation_not_described_as_high(self, designer, monkeypatch):
        import pandas as pd

        channels = designer.channel_columns
        correlations = pd.DataFrame(0.02, index=channels, columns=channels)
        monkeypatch.setattr(designer, "_spend_correlation", correlations)
        result = designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0)
        assert all("high spend correlation" not in rec.rationale for rec in result)

    def test_inverse_correlation_with_constant_channel(self, designer, monkeypatch):
        import pandas as pd

        spend = pd.DataFrame({"tv": [1, 2, 3], "search": [3, 2, 1], "social": 0})
        monkeypatch.setattr(designer, "_spend_correlation", spend.corr())

        metrics = designer._channel_metrics()
        assert metrics["tv"]["mean_correlation"] == pytest.approx(1.0)
        assert metrics["search"]["mean_correlation"] == pytest.approx(1.0)
        recommendations = designer.recommend(
            spend_changes=[0.2], durations=[4], min_snr=0
        )
        tv = next(rec for rec in recommendations if rec.channel == "tv")
        assert "Caution" in tv.rationale
        assert "|r| = 1.00" in tv.rationale


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
        """When spend_correlation is None, scoring still works."""
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
        idata = generate_experiment_fixture(
            channels=["ch1"],
            true_params={"ch1": {"lam": 1.0, "beta": 1.0, "alpha": 0.5}},
            fit_model=False,
            seed=1,
        )
        idata["constant_data"] = idata.constant_data.dataset.drop_vars(
            "residual_autocorr"
        )
        d = ExperimentDesigner.from_idata(idata)
        assert d._residual_autocorr == 0.0

    def test_from_idata_without_spend_correlation(self):
        """from_idata sets correlation to None when field is absent."""
        idata = generate_experiment_fixture(
            channels=["ch1"],
            true_params={"ch1": {"lam": 1.0, "beta": 1.0, "alpha": 0.5}},
            fit_model=False,
            seed=1,
        )
        idata["constant_data"] = idata.constant_data.dataset.drop_vars(
            "spend_correlation"
        )
        d = ExperimentDesigner.from_idata(idata)
        assert d._spend_correlation is None


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
        result = d._strongest_spend_correlation("solo")
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

    def test_plot_lift_distributions_single_cell(self, designer):
        import matplotlib.pyplot as plt

        fig, axes = designer.plot_lift_distributions(
            "tv", spend_changes=[0.2], durations=[4]
        )
        assert axes.shape == (1, 1)
        plt.close(fig)

    def test_plot_saturation_curve(self, designer):
        fig, _ax = designer.plot_saturation_curve(
            "tv",
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_saturation_sweep_agrees_with_pointwise_graph(self, designer):
        import matplotlib.pyplot as plt

        figure, axes = designer.plot_saturation_curve("tv")
        levels, means = axes.lines[0].get_data()
        for i in (0, 57, 199):
            spend = designer._make_spend_array(designer.l_max + 1)
            spend[:, 0] = levels[i]
            mean = designer._eval_fn(spend)[:, -1, 0].mean()
            assert means[i] == pytest.approx(mean)
        plt.close(figure)

    def test_plot_saturation_curve_existing_ax(self, designer):
        """Saturation curve on a pre-existing axes."""
        import matplotlib.pyplot as plt

        _fig0, ax0 = plt.subplots()
        fig, ax = designer.plot_saturation_curve("tv", ax=ax0)
        assert fig is not None
        assert ax is ax0
        plt.close(fig)

    def test_plot_saturation_curve_with_spend_levels(self, designer):
        """Saturation curve with spend-level markers."""
        import matplotlib.pyplot as plt

        fig, _ax = designer.plot_saturation_curve("tv", spend_levels=[0.2, 0.5])
        assert fig is not None
        plt.close(fig)

    def test_plot_adstock_ramp(self, designer):
        fig, _ax = designer.plot_adstock_ramp(max_weeks=8)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestEvalContributions:
    """Test the graph-based evaluation function."""

    def test_eval_contributions_shape(self, designer):
        spend = designer._make_spend_array(10)
        contrib = designer._eval_fn(spend)
        assert contrib.shape[0] == designer.n_draws
        assert contrib.shape[1] == 10
        assert contrib.shape[2] == len(designer.channel_columns)

    def test_eval_contributions_positive(self, designer):
        spend = designer._make_spend_array(10)
        contrib = designer._eval_fn(spend)
        assert np.all(contrib >= 0)
