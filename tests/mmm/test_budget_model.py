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
"""Tests for the budget model (control-function) effect."""

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from pymc_extras.prior import Prior

from pymc_marketing.mmm import (
    BudgetModelEffect,
    GeometricAdstock,
    LogisticSaturation,
    MichaelisMentenSaturation,
    lift_test_design,
)
from pymc_marketing.mmm.mmm import MMM
from pymc_marketing.mmm.synthetic_data import simulate_endogenous_spend_market
from pymc_marketing.serialization import serialization
from tests.mmm.conftest import mock_fit

N_WEEKS = 30
DATES = pd.date_range("2023-01-02", periods=N_WEEKS, freq="W-MON")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "date": DATES,
            "tv": rng.uniform(100, 500, N_WEEKS),
            "digital": rng.uniform(50, 200, N_WEEKS),
            "c1": rng.normal(size=N_WEEKS),
        }
    )
    y = pd.Series(X["tv"] + X["digital"] + rng.normal(0, 10, N_WEEKS), name="y")
    return X, y


@pytest.fixture(scope="module")
def panel_data():
    rng = np.random.default_rng(1)
    dates = DATES[:20]
    rows = [
        (d, g, rng.uniform(100, 500), rng.uniform(50, 200), rng.normal())
        for g in ("A", "B")
        for d in dates
    ]
    X = pd.DataFrame(rows, columns=["date", "geo", "tv", "digital", "c1"])
    y = pd.Series(X["tv"] + X["digital"] + rng.normal(0, 10, len(X)), name="y")
    return X, y


@pytest.fixture(scope="module")
def design():
    return pd.DataFrame(
        {
            "channel": ["tv", "digital"],
            "start_date": [DATES[10], DATES[20]],
            "end_date": [DATES[13], DATES[21]],
            "mode": ["shift", "set"],
            "delta_x": [-50.0, np.nan],
        }
    )


def _make_mmm(**kwargs) -> MMM:
    return MMM(
        date_column="date",
        channel_columns=["tv", "digital"],
        control_columns=["c1"],
        target_column="y",
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **kwargs,
    )


@pytest.fixture(scope="module")
def fitted(data, design):
    X, y = data
    effect = BudgetModelEffect(design=design)
    mmm = _make_mmm(yearly_seasonality=2).add_mu_effect(effect)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mock_fit(mmm, X, y, random_seed=1)
    return mmm, effect


# --------------------------------------------------------------- design helper
class TestLiftTestDesign:
    def test_shift_and_set(self, design):
        shift, holdout = lift_test_design(
            design, dates=DATES, channels=["tv", "digital"]
        )
        assert shift.dims == ("date", "channel")
        np.testing.assert_allclose(shift.sel(channel="tv").values[10:14], -50.0)
        assert float(shift.sel(channel="tv").sum()) == pytest.approx(-200.0)
        assert float(abs(shift.sel(channel="digital")).sum()) == 0.0
        assert holdout.sel(channel="digital").values[20:22].all()
        assert int(holdout.sum()) == 2

    def test_default_mode_is_shift(self):
        df = pd.DataFrame(
            {
                "channel": ["tv"],
                "start_date": [DATES[0]],
                "end_date": [DATES[1]],
                "delta_x": [5.0],
            }
        )
        shift, holdout = lift_test_design(df, dates=DATES, channels=["tv"])
        assert float(shift.sum()) == 10.0
        assert not holdout.any()

    def test_dims(self):
        df = pd.DataFrame(
            {
                "channel": ["tv"],
                "geo": ["B"],
                "start_date": [DATES[2]],
                "end_date": [DATES[3]],
                "delta_x": [1.0],
            }
        )
        shift, _ = lift_test_design(
            df, dates=DATES, channels=["tv"], dim_coords={"geo": ["A", "B"]}
        )
        assert shift.dims == ("date", "geo", "channel")
        assert float(shift.sel(geo="A").sum()) == 0.0
        assert float(shift.sel(geo="B").sum()) == 2.0

    @pytest.mark.parametrize(
        "row, match",
        [
            ({"channel": "radio"}, "not one of"),
            ({"start_date": DATES[5], "end_date": DATES[4]}, "after end_date"),
            ({"end_date": DATES[-1] + pd.Timedelta(weeks=2)}, "outside"),
            ({"mode": "halve"}, "mode must be"),
            ({"delta_x": np.nan}, "require delta_x"),
        ],
    )
    def test_validation(self, row, match):
        base = {
            "channel": "tv",
            "start_date": DATES[2],
            "end_date": DATES[4],
            "mode": "shift",
            "delta_x": 1.0,
        }
        df = pd.DataFrame([{**base, **row}])
        with pytest.raises(ValueError, match=match):
            lift_test_design(df, dates=DATES, channels=["tv"])

    def test_overlap_raises(self):
        df = pd.DataFrame(
            {
                "channel": ["tv", "tv"],
                "start_date": [DATES[2], DATES[4]],
                "end_date": [DATES[5], DATES[6]],
                "delta_x": [1.0, 2.0],
            }
        )
        with pytest.raises(ValueError, match="overlaps"):
            lift_test_design(df, dates=DATES, channels=["tv"])

    def test_missing_columns(self):
        with pytest.raises(ValueError, match="missing required columns"):
            lift_test_design(
                pd.DataFrame({"channel": ["tv"]}), dates=DATES, channels=["tv"]
            )

    def test_missing_dim_column(self, design):
        with pytest.raises(ValueError, match="geo"):
            lift_test_design(
                design,
                dates=DATES,
                channels=["tv", "digital"],
                dim_coords={"geo": ["A"]},
            )


# ------------------------------------------------------------------- building
class TestBuild:
    def test_named_vars_and_dims(self, fitted):
        mmm, _ = fitted
        model = mmm.model
        for name in [
            "budget_chosen_spend",
            "budget_active",
            "budget_spend_intercept",
            "budget_spend_control_coef",
            "budget_spend_fourier_coef",
            "budget_spend_sigma",
            "budget_spend_mu",
            "budget_surprise",
            "budget_gamma",
            "budget_spend",
        ]:
            assert name in model.named_vars, name
        assert model.named_vars_to_dims["budget_effect_contribution"] == ("date",)
        assert model.named_vars_to_dims["budget_surprise"] == (
            "date",
            "budget_channel",
        )
        assert list(model.coords["budget_channel"]) == ["tv", "digital"]
        assert np.isfinite(model.compile_logp()(model.initial_point()))

    def test_gradient_is_available(self, fitted):
        # NUTS needs dlogp; a Potential over xtensor ops would not provide it.
        model = fitted[0].model
        grad = model.compile_dlogp()(model.initial_point())
        assert np.isfinite(grad).all()

    def test_masked_cells_contribute_constant_logp(self, fitted):
        model = fitted[0].model
        logp_fn = model.compile_fn(
            model.logp(vars=[model["budget_spend"]], sum=False)[0],
            inputs=model.value_vars,
            on_unused_input="ignore",
        )
        point = model.initial_point()
        values = logp_fn(point)
        sigma_key = "budget_spend_sigma_log__"
        point[sigma_key] = point[sigma_key] + 1.0
        shifted = logp_fn(point)
        # Holdout cells (digital, weeks 20-21) do not depend on the parameters.
        np.testing.assert_allclose(values[20:22, 1], shifted[20:22, 1])
        np.testing.assert_allclose(values[20:22, 1], -0.5 * np.log(2 * np.pi))

    def test_chosen_spend_removes_design_shift(self, fitted, data):
        _, effect = fitted
        X, _ = data
        chosen = effect._factual["chosen"].sel(budget_channel="tv")
        scale = float(effect._spend_scale.sel(budget_channel="tv"))
        expected = X["tv"].to_numpy().copy()
        expected[10:14] += 50.0
        np.testing.assert_allclose(chosen.values * scale, expected)

    def test_holdout_is_masked(self, fitted):
        mmm, _ = fitted
        surprise = mmm.idata.posterior["budget_surprise"]
        assert (
            float(abs(surprise.sel(budget_channel="digital").isel(date=[20, 21])).max())
            == 0.0
        )
        assert (
            float(abs(surprise.sel(budget_channel="digital").isel(date=[5])).max()) > 0
        )

    def test_decomposition_includes_effect(self, fitted):
        mmm, _ = fitted
        assert mmm.mu_effects[0].contribution_var_name == "budget_effect_contribution"
        assert "budget_effect_contribution" in mmm.idata.posterior

    def test_channel_subset(self, data):
        X, y = data
        mmm = _make_mmm().add_mu_effect(
            BudgetModelEffect(channels=["tv"], fourier_order=0, use_controls=False)
        )
        mmm.build_model(X, y)
        assert list(mmm.model.coords["budget_channel"]) == ["tv"]
        assert "budget_spend_fourier_coef" not in mmm.model.named_vars
        assert "budget_spend_control_coef" not in mmm.model.named_vars

    def test_unknown_channel(self, data):
        X, y = data
        mmm = _make_mmm().add_mu_effect(BudgetModelEffect(channels=["radio"]))
        with pytest.raises(ValueError, match="Unknown channels"):
            mmm.build_model(X, y)

    def test_custom_prior(self, data):
        X, y = data
        gamma = Prior("Normal", mu=0, sigma=2, dims="budget_channel")
        mmm = _make_mmm().add_mu_effect(BudgetModelEffect(gamma_prior=gamma))
        mmm.build_model(X, y)
        assert mmm.model.named_vars_to_dims["budget_gamma"] == ("budget_channel",)

    def test_design_missing_column(self):
        with pytest.raises(ValueError, match="missing required columns"):
            BudgetModelEffect(design=pd.DataFrame({"channel": ["tv"]}))

    def test_panel(self, panel_data):
        X, y = panel_data
        design = pd.DataFrame(
            {
                "channel": ["tv"],
                "geo": ["B"],
                "start_date": [DATES[5]],
                "end_date": [DATES[8]],
                "delta_x": [-40.0],
            }
        )
        effect = BudgetModelEffect(design=design, channels=["tv"])
        mmm = _make_mmm(dims=("geo",)).add_mu_effect(effect)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_fit(mmm, X, y, random_seed=1)
        dims = mmm.model.named_vars_to_dims
        assert dims["budget_effect_contribution"] == ("date", "geo")
        assert dims["budget_spend_intercept"] == ("geo", "budget_channel")
        summary = effect.exogeneity_summary(mmm)
        assert set(summary["geo"]) == {"A", "B"}
        # gamma is pooled over geo, so the design in B identifies it for A too.
        assert summary["identified_by_design"].all()


# ----------------------------------------------- interventions and predictions
class TestInterventions:
    def test_not_reachable_from_channel_data(self, fitted):
        mmm, _ = fitted
        assert not mmm._effects_carry_media_response()

    def test_do_channel_data_leaves_contribution_unchanged(self, fitted):
        mmm, _ = fitted
        model = mmm.model
        post = mmm.idata.posterior.isel(chain=0, draw=0)
        point = {rv.name: post[rv.name].values for rv in model.free_RVs}
        f0 = model.compile_fn(
            model["budget_effect_contribution"],
            inputs=model.free_RVs,
            on_unused_input="ignore",
        )
        intervened = pm.do(model, {"channel_data": np.zeros((N_WEEKS, 2))})
        f1 = intervened.compile_fn(
            intervened["budget_effect_contribution"],
            inputs=intervened.free_RVs,
            on_unused_input="ignore",
        )
        np.testing.assert_allclose(f0(point), f1(point))

    def test_future_dates_have_zero_contribution(self, fitted):
        mmm, _ = fitted
        X_new = pd.DataFrame(
            {
                "date": pd.date_range(
                    DATES[-1] + pd.Timedelta(weeks=1), periods=5, freq="W-MON"
                ),
                "tv": 300.0,
                "digital": 100.0,
                "c1": 0.0,
            }
        )
        pp = mmm.sample_posterior_predictive(
            X_new,
            extend_idata=False,
            var_names=["budget_effect_contribution", "y"],
            random_seed=1,
        )
        assert float(abs(pp["budget_effect_contribution"]).max()) == 0.0

    def test_modified_spend_in_sample_keeps_factual_surprise(self, fitted, data):
        mmm, _ = fitted
        X, _ = data
        X_mod = X.assign(tv=0.0, digital=1000.0)
        pp = mmm.sample_posterior_predictive(
            X_mod,
            extend_idata=False,
            var_names=["budget_effect_contribution"],
            random_seed=1,
        )
        new = pp["budget_effect_contribution"].transpose(..., "date").values
        factual = mmm.idata.posterior["budget_effect_contribution"]
        np.testing.assert_allclose(
            new.reshape(-1, N_WEEKS),
            factual.transpose(..., "date").values.reshape(-1, N_WEEKS),
        )

    def test_incrementality_runs(self, fitted):
        mmm, _ = fitted
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            inc = mmm.incrementality.compute_incremental_contribution(
                frequency="all_time"
            )
        assert "channel" in inc.dims

    def test_budget_optimizer_runs(self, fitted):
        mmm, _ = fitted
        start = DATES[-1] + pd.Timedelta(weeks=1)
        optimizer = mmm.budget_optimizer(
            start_date=start, end_date=start + pd.Timedelta(weeks=3)
        )
        _, result = optimizer.allocate_budget(total_budget=500)
        assert result.success


# -------------------------------------------------------------- serialization
class TestSerialization:
    def test_dict_round_trip(self, design):
        effect = BudgetModelEffect(
            design=design,
            channels=["tv"],
            fourier_order=3,
            gamma_prior=Prior("Normal", mu=0, sigma=1, dims="budget_channel"),
        )
        restored = serialization.deserialize(serialization.serialize(effect))
        assert isinstance(restored, BudgetModelEffect)
        assert restored.channels == ["tv"]
        assert restored.fourier_order == 3
        assert restored.gamma_prior == effect.gamma_prior
        pd.testing.assert_frame_equal(restored.design, effect.design)

    def test_save_load(self, fitted, tmp_path):
        mmm, effect = fitted
        path = tmp_path / "budget.nc"
        mmm.save(str(path))
        loaded = MMM.load(str(path))
        restored = loaded.mu_effects[0]
        assert isinstance(restored, BudgetModelEffect)
        pd.testing.assert_frame_equal(restored.design, effect.design)
        assert "budget_effect_contribution" in loaded.model.named_vars


# ---------------------------------------------------------------- diagnostics
class TestExogeneitySummary:
    def test_schema(self, fitted):
        mmm, effect = fitted
        summary = effect.exogeneity_summary(mmm)
        assert list(summary["channel"]) == ["tv", "digital"]
        for col in [
            "gamma_mean",
            "gamma_lower",
            "gamma_upper",
            "gamma_per_spend_unit",
            "prob_positive",
            "prior_sd",
            "posterior_sd",
            "contraction",
            "identified_by_design",
            "note",
        ]:
            assert col in summary.columns, col
        assert (summary["gamma_lower"] <= summary["gamma_upper"]).all()
        assert summary["identified_by_design"].all()

    def test_flags_functional_form_identification(self, data):
        X, y = data
        effect = BudgetModelEffect()
        mmm = _make_mmm().add_mu_effect(effect)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_fit(mmm, X, y, random_seed=1)
        summary = effect.exogeneity_summary(mmm)
        assert not summary["identified_by_design"].any()
        assert summary["note"].str.contains("functional form").all()

    def test_requires_posterior(self, data):
        X, y = data
        effect = BudgetModelEffect()
        mmm = _make_mmm().add_mu_effect(effect)
        mmm.build_model(X, y)
        with pytest.raises(RuntimeError, match="fit the model first"):
            effect.exogeneity_summary(mmm)


# ------------------------------------------------------ lift-test interaction
def test_lift_likelihood_double_count_warning(data, design):
    X, y = data
    mmm = _make_mmm().add_mu_effect(BudgetModelEffect(design=design))
    mmm.build_model(X, y)
    df_lift = pd.DataFrame(
        {
            "channel": ["tv"],
            "x": [300.0],
            "delta_x": [-50.0],
            "delta_y": [-20.0],
            "sigma": [5.0],
        }
    )
    with pytest.warns(UserWarning, match="counts them twice"):
        mmm.add_lift_test_measurements(df_lift)


def test_lift_likelihood_no_warning_for_untested_channel(data):
    X, y = data
    design = pd.DataFrame(
        {
            "channel": ["digital"],
            "start_date": [DATES[3]],
            "end_date": [DATES[4]],
            "delta_x": [5.0],
        }
    )
    mmm = _make_mmm().add_mu_effect(BudgetModelEffect(design=design))
    mmm.build_model(X, y)
    df_lift = pd.DataFrame(
        {
            "channel": ["tv"],
            "x": [300.0],
            "delta_x": [-50.0],
            "delta_y": [-20.0],
            "sigma": [5.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mmm.add_lift_test_measurements(df_lift)


# ------------------------------------------------------------------- recovery
@pytest.mark.parametrize(
    "scenario", ["forecast", "target_chasing", "search", "observed_only"]
)
def test_synthetic_market(scenario):
    market = simulate_endogenous_spend_market(scenario, random_seed=1)
    again = simulate_endogenous_spend_market(scenario, random_seed=1)
    pd.testing.assert_frame_equal(market.data, again.data)
    assert len(market.data) == 120
    assert market.lift_test["delta_x"].item() < 0
    assert market.lift_test["delta_y"].item() < 0
    shift, holdout = lift_test_design(
        market.design, dates=market.data["date"], channels=["tv", "digital"]
    )
    assert int(holdout.sum()) == (4 if scenario == "search" else 0)
    assert np.isclose(float(shift.sum()), 0 if scenario == "search" else -6.0)


def test_synthetic_market_invalid_scenario():
    with pytest.raises(ValueError, match="scenario must be one of"):
        simulate_endogenous_spend_market("random")


@pytest.mark.slow
def test_budget_model_recovers_marginal_return():
    market = simulate_endogenous_spend_market("forecast", random_seed=7)
    X, y = market.data.drop(columns="y"), market.data["y"]

    def fit(effect):
        mmm = MMM(
            date_column="date",
            channel_columns=["tv", "digital"],
            control_columns=["inflation", "unemployment"],
            target_column="y",
            yearly_seasonality=4,
            adstock=GeometricAdstock(l_max=8),
            saturation=MichaelisMentenSaturation(),
        )
        if effect is not None:
            mmm.add_mu_effect(effect)
        mmm.fit(
            X,
            y,
            draws=500,
            tune=1000,
            chains=2,
            target_accept=0.95,
            random_seed=3,
            progressbar=False,
        )
        post = mmm.idata.posterior
        lam = post["saturation_lam"].sel(channel="tv") * float(
            mmm.scalers["_channel"].sel(channel="tv")
        )
        beta = post["saturation_alpha"].sel(channel="tv") * float(
            mmm.scalers["_target"]
        )
        x = market.operating_point
        return float((beta * lam / (lam + x) ** 2).mean()), mmm

    plain_mr, _ = fit(None)
    effect = BudgetModelEffect(design=market.design)
    budget_mr, mmm = fit(effect)

    truth = market.true_marginal_return
    assert abs(budget_mr - truth) < abs(plain_mr - truth)
    summary = effect.exogeneity_summary(mmm)
    assert summary.loc[summary["channel"] == "tv", "prob_positive"].item() > 0.9
