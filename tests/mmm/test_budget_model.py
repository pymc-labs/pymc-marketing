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

import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.mmm import (
    BudgetModelEffect,
    GeometricAdstock,
    LogisticSaturation,
    MichaelisMentenSaturation,
    lift_test_design,
)
from pymc_marketing.mmm.additive_effect import FourierEffect, LinearTrendEffect
from pymc_marketing.mmm.budget_model import exogeneity_summary
from pymc_marketing.mmm.data_conversion import to_mmm_dataset
from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.mmm import MMM, BudgetOptimizerWrapper
from pymc_marketing.mmm.synthetic_data import simulate_endogenous_spend_market
from pymc_marketing.mmm.time_slice_cross_validation import TimeSliceCrossValidator
from pymc_marketing.serialization import serialization
from pymc_marketing.special_priors import LogNormalPrior
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
        mmm, effect = fitted
        X, _ = data
        chosen = mmm.idata.constant_data["budget_chosen_spend"].sel(budget_channel="tv")
        scale = float(effect._training_data(mmm).spend_scale.sel(channel="tv"))
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

    def test_log_link(self, data, design):
        X, y = data
        mmm = _make_mmm(link="log").add_mu_effect(BudgetModelEffect(design=design))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mmm.build_model(X, y)
        assert mmm.model.named_vars_to_dims["budget_effect_contribution"] == ("date",)
        assert np.isfinite(mmm.model.compile_logp()(mmm.model.initial_point()))

    def test_prior_with_wrong_channel_dim(self, data):
        X, y = data
        effect = BudgetModelEffect(gamma_prior=Prior("Normal", sigma=1, dims="channel"))
        mmm = _make_mmm().add_mu_effect(effect)
        with pytest.raises(ValueError, match="not 'channel'"):
            mmm.build_model(X, y)

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

    def test_every_field_round_trips(self, design):
        # Guards against a new field being silently dropped on save/load.
        prior = Prior("Normal", mu=0, sigma=2, dims="other_channel")
        values = {
            "prefix": "other",
            "channels": ["tv"],
            "use_controls": False,
            "fourier_order": 1,
            "trend": True,
            "instruments": ["cost_shock"],
            "design": design,
            "surprise_lags": 2,
            "surprise_out_of_sample": "observed",
            **{
                name: prior
                for name in BudgetModelEffect.model_fields
                if name.endswith("_prior")
            },
        }
        assert set(values) == set(BudgetModelEffect.model_fields)
        effect = BudgetModelEffect(**values)
        restored = serialization.deserialize(serialization.serialize(effect))
        for name in values:
            if name == "design":
                pd.testing.assert_frame_equal(restored.design, effect.design)
            else:
                assert getattr(restored, name) == getattr(effect, name), name

    def test_save_load(self, fitted, tmp_path):
        mmm, effect = fitted
        path = tmp_path / "budget.nc"
        mmm.save(str(path))
        loaded = MMM.load(str(path))
        restored = loaded.mu_effects[0]
        assert isinstance(restored, BudgetModelEffect)
        pd.testing.assert_frame_equal(restored.design, effect.design)
        assert "budget_effect_contribution" in loaded.model.named_vars
        assert loaded == mmm

    def test_equality(self, design):
        first = _make_mmm().add_mu_effect(BudgetModelEffect(design=design))
        second = _make_mmm().add_mu_effect(BudgetModelEffect(design=design))
        assert first == second
        changed = design.assign(delta_x=[-10.0, np.nan])
        assert first != _make_mmm().add_mu_effect(BudgetModelEffect(design=changed))

    def test_set_only_design_round_trip(self):
        design = pd.DataFrame(
            {
                "channel": ["digital"],
                "start_date": [DATES[3]],
                "end_date": [DATES[4]],
                "mode": ["set"],
                "delta_x": [None],
            }
        )
        restored = BudgetModelEffect.from_dict(
            BudgetModelEffect(design=design).to_dict()
        )
        shift, holdout = lift_test_design(
            restored.design, dates=DATES, channels=["tv", "digital"]
        )
        assert int(holdout.sum()) == 2
        assert float(abs(shift).sum()) == 0.0


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
        np.testing.assert_allclose(summary["prior_sd"], 0.5)
        pd.testing.assert_frame_equal(summary, effect.exogeneity_summary(mmm))

    def test_module_function_matches_method(self, fitted):
        mmm, effect = fitted
        pd.testing.assert_frame_equal(
            exogeneity_summary(mmm, prefix="budget"), effect.exogeneity_summary(mmm)
        )

    def test_sampled_prior_sd_is_reproducible(self, data):
        X, y = data
        effect = BudgetModelEffect(
            gamma_prior=Prior("StudentT", nu=3, sigma=0.5, dims="budget_channel")
        )
        mmm = _make_mmm().add_mu_effect(effect)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_fit(mmm, X, y, random_seed=1)
        first, second = effect.exogeneity_summary(mmm), effect.exogeneity_summary(mmm)
        pd.testing.assert_series_equal(first["prior_sd"], second["prior_sd"])

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
def test_lift_test_design_edge_cases():
    base = {"channel": "tv", "delta_x": 1.0}
    missing_mode = pd.DataFrame(
        [{**base, "start_date": DATES[2], "end_date": DATES[3], "mode": np.nan}]
    )
    shift, _ = lift_test_design(missing_mode, dates=DATES, channels=["tv"])
    assert float(shift.sum()) == 2.0

    with pytest.raises(ValueError, match="not in the model coords"):
        lift_test_design(
            pd.DataFrame(
                [{**base, "geo": "Z", "start_date": DATES[2], "end_date": DATES[3]}]
            ),
            dates=DATES,
            channels=["tv"],
            dim_coords={"geo": ["A"]},
        )

    overrun = pd.DataFrame(
        [
            {
                **base,
                "start_date": DATES[-2],
                "end_date": DATES[-1] + pd.Timedelta(weeks=3),
            }
        ]
    )
    shift, _ = lift_test_design(overrun, dates=DATES, channels=["tv"])
    assert float(shift.sum()) == 2.0

    between_weeks = {
        "start_date": DATES[2] + pd.Timedelta(days=1),
        "end_date": DATES[2] + pd.Timedelta(days=3),
    }
    with pytest.raises(ValueError, match="contains no model dates"):
        lift_test_design(
            pd.DataFrame([{**base, **between_weeks}]), dates=DATES, channels=["tv"]
        )


def test_instruments(data, tmp_path):
    X, y = data
    X = X.assign(cost_shock=np.random.default_rng(3).normal(size=N_WEEKS))
    effect = BudgetModelEffect(instruments=["cost_shock"])
    assert effect.data_vars == ["cost_shock"]
    mmm = _make_mmm().add_mu_effect(effect)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mock_fit(mmm, X, y, random_seed=1)
    assert "budget_instrument_cost_shock_coef" in mmm.model.named_vars
    in_sample = mmm.sample_posterior_predictive(
        X, extend_idata=False, var_names=["budget_spend_mu"], random_seed=1
    )
    assert np.isfinite(in_sample["budget_spend_mu"]).all()

    # Instruments only move the spend equation, so predictions do not need them.
    future = X.drop(columns="cost_shock").assign(
        date=X["date"] + pd.Timedelta(weeks=N_WEEKS)
    )
    pp = mmm.sample_posterior_predictive(
        future,
        extend_idata=False,
        var_names=["budget_effect_contribution"],
        random_seed=1,
    )
    assert float(abs(pp["budget_effect_contribution"]).max()) == 0.0

    path = tmp_path / "instruments.nc"
    mmm.save(str(path))
    loaded = MMM.load(str(path))
    assert "budget_instrument_cost_shock_coef" in loaded.model.named_vars


def test_panel_instrument_not_needed_for_prediction(panel_data):
    X, y = panel_data
    X = X.assign(cost_shock=np.random.default_rng(3).normal(size=len(X)))
    mmm = _make_mmm(dims=("geo",)).add_mu_effect(
        BudgetModelEffect(instruments=["cost_shock"])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mock_fit(mmm, X, y, random_seed=1)
    future = X.drop(columns="cost_shock").assign(
        date=X["date"] + pd.Timedelta(weeks=20)
    )
    pp = mmm.sample_posterior_predictive(
        future, extend_idata=False, var_names=["budget_spend_mu"], random_seed=1
    )
    assert np.isfinite(pp["budget_spend_mu"]).all()


def test_instrument_follows_model_dim_order(panel_data):
    X, y = panel_data
    X = X.assign(cost_shock=np.random.default_rng(3).normal(size=len(X)))
    mmm = _make_mmm(dims=("geo",)).add_mu_effect(
        BudgetModelEffect(instruments=["cost_shock"])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mock_fit(mmm, X, y, random_seed=1)
    dataset = mmm._posterior_predictive_data_transformation(X)
    dataset["cost_shock"] = dataset["cost_shock"].transpose("geo", "date")
    model = mmm._set_xarray_data(dataset, model=mmm.model.copy())
    mmm.mu_effects[0].set_data(mmm, model, dataset)
    np.testing.assert_allclose(
        model["cost_shock"].eval(),
        mmm.xarray_dataset["cost_shock"].transpose("date", "geo").values,
    )


def test_missing_instrument_keeps_factual_surprise_in_sample(data):
    X, y = data
    X = X.assign(cost_shock=np.random.default_rng(3).normal(size=N_WEEKS))
    mmm = _make_mmm().add_mu_effect(BudgetModelEffect(instruments=["cost_shock"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mock_fit(mmm, X, y, random_seed=1)
    pp = mmm.sample_posterior_predictive(
        X.drop(columns="cost_shock"),
        extend_idata=False,
        var_names=["budget_surprise"],
        random_seed=1,
    )
    np.testing.assert_allclose(
        pp["budget_surprise"].mean("sample").transpose("date", "budget_channel"),
        mmm.idata.posterior["budget_surprise"]
        .mean(("chain", "draw"))
        .transpose("date", "budget_channel"),
    )


def test_instrument_shared_between_effects(data):
    X, y = data
    X = X.assign(cost_shock=np.random.default_rng(3).normal(size=N_WEEKS))
    mmm = (
        _make_mmm()
        .add_mu_effect(BudgetModelEffect(instruments=["cost_shock"], channels=["tv"]))
        .add_mu_effect(
            BudgetModelEffect(
                prefix="dig", instruments=["cost_shock"], channels=["digital"]
            )
        )
    )
    mmm.build_model(X, y)
    assert {
        "budget_instrument_cost_shock_coef",
        "dig_instrument_cost_shock_coef",
    } <= set(mmm.model.named_vars)


@pytest.mark.parametrize(
    "name, dims, match",
    [
        ("unknown", None, "not in the training data"),
        ("static", ("source",), "must have a 'date' dim"),
        ("by_source", ("date", "source"), "beyond the model's"),
        ("target_scale", ("date",), "Cannot reuse model variable"),
    ],
)
def test_instrument_validation(data, name, dims, match):
    X, y = data
    ds = to_mmm_dataset(
        X, date_column="date", channel_columns=["tv", "digital"], control_columns=["c1"]
    )
    if dims is not None:
        coords = {"date": ds["date"], "source": ["a", "b"]}
        shape = tuple(len(coords[d]) for d in dims)
        ds[name] = xr.DataArray(
            np.ones(shape), dims=dims, coords={d: coords[d] for d in dims}
        )
    y_da = xr.DataArray(y.to_numpy(), dims="date", coords={"date": ds["date"]})
    mmm = _make_mmm().add_mu_effect(BudgetModelEffect(instruments=[name]))
    with pytest.raises(ValueError, match=match):
        mmm.build_model(ds, y_da)


class TestSurpriseLags:
    @pytest.fixture(scope="class")
    def lagged(self, data, design):
        X, y = data
        effect = BudgetModelEffect(design=design, surprise_lags=2)
        mmm = _make_mmm().add_mu_effect(effect)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_fit(mmm, X, y, random_seed=1)
        return mmm

    def test_structure(self, lagged):
        dims = lagged.model.named_vars_to_dims
        assert dims["budget_gamma_lag"] == ("budget_lag", "budget_channel")
        assert list(lagged.model.coords["budget_lag"]) == [1, 2]
        grad = lagged.model.compile_dlogp()(lagged.model.initial_point())
        assert np.isfinite(grad).all()

    def test_contribution_sums_lagged_surprises(self, lagged):
        post = lagged.idata.posterior.isel(chain=0, draw=0)
        surprise = post["budget_surprise"]
        gamma, gamma_lag = post["budget_gamma"], post["budget_gamma_lag"]
        expected = (surprise * gamma).sum("budget_channel")
        for lag in (1, 2):
            shifted = surprise.shift(date=lag, fill_value=0.0)
            expected = expected + (shifted * gamma_lag.sel(budget_lag=lag)).sum(
                "budget_channel"
            )
        np.testing.assert_allclose(
            post["budget_effect_contribution"].values, expected.values, atol=1e-10
        )

    def test_future_dates_have_zero_contribution(self, lagged, data):
        X, _ = data
        future = X.assign(date=X["date"] + pd.Timedelta(weeks=N_WEEKS))
        pp = lagged.sample_posterior_predictive(
            future,
            extend_idata=False,
            var_names=["budget_effect_contribution"],
            random_seed=1,
        )
        assert float(abs(pp["budget_effect_contribution"]).max()) == 0.0

    def test_single_lag_gradient(self, data):
        X, y = data
        mmm = _make_mmm().add_mu_effect(BudgetModelEffect(surprise_lags=1))
        mmm.build_model(X, y)
        grad = mmm.model.compile_dlogp()(mmm.model.initial_point())
        assert np.isfinite(grad).all()

    def test_summary_reports_every_lag(self, lagged):
        summary = lagged.mu_effects[0].exogeneity_summary(lagged)
        assert list(summary.columns[:2]) == ["channel", "lag"]
        assert sorted(summary["lag"].unique()) == [0, 1, 2]
        assert len(summary) == 6
        np.testing.assert_allclose(summary["prior_sd"], 0.5)

    def test_round_trip_and_validation(self, data):
        effect = BudgetModelEffect(surprise_lags=3)
        restored = BudgetModelEffect.from_dict(effect.to_dict())
        assert restored.surprise_lags == 3
        X, y = data
        too_many = _make_mmm().add_mu_effect(BudgetModelEffect(surprise_lags=N_WEEKS))
        with pytest.raises(ValueError, match="must be smaller than the number"):
            too_many.build_model(X, y)


class TestIdentificationGuards:
    def test_fourier_order_follows_sales_seasonality(self, data):
        X, y = data
        mmm = _make_mmm(yearly_seasonality=3).add_mu_effect(BudgetModelEffect())
        mmm.build_model(X, y)
        assert len(mmm.model.coords["budget_fourier"]) == 6

        no_season = _make_mmm().add_mu_effect(BudgetModelEffect())
        no_season.build_model(X, y)
        assert "budget_spend_fourier_coef" not in no_season.model.named_vars

    def test_fourier_order_follows_fourier_effect(self, data):
        X, y = data
        mmm = (
            _make_mmm()
            .add_mu_effect(FourierEffect(fourier=YearlyFourier(n_order=3)))
            .add_mu_effect(BudgetModelEffect())
        )
        mmm.build_model(X, y)
        assert len(mmm.model.coords["budget_fourier"]) == 6

    def test_fourier_order_above_sales_warns(self, data):
        X, y = data
        mmm = _make_mmm().add_mu_effect(BudgetModelEffect(fourier_order=2))
        with pytest.warns(UserWarning, match="act as instruments"):
            mmm.build_model(X, y)

    def test_trend(self, data):
        X, y = data
        mmm = _make_mmm().add_mu_effect(BudgetModelEffect(trend=True))
        with pytest.warns(UserWarning, match="trend acts as an instrument"):
            mmm.build_model(X, y)
        assert "budget_spend_trend_coef" in mmm.model.named_vars

        with_trend = (
            _make_mmm()
            .add_mu_effect(LinearTrendEffect(trend=LinearTrend(), prefix="trend"))
            .add_mu_effect(BudgetModelEffect(trend=True))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with_trend.build_model(X, y)

    def test_trend_predictions(self, data):
        X, y = data
        effect = BudgetModelEffect(trend=True)
        mmm = _make_mmm().add_mu_effect(effect)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_fit(mmm, X, y, random_seed=1)
        future = X.assign(date=X["date"] + pd.Timedelta(weeks=N_WEEKS))
        pp = mmm.sample_posterior_predictive(
            future, extend_idata=False, var_names=["budget_spend_mu"], random_seed=1
        )
        assert np.isfinite(pp["budget_spend_mu"]).all()

    @pytest.mark.parametrize(
        "tv_spend, delta_x",
        [(0.0, -50.0), (300.0, 1_000.0)],
        ids=["zero-spend", "negative-chosen"],
    )
    def test_unrealised_shift_warns(self, data, tv_spend, delta_x):
        X, y = data
        X = X.copy()
        X.loc[10, "tv"] = tv_spend
        design = pd.DataFrame(
            {
                "channel": ["tv"],
                "start_date": [DATES[10]],
                "end_date": [DATES[11]],
                "delta_x": [delta_x],
            }
        )
        mmm = _make_mmm().add_mu_effect(BudgetModelEffect(design=design))
        with pytest.warns(UserWarning, match="may not have been fully realised"):
            mmm.build_model(X, y)

    def test_design_rows_and_columns(self, data, design):
        X, y = data
        extra_column = design.assign(measured_at=pd.Timestamp("2024-01-01"))
        mmm = _make_mmm().add_mu_effect(BudgetModelEffect(design=extra_column))
        with pytest.raises(ValueError, match="does not use"):
            mmm.build_model(X, y)

        typo = design.assign(channel=["tv", "digitl"])
        mmm = _make_mmm().add_mu_effect(BudgetModelEffect(design=typo))
        with pytest.raises(ValueError, match="the MMM does not have"):
            mmm.build_model(X, y)

        # A design for a channel that is not modelled is dropped, like one
        # outside the data.
        mmm = _make_mmm().add_mu_effect(
            BudgetModelEffect(design=design, channels=["tv"])
        )
        with pytest.warns(UserWarning, match="Dropping 1 BudgetModelEffect design"):
            mmm.build_model(X, y)

    def test_flighted_channel_warns(self, data):
        X, y = data
        X = X.assign(tv=np.where(np.arange(N_WEEKS) % 3 == 0, 0.0, X["tv"]))
        mmm = _make_mmm().add_mu_effect(BudgetModelEffect())
        with pytest.warns(UserWarning, match="zero spend in more than 20%"):
            mmm.build_model(X, y)

    def test_instance_serves_each_model_it_is_in(self, data):
        # State is derived from the model, so a shared instance cannot mix up
        # two models trained on different data.
        X, y = data
        effect = BudgetModelEffect()
        first = _make_mmm().add_mu_effect(effect)
        second = _make_mmm().add_mu_effect(effect)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_fit(first, X, y, random_seed=1)
            mock_fit(second, X.iloc[:20], y.iloc[:20], random_seed=1)
        pp = first.sample_posterior_predictive(
            X, extend_idata=False, var_names=["budget_surprise"], random_seed=1
        )
        assert pp.sizes["date"] == N_WEEKS
        assert (
            float(abs(pp["budget_surprise"]).min(["sample", "budget_channel"]).max())
            > 0
        )
        assert not effect.exogeneity_summary(first).empty
        assert not effect.exogeneity_summary(second).empty

    def test_fitted_model_can_be_copied_and_pickled(self, fitted, data):
        mmm, effect = fitted
        X, _ = data
        copied = copy.deepcopy(mmm)
        pp = copied.sample_posterior_predictive(
            X, extend_idata=False, var_names=["y"], random_seed=1
        )
        assert np.isfinite(pp["y"]).all()
        pd.testing.assert_frame_equal(
            copied.mu_effects[0].exogeneity_summary(copied),
            effect.exogeneity_summary(mmm),
        )
        restored = pickle.loads(pickle.dumps(effect))  # noqa: S301
        assert restored.to_dict() == effect.to_dict()

    def test_summary_from_original_instance_after_load(self, fitted, tmp_path):
        mmm, effect = fitted
        path = tmp_path / "budget.nc"
        mmm.save(str(path))
        loaded = MMM.load(str(path))
        pd.testing.assert_frame_equal(
            effect.exogeneity_summary(loaded),
            loaded.mu_effects[0].exogeneity_summary(loaded),
        )

    def test_summary_without_attached_effect_raises(self, fitted):
        mmm, effect = fitted
        effects = mmm.mu_effects
        try:
            mmm.mu_effects = []
            with pytest.raises(RuntimeError, match="has no BudgetModelEffect"):
                effect.exogeneity_summary(mmm)
        finally:
            mmm.mu_effects = effects

    def test_observed_surprise_out_of_sample(self, data):
        X, y = data
        effect = BudgetModelEffect(surprise_out_of_sample="observed")
        mmm = _make_mmm().add_mu_effect(effect)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_fit(mmm, X, y, random_seed=1)
        future = X.assign(date=X["date"] + pd.Timedelta(weeks=N_WEEKS))
        pp = mmm.sample_posterior_predictive(
            future,
            extend_idata=False,
            var_names=["budget_effect_contribution"],
            random_seed=1,
        )
        assert float(abs(pp["budget_effect_contribution"]).max()) > 0.0

    def test_observed_surprise_applies_design_on_new_dates(self, data):
        # A CV fold trained on weeks 0-23 whose test window holds the experiment.
        X, y = data
        design = pd.DataFrame(
            {
                "channel": ["digital", "tv"],
                "start_date": [DATES[25], DATES[26]],
                "end_date": [DATES[26], DATES[27]],
                "mode": ["set", "shift"],
                "delta_x": [np.nan, -50.0],
            }
        )
        effect = BudgetModelEffect(design=design, surprise_out_of_sample="observed")
        mmm = _make_mmm().add_mu_effect(effect)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_fit(mmm, X.iloc[:24], y.iloc[:24], random_seed=1)
        pp = mmm.sample_posterior_predictive(
            X,
            extend_idata=False,
            var_names=["budget_surprise", "budget_spend_mu"],
            random_seed=1,
        )
        surprise = pp["budget_surprise"].transpose("sample", "date", "budget_channel")
        spend_mu = pp["budget_spend_mu"].transpose("sample", "date", "budget_channel")
        holdout = surprise.sel(budget_channel="digital").isel(date=[25, 26])
        assert float(abs(holdout).max()) == 0.0

        scale = float(effect._training_data(mmm).spend_scale.sel(channel="tv"))
        chosen_tv = (X["tv"].to_numpy()[26:28] + 50.0) / scale
        expected = chosen_tv - spend_mu.sel(budget_channel="tv").isel(date=[26, 27])
        np.testing.assert_allclose(
            surprise.sel(budget_channel="tv").isel(date=[26, 27]).values,
            expected.values,
        )

    def test_observed_surprise_requires_spend(self, data):
        X, y = data
        mmm = _make_mmm().add_mu_effect(
            BudgetModelEffect(surprise_out_of_sample="observed")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_fit(mmm, X.iloc[:24], y.iloc[:24], random_seed=1)
        dataset = mmm._posterior_predictive_data_transformation(X)
        model = mmm._set_xarray_data(dataset, model=mmm.model.copy())
        with pytest.raises(ValueError, match="has no channel spend"):
            mmm.mu_effects[0].set_data(mmm, model, dataset.drop_vars("_channel"))

    def test_observed_surprise_requires_instruments(self, data):
        X, y = data
        X = X.assign(cost_shock=np.random.default_rng(3).normal(size=N_WEEKS))
        effect = BudgetModelEffect(
            instruments=["cost_shock"], surprise_out_of_sample="observed"
        )
        mmm = _make_mmm().add_mu_effect(effect)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_fit(mmm, X, y, random_seed=1)
        future = X.drop(columns="cost_shock").assign(
            date=X["date"] + pd.Timedelta(weeks=N_WEEKS)
        )
        with pytest.raises(ValueError, match="is required on new dates"):
            mmm.sample_posterior_predictive(
                future, extend_idata=False, var_names=["y"], random_seed=1
            )

    def test_optimizer_warns_in_observed_mode(self, data):
        X, y = data
        mmm = _make_mmm().add_mu_effect(
            BudgetModelEffect(surprise_out_of_sample="observed")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_fit(mmm, X, y, random_seed=1)
        start = DATES[-1] + pd.Timedelta(weeks=1)
        with pytest.warns(UserWarning, match="not budget scenarios"):
            mmm.create_optimization_model(start, start + pd.Timedelta(weeks=3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            wrapper = BudgetOptimizerWrapper(
                model=mmm,
                start_date=str(start.date()),
                end_date=str((start + pd.Timedelta(weeks=3)).date()),
            )
        with pytest.warns(UserWarning, match="not budget scenarios"):
            wrapper.optimization_model(4)

    def test_cross_validation_with_late_design(self, data):
        X, y = data
        design = pd.DataFrame(
            {
                "channel": ["tv"],
                "start_date": [DATES[26]],
                "end_date": [DATES[27]],
                "delta_x": [-50.0],
            }
        )
        mmm = _make_mmm().add_mu_effect(BudgetModelEffect(design=design))
        cv = TimeSliceCrossValidator(n_init=24, forecast_horizon=2, date_column="date")
        sampler_config = {
            "draws": 10,
            "tune": 10,
            "chains": 1,
            "progressbar": False,
            "random_seed": 1,
        }
        with pytest.warns(UserWarning, match="Dropping 1 BudgetModelEffect design"):
            cv.run(X, y, mmm=mmm, sampler_config=sampler_config)
        assert len(cv._cv_results) == cv.get_n_splits(X, y)


@pytest.mark.slow
def test_negative_control_gamma_interval_contains_zero():
    market = simulate_endogenous_spend_market("observed_only", random_seed=11)
    X, y = market.data.drop(columns="y"), market.data["y"]
    effect = BudgetModelEffect(design=market.design)
    mmm = MMM(
        date_column="date",
        channel_columns=["tv", "digital"],
        control_columns=["inflation", "unemployment"],
        target_column="y",
        yearly_seasonality=4,
        adstock=GeometricAdstock(l_max=8),
        saturation=MichaelisMentenSaturation(),
    ).add_mu_effect(effect)
    mmm.fit(
        X,
        y,
        nuts_sampler="pymc",
        draws=500,
        tune=1000,
        chains=2,
        target_accept=0.95,
        random_seed=11,
        progressbar=False,
    )
    tv = effect.exogeneity_summary(mmm).set_index("channel").loc["tv"]
    # Looser than the 94% interval, so numerical drift across releases cannot
    # flip a single-seed false alarm that happens ~12% of the time.
    assert 0.02 < tv["prob_positive"] < 0.98


def test_guards(fitted):
    mmm, _ = fitted
    assert BudgetModelEffect(design=None).design is None
    unbuilt = BudgetModelEffect(prefix="unbuilt")
    with pytest.raises(RuntimeError, match="Build the MMM with this effect"):
        unbuilt.set_data(mmm, mmm.model, None)
    with pytest.raises(ValueError, match="interval_prob must be in"):
        mmm.mu_effects[0].exogeneity_summary(mmm, interval_prob=1.5)


def test_from_dict_registered_prior():
    effect = BudgetModelEffect(
        spend_sigma_prior=LogNormalPrior(mean=0.2, std=0.1, dims="budget_channel")
    )
    restored = BudgetModelEffect.from_dict(effect.to_dict())
    assert isinstance(restored.spend_sigma_prior, LogNormalPrior)


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


@pytest.mark.parametrize("tested", [["digital"], []], ids=["other-channel", "none"])
def test_lift_likelihood_no_warning_for_untested_channel(data, tested):
    X, y = data
    design = pd.DataFrame(
        {
            "channel": tested,
            "start_date": [DATES[3]] * len(tested),
            "end_date": [DATES[4]] * len(tested),
            "delta_x": [5.0] * len(tested),
        }
    )
    effect = BudgetModelEffect(design=design if tested else None)
    mmm = _make_mmm().add_mu_effect(effect)
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
            nuts_sampler="pymc",
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
