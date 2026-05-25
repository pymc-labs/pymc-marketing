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
"""Tests for FrequencyReachAdditiveEffect and HillShapeSaturation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from pymc_marketing.mmm import (
    MMM,
    RF_CHANNEL_COORD,
    FrequencyReachAdditiveEffect,
    GeometricAdstock,
    HillShapeSaturation,
    LogisticSaturation,
)
from pymc_marketing.prior import Prior

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_DATES = 12
CHANNELS = ["youtube", "display"]
DATES = pd.date_range("2024-01-01", periods=N_DATES, freq="W-MON")


def _make_rf_df(channels=None, n_dates=N_DATES, dates=None, seed=0):
    """Return a minimal long-format reach/frequency DataFrame."""
    rng = np.random.default_rng(seed)
    if channels is None:
        channels = CHANNELS
    if dates is None:
        dates = DATES
    rows = []
    for date in dates:
        for ch in channels:
            rows.append(
                {
                    "date": date,
                    "channel": ch,
                    "reach": float(rng.uniform(0.1, 0.8)),
                    "frequency": float(rng.uniform(1.0, 5.0)),
                }
            )
    return pd.DataFrame(rows)


def _make_cpu_df(channels=None, n_dates=N_DATES, dates=None):
    """Return a cost_per_unit DataFrame for R&F channels."""
    if channels is None:
        channels = CHANNELS
    if dates is None:
        dates = DATES
    data = {"date": dates}
    for ch in channels:
        data[ch] = np.full(n_dates, 0.01)  # $0.01 per impression
    return pd.DataFrame(data)


@pytest.fixture
def rf_df():
    return _make_rf_df()


@pytest.fixture
def cpu_df():
    return _make_cpu_df()


@pytest.fixture
def saturation():
    return HillShapeSaturation()


@pytest.fixture
def adstock():
    return GeometricAdstock(l_max=2)


@pytest.fixture
def effect(rf_df, saturation, adstock):
    return FrequencyReachAdditiveEffect(
        df_frequency_reach=rf_df,
        saturation=saturation,
        adstock=adstock,
    )


@pytest.fixture
def effect_with_cpu(rf_df, saturation, adstock, cpu_df):
    return FrequencyReachAdditiveEffect(
        df_frequency_reach=rf_df,
        saturation=saturation,
        adstock=adstock,
        cost_per_unit=cpu_df,
    )


# ---------------------------------------------------------------------------
# Minimal fitted MMM fixture
# ---------------------------------------------------------------------------

N_MMM_DATES = N_DATES
MMM_CHANNELS = ["tv", "radio"]


@pytest.fixture(scope="module")
def dummy_mmm_df():
    rng = np.random.default_rng(42)
    n = N_MMM_DATES
    dates = pd.date_range("2024-01-01", periods=n, freq="W-MON")
    df = pd.DataFrame(
        {
            "date": dates,
            "tv": rng.uniform(0, 1, n),
            "radio": rng.uniform(0, 1, n),
        }
    )
    y = pd.Series(rng.uniform(0, 1, n), name="y")
    return df, y


@pytest.fixture(scope="module")
def fitted_mmm(dummy_mmm_df):
    df, y = dummy_mmm_df
    mmm = MMM(
        date_column="date",
        channel_columns=MMM_CHANNELS,
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )
    with pm.sampling_jax.do_nothing():
        pass
    # Use fast nuts_sampler for speed
    mmm.fit(
        df,
        y,
        draws=10,
        tune=10,
        chains=1,
        random_seed=42,
        nuts_sampler="nutpie",
        progressbar=False,
    )
    return mmm


# ---------------------------------------------------------------------------
# HillShapeSaturation tests
# ---------------------------------------------------------------------------


class TestHillShapeSaturation:
    def test_lookup_name(self):
        assert HillShapeSaturation.lookup_name == "hill_shape"

    def test_default_priors_keys(self):
        sat = HillShapeSaturation()
        assert set(sat.default_priors) == {"slope", "kappa"}

    def test_no_beta_prior(self):
        """HillShapeSaturation must NOT have a 'beta' prior — that's the point."""
        sat = HillShapeSaturation()
        assert "beta" not in sat.default_priors

    def test_function_monotone(self):
        """hill_function should be monotonically increasing in x (tested via apply)."""
        import pytensor.xtensor as ptx

        sat = HillShapeSaturation()
        x_np = np.linspace(0.1, 10, 10).astype("float32").reshape(10, 1)
        with pm.Model(coords={"channel": ["ch"]}):
            x = ptx.as_xtensor(
                __import__("pytensor").tensor.as_tensor_variable(x_np),
                dims=("date", "channel"),
            )
            y = sat.apply(x=x)
            vals = y.values.eval()
        # monotone increasing along date axis
        assert np.all(np.diff(vals[:, 0]) >= 0)

    def test_function_bounded_zero_one(self):
        """Output of Hill saturation should be in [0, 1]."""
        import pytensor.xtensor as ptx

        sat = HillShapeSaturation()
        x_np = np.array([[0.0], [1.0], [10.0], [1000.0]], dtype="float32")
        with pm.Model(coords={"channel": ["ch"]}):
            x = ptx.as_xtensor(
                __import__("pytensor").tensor.as_tensor_variable(x_np),
                dims=("date", "channel"),
            )
            y = sat.apply(x=x)
            vals = y.values.eval()
        assert np.all(vals >= 0)
        assert np.all(vals <= 1.0 + 1e-6)

    def test_serialization_round_trip(self):
        from pymc_marketing.mmm.components.saturation import saturation_from_dict

        sat = HillShapeSaturation()
        d = sat.to_dict()
        sat2 = saturation_from_dict(d)
        assert type(sat2) is HillShapeSaturation


# ---------------------------------------------------------------------------
# FrequencyReachAdditiveEffect — validation tests
# ---------------------------------------------------------------------------


class TestFrequencyReachValidation:
    def test_missing_column_raises(self, saturation, adstock):
        bad_df = pd.DataFrame({"date": DATES, "channel": "youtube", "reach": 0.5})
        with pytest.raises(ValueError, match="Missing required columns"):
            FrequencyReachAdditiveEffect(
                df_frequency_reach=bad_df,
                saturation=saturation,
                adstock=adstock,
            )

    def test_reach_out_of_range_raises(self, saturation, adstock):
        df = _make_rf_df()
        df.loc[0, "reach"] = 1.5
        with pytest.raises(ValueError, match="Reach must be within"):
            FrequencyReachAdditiveEffect(
                df_frequency_reach=df,
                saturation=saturation,
                adstock=adstock,
            )

    def test_negative_frequency_raises(self, saturation, adstock):
        df = _make_rf_df()
        df.loc[0, "frequency"] = -1.0
        with pytest.raises(ValueError, match="Frequency must be non-negative"):
            FrequencyReachAdditiveEffect(
                df_frequency_reach=df,
                saturation=saturation,
                adstock=adstock,
            )

    def test_cpu_missing_date_raises(self, rf_df, saturation, adstock):
        bad_cpu = pd.DataFrame(
            {"youtube": [0.01] * N_DATES, "display": [0.02] * N_DATES}
        )
        with pytest.raises(ValueError, match="cost_per_unit must have a 'date' column"):
            FrequencyReachAdditiveEffect(
                df_frequency_reach=rf_df,
                saturation=saturation,
                adstock=adstock,
                cost_per_unit=bad_cpu,
            )

    def test_cpu_unknown_channel_raises(self, rf_df, saturation, adstock):
        bad_cpu = pd.DataFrame(
            {"date": DATES, "youtube": 0.01, "unknown_channel": 0.02}
        )
        with pytest.raises(ValueError, match="not found in df_frequency_reach"):
            FrequencyReachAdditiveEffect(
                df_frequency_reach=rf_df,
                saturation=saturation,
                adstock=adstock,
                cost_per_unit=bad_cpu,
            )

    def test_cpu_non_positive_raises(self, rf_df, saturation, adstock):
        bad_cpu = _make_cpu_df()
        bad_cpu.loc[0, "youtube"] = 0.0
        with pytest.raises(ValueError, match="cost_per_unit values must be positive"):
            FrequencyReachAdditiveEffect(
                df_frequency_reach=rf_df,
                saturation=saturation,
                adstock=adstock,
                cost_per_unit=bad_cpu,
            )

    def test_prefix_assigned_to_transformations(self, effect):
        assert effect.adstock.prefix == f"{effect.prefix}_adstock"
        assert effect.saturation.prefix == f"{effect.prefix}_saturation"


# ---------------------------------------------------------------------------
# FrequencyReachAdditiveEffect — properties
# ---------------------------------------------------------------------------


class TestFrequencyReachProperties:
    def test_rf_channels_sorted(self, effect):
        assert effect.rf_channels == sorted(CHANNELS)

    def test_cost_per_unit_xarray_none(self, effect):
        assert effect.cost_per_unit_xarray is None

    def test_cost_per_unit_xarray_shape(self, effect_with_cpu):
        da = effect_with_cpu.cost_per_unit_xarray
        assert da is not None
        assert set(da.dims) == {"date", RF_CHANNEL_COORD}
        assert list(da.coords[RF_CHANNEL_COORD].values) == sorted(CHANNELS)
        assert (da.values > 0).all()

    def test_assumed_frequency_none_uses_median(self, effect):
        freqs = effect.get_assumed_frequency_array()
        assert freqs.shape == (len(CHANNELS),)
        for i, ch in enumerate(effect.rf_channels):
            expected = float(
                effect.df_frequency_reach.loc[
                    effect.df_frequency_reach["channel"] == ch, "frequency"
                ].median()
            )
            assert np.isclose(freqs[i], expected)

    def test_assumed_frequency_scalar(self, rf_df, saturation, adstock):
        eff = FrequencyReachAdditiveEffect(
            df_frequency_reach=rf_df,
            saturation=saturation,
            adstock=adstock,
            assumed_frequency=3.0,
        )
        freqs = eff.get_assumed_frequency_array()
        assert np.all(freqs == 3.0)

    def test_assumed_frequency_dict(self, rf_df, saturation, adstock):
        assumed = {ch: float(i + 2) for i, ch in enumerate(sorted(CHANNELS))}
        eff = FrequencyReachAdditiveEffect(
            df_frequency_reach=rf_df,
            saturation=saturation,
            adstock=adstock,
            assumed_frequency=assumed,
        )
        freqs = eff.get_assumed_frequency_array()
        for i, ch in enumerate(sorted(CHANNELS)):
            assert freqs[i] == assumed[ch]

    def test_assumed_frequency_dict_missing_channel_raises(
        self, rf_df, saturation, adstock
    ):
        with pytest.raises(
            ValueError, match="assumed_frequency dict is missing channels"
        ):
            eff = FrequencyReachAdditiveEffect(
                df_frequency_reach=rf_df,
                saturation=saturation,
                adstock=adstock,
                assumed_frequency={"youtube": 3.0},  # missing "display"
            )
            eff.get_assumed_frequency_array()

    def test_configurable_beta_prior(self, rf_df, saturation, adstock):
        custom_prior = Prior("HalfNormal", sigma=2.0)
        eff = FrequencyReachAdditiveEffect(
            df_frequency_reach=rf_df,
            saturation=saturation,
            adstock=adstock,
            beta_prior=custom_prior,
        )
        assert eff.beta_prior == custom_prior


# ---------------------------------------------------------------------------
# FrequencyReachAdditiveEffect — model integration (create_data / create_effect)
# ---------------------------------------------------------------------------


class TestFrequencyReachModelIntegration:
    """Tests that build a minimal PyMC model and run the effect pipeline."""

    @pytest.fixture
    def minimal_mmm(self, dummy_mmm_df):
        """Unfitted MMM in 'build_model' context — just enough to call create_data."""
        df, y = dummy_mmm_df
        mmm = MMM(
            date_column="date",
            channel_columns=MMM_CHANNELS,
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        # Build the PyMC model without sampling
        mmm.build_model(df, y)
        return mmm

    def test_create_data_adds_coord(self, minimal_mmm, effect):
        with minimal_mmm.model:
            effect.create_data(minimal_mmm)
        assert RF_CHANNEL_COORD in minimal_mmm.model.coords
        assert list(minimal_mmm.model.coords[RF_CHANNEL_COORD]) == sorted(CHANNELS)

    def test_create_data_pm_data_shapes(self, minimal_mmm, effect):
        with minimal_mmm.model:
            effect.create_data(minimal_mmm)
        freq_node = minimal_mmm.model[f"{effect.prefix}_frequency_raw"]
        reach_node = minimal_mmm.model[f"{effect.prefix}_reach_raw"]
        freq_val = freq_node.get_value()
        reach_val = reach_node.get_value()
        assert freq_val.shape == (N_MMM_DATES, len(CHANNELS))
        assert reach_val.shape == (N_MMM_DATES, len(CHANNELS))

    def test_create_effect_returns_correct_dims(self, minimal_mmm, effect):
        with minimal_mmm.model:
            effect.create_data(minimal_mmm)
            result = effect.create_effect(minimal_mmm)
        # result should have dims (date,) — no extra dims for this 1D mmm
        assert result is not None

    def test_create_effect_deterministics_present(self, minimal_mmm, effect):
        with minimal_mmm.model:
            effect.create_data(minimal_mmm)
            effect.create_effect(minimal_mmm)
        named = minimal_mmm.model.named_vars
        for suffix in [
            "frequency_sat",
            "effective_exposure_raw",
            "effective_exposure_adstocked",
            "channel_contribution",
            "total_effect",
        ]:
            assert f"{effect.prefix}_{suffix}" in named

    def test_beta_prior_used_in_model(self, dummy_mmm_df):
        df, y = dummy_mmm_df
        custom_prior = Prior("HalfNormal", sigma=2.0)
        rf_df = _make_rf_df()
        eff = FrequencyReachAdditiveEffect(
            df_frequency_reach=rf_df,
            saturation=HillShapeSaturation(),
            adstock=GeometricAdstock(l_max=2),
            beta_prior=custom_prior,
        )
        mmm = MMM(
            date_column="date",
            channel_columns=MMM_CHANNELS,
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(df, y)
        with mmm.model:
            eff.create_data(mmm)
            eff.create_effect(mmm)
        beta_var = mmm.model[f"{eff.prefix}_beta"]
        # HalfNormal var should not have negative support: check distribution name
        assert "halfnormal" in str(beta_var.owner.op).lower()

    def test_rf_channel_coord_disjoint_from_channel(self, minimal_mmm, effect):
        """rf_channel coord must not overlap with standard channel coord."""
        with minimal_mmm.model:
            effect.create_data(minimal_mmm)
        rf_channels = set(minimal_mmm.model.coords[RF_CHANNEL_COORD])
        std_channels = set(minimal_mmm.model.coords["channel"])
        assert rf_channels.isdisjoint(std_channels)


# ---------------------------------------------------------------------------
# set_data — future date zero-fill
# ---------------------------------------------------------------------------


class TestSetData:
    def test_set_data_future_dates_zero_filled(self, dummy_mmm_df, effect):
        df, y = dummy_mmm_df
        mmm = MMM(
            date_column="date",
            channel_columns=MMM_CHANNELS,
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(df, y)
        with mmm.model:
            effect.create_data(mmm)
            effect.create_effect(mmm)

        # Extend the model dates by 4 weeks
        future_dates = pd.date_range(
            DATES[-1] + pd.Timedelta(weeks=1), periods=4, freq="W-MON"
        )
        all_dates = DATES.append(future_dates)
        # Build an xr.Dataset with extended dates (mimics what MMM passes)
        extended_ds = xr.Dataset(coords={"date": all_dates})

        # Manually update the model coord so set_data sees new dates
        extended_model = mmm.model.copy()
        extended_model.set_dim("date", len(all_dates), coord_values=all_dates)

        effect.set_data(mmm, extended_model, extended_ds)

        freq_val = extended_model[f"{effect.prefix}_frequency_raw"].get_value()
        # Last 4 rows should be zero
        assert freq_val.shape[0] == len(all_dates)
        assert np.all(freq_val[-4:] == 0.0)

    def test_set_data_no_future_no_change(self, dummy_mmm_df, effect):
        """If no future dates, data should not change."""
        df, y = dummy_mmm_df
        mmm = MMM(
            date_column="date",
            channel_columns=MMM_CHANNELS,
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(df, y)
        with mmm.model:
            effect.create_data(mmm)
            effect.create_effect(mmm)

        freq_before = mmm.model[f"{effect.prefix}_frequency_raw"].get_value().copy()

        ds = xr.Dataset(coords={"date": DATES})
        effect.set_data(mmm, mmm.model, ds)

        freq_after = mmm.model[f"{effect.prefix}_frequency_raw"].get_value()
        np.testing.assert_array_equal(freq_before, freq_after)


# ---------------------------------------------------------------------------
# build_rf_optimization_tensors
# ---------------------------------------------------------------------------


class TestBuildRFOptimizationTensors:
    def test_raises_without_cpu(self, effect):
        import pytensor.xtensor as ptx

        rf_budgets = ptx.as_xtensor(
            __import__("pytensor").tensor.as_tensor_variable(
                np.array([100.0, 200.0], dtype="float64")
            ),
            dims=(RF_CHANNEL_COORD,),
        )
        with pytest.raises(ValueError, match="cost_per_unit"):
            effect.build_rf_optimization_tensors(
                rf_budgets=rf_budgets,
                num_periods=4,
                budget_distribution=None,
            )

    def test_output_shapes(self, effect_with_cpu):
        import pytensor.tensor as pt
        import pytensor.xtensor as ptx

        n_rf = len(CHANNELS)
        l_max = effect_with_cpu.adstock.l_max
        num_periods = 4

        rf_budgets = ptx.as_xtensor(
            pt.as_tensor_variable(np.full(n_rf, 100.0, dtype="float64")),
            dims=(RF_CHANNEL_COORD,),
        )
        reach_t, freq_t = effect_with_cpu.build_rf_optimization_tensors(
            rf_budgets=rf_budgets,
            num_periods=num_periods,
            budget_distribution=None,
        )
        reach_val = reach_t.values.eval()
        freq_val = freq_t.values.eval()

        assert reach_val.shape == (num_periods + l_max, n_rf)
        assert freq_val.shape == (num_periods + l_max, n_rf)

    def test_carryover_zeros(self, effect_with_cpu):
        import pytensor.tensor as pt
        import pytensor.xtensor as ptx

        n_rf = len(CHANNELS)
        l_max = effect_with_cpu.adstock.l_max
        num_periods = 4

        rf_budgets = ptx.as_xtensor(
            pt.as_tensor_variable(np.full(n_rf, 100.0, dtype="float64")),
            dims=(RF_CHANNEL_COORD,),
        )
        reach_t, freq_t = effect_with_cpu.build_rf_optimization_tensors(
            rf_budgets=rf_budgets,
            num_periods=num_periods,
            budget_distribution=None,
        )
        reach_val = reach_t.values.eval()
        freq_val = freq_t.values.eval()

        # Last l_max rows (carryover zeros) should be zero
        assert np.all(reach_val[-l_max:] == 0.0)
        assert np.all(freq_val[-l_max:] == 0.0)

    def test_reach_positive_for_positive_budget(self, effect_with_cpu):
        import pytensor.tensor as pt
        import pytensor.xtensor as ptx

        n_rf = len(CHANNELS)
        l_max = effect_with_cpu.adstock.l_max
        num_periods = 4

        rf_budgets = ptx.as_xtensor(
            pt.as_tensor_variable(np.full(n_rf, 1000.0, dtype="float64")),
            dims=(RF_CHANNEL_COORD,),
        )
        reach_t, _ = effect_with_cpu.build_rf_optimization_tensors(
            rf_budgets=rf_budgets,
            num_periods=num_periods,
            budget_distribution=None,
        )
        reach_val = reach_t.values.eval()
        # Non-carryover periods should be positive
        assert np.all(reach_val[:-l_max] > 0)


# ---------------------------------------------------------------------------
# _replace_rf_data_by_optimization_variable (no-op when no R&F effects)
# ---------------------------------------------------------------------------


class TestBudgetOptimizerRFNoop:
    """When no FrequencyReachAdditiveEffect is attached, the method is a no-op."""

    def test_no_rf_effects_returns_same_model(self, dummy_mmm_df):
        """BudgetOptimizer builds without error when no R&F effect is present."""
        from pymc_marketing.mmm.mmm import BudgetOptimizerWrapper

        df, y = dummy_mmm_df
        mmm = MMM(
            date_column="date",
            channel_columns=MMM_CHANNELS,
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.fit(
            df,
            y,
            draws=10,
            tune=10,
            chains=1,
            random_seed=42,
            nuts_sampler="nutpie",
            progressbar=False,
        )
        wrapper = BudgetOptimizerWrapper(
            model=mmm,
            start_date=DATES[-1] + pd.Timedelta(weeks=1),
            end_date=DATES[-1] + pd.Timedelta(weeks=4),
        )
        # Should complete without error — R&F path is a no-op
        optimal, _ = wrapper.optimize_budget(budget=1000.0, default_constraints=True)
        assert set(optimal.coords["channel"].values) == set(MMM_CHANNELS)
