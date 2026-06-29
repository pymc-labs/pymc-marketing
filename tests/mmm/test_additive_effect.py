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
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm import MMM, BudgetOptimizerWrapper
from pymc_marketing.mmm.additive_effect import (
    DiscountedEventEffect,
    EventAdditiveEffect,
    FourierEffect,
    LinearTrendEffect,
    MuEffect,
    OptimizableMuEffect,
)
from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer
from pymc_marketing.mmm.fourier import MonthlyFourier, WeeklyFourier, YearlyFourier
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.serialization import DeserializationContext, serialization


@pytest.fixture(scope="function")
def create_mock_mmm():
    class MMM:
        pass

    def func(dims, model):
        mmm = MMM()

        mmm.dims = dims
        mmm.model = model

        return mmm

    return func


@pytest.fixture(scope="function")
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=52, freq="W-MON", name="date")


@pytest.fixture(scope="function")
def new_dates(dates) -> pd.DatetimeIndex:
    last_date = dates.max()

    return pd.date_range(
        last_date + pd.Timedelta(days=7),
        periods=26,
        freq="W-MON",
        name="date",
    )


def set_new_model_dates(dates):
    # Just changing the coordinates of the model
    model = pm.modelcontext(None)
    model.set_dim("date", len(dates), coord_values=dates)


@pytest.fixture(scope="function")
def create_fourier_model(dates):
    def create_model(coords) -> pm.Model:
        coords = coords | {"date": dates}
        return pm.Model(coords=coords)

    return create_model


@pytest.mark.parametrize(
    "fourier",
    [
        WeeklyFourier(n_order=10, prefix="weekly"),
        MonthlyFourier(n_order=10, prefix="monthly"),
        YearlyFourier(n_order=10, prefix="yearly"),
    ],
    ids=["weekly", "monthly", "yearly"],
)
@pytest.mark.parametrize(
    "dims, coords",
    [
        ((), {}),
        (("geo",), {"geo": ["A", "B"]}),
    ],
    ids=["no_dims", "with_dims"],
)
def test_fourier_effect(
    create_mock_mmm,
    new_dates,
    create_fourier_model,
    fourier,
    dims,
    coords,
) -> None:
    effect = FourierEffect(fourier=fourier)

    mmm = create_mock_mmm(
        dims=dims,
        model=create_fourier_model(coords=coords),
    )

    with mmm.model:
        effect.create_data(mmm)

    assert set(mmm.model.named_vars) == set([f"{fourier.prefix}_day"])
    assert set(mmm.model.coords) == {"date", *dims}

    with mmm.model:
        # Should just be broadcastable with target.
        # Not necessarily the same shape
        created_variable = effect.create_effect(mmm)

    assert set(created_variable.dims) - set(mmm.dims) == {"date"}

    # Variables created: data, beta coefficients, raw components (per mode), final contribution
    assert set(mmm.model.named_vars) == {
        f"{fourier.prefix}_day",
        f"{fourier.prefix}_beta",
        f"{fourier.prefix}_components",
        f"{fourier.prefix}_contribution",
    }
    assert set(mmm.model.coords) == {"date", *dims, fourier.prefix}

    with mmm.model:
        idata = pm.sample_prior_predictive()
        set_new_model_dates(new_dates)
        effect.set_data(mmm, mmm.model, None)

        idata.update(
            pm.sample_posterior_predictive(
                idata.prior,
                var_names=[f"{fourier.prefix}_contribution"],
            ),
        )

    effect_predictions = idata.posterior_predictive[f"{fourier.prefix}_contribution"]
    np.testing.assert_allclose(effect_predictions.notnull().mean().item(), 1.0)
    pd.testing.assert_index_equal(effect_predictions.date.to_index(), new_dates)


@pytest.mark.parametrize(
    "prior_dims",
    [
        (),
        ("weekly",),
        ("weekly", "geo"),
        ("geo", "weekly"),
    ],
    ids=["no-dims", "exclude", "include", "include_reverse"],
)
def test_fourier_effect_multidimensional(
    create_mock_mmm,
    create_fourier_model,
    prior_dims,
) -> None:
    mmm = create_mock_mmm(
        dims=("geo",),
        model=create_fourier_model(coords={"geo": ["A", "B"]}),
    )

    prefix = "weekly"
    prior = Prior("Laplace", mu=0, b=0.1, dims=prior_dims)
    fourier = WeeklyFourier(n_order=10, prefix=prefix, prior=prior)
    fourier_effect = FourierEffect(fourier=fourier)

    with mmm.model:
        fourier_effect.create_data(mmm)
        effect = fourier_effect.create_effect(mmm)
        pm.sample_prior_predictive()

    assert set(effect.dims) == ({"date", *prior_dims} - {"weekly"})


@pytest.mark.parametrize(
    "fourier_cls,prefix",
    [
        (WeeklyFourier, "weekly"),
        (MonthlyFourier, "monthly"),
        (YearlyFourier, "yearly"),
    ],
    ids=["weekly", "monthly", "yearly"],
)
def test_fourier_components_sum_to_contribution(
    create_mock_mmm, create_fourier_model, fourier_cls, prefix
):
    """Ensure <prefix>_contribution is the sum over the internal fourier components.

    The additive effect should expose:
      - <prefix>_components : (date, fourier[, extra dims])
      - <prefix>_contribution : (date[, extra dims]) == sum_{fourier} components
    """
    fourier = fourier_cls(n_order=4, prefix=prefix)
    effect = FourierEffect(fourier=fourier)

    mmm = create_mock_mmm(dims=(), model=create_fourier_model(coords={}))

    with mmm.model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
        idata = pm.sample_prior_predictive(draws=5)

    components = idata.prior[f"{prefix}_components"]  # dims: chain, draw, date, fourier
    contribution = idata.prior[f"{prefix}_contribution"]  # dims: chain, draw, date

    summed = components.sum(dim=prefix)

    # Align dims just in case ordering differs (should not) and compare
    xr.testing.assert_allclose(summed, contribution)


@pytest.fixture(scope="function")
def linear_trend_model(dates) -> pm.Model:
    coords = {"date": dates}
    return pm.Model(coords=coords)


@pytest.mark.parametrize(
    "mmm_dims, priors, linear_trend_dims, deterministic_dims",
    [
        pytest.param((), {}, (), ("date",), id="scalar"),
        pytest.param(
            ("geo", "product"),
            {},
            ("geo", "product"),
            ("date", None, None),
            id="2d",
        ),
        pytest.param(
            ("geo", "product"),
            {"delta": Prior("Normal", dims=("geo", "changepoint"))},
            ("geo", "product"),
            ("date", None, None),
            id="missing-product-dim-in-delta",
        ),
    ],
)
def test_linear_trend_effect(
    create_mock_mmm,
    new_dates,
    linear_trend_model,
    mmm_dims,
    priors,
    linear_trend_dims,
    deterministic_dims,
) -> None:
    prefix = "linear_trend"
    effect = LinearTrendEffect(
        trend=LinearTrend(priors=priors, dims=linear_trend_dims),
        prefix=prefix,
    )

    mmm = create_mock_mmm(dims=mmm_dims, model=linear_trend_model)

    mmm.model.add_coords({dim: ["dummy1", "dummy2"] for dim in linear_trend_dims})

    with mmm.model:
        effect.create_data(mmm)

    assert set(mmm.model.named_vars) == {f"{prefix}_t"}
    assert set(mmm.model.coords) == {"date"}.union(linear_trend_dims)
    assert effect.linear_trend_first_date == mmm.model.coords["date"][0]

    with mmm.model:
        pmd.Deterministic(
            "effect",
            effect.create_effect(mmm),
        )

    assert set(mmm.model.named_vars) == {
        "delta",
        "effect",
        f"{prefix}_effect_contribution",
        f"{prefix}_t",
    }
    assert set(mmm.model.coords) == {"date", "changepoint"}.union(linear_trend_dims)

    with mmm.model:
        idata = pm.sample_prior_predictive()
        set_new_model_dates(new_dates)
        effect.set_data(mmm, mmm.model, None)

        idata.update(
            pm.sample_posterior_predictive(
                idata.prior,
                var_names=["effect"],
            ),
        )

    effect_predictions = idata.posterior_predictive.effect
    np.testing.assert_allclose(effect_predictions.notnull().mean().item(), 1.0)
    pd.testing.assert_index_equal(effect_predictions.date.to_index(), new_dates)


class TestMuEffectRoundtrips:
    @pytest.mark.parametrize(
        "fourier_cls_name",
        ["YearlyFourier", "MonthlyFourier", "WeeklyFourier"],
    )
    def test_fourier_effect_all_fourier_types(self, fourier_cls_name):
        import pymc_marketing.mmm.fourier as fourier_mod

        fourier_cls = getattr(fourier_mod, fourier_cls_name)
        original = FourierEffect(
            fourier=fourier_cls(
                n_order=5,
                prefix="custom_fourier",
                prior=Prior("Laplace", mu=0.5, b=2.0),
            ),
            date_dim_name="custom_date",
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is FourierEffect
        assert type(restored.fourier) is fourier_cls
        assert restored.fourier.n_order == 5
        assert restored.fourier.prefix == "custom_fourier"
        assert restored.date_dim_name == "custom_date"
        assert restored == original

    def test_linear_trend_effect_all_parameters(self):
        original = LinearTrendEffect(
            trend=LinearTrend(
                n_changepoints=8,
                include_intercept=True,
                priors={
                    "delta": Prior("Laplace", mu=0, b=0.5, dims="changepoint"),
                    "k": Prior("Normal", mu=0.1, sigma=0.1),
                },
            ),
            prefix="custom_trend",
            date_dim_name="custom_date",
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is LinearTrendEffect
        assert restored.prefix == "custom_trend"
        assert restored.date_dim_name == "custom_date"
        assert type(restored.trend) is LinearTrend
        assert restored.trend.n_changepoints == 8
        assert restored.trend.include_intercept is True
        assert restored == original

    def test_custom_mu_effect_roundtrip(self):
        """User-defined MuEffect subclasses auto-register and round-trip."""
        from pymc_marketing.mmm.additive_effect import MuEffect

        class UserEffect(MuEffect):
            my_param: float = 3.14
            my_str: str = "default"

            def create_data(self, mmm):
                pass

            def create_effect(self, mmm):
                pass

            def set_data(self, mmm, model, X):
                pass

        original = UserEffect(my_param=2.71, my_str="custom_value")
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is UserEffect
        assert restored.my_param == 2.71
        assert restored.my_str == "custom_value"
        assert restored == original


class TestEventAdditiveEffectRoundtrips:
    def test_to_dict_serializes_all_fields(self):
        from pymc_marketing.mmm.additive_effect import EventAdditiveEffect
        from pymc_marketing.mmm.events import EventEffect, GaussianBasis

        df = pd.DataFrame(
            {
                "name": ["event1"],
                "start_date": ["2024-01-01"],
                "end_date": ["2024-01-07"],
            }
        )
        basis = GaussianBasis(
            prefix="ev_basis",
            priors={"sigma": Prior("Gamma", mu=5, sigma=2)},
        )
        effect = EventEffect(
            basis=basis,
            effect_size=Prior("Normal", mu=0.5, sigma=2.0),
            dims="custom_events",
        )
        eae = EventAdditiveEffect(
            df_events=df,
            prefix="custom_events",
            effect=effect,
            reference_date="2024-06-01",
            date_dim_name="custom_date",
        )
        data = eae.to_dict()
        assert "__type__" in data
        assert "effect" in data
        assert "df_events_group" in data
        assert data["df_events_group"] == "supplementary_data_custom_events"
        assert data["prefix"] == "custom_events"
        assert data["reference_date"] == "2024-06-01"
        assert data["date_dim_name"] == "custom_date"

    def test_roundtrip_all_parameters_with_mock_context(self):
        import xarray as xr

        from pymc_marketing.mmm.additive_effect import EventAdditiveEffect
        from pymc_marketing.mmm.events import EventEffect, GaussianBasis

        df = pd.DataFrame(
            {
                "name": ["ev1", "ev2", "ev3"],
                "start_date": ["2024-01-01", "2024-06-01", "2024-12-01"],
                "end_date": ["2024-01-07", "2024-06-07", "2024-12-07"],
            }
        )
        basis = GaussianBasis(
            prefix="ev_basis",
            priors={"sigma": Prior("Gamma", mu=5, sigma=2)},
        )
        effect = EventEffect(
            basis=basis,
            effect_size=Prior("Normal", mu=0.5, sigma=2.0),
            dims="custom_events",
        )
        original = EventAdditiveEffect(
            df_events=df,
            prefix="custom_events",
            effect=effect,
            reference_date="2024-06-01",
            date_dim_name="custom_date",
        )

        data = serialization.serialize(original)

        ds = xr.Dataset.from_dataframe(df.set_index("name"))
        fake_idata_dict = {"supplementary_data_custom_events": ds}

        class MockIdata:
            def __getitem__(self, key):
                return fake_idata_dict[key]

        ctx = DeserializationContext(idata=MockIdata())
        restored = serialization.deserialize(data, context=ctx)

        assert type(restored) is EventAdditiveEffect
        assert len(restored.df_events) == 3
        assert restored.prefix == "custom_events"
        assert restored.reference_date == "2024-06-01"
        assert restored.date_dim_name == "custom_date"
        assert type(restored.effect) is EventEffect
        assert type(restored.effect.basis) is GaussianBasis


@pytest.mark.parametrize(
    "type_key",
    [
        "pymc_marketing.mmm.additive_effect.FourierEffect",
        "pymc_marketing.mmm.additive_effect.LinearTrendEffect",
        "pymc_marketing.mmm.additive_effect.EventAdditiveEffect",
    ],
    ids=lambda s: s.rsplit(".", 1)[-1],
)
def test_additive_effect_type_registered(type_key):
    assert type_key in serialization._registry, f"{type_key} not registered"


# ---------------------------------------------------------------------------
# OptimizableMuEffect
# ---------------------------------------------------------------------------

# A minimal concrete OptimizableMuEffect used only in the tests below.
# It stores fixed data (zeros) in a pmd.Data node and replaces it during
# optimization with spend / cost_per_unit distributed uniformly over time.
#
# NOTE: All three MuEffect abstract methods (create_data, create_effect,
# set_data) follow exactly the same patterns as FourierEffect and
# EventAdditiveEffect: they use the mmm object, pmd.Data, and pmd.Deterministic
# — no raw pm.Model construction.


class _DummyOptimizableEffect(OptimizableMuEffect):
    """Concrete OptimizableMuEffect for testing.

    Registers a ``(date, rf_channel)`` pmd.Data node and returns the
    channel-sum as its effect contribution.  Designed to be added to a
    real MMM via ``mmm.add_mu_effect(...)`` before ``build_model``.
    """

    rf_channels: list[str]
    cost_per_unit: float = 1.0

    @property
    def contribution_var_name(self) -> str:
        return "dummy_rf_contribution"

    @property
    def budget_dim(self) -> str:
        return "rf_channel"

    def create_data(self, mmm) -> None:
        import pymc.dims as pmd_local

        n = len(self.rf_channels)
        n_dates = len(mmm.model.coords["date"])
        mmm.model.add_coord("rf_channel", self.rf_channels)
        pmd_local.Data(
            "dummy_rf_impressions",
            np.zeros((n_dates, n), dtype="float64"),
            dims=("date", "rf_channel"),
        )

    def create_effect(self, mmm) -> XTensorVariable:
        import pymc.dims as pmd_local

        data = mmm.model["dummy_rf_impressions"]  # (date, rf_channel)
        return pmd_local.Deterministic(
            "dummy_rf_contribution",
            data.sum(dim="rf_channel"),
        )

    def set_data(self, mmm, model, X) -> None:
        n = len(self.rf_channels)
        n_dates = len(model.coords["date"])
        model["dummy_rf_impressions"].set_value(np.zeros((n_dates, n), dtype="float64"))

    def replace_for_optimization(
        self,
        budget_slice,
        num_periods: int,
        budget_distribution,
    ) -> dict:
        import pytensor.tensor as pt
        import pytensor.xtensor as ptx

        impressions = budget_slice / self.cost_per_unit
        ones_date = ptx.as_xtensor(pt.ones((num_periods,)), dims=("date",))
        return {"dummy_rf_impressions": ones_date * impressions}

    @property
    def budget_channel_names(self) -> list[str]:
        return list(self.rf_channels)

    def set_budget_for_sampling(self, budget_per_item, model) -> None:
        n = len(self.rf_channels)
        n_dates = len(model.coords["date"])
        impressions = np.full((n_dates, n), budget_per_item / self.cost_per_unit)
        pm.set_data({"dummy_rf_impressions": impressions}, model=model)


@pytest.fixture(scope="module")
def dummy_mmm_df() -> tuple[dict, pd.DataFrame, pd.Series, list[str]]:
    """Minimal DataFrame + kwargs for MMM.build_model in OptimizableMuEffect tests."""
    n = 10
    channels = ["std_1", "std_2"]
    df = pd.DataFrame(
        {
            "date_week": pd.date_range("2024-01-01", periods=n, freq="W"),
            "std_1": np.linspace(0, 1, n),
            "std_2": np.linspace(0, 1, n),
        }
    )
    y = pd.Series(np.ones(n), name="y")
    df_kwargs = {
        "date_column": "date_week",
        "channel_columns": channels,
    }
    return df_kwargs, df, y, channels


@pytest.fixture(scope="module")
def dummy_mmm_idata(dummy_mmm_df: pd.DataFrame):
    """idata with all model parameters needed by BudgetOptimizer.

    Mirrors the pattern from test_budget_optimizer.py::dummy_idata.
    """
    _df_kwargs, _df, _y, channels = dummy_mmm_df
    n_dates = 2
    return az.from_dict(
        posterior={
            "saturation_lam": [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
            "saturation_beta": [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]],
            "adstock_alpha": [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]],
            "channel_contribution": np.ones((2, 2, len(channels), n_dates)),
        },
        coords={
            "chain": [0, 1],
            "draw": [0, 1],
            "channel": channels,
            "date": list(range(n_dates)),
        },
        dims={
            "saturation_lam": ["chain", "draw", "channel"],
            "saturation_beta": ["chain", "draw", "channel"],
            "adstock_alpha": ["chain", "draw", "channel"],
            "channel_contribution": ["chain", "draw", "channel", "date"],
        },
    )


def _build_budget_wrapper(mmm, df, start_date="2024-03-10", end_date="2024-04-21"):
    """Return a BudgetOptimizerWrapper for the given MMM over a short window."""
    return BudgetOptimizerWrapper(model=mmm, start_date=start_date, end_date=end_date)


class TestOptimizableMuEffect:
    """Tests for the OptimizableMuEffect abstract base class and its
    integration with BudgetOptimizer via a real MMM."""

    def test_cannot_instantiate_directly(self):
        """OptimizableMuEffect is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            OptimizableMuEffect()

    def test_missing_replace_for_optimization_raises(self):
        """Subclass missing replace_for_optimization cannot be instantiated."""

        class MissingReplace(OptimizableMuEffect):
            @property
            def optimizable_channel_names(self) -> list[str]:
                return []

            def create_data(self, mmm) -> None:
                pass

            def create_effect(self, mmm) -> XTensorVariable:
                return ptx.xtensor("dummy", shape=(1,), dims=("date",))

            def set_data(self, mmm, model, X) -> None:
                pass

        with pytest.raises(TypeError, match="abstract"):
            MissingReplace()

    @pytest.mark.parametrize(
        "effect_cls",
        [FourierEffect, LinearTrendEffect, EventAdditiveEffect],
        ids=["FourierEffect", "LinearTrendEffect", "EventAdditiveEffect"],
    )
    def test_time_based_effects_are_not_optimizable(self, effect_cls):
        """Time-based MuEffects must NOT inherit OptimizableMuEffect."""
        assert not issubclass(effect_cls, OptimizableMuEffect)
        assert issubclass(effect_cls, MuEffect)

    def test_budgets_flat_extended_for_optimizable_effect(
        self, dummy_mmm_df, dummy_mmm_idata
    ):
        """BudgetOptimizer._budgets_flat size equals n_standard + n_effect channels.

        Uses a real MMM with a _DummyOptimizableEffect to verify that the
        optimizer correctly discovers and sizes the combined budget variable.
        """
        from pymc_marketing.mmm.components.adstock import GeometricAdstock
        from pymc_marketing.mmm.components.saturation import LogisticSaturation

        df_kwargs, X, y, std_channels = dummy_mmm_df
        rf_channels = ["rf_a", "rf_b"]
        effect = _DummyOptimizableEffect(rf_channels=rf_channels)

        mmm = MMM(
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            **df_kwargs,
        )
        mmm.add_mu_effect(effect)
        mmm.build_model(X=X, y=y)
        mmm.idata = dummy_mmm_idata

        wrapper = _build_budget_wrapper(mmm, X)
        optimizer = BudgetOptimizer(model=wrapper, num_periods=wrapper.num_periods)
        expected = len(std_channels) + len(rf_channels)
        actual = optimizer._budgets_flat.type.shape[0]
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_replace_optimizable_mueffects_calls_replace_for_optimization(
        self, dummy_mmm_df, dummy_mmm_idata
    ):
        """BudgetOptimizer calls replace_for_optimization once per effect with correct num_periods."""
        from unittest.mock import patch

        from pymc_marketing.mmm.components.adstock import GeometricAdstock
        from pymc_marketing.mmm.components.saturation import LogisticSaturation

        df_kwargs, X, y, _ = dummy_mmm_df
        effect = _DummyOptimizableEffect(rf_channels=["rf_a"])

        mmm = MMM(
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            **df_kwargs,
        )
        mmm.add_mu_effect(effect)
        mmm.build_model(X=X, y=y)
        mmm.idata = dummy_mmm_idata

        wrapper = _build_budget_wrapper(mmm, X)
        num_periods = wrapper.num_periods

        original_replace = effect.replace_for_optimization

        call_log: list[dict] = []

        def recording_replace(_self, **kwargs):
            call_log.append(kwargs)
            return original_replace(**kwargs)

        with patch.object(
            _DummyOptimizableEffect, "replace_for_optimization", recording_replace
        ):
            BudgetOptimizer(model=wrapper, num_periods=num_periods)

        assert len(call_log) == 1
        assert call_log[0]["num_periods"] == num_periods

    def test_no_optimizable_effects_noop(self, dummy_mmm_df, dummy_mmm_idata):
        """_budgets_flat size equals only the standard channel count when no OptimizableMuEffect is present."""
        from pymc_marketing.mmm.components.adstock import GeometricAdstock
        from pymc_marketing.mmm.components.saturation import LogisticSaturation

        df_kwargs, X, y, std_channels = dummy_mmm_df

        mmm = MMM(
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            **df_kwargs,
        )
        mmm.build_model(X=X, y=y)
        mmm.idata = dummy_mmm_idata

        wrapper = _build_budget_wrapper(mmm, X)
        optimizer = BudgetOptimizer(model=wrapper, num_periods=wrapper.num_periods)
        assert optimizer._budgets_flat.type.shape[0] == len(std_channels)


# ---------------------------------------------------------------------------
# DiscountedEventEffect
# ---------------------------------------------------------------------------


def _make_df_events(names=("black_friday", "summer_sale")):
    """Minimal df_events for DiscountedEventEffect tests."""
    n = len(names)
    return pd.DataFrame(
        {
            "name": list(names),
            "start_date": ["2024-11-29", "2024-07-01"][:n],
            "end_date": ["2024-12-02", "2024-07-07"][:n],
            "discount_pct": [0.20, 0.15][:n],
        }
    )


@pytest.fixture(scope="module")
def discounted_event_df():
    """DataFrame covering both promotional event windows for MMM build tests."""
    # Weekly data spanning 2024, so both event windows fall inside the training range
    n = 52
    dates = pd.date_range("2024-01-01", periods=n, freq="W")
    channels = ["std_1", "std_2"]
    df = pd.DataFrame(
        {
            "date_week": dates,
            "std_1": np.linspace(0, 1, n),
            "std_2": np.linspace(0, 1, n),
        }
    )
    y = pd.Series(np.ones(n) * 100.0, name="y")
    df_kwargs = {"date_column": "date_week", "channel_columns": channels}
    return df_kwargs, df, y, channels


@pytest.fixture(scope="module")
def discounted_event_idata(discounted_event_df):
    """Minimal idata for BudgetOptimizer construction in DiscountedEventEffect tests."""
    _df_kwargs, _df, _y, channels = discounted_event_df
    n_dates = 2
    promo_events = ["black_friday", "summer_sale"]
    n_events = len(promo_events)
    return az.from_dict(
        posterior={
            "saturation_lam": [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
            "saturation_beta": [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]],
            "adstock_alpha": [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]],
            "channel_contribution": np.ones((2, 2, len(channels), n_dates)),
            "promo_beta": np.ones((2, 2, n_events)) * 2.0,
        },
        coords={
            "chain": [0, 1],
            "draw": [0, 1],
            "channel": channels,
            "date": list(range(n_dates)),
            "promo": promo_events,
        },
        dims={
            "saturation_lam": ["chain", "draw", "channel"],
            "saturation_beta": ["chain", "draw", "channel"],
            "adstock_alpha": ["chain", "draw", "channel"],
            "channel_contribution": ["chain", "draw", "channel", "date"],
            "promo_beta": ["chain", "draw", "promo"],
        },
    )


def _build_discounted_mmm(df_kwargs, X, y, df_events):
    """Helper: build an MMM with a DiscountedEventEffect attached."""
    from pymc_marketing.mmm.components.adstock import GeometricAdstock
    from pymc_marketing.mmm.components.saturation import LogisticSaturation

    effect = DiscountedEventEffect(
        df_events=df_events,
        prefix="promo",
    )
    mmm = MMM(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )
    mmm.add_mu_effect(effect)
    mmm.build_model(X=X, y=y)
    return mmm, effect


class TestDiscountedEventEffect:
    """Tests for DiscountedEventEffect and its integration with BudgetOptimizer."""

    # ------------------------------------------------------------------
    # Contract properties
    # ------------------------------------------------------------------

    def test_is_optimizable_mu_effect(self):
        effect = DiscountedEventEffect(
            df_events=_make_df_events(),
            prefix="promo",
        )
        assert isinstance(effect, OptimizableMuEffect)
        assert isinstance(effect, MuEffect)

    def test_contribution_var_name(self):
        effect = DiscountedEventEffect(
            df_events=_make_df_events(),
            prefix="promo",
        )
        assert effect.contribution_var_name == "promo_effect_contribution"

    def test_budget_dim_defaults_to_prefix(self):
        effect = DiscountedEventEffect(
            df_events=_make_df_events(),
            prefix="promo",
        )
        assert effect.budget_dim == "promo"

    def test_not_event_additive_effect(self):
        """DiscountedEventEffect is NOT a plain EventAdditiveEffect."""
        effect = DiscountedEventEffect(
            df_events=_make_df_events(),
            prefix="promo",
        )
        assert not isinstance(effect, EventAdditiveEffect)

    @pytest.mark.parametrize(
        "df_bad, match",
        [
            (
                pd.DataFrame({"name": ["black_friday"]}),
                "start_date",
            ),
        ],
    )
    def test_missing_required_columns_raises(self, df_bad, match):
        with pytest.raises(ValueError, match=match):
            DiscountedEventEffect(df_events=df_bad, prefix="promo")

    # ------------------------------------------------------------------
    # replace_for_optimization
    # ------------------------------------------------------------------

    def test_replace_for_optimization_returns_correct_key(self):
        import pytensor.xtensor as ptx

        df_events = _make_df_events()
        effect = DiscountedEventEffect(
            df_events=df_events,
            prefix="promo",
        )
        effect._event_revenue = np.array([1_000.0, 800.0])
        budget_slice = ptx.as_xtensor(
            np.array([100.0, 200.0], dtype="float64"),
            dims=("promo",),
        )
        result = effect.replace_for_optimization(
            budget_slice=budget_slice,
            num_periods=10,
            budget_distribution=None,
        )
        assert "promo_discount_pct" in result

    def test_replace_for_optimization_spend_shape(self):
        import pytensor.xtensor as ptx

        df_events = _make_df_events()
        effect = DiscountedEventEffect(
            df_events=df_events,
            prefix="promo",
        )
        effect._event_revenue = np.array([1_000.0, 800.0])
        budget_slice = ptx.as_xtensor(
            np.array([100.0, 200.0], dtype="float64"),
            dims=("promo",),
        )
        result = effect.replace_for_optimization(
            budget_slice=budget_slice,
            num_periods=10,
            budget_distribution=None,
        )
        spend = result["promo_discount_pct"]
        assert "promo" in spend.type.dims

    # ------------------------------------------------------------------
    # Optimizer integration
    # ------------------------------------------------------------------

    def test_budgets_flat_extended_for_discounted_effect(
        self, discounted_event_df, discounted_event_idata
    ):
        """_budgets_flat size = n_std_channels + n_events."""
        df_kwargs, X, y, std_channels = discounted_event_df
        df_events = _make_df_events()

        mmm, _ = _build_discounted_mmm(df_kwargs, X, y, df_events)
        mmm.idata = discounted_event_idata

        wrapper = BudgetOptimizerWrapper(
            model=mmm, start_date="2024-06-01", end_date="2024-12-31"
        )
        optimizer = BudgetOptimizer(model=wrapper, num_periods=wrapper.num_periods)

        expected = len(std_channels) + len(df_events)
        assert optimizer._budgets_flat.type.shape[0] == expected

    def test_allocate_budget_end_to_end(
        self, discounted_event_df, discounted_event_idata
    ):
        """allocate_budget runs without error and returns sane allocations."""
        df_kwargs, X, y, _std_channels = discounted_event_df
        df_events = _make_df_events()

        mmm, _ = _build_discounted_mmm(df_kwargs, X, y, df_events)
        mmm.idata = discounted_event_idata

        wrapper = BudgetOptimizerWrapper(
            model=mmm, start_date="2024-06-01", end_date="2024-12-31"
        )
        total_budget = 1000.0
        optimizer = BudgetOptimizer(model=wrapper, num_periods=wrapper.num_periods)
        optimal, _result = optimizer.allocate_budget(
            total_budget=total_budget,
            budget_bounds=None,
        )

        # All values finite and non-negative
        assert np.all(np.isfinite(optimal.values))
        assert np.all(optimal.values >= 0)

        # Event channel names appear in the output
        for name in df_events["name"]:
            assert name in optimal.coords["channel"].values

        # Total spend does not exceed budget (constraint respected)
        assert float(optimal.values.sum()) <= total_budget * 1.01  # 1% tolerance

    def test_beta_is_free_rv(self, discounted_event_df):
        """promo_beta must appear as a free RV, one entry per event."""
        df_kwargs, X, y, _ = discounted_event_df
        df_events = _make_df_events()
        mmm, _ = _build_discounted_mmm(df_kwargs, X, y, df_events)
        free_rv_names = [rv.name for rv in mmm.model.free_RVs]
        assert any("promo_beta" in name for name in free_rv_names)

    # ------------------------------------------------------------------
    # budget_channel_names and set_budget_for_sampling
    # ------------------------------------------------------------------

    def test_budget_channel_names_returns_event_names(self):
        """budget_channel_names must exactly match df_events['name'] in order."""
        df_events = _make_df_events()
        effect = DiscountedEventEffect(df_events=df_events, prefix="promo")
        assert effect.budget_channel_names == df_events["name"].tolist()

    def test_set_budget_for_sampling_updates_discount_pct(self, discounted_event_df):
        """set_budget_for_sampling converts monetary budgets to discount_pct."""
        df_kwargs, X, y, _ = discounted_event_df
        df_events = _make_df_events()
        mmm, effect = _build_discounted_mmm(df_kwargs, X, y, df_events)
        model = mmm.model.copy()

        # budgets → pct = budget / event_revenue (auto-computed from y)
        budgets = np.array([200.0, 120.0])
        effect.set_budget_for_sampling(budgets, model)

        updated_pct = model["promo_discount_pct"].get_value()
        expected = budgets / effect._event_revenue
        np.testing.assert_allclose(updated_pct, expected)

    def test_set_budget_for_sampling_zero_revenue_guard(self, discounted_event_df):
        """set_budget_for_sampling must not divide by zero when event_revenue == 0."""
        df_kwargs, X, y, _ = discounted_event_df
        df_events = _make_df_events(names=("zero_rev_event",))
        mmm, effect = _build_discounted_mmm(df_kwargs, X, y, df_events)
        model = mmm.model.copy()

        # Simulate a window with zero observed revenue (e.g. event outside training range).
        effect._event_revenue = np.array([0.0])

        # Should not raise; safe_revenue guard replaces 0 with 1.0.
        budgets = np.array([500.0])
        effect.set_budget_for_sampling(budgets, model)

        updated_pct = model["promo_discount_pct"].get_value()
        np.testing.assert_allclose(updated_pct, np.array([500.0]))

    def test_sample_response_distribution_no_crash_with_effect_channels(
        self, discounted_event_df, discounted_event_idata
    ):
        """sample_response_distribution must not crash when allocation_strategy
        contains effect channel names alongside standard channel names, and must
        update discount_pct on the model before sampling."""
        from unittest.mock import patch

        df_kwargs, X, y, std_channels = discounted_event_df
        df_events = _make_df_events()

        mmm, _effect = _build_discounted_mmm(df_kwargs, X, y, df_events)
        mmm.idata = discounted_event_idata

        wrapper = BudgetOptimizerWrapper(
            model=mmm, start_date="2024-06-01", end_date="2024-12-31"
        )

        # Construct an allocation_strategy exactly as allocate_budget() returns:
        # standard channels + effect channel names in a single DataArray.
        all_channels = std_channels + df_events["name"].tolist()
        allocation_strategy = xr.DataArray(
            np.ones(len(all_channels)) * 100.0,
            coords={"channel": all_channels},
            dims=["channel"],
        )
        allocation_strategy.attrs["channel_type"] = {
            ch: "effect" if ch in df_events["name"].tolist() else "media"
            for ch in all_channels
        }

        # Add channel_contribution_original_scale to model.named_vars so the
        # guard in sample_response_distribution passes.
        with mmm.model:
            pmd.Deterministic(
                "channel_contribution_original_scale",
                mmm.model["channel_contribution"],
                dims=("date", "channel"),
            )

        # Fake return value: channel dim uses only the standard channels, matching
        # what sample_posterior_predictive actually returns.
        fake_response = xr.Dataset(
            {
                "y": xr.DataArray(np.ones((2, 2, 10)), dims=["chain", "draw", "date"]),
                "channel_contribution": xr.DataArray(
                    np.ones((2, 2, len(std_channels), 10)),
                    coords={"channel": std_channels},
                    dims=["chain", "draw", "channel", "date"],
                ),
                "channel_contribution_original_scale": xr.DataArray(
                    np.ones((2, 2, len(std_channels), 10)),
                    coords={"channel": std_channels},
                    dims=["chain", "draw", "channel", "date"],
                ),
                "total_media_contribution_original_scale": xr.DataArray(
                    np.ones((2, 2, 10)), dims=["chain", "draw", "date"]
                ),
            }
        )

        discount_pct_during_call = {}

        def _fake_spp(
            self_mmm,
            X,
            extend_idata,
            include_last_observations,
            var_names,
            progressbar,
            **kwargs,
        ):
            # Capture discount_pct as seen by the original model mid-call.
            discount_pct_during_call["value"] = (
                self_mmm.model["promo_discount_pct"].get_value().copy()
            )
            return fake_response

        with patch.object(type(mmm), "sample_posterior_predictive", _fake_spp):
            result = wrapper.sample_response_distribution(
                allocation_strategy=allocation_strategy
            )

        # Discount was updated to budget-derived values during the call.
        expected_pct = np.array([100.0, 100.0]) / _effect._event_revenue
        np.testing.assert_allclose(discount_pct_during_call["value"], expected_pct)

        # Result is a valid Dataset.
        assert isinstance(result, xr.Dataset)

        # allocation only contains media channels — event channels must NOT appear
        # in the channel coord (which would cause NaN expansion in channel_contribution).
        alloc_channels = result["allocation"].coords["channel"].values
        for event_name in df_events["name"]:
            assert event_name not in alloc_channels, (
                f"Event channel '{event_name}' leaked into allocation coord; "
                "this would expand channel_contribution with NaN."
            )
        for std_ch in std_channels:
            assert std_ch in alloc_channels

        # channel_contribution must not contain NaN for any media channel.
        for std_ch in std_channels:
            cc_slice = result["channel_contribution"].sel(channel=std_ch)
            assert not np.any(np.isnan(cc_slice.values)), (
                f"channel_contribution is NaN for media channel '{std_ch}'"
            )

    def test_channel_contribution_var_name(self):
        """channel_contribution_var_name returns the per-event Deterministic name."""
        df_events = _make_df_events()
        effect = DiscountedEventEffect(df_events=df_events, prefix="promo")
        assert effect.channel_contribution_var_name == "promo_channel_contribution"

    def test_channel_contribution_var_name_default_none_for_base(self):
        """OptimizableMuEffect base default returns None (no per-channel tracking)."""
        from pymc_marketing.mmm.additive_effect import OptimizableMuEffect

        dummy = _DummyOptimizableEffect(rf_channels=["a"])
        assert isinstance(dummy, OptimizableMuEffect)
        assert dummy.channel_contribution_var_name is None

    def test_create_effect_registers_per_event_contribution(self, discounted_event_df):
        """create_effect must register promo_channel_contribution in the model."""
        df_kwargs, X, y, _ = discounted_event_df
        df_events = _make_df_events()
        mmm, _ = _build_discounted_mmm(df_kwargs, X, y, df_events)
        assert "promo_channel_contribution" in mmm.model.named_vars
