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
import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.additive_effect import (
    DiscountedEventEffect,
    FourierEffect,
    LinearTrendEffect,
)
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
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


def _make_discount_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["black_friday", "summer_sale"],
            "start_date": ["2025-03-03", "2025-01-20"],
            "end_date": ["2025-03-24", "2025-02-10"],
            "discount_pct": [0.30, 0.20],
        }
    )


class TestDiscountedEventEffectValidation:
    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="missing required columns"):
            DiscountedEventEffect(
                df_events=pd.DataFrame({"name": ["a"], "start_date": ["2024-01-01"]}),
                prefix="promo",
            )

    def test_inverted_bounds_raise(self):
        with pytest.raises(ValueError, match="discount_min"):
            DiscountedEventEffect(
                df_events=_make_discount_events(),
                prefix="promo",
                discount_min=0.5,
                discount_max=0.2,
            )

    def test_discount_pct_out_of_range_raises(self):
        df = _make_discount_events()
        df["discount_pct"] = [1.5, 0.2]
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            DiscountedEventEffect(df_events=df, prefix="promo")


class TestDiscountedEventEffectUnits:
    def test_lever_bounds_native_units(self):
        effect = DiscountedEventEffect(
            df_events=_make_discount_events(),
            prefix="promo",
            discount_min=0.05,
            discount_max=0.4,
        )
        assert effect.lever_bounds == [(0.05, 0.4), (0.05, 0.4)]

    def test_window_mask_marks_event_dates(self, dates):
        effect = DiscountedEventEffect(
            df_events=_make_discount_events(), prefix="promo"
        )
        window = effect._window_mask(dates)
        assert window.dims == ("date", "promo")
        assert list(window.coords["promo"].values) == ["black_friday", "summer_sale"]
        in_window = (dates >= pd.Timestamp("2025-03-03")) & (
            dates <= pd.Timestamp("2025-03-24")
        )
        np.testing.assert_array_equal(
            window.sel(promo="black_friday").values, in_window.astype(float)
        )

    def test_lever_var_name(self):
        effect = DiscountedEventEffect(
            df_events=_make_discount_events(), prefix="promo"
        )
        assert effect.lever_var_name == "promo_data"

    def test_lever_var_name_without_prefix_raises(self):
        from pymc_marketing.mmm.additive_effect import OptimizableMuEffect

        class NoPrefixEffect(OptimizableMuEffect):
            def create_data(self, mmm):  # pragma: no cover - contract stub
                pass

            def create_effect(self, mmm):  # pragma: no cover - contract stub
                pass

            def set_data(self, mmm, model, X):  # pragma: no cover - contract stub
                pass

        with pytest.raises(NotImplementedError, match="lever_var_name"):
            _ = NoPrefixEffect().lever_var_name

    def test_events_sharing_model_dates(self, dates):
        df = _make_discount_events()  # disjoint windows
        effect = DiscountedEventEffect(df_events=df, prefix="promo")
        window = effect._window_mask(dates)
        assert effect._events_sharing_model_dates(window) == []

        df_overlap = df.copy()
        df_overlap.loc[1, ["start_date", "end_date"]] = ["2025-03-10", "2025-04-01"]
        effect = DiscountedEventEffect(df_events=df_overlap, prefix="promo")
        window = effect._window_mask(dates)
        assert effect._events_sharing_model_dates(window) == [
            "black_friday",
            "summer_sale",
        ]

    def test_to_dict_and_idata_group(self):
        effect = DiscountedEventEffect(
            df_events=_make_discount_events(),
            prefix="promo",
            discount_min=0.0,
            discount_max=0.35,
        )
        d = effect.to_dict()
        assert d["df_events_group"] == "supplementary_data_promo"
        assert d["discount_max"] == 0.35
        groups = effect.idata_groups()
        assert "supplementary_data_promo" in groups
        assert list(groups["supplementary_data_promo"]["name"].values) == [
            "black_friday",
            "summer_sale",
        ]

    def test_type_registered_for_serialization(self):
        key = "pymc_marketing.mmm.additive_effect.DiscountedEventEffect"
        assert key in serialization._registry


def test_discounted_event_effect_save_load_roundtrip(mock_pymc_sample, tmp_path):
    """Save/load preserves df_events (names + discount_pct) and native bounds."""
    date_range = pd.date_range("2023-01-01", periods=16, freq="W")
    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        {
            "date": date_range,
            "ch1": rng.uniform(100, 500, size=len(date_range)),
            "ch2": rng.uniform(100, 500, size=len(date_range)),
        }
    )
    y = pd.Series(rng.uniform(500, 1500, size=len(date_range)), name="target")

    effect = DiscountedEventEffect(
        df_events=pd.DataFrame(
            {
                "name": ["spring_sale"],
                "start_date": ["2023-02-01"],
                "end_date": ["2023-03-15"],
                "discount_pct": [0.15],
            }
        ),
        prefix="promo",
        discount_min=0.0,
        discount_max=0.4,
    )
    mmm = MMM(
        date_column="date",
        channel_columns=["ch1", "ch2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    ).add_mu_effect(effect)
    mmm.fit(X, y, random_seed=3)

    path = tmp_path / "mmm_discount.nc"
    mmm.save(str(path))
    loaded = type(mmm).load(str(path))

    (restored,) = [e for e in loaded.mu_effects if isinstance(e, DiscountedEventEffect)]
    assert restored.df_events["name"].tolist() == ["spring_sale"]
    assert restored.discount_max == 0.4
    assert restored.lever_bounds == [(0.0, 0.4)]
    np.testing.assert_allclose(
        restored.df_events["discount_pct"].to_numpy(dtype="float64"), [0.15]
    )


# ---------------------------------------------------------------------------
# DiscountedEventEffect: build-based tests (both links)
# ---------------------------------------------------------------------------


def _make_two_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["spring_sale", "fall_sale"],
            "start_date": ["2023-03-01", "2023-09-01"],
            "end_date": ["2023-03-31", "2023-09-30"],
            "discount_pct": [0.30, 0.20],
        }
    )


def _build_discount_mmm(link: str, df_events: pd.DataFrame | None = None) -> MMM:
    rng = np.random.default_rng(1)
    dates_ = pd.date_range("2023-01-01", periods=52, freq="W")
    X = pd.DataFrame(
        {
            "date": dates_,
            "ch1": rng.uniform(100, 500, size=len(dates_)),
            "ch2": rng.uniform(100, 500, size=len(dates_)),
        }
    )
    y = pd.Series(rng.uniform(500, 1500, size=len(dates_)), name="target")
    effect = DiscountedEventEffect(
        df_events=df_events if df_events is not None else _make_two_events(),
        prefix="promo",
        discount_min=0.05,
        discount_max=0.45,
    )
    mmm = MMM(
        date_column="date",
        channel_columns=["ch1", "ch2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        link=link,
    ).add_mu_effect(effect)
    mmm.build_model(X, y)
    return mmm


@pytest.mark.parametrize("link", ["identity", "log"])
def test_discount_create_data_registers_nodes(link):
    mmm = _build_discount_mmm(link)
    model = mmm.model
    assert "promo_window" in model.named_vars
    assert "promo_data" in model.named_vars
    assert "promo_event_contribution" in model.named_vars
    assert "promo_effect_contribution" in model.named_vars
    assert "total_response_original_scale" in model.named_vars
    # The r_k machinery is gone
    assert "promo_revenue_per_period" not in model.named_vars
    # Lever initialised at the historical depths
    np.testing.assert_allclose(model["promo_data"].get_value(), [0.30, 0.20])


def test_discount_forward_formula_log():
    """Numeric check: contribution = window * (ln(1-d) + beta*ln(1+d))."""
    mmm = _build_discount_mmm("log")
    beta_val = np.array([2.0, 1.5])
    d = np.array([0.30, 0.20])
    fixed = pm.do(mmm.model, {"promo_beta": beta_val})
    contrib, window = pm.draw(
        [fixed["promo_event_contribution"], fixed["promo_window"]], random_seed=1
    )
    expected = window * (np.log1p(-d) + beta_val * np.log1p(d))[None, :]
    np.testing.assert_allclose(contrib, expected, rtol=1e-10)
    # d=0 => no contribution; both events lift is finite and event-specific
    assert not np.allclose(expected[window.astype(bool)], 0.0)


def test_discount_forward_formula_identity():
    """Numeric check: contribution = mu_base * ((1-d)(1+d)^beta - 1) * window.

    The multiplier is exact (identical to the log link's factor), so both
    links share the same standalone optimum.
    """
    mmm = _build_discount_mmm("identity")
    beta_val = np.array([2.0, 1.5])
    d = np.array([0.30, 0.20])
    fixed = pm.do(mmm.model, {"promo_beta": beta_val})
    contrib, window, intercept, channel = pm.draw(
        [
            fixed["promo_event_contribution"],
            fixed["promo_window"],
            fixed["intercept_contribution"],
            fixed["channel_contribution"],
        ],
        random_seed=1,
    )
    # The fixture has no controls/seasonality: baseline = intercept + media
    baseline = intercept + channel.sum(axis=-1)  # (date,)
    mult = (1.0 - d) * (1.0 + d) ** beta_val - 1.0
    expected = baseline[:, None] * (window * mult[None, :])
    np.testing.assert_allclose(contrib, expected, rtol=1e-8)


def test_discount_overlap_guard():
    """Windows sharing a model date raise under identity, compose under log."""
    df = _make_two_events()
    df.loc[1, ["start_date", "end_date"]] = ["2023-03-15", "2023-04-15"]
    with pytest.raises(ValueError, match="same model date"):
        _build_discount_mmm("identity", df_events=df)
    mmm = _build_discount_mmm("log", df_events=df)
    assert "promo_event_contribution" in mmm.model.named_vars


def test_discount_calendar_overlap_without_shared_model_date_builds():
    """Calendar-overlapping windows that resolve to disjoint model dates are fine.

    On weekly data, two short windows can overlap on the calendar while never
    flagging the same model date -- no double-counting occurs, so the
    identity-link guard must not fire.
    """
    # The fixture's model dates are Sundays. short_a (03-01..03-08) and
    # short_b (03-06..03-11) overlap on the calendar (03-06..03-08) but only
    # short_a contains a model date (Sunday 03-05) -- no shared model date.
    df = pd.DataFrame(
        {
            "name": ["short_a", "short_b"],
            "start_date": ["2023-03-01", "2023-03-06"],
            "end_date": ["2023-03-08", "2023-03-11"],
            "discount_pct": [0.10, 0.10],
        }
    )
    mmm = _build_discount_mmm("identity", df_events=df)
    assert "promo_event_contribution" in mmm.model.named_vars


def test_discount_log_link_rejects_full_discount():
    """discount_max or discount_pct >= 1 is -inf under the log link: raise."""
    df = _make_two_events()
    with pytest.raises(ValueError, match="strictly below 1"):
        effect = DiscountedEventEffect(
            df_events=df, prefix="promo", discount_min=0.05, discount_max=1.0
        )
        rng = np.random.default_rng(1)
        dates_ = pd.date_range("2023-01-01", periods=52, freq="W")
        X = pd.DataFrame(
            {
                "date": dates_,
                "ch1": rng.uniform(100, 500, size=len(dates_)),
                "ch2": rng.uniform(100, 500, size=len(dates_)),
            }
        )
        y = pd.Series(rng.uniform(500, 1500, size=len(dates_)), name="target")
        MMM(
            date_column="date",
            channel_columns=["ch1", "ch2"],
            target_column="target",
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            link="log",
        ).add_mu_effect(effect).build_model(X, y)


def test_discount_identity_baseline_includes_non_optimizable_effects():
    """Non-optimizable mu effects (e.g. Fourier seasonality) are repriced.

    build_model applies non-optimizable effects before stashing _mu_baseline,
    so the discount multiplier covers them; optimizable effects apply after
    the stash and never reprice each other.
    """
    rng = np.random.default_rng(1)
    dates_ = pd.date_range("2023-01-01", periods=52, freq="W")
    X = pd.DataFrame(
        {
            "date": dates_,
            "ch1": rng.uniform(100, 500, size=len(dates_)),
            "ch2": rng.uniform(100, 500, size=len(dates_)),
        }
    )
    y = pd.Series(rng.uniform(500, 1500, size=len(dates_)), name="target")
    fourier_effect = FourierEffect(fourier=YearlyFourier(n_order=2))
    discount = DiscountedEventEffect(
        df_events=_make_two_events(),
        prefix="promo",
        discount_min=0.05,
        discount_max=0.45,
    )
    mmm = MMM(
        date_column="date",
        channel_columns=["ch1", "ch2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        link="identity",
    )
    # Order in mu_effects should not matter: the optimizable effect is
    # applied after the stash regardless.
    mmm.add_mu_effect(discount).add_mu_effect(fourier_effect)
    mmm.build_model(X, y)

    beta_val = np.array([2.0, 1.5])
    d = np.array([0.30, 0.20])
    fixed = pm.do(mmm.model, {"promo_beta": beta_val})
    contrib, window, intercept, channel, fourier_contrib = pm.draw(
        [
            fixed["promo_event_contribution"],
            fixed["promo_window"],
            fixed["intercept_contribution"],
            fixed["channel_contribution"],
            fixed[fourier_effect.contribution_var_name],
        ],
        random_seed=1,
    )
    baseline = intercept + channel.sum(axis=-1) + fourier_contrib  # (date,)
    mult = (1.0 - d) * (1.0 + d) ** beta_val - 1.0
    expected = baseline[:, None] * (window * mult[None, :])
    np.testing.assert_allclose(contrib, expected, rtol=1e-8)


def test_discount_two_effects_compose_multiplicatively():
    """Two repricing effects sharing a model date give mu_base*(1+m1)(1+m2).

    build_model refreshes _mu_baseline after each optimizable effect, so
    multipliers compose (as under the log link) instead of summing on the
    same baseline -- the additive cross-effect double-count.
    """
    rng = np.random.default_rng(1)
    dates_ = pd.date_range("2023-01-01", periods=52, freq="W")
    X = pd.DataFrame(
        {
            "date": dates_,
            "ch1": rng.uniform(100, 500, size=len(dates_)),
            "ch2": rng.uniform(100, 500, size=len(dates_)),
        }
    )
    y = pd.Series(rng.uniform(500, 1500, size=len(dates_)), name="target")

    def make(prefix, d):
        return DiscountedEventEffect(
            df_events=pd.DataFrame(
                {
                    "name": [f"{prefix}_event"],
                    "start_date": ["2023-03-01"],
                    "end_date": ["2023-03-31"],  # same window: shared dates
                    "discount_pct": [d],
                }
            ),
            prefix=prefix,
        )

    mmm = (
        MMM(
            date_column="date",
            channel_columns=["ch1", "ch2"],
            target_column="target",
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            link="identity",
        )
        .add_mu_effect(make("promo", 0.30))
        .add_mu_effect(make("loyalty", 0.20))
    )
    mmm.build_model(X, y)

    beta1, beta2 = 2.0, 1.5
    d1, d2 = 0.30, 0.20
    fixed = pm.do(
        mmm.model,
        {"promo_beta": np.array([beta1]), "loyalty_beta": np.array([beta2])},
    )
    c1, c2, window, intercept, channel = pm.draw(
        [
            fixed["promo_effect_contribution"],
            fixed["loyalty_effect_contribution"],
            fixed["promo_window"],
            fixed["intercept_contribution"],
            fixed["channel_contribution"],
        ],
        random_seed=1,
    )
    baseline = intercept + channel.sum(axis=-1)  # (date,)
    m1 = (1 - d1) * (1 + d1) ** beta1 - 1.0
    m2 = (1 - d2) * (1 + d2) ** beta2 - 1.0
    w = window[:, 0]  # shared window (date,)
    # Total on shared dates: mu_base * ((1+m1)(1+m2) - 1), NOT mu_base*(m1+m2)
    expected_total = baseline * w * ((1 + m1) * (1 + m2) - 1.0)
    np.testing.assert_allclose(c1 + c2, expected_total, rtol=1e-8)
    additive_double_count = baseline * w * (m1 + m2)
    assert not np.allclose(c1 + c2, additive_double_count)


def test_total_response_not_registered_without_optimizable_effect():
    """No silent posterior addition: the objective node is gated.

    Plain models (and models with only non-optimizable mu effects) must not
    gain total_response_original_scale.
    """
    rng = np.random.default_rng(1)
    dates_ = pd.date_range("2023-01-01", periods=20, freq="W")
    X = pd.DataFrame(
        {
            "date": dates_,
            "ch1": rng.uniform(100, 500, size=len(dates_)),
            "ch2": rng.uniform(100, 500, size=len(dates_)),
        }
    )
    y = pd.Series(rng.uniform(500, 1500, size=len(dates_)), name="target")

    plain = MMM(
        date_column="date",
        channel_columns=["ch1", "ch2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )
    plain.build_model(X, y)
    assert "total_response_original_scale" not in plain.model.named_vars

    with_fourier = MMM(
        date_column="date",
        channel_columns=["ch1", "ch2"],
        target_column="target",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    ).add_mu_effect(FourierEffect(fourier=YearlyFourier(n_order=2)))
    with_fourier.build_model(X, y)
    assert "total_response_original_scale" not in with_fourier.model.named_vars


def test_discount_identity_requires_mu_baseline(dates):
    """A custom model without the _mu_baseline stash gets a clear error."""
    from types import SimpleNamespace

    effect = DiscountedEventEffect(df_events=_make_discount_events(), prefix="promo")
    with pm.Model(coords={"date": dates}) as model:
        stub = SimpleNamespace(model=model, link="identity")
        effect.create_data(stub)
        with pytest.raises(ValueError, match="_mu_baseline"):
            effect.create_effect(stub)


def test_discount_multidim_build(panel_mmm_data):
    """Panel MMM + DiscountedEventEffect builds and broadcasts over dims.

    Previously broke with a TypeError on ``float(scalers._target)``.
    """
    X, y = panel_mmm_data["X"], panel_mmm_data["y"]
    effect = DiscountedEventEffect(
        df_events=pd.DataFrame(
            {
                "name": ["promo_week"],
                "start_date": ["2023-02-01"],
                "end_date": ["2023-02-21"],
                "discount_pct": [0.2],
            }
        ),
        prefix="promo",
    )
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    ).add_mu_effect(effect)
    mmm.build_model(X, y)
    dims = tuple(mmm.model.named_vars_to_dims["promo_event_contribution"])
    assert set(dims) == {"date", "promo", "country"}
    contrib = pm.draw(mmm.model["promo_event_contribution"], random_seed=1)
    assert np.isfinite(contrib).all()
