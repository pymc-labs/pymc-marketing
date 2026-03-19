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

from pymc_marketing.mmm.additive_effect import (
    EventAdditiveEffect,
    FourierEffect,
    LinearTrendEffect,
    MuEffect,
)
from pymc_marketing.mmm.fourier import MonthlyFourier, WeeklyFourier, YearlyFourier
from pymc_marketing.mmm.linear_trend import LinearTrend


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

        idata.extend(
            pm.sample_posterior_predictive(
                idata.prior,
                var_names=[f"{fourier.prefix}_contribution"],
            )
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
        idata = pm.sample_prior_predictive(samples=5)

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

        idata.extend(
            pm.sample_posterior_predictive(
                idata.prior,
                var_names=["effect"],
            )
        )

    effect_predictions = idata.posterior_predictive.effect
    np.testing.assert_allclose(effect_predictions.notnull().mean().item(), 1.0)
    pd.testing.assert_index_equal(effect_predictions.date.to_index(), new_dates)


class TestMuEffectTypeRegistry:
    def test_mu_effect_not_registered(self):
        """MuEffect itself (abstract) should not be in the registry."""
        from pymc_marketing.serialization import registry

        type_key = f"{MuEffect.__module__}.{MuEffect.__qualname__}"
        assert type_key not in registry._registry

    @pytest.mark.parametrize(
        "cls",
        [FourierEffect, LinearTrendEffect, EventAdditiveEffect],
        ids=lambda c: c.__name__,
    )
    def test_concrete_subclass_registered(self, cls):
        from pymc_marketing.serialization import registry

        type_key = f"{cls.__module__}.{cls.__qualname__}"
        assert type_key in registry._registry

    def test_custom_mu_effect_auto_registered(self):
        """User-defined MuEffect subclasses should be auto-registered."""
        from pymc_marketing.serialization import registry

        class CustomTestEffect(MuEffect):
            my_val: int = 42

            def create_data(self, mmm):
                pass

            def create_effect(self, mmm):
                pass

            def set_data(self, mmm, model, X):
                pass

        type_key = f"{CustomTestEffect.__module__}.{CustomTestEffect.__qualname__}"
        assert type_key in registry._registry


class TestLinearTrendEffectSerialization:
    """Tests for TypeRegistry-based round-trip serialization of LinearTrendEffect."""

    def test_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.mmm.additive_effect import LinearTrendEffect
        from pymc_marketing.mmm.linear_trend import LinearTrend
        from pymc_marketing.serialization import registry

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
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is LinearTrendEffect
        assert restored.prefix == "custom_trend"
        assert restored.date_dim_name == "custom_date"
        assert type(restored.trend) is LinearTrend
        assert restored.trend.n_changepoints == 8
        assert restored.trend.include_intercept is True
        assert restored.trend.priors["delta"] == Prior(
            "Laplace", mu=0, b=0.5, dims="changepoint"
        )
        assert restored == original


class TestFourierEffectSerialization:
    """Tests for TypeRegistry-based round-trip serialization of FourierEffect."""

    @pytest.mark.parametrize(
        "fourier_cls",
        [YearlyFourier, MonthlyFourier, WeeklyFourier],
        ids=lambda c: c.__name__,
    )
    def test_fourier_effect_roundtrip_all_fourier_types(self, fourier_cls):
        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry

        original = FourierEffect(
            fourier=fourier_cls(
                n_order=5,
                prefix="custom_fourier",
                prior=Prior("Laplace", mu=0.5, b=2.0),
            ),
            date_dim_name="custom_date",
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is FourierEffect
        assert type(restored.fourier) is fourier_cls
        assert restored.fourier.n_order == 5
        assert restored.fourier.prefix == "custom_fourier"
        assert restored.date_dim_name == "custom_date"
        assert restored == original


class TestEventAdditiveEffectSerialization:
    """Tests for TypeRegistry-based round-trip serialization of EventAdditiveEffect."""

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
        from pymc_marketing.serialization import (
            DeserializationContext,
            registry,
        )

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

        data = registry.serialize(original)

        ds = xr.Dataset.from_dataframe(df.set_index("name"))
        fake_idata_dict = {"supplementary_data_custom_events": ds}

        class MockIdata:
            def __getitem__(self, key):
                return fake_idata_dict[key]

        ctx = DeserializationContext(idata=MockIdata())
        restored = registry.deserialize(data, context=ctx)

        assert type(restored) is EventAdditiveEffect
        assert len(restored.df_events) == 3
        assert restored.prefix == "custom_events"
        assert restored.reference_date == "2024-06-01"
        assert restored.date_dim_name == "custom_date"
        assert type(restored.effect) is EventEffect
        assert type(restored.effect.basis) is GaussianBasis
