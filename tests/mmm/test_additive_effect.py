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
    ControlEffect,
    FourierEffect,
    LinearTrendEffect,
)
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
        "pymc_marketing.mmm.additive_effect.ControlEffect",
    ],
    ids=lambda s: s.rsplit(".", 1)[-1],
)
def test_additive_effect_type_registered(type_key):
    assert type_key in serialization._registry, f"{type_key} not registered"


# ---------------------------------------------------------------------------
# ControlEffect tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def control_dates() -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=10, freq="W-MON", name="date")


@pytest.fixture(scope="function")
def control_xarray(control_dates) -> xr.Dataset:
    """Minimal xarray Dataset mimicking what MMM._set_xarray_data produces."""
    n_dates = len(control_dates)
    control_cols = ["price_index", "promo"]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_dates, len(control_cols)))
    control_da = xr.DataArray(
        data,
        dims=["date", "control"],
        coords={"date": control_dates, "control": control_cols},
        name="_control",
    )
    return xr.Dataset({"_control": control_da})


@pytest.fixture(scope="function")
def control_model(control_dates) -> pm.Model:
    return pm.Model(coords={"date": control_dates})


@pytest.fixture(scope="function")
def mock_mmm_control(control_model):
    class MMM:
        pass

    mmm = MMM()
    mmm.dims = ()
    mmm.model = control_model
    return mmm


def _make_panel_xarray(dates, geos, control_cols, seed=0):
    """Build a panel _control DataArray with dims (date, geo, control)."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(dates), len(geos), len(control_cols)))
    return xr.Dataset(
        {
            "_control": xr.DataArray(
                data,
                dims=["date", "geo", "control"],
                coords={"date": dates, "geo": geos, "control": control_cols},
            )
        }
    )


class TestControlEffect:
    def test_create_data_registers_shared_variable(
        self, mock_mmm_control, control_xarray, control_model
    ):
        effect = ControlEffect()
        with control_model:
            effect.create_data(mock_mmm_control, X=control_xarray)

        assert "control_data" in control_model.named_vars
        assert "control" in control_model.coords

    def test_create_data_with_no_X_does_nothing(self, mock_mmm_control, control_model):
        effect = ControlEffect()
        with control_model:
            effect.create_data(mock_mmm_control, X=None)

        assert "control_data" not in control_model.named_vars

    def test_create_data_with_missing_control_key_does_nothing(
        self, mock_mmm_control, control_dates, control_model
    ):
        effect = ControlEffect()
        X_no_control = xr.Dataset({"other": xr.DataArray([1, 2], dims=["date"])})
        with control_model:
            effect.create_data(mock_mmm_control, X=X_no_control)

        assert "control_data" not in control_model.named_vars

    def test_create_data_idempotent_coord(
        self, mock_mmm_control, control_xarray, control_model
    ):
        """Calling create_data twice must not raise due to duplicate coord."""
        effect = ControlEffect()
        with control_model:
            effect.create_data(mock_mmm_control, X=control_xarray)
            # Second call should not raise even though "control" coord exists
            effect.create_data(mock_mmm_control, X=control_xarray)

    def test_create_effect_produces_deterministic(
        self, mock_mmm_control, control_xarray, control_model
    ):
        effect = ControlEffect()
        with control_model:
            effect.create_data(mock_mmm_control, X=control_xarray)
            result = effect.create_effect(mock_mmm_control)

        assert "control_contribution" in control_model.named_vars
        # Result should have summed over "control" dim — only date dim remains
        assert "control" not in result.type.dims

    def test_create_effect_raises_without_create_data(
        self, mock_mmm_control, control_model
    ):
        """create_effect must raise clearly when create_data was not called first."""
        effect = ControlEffect()
        with control_model:
            with pytest.raises(RuntimeError, match="create_data"):
                effect.create_effect(mock_mmm_control)

    def test_set_data_updates_shared_variable(
        self, mock_mmm_control, control_xarray, control_model
    ):
        effect = ControlEffect()
        with control_model:
            effect.create_data(mock_mmm_control, X=control_xarray)
            effect.create_effect(mock_mmm_control)

        # New dataset with same structure but different values
        n_dates = len(control_xarray.coords["date"])
        rng = np.random.default_rng(42)
        new_data = rng.standard_normal((n_dates, 2))
        new_control_da = xr.DataArray(
            new_data,
            dims=["date", "control"],
            coords={
                "date": control_xarray.coords["date"],
                "control": control_xarray.coords["control"],
            },
            name="_control",
        )
        new_X = xr.Dataset({"_control": new_control_da})

        # Should not raise
        effect.set_data(mock_mmm_control, control_model, new_X)

    def test_set_data_with_none_does_nothing(
        self, mock_mmm_control, control_xarray, control_model
    ):
        effect = ControlEffect()
        with control_model:
            effect.create_data(mock_mmm_control, X=control_xarray)
            effect.create_effect(mock_mmm_control)

        # Should not raise
        effect.set_data(mock_mmm_control, control_model, None)

    def test_set_data_with_no_control_key_does_nothing(
        self, mock_mmm_control, control_xarray, control_model
    ):
        effect = ControlEffect()
        with control_model:
            effect.create_data(mock_mmm_control, X=control_xarray)
            effect.create_effect(mock_mmm_control)

        X_no_control = xr.Dataset()
        # Should not raise
        effect.set_data(mock_mmm_control, control_model, X_no_control)

    def test_to_dict_round_trip(self):
        prior = Prior("Normal", mu=0, sigma=1, dims="control")
        effect = ControlEffect(prior=prior, date_dim_name="time")

        data = serialization.serialize(effect)
        restored = serialization.deserialize(data)

        assert type(restored) is ControlEffect
        assert restored.date_dim_name == "time"
        assert restored.prior._distribution == "Normal"

    def test_default_prior(self):
        effect = ControlEffect()
        assert effect.prior._distribution == "Normal"
        assert effect.prior.parameters["mu"] == 0
        assert effect.prior.parameters["sigma"] == 2
        assert "control" in (effect.prior.dims or ())

    @pytest.mark.parametrize(
        "input_prior,expected_dims",
        [
            (Prior("Normal"), ("control",)),
            (Prior("Normal", dims="control"), ("control",)),
            (Prior("Normal", mu=0, sigma=1), ("control",)),
            (Prior("Normal", dims="geo"), ("geo",)),
        ],
        ids=["no-dims", "already-control", "kwargs-no-dims", "explicit-other-dim"],
    )
    def test_prior_validator_ensures_control_dim(self, input_prior, expected_dims):
        """Prior with dims=None is defaulted to 'control'; explicit dims are untouched."""
        effect = ControlEffect(prior=input_prior)
        assert effect.prior.dims == expected_dims

    @pytest.mark.parametrize(
        "dims,extra_coords",
        [
            ((), {}),
            (("geo",), {"geo": ["A", "B"]}),
        ],
        ids=["scalar", "panel-geo"],
    )
    def test_create_effect_multidimensional(self, control_dates, dims, extra_coords):
        """ControlEffect must produce correct dims for both scalar and panel MMMs."""
        control_cols = ["price_index", "promo"]
        geos = extra_coords.get("geo", [])

        coords = {"date": control_dates, **extra_coords}
        model = pm.Model(coords=coords)

        class MockMMM:
            pass

        mmm = MockMMM()
        mmm.dims = dims
        mmm.model = model

        if dims:
            X = _make_panel_xarray(control_dates, geos, control_cols)
            prior = Prior("Normal", mu=0, sigma=2, dims=(*dims, "control"))
        else:
            rng = np.random.default_rng(0)
            data = rng.standard_normal((len(control_dates), len(control_cols)))
            X = xr.Dataset(
                {
                    "_control": xr.DataArray(
                        data,
                        dims=["date", "control"],
                        coords={"date": control_dates, "control": control_cols},
                    )
                }
            )
            prior = Prior("Normal", mu=0, sigma=2, dims="control")

        effect = ControlEffect(prior=prior)

        with model:
            effect.create_data(mmm, X=X)
            result = effect.create_effect(mmm)

        assert "control_contribution" in model.named_vars
        assert "control" not in result.type.dims
        # date dim should be present
        assert "date" in result.type.dims
        # geo dim should be present for panel models
        for dim in dims:
            assert dim in result.type.dims
