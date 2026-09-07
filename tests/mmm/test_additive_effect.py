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
import pytensor.tensor as pt
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.mmm.additive_effect import (
    ControlMuEffect,
    DataVarMuEffect,
    FourierEffect,
    LinearTrendEffect,
    MediaMuEffect,
)
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.fourier import MonthlyFourier, WeeklyFourier, YearlyFourier
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.media_transformation import MediaTransformation
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


class TestDataVarMuEffect:
    """Tests for the DataVarMuEffect base class."""

    def test_instantiation_error(self):
        """DataVarMuEffect is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataVarMuEffect(data_vars=["x"], prefix="test")  # type: ignore

    def _make_mock_mmm(self, model, ds):
        """Create a minimal mock MMM for testing."""
        MockMMM = type(
            "MockMMM", (), {"dims": (), "model": model, "xarray_dataset": ds}
        )
        return MockMMM()

    def test_concrete_subclass_create_data(self, dates):
        """Subclass registers data variables from xarray_dataset as pm.Data."""
        rng = np.random.default_rng(42)
        ds = xr.Dataset(
            {"feature": (("date",), rng.normal(size=len(dates)))},
            coords={"date": dates},
        )
        model = pm.Model(coords={"date": dates})

        mmm = self._make_mock_mmm(model, ds)

        class TestEffect(DataVarMuEffect):
            data_vars: list[str] = ["feature"]
            prefix: str = "test"

            def create_effect(self, mmm):  # type: ignore
                return pt.as_tensor(0.0)

            def set_data(self, mmm, model, X):  # type: ignore
                pass

        effect = TestEffect()

        with mmm.model:
            effect.create_data(mmm)

        assert "feature" in mmm.model.named_vars

    def test_set_data_updates_variables(self, dates, new_dates):
        """set_data updates pm.Data from new dataset."""
        rng = np.random.default_rng(42)
        ds = xr.Dataset(
            {"feature": (("date",), rng.normal(size=len(dates)))},
            coords={"date": dates},
        )
        new_ds = xr.Dataset(
            {"feature": (("date",), rng.normal(size=len(new_dates)))},
            coords={"date": new_dates},
        )
        model = pm.Model(coords={"date": dates})

        mmm = self._make_mock_mmm(model, ds)

        class TestEffect(DataVarMuEffect):
            data_vars: list[str] = ["feature"]
            prefix: str = "test"

            def create_effect(self, mmm):  # type: ignore
                return pt.as_tensor(0.0)

            def set_data(self, mmm, model, X):  # type: ignore
                for var_name in self.data_vars:
                    if var_name in X.data_vars:
                        pm.set_data({var_name: X[var_name].values}, model=model)

        effect = TestEffect()

        with mmm.model:
            effect.create_data(mmm)
            model_copy = model.copy()
            model_copy.set_dim("date", len(new_dates), coord_values=new_dates)
            effect.set_data(mmm, model_copy, new_ds)


class TestMediaMuEffect:
    """Tests for MediaMuEffect."""

    @pytest.mark.parametrize(
        "effect_dims, mmm_dims, extra_coords, prefix",
        [
            pytest.param((), (), {}, "national", id="no_dims"),
            pytest.param(
                ("product",),
                ("product", "geo"),
                {"product": ["A", "B"]},
                "product",
                id="single_dim",
            ),
            pytest.param(
                ("product", "geo"),
                ("product", "geo"),
                {"product": ["A", "B"], "geo": ["X", "Y"]},
                "full",
                id="multiple_dims",
            ),
        ],
    )
    def test_create_effect(
        self,
        dates,
        effect_dims,
        mmm_dims,
        extra_coords,
        prefix,
    ):
        all_dims = ("date", *effect_dims, "channel")
        n_dates = len(dates)

        shape = [n_dates]
        coords = {"date": dates, "channel": ["tv", "digital"]}
        for dim_name in effect_dims:
            vals = extra_coords[dim_name]
            shape.append(len(vals))
            coords[dim_name] = vals
        shape.append(len(coords["channel"]))

        rng = np.random.default_rng(42)
        data_vals = rng.exponential(size=shape)

        ds = xr.Dataset(
            {"media_data": (all_dims, data_vals)},
            coords=coords,
        )

        model_coords = {"date": dates, "channel": ["tv", "digital"]}
        model_coords.update(extra_coords)
        model = pm.Model(coords=model_coords)

        MockMMM = type(
            "MockMMM", (), {"dims": mmm_dims, "model": model, "xarray_dataset": ds}
        )
        mmm = MockMMM()

        effect = MediaMuEffect(
            data_vars=["media_data"],
            effect_dims=effect_dims,
            media_transformation=MediaTransformation(
                adstock=GeometricAdstock(l_max=4),
                saturation=LogisticSaturation(),
                adstock_first=True,
                dims=(*effect_dims, "channel"),
            ),
            prefix=prefix,
        )

        with mmm.model:
            effect.create_data(mmm)
            contribution = effect.create_effect(mmm)

        # Check the deterministic was registered
        var_name = f"{prefix}_effect_contribution"
        assert var_name in mmm.model.named_vars
        # Check dims of the returned tensor
        contrib_dims = set(contribution.dims)
        assert "date" in contrib_dims
        for dim in effect_dims:
            assert dim in contrib_dims
        assert "channel" not in contrib_dims

        # Check contribution_var_name property
        assert effect.contribution_var_name == var_name

        # Verify we can sample prior predictive
        with mmm.model:
            pm.sample_prior_predictive(draws=5, random_seed=rng)

    @pytest.mark.parametrize(
        "effect_dims, mmm_dims, extra_coords, prefix",
        [
            pytest.param((), (), {}, "national", id="no_dims"),
            pytest.param(
                ("product",),
                ("product", "geo"),
                {"product": ["A", "B"]},
                "product",
                id="single_dim",
            ),
        ],
    )
    def test_set_data(
        self,
        dates,
        new_dates,
        effect_dims,
        mmm_dims,
        extra_coords,
        prefix,
    ):
        all_dims = ("date", *effect_dims, "channel")
        n_dates = len(dates)

        shape = [n_dates]
        coords = {"date": dates, "channel": ["tv", "digital"]}
        for dim_name in effect_dims:
            vals = extra_coords[dim_name]
            shape.append(len(vals))
            coords[dim_name] = vals
        shape.append(len(coords["channel"]))

        rng = np.random.default_rng(42)

        ds = xr.Dataset(
            {"media_data": (all_dims, rng.exponential(size=shape))},
            coords=coords,
        )

        # Prediction data with new dates
        new_shape = [len(new_dates)]
        for dim_name in effect_dims:
            new_shape.append(len(extra_coords[dim_name]))
        new_shape.append(len(coords["channel"]))

        new_coords = coords.copy()
        new_coords["date"] = new_dates
        new_ds = xr.Dataset(
            {"media_data": (all_dims, rng.exponential(size=new_shape))},
            coords=new_coords,
        )

        model_coords = {"date": dates, "channel": ["tv", "digital"]}
        model_coords.update(extra_coords)
        model = pm.Model(coords=model_coords)

        MockMMM = type(
            "MockMMM", (), {"dims": mmm_dims, "model": model, "xarray_dataset": ds}
        )
        mmm = MockMMM()

        effect = MediaMuEffect(
            data_vars=["media_data"],
            effect_dims=effect_dims,
            media_transformation=MediaTransformation(
                adstock=GeometricAdstock(l_max=4),
                saturation=LogisticSaturation(),
                adstock_first=True,
                dims=(*effect_dims, "channel"),
            ),
            prefix=prefix,
        )

        with mmm.model:
            effect.create_data(mmm)
            model_copy = model.copy()
            model_copy.set_dim("date", len(new_dates), coord_values=new_dates)
            effect.set_data(mmm, model_copy, new_ds)

        # Check that set_data didn't error and variables exist
        var_name = "media_data"
        assert var_name in model_copy.named_vars

    def test_serialization_roundtrip(self):
        original = MediaMuEffect(
            data_vars=["media_data"],
            media_transformation=MediaTransformation(
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                adstock_first=True,
                dims=("product", "channel"),
            ),
            prefix="test_media",
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is MediaMuEffect
        assert restored.data_vars == ["media_data"]
        assert restored.effect_dims == ("product",)
        assert restored.prefix == "test_media"
        assert type(restored.media_transformation) is MediaTransformation
        assert type(restored.media_transformation.adstock) is GeometricAdstock
        assert restored.media_transformation.adstock.l_max == 8
        assert type(restored.media_transformation.saturation) is LogisticSaturation
        assert restored.media_transformation.adstock_first is True
        assert restored == original


class TestControlMuEffect:
    """Tests for ControlMuEffect."""

    @pytest.mark.parametrize(
        "ctrl_dims, mmm_dims, extra_coords, prefix",
        [
            pytest.param(
                (),
                ("product", "geo"),
                {"product": ["A", "B"], "geo": ["X", "Y"]},
                "national",
                id="national_controls",
            ),
            pytest.param(
                ("product",),
                ("product", "geo"),
                {"product": ["A", "B"], "geo": ["X", "Y"]},
                "product_ctrl",
                id="product_controls",
            ),
        ],
    )
    def test_create_effect(
        self,
        dates,
        ctrl_dims,
        mmm_dims,
        extra_coords,
        prefix,
    ):
        var_name = "control_data"
        all_dims = ("date", *ctrl_dims, "control")

        n_dates = len(dates)
        shape = [n_dates]
        coords = {"date": dates, "control": ["c1", "c2"]}
        for dim_name in ctrl_dims:
            vals = extra_coords[dim_name]
            shape.append(len(vals))
            coords[dim_name] = vals
        shape.append(len(coords["control"]))

        rng = np.random.default_rng(42)
        data_vals = rng.normal(size=shape)

        ds = xr.Dataset(
            {var_name: (all_dims, data_vals)},
            coords=coords,
        )

        model_coords = {"date": dates, "control": ["c1", "c2"]}
        model_coords.update(extra_coords)
        model = pm.Model(coords=model_coords)

        MockMMM = type(
            "MockMMM", (), {"dims": mmm_dims, "model": model, "xarray_dataset": ds}
        )
        mmm = MockMMM()

        effect = ControlMuEffect(
            data_vars=[var_name],
            prefix=prefix,
        )

        with mmm.model:
            effect.create_data(mmm)
            contribution = effect.create_effect(mmm)

        # Check deterministic was registered
        assert f"{prefix}_effect_contribution" in mmm.model.named_vars

        # Check dims: should have date and effect dims, not control
        contrib_dims = set(contribution.dims)
        assert "date" in contrib_dims
        for dim in ctrl_dims:
            assert dim in contrib_dims
        assert "control" not in contrib_dims

        # Check contribution_var_name
        assert effect.contribution_var_name == f"{prefix}_effect_contribution"

        # Can sample prior predictives
        with mmm.model:
            pm.sample_prior_predictive(draws=5, random_seed=rng)

    def test_serialization_roundtrip(self):
        original = ControlMuEffect(
            data_vars=["ctrl_national"],
            prefix="test_ctrl",
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is ControlMuEffect
        assert restored.data_vars == ["ctrl_national"]
        assert restored.prefix == "test_ctrl"
        assert restored == original
