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

"""Tests for pymc_marketing.terms_gp."""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytest
import xarray as xr
from pymc_extras.prior import Prior, VariableFactory
from pytensor.graph.basic import Variable as PTVariable

from pymc_marketing.hsgp_kwargs import CovFunc
from pymc_marketing.mmm.hsgp import HSGP, HSGPPeriodic, SoftPlusHSGP
from pymc_marketing.mmm.tvp import infer_time_index
from pymc_marketing.model_graph import deterministics_to_flat
from pymc_marketing.serialization import serialization
from pymc_marketing.terms import (
    Dot,
    Intercept,
    ModelTerm,
    Named,
    Sum,
    build_param,
    collect_coords,
    frozen_deterministics,
    register_data,
    set_data,
)
from pymc_marketing.terms_gp import HSGPPeriodicTerm, HSGPTerm, SoftPlusHSGPTerm


@pytest.fixture
def ds():
    """Dataset with a datetime date coordinate, media, and channel dims."""
    rng = np.random.default_rng(42)
    n = 40
    dates = pd.date_range("2023-01-02", periods=n, freq="W-MON")
    return xr.Dataset(
        {
            "media": (("date", "channel"), rng.normal(size=(n, 2)) ** 2),
        },
        coords={"date": dates, "channel": ["A", "B"]},
    )


@pytest.fixture
def ds_product():
    """Dataset with a datetime date coordinate and a product dimension."""
    dates = pd.date_range("2023-01-02", periods=40, freq="W-MON")
    return xr.Dataset({}, coords={"date": dates, "product": ["EU", "US", "JP"]})


@pytest.fixture
def ds_num():
    """Dataset with a numeric time data variable and features."""
    rng = np.random.default_rng(42)
    n = 52
    return xr.Dataset(
        {
            "time": ("t", np.arange(n, dtype=float)),
            "x": (("t", "feature"), rng.normal(size=(n, 3))),
        },
        coords={"t": np.arange(n)},
    )


@dataclass(kw_only=True)
class ChannelScaled(ModelTerm):
    """Media-like term: data * channel scale, output (date, channel)."""

    var_name: str
    name: str
    prior: VariableFactory

    def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
        return {k: v.values.tolist() for k, v in ds[self.var_name].coords.items()}

    def register_data(self, ds: xr.Dataset) -> None:
        model = pm.modelcontext(None)
        if self.var_name not in model:
            pmd.Data(self.var_name, ds[self.var_name])

    def set_data(self, ds: xr.Dataset, model: pm.Model | None = None) -> None:
        if self.var_name in ds:
            da = ds[self.var_name]
            coords = {dim: ds[dim].values for dim in da.dims if dim in ds.coords}
            pm.set_data({self.var_name: da.values}, model=model, coords=coords)

    def create_variable(self) -> PTVariable:
        model = pm.modelcontext(None)
        data = model[self.var_name]
        scale = self.prior.create_variable(self.name, xdist=True)
        return data * scale


def media_term(ds=None, name="scale"):
    """A media-chain stand-in with (date, channel) output."""
    return ChannelScaled(
        var_name="media",
        name=name,
        prior=Prior("HalfNormal", sigma=1, dims="channel"),
    )


def test_no_arg_defaults():
    term = HSGPTerm()
    assert term.var_name == "date"
    assert term.name == "hsgp"
    assert term.m is None
    assert term.L is None
    assert term.eta is None
    assert term.ls is None


def test_softplus_defaults():
    term = SoftPlusHSGPTerm()
    assert term.var_name == "date"
    assert term.name == "tvp"
    assert term.m is None


def test_coordinate_build(ds):
    """No-arg term resolves the date coordinate into a per-term index."""
    trend = HSGPTerm(name="trend")
    mu = Intercept("intercept") + trend
    coords = collect_coords(mu, ds=ds)
    assert "date" in coords
    with pm.Model(coords=coords) as model:
        register_data(mu, ds=ds)
        build_param(mu)
        assert trend.index_var == "trend_index"
        assert trend.index_var in model
        assert "trend_m" in model.coords
        assert trend.resolved.m is not None
        assert trend.resolved.L is not None
        assert trend.resolved.X_mid == pytest.approx(
            19.5
        )  # in observation periods, not days
        assert trend.time_resolution == 7  # inferred from the weekly date spacing
        assert trend.time_dim == "date"


def test_numeric_data_var_build(ds_num):
    """A numeric data variable is indexed under its own _index name."""
    trend = HSGPTerm(var_name="time", name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds_num)) as model:
        register_data(trend, ds=ds_num)
        effect = build_param(trend)
        assert isinstance(effect, PTVariable)
        assert trend.index_var == "trend_index"
        assert trend.index_var in model
        assert np.allclose(
            model[trend.index_var].get_value(), np.arange(52, dtype=float)
        )
        assert trend.time_dim == "t"


def test_datetime_data_var_conversion():
    """A datetime data variable (not coordinate) is converted too."""
    n = 30
    ds = xr.Dataset(
        {"when": ("t", pd.date_range("2024-01-01", periods=n, freq="D"))},
        coords={"t": np.arange(n)},
    )
    trend = HSGPTerm(var_name="when", name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds)) as model:
        register_data(trend, ds=ds)
        build_param(trend)
        assert trend.index_var == "trend_index"
        assert np.allclose(
            model[trend.index_var].get_value(), np.arange(n, dtype=float)
        )


def test_time_resolution(ds):
    """time_resolution divides the day offsets."""
    trend = HSGPTerm(name="trend", time_resolution=7)
    with pm.Model(coords=collect_coords(trend, ds=ds)) as model:
        register_data(trend, ds=ds)
        build_param(trend)
        assert trend.resolved.X_mid == pytest.approx(19.5)
        assert np.allclose(model[trend.index_var].get_value(), np.arange(40) * 7 / 7)


@pytest.mark.parametrize("freq", ["W", "D", "2W", "3D"])
def test_time_resolution_inferred_matches_mmm_convention(freq):
    """The time index is in observation periods, matching the MMM convention.

    ``MMM`` sets ``(dates[1] - dates[0]).days`` as its time resolution so the
    numeric index counts periods rather than days. The term must infer the same
    resolution, otherwise the deferred ``m`` / ``L`` / lengthscale heuristics are
    computed on an axis scaled by the sampling cadence.
    """
    dates = pd.date_range("2024-01-01", periods=104, freq=freq)
    ds = xr.Dataset({}, coords={"date": dates})
    expected_res = max(round(float((dates[1] - dates[0]).days)), 1)

    trend = HSGPTerm(name="trend")
    assert trend.time_resolution is None  # not resolved until it sees data

    with pm.Model(coords=collect_coords(trend, ds=ds)) as model:
        register_data(trend, ds=ds)
        build_param(trend)
        index = model[trend.index_var].get_value()

    assert trend.time_resolution == expected_res
    mmm_index = infer_time_index(pd.Series(dates), pd.Series(dates), expected_res)
    np.testing.assert_allclose(index, mmm_index)


def test_deferred_hyperparameters_stay_in_period_units():
    """Deferred ``m`` / ``L`` must not blow up on a realistic weekly dataset.

    With the index in days instead of periods, the default weekly MMM cadence
    resolves ~4000 basis functions for ~100 observations.
    """
    dates = pd.date_range("2021-01-03", periods=104, freq="W")
    ds = xr.Dataset({}, coords={"date": dates})
    trend = HSGPTerm(name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds)):
        register_data(trend, ds=ds)
        build_param(trend)

    assert trend.resolved.m < 104 * 6
    assert trend.resolved.L < 104 * 6


def test_time_resolution_independent_of_row_order():
    """The same dates in any order resolve the same term hyperparameters.

    ``time_resolution`` was inferred from the first two values and the anchor
    fell back to the first element, so a shuffled time reference resolved a
    different resolution, ``X_mid``, and basis size than the sorted one.
    """
    dates = pd.date_range("2024-01-01", periods=40, freq="7D")
    shuffled = np.random.default_rng(0).permutation(dates.values)
    ds_sorted = xr.Dataset({}, coords={"date": dates})
    ds_shuffled = xr.Dataset({}, coords={"date": shuffled})

    def resolved(ds_build):
        term = HSGPTerm(name="trend")
        with pm.Model(coords=collect_coords(term, ds=ds_build)):
            register_data(term, ds=ds_build)
            build_param(term)
        return term

    term_sorted = resolved(ds_sorted)
    term_shuffled = resolved(ds_shuffled)

    assert term_shuffled.time_resolution == term_sorted.time_resolution
    assert term_shuffled.X_mid == pytest.approx(term_sorted.X_mid)
    assert term_shuffled.m == term_sorted.m
    assert term_shuffled.L == pytest.approx(term_sorted.L)


def test_explicit_time_resolution_wins_over_inference():
    """An explicit ``time_resolution`` is never overwritten by inference."""
    dates = pd.date_range("2024-01-01", periods=30, freq="W")
    ds = xr.Dataset({}, coords={"date": dates})
    trend = HSGPTerm(name="trend", time_resolution=1)
    with pm.Model(coords=collect_coords(trend, ds=ds)) as model:
        register_data(trend, ds=ds)
        build_param(trend)
        index = model[trend.index_var].get_value()
    assert trend.time_resolution == 1
    assert index[-1] > 100  # day offsets, as explicitly requested


def test_numeric_reference_keeps_resolution_one(ds_num):
    """A numeric reference is passed through unchanged, with resolution 1."""
    trend = HSGPTerm(var_name="time", name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds_num)):
        register_data(trend, ds=ds_num)
        build_param(trend)
    assert trend.time_resolution == 1


def test_deferred_values_cached(ds):
    """Resolution happens once; later registrations reuse the values."""
    trend = HSGPTerm(name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds)):
        register_data(trend, ds=ds)
        build_param(trend)
    m, L, X_mid = trend.m, trend.L, trend.X_mid
    with pm.Model(coords=collect_coords(trend, ds=ds)):
        register_data(trend, ds=ds)
        build_param(trend)
        assert trend.m == m
        assert trend.L == L
        assert trend.X_mid == X_mid


def test_fitting_does_not_overwrite_the_declared_recipe(ds):
    """Resolution is recorded separately from the declared recipe.

    Filling ``m`` / ``L`` / ``eta`` / ``ls`` / ``X_mid`` back into the
    constructor fields made the first registration silently become the
    term's configuration: after a build the recipe could no longer tell
    user config from learned state, and there was no way back.
    """
    trend = HSGPTerm(name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds)):
        register_data(trend, ds=ds)
        build_param(trend)

    assert trend.m is None
    assert trend.L is None
    assert trend.eta is None
    assert trend.ls is None
    assert trend.X_mid is None

    assert trend.resolved is not None
    assert trend.resolved.m is not None
    assert trend.resolved.X_mid == pytest.approx(19.5)


def test_declared_hyperparameters_win_over_resolution(ds):
    """Explicit values always win during resolution."""
    eta = Prior("Exponential", lam=1)
    ls = Prior("InverseGamma", alpha=2, beta=1)
    trend = HSGPTerm(name="trend", eta=eta, ls=ls, m=15, L=100)
    with pm.Model(coords=collect_coords(trend, ds=ds)):
        register_data(trend, ds=ds)
        build_param(trend)
        assert trend.m == 15
        assert trend.L == 100
        assert trend.eta is eta
        assert trend.ls is ls
        assert trend.resolved.m == 15
        assert trend.resolved.L == 100
        assert trend.resolved.eta is eta
        assert trend.resolved.ls is ls


def test_float_hyperparams(ds):
    """Float eta/ls bypass the priors."""
    trend = HSGPTerm(name="trend", eta=1.0, ls=2.0, m=15, L=100)
    with pm.Model(coords=collect_coords(trend, ds=ds)):
        register_data(trend, ds=ds)
        effect = build_param(trend)
        assert isinstance(effect, PTVariable)


def test_register_data_dedup(ds):
    """Two terms on the same time reference each register their own index."""
    trend = HSGPTerm(name="trend")
    seasonality = HSGPPeriodicTerm(
        name="seasonality",
        scale=Prior("HalfNormal", sigma=1),
        ls=Prior("InverseGamma", alpha=2, beta=1),
        period=52,
        m=20,
    )
    mu = trend + seasonality
    with pm.Model(coords=collect_coords(mu, ds=ds)) as model:
        register_data(mu, ds=ds)
        build_param(mu)
        assert trend.index_var in model
        assert seasonality.index_var in model


def test_terms_sharing_reference_keep_their_own_index(ds):
    """Two terms on one time reference each get their own time index.

    Each term resolves ``time_resolution`` (and the anchor) from its own
    recipe, so a shared index would force one term's basis onto the other's
    units. Separate indexes keep every basis on the axis it was resolved
    for, with no cross-term clobbering.
    """
    trend = HSGPTerm(name="trend", time_resolution=7)
    seasonality = HSGPPeriodicTerm(
        name="seasonality",
        time_resolution=1,
        scale=Prior("HalfNormal", sigma=1),
        ls=Prior("InverseGamma", alpha=2, beta=1),
        period=52,
        m=20,
    )
    mu = trend + seasonality
    with pm.Model(coords=collect_coords(mu, ds=ds)) as model:
        register_data(mu, ds=ds)
        build_param(mu)
        # resolution divides the day offsets: res=7 -> weeks, res=1 -> days
        np.testing.assert_allclose(model[trend.index_var].get_value(), np.arange(40))
        np.testing.assert_allclose(
            model[seasonality.index_var].get_value(), np.arange(40) * 7.0
        )


def test_set_data_after_model_context_exits(ds):
    """``set_data`` with an explicit model works outside the model context.

    The lifecycle docs offer ``set_data`` as a standalone step with an
    explicit ``model`` argument, so it must not require an active ``with``
    block.
    """
    trend = HSGPTerm(name="trend", time_resolution=7)
    with pm.Model(coords=collect_coords(trend, ds=ds)) as model:
        register_data(trend, ds=ds)
        build_param(trend)

    future = xr.Dataset(
        {},
        coords={
            "date": pd.date_range(
                start=ds.coords["date"].values[-1], periods=6, freq="7D"
            )[1:]
        },
    )
    trend.set_data(ds=future, model=model)
    index = model[trend.index_var].get_value()
    # res=7 -> the future window starts one period after the training range
    assert index[0] == pytest.approx(40)


def test_name_collision_raises(ds):
    """Two no-argument terms collide on every prefixed variable."""
    mu = HSGPTerm() + HSGPTerm()
    coords = collect_coords(mu, ds=ds)
    with pm.Model(coords=coords):
        register_data(mu, ds=ds)
        with pytest.raises(ValueError, match="already exists"):
            build_param(mu)


def test_name_collision_hint_names_the_term(ds):
    """The collision error points at the offending term and the fix."""
    mu = HSGPTerm(name="trend") + HSGPTerm(name="trend")
    coords = collect_coords(mu, ds=ds)
    with pm.Model(coords=coords):
        register_data(mu, ds=ds)
        with pytest.raises(ValueError, match="distinct `name=`"):
            build_param(mu)


def test_name_collision_hint_only_for_real_collisions(ds):
    """The naming hint fires on collisions, not unrelated build errors.

    An extra dim missing from the model coords is reported by PyMC with the
    term's basis coordinate in it, which the hint's name-prefix match used
    to swallow into a misleading "variable already exists" message.
    """
    trend = HSGPTerm(name="trend", dims="channel", m=20, L=200, eta=1.0, ls=1.0)
    with pm.Model(coords={"date": ds.coords["date"].values}):
        register_data(trend, ds=ds)
        with pytest.raises(ValueError, match="part of the model coords") as excinfo:
            build_param(trend)
    assert "already exists" not in str(excinfo.value)


def test_non_scalar_prior_raises(ds):
    """Dimensional eta/ls priors are rejected."""
    trend = HSGPTerm(
        name="trend", ls=Prior("InverseGamma", alpha=2, beta=1, dims="channel")
    )
    with pm.Model(coords=collect_coords(trend, ds=ds)):
        register_data(trend, ds=ds)
        with pytest.raises(ValueError, match="must be a scalar"):
            build_param(trend)


def test_create_variable_before_register_raises(ds):
    """create_variable without registration fails clearly."""
    trend = HSGPTerm(name="trend", m=15, L=100, eta=1.0, ls=1.0)
    with pm.Model():
        with pytest.raises(ValueError, match="register_data"):
            trend.create_variable()


def test_set_data_anchored_index(ds):
    """set_data anchors shifted dates to the training first date."""
    trend = HSGPTerm(name="trend", time_resolution=1)
    mu = Intercept("intercept") + trend
    with pm.Model(coords=collect_coords(mu, ds=ds)) as model:
        register_data(mu, ds=ds)
        build_param(mu)
        X_mid = trend.X_mid

        shifted = ds.assign_coords(date=ds.coords["date"].values + pd.Timedelta(days=7))
        set_data(mu, ds=shifted, model=model)
        assert trend.X_mid == X_mid
        np.testing.assert_allclose(
            model[trend.index_var].get_value(), np.arange(40) * 7 + 7
        )
        assert model.coords["date"][0] == shifted.coords["date"].values[0]


def test_set_data_before_register_raises(ds):
    trend = HSGPTerm(name="trend")
    with pm.Model():
        with pytest.raises(ValueError, match="register_data"):
            set_data(trend, ds=ds, model=pm.modelcontext(None))


def test_roundtrip_recipe_keeps_the_training_anchor(ds):
    """A serialized recipe carries the training range and X_mid.

    Without these, a reloaded term re-derives them from whatever window it is
    next given, which silently re-anchors the time index.
    """
    trend = HSGPTerm(name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds)):
        register_data(trend, ds=ds)
        build_param(trend)

    payload = json.loads(json.dumps(serialization.serialize(trend)))
    assert "X_mid" in payload
    assert "first_date" in payload

    restored = HSGPTerm.from_dict(payload)
    assert restored.first_date == trend.first_date
    assert restored.X_mid == trend.X_mid


def test_register_data_refuses_window_before_training_anchor(ds):
    """A window starting before the training anchor is refused.

    The anchor and basis are frozen on the training data. A window that begins
    earlier would need the time index to run negative, outside the learned
    domain, so it is rejected rather than silently extrapolated.
    """
    trend = HSGPTerm(name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds)):
        register_data(trend, ds=ds)
        build_param(trend)
        payload = json.loads(json.dumps(serialization.serialize(trend)))

    # a window entirely before the training start
    before = xr.Dataset(
        {},
        coords={
            "date": pd.date_range(
                end=ds.coords["date"].values[0], periods=4, freq="7D"
            )[:-1]
        },
    )
    restored = HSGPTerm.from_dict(payload)
    with pm.Model(coords=collect_coords(restored, ds=before)):
        with pytest.raises(ValueError, match="before the training"):
            register_data(restored, ds=before)


def test_tvp_media_broadcasting(ds):
    """SoftPlusHSGPTerm() * media broadcasts (date,) over (date, channel)."""
    tvp_media = SoftPlusHSGPTerm(name="tvp") * media_term()
    coords = collect_coords(tvp_media, ds=ds)
    assert "date" in coords
    assert "channel" in coords
    with pm.Model(coords=coords):
        register_data(tvp_media, ds=ds)
        effect = build_param(tvp_media)
        evaluated = effect.eval()
        assert evaluated.shape == (40, 2)


def test_outcome_equation_prior(ds):
    """Full outcome equation: intercept + trend + Named(tvp * media)."""
    trend = HSGPTerm(name="trend", m=20, L=300)
    tvp_media = Named(
        "tvp_media",
        SoftPlusHSGPTerm(name="tvp", m=20, L=300) * media_term(),
        dims=("date", "channel"),
    )
    mu = Intercept("intercept") + trend + tvp_media
    coords = collect_coords(mu, ds=ds)
    with pm.Model(coords=coords):
        register_data(mu, ds=ds)
        build_param(mu)
        idata = pm.sample_prior_predictive(
            random_seed=42, var_names=["tvp", "tvp_media"]
        )

    # the GP factor is positive with mean one over the time dim
    gp = idata.prior["tvp"]
    assert gp.dims == ("chain", "draw", "date")
    gp_values = gp.values
    assert (gp_values > 0).all()
    assert np.allclose(gp_values.mean(axis=-1), 1.0)

    # the product broadcasts across the channel dim
    product = idata.prior["tvp_media"]
    assert product.dims == ("chain", "draw", "date", "channel")
    assert (product.values > 0).all()


def test_higher_dim_coefs(ds):
    """dims adds one GP curve per group."""
    trend = HSGPTerm(name="trend", dims="channel", m=20, L=300, eta=1.0, ls=2.0)
    with pm.Model(coords=collect_coords(trend, ds=ds)) as model:
        register_data(trend, ds=ds)
        effect = build_param(trend)
        assert effect.eval().shape == (40, 2)
        assert "trend_hsgp_coefs" in model.named_vars


def test_periodic_requires_args():
    with pytest.raises(TypeError):
        HSGPPeriodicTerm()


def test_periodic_build(ds):
    seasonality = HSGPPeriodicTerm(
        name="seasonality",
        scale=Prior("HalfNormal", sigma=1),
        ls=Prior("InverseGamma", alpha=2, beta=1),
        period=52,
        m=20,
    )
    with pm.Model(coords=collect_coords(seasonality, ds=ds)) as model:
        register_data(seasonality, ds=ds)
        effect = build_param(seasonality)
        assert isinstance(effect, PTVariable)
        assert len(model.coords["seasonality_m"]) == 20 * 2 - 1


def test_string_cov_func_roundtrips():
    """A string ``cov_func`` is coerced and serializes like the enum.

    ``HSGP`` accepts a string covariance name, and recipes are serialized at
    sampling time after a fit, so a string passed at construction must not
    crash ``to_dict``.
    """
    term = HSGPTerm(cov_func="matern52", m=15, L=100, eta=1.0, ls=1.0)
    assert term.cov_func is CovFunc.Matern52
    restored = _roundtrip(term)
    assert restored.cov_func is CovFunc.Matern52


def test_serialize_hsgp_term_roundtrip():
    term = HSGPTerm(
        var_name="time",
        name="trend",
        eta=Prior("Exponential", lam=1),
        m=15,
        L=100,
        dims=("channel",),
    )
    restored = serialization.deserialize(serialization.serialize(term))
    assert restored == term
    assert restored.dims == ("channel",)


@pytest.mark.parametrize("dims", ["channel", ("channel", "product")], ids=str)
def test_serialize_json_roundtrip_dims(dims):
    """string and tuple dims both survive a JSON hop as tuples."""
    term = HSGPTerm(name="trend", dims=dims, m=15, L=100)
    assert term.dims == (("channel",) if dims == "channel" else dims)
    data = json.loads(json.dumps(serialization.serialize(term)))
    restored = serialization.deserialize(data)
    assert restored == term
    assert restored.dims == (("channel",) if dims == "channel" else dims)


@pytest.mark.parametrize("term_cls", [SoftPlusHSGPTerm, HSGPTerm])
def test_serialize_json_roundtrip_no_dims(term_cls):
    """JSON hop with no dims configured."""
    term = term_cls(name="trend", m=15, L=100)
    data = json.loads(json.dumps(serialization.serialize(term)))
    restored = serialization.deserialize(data)
    assert restored == term
    assert restored.dims is None


def test_serialize_deferred_roundtrip():
    term = HSGPTerm()
    restored = serialization.deserialize(serialization.serialize(term))
    assert restored.var_name == "date"
    assert restored.name == "hsgp"
    assert restored.m is None
    assert restored.X_mid is None


def test_serialize_resolved_keeps_m_l(ds):
    """Resolved m/L persist, and the frozen training state round-trips."""
    term = HSGPTerm(name="trend", m=15, L=100, eta=1.0, ls=1.0)
    with pm.Model(coords=collect_coords(term, ds=ds)):
        register_data(term, ds=ds)
        build_param(term)
    data = serialization.serialize(term)
    assert data["m"] == 15
    assert data["L"] == 100
    assert data["X_mid"] == term.X_mid
    assert data["first_date"] is not None
    restored = serialization.deserialize(data)
    assert restored.m == 15
    assert restored.L == 100
    assert restored.X_mid == term.X_mid
    assert restored.first_date == term.first_date
    assert restored.last_date == term.last_date


def test_serialize_softplus_roundtrip():
    term = SoftPlusHSGPTerm(m=15, L=100)
    restored = serialization.deserialize(serialization.serialize(term))
    assert restored == term
    assert type(restored).__name__ == "SoftPlusHSGPTerm"


def test_serialize_periodic_roundtrip():
    term = HSGPPeriodicTerm(
        scale=Prior("HalfNormal", sigma=1),
        ls=Prior("InverseGamma", alpha=2, beta=1),
        period=52,
        m=20,
    )
    restored = serialization.deserialize(serialization.serialize(term))
    assert restored == term


def test_serialize_composition_roundtrip(ds):
    expr = HSGPTerm(name="trend", m=15, L=100) + Intercept("intercept")
    restored = serialization.deserialize(serialization.serialize(expr))
    assert isinstance(restored, Sum)
    assert isinstance(restored.terms[0], HSGPTerm)


def _trained(term, ds):
    """Register and build ``term`` on ``ds``.

    Returns ``(term, index)`` where ``index`` is the registered time index.
    """
    with pm.Model(coords=collect_coords(term, ds=ds)) as model:
        register_data(term, ds=ds)
        build_param(term)
        index = model[term.index_var].get_value().copy()
    return term, index


def _roundtrip(term):
    """Round-trip a term through JSON, as saving and reloading a model does."""
    payload = json.loads(json.dumps(serialization.serialize(term)))
    return serialization.deserialize(payload)


def test_trained_hsgp_term_roundtrips_equal(ds):
    """A trained HSGPTerm keeps its frozen training state through a round-trip.

    Every ``init=False`` field is part of dataclass equality, so this is the
    round-trip check for the frozen state (``X_mid``, the date anchors, the
    inferred resolution and ``time_dim``). Registering first is what makes it
    meaningful: an untrained term has every field ``None`` on both sides and
    compares equal vacuously.

    The frozen fields are compared individually rather than via whole-dataclass
    ``==`` because the deferred ``eta`` / ``ls`` priors resolve to a
    ``Prior`` holding a resolved hyperparameter, and ``Prior.__eq__`` is not
    safe for those (pymc_extras).
    """
    term, _ = _trained(HSGPTerm(name="g", eta=1.0, ls=1.0), ds)
    restored = _roundtrip(term)
    for field in ("X_mid", "time_dim", "time_resolution"):
        assert getattr(restored, field) == getattr(term, field)
    assert restored.first_date == term.first_date
    assert restored.last_date == term.last_date
    assert restored == term  # explicit scalar priors are equality-safe


def test_deferred_term_roundtrips_frozen_state(ds):
    """A deferred term's frozen state survives the round-trip.

    The deferred ``eta`` / ``ls`` are excluded from the equality check (see
    ``test_trained_hsgp_term_roundtrips_equal``) but must still be present and
    equivalent on the restored term.
    """
    term, _ = _trained(HSGPTerm(name="g"), ds)
    assert term.resolved is not None
    assert term.resolved.m is not None and term.resolved.ls is not None
    restored = _roundtrip(term)
    assert restored.resolved is not None
    for field in ("X_mid", "m", "L"):
        assert getattr(restored.resolved, field) == getattr(term.resolved, field)
    assert restored.time_dim == term.time_dim
    assert restored.first_date == term.first_date
    assert restored.last_date == term.last_date


def test_trained_periodic_term_roundtrips_equal(ds):
    """A trained HSGPPeriodicTerm round-trips its frozen training state too."""
    term, _ = _trained(
        HSGPPeriodicTerm(name="g", scale=1.0, ls=1.0, period=52, m=10), ds
    )
    for field in ("X_mid", "time_dim", "time_resolution"):
        assert getattr(_roundtrip(term), field) == getattr(term, field)


def test_restored_term_rebuilds_identically(ds):
    """A restored term rebuilds a model with the same variables and time index."""
    original, expected_index = _trained(HSGPTerm(name="trend"), ds)
    with pm.Model(coords=collect_coords(original, ds=ds)) as model:
        register_data(original, ds=ds)
        build_param(original)
        expected_vars = set(model.named_vars)

    restored = _roundtrip(original)
    with pm.Model(coords=collect_coords(restored, ds=ds)) as model:
        register_data(restored, ds=ds)
        build_param(restored)
        assert set(model.named_vars) == expected_vars
        np.testing.assert_allclose(
            model[restored.index_var].get_value(), expected_index
        )


def test_restored_term_set_data(ds):
    """A restored term can update the time index for prediction.

    ``time_dim`` gates ``set_data``; if it did not survive serialization, a
    reloaded term could not be used for out-of-sample prediction at all.
    """
    future = xr.Dataset(
        {},
        coords={
            "date": pd.date_range(
                start=ds.coords["date"].values[-1], periods=6, freq="7D"
            )[1:]
        },
    )
    original, training_index = _trained(HSGPTerm(name="trend"), ds)

    restored = _roundtrip(original)
    with pm.Model(coords=collect_coords(restored, ds=ds)) as model:
        register_data(restored, ds=ds)
        build_param(restored)
        set_data(restored, ds=future, model=model)
        index = model[restored.index_var].get_value()

    assert index[0] == pytest.approx(training_index[-1] + 1)


def test_restored_term_rebuild_on_training_window_matches_in_model(ds):
    """Reload contract: rebuild on the training window, then ``set_data``.

    A restored term rebuilt on the recorded training data must produce the
    same curve as the original in-model flow (rebuild, then ``set_data`` for
    the prediction window). This is the reference behavior the reload
    contract guarantees.
    """
    future = xr.Dataset(
        {},
        coords={
            "date": pd.date_range(
                start=ds.coords["date"].values[-1], periods=6, freq="7D"
            )[1:]
        },
    )

    def curve(term, ds_build, predict):
        """Draw from a (possibly restored) term, optionally after ``set_data``."""
        with pm.Model(coords=collect_coords(term, ds=ds_build)) as model:
            register_data(term, ds=ds_build)
            variable = build_param(term)
            if predict:
                set_data(term, ds=future, model=model)
            return pm.draw(variable, draws=5, random_seed=7)

    trained = HSGPTerm(name="trend")
    reference = curve(trained, ds, predict=True)
    restored = _roundtrip(trained)
    np.testing.assert_allclose(curve(restored, ds, predict=True), reference)


def test_register_data_on_fitted_term_refuses_other_window(ds):
    """A fitted term only rebuilds on its recorded training window.

    Rebuilding on any other window (a future-only window, or train + future)
    would center and size the basis on data the term was never fit with, and
    the resulting GP is silently different from the trained one. The reload
    contract is: rebuild on the training window, then ``set_data`` for
    prediction windows.
    """
    trained, _ = _trained(HSGPTerm(name="trend"), ds)
    restored = _roundtrip(trained)

    future = xr.Dataset(
        {},
        coords={
            "date": pd.date_range(
                start=ds.coords["date"].values[-1], periods=6, freq="7D"
            )[1:]
        },
    )
    with pm.Model(coords=collect_coords(restored, ds=future)):
        with pytest.raises(ValueError, match="training window"):
            register_data(restored, ds=future)

    # train + future is refused too: only the exact training window rebuilds
    extended = xr.Dataset(
        {},
        coords={
            "date": pd.date_range(
                start=ds.coords["date"].values[0],
                periods=len(ds.coords["date"]) + 6,
                freq="7D",
            )
        },
    )
    restored = _roundtrip(trained)
    with pm.Model(coords=collect_coords(restored, ds=extended)):
        with pytest.raises(ValueError, match="training window"):
            register_data(restored, ds=extended)


def test_set_data_on_restored_term_without_time_dim_explains(ds):
    """A recipe missing ``time_dim`` gets an actionable error, not a wrong model."""
    term, _ = _trained(HSGPTerm(name="trend"), ds)
    payload = json.loads(json.dumps(serialization.serialize(term)))
    del payload["time_dim"]  # a recipe saved before time_dim was recorded

    restored = HSGPTerm.from_dict(payload)
    future = xr.Dataset({}, coords={"date": ds.coords["date"].values[-1:]})
    with pm.Model():
        with pytest.raises(ValueError, match="does not record its time dimension"):
            set_data(restored, ds=future, model=pm.modelcontext(None))


def test_multi_dim_term_roundtrips_frozen_state(ds_product):
    """A GP over (date, product) keeps its frozen state and its dims.

    ``dims`` adds one GP curve per group, so the extra dimension is part of the
    recipe and must survive a round-trip as a tuple. The time index itself
    stays one-dimensional: only the coefficients are per-product.
    """
    term, index = _trained(
        HSGPTerm(name="trend", dims="product", m=20, L=200, eta=1.0, ls=1.0),
        ds_product,
    )
    restored = _roundtrip(term)

    for field in ("X_mid", "time_dim", "time_resolution", "m", "L"):
        assert getattr(restored, field) == getattr(term, field)
    assert restored.first_date == term.first_date
    assert restored.last_date == term.last_date
    assert restored.dims == ("product",)
    assert restored.extra_coords == {"product": ["EU", "US", "JP"]}
    assert index.shape == (len(ds_product.coords["date"]),)  # time axis is shared


def test_register_data_on_fitted_term_refuses_reordered_coords(ds_product):
    """A fitted multi-dim term refuses a rebuilt window with reordered coords.

    Each extra dim gets one GP curve per coordinate, positioned by order. A
    rebuild on reordered coordinates would silently report each curve under
    the wrong label, so it is refused wherever the window is registered.
    """
    original, _ = _trained(
        HSGPTerm(name="trend", dims="product", m=20, L=200, eta=1.0, ls=1.0),
        ds_product,
    )
    restored = _roundtrip(original)

    reordered = ds_product.assign_coords(product=["JP", "EU", "US"])
    with pm.Model(coords=collect_coords(restored, ds=reordered)):
        with pytest.raises(ValueError, match="reordered"):
            register_data(restored, ds=reordered)


def test_multi_dim_term_rebuilds_with_one_curve_per_product(ds_product):
    """The reloaded multi-dim term rebuilds the same per-product curves."""
    original, _ = _trained(
        HSGPTerm(name="trend", dims="product", m=20, L=200, eta=1.0, ls=1.0),
        ds_product,
    )
    with pm.Model(coords=collect_coords(original, ds=ds_product)) as model:
        register_data(original, ds=ds_product)
        expected_shape = build_param(original).eval().shape
        expected_vars = set(model.named_vars)

    restored = _roundtrip(original)
    with pm.Model(coords=collect_coords(restored, ds=ds_product)) as model:
        register_data(restored, ds=ds_product)
        effect = build_param(restored)
        assert set(model.named_vars) == expected_vars
        assert effect.eval().shape == expected_shape
    assert effect.eval().shape == (
        len(ds_product.coords["date"]),
        len(ds_product.coords["product"]),
    )


def test_multi_dim_term_set_data(ds_product):
    """A reloaded multi-dim term predicts out of sample on the training axis."""
    future = xr.Dataset(
        {},
        coords={
            "date": pd.date_range(
                start=ds_product.coords["date"].values[-1], periods=5, freq="7D"
            )[1:],
            "product": ds_product.coords["product"].values,
        },
    )
    original, training_index = _trained(
        HSGPTerm(name="trend", dims="product", m=20, L=200, eta=1.0, ls=1.0),
        ds_product,
    )

    restored = _roundtrip(original)
    with pm.Model(coords=collect_coords(restored, ds=ds_product)) as model:
        register_data(restored, ds=ds_product)
        build_param(restored)
        set_data(restored, ds=future, model=model)
        index = model[restored.index_var].get_value()

    # the shared time index continues past training, per-product curves intact
    assert index[0] == pytest.approx(training_index[-1] + 1)
    assert index.ndim == 1


def test_multi_dim_term_refuses_unseen_products(ds_product):
    """Predicting on products the GP was not trained for is refused.

    A GP curve per product cannot be extrapolated to an unseen product, so the
    coordinate mismatch must not pass silently.
    """
    term, _ = _trained(
        HSGPTerm(name="trend", dims="product", m=20, L=200, eta=1.0, ls=1.0),
        ds_product,
    )
    unseen = xr.Dataset(
        {},
        coords={
            "date": pd.date_range(
                start=ds_product.coords["date"].values[-1], periods=3, freq="7D"
            )[1:],
            "product": ["EU", "US", "JP", "BR"],
        },
    )
    with pm.Model(coords=collect_coords(term, ds=ds_product)) as model:
        register_data(term, ds=ds_product)
        build_param(term)
        with pytest.raises(ValueError, match="does not match the training data"):
            set_data(term, ds=unseen, model=model)


def test_multi_dim_term_refuses_reordered_products(ds_product):
    """A reordered product set is refused rather than silently mislabelled.

    Each extra-dim coordinate gets its own GP curve, positioned by order, so
    reordering the prediction window would report the curve trained on one
    product under another product's label.
    """
    term, _ = _trained(
        HSGPTerm(name="trend", dims="product", m=20, L=200, eta=1.0, ls=1.0),
        ds_product,
    )
    reordered = xr.Dataset(
        {},
        coords={
            "date": pd.date_range(
                start=ds_product.coords["date"].values[-1], periods=3, freq="7D"
            )[1:],
            "product": ["JP", "EU", "US"],
        },
    )
    with pm.Model(coords=collect_coords(term, ds=ds_product)) as model:
        register_data(term, ds=ds_product)
        build_param(term)
        with pytest.raises(ValueError, match="does not match the training data"):
            set_data(term, ds=reordered, model=model)


def test_multi_dim_term_refuses_missing_product_coord(ds_product):
    """A prediction window without the product coordinate is refused."""
    term, _ = _trained(
        HSGPTerm(name="trend", dims="product", m=20, L=200, eta=1.0, ls=1.0),
        ds_product,
    )
    no_product = xr.Dataset(
        {},
        coords={
            "date": pd.date_range(
                start=ds_product.coords["date"].values[-1], periods=3, freq="7D"
            )[1:]
        },
    )
    with pm.Model(coords=collect_coords(term, ds=ds_product)) as model:
        register_data(term, ds=ds_product)
        build_param(term)
        with pytest.raises(ValueError, match="no 'product' coordinate"):
            set_data(term, ds=no_product, model=model)


def test_multi_dim_term_rejects_bad_products_after_reload(ds_product):
    """The extra-dim guard survives serialization.

    ``extra_coords`` is the only record of which coordinates the curves were
    fit on, so it has to round-trip for the guard to work on a reloaded term.
    """
    term, _ = _trained(
        HSGPTerm(name="trend", dims="product", m=20, L=200, eta=1.0, ls=1.0),
        ds_product,
    )
    restored = _roundtrip(term)
    assert restored.extra_coords == {"product": ["EU", "US", "JP"]}

    reordered = xr.Dataset(
        {},
        coords={
            "date": pd.date_range(
                start=ds_product.coords["date"].values[-1], periods=3, freq="7D"
            )[1:],
            "product": ["US", "EU", "JP"],
        },
    )
    with pm.Model(coords=collect_coords(restored, ds=ds_product)) as model:
        register_data(restored, ds=ds_product)
        build_param(restored)
        with pytest.raises(ValueError, match="does not match the training data"):
            set_data(restored, ds=reordered, model=model)


def _sample_prior_both(term, existing):
    """Sample priors from the existing HSGP class and the term; compare."""
    n = 52
    X = np.arange(n, dtype=float)
    coords = {"time": np.arange(n)}
    ds = xr.Dataset({}, coords=coords)

    with pm.Model(coords=coords):
        existing.register_data(X)
        existing.create_variable("f", xdist=True)
        idata_existing = pm.sample_prior_predictive(random_seed=42)

    with pm.Model(coords=collect_coords(term, ds=ds)):
        register_data(term, ds=ds)
        build_param(term)
        idata_term = pm.sample_prior_predictive(random_seed=42)

    return (
        idata_existing.prior["f"].values,
        idata_term.prior["f"].values,
    )


def test_equivalence_hsgp():
    """Exact prior draws match the existing HSGP class."""
    eta = Prior("Exponential", lam=1)
    ls = Prior("InverseGamma", alpha=2, beta=1)
    existing = HSGP(eta=eta, ls=ls, m=20, L=150, dims="time")
    term = HSGPTerm(var_name="time", name="f", eta=eta, ls=ls, m=20, L=150)
    draws_existing, draws_term = _sample_prior_both(term, existing)
    np.testing.assert_allclose(draws_existing, draws_term)


def test_equivalence_periodic():
    """Exact prior draws match the existing HSGPPeriodic class."""
    scale = Prior("HalfNormal", sigma=1)
    ls = Prior("InverseGamma", alpha=2, beta=1)
    existing = HSGPPeriodic(
        scale=scale, ls=ls, m=20, cov_func="periodic", period=52, dims="time"
    )
    term = HSGPPeriodicTerm(
        var_name="time", name="f", scale=scale, ls=ls, period=52, m=20
    )
    draws_existing, draws_term = _sample_prior_both(term, existing)
    np.testing.assert_allclose(draws_existing, draws_term)


def test_time_index_parity_with_mmm_training_index(ds):
    """The term's time index matches MMM's own training index.

    ``MMM`` builds its latent-process time index as ``np.arange(n)`` (see
    ``MMM._time_index``) and sets ``(dates[1] - dates[0]).days`` as the time
    resolution, so a ``time_varying_media`` HSGP sees periods. The term must
    derive the same axis, otherwise the deferred hyperparameters -- and the
    resulting media multiplier -- do not match the MMM it is meant to mirror.
    """
    dates = pd.DatetimeIndex(ds.coords["date"].values)
    n = len(dates)
    mmm_index = np.arange(n)
    mmm_resolution = (dates[1] - dates[0]).days

    trend = HSGPTerm(name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds)) as model:
        register_data(trend, ds=ds)
        build_param(trend)
        index = model[trend.index_var].get_value()

    assert trend.time_resolution == mmm_resolution
    np.testing.assert_allclose(index, mmm_index)


def test_mmm_style_media_contribution(ds):
    """The canonical MMM media composition builds a per-channel contribution.

    ``MMM`` computes ``channel_contribution = baseline * media_latent_process``,
    where the latent process is a ``SoftPlusHSGP`` over the time index that
    broadcasts across the channel dimension. The term equivalent --
    ``SoftPlusHSGPTerm() * Dot(media)`` -- must build the same way, with a
    strictly positive, mean-one multiplier so the composition is meaningful.
    """
    media = Dot(var_name="media", prior=Prior("Normal", dims="channel"))
    tvp_media = SoftPlusHSGPTerm() * media
    mu = Intercept("intercept") + tvp_media
    with pm.Model(coords=collect_coords(mu, ds=ds)) as model:
        register_data(mu, ds=ds)
        build_param(mu)
        # the multiplier is a named GP output alongside the media coefficients
        assert "tvp" in model.named_vars
        assert "media_beta" in model.named_vars
        multiplier = (
            pm.sample_prior_predictive(random_seed=7, draws=5, var_names=["tvp"])
            .prior["tvp"]
            .values
        )

    # strictly positive, and mean one over the time dimension
    assert (multiplier > 0).all()
    np.testing.assert_allclose(multiplier.mean(axis=-1), 1.0, atol=1e-6)


def test_equivalence_softplus():
    """Exact prior draws match the existing SoftPlusHSGP class."""
    eta = Prior("Exponential", lam=1)
    ls = Prior("InverseGamma", alpha=2, beta=1)
    existing = SoftPlusHSGP(eta=eta, ls=ls, m=20, L=150, dims="time")
    term = SoftPlusHSGPTerm(var_name="time", name="f", eta=eta, ls=ls, m=20, L=150)
    draws_existing, draws_term = _sample_prior_both(term, existing)
    np.testing.assert_allclose(draws_existing, draws_term)


def test_defaults_match_parameterize_from_data():
    """Deferred resolution mirrors HSGP.parameterize_from_data assumptions."""
    n = 52
    X = np.arange(n, dtype=float)
    expected = HSGP.parameterize_from_data(X=X, dims="time")

    ds = xr.Dataset({}, coords={"time": X})
    term = HSGPTerm(var_name="time", name="f")
    with pm.Model(coords=collect_coords(term, ds=ds)):
        register_data(term, ds=ds)
        build_param(term)

    assert term.resolved.m == expected.m
    assert term.resolved.L == expected.L
    assert term.resolved.X_mid == expected.X_mid
    assert type(term.resolved.eta).__name__ == "Prior"
    assert type(term.resolved.ls).__name__ == "Prior"


def test_set_data_refuses_window_before_training_anchor(ds):
    """set_data enforces the anchor, not only register_data.

    The anchor is frozen on the training data, so a fitted term must refuse a
    prediction window that begins earlier: it would place the time index before
    zero, outside the learned basis, and extrapolate silently.
    """
    trend = HSGPTerm(name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds)) as model:
        register_data(trend, ds=ds)
        build_param(trend)
        anchor = trend.first_date

        before = xr.Dataset(
            {},
            coords={
                "date": pd.date_range(
                    pd.Timestamp(anchor) - pd.Timedelta(weeks=6),
                    periods=12,
                    freq="W-MON",
                )
            },
        )
        with pytest.raises(ValueError, match="before the training anchor"):
            set_data(trend, ds=before, model=model)

        # the shared index is untouched by the rejected window
        assert np.asarray(model[trend.index_var].get_value()).min() == 0.0


def test_set_data_allows_window_extending_past_training(ds):
    """A window that extends the training range is the normal case, and works."""
    trend = HSGPTerm(name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds)) as model:
        register_data(trend, ds=ds)
        build_param(trend)

        future = xr.Dataset(
            {},
            coords={
                "date": pd.date_range(
                    ds.coords["date"].values[-1], periods=8, freq="W-MON"
                )
            },
        )
        set_data(trend, ds=future, model=model)
        assert np.asarray(model[trend.index_var].get_value()).max() > 0.0


def test_shared_time_reference_with_matching_resolution_builds(ds):
    """Sharing a reference with matching resolutions builds cleanly."""
    trend = HSGPTerm(name="trend", time_resolution=7)
    season = SoftPlusHSGPTerm(name="season", time_resolution=7)
    mu = Intercept("intercept") + trend + season

    with pm.Model(coords=collect_coords(mu, ds=ds)) as model:
        register_data(mu, ds=ds)
        build_param(mu)

    assert trend.index_var in model
    assert season.index_var in model
    assert trend.X_mid == pytest.approx(season.X_mid)


def test_anchor_uses_earliest_date_not_first_element():
    """An unsorted time reference anchors on its earliest date."""
    base = pd.date_range("2023-01-02", periods=40, freq="W-MON")
    order = [
        20,
        3,
        39,
        11,
        0,
        25,
        7,
        33,
        15,
        1,
        28,
        19,
        35,
        9,
        30,
        12,
        22,
        5,
        38,
        17,
        26,
        2,
        31,
        14,
        23,
        8,
        37,
        18,
        27,
        6,
        34,
        13,
        24,
        10,
        36,
        16,
        29,
        4,
        32,
        21,
    ]
    shuffled = xr.Dataset(
        {},
        coords={"date": xr.DataArray(base[order].values, dims="date")},
    )
    # explicit m/L so the test exercises the anchor, not the data-driven
    # recommendation (which assumes a sorted, evenly spaced index).
    trend = HSGPTerm(name="trend", var_name="date", m=10, L=30)
    with pm.Model(coords=collect_coords(trend, ds=shuffled)):
        register_data(trend, ds=shuffled)
    assert trend.first_date == base.min()
    assert trend.last_date == base.max()


def test_string_extra_coords_survive_serialization(ds):
    """Numeric-looking string coords stay strings after a reload.

    ``np.datetime64`` parses far more than ISO timestamps (``"12"`` becomes
    year 12, ``"NaT"`` becomes not-a-time, ``"today"`` becomes today), so a
    string coordinate must not be re-typed by guesswork on deserialize: the
    restored term would then refuse the identical string coordinate.
    """
    data = ds.assign_coords(store=["12", "2020"])
    original, _ = _trained(
        HSGPTerm(name="trend", dims="store", m=20, L=200, eta=1.0, ls=1.0),
        data,
    )
    restored = _roundtrip(original)
    assert restored.extra_coords == {"store": ["12", "2020"]}
    assert all(isinstance(v, str) for v in restored.extra_coords["store"])

    # and the restored term accepts the identical string coordinate
    with pm.Model(coords=collect_coords(restored, ds=data)):
        register_data(restored, ds=data)
        build_param(restored)


def test_frozen_deterministics_collected_for_forecast(ds):
    """Forecasting a SoftPlus term keeps the training mean of one.

    ``{name}_f_mean`` is a mean over the time dimension; recomputing it on a
    prediction window renormalizes every draw to the new window and erases
    the time variation. Collecting the term's frozen deterministics across a
    recipe and replacing them with ``deterministics_to_flat`` keeps the
    training normalization instead.
    """
    recipe = SoftPlusHSGPTerm(name="tvp") * media_term()
    with pm.Model(coords=collect_coords(recipe, ds=ds)) as model:
        register_data(recipe, ds=ds)
        build_param(recipe)

    assert frozen_deterministics(recipe) == ["tvp_f_mean"]

    n_channel = len(ds.coords["channel"])
    future_dates = pd.date_range(
        start=ds.coords["date"].values[-1], periods=8, freq="7D"
    )[1:]
    future = xr.Dataset(
        {
            "media": (
                ("date", "channel"),
                np.ones((len(future_dates), n_channel)),
            ),
        },
        coords={"date": future_dates},
    )

    def forecast(freeze):
        """Draw the tvp multiplier on the future window, optionally frozen."""
        if freeze:
            with model:
                f_mean = pm.draw(model["tvp_f_mean"], draws=20, random_seed=7)
            mock = xr.Dataset(
                {"tvp_f_mean": (("chain", "draw"), f_mean[None, :])},
                coords={"chain": [0], "draw": np.arange(20)},
            )
            target = deterministics_to_flat(model, ["tvp_f_mean"])
            with target:
                set_data(recipe, ds=future, model=target)
                draws = (
                    pm.sample_posterior_predictive(
                        mock, model=target, var_names=["tvp"], random_seed=7
                    )
                    .posterior_predictive["tvp"]
                    .values.reshape(20, -1)
                )
        else:
            with model:
                set_data(recipe, ds=future, model=model)
                draws = pm.draw(model["tvp"], draws=20, random_seed=7)
        return np.abs(draws.mean(axis=1) - 1.0)  # deviation from mean one

    # without freezing, every draw renormalizes to exactly mean one over the
    # prediction window: the time variation is erased
    np.testing.assert_allclose(forecast(freeze=False), 0.0, atol=1e-10)

    # with the frozen deterministic, the training normalization survives
    assert forecast(freeze=True).std() > 0


def test_datetime_extra_coord_survives_serialization():
    """A datetime extra dim round-trips instead of becoming an epoch integer."""
    dates = pd.date_range("2023-01-02", periods=20, freq="W-MON")
    monthly = pd.date_range("2023-01-01", periods=3, freq="MS")
    data = xr.Dataset(
        {},
        coords={
            "date": dates,
            "month": xr.DataArray(
                np.array(monthly, dtype="datetime64[ns]"), dims="month"
            ),
        },
    )
    term = HSGPPeriodicTerm(
        name="periodic",
        var_name="date",
        period=52,
        m=10,
        scale=1.0,
        ls=1.0,
        dims="month",
    )
    with pm.Model(coords=collect_coords(term, ds=data)):
        register_data(term, ds=data)
        payload = json.loads(json.dumps(serialization.serialize(term)))

    assert all(isinstance(v, str) for v in payload["extra_coords"]["month"])
    restored = serialization.deserialize(payload)
    assert restored.extra_coords["month"] == [
        np.datetime64("2023-01-01"),
        np.datetime64("2023-02-01"),
        np.datetime64("2023-03-01"),
    ]

    # and the restored term accepts the identical datetime coordinate
    with pm.Model(coords=collect_coords(restored, ds=data)):
        register_data(restored, ds=data)
        set_data(restored, ds=data, model=pm.modelcontext(None))
