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

from pymc_marketing.mmm.hsgp import HSGP, HSGPPeriodic, SoftPlusHSGP
from pymc_marketing.serialization import serialization
from pymc_marketing.terms import (
    Intercept,
    ModelTerm,
    Named,
    Sum,
    build_param,
    collect_coords,
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
    """No-arg term resolves the date coordinate into date_index."""
    trend = HSGPTerm(name="trend")
    mu = Intercept("intercept") + trend
    coords = collect_coords(mu, ds=ds)
    assert "date" in coords
    with pm.Model(coords=coords) as model:
        register_data(mu, ds=ds)
        build_param(mu)
        assert trend.index_var == "date_index"
        assert "date_index" in model
        assert "trend_m" in model.coords
        assert trend.m is not None
        assert trend.L is not None
        assert trend.X_mid == pytest.approx(19.5 * 7)
        assert trend.time_dim == "date"


def test_numeric_data_var_build(ds_num):
    """A numeric data variable is indexed under its own _index name."""
    trend = HSGPTerm(var_name="time", name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds_num)) as model:
        register_data(trend, ds=ds_num)
        effect = build_param(trend)
        assert isinstance(effect, PTVariable)
        assert trend.index_var == "time_index"
        assert "time_index" in model
        assert np.allclose(model["time_index"].get_value(), np.arange(52, dtype=float))
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
        assert trend.index_var == "when_index"
        assert np.allclose(model["when_index"].get_value(), np.arange(n, dtype=float))


def test_time_resolution(ds):
    """time_resolution divides the day offsets."""
    trend = HSGPTerm(name="trend", time_resolution=7)
    with pm.Model(coords=collect_coords(trend, ds=ds)) as model:
        register_data(trend, ds=ds)
        build_param(trend)
        assert trend.X_mid == pytest.approx(19.5)
        assert np.allclose(model["date_index"].get_value(), np.arange(40) * 7 / 7)


def test_deferred_values_cached(ds):
    """Resolution happens once; later registrations reuse the values."""
    trend = HSGPTerm(name="trend")
    with pm.Model(coords=collect_coords(trend, ds=ds)):
        register_data(trend, ds=ds)
        build_param(trend)
    m, L, X_mid = trend.m, trend.L, trend.X_mid

    shifted = ds.assign_coords(date=ds.coords["date"].values + pd.Timedelta(days=1))
    with pm.Model(coords=collect_coords(trend, ds=shifted)):
        register_data(trend, ds=shifted)
        build_param(trend)
        assert trend.m == m
        assert trend.L == L
        assert trend.X_mid == X_mid


def test_explicit_values_win(ds):
    """Explicit m/L/eta/ls are kept during resolution."""
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


def test_float_hyperparams(ds):
    """Float eta/ls bypass the priors."""
    trend = HSGPTerm(name="trend", eta=1.0, ls=2.0, m=15, L=100)
    with pm.Model(coords=collect_coords(trend, ds=ds)):
        register_data(trend, ds=ds)
        effect = build_param(trend)
        assert isinstance(effect, PTVariable)


def test_register_data_dedup(ds):
    """Two terms on the same time reference share one data variable."""
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
        assert "date_index" in model


def test_name_collision_raises(ds):
    """Two no-argument terms collide on every prefixed variable."""
    mu = HSGPTerm() + HSGPTerm()
    coords = collect_coords(mu, ds=ds)
    with pm.Model(coords=coords):
        register_data(mu, ds=ds)
        with pytest.raises(ValueError, match="already exists"):
            build_param(mu)


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
        assert np.allclose(model["date_index"].get_value(), np.arange(40) * 7 + 7)
        assert model.coords["date"][0] == shifted.coords["date"].values[0]


def test_set_data_before_register_raises(ds):
    trend = HSGPTerm(name="trend")
    with pm.Model():
        with pytest.raises(ValueError, match="register_data"):
            set_data(trend, ds=ds, model=pm.modelcontext(None))


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
    """Resolved m/L persist; X_mid stays excluded."""
    term = HSGPTerm(name="trend", m=15, L=100, eta=1.0, ls=1.0)
    with pm.Model(coords=collect_coords(term, ds=ds)):
        register_data(term, ds=ds)
        build_param(term)
    data = serialization.serialize(term)
    assert data["m"] == 15
    assert data["L"] == 100
    assert "X_mid" not in data
    restored = serialization.deserialize(data)
    assert restored.m == 15
    assert restored.L == 100


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

    assert term.m == expected.m
    assert term.L == expected.L
    assert term.X_mid == expected.X_mid
    assert type(term.eta).__name__ == "Prior"
    assert type(term.ls).__name__ == "Prior"
