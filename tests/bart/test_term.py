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
"""Tests for the ``pymc_marketing.bart`` term."""

import sys
from typing import Any

import numpy as np
import pytest
import xarray as xr
from pymc_bart.split_rules import ContinuousSplitRule, OneHotSplitRule
from pymc_extras.prior import Prior

from pymc_marketing.bart import Bart
from pymc_marketing.serialization import serialization
from pymc_marketing.terms import (
    Dot,
    Intercept,
    build_param,
    collect_coords,
    collect_terms,
    register_data,
)


@pytest.fixture
def ds() -> xr.Dataset:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=40)
    return xr.Dataset(
        {"X": (("obs", "feature"), X), "y_obs": (("obs",), y)},
        coords={"obs": np.arange(40), "feature": ["a", "b", "c"]},
    )


def make_term(**kwargs: Any) -> Bart:
    defaults: dict[str, Any] = {
        "var_name": "X",
        "y_name": "y_obs",
        "m": 20,
        "split_rules": [ContinuousSplitRule()] * 3,
        "name": "bart",
    }
    defaults.update(kwargs)
    return Bart(**defaults)


def test_invalid_response_raises() -> None:
    with pytest.raises(ValueError, match="response"):
        make_term(response="quadratic")


def test_invalid_m_raises() -> None:
    with pytest.raises(ValueError, match="m must be"):
        make_term(m=0)


def test_get_coords(ds: xr.Dataset) -> None:
    term = make_term()
    coords = term.get_coords(ds)
    assert set(coords) == {"obs", "feature"}
    assert coords["feature"] == ["a", "b", "c"]


def test_register_data_is_guarded(ds: xr.Dataset) -> None:
    term = make_term()
    with __import__("pymc").Model(coords=term.get_coords(ds)) as model:
        term.register_data(ds)
        term.register_data(ds)
        assert "X" in model
        assert "y_obs" in model


def test_set_data_resizes_obs(ds: xr.Dataset) -> None:
    term = make_term()
    import pymc as pm
    import pymc.dims as pmd
    import pymc_bart as pmb
    from pytensor.xtensor.type import as_xtensor

    coords = term.get_coords(ds)
    with pm.Model(coords=coords) as model:
        term.register_data(ds)
        mu_plain = pmb.BART(
            "bart",
            X=model["X"].values,
            Y=model["y_obs"].values,
            m=term.m,
            split_rules=term.split_rules,
            dims=("obs",),
        )
        mu_det = pmd.Deterministic("mu", as_xtensor(mu_plain, dims=("obs",)))
        pmd.Normal("y", mu=mu_det, sigma=1.0, observed=model["y_obs"], dims="obs")
        term.set_data(ds, model=model)

    ds_new = ds.isel(obs=slice(0, 7)).assign_coords(
        obs=np.arange(7), feature=["a", "b", "c"]
    )
    term.set_data(ds_new, model=model)
    assert model["X"].values.eval().shape == (7, 3)


def test_sample_vars() -> None:
    assert make_term().sample_vars == ["bart"]


def test_compose_with_intercept(ds: xr.Dataset) -> None:
    import pymc as pm
    import pymc.dims as pmd

    mu = Intercept(name="intercept") + make_term()
    coords = collect_coords(mu, ds=ds)
    coords["obs"] = ds.coords["obs"].values.tolist()
    with pm.Model(coords=coords) as model:
        register_data(mu, ds=ds)
        pmd.Normal(
            "y",
            mu=build_param(mu),
            sigma=Prior("HalfNormal", sigma=1.0).create_variable("sigma", xdist=True),
            observed=model["y_obs"],
            dims="obs",
        )
        prior = pm.sample_prior_predictive(draws=2)
    assert prior.prior["intercept"].dims == ("chain", "draw")
    assert set(prior.prior["bart"].dims) == {"chain", "draw", "obs"}


def test_collect_terms_finds_bart(ds: xr.Dataset) -> None:
    mu = Intercept(name="intercept") + make_term()
    terms = collect_terms([mu])
    bart_terms = [t for t in terms if isinstance(t, Bart)]
    assert len(bart_terms) == 1


def test_serialization_roundtrip() -> None:
    term = make_term(split_rules=[OneHotSplitRule(), ContinuousSplitRule()])
    data = serialization.serialize(term)
    assert data["__type__"] == "pymc_marketing.bart.Bart"
    rebuilt = serialization.deserialize(data)
    assert isinstance(rebuilt, Bart)
    assert rebuilt.m == term.m
    assert rebuilt.alpha == term.alpha
    assert rebuilt.beta == term.beta
    assert rebuilt.response == term.response
    assert rebuilt.name == term.name
    assert rebuilt.var_name == term.var_name
    assert rebuilt.y_name == term.y_name
    rule_types = [type(rule) for rule in rebuilt.split_rules]
    assert rule_types == [OneHotSplitRule, ContinuousSplitRule]


def test_deserialize_unknown_split_rule_raises() -> None:
    data = make_term().to_dict()
    data["split_rules"] = ["NotASplitRule"]
    with pytest.raises(ValueError, match="NotASplitRule"):
        Bart.from_dict(data)


def test_create_variable_without_pymc_bart_raises(ds: xr.Dataset, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pymc_bart", None)

    import pymc_marketing.bart as bart_module

    monkeypatch.setattr(bart_module, "pmb", None)

    term = make_term()
    import pymc as pm

    with pm.Model(coords=term.get_coords(ds)):
        term.register_data(ds)
        with pytest.raises(ImportError, match="pymc-marketing\\[pie\\]"):
            build_param(term)


def test_bart_dot_swappable(ds: xr.Dataset) -> None:
    """Bart and Dot terms produce identically-dimensioned contributions."""
    import pymc as pm
    import pymc.dims as pmd

    linear_mean = Dot(
        var_name="X", name="mu_coef", prior=Prior("Normal", dims="feature")
    )
    for recipe in (make_term(), linear_mean):
        coords = collect_coords(recipe, ds=ds)
        coords["obs"] = ds.coords["obs"].values.tolist()
        with pm.Model(coords=coords):
            register_data(recipe, ds=ds)
            pmd.Deterministic("mu", build_param(recipe), dims="obs")
            prior = pm.sample_prior_predictive(draws=2)
        assert prior.prior["mu"].dims == ("chain", "draw", "obs")
        assert prior.prior["mu"].shape[-1] == 40
