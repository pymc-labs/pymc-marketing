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
"""Tests for NestedZeroSumPrior and its sigma options IndependentSigmas and VarianceBudget."""

import json
import re
import textwrap

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytensor.tensor as pt
import pytest
import yaml
from pymc_extras.deserialize import deserialize
from pymc_extras.prior import Prior

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml
from pymc_marketing.mmm.mmm import MMM
from pymc_marketing.serialization import serialization
from pymc_marketing.special_priors import (
    IndependentSigmas,
    NestedZeroSumPrior,
    VarianceBudget,
)

G = Prior("Gamma", alpha=2, beta=20)
MU = Prior("Normal", dims="channel")
FAM = {"b0": 0, "b1": 0, "b2": 1, "b3": 1}  # brand -> family
TIER = {f"d{i}": t for i, t in enumerate([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])}  # dma -> tier
COORDS = {"channel": ["c0", "c1", "c2"], "brand": list(FAM), "dma": list(TIER)}
NEST = {"family": {}, "brand": {"within": "family", "map": FAM}}
LEVEL_HN = Prior("HalfNormal", sigma=0.1, dims="beta_sigma_level")
LEVEL_GAMMA = Prior(  # hierarchical: the levels share an unknown mean scale
    "Gamma", mu=Prior("HalfNormal", sigma=0.2), sigma=0.05, dims="beta_sigma_level"
)


class ThreeScales:
    """A factory that declares the level dim but ignores it."""

    dims = ("beta_sigma_level",)

    def create_variable(self, name):
        return pm.HalfNormal(name, shape=3)


class XScales:
    """A factory that returns its scales as an XTensorVariable."""

    dims = ("beta_sigma_level",)

    def create_variable(self, name):
        return pmd.HalfNormal(name, sigma=0.2, dims=self.dims)


def nzs(terms, interactions=None, **kwargs):
    """A NestedZeroSumPrior with the same sigma prior on every level, unless given."""
    levels = [*terms, *(interactions or {})]
    kwargs.setdefault("sigma", IndependentSigmas({lv: G for lv in levels}))
    kwargs.setdefault("mu", MU)
    return NestedZeroSumPrior(terms=terms, interactions=interactions, **kwargs)


def build(prior, coords=COORDS):
    with pm.Model(coords=coords) as model:
        beta = prior.create_variable("beta", xdist=True)
    return model, beta


def beta_at(model, beta, point):
    """Beta as a function of the model's value variables."""
    fn = model.compile_fn(
        model.replace_rvs_by_values([beta.values])[0],
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    return fn(point)


# --- Constructor checks -------------------------------------------------------


@pytest.mark.parametrize(
    ("terms", "interactions", "dims"),
    [
        (NEST, None, ("brand", "channel")),
        (
            {**NEST, "dma": {}},
            {"brand_dma": {"dims": ["brand", "dma"]}},
            ("brand", "dma", "channel"),
        ),
        # nesting without a family term
        (
            {"brand": {"within": "family", "map": {}, "parent_effect": False}},
            None,
            ("brand", "channel"),
        ),
    ],
)
def test_dims_are_derived_from_the_terms(terms, interactions, dims):
    sigma = IndependentSigmas({lv: G for lv in [*terms, *(interactions or {})]})
    prior = NestedZeroSumPrior(
        mu=MU, terms=terms, interactions=interactions, sigma=sigma
    )
    assert prior.dims == dims
    assert prior.parameters["interactions"] == (interactions or {})
    assert prior.parameters["sigma"] is sigma


def test_raw_dicts_are_deserialized():  # YAML resolve() hands nested priors over as plain dicts
    sigma = {
        "__type__": "pymc_marketing.special_priors.IndependentSigmas",
        "priors": {lv: G.to_dict() for lv in NEST},
    }
    prior = NestedZeroSumPrior(
        mu={"distribution": "Normal", "dims": "channel"}, terms=NEST, sigma=sigma
    )
    assert isinstance(prior.parameters["mu"], Prior)
    assert isinstance(prior.parameters["sigma"], IndependentSigmas)
    assert prior.parameters["sigma"].priors["brand"] == G


@pytest.mark.parametrize(
    "mu", [0.5, 1, [0.1, -0.2, 0.3]], ids=["float", "int", "per_channel"]
)
def test_constant_mu_fixes_the_channel_mean_and_round_trips(mu):
    prior = nzs({"brand": {}}, mu=mu)
    model, beta = build(prior)
    assert "beta_mu" not in model.named_vars
    # brand sums to zero, so the brand mean of log beta is mu
    np.testing.assert_allclose(
        np.log(pm.draw(beta.values)).mean(axis=0), np.broadcast_to(mu, 3), atol=1e-6
    )
    data = json.loads(json.dumps(prior.to_dict()))
    assert NestedZeroSumPrior.from_dict(data).to_dict() == prior.to_dict()


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (
            lambda: IndependentSigmas({"brand": 0.5}),
            r"IndependentSigmas priors\['brand'\] must be a Prior",
        ),
        (
            lambda: VarianceBudget(0.5, {"brand": 1.0}),
            "VarianceBudget total must be a Prior",
        ),
        (lambda: nzs({"brand": {}}, sigma=0.5), "`sigma` must be IndependentSigmas"),
        (  # a dimmed scale broadcast onto the wrong axis and broke the zero sum
            lambda: IndependentSigmas({"brand": Prior("HalfNormal", dims="channel")}),
            r"IndependentSigmas priors\['brand'\] must be scalar \(no dims\)",
        ),
        (
            lambda: VarianceBudget(Prior("HalfNormal", dims="channel"), {"brand": 1.0}),
            r"VarianceBudget total must be scalar \(no dims\)",
        ),
        (lambda: VarianceBudget(G, {"brand": 1.0}, -1), "concentration > 0"),
        (lambda: VarianceBudget(G, {}), "non-empty share"),
    ],
    ids=[
        "independent",
        "budget",
        "sigma",
        "independent_dims",
        "budget_total_dims",
        "budget_concentration",
        "budget_empty_share",
    ],
)
def test_invalid_sigma_raises_naming_the_field(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_the_callers_spec_is_not_mutated():
    terms = {"family": {}, "brand": {"within": "family", "map": {101: 0, 102: 1}}}
    nzs(terms)
    assert list(terms["brand"]["map"]) == [101, 102]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"terms": {**NEST, "family": {"sigma": G}}},
            "carry their own sigma",
        ),
        ({"terms": NEST, "sigma": None}, "keyed by exactly"),
        (
            {"terms": NEST, "sigma": IndependentSigmas({"family": G})},
            "keyed by exactly",
        ),
        (
            {
                "terms": NEST,
                "sigma": VarianceBudget(G, {"family": 0.5, "brand": 0.5}),
                "interactions": {"bd": {"dims": ["brand", "dma"]}},
            },
            "keyed by exactly",
        ),
        (
            {
                "terms": {"brand": {"within": "family", "map": {}}},
                "sigma": IndependentSigmas({"brand": G}),
            },
            r"within: family`, which is not a term.*parent_effect: false",
        ),
        (
            {"terms": {"family": {}, "brand": {"within": "family"}}},
            "has no `map`",
        ),
        (
            {"terms": NEST, "interactions": {"bd": {"dims": ["brand", "dma"]}}},
            r"interaction dims \['dma'\] are not terms that are model coords",
        ),
        (
            {"terms": NEST, "interactions": {"bf": {"dims": ["brand", "family"]}}},
            r"interaction dims \['family'\] are not terms that are model coords",
        ),
        ({"terms": NEST, "dims": ("channel", "brand")}, "leave dims unset"),
        ({"terms": NEST, "foo": 1}, "Parameters must be mu, terms, sigma"),
        ({"terms": NEST, "link": "logit"}, "link must be 'log' or 'identity'"),
        (
            {"terms": NEST, "dims": ("brand", "channel", "control")},
            r"more than one base dim \['channel', 'control'\]",
        ),
        ({"terms": NEST, "dims": ("brand",)}, r"mu has dims \('channel',\); it must"),
        (
            {
                "terms": {
                    **NEST,
                    "region": {},
                    "family": {"within": "region", "map": {0: "r0"}},
                }
            },
            "nesting is at most two levels per tree",
        ),
        (  # dims were unset, yet the message said to leave them unset
            {"terms": {"channel": {}, "brand": {"within": "channel", "map": FAM}}},
            "term 'channel' is named like the default base dim",
        ),
        ({"terms": NEST, "mu": "abc"}, "constant mu must be a number"),
        (
            {"terms": NEST, "mu": [0.1, 0.2, 0.3, 0.4], "dims": ("brand",)},
            r"one number per label of the base dims \(\)",
        ),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    terms = kwargs.pop("terms")
    with pytest.raises(ValueError, match=match):
        nzs(terms, **kwargs)


def test_identity_link_is_the_log_of_the_log_link():
    beta_log, beta_id = (
        pm.draw(build(nzs(NEST, link=link))[1].values, 20, random_seed=1)
        for link in ("log", "identity")
    )
    assert (beta_id < 0).any()
    np.testing.assert_allclose(np.exp(beta_id), beta_log, rtol=1e-6)


CROSS = {"brand": {}, "dma": {}}


@pytest.mark.parametrize(
    ("terms", "interactions", "dims", "mu", "base"),
    [
        (NEST, None, ("brand",), Prior("Normal"), ()),
        (NEST, None, ("brand",), 0.5, ()),
        (CROSS, {"brand_dma": {"dims": ["brand", "dma"]}}, ("brand", "dma"), 0.5, ()),
        (
            NEST,
            None,
            ("brand", "control"),
            Prior("Normal", dims="control"),
            ("control",),
        ),
        (NEST, None, ("brand", "control"), [0.1, -0.2], ("control",)),
    ],
    ids=["none", "none_constant", "none_crossed", "control", "control_constant"],
)
def test_base_dim_is_optional_and_need_not_be_channel(
    terms, interactions, dims, mu, base
):
    prior = nzs(terms, interactions, mu=mu, dims=dims, link="identity")
    coords = {"brand": list(FAM), "dma": list(TIER), "control": ["x0", "x1"]}
    model, beta = build(prior, coords)
    assert beta.dims == dims
    n_base = tuple(len(coords[d]) for d in base)
    offsets = [
        v for v in model.free_RVs if "sigma" not in v.name and v.name != "beta_mu"
    ]
    assert {v.type.shape[: len(base)] for v in offsets} == {n_base}
    # every offset is zero-sum over a hierarchy dim, so the hierarchy mean of beta is mu
    mu_var = model["beta_mu"] if "beta_mu" in model.named_vars else pt.as_tensor(mu)
    beta_draw, mu_draw = pm.draw([beta.values, mu_var], random_seed=2)
    hierarchy_axes = tuple(range(len(dims) - len(base)))
    np.testing.assert_allclose(
        beta_draw.mean(axis=hierarchy_axes), np.broadcast_to(mu_draw, n_base), atol=1e-6
    )


# --- Regressions: one test per fixed bug --------------------------------------


def test_parent_effect_false_with_a_parent_term_raises():
    # was silently ignored: beta still depended on beta_family
    terms = {
        "family": {},
        "brand": {"within": "family", "map": FAM, "parent_effect": False},
    }
    with pytest.raises(
        ValueError, match="brand sets `parent_effect: false` but family is a term"
    ):
        nzs(terms)


def test_mixed_str_and_int_parent_labels_raise():  # np.unique merged 0 and "0" into one parent
    brand_map = {"b0": 0, "b1": 0, "b2": "0", "b3": "0"}
    with pytest.raises(ValueError, match="mixes str and non-str family labels"):
        nzs({"family": {}, "brand": {"within": "family", "map": brand_map}})


@pytest.mark.parametrize(
    "share",
    [
        {"family": 0.5, "brand": 0.6},
        {"family": 0.0, "brand": 1.0},
        {"family": -0.1, "brand": 1.1},
    ],
)
def test_variance_budget_shares_off_the_simplex_raise(share):
    # sum != 1 rescaled the concentration; a share <= 0 built an invalid Dirichlet
    with pytest.raises(ValueError, match="must be > 0 for every level and sum to 1"):
        VarianceBudget(G, share)


@pytest.mark.parametrize(
    ("terms", "interactions", "match"),
    [
        (
            # the pre-rename British spelling of the key is now rejected, not ignored
            {"brand": {"centred": True}},
            None,
            r"brand: unknown keys \['centred'\]; "
            r"allowed: \['centered', 'within', 'map', 'parent_effect'\]",
        ),
        (
            {
                "family": {},
                "brand": {"within": "family", "map": FAM, "parent_efect": 0},
            },
            None,
            r"unknown keys \['parent_efect'\]",
        ),
        (
            {"family": {}, "brand": {"whithin": "family", "map": FAM}},
            None,
            r"unknown keys \['whithin'\]",
        ),
        (
            {"brand": {}},
            {"i": {"dims": ["brand"], "centred": True}},
            r"i: unknown keys \['centred'\]; allowed: \['dims', 'centered'\]",
        ),
    ],
)
def test_unknown_spec_keys_raise(terms, interactions, match):
    # a misspelled key was ignored (e.g. the level was built non-centered)
    with pytest.raises(ValueError, match=match):
        nzs(terms, interactions)


@pytest.mark.parametrize(
    ("terms", "dims"),
    [
        ({"brand": {}}, ("brand", "channel")),
        (
            {"brand": {}, "tier": {}, "dma": {"within": "tier", "map": TIER}},
            ("brand", "dma", "channel"),
        ),
    ],
)
def test_crossed_term_without_within_is_in_the_dims_and_builds(terms, dims):
    # was left out of dims, so the transpose crashed
    prior = nzs(terms)
    model, beta = build(prior)
    assert prior.dims == beta.dims == dims
    assert "beta_brand" in model.named_vars


def test_int_child_labels_survive_the_json_round_trip():
    # JSON made the map keys str, so the reloaded prior missed every int label
    brand_map = {101: 0, 102: 0, 103: 1, 104: 1}
    prior = nzs({"family": {}, "brand": {"within": "family", "map": brand_map}})
    loaded = deserialize(json.loads(json.dumps(prior.to_dict())))
    coords = COORDS | {"brand": list(brand_map)}
    (model_p, beta_p), (model_l, beta_l) = build(prior, coords), build(loaded, coords)
    point = {
        k: v + 0.1 * np.arange(v.size).reshape(v.shape)
        for k, v in model_p.initial_point().items()
    }
    assert loaded == prior
    assert model_p.compile_logp()(point) == model_l.compile_logp()(point)
    np.testing.assert_array_equal(
        beta_at(model_p, beta_p, point), beta_at(model_l, beta_l, point)
    )


GROUPED = {  # one parent term, two children within it
    "grp": {},
    "brand": {"within": "grp", "map": {b: f"g{i // 2}" for i, b in enumerate(FAM)}},
    "dma": {"within": "grp", "map": {d: f"g{i // 5}" for i, d in enumerate(TIER)}},
}


@pytest.mark.parametrize(
    ("make", "coords", "match"),
    [
        (
            lambda: nzs(GROUPED),
            COORDS,
            r"terms \['grp'\] have more than one child `within` them",
        ),
        (
            lambda: nzs(
                NEST,
                {"family": {"dims": ["brand"]}},
                sigma=IndependentSigmas({"family": G, "brand": G}),
            ),
            COORDS,
            r"interactions \['family'\] are named like terms",
        ),
        (
            lambda: nzs({"tier": {}, "dma": {"within": "tier", "map": TIER}}),
            COORDS | {"tier": [0, 1, 2]},
            "dma is `within: tier`, which is a model coord",
        ),
        (
            lambda: nzs(NEST, {"brand_0": {"dims": ["brand"]}}),
            COORDS,
            r"RV names \['beta_brand_0'\] are generated twice",
        ),
        (
            lambda: nzs({"sigma_brand": {}, "brand": {}}),
            COORDS | {"sigma_brand": ["x", "y"]},
            r"RV names \['beta_sigma_brand'\] are generated twice",
        ),
        (
            lambda: nzs(NEST, sigma=G),
            COORDS,
            "`sigma` must be IndependentSigmas or VarianceBudget",
        ),
        (
            lambda: nzs(NEST, sigma=Prior("HalfNormal", dims="beta_sigma_levels")),
            COORDS,
            r"or a prior with the single dim `<name>_sigma_level` \(got dims \('beta_sigma_levels',\)\)",
        ),
        (
            lambda: nzs(NEST, sigma=Prior("HalfNormal", dims="gamma_sigma_level")),
            COORDS,
            r"for beta it must have the single dim 'beta_sigma_level'",
        ),
        (
            lambda: nzs(
                NEST, sigma=Prior("HalfNormal", dims=("beta_sigma_level", "channel"))
            ),
            COORDS,
            "or a prior with the single dim `<name>_sigma_level`",
        ),
        (  # a term `sigma` creates beta_sigma, the factory's own variable
            lambda: nzs({"sigma": {}, "brand": {}}, sigma=LEVEL_HN),
            COORDS | {"sigma": ["x", "y"]},
            r"RV names \['beta_sigma'\] are generated twice",
        ),
        (
            lambda: nzs(NEST, sigma=ThreeScales()),
            COORDS,
            r"`sigma` returned shape \(3,\); expected \(2,\)",
        ),
        (
            lambda: nzs(NEST, sigma=LEVEL_HN),
            COORDS | {"beta_sigma_level": ["brand", "family"]},
            r"coord 'beta_sigma_level' has labels \['brand', 'family'\], "
            r"but the levels of beta \(terms, then interactions\) are \['family', 'brand'\]",
        ),
        (
            lambda: nzs(NEST),
            {k: v for k, v in COORDS.items() if k != "channel"},
            "the model has no `channel` coord",
        ),
        (
            lambda: nzs({"brand": {}}, {"i": {}}),
            COORDS,
            r"interactions \['i'\] have no `dims`",
        ),
        (
            lambda: nzs(
                {"family": {}, "brand": {"within": "family", "map": FAM | {"b3": None}}}
            ),
            COORDS,
            "`map` has a None family label",
        ),
        (
            lambda: nzs({"region": {}, "brand": {}}),
            COORDS,
            r"term region: not a model coord",
        ),
        (
            lambda: nzs(
                {
                    "family": {},
                    "brand": {"within": "family", "map": FAM},
                    "tier": {},
                    "dma": {"within": "tier", "map": {"d0": 0}},
                }
            ),
            COORDS,
            r"term dma: `map` has no tier for data labels \['d1'",
        ),
        (
            lambda: nzs(
                {"family": {}, "brand": {"within": "family", "map": FAM | {"b9": 1}}}
            ),
            COORDS,
            r"term brand: `map` keys \['b9'\] are not brand labels in the data",
        ),
        (  # was pytensor's Alloc error
            lambda: nzs(NEST, mu=[0.1, 0.2]),
            COORDS,
            r"constant mu has shape \(2,\); expected \(\) or \(3,\)",
        ),
        (  # hyperprior names are not checked: pymc's own error
            lambda: nzs(
                {"sigma_sigma": {}},
                sigma=Prior("HalfNormal", sigma=G, dims="beta_sigma_level"),
            ),
            COORDS | {"sigma_sigma": ["x", "y"]},
            "Variable name beta_sigma_sigma already exists",
        ),
        (
            lambda: nzs({"mu_sigma": {}}, mu=Prior("Normal", sigma=G, dims="channel")),
            COORDS | {"mu_sigma": ["x", "y"]},
            "Variable name beta_mu_sigma already exists",
        ),
    ],
    ids=[
        "shared_parent",
        "interaction_named_like_term",
        "parent_is_coord",
        "rv_name_collision",
        "sigma_name_collision",
        "sigma_is_prior",
        "sigma_dim_not_a_level_dim",
        "sigma_dim_of_another_prior",
        "sigma_with_two_dims",
        "factory_sigma_name_collision",
        "sigma_of_the_wrong_length",
        "level_coord_with_other_labels",
        "no_channel",
        "interaction_without_dims",
        "none_parent",
        "orphan_term",
        "map_misses_a_label",
        "map_has_an_extra_label",
        "constant_mu_of_the_wrong_length",
        "sigma_hyperprior_name_collision",
        "mu_hyperprior_name_collision",
    ],
)
def test_spec_problems_raise_a_valueerror_naming_them(make, coords, match):
    # were: duplicate RV, AttributeError, KeyError, TypeError
    with pytest.raises(ValueError, match=match):
        build(make(), coords)


def test_variance_budget_level_coord_is_named_per_prior():
    # a plain `level` clashed with a user `level` coord (or a second budget)
    prior = nzs(NEST, sigma=VarianceBudget(G, {"family": 0.5, "brand": 0.5}))
    model, _ = build(prior, COORDS | {"level": ["a", "b"]})
    assert list(model.coords["beta_sigma_level"]) == ["family", "brand"]
    assert model.coords["level"] == ("a", "b")


@pytest.mark.parametrize("centered", [True, False])
def test_from_dict_defaults_centered_to_false_like_init(centered):
    # and centered=True still survives the JSON round trip
    prior = nzs(NEST, centered=centered)
    data = {"kwargs": {"mu": MU, "terms": NEST, "sigma": prior.parameters["sigma"]}}
    assert NestedZeroSumPrior.from_dict(data).centered is False
    assert deserialize(json.loads(json.dumps(prior.to_dict()))).centered is centered


@pytest.mark.parametrize(
    ("terms", "interactions", "coords", "match"),
    [
        (
            {"family": {}, "brand": {"within": "family", "map": {b: b for b in FAM}}},
            None,
            COORDS,
            "every family has one brand",
        ),
        (
            {"family": {}, "brand": {"within": "family", "map": {b: 0 for b in FAM}}},
            None,
            COORDS,
            "term family has one label",
        ),
        ({"brand": {}}, None, COORDS | {"brand": ["b0"]}, "term brand has one label"),
        (
            {"brand": {}, "dma": {}},
            {"bd": {"dims": ["brand", "dma"]}},
            COORDS | {"dma": ["d0"]},
            r"interaction bd: dims \['dma'\] have one label",
        ),
    ],
    ids=[
        "singleton_children",
        "one_parent",
        "one_crossed_label",
        "one_interaction_label",
    ],
)
def test_unused_sigma_warns(terms, interactions, coords, match):
    # that level's offsets are identically 0
    with pytest.warns(UserWarning, match=match):
        build(nzs(terms, interactions), coords)


def test_a_term_named_mu_builds_with_a_constant_mu():
    # no beta_mu is created, so there is no collision (was a false positive)
    model, _ = build(nzs({"mu": {}}, mu=0.0), COORDS | {"mu": ["x", "y"]})
    assert "beta_mu" in model.named_vars and "beta_sigma_mu" in model.named_vars


def test_a_str_dims_is_one_dim():  # was split into characters, also by from_dict
    prior = nzs({"brand": {}}, mu=Prior("Normal"), dims="brand")
    assert prior.dims == ("brand",)
    assert NestedZeroSumPrior.from_dict(prior.to_dict() | {"dims": "brand"}) == prior


def json_round_trip(prior):
    return NestedZeroSumPrior.from_dict(json.loads(json.dumps(prior.to_dict())))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            nzs(NEST),
            nzs(
                {
                    "family": {},
                    "brand": {"within": "family", "map": dict(reversed(FAM.items()))},
                }
            ),
        ),
        (
            nzs(
                NEST,
                sigma=IndependentSigmas(
                    {"family": G, "brand": Prior("HalfNormal", sigma=0.3)}
                ),
            ),
            nzs(
                NEST,
                sigma=IndependentSigmas(
                    {"brand": Prior("HalfNormal", sigma=0.3), "family": G}
                ),
            ),
        ),
        (nzs(NEST, link="log"), nzs(NEST)),
        (json_round_trip(p := nzs(NEST, mu=np.array([0.1, -0.2, 0.3]))), p),
        (json_round_trip(p := nzs({}, mu=Prior("Normal"), dims=())), p),
    ],
    ids=[
        "map_order",
        "sigma_order",
        "explicit_log_link",
        "ndarray_mu_round_trip",
        "no_dims_round_trip",
    ],
)
def test_equivalent_priors_are_equal_and_hash_equal(a, b):
    assert a == b and hash(a) == hash(b) and a.to_dict() == b.to_dict()


def test_parent_effect_false_builds_nesting_without_the_parent_term():
    terms = {"dma": {"within": "tier", "map": TIER, "parent_effect": False}}
    model, _ = build(nzs(terms))
    assert "beta_tier" not in model.named_vars
    assert {f"beta_dma_{p}" for p in range(3)} <= set(model.named_vars)


# --- Properties of the built prior --------------------------------------------

HN = Prior("HalfNormal", sigma=0.5)
F3 = {
    "b0": "z",
    "b1": "a",
    "b2": "a",
    "b3": "m",
}  # str parents, unsorted, two singletons
CROSSED = {"brand": {}, "tier": {}, "dma": {"within": "tier", "map": TIER}}


def nested(fam=FAM):
    return {
        "family": {},
        "brand": {"within": "family", "map": fam},
        "tier": {},
        "dma": {"within": "tier", "map": TIER},
    }


def build_full(
    terms,
    centered=True,
    sigma=None,
    interaction_dims=("brand", "dma"),
    brands=tuple(FAM),
    dmas=tuple(TIER),
    tier=TIER,
):
    """Beta over (brand, dma, channel) with the given terms plus a brand x dma interaction."""
    terms = {
        k: {**t, "centered": centered} | ({"map": tier} if k == "dma" else {})
        for k, t in terms.items()
    }
    interactions = {"brand_dma": {"dims": list(interaction_dims), "centered": centered}}
    prior = NestedZeroSumPrior(
        mu=MU,
        terms=terms,
        interactions=interactions,
        sigma=sigma or IndependentSigmas({lv: HN for lv in [*terms, "brand_dma"]}),
    )
    coords = {"channel": ["c0", "c1", "c2"], "brand": list(brands), "dma": list(dmas)}
    return build(prior, coords)


def log_beta_fn(model, beta):
    """Log beta as a function of {free RV name: value}."""
    fn = pytensor.function(
        model.free_RVs, pt.log(beta.values), on_unused_input="ignore"
    )
    return lambda values: fn(*(values[rv.name] for rv in model.free_RVs))


def term_only(model, beta, level, n=20):
    """Level `level`'s contribution to log beta alone, on n prior draws: (draw, brand, dma, channel).

    Mu and every zero-sum variable but the level's are set to 0.
    """
    fn = log_beta_fn(model, beta)
    names = [rv.name for rv in model.free_RVs]
    draws = dict(zip(names, pm.draw(model.free_RVs, n, random_seed=1), strict=True))

    def keep(rv_name):  # beta_brand, beta_brand_0, beta_brand_a; not beta_brand_dma
        return "sigma" in rv_name or re.fullmatch(rf"beta_{level}(_\w)?", rv_name)

    return np.stack(
        [
            fn({k: v[i] if keep(k) else 0 * v[i] for k, v in draws.items()})
            for i in range(n)
        ]
    )


@pytest.mark.parametrize("centered", [True, False])
@pytest.mark.parametrize(
    ("terms", "level", "axis", "groups"),
    [
        (
            nested(),
            "family",
            1,
            [[0, 2]],
        ),  # one brand per family: the families sum to 0
        (nested(), "brand", 1, [[0, 1], [2, 3]]),  # within each family
        (nested(), "tier", 2, [[0, 4, 7]]),
        (nested(), "dma", 2, [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]),  # within each tier
        (nested(), "brand_dma", 1, [range(4)]),  # interaction: both axes
        (nested(), "brand_dma", 2, [range(10)]),
        (CROSSED, "brand", 1, [range(4)]),  # crossed term: over the whole coord
    ],
    ids=[
        "family",
        "brand_in_family",
        "tier",
        "dma_in_tier",
        "interaction_brand_axis",
        "interaction_dma_axis",
        "crossed_brand",
    ],
)
@pytest.mark.parametrize("sigma", [None, LEVEL_HN], ids=["independent", "prior"])
def test_zero_sum_on_every_intended_axis(terms, level, axis, groups, centered, sigma):
    x = term_only(*build_full(terms, centered, sigma), level)
    assert np.abs(x).max() > 1e-3
    for g in groups:
        assert np.abs(x.take(list(g), axis).sum(axis)).max() < 1e-12


@pytest.mark.parametrize("centered", [True, False])
@pytest.mark.parametrize(
    ("brands", "dmas", "fam", "tier"),
    [
        (list(FAM), list(TIER), FAM, TIER),
        (  # permuted coords and map key order
            ["b3", "b0", "b2", "b1"],
            ["d9", "d0", "d5", "d1", "d7", "d2", "d3", "d4", "d6", "d8"],
            dict(reversed(FAM.items())),
            dict(sorted(TIER.items(), key=lambda kv: -int(kv[0][1:]))),
        ),
        (list(FAM), list(TIER), F3, TIER),
    ],
    ids=["base", "permuted", "str_parents_singletons"],
)
def test_each_cell_gets_its_own_parent_and_child_offsets(
    brands, dmas, fam, tier, centered
):
    model, beta = build_full(nested(fam), centered, brands=brands, dmas=dmas, tier=tier)
    rng = np.random.default_rng(0)
    # any values, not only zero-sum ones
    v = {rv.name: rng.normal(size=tuple(rv.shape.eval())) for rv in model.free_RVs}

    def scale(level):
        return 1.0 if centered else v[f"beta_sigma_{level}"]

    families = list(np.unique(list(fam.values())))
    tiers = list(np.unique(list(tier.values())))
    expected = np.zeros((len(brands), len(dmas), 3))
    for i, b in enumerate(brands):
        for j, d in enumerate(dmas):
            # siblings, in coord order
            sib_b = [x for x in brands if fam[x] == fam[b]]
            sib_d = [x for x in dmas if tier[x] == tier[d]]
            expected[i, j] = (
                v["beta_mu"]
                + scale("family") * v["beta_family"][:, families.index(fam[b])]
                + scale("tier") * v["beta_tier"][:, tiers.index(tier[d])]
                + scale("brand_dma") * v["beta_brand_dma"][:, i, j]
            )
            if len(sib_b) > 1:
                expected[i, j] += (
                    scale("brand") * v[f"beta_brand_{fam[b]}"][:, sib_b.index(b)]
                )
            if len(sib_d) > 1:
                expected[i, j] += (
                    scale("dma") * v[f"beta_dma_{tier[d]}"][:, sib_d.index(d)]
                )
    assert np.abs(log_beta_fn(model, beta)(v) - expected).max() < 1e-12


@pytest.mark.parametrize("centered", [True, False])
def test_singleton_children_get_no_within_offset_but_the_parent_effect(centered):
    model, beta = build_full(nested(F3), centered)
    brand, family = term_only(model, beta, "brand"), term_only(model, beta, "family")
    assert np.all(brand[:, [0, 3]] == 0)  # b0 and b3 are alone in their family
    assert np.abs(brand[:, [1, 2]]).min() > 0
    assert np.abs(family[:, [0, 3]]).min() > 0


def test_variance_budget_sigma_is_total_times_sqrt_share():
    share = {"family": 0.3, "brand": 0.3, "tier": 0.2, "dma": 0.1, "brand_dma": 0.1}
    model, _ = build_full(nested(), sigma=VarianceBudget(HN, share))
    sigma, total, shares = pm.draw(
        [model["beta_sigma"], model["beta_sigma_total"], model["beta_sigma_share"]],
        200,
        random_seed=0,
    )
    np.testing.assert_allclose(sigma, total[:, None] * np.sqrt(shares), rtol=1e-12)


@pytest.mark.parametrize("centered", [True, False])
def test_per_cell_variance_around_mu_matches_the_docstring(centered):
    # Var(log beta - mu) per cell = E[total^2] * sum_l c_l share_l, c_l = 1 - 1/J per zero-sum axis
    share = {
        "family": 0.49261084,
        "brand": 0.49261084,
        "tier": 0.0049261084,
        "dma": 0.0049261084,
        "brand_dma": 0.0049261084,
    }
    total = Prior("Gamma", mu=0.43, sigma=0.11)
    model, beta = build_full(nested(), centered, VarianceBudget(total, share))
    log_beta, mu, t = pm.draw(
        [pt.log(beta.values), model["beta_mu"], model["beta_sigma_total"]],
        4000,
        random_seed=2,
    )
    c_dma = np.array([3 / 4] * 4 + [2 / 3] * 6)  # tiers of 4, 3 and 3 DMAs
    predicted = (t**2).mean() * (
        share["family"] / 2
        + share["brand"] / 2
        + share["tier"] * 2 / 3
        + share["dma"] * c_dma
        + share["brand_dma"] * 3 / 4 * 9 / 10
    )
    ratio = (log_beta - mu[:, None, None, :]).var(0) / predicted[None, :, None]
    assert abs(ratio.mean() - 1) < 0.05
    assert 0.85 < ratio.min() and ratio.max() < 1.15


@pytest.mark.parametrize("centered", [True, False])
def test_reordered_spec_gives_identical_beta_and_logp(centered):
    reordered = {k: nested()[k] for k in ("tier", "dma", "family", "brand")}
    model_a, beta_a = build_full(nested(), centered)
    model_b, beta_b = build_full(reordered, centered, interaction_dims=("dma", "brand"))
    assert beta_a.dims == ("brand", "dma", "channel")
    assert beta_b.dims == ("dma", "brand", "channel")
    rng = np.random.default_rng(1)
    point_a = {
        k: x + 0.5 * rng.normal(size=np.shape(x))
        for k, x in sorted(model_a.initial_point().items())
    }
    # the interaction's axes are swapped
    point_b = point_a | {
        k: np.swapaxes(point_a[k], 1, 2)
        for k in point_a
        if k.startswith("beta_brand_dma")
    }
    np.testing.assert_allclose(
        beta_at(model_a, beta_a, point_a),
        np.swapaxes(beta_at(model_b, beta_b, point_b), 0, 1),
        rtol=1e-13,
        atol=0,
    )
    np.testing.assert_allclose(
        model_a.compile_logp()(point_a),
        model_b.compile_logp()(point_b),
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize("prior", [LEVEL_HN, LEVEL_GAMMA], ids=["flat", "hierarchical"])
def test_single_variable_sigma_entry_i_scales_level_i(prior):
    # Non-centered, sigma = e_i leaves only level i in log beta, and zeroing level i's offsets removes it
    model, beta = build_full(nested(), centered=False, sigma=prior)
    levels = ["family", "brand", "tier", "dma", "brand_dma"]
    assert model.named_vars_to_dims["beta_sigma"] == ("beta_sigma_level",)
    assert list(model.coords["beta_sigma_level"]) == levels
    fn = log_beta_fn(model, beta)
    names = [rv.name for rv in model.free_RVs]
    draws = dict(zip(names, pm.draw(model.free_RVs, random_seed=1), strict=True))
    draws["beta_mu"] = 0 * draws["beta_mu"]
    for i, lv in enumerate(levels):
        draws["beta_sigma"] = np.eye(len(levels))[i]
        assert np.abs(fn(draws)).max() > 1e-3
        own_zeroed = {
            k: 0 * v if re.fullmatch(rf"beta_{lv}(_\w)?", k) else v
            for k, v in draws.items()
        }
        assert np.abs(fn(own_zeroed)).max() == 0


@pytest.mark.parametrize(
    "coords",
    [COORDS, COORDS | {"beta_sigma_level": ["family", "brand"]}],
    ids=["coord_added_here", "coord_already_there"],
)
@pytest.mark.parametrize("centered", [True, False])
def test_xtensor_sigma_is_unwrapped(coords, centered):
    model, beta = build(nzs(NEST, centered=centered, sigma=XScales()), coords)
    assert list(model.coords["beta_sigma_level"]) == ["family", "brand"]
    assert model.named_vars_to_dims["beta_sigma"] == ("beta_sigma_level",)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert np.all(pm.draw(beta, 10, random_seed=1) > 0)


# --- Serialization ------------------------------------------------------------

IND = IndependentSigmas({"family": G, "brand": Prior("HalfNormal", sigma=0.3)})
BUD = VarianceBudget(
    total=Prior("TruncatedNormal", mu=0.4, sigma=0.1, lower=0.0),
    share={"family": 0.3, "brand": 0.7},
    concentration=100,
)
SP = NestedZeroSumPrior(
    mu=Prior("Normal", mu=[0.0, 1.0, 2.0], sigma=0.3, dims="channel"),
    sigma=BUD,
    terms={
        "family": {"centered": True},
        "brand": {"within": "family", "map": {"b0": 0, "b1": 1}},
    },
)
SP_PRIOR = NestedZeroSumPrior(
    mu=MU, terms=NEST, sigma=LEVEL_GAMMA
)  # sigma goes through Prior.to_dict and deserialize
OBJS = [IND, BUD, SP_PRIOR, SP]


@pytest.mark.parametrize("obj", OBJS, ids=lambda o: type(o).__name__)
class TestRoundTrip:
    def test_to_dict_from_dict(self, obj):
        data = obj.to_dict()
        assert type(obj).from_dict(data).to_dict() == data

    def test_pymc_marketing_registry(self, obj):  # what fit() and save() use
        data = serialization.serialize(obj)
        assert data["__type__"] == f"pymc_marketing.special_priors.{type(obj).__name__}"
        loaded = serialization.deserialize(data)
        assert type(loaded) is type(obj)
        assert loaded.to_dict() == obj.to_dict()

    def test_pymc_extras_deserialize(
        self, obj
    ):  # what MMM.load uses for the saturation
        loaded = deserialize(serialization.serialize(obj))
        assert type(loaded) is type(obj)
        assert loaded.to_dict() == obj.to_dict()


def test_the_deserializer_can_distinguish_between_the_classes():
    loaded = [deserialize(serialization.serialize(o)) for o in OBJS]
    assert [type(o) for o in loaded] == [type(o) for o in OBJS]
    assert loaded[-1] == SP
    assert loaded[2] == SP_PRIOR
    assert SP.to_dict()["special_prior"] == "NestedZeroSumPrior"
    assert isinstance(deserialize(G.to_dict()), Prior)  # plain Priors still go to Prior


@pytest.mark.parametrize("dumps", [json.dumps, yaml.safe_dump])
def test_link_and_base_dims_round_trip(dumps):
    prior = nzs(
        NEST,
        mu=Prior("Normal", dims="control"),
        dims=("brand", "control"),
        link="identity",
    )
    loaded = NestedZeroSumPrior.from_dict(yaml.safe_load(dumps(prior.to_dict())))
    assert loaded == prior and loaded.parameters["link"] == "identity"
    assert deserialize(serialization.serialize(prior)) == prior


def test_nested_sigma_survives_round_trip():
    loaded = deserialize(serialization.serialize(SP))
    assert isinstance(loaded.parameters["sigma"], VarianceBudget)
    assert loaded.parameters["sigma"].total == BUD.total
    assert loaded.parameters["mu"] == SP.parameters["mu"]


# --- MMM ----------------------------------------------------------------------

MMM_FAM = {"b0": "f0", "b1": "f0", "b2": "f0", "b3": "f1"}


@pytest.fixture(scope="module")
def panel():
    """A small (date, brand, dma) panel with two channels and one control."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=12, freq="W-MON")
    index = pd.MultiIndex.from_product(
        [dates, list(MMM_FAM), ["d0", "d1"]], names=["date", "brand", "dma"]
    )
    X = index.to_frame(index=False)
    X["c0"] = rng.uniform(0, 10, len(X))
    X["c1"] = rng.uniform(0, 5, len(X))
    X["x0"] = rng.normal(size=len(X))
    y = pd.Series(
        1 + 0.1 * X["c0"] + 0.2 * X["c1"] + rng.normal(0, 0.1, len(X)), name="y"
    )
    return X, y


def make_mmm(sigma):
    beta = NestedZeroSumPrior(
        mu=Prior("Normal", mu=0, sigma=0.5, dims="channel"),
        terms={"family": {}, "brand": {"within": "family", "map": MMM_FAM}, "dma": {}},
        interactions={"brand_dma": {"dims": ["brand", "dma"]}},
        sigma=sigma,
    )
    return MMM(
        date_column="date",
        channel_columns=["c0", "c1"],
        control_columns=["x0"],
        target_column="y",
        dims=("brand", "dma"),
        adstock=GeometricAdstock(
            l_max=2, priors={"alpha": Prior("Beta", alpha=2, beta=3, dims="channel")}
        ),
        saturation=LogisticSaturation(
            priors={
                "beta": beta,
                "lam": Prior("Gamma", alpha=3, beta=2, dims="channel"),
            }
        ),
    )


LEVELS = ["family", "brand", "dma", "brand_dma"]
SIGMAS = {
    "independent": IndependentSigmas(
        {lv: Prior("HalfNormal", sigma=0.2) for lv in LEVELS}
    ),
    "budget": VarianceBudget(
        Prior("HalfNormal", sigma=0.4), dict.fromkeys(LEVELS, 0.25)
    ),
    "prior": Prior(
        "HalfNormal",
        sigma=Prior("HalfNormal", sigma=0.2),
        dims="saturation_beta_sigma_level",
    ),
}


def test_mmm_builds_with_the_prior_on_saturation_beta(panel):
    X, y = panel
    mmm = make_mmm(SIGMAS["independent"])
    mmm.build_model(X, y)
    beta = mmm.model["saturation_beta"]
    assert mmm.model.named_vars_to_dims["saturation_beta"] == (
        "brand",
        "dma",
        "channel",
    )
    assert {"saturation_beta_family", "saturation_beta_brand_f0"} <= set(
        mmm.model.named_vars
    )
    # f1 has one brand, so there is no zero-sum vector for it
    assert "saturation_beta_brand_f1" not in mmm.model.named_vars
    assert np.all(pm.draw(beta, 10, random_seed=1) > 0)


@pytest.mark.parametrize("sigma", ["independent", "budget", "prior"])
def test_mmm_save_load_round_trip(sigma, panel, tmp_path, mock_pymc_sample):
    X, y = panel
    mmm = make_mmm(SIGMAS[sigma])
    mmm.fit(X, y, chains=1, draws=10, tune=10, random_seed=42)
    assert "saturation_beta" in mmm.idata.posterior
    mmm.save(str(tmp_path / "mmm.nc"))
    loaded = MMM.load(str(tmp_path / "mmm.nc"), check=True)
    original_beta = mmm.saturation.function_priors["beta"]
    loaded_beta = loaded.saturation.function_priors["beta"]
    assert loaded.id == mmm.id
    assert type(loaded_beta) is NestedZeroSumPrior
    assert loaded_beta == original_beta
    assert type(loaded_beta.parameters["sigma"]) is type(SIGMAS[sigma])
    point = mmm.model.initial_point()
    assert loaded.model.compile_logp()(point) == mmm.model.compile_logp()(point)


def docstring_blocks(cls, language):
    """The ``.. code-block:: <language>`` blocks of a class docstring, dedented."""
    pattern = rf"^( *)\.\. code-block:: {language}\n\n((?:\s*\n|\1 +.*\n)+)"
    return [textwrap.dedent(b) for _, b in re.findall(pattern, cls.__doc__, re.M)]


@pytest.mark.parametrize("cls", [NestedZeroSumPrior, IndependentSigmas, VarianceBudget])
def test_docstring_examples_build(cls, panel, tmp_path):
    namespace = {}
    for python_example in docstring_blocks(
        cls, "python"
    ):  # later blocks reuse the first's imports
        exec(python_example, namespace)  # noqa: S102
    assert isinstance(namespace["mmm"], MMM)

    (yaml_example,) = docstring_blocks(cls, "yaml")
    config = {
        "model": {
            "class": "pymc_marketing.mmm.mmm.MMM",
            "kwargs": {
                "date_column": "date",
                "channel_columns": ["c0", "c1"],
                "target_column": "y",
                "dims": ["brand", "dma"],
                "adstock": {
                    "class": "pymc_marketing.mmm.GeometricAdstock",
                    "kwargs": {"l_max": 2},
                },
                **yaml.safe_load(yaml_example),
            },
        }
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    X, y = panel
    mmm = build_mmm_from_yaml(path, X=X, y=y)
    beta = mmm.saturation.function_priors["beta"]
    assert isinstance(beta, NestedZeroSumPrior)
    assert isinstance(beta.parameters["sigma"], IndependentSigmas | VarianceBudget)
    assert "saturation_beta" in mmm.model.named_vars
