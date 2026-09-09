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

"""Tests for pymc_marketing.terms."""

import json
from dataclasses import dataclass

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import pytest
import xarray as xr
from pymc_extras.prior import CUSTOM_TRANSFORMS, Prior
from pytensor.graph.basic import Variable as PTVariable

from pymc_marketing.model_builder import ModelBuilder
from pymc_marketing.r2d2 import R2D2
from pymc_marketing.serialization import (
    DeferredFactory,
    SerializationError,
    serialization,
)
from pymc_marketing.terms import (
    Dot,
    Intercept,
    ModelTerm,
    Named,
    Parameter,
    Product,
    Ref,
    Sum,
    Transform,
    _deserialize_child,
    _serialize_child,
    build_param,
    collect_coords,
    collect_terms,
    get_coords,
    register_data,
    set_data,
)


@pytest.fixture
def simple_ds():
    """Dataset with a 2D feature variable and a target."""
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "x": (("obs", "feature"), rng.normal(size=(50, 3))),
            "y": ("obs", rng.normal(size=50)),
        },
        coords={"obs": range(50), "feature": list("ABC")},
    )


def test_modelterm_add():
    result = Intercept(name="a") + Intercept(name="b")
    assert isinstance(result, Sum)
    assert len(result.terms) == 2


def test_modelterm_add_int():
    result = Intercept(name="a") + 5
    assert isinstance(result, Sum)
    assert result.terms[0] == Intercept(name="a")
    assert result.terms[1] == 5


def test_modelterm_radd_int():
    result = 3 + Intercept(name="a")
    assert isinstance(result, Sum)
    assert result.terms[0] == 3
    assert result.terms[1] == Intercept(name="a")


def test_modelterm_radd_zero():
    result = 0 + Intercept(name="a")
    assert result == Intercept(name="a")


def test_modelterm_mul():
    result = Intercept(name="a") * Intercept(name="b")
    assert isinstance(result, Product)
    assert result.left == Intercept(name="a")
    assert result.right == Intercept(name="b")


def test_modelterm_sub():
    result = Intercept(name="a") - Intercept(name="b")
    assert isinstance(result, Sum)
    assert len(result.terms) == 2
    assert result.terms[0] == Intercept(name="a")
    assert isinstance(result.terms[1], Product)
    assert result.terms[1].left == -1
    assert result.terms[1].right == Intercept(name="b")


def test_modelterm_rsub():
    result = 5 - Intercept(name="a")
    assert isinstance(result, Sum)
    assert result.terms[0] == 5
    assert isinstance(result.terms[1], Product)


def test_modelterm_neg():
    result = -Intercept(name="a")
    assert isinstance(result, Product)
    assert result.left == -1
    assert result.right == Intercept(name="a")


def test_sum_add_sum():
    t1 = Sum([Intercept(name="a")])
    t2 = Sum([Intercept(name="b")])
    result = t1 + t2
    assert len(result.terms) == 2


def test_sum_add_term():
    result = Sum([Intercept(name="a")]) + Dot(
        var_name="x", prior=Prior("Normal", dims="feature")
    )
    assert len(result.terms) == 2


def test_sum_add_int():
    result = Sum([Intercept(name="a")]) + 3
    assert result.terms[-1] == 3


def test_sum_radd_zero():
    result = 0 + Sum([Intercept(name="a")])
    assert isinstance(result, Sum)
    assert len(result.terms) == 1


def test_sum_mul():
    result = Sum([Intercept(name="a")]) * Intercept(name="b")
    assert isinstance(result, Product)


def test_sum_get_coords(simple_ds):
    tl = Sum([Dot(var_name="x", prior=Prior("Normal", dims="feature"))])
    coords = tl.get_coords(simple_ds)
    assert "feature" in coords


def test_intercept_create_variable():
    with pm.Model():
        effect = Intercept(name="intercept").create_variable()
        assert isinstance(effect, PTVariable)


def test_dot_register_data(simple_ds):
    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    coords = dot.get_coords(simple_ds)
    with pm.Model(coords=coords) as model:
        dot.register_data(simple_ds)
        data = pm.modelcontext(None)["x"]
        assert data is not None
        assert "feature" in model.coords


def test_dot_get_coords(simple_ds):
    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    coords = dot.get_coords(simple_ds)
    assert "feature" in coords
    assert coords["feature"] == list("ABC")


def test_dot_create_variable(simple_ds):
    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    coords = dot.get_coords(simple_ds)
    with pm.Model(coords=coords):
        dot.register_data(simple_ds)
        effect = dot.create_variable()
        assert isinstance(effect, PTVariable)


def test_dot_set_data(simple_ds):
    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    coords = dot.get_coords(simple_ds)
    with pm.Model(coords=coords) as model:
        dot.register_data(simple_ds)
        ds2 = simple_ds.copy()
        ds2["x"] = xr.DataArray(
            np.roll(simple_ds["x"].values, 1, axis=0), dims=("obs", "feature")
        )
        dot.set_data(ds2)
        assert np.allclose(model["x"].get_value(), ds2["x"].values)


def test_transform_create_variable():
    with pm.Model():
        inner = Intercept(name="sigma")
        transformed = Transform(inner, func=ptx.math.exp)
        result = transformed.create_variable()
        assert isinstance(result, PTVariable)


def test_transform_with_dot(simple_ds):
    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    transformed = Transform(dot, func=ptx.math.exp)
    coords = transformed.get_coords(simple_ds)
    with pm.Model(coords=coords):
        transformed.register_data(simple_ds)
        result = transformed.create_variable()
        assert isinstance(result, PTVariable)


def test_transform_set_data(simple_ds):
    inner = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    transformed = Transform(inner, func=ptx.math.exp)
    coords = transformed.get_coords(simple_ds)
    with pm.Model(coords=coords) as model:
        transformed.register_data(simple_ds)
        ds2 = simple_ds.copy()
        ds2["x"] = xr.DataArray(
            np.roll(simple_ds["x"].values, 1, axis=0), dims=("obs", "feature")
        )
        transformed.set_data(ds=ds2, model=model)
        assert np.allclose(model["x"].get_value(), ds2["x"].values)


def test_build_param_int():
    assert build_param(5) == 5


def test_build_param_float():
    assert build_param(3.14) == pytest.approx(3.14)


def test_build_param_intercept():
    with pm.Model():
        result = build_param(Intercept(name="intercept"))
        assert isinstance(result, PTVariable)


def test_build_param_dot(simple_ds):
    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    coords = dot.get_coords(simple_ds)
    with pm.Model(coords=coords):
        dot.register_data(simple_ds)
        result = build_param(dot)
        assert isinstance(result, PTVariable)


def test_build_param_sum(simple_ds):
    terms = Intercept(name="intercept") + Dot(
        var_name="x", prior=Prior("Normal", dims="feature")
    )
    coords = get_coords(terms, simple_ds)
    with pm.Model(coords=coords):
        register_data(terms, ds=simple_ds)
        result = build_param(terms)
        assert isinstance(result, PTVariable)


def test_build_param_multiplicative():
    with pm.Model():
        result = build_param(Product(Intercept(name="a"), 2.0))
        assert isinstance(result, PTVariable)


def test_build_param_prior():
    with pm.Model():
        result = build_param(Prior("Normal", mu=0, sigma=1), name="test")
        assert isinstance(result, PTVariable)


def test_build_param_transform():
    with pm.Model():
        result = build_param(Transform(Intercept(name="sigma"), func=ptx.math.exp))
        assert isinstance(result, PTVariable)


def test_build_param_variable_factory():
    """Custom VariableFactory is accepted by build_param."""

    class _CustomFactory:
        dims = None

        def create_variable(self, name, xdist=False):
            return pt.as_tensor_variable(42.0)

    with pm.Model():
        result = build_param(_CustomFactory(), name="custom")
        pt_value = result.eval()
        assert pt_value == 42.0


def test_build_param_dataarray():
    da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="obs")
    with pm.Model():
        result = build_param(da)
        assert isinstance(result, PTVariable)


def test_build_param_unknown_raises():
    with pm.Model():
        with pytest.raises(TypeError, match="Cannot build param"):
            build_param("not_a_term")


def test_collect_terms_flat():
    terms = [Intercept(name="a"), Intercept(name="b")]
    result = collect_terms(terms)
    assert len(result) == 2


def test_collect_terms_nested():
    terms = [
        Intercept(name="a") + Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    ]
    result = collect_terms(terms)
    assert len(result) == 2


def test_collect_terms_multiplicative():
    terms = [Intercept(name="a") * Intercept(name="b")]
    result = collect_terms(terms)
    assert len(result) == 2


def test_collect_terms_skips_constants():
    terms = [Intercept(name="a"), 5, 3.0]
    result = collect_terms(terms)
    assert len(result) == 1


def test_collect_terms_includes_transform():
    terms = [Transform(Intercept(name="a"), func=ptx.math.exp)]
    result = collect_terms(terms)
    assert len(result) == 1


def test_collect_coords(simple_ds):
    """collect_coords merges coordinates from multiple term trees."""
    mu = Intercept(name="mu") + Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    sigma = Transform(Intercept(name="sigma"), func=ptx.math.exp)
    coords = collect_coords(mu, sigma, ds=simple_ds)
    assert "feature" in coords


def test_dot_dedup_data(simple_ds):
    """Two Dot terms with same data_var share the pmd.Data."""
    dot_a = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    dot_b = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    terms = dot_a + dot_b
    coords = get_coords(terms, simple_ds)
    with pm.Model(coords=coords) as m:
        register_data(terms, ds=simple_ds)
        assert "x" in m


def test_dot_duplicate_prior_name_errors(simple_ds):
    """Same data_var AND same prior name clash on variable names."""
    dot_a = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    dot_b = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    terms = dot_a + dot_b
    coords = get_coords(terms, simple_ds)
    with pm.Model(coords=coords):
        register_data(terms, ds=simple_ds)
        with pytest.raises(ValueError, match="already exists"):
            build_param(terms)


def test_add_coords_dynamic(simple_ds):
    """Custom term overrides add_coords to add model-internal coords."""

    @dataclass
    class _GroupTerm(ModelTerm):
        data_source: str

        def get_coords(self, ds):
            return {
                k: v.values.tolist() for k, v in ds[self.data_source].coords.items()
            }

        def add_coords(self, ds):
            model = pm.modelcontext(None)
            unique = list(dict.fromkeys(ds[self.data_source].values))
            model.add_coords({self.data_source: unique})

        def register_data(self, ds):
            pmd.Data(f"{self.data_source}_idx", np.arange(len(ds["obs"])), dims="obs")

    ds = xr.Dataset(
        {"group": ("obs", ["a", "b", "a", "b", "c"] * 10), "y": ("obs", np.arange(50))},
        coords={"obs": range(50)},
    )
    mu = _GroupTerm("group") + Intercept(name="mu")
    coords = collect_coords(mu, ds=ds)
    with pm.Model(coords=coords) as m:
        register_data(mu, ds=ds)
        assert "group" in m.coords


def test_dot_set_data_renamed_dim(simple_ds):
    """set_data with same shape but renamed coordinate labels."""
    ds2 = simple_ds.copy()
    ds2 = ds2.assign_coords(feature=["X", "Y", "Z"])

    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    coords = get_coords(dot, simple_ds)
    with pm.Model(coords=coords) as model:
        dot.register_data(simple_ds)
        dot.set_data(ds2)
        assert np.allclose(model["x"].get_value(), ds2["x"].values)


def test_build_param_with_subtraction():
    """Subtraction produces Sum with Product(-1, term)."""
    terms = Intercept(name="a") - Intercept(name="b")
    assert isinstance(terms, Sum)
    assert isinstance(terms.terms[1], Product)
    with pm.Model():
        result = build_param(terms)
        assert isinstance(result, PTVariable)


def test_collect_terms_deep_nesting(simple_ds):
    """collect_terms flattens deeply nested structures."""
    terms = Product(
        Intercept(name="a"),
        Sum(
            [
                Intercept(name="b"),
                Dot(var_name="x", prior=Prior("Normal", dims="feature")),
            ]
        ),
    )
    result = collect_terms([terms])
    assert len(result) == 3


def test_register_data_skips_literals(simple_ds):
    """register_data ignores int/float terms in a Sum."""
    terms = Sum([Intercept(name="a"), 5, 3.0])
    coords = collect_coords(terms, ds=simple_ds)
    with pm.Model(coords=coords) as model:
        register_data(terms, ds=simple_ds)
        # Literals are skipped; Intercept has no data to register
        assert len(model.data_vars) == 0


def test_product_rmul():
    p = Product(Intercept(name="a"), 2.0)
    result = 3 * p
    assert isinstance(result, Product)


def test_sum_rmul():
    s = Sum([Intercept(name="a")])
    result = 3 * s
    assert isinstance(result, Product)


def test_modelterm_set_data_default():
    t = Intercept(name="x")
    t.set_data(xr.Dataset(), model=None)  # should not raise


def test_modelterm_add_coords_default():
    t = Intercept(name="x")
    with pm.Model():
        t.add_coords(xr.Dataset())  # should not raise


def test_custom_term():
    @dataclass
    class _Custom(ModelTerm):
        value: float = 1.0

        def create_variable(self):
            return pt.as_tensor_variable(self.value)

    with pm.Model():
        result = build_param(_Custom(42.0))
        assert isinstance(result, PTVariable)


def test_dot_default_name(simple_ds):
    """Dot without `name` defaults to `{var_name}_beta` (unchanged behavior)."""
    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    assert dot.name == "x_beta"


def test_dot_distinct_name_no_collision(simple_ds):
    """CLV: alpha/beta branches share data but need distinct coef names.

    Two Dot terms referencing the same ``var_name`` with distinct ``name``
    should build separate coefficient variables without colliding.
    """
    dot_a = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    dot_b = Dot(var_name="x", prior=Prior("Normal", dims="feature"), name="x_coef_b")
    mu = dot_a + dot_b
    coords = collect_coords(mu, ds=simple_ds)
    with pm.Model(coords=coords) as model:
        register_data(mu, ds=simple_ds)
        build_param(mu)
        assert "x_beta" in model.named_vars
        assert "x_coef_b" in model.named_vars


def test_collect_coords_subtraction(simple_ds):
    """CLV gotcha: `baseline - dot(...)` coordinates must be collected.

    Regression test for `Sum.get_coords` dropping Product children (the
    Product(-1, Dot) produced by subtraction).
    """
    mu = Intercept(name="a") - Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    coords = collect_coords(mu, ds=simple_ds)
    assert "feature" in coords


def test_collect_coords_multiplication(simple_ds):
    """CLV gotcha: `a + b * c` must collect coordinates from the Product child."""
    mu = Intercept(name="a") + Dot(
        var_name="x", prior=Prior("Normal", dims="feature")
    ) * Transform(Intercept(name="scale"), func=ptx.math.exp)
    coords = collect_coords(mu, ds=simple_ds)
    assert "feature" in coords


def test_set_data_subtraction(simple_ds):
    """CLV gotcha: `baseline - dot(...)` must still update the Dot's data on prediction.

    Regression test for `Sum.set_data` dropping Product children, which would
    silently leave the covariate `pmd.Data` stale during prediction.
    """
    mu = Intercept(name="a") - Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    coords = collect_coords(mu, ds=simple_ds)
    with pm.Model(coords=coords) as model:
        register_data(mu, ds=simple_ds)
        build_param(mu)
        ds2 = simple_ds.copy()
        ds2["x"] = xr.DataArray(
            np.roll(simple_ds["x"].values, 1, axis=0), dims=("obs", "feature")
        )
        set_data(mu, ds=ds2, model=model)
        updated = model["x"].get_value()
    assert np.allclose(updated, ds2["x"].values)


def test_set_data_multiplication(simple_ds):
    """CLV gotcha: `a + b * c` must still update the Dot's data on prediction.

    Mirror of `test_set_data_subtraction` for the `*` trigger (Product child
    inside a Sum).
    """
    mu = Intercept(name="a") + Dot(
        var_name="x", prior=Prior("Normal", dims="feature")
    ) * Transform(Intercept(name="scale"), func=ptx.math.exp)
    coords = collect_coords(mu, ds=simple_ds)
    with pm.Model(coords=coords) as model:
        register_data(mu, ds=simple_ds)
        build_param(mu)
        ds2 = simple_ds.copy()
        ds2["x"] = xr.DataArray(
            np.roll(simple_ds["x"].values, 1, axis=0), dims=("obs", "feature")
        )
        set_data(mu, ds=ds2, model=model)
        updated = model["x"].get_value()
    assert np.allclose(updated, ds2["x"].values)


def test_register_data_subtraction(simple_ds):
    """`register_data` of `a - b` must register the subtracted Dot's data."""
    mu = Intercept(name="a") - Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    coords = collect_coords(mu, ds=simple_ds)
    with pm.Model(coords=coords) as model:
        register_data(mu, ds=simple_ds)
        assert "x" in model


@pytest.fixture
def ds_with_dims():
    """Dataset with an extra non-obs dimension for free-parameter dims."""
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "x": (("obs", "feature"), rng.normal(size=(20, 3))),
            "y": ("obs", rng.normal(size=20)),
        },
        coords={
            "obs": range(20),
            "feature": list("ABC"),
            "product": ["p1", "p2", "p3"],
        },
    )


def test_parameter_requires_name():
    """Parameter without a name raises TypeError."""
    with pytest.raises(TypeError):
        Parameter()


def test_parameter_get_coords_with_dims(ds_with_dims):
    """Parameter with prior.dims extracts matching coordinates."""
    p = Parameter("alpha", prior=Prior("Normal", dims="product"))
    coords = p.get_coords(ds_with_dims)
    assert "product" in coords
    assert coords["product"] == ["p1", "p2", "p3"]


def test_parameter_get_coords_no_dims(ds_with_dims):
    """Parameter without dims returns empty coords."""
    p = Parameter("mu")
    coords = p.get_coords(ds_with_dims)
    assert coords == {}


def test_parameter_get_coords_missing_dim():
    """Parameter with dims not in ds returns empty coords."""
    ds = xr.Dataset({"x": ("obs", [1, 2])}, coords={"obs": range(2)})
    p = Parameter("alpha", prior=Prior("Normal", dims="nonexistent"))
    coords = p.get_coords(ds)
    assert coords == {}


def test_intercept_get_coords_with_dims(ds_with_dims):
    """Intercept (subclass of Parameter) also extracts prior.dims coords."""
    p = Intercept("baseline", prior=Prior("Normal", dims="product"))
    coords = p.get_coords(ds_with_dims)
    assert "product" in coords
    assert coords["product"] == ["p1", "p2", "p3"]


def test_intercept_default_name():
    """Intercept defaults name to 'intercept'."""
    p = Intercept()
    assert p.name == "intercept"


def test_collect_coords_includes_parameter_dims(ds_with_dims):
    """collect_coords picks up coords from a dimmed Parameter in a Sum."""
    mu = Intercept("a") + Parameter("alpha", prior=Prior("Normal", dims="product"))
    coords = collect_coords(mu, ds=ds_with_dims)
    assert "product" in coords


def test_parameter_create_variable():
    """Parameter create_variable builds a named tensor."""
    with pm.Model():
        p = Parameter("alpha", prior=Prior("Normal", dims="product"))
        with pm.Model(coords={"product": ["p1", "p2", "p3"]}):
            v = p.create_variable()
            assert v is not None


def test_serialize_parameter_roundtrip():
    term = Parameter("alpha", prior=Prior("Normal", mu=0, sigma=1, dims="product"))
    restored = serialization.deserialize(serialization.serialize(term))
    assert restored == term
    assert restored.prior.dims == ("product",)


def test_serialize_dot_roundtrip():
    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"), name="x_coef")
    restored = serialization.deserialize(serialization.serialize(dot))
    assert restored == dot
    assert restored.name == "x_coef"


def test_serialize_intercept_subclass_roundtrip():
    term = Intercept("baseline", prior=Prior("Normal"))
    restored = serialization.deserialize(serialization.serialize(term))
    assert isinstance(restored, Intercept)
    assert restored == term


def test_serialize_composition_roundtrip():
    alpha = Parameter("alpha_scale", prior=Prior("HalfFlat")) * Transform(
        -Dot(
            var_name="purchase_data",
            name="purchase_coefficient_alpha",
            prior=Prior("Normal", mu=0, sigma=1, dims="purchase_covariate"),
        ),
        func=ptx.math.exp,
    )
    restored = serialization.deserialize(serialization.serialize(alpha))
    assert restored == alpha


def test_serialize_sum_with_literals_roundtrip():
    expr = Intercept(name="a") + 5 - Intercept(name="b")
    restored = serialization.deserialize(serialization.serialize(expr))
    assert restored == expr


def test_serialize_unregistered_func_raises():
    with pytest.raises(SerializationError, match="not serializable"):
        serialization.serialize(Transform(Parameter("x"), func=pt.sqrt))


def test_serialize_registered_custom_transform_roundtrip(monkeypatch):
    def square(x):
        return x**2

    monkeypatch.setitem(CUSTOM_TRANSFORMS, "square", square)
    term = Transform(Parameter("x"), func=square)
    restored = serialization.deserialize(serialization.serialize(term))
    assert restored == term
    assert restored.func is square


def test_restored_term_builds(simple_ds):
    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    restored = serialization.deserialize(serialization.serialize(dot))
    coords = collect_coords(restored, ds=simple_ds)
    with pm.Model(coords=coords):
        register_data(restored, ds=simple_ds)
        assert isinstance(build_param(restored), PTVariable)


def test_serialize_shared_r2d2_decomposition_through_parameter():
    """Regression: R2D2 splits wrapped in terms must share one decomposition.

    Each deserialized R2D2Split otherwise builds its own decomposition, and
    the second ``create_variable`` call collides on the ``r2d2_*`` names.
    """
    r2d2 = R2D2(
        r2=Prior("Beta", alpha=2, beta=2),
        total_sigma=Prior("HalfNormal"),
        dims={"control": "control", "fourier": "fourier"},
    )
    config = {
        "a": Parameter("a", prior=r2d2.split("control")),
        "b": Parameter("b", prior=r2d2.split("fourier")),
    }
    serialized = serialization.serialize_model_config(config)
    loaded = serialization.deserialize_model_config(json.loads(json.dumps(serialized)))

    assert loaded["a"].prior.decomposition is loaded["b"].prior.decomposition

    with pm.Model(coords={"control": ["c1", "c2"], "fourier": ["f1", "f2"]}) as model:
        loaded["a"].prior.create_variable("a_coef")
        loaded["b"].prior.create_variable("b_coef")
        assert model["a_coef"] is not None
        assert model["b_coef"] is not None


@pytest.mark.parametrize(
    "literal",
    [
        True,
        False,
        5,
        2.5,
        np.int64(3),
        np.float64(2.5),
        np.bool_(True),
    ],
)
def test_serialize_json_roundtrip_with_numpy_literals(literal):
    """Term children that are numpy scalars or bools survive a JSON dump."""
    term = Parameter("a") * literal
    serialized = json.dumps(serialization.serialize(term))
    restored = serialization.deserialize(json.loads(serialized))

    assert restored == term
    assert restored.right == literal
    assert not isinstance(restored.right, (np.number, np.bool_))


def test_serialize_model_config_with_term_roundtrip():
    """Terms in model_config round-trip through serialize/deserialize_model_config."""
    config = {
        "mu": Intercept(name="a")
        + Dot(var_name="x", prior=Prior("Normal", dims="feature")),
        "sigma": Transform(Parameter("sigma"), func=ptx.math.exp),
    }
    serialized = serialization.serialize_model_config(config)
    json.dumps(serialized)
    loaded = serialization.deserialize_model_config(json.loads(json.dumps(serialized)))

    assert loaded["mu"] == config["mu"]
    assert loaded["sigma"] == config["sigma"]


def test_deserialize_unknown_func_raises():
    term = Transform(Parameter("x"), func=ptx.math.exp)
    data = serialization.serialize(term)
    data["func"] = "nope"
    with pytest.raises(SerializationError, match="Unknown serialized function"):
        serialization.deserialize(data)


def test_deserialize_non_callable_func_raises():
    """Function names from a file must resolve to callables, not modules."""
    term = Transform(Parameter("x"), func=ptx.math.exp)
    data = serialization.serialize(term)
    data["func"] = "basic"
    with pytest.raises(SerializationError, match="callable"):
        serialization.deserialize(data)


@dataclass
class UnregisteredWithToDict(ModelTerm):
    """Custom term with to_dict but no @serialization.register."""

    def to_dict(self) -> dict:
        return {"k": 3}


@dataclass
class UnregisteredBare(ModelTerm):
    """Custom term without to_dict and no @serialization.register."""


@pytest.mark.parametrize("term", [UnregisteredWithToDict(), UnregisteredBare()])
def test_serialize_unregistered_custom_term_raises(term):
    """Unregistered custom terms fail at serialize time, not load time."""
    with pytest.raises(SerializationError, match=r"serialization\.register"):
        serialization.serialize(Sum(terms=[term]))


def test_serialize_deferred_factory_roundtrip():
    deferred = DeferredFactory(factory="builtins.dict", kwargs={"a": 1})
    term = Parameter("a", prior=deferred)
    restored = serialization.deserialize(serialization.serialize(term))
    assert restored.prior == deferred


def test_serialize_data_array_child_roundtrip():
    """xr.DataArray children serialize via to_dict and load via pymc-extras."""
    da = xr.DataArray([1.0, 2.0], dims="d")
    term = Parameter("a", prior=da)
    serialized = serialization.serialize(term)
    json.dumps(serialized)
    restored = serialization.deserialize(json.loads(json.dumps(serialized)))
    assert restored.prior.equals(da)


class _TermAttrsModel(ModelBuilder):
    """Minimal builder to exercise create_idata_attrs with terms in config."""

    _model_type = "terms_attrs_test"
    version = "0.1"

    @property
    def default_model_config(self) -> dict:
        return {"mu": None}

    @property
    def default_sampler_config(self) -> dict:
        return {}

    @property
    def output_var(self) -> str:
        return "y"

    @property
    def _serializable_model_config(self) -> dict:
        return self.model_config

    def build_model(self, X=None, y=None, **kwargs):
        pass

    def build_from_idata(self, idata):
        pass

    def _data_setter(self, X, y=None):
        pass


def test_model_builder_attrs_roundtrip_with_term():
    """Terms in model_config survive the create_idata_attrs JSON dump."""
    term = Intercept(name="a") + Dot(
        var_name="x", prior=Prior("Normal", dims="feature")
    )
    model = _TermAttrsModel(model_config={"mu": term})

    attrs = model.create_idata_attrs()
    loaded = serialization.deserialize_model_config(json.loads(attrs["model_config"]))

    assert loaded["mu"] == term


def test_frozen_term_dataclass_walk_shares_decomposition():
    """Frozen custom terms do not bypass decomposition sharing (serialization walk)."""
    from dataclasses import dataclass

    r2d2 = R2D2(
        r2=Prior("Beta", mu=0.8, sigma=0.4),
        total_sigma=Prior("LogNormal", mu=0, sigma=1),
        dims={"control": "control", "fourier": "fourier"},
    )

    @serialization.register
    @dataclass(frozen=True)
    class FrozenHolder:
        prior: object

        def to_dict(self):
            return {"prior": _serialize_child(self.prior)}

        @classmethod
        def from_dict(cls, data):
            return cls(prior=_deserialize_child(data["prior"]))

    config = {
        "a": Parameter("a", prior=r2d2.split("control")),
        "b": FrozenHolder(prior=r2d2.split("fourier")),
    }
    loaded = serialization.deserialize_model_config(
        json.loads(json.dumps(serialization.serialize_model_config(config)))
    )

    assert loaded["a"].prior.decomposition is loaded["b"].prior.decomposition


def test_resolve_func_allows_registered_underscore_name(monkeypatch):
    """Explicitly registered underscore names are loadable (not module noise)."""

    def _secret(x):
        return x

    monkeypatch.setitem(CUSTOM_TRANSFORMS, "_secret", _secret)
    term = Transform(Parameter("x"), func=_secret)
    restored = serialization.deserialize(serialization.serialize(term))
    assert restored.func is _secret


def test_deserialize_custom_factory_error_names_register_deserialization():
    """A VariableFactory pymc-extras cannot read back fails with guidance."""

    class WriteOnlyFactory:
        dims = None

        def create_variable(self, name, xdist=False):
            return pt.as_tensor_variable(1.0)

        def to_dict(self):
            return {"k": 2}

    term = Parameter("a", prior=WriteOnlyFactory())
    serialized = serialization.serialize(term)
    with pytest.raises(SerializationError, match="register_deserialization"):
        serialization.deserialize(serialized)


def test_deserialize_child_passthrough_non_dict():
    """Non-dict children pass through (defensive; _serialize_child emits dicts)."""
    from pymc_marketing.terms import _deserialize_child

    assert _deserialize_child(52) == 52
def test_named_builds_pmd_deterministic():
    coords = {"product": ["p1", "p2"]}
    with pm.Model(coords=coords) as model:
        term = Named(
            "sigma",
            Parameter("scale", prior=Prior("HalfNormal", dims="product")),
            dims="product",
        )
        term.create_variable()

    assert "sigma" in model.named_vars
    assert model.named_vars_to_dims["sigma"] == ("product",)


def test_named_dims_none_scalar():
    with pm.Model() as model:
        term = Named(
            "sigma",
            Parameter("scale", prior=Prior("HalfNormal")),
            dims=None,
        )
        term.create_variable()

    assert "sigma" in model.named_vars
    assert model.named_vars_to_dims["sigma"] == ()


def test_named_delegates_lifecycle(simple_ds):
    term = Named(
        "effect",
        Transform(
            Dot(var_name="x", prior=Prior("Normal", dims="feature")),
            func=ptx.math.exp,
        ),
        dims="obs",
    )
    coords = term.get_coords(simple_ds)
    assert "feature" in coords

    with pm.Model(coords=coords) as model:
        term.register_data(simple_ds)
        term.create_variable()

    assert "effect" in model.named_vars
    assert model.named_vars_to_dims["effect"] == ("obs",)

    ds2 = simple_ds.copy()
    ds2["x"] = xr.DataArray(
        np.roll(simple_ds["x"].values, 1, axis=0), dims=("obs", "feature")
    )
    term.set_data(ds2, model=model)
    assert np.allclose(model["x"].get_value(), ds2["x"].values)


def test_ref_resolves_built_variable():
    coords = {"product": ["p1", "p2"]}
    with pm.Model(coords=coords) as model:
        scale = Named(
            "a_scale",
            Parameter("phi", prior=Prior("Uniform", lower=0, upper=1, dims="product")),
            dims="product",
        )
        scale.create_variable()

        effect = Named(
            "a",
            Ref("a_scale")
            * Parameter("kappa", prior=Prior("HalfNormal", sigma=1, dims="product")),
            dims="product",
        )
        effect.create_variable()

    assert "a_scale" in model.named_vars
    assert "a" in model.named_vars
    assert model.named_vars_to_dims["a"] == ("product",)


def test_ref_missing_raises():
    with pm.Model():
        ref = Ref("missing")
        with pytest.raises(KeyError):
            ref.create_variable()


def test_serialize_named_roundtrip():
    term = Named(
        "alpha",
        Parameter("alpha_scale", prior=Prior("HalfFlat"))
        * Transform(
            Dot(
                var_name="purchase_data",
                name="purchase_coefficient_alpha",
                prior=Prior("Normal", mu=0, sigma=1, dims="purchase_covariate"),
            ),
            func=ptx.math.exp,
        ),
        dims="customer_id",
    )
    restored = serialization.deserialize(serialization.serialize(term))
    assert restored == term
    assert restored.dims == "customer_id"


def test_serialize_ref_roundtrip():
    term = Ref("a_scale")
    restored = serialization.deserialize(serialization.serialize(term))
    assert restored == term
