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

from dataclasses import dataclass

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import pytensor.xtensor.math as ptx
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from pytensor.graph.basic import Variable as PTVariable

from pymc_marketing.terms import (
    Dot,
    Intercept,
    ModelTerm,
    Parameter,
    Product,
    Sum,
    Transform,
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
        transformed = Transform(inner, func=ptx.exp)
        result = transformed.create_variable()
        assert isinstance(result, PTVariable)


def test_transform_with_dot(simple_ds):
    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    transformed = Transform(dot, func=ptx.exp)
    coords = transformed.get_coords(simple_ds)
    with pm.Model(coords=coords):
        transformed.register_data(simple_ds)
        result = transformed.create_variable()
        assert isinstance(result, PTVariable)


def test_transform_set_data(simple_ds):
    inner = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    transformed = Transform(inner, func=ptx.exp)
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
        result = build_param(Transform(Intercept(name="sigma"), func=ptx.exp))
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
    terms = [Transform(Intercept(name="a"), func=ptx.exp)]
    result = collect_terms(terms)
    assert len(result) == 1


def test_collect_coords(simple_ds):
    """collect_coords merges coordinates from multiple term trees."""
    mu = Intercept(name="mu") + Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    sigma = Transform(Intercept(name="sigma"), func=ptx.exp)
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


def test_product_add():
    p = Product(Intercept(name="a"), 2.0)
    result = p + Intercept(name="b")
    assert isinstance(result, Sum)
    assert result.terms[0] == p
    assert result.terms[1] == Intercept(name="b")


def test_product_radd_int():
    p = Product(Intercept(name="a"), 2.0)
    result = 3 + p
    assert isinstance(result, Sum)
    assert result.terms[0] == 3
    assert result.terms[1] == p


def test_product_radd_zero():
    p = Product(Intercept(name="a"), 2.0)
    result = 0 + p
    assert result == p


def test_product_add_product():
    result = -Intercept(name="a") + 3 * Parameter(name="b", prior=Prior("Normal"))
    assert isinstance(result, Sum)
    assert len(result.terms) == 2
    assert all(isinstance(term, Product) for term in result.terms)


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
    ) * Transform(Intercept(name="scale"), func=ptx.exp)
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
    ) * Transform(Intercept(name="scale"), func=ptx.exp)
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
