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

"""Observable stochastic and lifecycle contracts for experimental equations."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from pymc_extras.prior import Prior
from scipy.special import gammaln

from pymc_marketing.experimental.terms import SCM
from pymc_marketing.experimental.terms._graph import (
    BuildContext,
    Data,
    Equation,
    copy_prior,
    specification_key,
    walk,
)
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.terms import (
    Dot,
    ModelTerm,
    Parameter,
    Transform,
    build_param,
    get_coords,
    register_data,
)


def _normal_logp(value: Any, mu: Any, sigma: float) -> Any:
    return -0.5 * ((value - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def test_joint_equations_use_observed_intermediate_in_joint_logp():
    ds = xr.Dataset(
        {
            "x": ("date", [1.0, 2.0, 3.0]),
            "spend": ("date", [2.0, 4.0, 3.0]),
            "sales": ("date", [5.0, 6.0, 4.0]),
        },
        coords={"date": [0, 1, 2]},
    )
    theta = Parameter("theta", Prior("Normal", sigma=2))
    spend = Equation(
        mu=theta * Data("x"),
        likelihood=Prior("Normal", sigma=1),
        name="B",
        observed="spend",
    )
    sales = Equation(
        mu=spend + theta,
        likelihood=Prior("Normal", sigma=0.5),
        name="Y",
        observed="sales",
    )
    with pm.Model() as model:
        BuildContext(ds).build(sales)
    theta_value = 0.3
    expected = (
        _normal_logp(theta_value, 0, 2)
        + _normal_logp(ds.spend.values, theta_value * ds.x.values, 1).sum()
        + _normal_logp(ds.sales.values, ds.spend.values + theta_value, 0.5).sum()
    )
    assert_allclose(model.compile_logp()({"theta": theta_value}), expected)
    assert {variable.name for variable in model.observed_RVs} == {"B", "Y"}


def test_repeated_custom_lifecycle_uses_one_clone_and_shared_parameter():
    @dataclass
    class Calibrated(ModelTerm):
        child: Any

        def get_coords(self, ds):
            self.weight = ds.attrs["weight"]
            return get_coords(self.child, ds)

        def add_coords(self, ds):
            self.child.add_coords(ds)

        def register_data(self, ds):
            register_data(self.child, ds=ds)
            self.calibration = pmd.Data("calibration", np.asarray(self.weight))

        def create_variable(self):
            return self.calibration * build_param(self.child)

    theta = Parameter("theta", Prior("Normal"))
    calibrated = Calibrated(theta)
    expression = Transform(calibrated + calibrated, pmd.math.exp) + calibrated * theta
    ds = xr.Dataset(attrs={"weight": 3.0})
    before = specification_key(expression)
    with pm.Model() as model:
        context = BuildContext(ds)
        value = context.build(expression)
        evaluate = pytensor.function([model["theta"]], value)
    assert_allclose(evaluate(0.5), np.exp(3.0) + 0.75)
    assert {variable.name for variable in model.free_RVs} == {"theta"}
    assert specification_key(expression) == before
    assert not hasattr(calibrated, "calibration")
    assert not hasattr(calibrated, "weight")


def test_shared_terms_in_custom_containers_retain_identity():
    @dataclass
    class Group(ModelTerm):
        children: dict[str, list[Any]]

        def get_coords(self, ds):
            return get_coords(self.children["terms"][0], ds)

        def register_data(self, ds):
            for child in self.children["terms"]:
                register_data(child, ds=ds)

        def create_variable(self):
            return sum(build_param(child) for child in self.children["terms"])

    theta = Parameter("theta", Prior("Normal"))
    group = Group({"terms": [theta, Transform(theta, pmd.math.exp)]})
    with pm.Model() as model:
        value = BuildContext(xr.Dataset()).build(group + theta)
        evaluate = pytensor.function([model["theta"]], value)
    assert_allclose(evaluate(0.4), 0.8 + np.exp(0.4))
    assert {variable.name for variable in model.free_RVs} == {"theta"}


def test_prior_recipe_reuse_does_not_imply_parameter_sharing():
    recipe = Prior("Normal", sigma=2)
    left = Equation(
        parameters={"mu": recipe},
        likelihood=Prior("Normal", sigma=1),
        name="left",
        dims=(),
    )
    right = Equation(
        parameters={"mu": recipe},
        likelihood=Prior("Normal", sigma=1),
        name="right",
        dims=(),
    )
    with pm.Model() as model:
        BuildContext(xr.Dataset()).build(left + right)
    assert {variable.name for variable in model.free_RVs} == {
        "left_mu",
        "left",
        "right_mu",
        "right",
    }


def test_copy_prior_preserves_nested_multivariate_core_dimensions():
    alpha = xr.DataArray([1.0, 2.0, 3.0], dims="category")
    prior = Prior(
        "Dirichlet",
        a=Prior("Dirichlet", a=alpha, dims="category", core_dims="category"),
        dims="category",
        core_dims="category",
    )
    before = specification_key(prior)
    with pm.Model(coords={"category": ["a", "b", "c"]}) as model:
        variable = copy_prior(prior).create_variable("probabilities", xdist=True)
        outer, inner = pm.draw(
            [variable, model["probabilities_a"]], draws=5, random_seed=124
        )
    assert_allclose(outer.sum(axis=-1), 1)
    assert_allclose(inner.sum(axis=-1), 1)
    assert specification_key(prior) == before


def test_non_mu_distribution_parameters_build_real_likelihood():
    ds = xr.Dataset(
        {"score": ("date", [-1.0, 0.0, 2.0]), "amount": ("date", [0.2, 1.0, 3.0])},
        coords={"date": [0, 1, 2]},
    )
    equation = Equation(
        parameters={"alpha": Transform(Data("score"), pmd.math.exp)},
        likelihood=Prior("Gamma", beta=2),
        name="amount_process",
        observed="amount",
    )
    with pm.Model() as model:
        BuildContext(ds).build(equation)
    alpha = np.exp(ds.score.values)
    expected = (
        alpha * np.log(2)
        - gammaln(alpha)
        + (alpha - 1) * np.log(ds.amount.values)
        - 2 * ds.amount.values
    ).sum()
    assert_allclose(model.compile_logp()({}), expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mu": 0, "parameters": {"mu": 1}},
        {"mu": 0, "likelihood": Prior("Normal", mu=1)},
        {"parameters": {"sigma": 2}, "likelihood": Prior("Normal", sigma=1)},
    ],
)
def test_duplicate_distribution_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Equation(**kwargs)


def test_walk_and_build_distinguish_shared_dag_from_true_cycle():
    theta = Parameter("theta", Prior("Normal"))
    intermediate = Equation(mu=theta, name="B", dims=())
    root = Equation(mu=intermediate + intermediate, name="Y", dims=())
    assert sum(node is intermediate for node in walk(root)) == 1
    intermediate.mu = root
    with pytest.raises(ValueError, match="Cycle"):
        list(walk(root))
    with pm.Model(), pytest.raises(ValueError, match="Cycle"):
        BuildContext(xr.Dataset()).build(root)


@pytest.mark.parametrize("kind", ["parameter", "equation"])
def test_distinct_graph_nodes_cannot_own_the_same_model_name(kind):
    if kind == "parameter":
        left = Parameter("same", Prior("Normal"))
        right = Parameter("same", Prior("Normal"))
    else:
        left = Equation(name="same", likelihood=Prior("Normal"), dims=())
        right = Equation(name="same", likelihood=Prior("Normal"), dims=())
    with pm.Model(), pytest.raises(ValueError, match="name"):
        BuildContext(xr.Dataset()).build(left + right)


def test_dot_cannot_silently_consume_an_equation_instead_of_raw_data():
    ds = xr.Dataset(
        {"x": (("date", "feature"), [[1.0], [2.0]])},
        coords={"date": [0, 1], "feature": ["a"]},
    )
    equation = Equation(name="x", likelihood=Prior("Normal"), dims=())
    dot = Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    with pm.Model(), pytest.raises(ValueError, match="name"):
        context = BuildContext(ds)
        context.build(equation)
        context.build(dot)


def test_distinct_dot_coefficients_can_share_raw_data():
    ds = xr.Dataset(
        {"x": (("date", "feature"), [[1.0], [2.0]])},
        coords={"date": [0, 1], "feature": ["a"]},
    )
    left = Dot(var_name="x", name="left", prior=Prior("Normal", dims="feature"))
    right = Dot(var_name="x", name="right", prior=Prior("Normal", dims="feature"))
    with pm.Model() as model:
        value = BuildContext(ds).build(left + right)
        evaluate = pytensor.function([model["left"], model["right"]], value)
    assert_allclose(evaluate([2.0], [3.0]), [5.0, 10.0])


def test_conditioned_equation_stops_missing_mechanism_inputs():
    ds = xr.Dataset({"spend": ("date", [3.0, 5.0])}, coords={"date": [0, 1]})
    ancestor = Equation(
        mu=Data("unavailable"), name="ancestor", observed="also_unavailable"
    )
    spend = Equation(mu=ancestor, name="B", observed="spend")
    root = Equation(
        mu=spend * 2,
        likelihood=Prior("Normal", sigma=1),
        name="Y",
        observed="sales",
        dims="date",
    )
    with pm.Model() as model:
        context = BuildContext(ds, prediction=True, condition_on=("B",))
        context.build(root)
        values = pm.draw(model["B"], random_seed=12)
    assert_allclose(values, ds.spend.values)
    assert {variable.name for variable in model.free_RVs} == {"Y"}
    assert "ancestor" not in model


def test_generated_equation_uses_measured_history_without_registering_future_nans():
    ds = xr.Dataset(
        {"spend": ("date", [10.0, 20.0, np.nan, np.nan])}, coords={"date": [0, 1, 2, 3]}
    )
    equation = Equation(
        mu=1, likelihood=Prior("Normal", sigma=1), name="B", observed="spend"
    )
    with pm.Model() as model:
        context = BuildContext(ds, prediction=True, history_length=2)
        downstream = context.build(equation)
        evaluate = pytensor.function([model["B"]], downstream)
    assert_allclose(evaluate([100.0, 200.0, 3.0, 4.0]), [10.0, 20.0, 3.0, 4.0])
    assert {variable.name for variable in model.free_RVs} == {"B"}
    assert context.equation_names[id(equation)] == "B"
    assert not model.data_vars


def test_missing_historical_observation_is_rejected():
    ds = xr.Dataset({"spend": ("date", [np.nan, np.nan])}, coords={"date": [0, 1]})
    equation = Equation(mu=1, name="B", observed="spend")
    with pm.Model(), pytest.raises(ValueError):
        BuildContext(ds, prediction=True, history_length=1).build(equation)


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_missing_training_observations_do_not_create_imputation_variables(missing):
    ds = xr.Dataset({"sales": ("date", [1.0, missing])}, coords={"date": [0, 1]})
    equation = Equation(mu=0, name="Y", observed="sales")
    with pm.Model(), pytest.raises(ValueError):
        BuildContext(ds).build(equation)


def test_equation_dims_follow_observations_then_likelihood_then_scalar():
    ds = xr.Dataset(
        {"y": (("date", "geo"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"date": [0, 1], "geo": ["a", "b"]},
    )
    scalar = Equation(mu=0, likelihood=Prior("Normal", sigma=1), name="scalar")
    regional = Equation(
        mu=0, likelihood=Prior("Normal", sigma=1, dims="geo"), name="regional"
    )
    observed = Equation(
        mu=scalar + regional, likelihood=Prior("Normal", sigma=1), observed="y"
    )
    with pm.Model() as model:
        BuildContext(ds).build(observed)
    assert model.named_vars_to_dims["scalar"] == ()
    assert model["scalar"].ndim == 0
    assert model.named_vars_to_dims["regional"] == ("geo",)
    assert model.named_vars_to_dims["y"] == ("date", "geo")
    expected = (
        _normal_logp(0.2, 0, 1)
        + _normal_logp(np.array([0.1, -0.1]), 0, 1).sum()
        + _normal_logp(ds.y.values, 0.2 + np.array([0.1, -0.1]), 1).sum()
    )
    assert_allclose(
        model.compile_logp()({"scalar": 0.2, "regional": [0.1, -0.1]}), expected
    )


def test_specification_key_detects_topology_callables_and_transformation_mutations():
    theta = Parameter("theta", Prior("Normal"))
    shared = theta + theta
    independent = theta + Parameter("theta", Prior("Normal"))
    assert specification_key(shared) != specification_key(independent)
    expression = Transform(theta, pmd.math.exp)
    before = specification_key(expression)
    expression.func = pmd.math.sigmoid
    assert specification_key(expression) != before
    adstock = GeometricAdstock(l_max=3)
    before = specification_key(adstock)
    adstock.l_max = 5
    assert specification_key(adstock) != before
    before = specification_key(adstock)
    adstock.function_priors["alpha"].parameters["alpha"] = 2
    assert specification_key(adstock) != before


def test_specification_key_preserves_core_dims_but_ignores_prior_runtime_state():
    prior = Prior("Normal", sigma=2)
    before = specification_key(prior)
    with pm.Model():
        prior.create_variable("theta", xdist=True)
    assert specification_key(prior) == before
    prior.core_dims = ("component",)
    assert specification_key(prior) != before


@pytest.mark.parametrize("dot_first", [False, True])
def test_data_and_dot_share_the_container_named_after_their_column(dot_first):
    ds = xr.Dataset(
        {
            "features": (("date", "feature"), [[1.0], [2.0], [3.0]]),
            "sales": ("date", [1.0, 2.0, 3.0]),
            "aux": ("date", [10.0, 20.0, 30.0]),
        },
        coords={"date": np.arange(3), "feature": ["a"]},
    )
    raw = Equation(
        observed="sales",
        mu=Data("features").sum("feature"),
        likelihood=Prior("Normal", sigma=0.1),
    )
    dot = Equation(
        observed="aux",
        mu=Dot(
            var_name="features",
            prior=Prior("Normal", mu=1.0, sigma=0.2, dims="feature"),
        ),
        likelihood=Prior("Normal", sigma=0.1),
    )
    with pm.Model() as model:
        context = BuildContext(ds)
        for equation in (dot, raw) if dot_first else (raw, dot):
            context.build(equation)
    assert [variable.name for variable in model.data_vars] == ["features"]
    updated = np.array([[2.0], [4.0], [6.0]])
    pm.set_data({"features": updated}, model=model)
    coefficient = np.array([0.7])
    expected = (
        _normal_logp(coefficient, 1.0, 0.2).sum()
        + _normal_logp(ds["sales"], updated.sum(axis=1), 0.1).sum()
        + _normal_logp(ds["aux"], updated.sum(axis=1) * coefficient[0], 0.1).sum()
    )
    assert_allclose(
        model.compile_logp()({model.free_RVs[0].name: coefficient}), float(expected)
    )


@pytest.mark.parametrize("data_first", [False, True])
def test_data_name_collisions_ask_for_a_distinct_model_variable_name(data_first):
    ds = xr.Dataset(
        {"search": ("date", [1.0, 2.0, 3.0]), "sales": ("date", [2.0, 3.0, 4.0])},
        coords={"date": np.arange(3)},
    )
    search = Equation(
        observed="search",
        mu=Parameter("search_level", Prior("Normal")),
        likelihood=Prior("Normal", sigma=1.0),
    )
    sales = Equation(
        observed="sales", mu=Data("search"), likelihood=Prior("Normal", sigma=1.0)
    )
    with (
        pm.Model(),
        pytest.raises(ValueError, match=r"Data\('search'\).*Equation\(name="),
    ):
        context = BuildContext(ds)
        for equation in (sales, search) if data_first else (search, sales):
            context.build(equation)


def test_discrete_observations_keep_their_joint_likelihood_with_continuous_inputs():
    values = [1, 3, 4]
    ds = xr.Dataset(
        {
            "y": ("geo", np.asarray(values, dtype="int32")),
            "aux": ("geo", [0.0, 0.7, 1.4]),
        },
        coords={"geo": ["a", "b", "c"], "category": [0, 1, 2]},
    )
    before = ds.copy(deep=True)
    likelihood = Prior("Poisson", mu=2.0)
    expected_y = ds["y"].values * np.log(2.0) - 2.0 - gammaln(ds["y"].values + 1)
    # The input container takes the column name "y", so the outcome needs its own name.
    outcome = Equation(name="count", observed="y", likelihood=likelihood)
    slope = Parameter("slope", Prior("Normal"))
    downstream = Equation(
        observed="aux",
        mu=slope * Data("y"),
        likelihood=Prior("Normal", sigma=0.2),
    )
    with pm.Model() as model:
        context = BuildContext(ds)
        # Register y as a continuous predictor before reading it as raw observations.
        context.build(downstream)
        context.build(outcome)
    expected = (
        expected_y.sum()
        + float(_normal_logp(0.7, 0.0, 1.0))
        + _normal_logp(ds["aux"], 0.7 * ds["y"], 0.2).sum()
    )
    assert_allclose(model.compile_logp()({"slope": 0.7}), float(expected), atol=1e-10)
    xr.testing.assert_identical(ds, before)


@pytest.mark.parametrize("value", [2**53 + 1, -(2**53 + 1)])
def test_integer_inputs_cannot_lose_units_during_float_conversion(value):
    ds = xr.Dataset(
        {"x": ("geo", np.asarray([value], dtype="int64"))}, coords={"geo": ["a"]}
    )
    with pm.Model(), pytest.raises(ValueError):
        BuildContext(ds).build(Data("x"))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        pytest.param({"name": ""}, ValueError, "name", id="empty-name"),
        pytest.param({"dims": ("",)}, ValueError, "Dimensions", id="empty-dimension"),
        pytest.param(
            {"dims": ("geo", "geo")},
            ValueError,
            "unique",
            id="duplicate-dimensions",
        ),
        pytest.param(
            {"parameters": [("mu", 0)]},
            TypeError,
            "mapping",
            id="nonmapping-parameters",
        ),
        pytest.param(
            {"likelihood": object()},
            TypeError,
            "Prior",
            id="nonrecipe-likelihood",
        ),
        pytest.param(
            {"parameters": {"observed": [1.0]}},
            ValueError,
            "observations",
            id="observations-in-parameters",
        ),
    ],
)
def test_invalid_equation_configuration(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        Equation(**kwargs)


def test_mu_setter_cannot_overwrite_a_likelihood_parameter() -> None:
    likelihood = Prior("Normal", mu=1.0, sigma=2.0)
    equation = Equation(observed="y", likelihood=likelihood)
    before = specification_key(equation)

    with pytest.raises(ValueError, match="mu"):
        equation.mu = 3.0

    assert equation.mu is None
    assert equation.dependencies() == ()
    assert specification_key(equation) == before
    assert likelihood.parameters["mu"] == 1.0


def test_numpy_and_symbolic_parameters_keep_their_values_and_recipes() -> None:
    scale = pmd.as_xtensor(np.asarray(2.0))
    likelihood = Prior("Normal", sigma=scale)
    equation = Equation(mu=0.0, observed="y", likelihood=likelihood)
    equation.mu = np.float32(1.25)
    symbolic = Equation(mu=scale, observed="z", likelihood=Prior("Normal", sigma=1.0))
    data = xr.Dataset(
        {"y": ("geo", [0.0, 2.0]), "z": ("geo", [1.0, 3.0])},
        coords={"geo": ["a", "b"]},
    )
    before = specification_key(equation, symbolic)

    model = SCM(equation, symbolic).build_model(data)

    expected = _normal_logp(data["y"].values, 1.25, 2.0).sum()
    expected += _normal_logp(data["z"].values, 2.0, 1.0).sum()
    assert_allclose(model.compile_logp()({}), expected)
    assert equation.mu == 1.25
    assert symbolic.mu is scale
    assert likelihood.parameters["sigma"] is scale
    assert specification_key(equation, symbolic) == before


@pytest.mark.parametrize(
    "likelihood",
    [
        pytest.param(
            Prior("Normal", mu=0.0, sigma=1.0, transform="exp"),
            id="transformed",
        ),
        pytest.param(
            Prior("Normal", mu=0.0, sigma=1.0, centered=False),
            id="noncentered",
        ),
    ],
)
def test_observed_equations_require_direct_centered_likelihoods(
    likelihood: Prior,
) -> None:
    data = xr.Dataset({"y": ("geo", [1.0])}, coords={"geo": ["a"]})
    equation = Equation(observed="y", likelihood=likelihood)
    before = specification_key(equation)

    with pytest.raises(ValueError, match="likelihood recipes"):
        SCM(equation).build_model(data)

    assert specification_key(equation) == before


def test_equation_dimensions_must_match_observation_dimensions() -> None:
    data = xr.Dataset(
        {"y": ("geo", [1.0, 2.0])},
        coords={"geo": ["a", "b"], "target": ["units", "revenue"]},
    )
    equation = Equation(observed="y", mu=0.0, dims="target")

    with pytest.raises(ValueError, match="Observation dimensions"):
        SCM(equation).build_model(data)


def test_nonnumeric_observations_are_rejected_before_sampling() -> None:
    data = xr.Dataset({"y": ("geo", ["one", "two"])}, coords={"geo": ["a", "b"]})
    with pytest.raises(ValueError, match="finite numeric"):
        SCM(Equation(observed="y", mu=0.0)).build_model(data)


@pytest.mark.parametrize(
    "predictor",
    [
        pytest.param(Data("missing"), id="raw-data"),
        pytest.param(
            Dot(var_name="missing", prior=Prior("Normal", dims="feature")),
            id="dot-data",
        ),
    ],
)
def test_missing_predictors_are_not_silently_ignored(predictor: ModelTerm) -> None:
    data = xr.Dataset(
        {"y": ("geo", [1.0, 2.0])},
        coords={"geo": ["a", "b"], "feature": ["x"]},
    )
    with pytest.raises(ValueError, match="Required data variable 'missing'"):
        SCM(Equation(observed="y", mu=predictor)).build_model(data)


def test_parameters_need_a_model_name() -> None:
    data = xr.Dataset({"y": ("geo", [1.0])}, coords={"geo": ["a"]})
    equation = Equation(observed="y", mu=Parameter(None, Prior("Normal")))
    with pytest.raises(ValueError, match="must have names"):
        SCM(equation).build_model(data)


@pytest.mark.parametrize("conflict", ["name", "coordinates"])
def test_build_context_respects_existing_model_metadata(conflict: str) -> None:
    data = xr.Dataset({"y": ("geo", [1.0, 2.0])}, coords={"geo": ["a", "b"]})
    labels = ["b", "a"] if conflict == "coordinates" else ["a", "b"]
    equation = Equation(observed="y", mu=0.0)

    with pm.Model(coords={"geo": labels}) as model:
        if conflict == "name":
            pm.Normal("y")
        variables_before = tuple(model.named_vars)
        with pytest.raises(ValueError):
            BuildContext(data).build(equation)

    assert tuple(model.named_vars) == variables_before
    assert model.coords["geo"] == tuple(labels)


def test_build_context_cannot_register_inputs_in_another_model() -> None:
    data = xr.Dataset({"x": ("geo", [1.0, 2.0])}, coords={"geo": ["a", "b"]})
    with pm.Model() as owner:
        context = BuildContext(data)
    with pm.Model() as other:
        with pytest.raises(RuntimeError, match="model where it was created"):
            context.build(Data("x"))
    assert not owner.named_vars
    assert not other.named_vars

    with owner:
        context.build(Data("x"))
    assert_allclose(owner["x"].get_value(), data["x"].values)


def test_exact_integer_input_boundaries_preserve_units_and_source_dtype() -> None:
    values = np.asarray([-(2**53), 2**53], dtype="int64")
    data = xr.Dataset({"x": ("geo", values)}, coords={"geo": ["a", "b"]})
    before = data.copy(deep=True)

    with pm.Model() as model:
        BuildContext(data).build(Data("x"))

    registered = model["x"].get_value()
    assert registered.dtype == np.dtype("float64")
    np.testing.assert_array_equal(registered, values)
    xr.testing.assert_identical(data, before)
