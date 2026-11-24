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
"""Tests for the Observed equation constructor."""

import pytensor.xtensor as ptx
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.terms import (
    Data,
    Equation,
    Observed,
    Parameter,
    Transform,
)


@pytest.fixture
def funnel_dataset() -> xr.Dataset:
    entered = [[1500.0, 1500.0], [520.0, 600.0]]
    converted = [[520.0, 600.0], [130.0, 190.0]]
    return xr.Dataset(
        {
            "entered": (("step", "variant"), entered),
            "converted": (("step", "variant"), converted),
        },
        coords={"step": ["signup", "paid"], "variant": ["control", "treatment"]},
    )


def step_probability() -> Transform:
    return Transform(
        Parameter(
            "step_logit",
            Prior("Normal", mu=0, sigma=1.5, dims=("step", "variant")),
        ),
        ptx.math.sigmoid,
    )


class TestObservedFactory:
    def test_returns_equation(self) -> None:
        eq = Observed(
            "Binomial", observed="converted", n="entered", p=step_probability()
        )
        assert type(eq) is Equation

    def test_string_parameters_become_data_references(self) -> None:
        eq = Observed(
            "Binomial", observed="converted", n="entered", p=step_probability()
        )
        n_term = eq.parameters["n"]
        assert isinstance(n_term, Data)
        assert n_term.var_name == "entered"

    def test_non_string_parameters_pass_through(self) -> None:
        p = step_probability()
        eq = Observed("Binomial", observed="converted", n="entered", p=p)
        assert eq.parameters["p"] is p
        assert eq.observed == "converted"
        assert eq.likelihood.distribution == "Binomial"

    def test_works_for_other_families(self) -> None:
        rate = Parameter("lam", Prior("Normal", dims=("step", "variant")))
        eq = Observed("Poisson", observed="sales", mu=rate)
        assert eq.likelihood.distribution == "Poisson"
        assert "mu" in eq.parameters

    def test_beta_binomial_via_ordinary_kwargs(self) -> None:
        kappa = Parameter("kappa", Prior("Gamma", alpha=2, beta=0.1, dims="step"))
        p = step_probability()
        eq = Observed(
            "BetaBinomial",
            observed="converted",
            n="entered",
            alpha=kappa * p,
            beta=kappa * (1 - p),
        )
        assert set(eq.parameters) == {"n", "alpha", "beta"}

    def test_prior_recipe_for_hyperpriors(self) -> None:
        likelihood = Prior("Binomial")
        eq = Observed(
            likelihood, observed="converted", n="entered", p=step_probability()
        )
        assert eq.likelihood is likelihood

    def test_unknown_parameter_raises(self) -> None:
        with pytest.raises(ValueError, match="has no parameters \\['mu'\\]"):
            Observed("Binomial", observed="converted", mu=1.0)

    def test_unknown_family_raises(self) -> None:
        with pytest.raises(Exception, match="Nope"):
            Observed("Nope", observed="converted", n="entered", p=step_probability())

    @pytest.mark.parametrize("observed", ["", None, 5])
    def test_observed_must_be_a_name(self, observed) -> None:
        with pytest.raises(TypeError, match="observed must be"):
            Observed("Binomial", observed=observed, n="entered", p=step_probability())

    def test_dist_type_error(self) -> None:
        with pytest.raises(TypeError, match="family name or a Prior"):
            Observed(5, observed="converted", n="entered", p=step_probability())


class TestObservedFit:
    def test_funnel_fit_and_predict(self, funnel_dataset) -> None:
        from pymc_marketing.experimental.terms import SCM

        equation = Observed(
            "Binomial",
            observed="converted",
            n="entered",
            p=step_probability().named("p"),
        )
        scm = SCM(equation)
        scm.fit(funnel_dataset, draws=50, tune=50, chains=1, random_seed=0)
        posterior = scm.idata["posterior"]
        assert "p" in posterior.data_vars
        assert "step_logit" in posterior.data_vars
        assert set(posterior["p"].dims) == {"chain", "draw", "step", "variant"}

        future = funnel_dataset.copy()
        future["entered"] = xr.full_like(funnel_dataset["entered"], 20000.0)
        future["converted"] = xr.zeros_like(funnel_dataset["converted"])
        predictive = scm.sample_posterior_predictive(future)
        assert "converted" in predictive.data_vars
