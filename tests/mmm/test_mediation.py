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
"""Tests for what an effect declares about mediation, and what the graph says back.

Nothing here fits a model.  The declarations are plain pydantic models, and the
registry and the role checks read a built graph and compile nothing.
"""

import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import pytest
from pydantic import ValidationError

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.additive_effect import (
    IncrementalitySpec,
    MediatorSpec,
    MuEffect,
)
from pymc_marketing.mmm.mediation import (
    PathPlan,
    ResolvedMediator,
    resolve_mediators,
    resolve_path_plan,
)
from pymc_marketing.mmm.spend_reach import resolve_channel_dependent_effects


class TestMediatorSpec:
    """What a declared node may be called, and what its reporting fields need."""

    @pytest.mark.parametrize(
        "name",
        [
            "total",
            "direct",
            "mediated",
            "spend_change",
            "induced_spend",
            "complete",
            "chain",
            "draw",
            "date",
            "channel",
        ],
    )
    def test_a_name_a_result_already_uses_is_refused(self, name):
        """A mediator's name becomes a variable of a ``Dataset`` next to these."""
        with pytest.raises(ValidationError, match="reserved"):
            MediatorSpec(name=name, variable="funnel_hub")

    @pytest.mark.parametrize("name", ["", "lf spend", "2nd_hub", "hub-a", "hub.a"])
    def test_a_name_that_is_not_an_identifier_is_refused(self, name):
        """The name is what a user types after ``result.``."""
        with pytest.raises(ValidationError, match="name"):
            MediatorSpec(name=name, variable="funnel_hub")

    @pytest.mark.parametrize(
        "fields, missing",
        [
            pytest.param({"is_spend": True}, "units", id="spend without a currency"),
            pytest.param({"scale": 2.5}, "units", id="rescaled without units"),
        ],
    )
    def test_a_reported_quantity_names_its_units(self, fields, missing):
        """Money and a rescaled movement both say what they are counted in."""
        with pytest.raises(ValidationError, match=missing):
            MediatorSpec(name="lf_spend", variable="funnel_spend", **fields)

    @pytest.mark.parametrize(
        "scale", [0.0, -1.0, float("inf"), float("nan"), True, "2"]
    )
    def test_a_scale_is_a_positive_finite_number(self, scale):
        """A scale multiplies a movement, so zero, a sign and ``inf`` are mistakes."""
        with pytest.raises(ValidationError, match="scale"):
            MediatorSpec(
                name="lf_spend", variable="funnel_spend", units="USD", scale=scale
            )

    def test_the_variable_is_stripped_and_not_empty(self):
        """The variable is looked up in the model by this exact text."""
        spec = MediatorSpec(name="hub", variable=" funnel_hub ")

        assert spec.variable == "funnel_hub"
        with pytest.raises(ValidationError, match="variable"):
            MediatorSpec(name="hub", variable="  ")

    def test_a_spec_is_immutable_and_closed(self):
        """A declaration cannot be edited after it is read, nor carry a typo."""
        spec = MediatorSpec(name="hub", variable="funnel_hub")

        with pytest.raises(ValidationError):
            spec.variable = "other"
        with pytest.raises(ValidationError, match="unit"):
            MediatorSpec(name="hub", variable="funnel_hub", unit="USD")


class TestRoleAndMediators:
    """The two fields a path request reads off an ``IncrementalitySpec``."""

    def test_an_empty_spec_still_opts_in_and_claims_nothing(self):
        """Opting in to totals keeps costing one line."""
        spec = IncrementalitySpec()

        assert spec.role is None
        assert spec.mediators == ()

    def test_a_mediator_role_names_at_least_one_node(self):
        """The claim is that spend reaches the term only through declared nodes."""
        with pytest.raises(ValidationError, match="at least one"):
            IncrementalitySpec(role="mediator")

    def test_a_direct_role_needs_no_node_and_may_declare_one(self):
        """A direct effect may still report a node it measures."""
        indicator = MediatorSpec(name="searches", variable="halo_searches")

        assert IncrementalitySpec(role="direct").mediators == ()
        assert IncrementalitySpec(role="direct", mediators=(indicator,)).mediators == (
            indicator,
        )

    def test_an_unknown_role_is_refused(self):
        """``"mixed"`` belongs to a later stage, and a typo to none."""
        with pytest.raises(ValidationError, match="role"):
            IncrementalitySpec(role="mixed")

    @pytest.mark.parametrize(
        "second, complaint",
        [
            pytest.param(
                {"name": "hub", "variable": "funnel_spend"},
                "distinct names",
                id="one name for two variables",
            ),
            pytest.param(
                {"name": "spend", "variable": "funnel_hub"},
                "distinct variables",
                id="one variable under two names",
            ),
            pytest.param(
                {"name": "funnel_hub", "variable": "funnel_spend"},
                "another mediator's variable",
                id="a name that is another node's variable",
            ),
        ],
    )
    def test_names_and_variables_do_not_collide(self, second, complaint):
        """A result is keyed by name and the graph by variable; neither may repeat."""
        first = MediatorSpec(name="hub", variable="funnel_hub")

        with pytest.raises(ValidationError, match=complaint):
            IncrementalitySpec(
                role="mediator", mediators=(first, MediatorSpec(**second))
            )

    def test_the_order_of_declaration_carries_no_meaning(self):
        """Two specs that declare the same nodes are equal, whatever their order."""
        hub = MediatorSpec(name="hub", variable="funnel_hub")
        spend = MediatorSpec(
            name="lf_spend", variable="funnel_spend", units="USD", is_spend=True
        )

        forward = IncrementalitySpec(role="mediator", mediators=(hub, spend))
        backward = IncrementalitySpec(role="mediator", mediators=(spend, hub))

        assert forward == backward
        assert [mediator.name for mediator in backward.mediators] == ["hub", "lf_spend"]

    def test_a_spec_survives_json(self):
        """A declaration is what a YAML config or a saved effect would hold."""
        spec = IncrementalitySpec(
            additional_carryover_lags=4,
            role="mediator",
            mediators=(
                MediatorSpec(name="hub", variable="funnel_hub"),
                MediatorSpec(
                    name="sem",
                    variable="funnel_mediator",
                    units="impressions",
                    scale=1250.0,
                ),
            ),
        )

        assert IncrementalitySpec.model_validate_json(spec.model_dump_json()) == spec
        assert IncrementalitySpec.model_validate(spec.model_dump(mode="json")) == spec


class _KeepsItsMediators(MuEffect):
    """An effect that stores its declaration as a field, with no ``to_dict`` of its own."""

    prefix: str
    mediators: tuple[MediatorSpec, ...] = ()

    def create_data(self, mmm) -> None:
        """No data of its own."""

    def create_effect(self, mmm):
        """Never built here."""
        raise NotImplementedError

    def set_data(self, mmm, model, X) -> None:
        """No data of its own."""

    def incrementality_spec(self) -> IncrementalitySpec:
        """Hand the stored declaration over."""
        return IncrementalitySpec(role="mediator", mediators=self.mediators)


def test_a_declaration_kept_as_a_field_is_saved_with_the_effect():
    """The default serialization writes the nodes and reads them back.

    The recommended home of a declaration is the method, built from fields the
    effect already saves.  An effect may keep it as a field instead, and then
    the plain-dict form a YAML config supplies has to validate too.
    """
    effect = _KeepsItsMediators(
        prefix="funnel",
        mediators=(
            MediatorSpec(
                name="lf_spend", variable="funnel_spend", units="USD", is_spend=True
            ),
        ),
    )

    restored = _KeepsItsMediators.model_validate(effect.model_dump(mode="json"))

    assert restored.incrementality_spec() == effect.incrementality_spec()


# ============================================================================
# The registry and the role checks, on built and unfitted models
# ============================================================================


class GraphShape(MuEffect):
    """One way of wiring spend into a term of the linear predictor.

    The declaration is a field, handed over as it is, so that a test states the
    graph (``kind``) and the claim about it (``spec``) side by side.
    """

    prefix: str
    kind: str
    spec: IncrementalitySpec | None = None
    reads: str | None = None
    frozen: tuple[str, ...] = ()

    def create_data(self, mmm) -> None:
        """No data of its own."""

    def set_data(self, mmm, model, X) -> None:
        """No data of its own."""

    def incrementality_spec(self) -> IncrementalitySpec | None:
        """Hand the stored declaration over."""
        return self.spec

    @property
    def frozen_deterministics(self) -> list[str]:
        """Whatever the test asked to be frozen."""
        return list(self.frozen)

    def create_effect(self, mmm):
        """Build the term the way ``kind`` says."""
        model = mmm.model
        prefix = self.prefix
        spend = mmm.channel_data_scaled.sum(dim="channel")
        beta = pmd.HalfNormal(f"{prefix}_beta", sigma=1.0)

        def term(value):
            return pmd.Deterministic(f"{prefix}_effect_contribution", value)

        def hub(value):
            return pmd.Deterministic(f"{prefix}_hub", value)

        match self.kind:
            case "direct":
                return term(beta * spend**2)
            case "hub":
                return term(beta * hub(pmd.math.tanh(spend)))
            case "chain":
                induced = pmd.Deterministic(
                    f"{prefix}_spend", hub(pmd.math.tanh(spend)) + 0.5
                )
                return term(beta * induced)
            case "two_route":
                return term(beta * hub(pmd.math.tanh(spend)) + beta * spend)
            case "under_hub":
                expression = pmd.math.tanh(spend)
                hub(expression)
                return term(beta * expression)
            case "reads":
                return term(beta * model[self.reads])
            case "hub_plus":
                return term(beta * hub(pmd.math.tanh(spend)) + model[self.reads])
            case "indicator":
                pmd.Deterministic(f"{prefix}_searches", pmd.math.tanh(spend))
                return term(beta * spend)
            case "indicator_under":
                expression = pmd.math.tanh(spend)
                pmd.Deterministic(f"{prefix}_searches", expression)
                return term(beta * expression)
            case "classic_hub":
                classic = pm.Deterministic(
                    f"{prefix}_hub", pt.tanh(spend.values), dims=spend.dims
                )
                return term(beta * ptx.as_xtensor(classic, dims=spend.dims))
            case "random_hub":
                return term(beta * pmd.Normal(f"{prefix}_hub", mu=spend, sigma=0.1))
            case "noisy_hub":
                latent = pmd.Normal(f"{prefix}_latent", mu=spend, sigma=0.1)
                return term(beta * hub(pmd.math.tanh(latent)))
            case "level":
                pmd.Deterministic(f"{prefix}_level", 2.0 * beta)
                return term(beta * hub(pmd.math.tanh(spend)))
            case "date_mean_hub":
                return term(beta * spend / hub(spend.mean(dim="date")))
            case "channel_hub":
                per_channel = hub(pmd.math.tanh(mmm.channel_data_scaled))
                return term(beta * per_channel.sum(dim="channel"))
            case "segment_hub":
                model.add_coords({"segment": ["new", "returning"]})
                weight = pmd.HalfNormal(f"{prefix}_weight", dims="segment")
                segmented = hub(pmd.math.tanh(spend) * weight)
                return term(beta * segmented.sum(dim="segment"))
        raise ValueError(self.kind)


def mediator(prefix: str, *nodes: str, **reporting) -> IncrementalitySpec:
    """A ``"mediator"`` declaration naming ``{prefix}_{node}`` for each node."""
    return IncrementalitySpec(
        role="mediator",
        mediators=tuple(
            MediatorSpec(
                name=f"{prefix}_{node}", variable=f"{prefix}_{node}", **reporting
            )
            for node in nodes
        ),
    )


DIRECT = IncrementalitySpec(role="direct")


def build(data, *effects, **kwargs) -> MMM:
    """An MMM on the funnel frame carrying *effects*, built and not fitted."""
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="y",
        adstock=GeometricAdstock(l_max=3),
        saturation=LogisticSaturation(),
        **kwargs,
    )
    for effect in effects:
        mmm.add_mu_effect(effect)
    mmm.build_model(data["X_df"], data["y_series"])
    return mmm


def registry_of(mmm: MMM) -> dict[str, ResolvedMediator]:
    """The registry of *mmm*, keyed by declared name."""
    effects = resolve_channel_dependent_effects(mmm)
    return {entry.name: entry for entry in resolve_mediators(mmm, effects)}


class TestResolveMediators:
    """What the registry says about each declared node."""

    def test_a_node_between_spend_and_its_term_is_usable_and_readable(
        self, funnel_mmm_data
    ):
        """The plain case: every field of the entry, for a hub and what it induces."""
        spec = IncrementalitySpec(
            role="mediator",
            mediators=(
                MediatorSpec(name="demand", variable="f_hub"),
                MediatorSpec(
                    name="lf_spend",
                    variable="f_spend",
                    units="USD",
                    scale=250.0,
                    is_spend=True,
                ),
            ),
        )
        mmm = build(funnel_mmm_data, GraphShape(prefix="f", kind="chain", spec=spec))

        registry = registry_of(mmm)

        assert registry["demand"] == ResolvedMediator(
            name="demand",
            variable="f_hub",
            effects=("GraphShape",),
            units=None,
            scale=1.0,
            is_spend=False,
            feeds_term=True,
            problem=None,
            unreadable=None,
        )
        assert registry["lf_spend"].readable
        assert (
            registry["lf_spend"].units,
            registry["lf_spend"].scale,
            registry["lf_spend"].is_spend,
        ) == ("USD", 250.0, True)

    def test_a_measurement_only_node_feeds_no_term(self, funnel_mmm_data):
        """An indicator of spend is reportable, and its return is in no numerator."""
        spec = IncrementalitySpec(
            role="direct",
            mediators=(MediatorSpec(name="searches", variable="f_searches"),),
        )
        mmm = build(
            funnel_mmm_data, GraphShape(prefix="f", kind="indicator", spec=spec)
        )

        entry = registry_of(mmm)["searches"]

        assert entry.readable
        assert not entry.feeds_term

    @pytest.mark.parametrize(
        "kind, variable, problem, options",
        [
            pytest.param(
                "hub", "f_nope", "not a variable of the model", {}, id="unknown"
            ),
            pytest.param(
                "random_hub",
                "f_hub",
                "not a registered Deterministic",
                {},
                id="a random variable",
            ),
            pytest.param(
                "hub", "f_hub", "frozen", {"frozen": ("f_hub",)}, id="a frozen node"
            ),
            pytest.param(
                "level",
                "f_level",
                "spend does not reach",
                {},
                id="spend does not reach it",
            ),
            pytest.param(
                "hub",
                "channel_contribution",
                "on the direct path",
                {},
                id="the direct path",
            ),
            pytest.param(
                "hub",
                "f_effect_contribution",
                "term an effect adds",
                {},
                id="an effect's own term",
            ),
            pytest.param(
                "hub", "mu", "linear predictor", {}, id="the linear predictor"
            ),
            pytest.param(
                "noisy_hub",
                "f_hub",
                "'f_latent'",
                {},
                id="behind a random variable",
            ),
        ],
    )
    def test_a_node_that_cannot_mark_a_cut_says_why(
        self, funnel_mmm_data, kind, variable, problem, options
    ):
        """A fault in a declaration is a field of its entry, not an exception."""
        spec = IncrementalitySpec(
            role="mediator", mediators=(MediatorSpec(name="node", variable=variable),)
        )
        mmm = build(
            funnel_mmm_data, GraphShape(prefix="f", kind=kind, spec=spec, **options)
        )

        entry = registry_of(mmm)["node"]

        assert problem in entry.problem
        assert not entry.readable

    @pytest.mark.parametrize(
        "kind, name, unreadable",
        [
            pytest.param("date_mean_hub", "node", "no 'date' dimension", id="no date"),
            pytest.param(
                "channel_hub", "node", "'channel' dimension", id="per channel"
            ),
            pytest.param(
                "segment_hub", "node", "('segment',)", id="a dimension of its own"
            ),
            pytest.param(
                "hub",
                "y",
                "another variable of the model",
                id="a model variable's name",
            ),
        ],
    )
    def test_a_node_that_cannot_be_reported_says_why(
        self, funnel_mmm_data, kind, name, unreadable
    ):
        """Such a node may still mark where an effect's routes are cut."""
        spec = IncrementalitySpec(
            role="mediator", mediators=(MediatorSpec(name=name, variable="f_hub"),)
        )
        mmm = build(funnel_mmm_data, GraphShape(prefix="f", kind=kind, spec=spec))

        entry = registry_of(mmm)[name]

        assert entry.problem is None
        assert unreadable in entry.unreadable
        assert not entry.readable

    def test_a_name_that_is_a_dimension_of_the_model_cannot_be_reported(
        self, panel_mmm_data
    ):
        """A result would carry the name twice, as a variable and as a dimension."""
        spec = IncrementalitySpec(
            role="mediator", mediators=(MediatorSpec(name="country", variable="f_hub"),)
        )
        mmm = MMM(
            channel_columns=["channel_1", "channel_2"],
            date_column="date",
            target_column="target",
            dims=("country",),
            adstock=GeometricAdstock(l_max=3),
            saturation=LogisticSaturation(),
        )
        mmm.add_mu_effect(GraphShape(prefix="f", kind="hub", spec=spec))
        mmm.build_model(panel_mmm_data["X"], panel_mmm_data["y"])

        assert "dimension of the model" in registry_of(mmm)["country"].unreadable

    @pytest.mark.parametrize(
        "second, unreadable",
        [
            pytest.param(
                MediatorSpec(name="hub", variable="g_hub"),
                "declared for",
                id="one name for two variables",
            ),
            pytest.param(
                MediatorSpec(name="demand", variable="f_hub"),
                "under the names",
                id="one variable under two names",
            ),
            pytest.param(
                MediatorSpec(name="hub", variable="f_hub", units="EUR", scale=2.0),
                "reporting fields",
                id="two sets of reporting fields",
            ),
        ],
    )
    def test_two_effects_that_disagree_about_a_node_make_it_unreadable(
        self, funnel_mmm_data, second, unreadable
    ):
        """The node can still be held; it cannot be reported under one name."""
        first = MediatorSpec(name="hub", variable="f_hub", units="USD", scale=3.0)
        mmm = build(
            funnel_mmm_data,
            GraphShape(
                prefix="f",
                kind="hub",
                spec=IncrementalitySpec(role="mediator", mediators=(first,)),
            ),
            GraphShape(
                prefix="g",
                kind="hub_plus",
                reads="f_hub",
                spec=IncrementalitySpec(role="mediator", mediators=(second,)),
            ),
        )

        registry = registry_of(mmm)

        assert all(unreadable in entry.unreadable for entry in registry.values())
        assert all(entry.problem is None for entry in registry.values())

    def test_a_hub_two_effects_read_is_one_entry(self, funnel_mmm_data):
        """One of them says how to report it and the other leaves the defaults."""
        reported = MediatorSpec(
            name="hub", variable="f_hub", units="USD", is_spend=True
        )
        mmm = build(
            funnel_mmm_data,
            GraphShape(
                prefix="f",
                kind="hub",
                spec=IncrementalitySpec(role="mediator", mediators=(reported,)),
            ),
            GraphShape(
                prefix="g",
                kind="reads",
                reads="f_hub",
                spec=IncrementalitySpec(
                    role="mediator",
                    mediators=(MediatorSpec(name="hub", variable="f_hub"),),
                ),
            ),
        )

        (entry,) = registry_of(mmm).values()

        assert entry.effects == ("GraphShape", "GraphShape")
        assert (entry.units, entry.is_spend, entry.readable) == ("USD", True, True)

    def test_an_effect_that_does_not_read_spend_may_declare_a_node(
        self, funnel_mmm_data
    ):
        """Its spec is read by the registry, which a total never does."""
        declaring = GraphShape(
            prefix="f",
            kind="level",
            spec=IncrementalitySpec(
                mediators=(MediatorSpec(name="hub", variable="f_hub"),)
            ),
        )
        mmm = build(funnel_mmm_data, declaring)

        assert registry_of(mmm)["hub"].readable

    def test_a_spec_that_was_built_around_validation_is_refused(self, funnel_mmm_data):
        """``model_construct`` skips the rules, so the registry applies them again."""

        class Forging(GraphShape):
            def incrementality_spec(self) -> IncrementalitySpec:
                return IncrementalitySpec.model_construct(
                    additional_carryover_lags=None,
                    evaluation_mode="auto",
                    role="mediator",
                    mediators=(),
                )

        mmm = build(funnel_mmm_data, Forging(prefix="f", kind="hub"))

        with pytest.raises(ValueError, match=r"(?s)Forging.*at least one"):
            registry_of(mmm)


def plan_of(mmm: MMM) -> PathPlan:
    """The path plan of *mmm*, through every check a path request runs."""
    effects = resolve_channel_dependent_effects(mmm)
    return resolve_path_plan(
        model=mmm, effects=effects, registry=resolve_mediators(mmm, effects)
    )


class TestRoleChecks:
    """Where the graph agrees with the declared roles, and where it contradicts them.

    A ``"mediator"`` term is dropped from a direct effect, which is the same as
    holding its nodes only if spend reaches the term through them alone and no
    kept term reads what lies behind them.
    """

    @pytest.mark.parametrize(
        "effects, held",
        [
            pytest.param(
                [GraphShape(prefix="f", kind="hub", spec=mediator("f", "hub"))],
                ("f_effect_contribution",),
                id="a mediator through its hub",
            ),
            pytest.param(
                [GraphShape(prefix="f", kind="chain", spec=mediator("f", "hub"))],
                ("f_effect_contribution",),
                id="the first of two nodes in a chain",
            ),
            pytest.param(
                [GraphShape(prefix="f", kind="chain", spec=mediator("f", "spend"))],
                ("f_effect_contribution",),
                id="the second of two nodes in a chain",
            ),
            pytest.param(
                [GraphShape(prefix="f", kind="classic_hub", spec=mediator("f", "hub"))],
                ("f_effect_contribution",),
                id="a hub registered with classic pm.Deterministic",
            ),
            pytest.param(
                [
                    GraphShape(prefix="g", kind="hub", spec=mediator("g", "hub")),
                    GraphShape(prefix="f", kind="hub", spec=mediator("f", "hub")),
                ],
                ("f_effect_contribution", "g_effect_contribution"),
                id="two mediator effects",
            ),
            pytest.param(
                [
                    GraphShape(prefix="f", kind="hub", spec=mediator("f", "hub")),
                    GraphShape(
                        prefix="g",
                        kind="reads",
                        reads="f_hub",
                        spec=mediator("f", "hub"),
                    ),
                ],
                ("f_effect_contribution", "g_effect_contribution"),
                id="two effects on one hub",
            ),
            pytest.param(
                [
                    GraphShape(prefix="f", kind="hub", spec=mediator("f", "hub")),
                    GraphShape(
                        prefix="g",
                        kind="hub_plus",
                        reads="f_effect_contribution",
                        spec=mediator("g", "hub"),
                    ),
                ],
                ("f_effect_contribution", "g_effect_contribution"),
                id="a mediator term that adds another mediator's term",
            ),
            pytest.param(
                [
                    GraphShape(prefix="f", kind="hub", spec=mediator("f", "hub")),
                    GraphShape(prefix="d", kind="direct", spec=DIRECT),
                ],
                ("f_effect_contribution",),
                id="a direct effect next to a mediator",
            ),
            pytest.param(
                [GraphShape(prefix="d", kind="direct", spec=DIRECT)],
                (),
                id="a direct effect alone",
            ),
            pytest.param(
                [
                    GraphShape(
                        prefix="d",
                        kind="indicator",
                        spec=IncrementalitySpec(
                            role="direct",
                            mediators=(
                                MediatorSpec(name="searches", variable="d_searches"),
                            ),
                        ),
                    )
                ],
                (),
                id="a direct effect that measures an indicator",
            ),
            pytest.param([], (), id="no effect at all"),
        ],
    )
    def test_a_coherent_model_gets_its_plan(self, funnel_mmm_data, effects, held):
        """The plan holds the term of every ``"mediator"`` effect, sorted."""
        assert plan_of(build(funnel_mmm_data, *effects)) == PathPlan(held_terms=held)

    @pytest.mark.parametrize(
        "effects, complaint",
        [
            pytest.param(
                [GraphShape(prefix="f", kind="two_route", spec=mediator("f", "hub"))],
                "'f_effect_contribution'.*past its declared nodes.*'f_hub'",
                id="a two-route effect declared as a mediator",
            ),
            pytest.param(
                [GraphShape(prefix="f", kind="under_hub", spec=mediator("f", "hub"))],
                "'f_effect_contribution'.*expression under.*'f_hub'",
                id="a term built from the expression under its node",
            ),
            pytest.param(
                [
                    GraphShape(
                        prefix="f",
                        kind="hub",
                        spec=IncrementalitySpec(
                            role="mediator",
                            mediators=(
                                MediatorSpec(
                                    name="base", variable="channel_contribution"
                                ),
                            ),
                        ),
                    )
                ],
                "'f_effect_contribution'.*none of its declared nodes.*direct path",
                id="a mediator whose only node is on the direct path",
            ),
            pytest.param(
                [
                    GraphShape(prefix="f", kind="hub", spec=mediator("f", "hub")),
                    GraphShape(prefix="g", kind="hub", spec=mediator("f", "hub")),
                ],
                "'g_effect_contribution'.*none of its declared nodes",
                id="a mediator that declares another effect's node",
            ),
            pytest.param(
                [
                    GraphShape(prefix="f", kind="hub", spec=mediator("f", "hub")),
                    GraphShape(prefix="d", kind="reads", reads="f_hub", spec=DIRECT),
                ],
                "'d_effect_contribution'.*reads the declared node 'f_hub'",
                id="a direct effect that reads a declared node",
            ),
            pytest.param(
                [
                    GraphShape(prefix="f", kind="hub", spec=mediator("f", "hub")),
                    GraphShape(
                        prefix="d",
                        kind="reads",
                        reads="f_effect_contribution",
                        spec=DIRECT,
                    ),
                ],
                "'d_effect_contribution'.*'f_effect_contribution'",
                id="a direct effect that reads a mediator's term",
            ),
            pytest.param(
                [
                    GraphShape(prefix="d", kind="direct", spec=DIRECT),
                    GraphShape(
                        prefix="g",
                        kind="hub_plus",
                        reads="d_effect_contribution",
                        spec=mediator("g", "hub"),
                    ),
                ],
                "'g_effect_contribution'.*past its declared nodes",
                id="a mediator term that adds a direct term",
            ),
            pytest.param(
                [
                    GraphShape(
                        prefix="d",
                        kind="indicator_under",
                        spec=IncrementalitySpec(
                            role="direct",
                            mediators=(
                                MediatorSpec(name="searches", variable="d_searches"),
                            ),
                        ),
                    )
                ],
                "'d_effect_contribution'.*expression under the declared node 'd_searches'",
                id="a direct term built from the expression under a node",
            ),
        ],
    )
    def test_a_contradicted_role_is_refused_by_name(
        self, funnel_mmm_data, effects, complaint
    ):
        """The message names the term and the variable the contradiction is about."""
        mmm = build(funnel_mmm_data, *effects)

        with pytest.raises(ValueError, match=complaint):
            plan_of(mmm)

    def test_an_effect_with_no_role_is_refused_by_name(self, funnel_mmm_data):
        """Opting in to totals says nothing about which path a term is on."""
        mmm = build(
            funnel_mmm_data,
            GraphShape(prefix="f", kind="hub", spec=IncrementalitySpec()),
        )

        with pytest.raises(
            NotImplementedError, match=r"GraphShape.*'f_effect_contribution'"
        ):
            plan_of(mmm)
