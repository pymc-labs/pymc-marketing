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
r"""Mediators of a spend counterfactual: what effects declare, and what the graph answers.

A ``mu_effect`` that reads spend can stand for two different things.  It can be
one more direct function of spend, such as a synergy between channels.  Or it
can be a mediated path: spend moves a node the effect models (latent demand,
lower-funnel spend), and that node moves the target.
:class:`~pymc_marketing.mmm.incrementality.Incrementality` can separate the two
only if each effect says which it is, through the ``role`` and ``mediators`` of
its :class:`~pymc_marketing.mmm.additive_effect.IncrementalitySpec`.

This module turns those declarations into what a request needs, reading the
graph and compiling nothing:

* :func:`resolve_mediators` lists every declared node with what is wrong with
  it, if anything.
* :func:`check_roles` refuses a model whose graph contradicts the declared
  roles.
* :func:`resolve_path_plan` returns the terms a direct effect holds fixed.
* :func:`select_mediators` picks the nodes a request reads, and refuses the
  ones it cannot.

**Why dropping a term is holding a mediator.**  A direct effect asks for the
response with spend moved and every mediator kept at the value it has under
actual spend.  The increment is assembled from the moves of
``channel_contribution`` and of each effect's term, so the cheap way to hold the
mediators is to leave the mediator effects' terms out.  That is the same
intervention only if two things are true, and the role checks are those two
things.  No term that stays reads anything behind a mediator's nodes, so the
kept terms move as they would with the nodes held.  And spend reaches each
mediator term only through its nodes and through other mediator terms, so with
every node held every mediator term stays where it was.

The roles themselves are the author's causal claim.  The graph cannot prove that
a node is a mediator and not one more transform of spend.  The checks see shared
variables only: a value recomputed from spend in two places is two variables to
the graph, and they cannot connect them.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError
from pytensor.compile.ops import ViewOp
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors, toposort
from pytensor.xtensor.basic import TensorFromXTensor, XTensorFromTensor
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.additive_effect import IncrementalitySpec, MediatorSpec
from pymc_marketing.mmm.spend_reach import (
    _PKG_PREFIX,
    CHANNEL_CONTRIBUTION,
    ChannelDependentEffect,
    _find_linear_predictor,
)
from pymc_marketing.names.mmm import Dim, Var
from pymc_marketing.pytensor_utils import blocked_between

if TYPE_CHECKING:
    from pymc import Model

    from pymc_marketing.mmm.mmm import MMM

__all__ = [
    "PathPlan",
    "ResolvedMediator",
    "check_roles",
    "resolve_mediators",
    "resolve_path_plan",
    "select_mediators",
]

_VALUE_KEEPING_OPS = (ViewOp, TensorFromXTensor, XTensorFromTensor)
"""Ops whose output is their input: a registration, or a change of tensor kind."""

_Reporting = tuple[str | None, float, bool]
"""``(units, scale, is_spend)``: how a node's movement is reported."""

_DEFAULT_REPORTING: _Reporting = (None, 1.0, False)
"""How a node nobody said how to report is reported."""

InducedInto = Literal["ratio", "decomposition"]
"""Where the movement of spend mediators goes: a ratio's denominator, or a decomposition."""


@dataclass(frozen=True)
class ResolvedMediator:
    """A declared mediator node, and whether it can be used.

    Parameters
    ----------
    name : str
        The declared name, which results are labelled with.
    variable : str
        The model variable it names.
    effects : tuple of str
        Class names of the effects that declare it, in ``mu_effects`` order.
    units : str or None
        What a movement is reported in.
    scale : float
        Reported units per model unit.
    is_spend : bool
        Whether the reported movement is money in the currency of channel spend.
    feeds_term : bool
        Whether a term of the linear predictor reads the node without passing
        through another term.  A node that only has a likelihood on it does not.
    full_axis : bool
        Whether an effect that declares it asks for every period to be
        evaluated on the full date axis (``evaluation_mode="full"``).  Honoured
        by a request that reads the node, also when the effect's own term does
        not read spend.
    declared_carryover_lags : int or None
        The widest ``additional_carryover_lags`` among the effects that declare
        it, also one whose own term does not read spend.  ``None`` when none of
        them declares one.  Where no probe can be placed, it is what bounds the
        node's carryover.
    dims : tuple of str
        The model's dimensions besides ``date`` that the node carries, in the
        model's order.  Empty for a variable the model does not have.
    problem : str or None
        Why the node cannot mark where an effect's routes from spend are cut.
    unreadable : str or None
        Why its movement cannot be reported under its name.
    """

    name: str
    variable: str
    effects: tuple[str, ...]
    units: str | None
    scale: float
    is_spend: bool
    feeds_term: bool
    full_axis: bool
    declared_carryover_lags: int | None
    dims: tuple[str, ...]
    problem: str | None
    unreadable: str | None

    @property
    def readable(self) -> bool:
        """Whether the node's movement can be reported."""
        return self.problem is None and self.unreadable is None


@dataclass(frozen=True)
class PathPlan:
    """What a direct effect holds fixed.

    Parameters
    ----------
    held_terms : tuple of str
        The contribution variables of every ``"mediator"`` effect, sorted.
        Empty for a model with nothing to hold.
    """

    held_terms: tuple[str, ...]


def _validated_spec(spec: object, label: str) -> IncrementalitySpec:
    """Validate a spec an effect returned, whatever it was built with.

    ``model_validate`` on an instance returns it untouched, so the fields are
    dumped first and the dump is validated against the library's own class.
    That brings ``model_construct``, ``model_copy(update=...)`` and a subclass
    that relaxed a rule back under the rules.

    Raises
    ------
    ValueError
        If *spec* is not an :class:`IncrementalitySpec`, or breaks its rules.
    """
    if not isinstance(spec, IncrementalitySpec):
        raise ValueError(
            f"The mu_effect {label!r} returned a {type(spec).__name__} from "
            "'incrementality_spec'.  A request that reads roles or mediator "
            "nodes needs an IncrementalitySpec."
        )
    try:
        return IncrementalitySpec.model_validate(spec.model_dump(mode="json"))
    except ValidationError as error:
        raise ValueError(
            f"The IncrementalitySpec of the mu_effect {label!r} is not valid: {error}"
        ) from error


def _expression_under(variable: Variable) -> list[Variable]:
    """List what a registered variable is a value-keeping copy of, nearest first.

    ``pm.Deterministic`` registers a copy of the expression it is given, and the
    classic one also converts between tensor kinds.  A term built on the
    expression instead of the registered variable reads these.
    """
    found = []
    while (
        variable.owner is not None
        and len(variable.owner.inputs) == 1
        and isinstance(variable.owner.op, _VALUE_KEEPING_OPS)
    ):
        variable = variable.owner.inputs[0]
        found.append(variable)
    return found


def _terms(
    model: MMM, effects: Sequence[ChannelDependentEffect]
) -> dict[str, Variable]:
    """Return the terms spend moves the linear predictor through, by name."""
    pymc_model = model.model
    return {
        name: pymc_model[name]
        for name in (
            CHANNEL_CONTRIBUTION,
            *(effect.contribution_var for effect in effects),
        )
    }


def _inside(terms: Mapping[str, Variable], name: str) -> set[Variable]:
    """Return a term and whatever it reads without passing through another term."""
    others = [term for other, term in terms.items() if other != name]
    return set(ancestors([terms[name]], blockers=others))


# ==================== The registry of declared nodes ====================


@dataclass
class _Declaration:
    """What the effects declare about one node under one name, merged.

    Parameters
    ----------
    variable : str
        The model variable the node is.
    effects : list of str
        Class names of the declaring effects, in ``mu_effects`` order.
    reporting : tuple
        ``(units, scale, is_spend)``, as the effect that sets them says.
    full_axis : bool
        Whether a declaring effect asks for ``evaluation_mode="full"``.
    carryover_lags : int or None
        The widest ``additional_carryover_lags`` a declaring effect declares.
    fault : str or None
        Why the declarations leave the node without a way to be reported.
    """

    variable: str
    effects: list[str] = field(default_factory=list)
    reporting: _Reporting = _DEFAULT_REPORTING
    full_axis: bool = False
    carryover_lags: int | None = None
    fault: str | None = None

    def merge(
        self,
        effect: str,
        mediator: MediatorSpec,
        *,
        full_axis: bool,
        carryover_lags: int | None,
    ) -> None:
        """Add one effect's declaration of the node.

        At most one effect may say how to report the node.  Two that say it
        differently leave it without a way to be reported.

        Parameters
        ----------
        effect : str
            Class name of the declaring effect.
        mediator : MediatorSpec
            Its declaration of the node.
        full_axis : bool
            Whether that effect asks for ``evaluation_mode="full"``.
        carryover_lags : int or None
            The ``additional_carryover_lags`` that effect declares, if any.
        """
        self.effects.append(effect)
        self.full_axis |= full_axis
        if carryover_lags is not None:
            self.carryover_lags = max(carryover_lags, self.carryover_lags or 0)
        reporting = (mediator.units, mediator.scale, mediator.is_spend)
        if reporting == _DEFAULT_REPORTING:
            return
        if self.reporting not in (_DEFAULT_REPORTING, reporting):
            self.fault = (
                f"{mediator.variable!r} is declared with two sets of "
                "reporting fields (units, scale, is_spend)"
            )
        self.reporting = reporting


def _collect_declarations(model: MMM) -> dict[tuple[str, str], _Declaration]:
    """Merge what every ``mu_effect`` declares, one entry per name and variable.

    Keyed by the pair, so that a name two effects use for two variables keeps
    both: neither can be reported under it, and each still cuts its own
    effect's routes.

    Parameters
    ----------
    model : MMM
        The model whose effects are read.

    Returns
    -------
    dict
        Per ``(name, variable)``, the merged declaration, with a fault where
        another entry shares its name or its variable.

    Raises
    ------
    ValueError
        If an effect returns something that is not a valid
        :class:`~pymc_marketing.mmm.additive_effect.IncrementalitySpec`.
    """
    declared: dict[tuple[str, str], _Declaration] = {}
    for effect in model.mu_effects:
        spec = getattr(effect, "incrementality_spec", lambda: None)()
        if spec is None:
            continue
        label = type(effect).__name__
        validated = _validated_spec(spec, label)
        for mediator in validated.mediators:
            declaration = declared.setdefault(
                (mediator.name, mediator.variable),
                _Declaration(variable=mediator.variable),
            )
            declaration.merge(
                label,
                mediator,
                full_axis=validated.evaluation_mode == "full",
                carryover_lags=validated.additional_carryover_lags,
            )
    _mark_name_clashes(declared)
    return declared


def _mark_name_clashes(declared: Mapping[tuple[str, str], _Declaration]) -> None:
    """Record a fault on every entry whose name or variable another entry shares.

    Parameters
    ----------
    declared : mapping
        Per ``(name, variable)``, the merged declaration.  Updated in place.
    """
    variables_of: dict[str, list[str]] = {}
    names_of: dict[str, list[str]] = {}
    for name, variable in declared:
        variables_of.setdefault(name, []).append(variable)
        names_of.setdefault(variable, []).append(name)
    for (name, variable), declaration in declared.items():
        if len(variables_of[name]) > 1:
            listed = " and for ".join(map(repr, sorted(variables_of[name])))
            declaration.fault = f"the name {name!r} is declared for {listed}"
        elif len(names_of[variable]) > 1:
            declaration.fault = (
                f"{variable!r} is declared under the names {sorted(names_of[variable])}"
            )


@dataclass(frozen=True)
class _SpendGraph:
    """What the checks on a declared node read off the model's graph, computed once.

    Parameters
    ----------
    pymc_model : pm.Model
        The model's graph.
    channel_data : Variable
        The spend a counterfactual moves.
    terms : dict
        The terms spend moves the linear predictor through, by name.
    term_reads : tuple of set
        For each term, what it reads without passing through another term.
    direct_path : set
        ``channel_contribution`` and everything upstream of it.
    predictor : Variable or None
        The linear predictor, where the graph exposes it.
    frozen : sequence of str
        The deterministics the posterior holds at their values.
    """

    pymc_model: Model
    channel_data: Variable
    terms: dict[str, Variable]
    term_reads: tuple[set[Variable], ...]
    direct_path: set[Variable]
    predictor: Variable | None
    frozen: Sequence[str]

    @classmethod
    def of(cls, model: MMM, effects: Sequence[ChannelDependentEffect]) -> _SpendGraph:
        """Read the graph facts of *model* with its channel-dependent *effects*."""
        pymc_model = model.model
        terms = _terms(model, effects)
        return cls(
            pymc_model=pymc_model,
            channel_data=pymc_model[Var.CHANNEL_DATA],
            terms=terms,
            term_reads=tuple(_inside(terms, name) for name in terms),
            direct_path=set(ancestors([terms[CHANNEL_CONTRIBUTION]])),
            predictor=_find_linear_predictor(model),
            frozen=model.frozen_deterministics,
        )

    def feeds_term(self, variable: str) -> bool:
        """Whether a term reads *variable* without passing through another term."""
        node = self.pymc_model.named_vars.get(variable)
        return any(node in reads for reads in self.term_reads)


def _cut_problem(graph: _SpendGraph, variable: str) -> str | None:
    """Say why a node cannot mark where an effect's routes from spend are cut.

    Parameters
    ----------
    graph : _SpendGraph
        The model's graph facts.
    variable : str
        The declared node.

    Returns
    -------
    str or None
        The reason, or ``None`` for a node that can be held.
    """
    pymc_model = graph.pymc_model
    if variable not in pymc_model.named_vars:
        return f"{variable!r} is not a variable of the model"
    node = pymc_model[variable]
    if node not in pymc_model.deterministics:
        return f"{variable!r} is not a registered Deterministic"
    if variable in graph.frozen:
        return f"{variable!r} is frozen at its posterior values"
    upstream = set(ancestors([node]))
    if graph.channel_data not in upstream:
        return f"spend does not reach {variable!r}"
    if node in graph.direct_path or not graph.direct_path.isdisjoint(
        _expression_under(node)
    ):
        return (
            f"{variable!r} is on the direct path: it is "
            f"{CHANNEL_CONTRIBUTION!r}, upstream of it or registered on "
            "either"
        )
    if variable in graph.terms:
        return f"{variable!r} is the term an effect adds to the linear predictor"
    if graph.predictor is not None and graph.predictor in upstream:
        return f"{variable!r} is the linear predictor or downstream of it"
    blocked = blocked_between(
        pymc_model, graph.channel_data, [node], frozen_deterministics=graph.frozen
    )
    if blocked:
        held = ", ".join(repr(var.name) for var in blocked)
        return (
            f"{held} lies between spend and {variable!r} and is held at its "
            "posterior values, so a counterfactual does not move the node"
        )
    return None


def _report_problem(model: MMM, name: str, declaration: _Declaration) -> str | None:
    """Say why a node's movement cannot be reported under its declared name.

    Parameters
    ----------
    model : MMM
        The model the node is a variable of.
    name : str
        The declared name.
    declaration : _Declaration
        What the effects declare about the node.

    Returns
    -------
    str or None
        The reason, or ``None`` for a movement that can be reported.  A
        variable the model does not have is the node's problem, not this one.
    """
    if declaration.fault is not None:
        return declaration.fault
    pymc_model = model.model
    variable = declaration.variable
    if variable not in pymc_model.named_vars:
        return None
    if not isinstance(pymc_model[variable], XTensorVariable):
        return (
            f"{variable!r} is a plain tensor, and a movement is read "
            "off a variable registered through pymc.dims"
        )
    if pymc_model[variable].type.dtype == "bool":
        return (
            f"{variable!r} is boolean, and a movement is a difference: "
            "register it as a number to report how often it changes"
        )
    dims = tuple(pymc_model.named_vars_to_dims.get(variable, ()))
    if Dim.DATE not in dims:
        return f"{variable!r} has no 'date' dimension"
    if Dim.CHANNEL in dims:
        return f"{variable!r} carries the 'channel' dimension"
    extra = tuple(dim for dim in dims if dim not in (Dim.DATE, *model.dims))
    if extra:
        return f"{variable!r} has dimensions {extra} the model does not have"
    if name in model.dims:
        return f"the name {name!r} is a dimension of the model"
    if name in pymc_model.named_vars and name != variable:
        return f"the name {name!r} is another variable of the model"
    return None


def _model_dims_of(model: MMM, variable: str) -> tuple[str, ...]:
    """Return the model's dimensions besides ``date`` that *variable* carries, in the model's order."""
    dims = model.model.named_vars_to_dims.get(variable, ())
    return tuple(dim for dim in model.dims if dim in dims)


def resolve_mediators(
    model: MMM, effects: Sequence[ChannelDependentEffect]
) -> tuple[ResolvedMediator, ...]:
    """List the mediator nodes the model's effects declare.

    Every ``mu_effect`` is asked for its spec, also one whose term does not
    depend on spend: such an effect may still declare a node that spend drives,
    such as an indicator with a likelihood on it.  A hub several effects read is
    declared by each of them, under one name and for one variable, and is one
    entry.  At most one of them says how to report it.

    A fault in a declaration is recorded on its entry and raises nothing here.
    It matters only to the request that needs the node.  A name that two
    effects use for two variables gives one entry per variable: neither can be
    reported under that name, and each still marks where its own effect's
    routes are cut.

    Parameters
    ----------
    model : MMM
        The model whose effects are read.
    effects : sequence of ChannelDependentEffect
        The effects :func:`~pymc_marketing.mmm.spend_reach.resolve_channel_dependent_effects`
        returned for it.

    Returns
    -------
    tuple of ResolvedMediator
        One entry per declared name and variable, sorted by both.

    Raises
    ------
    ValueError
        If an effect returns something that is not a valid
        :class:`~pymc_marketing.mmm.additive_effect.IncrementalitySpec`.
    """
    declared = _collect_declarations(model)
    graph = _SpendGraph.of(model, effects)
    resolved = []
    for name, variable in sorted(declared):
        declaration = declared[(name, variable)]
        units, scale, is_spend = declaration.reporting
        resolved.append(
            ResolvedMediator(
                name=name,
                variable=variable,
                effects=tuple(declaration.effects),
                units=units,
                scale=scale,
                is_spend=is_spend,
                feeds_term=graph.feeds_term(variable),
                full_axis=declaration.full_axis,
                declared_carryover_lags=declaration.carryover_lags,
                dims=_model_dims_of(model, variable),
                problem=_cut_problem(graph, variable),
                unreadable=_report_problem(model, name, declaration),
            )
        )
    return tuple(resolved)


# ==================== The role checks ====================


@dataclass(frozen=True)
class _MediatorFootprint:
    """What a ``"mediator"`` term owes to its declared nodes.

    Parameters
    ----------
    declared : list of str
        The variables of the nodes its effect declares.
    feeding : tuple of Variable
        The usable declared nodes the term reads without passing through
        another term.
    owed : frozenset of Variable
        The feeding nodes, the expressions under them, and whatever lies
        between them and the term, the term included.
    """

    declared: list[str]
    feeding: tuple[Variable, ...]
    owed: frozenset[Variable]


def _mediator_footprint(
    terms: Mapping[str, Variable],
    name: str,
    declared: Sequence[str],
    usable: Mapping[str, Variable],
) -> _MediatorFootprint:
    """Find what the ``"mediator"`` term *name* owes to its declared nodes.

    Parameters
    ----------
    terms : mapping
        The terms of the linear predictor, by name.
    name : str
        The mediator term.
    declared : sequence of str
        The variables of the nodes its effect declares.
    usable : mapping
        The declared nodes that can mark a cut, by variable.

    Returns
    -------
    _MediatorFootprint
        The nodes that feed the term and what depends on them inside it.
    """
    within = _inside(terms, name)
    feeding = tuple(
        usable[variable]
        for variable in declared
        if variable in usable and usable[variable] in within
    )
    # One pass in topological order marks what depends on the feeding nodes.
    downstream = set(feeding)
    for apply in toposort([terms[name]]):
        if not downstream.isdisjoint(apply.inputs):
            downstream.update(apply.outputs)
    owed = downstream & within
    for node in feeding:
        owed.update(_expression_under(node))
    return _MediatorFootprint(
        declared=list(declared), feeding=feeding, owed=frozenset(owed)
    )


@dataclass(frozen=True)
class _RoleGraph:
    """The graph facts the role checks read, computed once per request.

    Parameters
    ----------
    channel_data : Variable
        The spend a counterfactual moves.
    terms : dict
        The terms of the linear predictor, by name.
    usable : dict
        The declared nodes that can mark a cut, by variable.
    problems : dict
        Why each other declared node cannot, by variable.
    footprints : dict
        What each ``"mediator"`` term owes to its nodes, by term, in the order
        of the effects.
    names : dict
        The registered name of every named variable of the model.
    """

    channel_data: Variable
    terms: dict[str, Variable]
    usable: dict[str, Variable]
    problems: dict[str, str]
    footprints: dict[str, _MediatorFootprint]
    names: dict[Variable, str]

    @classmethod
    def of(
        cls,
        model: MMM,
        effects: Sequence[ChannelDependentEffect],
        registry: Sequence[ResolvedMediator],
        specs: Mapping[str, IncrementalitySpec],
    ) -> _RoleGraph:
        """Read the graph facts the role checks need."""
        pymc_model = model.model
        terms = _terms(model, effects)
        usable = {
            entry.variable: pymc_model[entry.variable]
            for entry in registry
            if entry.problem is None
        }
        return cls(
            channel_data=pymc_model[Var.CHANNEL_DATA],
            terms=terms,
            usable=usable,
            problems={
                entry.variable: entry.problem
                for entry in registry
                if entry.problem is not None
            },
            footprints={
                name: _mediator_footprint(
                    terms,
                    name,
                    [mediator.variable for mediator in spec.mediators],
                    usable,
                )
                for name, spec in specs.items()
                if spec.role == "mediator"
            },
            names={variable: name for name, variable in pymc_model.named_vars.items()},
        )


def _direct_term_contradictions(graph: _RoleGraph, name: str) -> list[str]:
    """List how the ``"direct"`` term *name* moves with a mediator.

    It contradicts its role when it reads a declared node or the expression
    under one (check 1 of :func:`check_roles`), or when it is built from what a
    ``"mediator"`` term owes to its nodes (check 2).

    Parameters
    ----------
    graph : _RoleGraph
        The graph facts of the request.
    name : str
        The direct term.

    Returns
    -------
    list of str
        One sentence per contradiction, empty for a term that keeps its role.
    """
    found: list[str] = []
    reads = _inside(graph.terms, name)
    for variable, node in graph.usable.items():
        if node in reads:
            found.append(
                f"the direct term {name!r} reads the declared node {variable!r}"
            )
        elif not reads.isdisjoint(_expression_under(node)):
            found.append(
                f"the direct term {name!r} reads the expression under "
                f"the declared node {variable!r}, which makes it move "
                "with the mediator"
            )
    upstream = set(ancestors([graph.terms[name]]))
    for mediator, footprint in graph.footprints.items():
        shared_nodes = footprint.owed & upstream
        if not shared_nodes:
            continue
        shared = sorted(graph.names[var] for var in shared_nodes if var in graph.names)
        found.append(
            f"the direct term {name!r} is built from what the "
            f"mediator term {mediator!r} owes to its nodes"
            + (f" ({', '.join(map(repr, shared))})" if shared else "")
        )
    return found


def _expression_hint(
    term: Variable,
    declared: Sequence[str],
    usable: Mapping[str, Variable],
    other_mediator_terms: Sequence[Variable],
) -> str:
    """Point at declared nodes a term reads around, through the expression under them.

    Parameters
    ----------
    term : Variable
        A ``"mediator"`` term.
    declared : sequence of str
        The variables of the nodes its effect declares.
    usable : mapping
        The declared nodes that can mark a cut, by variable.
    other_mediator_terms : sequence of Variable
        The other mediator terms, which count as cuts.

    Returns
    -------
    str
        A clause to append to a contradiction, or an empty string.
    """
    bypassed = [
        variable
        for variable in declared
        if variable in usable
        and not set(
            ancestors([term], blockers=[usable[variable], *other_mediator_terms])
        ).isdisjoint(_expression_under(usable[variable]))
    ]
    if not bypassed:
        return ""
    return f"; it reads the expression under {bypassed}, not the registered variable"


def _unfed_mediator_contradiction(graph: _RoleGraph, name: str, hint: str) -> str:
    """Describe a ``"mediator"`` term that reads none of its declared nodes.

    Parameters
    ----------
    graph : _RoleGraph
        The graph facts of the request.
    name : str
        The mediator term.
    hint : str
        What :func:`_expression_hint` found for it.

    Returns
    -------
    str
        The sentence, with why its nodes cannot be used and, when spend reaches
        it through another mediator term, what to register.
    """
    declared = graph.footprints[name].declared
    faults = "; ".join(
        graph.problems[variable] for variable in declared if variable in graph.problems
    )
    upstream = set(ancestors([graph.terms[name]]))
    through = [
        other
        for other in graph.footprints
        if other != name and graph.terms[other] in upstream
    ]
    via = (
        f"; spend reaches it through the mediator term {through}, and "
        "it needs a node of its own between that term and itself: "
        "register what it takes from the term as a Deterministic and "
        "declare that"
        if through
        else ""
    )
    return (
        f"the mediator term {name!r} reads none of its declared "
        f"nodes {declared}" + (f" ({faults})" if faults else "") + hint + via
    )


def _mediator_term_contradictions(graph: _RoleGraph, name: str) -> list[str]:
    """List how spend reaches the ``"mediator"`` term *name* around its declared nodes.

    Check 3 of :func:`check_roles`: the term reads at least one of its own
    declared nodes, and spend does not reach it once those nodes and the other
    mediator terms are cut out.

    Parameters
    ----------
    graph : _RoleGraph
        The graph facts of the request.
    name : str
        The mediator term.

    Returns
    -------
    list of str
        At most one sentence, empty for a term that keeps its role.
    """
    term = graph.terms[name]
    footprint = graph.footprints[name]
    others = [graph.terms[other] for other in graph.footprints if other != name]
    hint = _expression_hint(term, footprint.declared, graph.usable, others)
    if not footprint.feeding:
        return [_unfed_mediator_contradiction(graph, name, hint)]
    if graph.channel_data in set(
        ancestors([term], blockers=[*footprint.feeding, *others])
    ):
        return [
            f"spend reaches the mediator term {name!r} past its "
            f"declared nodes {footprint.declared}{hint}"
        ]
    return []


def _contradiction_error(found: Sequence[str]) -> ValueError:
    """Build the refusal of a model whose graph contradicts its declared roles."""
    return ValueError(
        "The declared roles contradict the model's graph: "
        + "; ".join(dict.fromkeys(found))
        + ".  A direct effect leaves the mediator terms out, which equals "
        "holding the mediators only if spend reaches each mediator term "
        "through its declared nodes alone and no direct term reads them.  "
        "Declare the nodes the term is built from, build the term on the "
        "registered variables, or change the role."
    )


def check_roles(
    model: MMM,
    effects: Sequence[ChannelDependentEffect],
    registry: Sequence[ResolvedMediator],
    specs: Mapping[str, IncrementalitySpec],
) -> None:
    """Refuse a model whose graph contradicts the roles its effects declare.

    A direct effect is computed by leaving the ``"mediator"`` terms out, which
    equals holding their nodes only when the graph has this shape:

    1. No ``"direct"`` term reads a declared node, or the expression a declared
       node was registered from.
    2. No ``"direct"`` term has among its ancestors anything a ``"mediator"``
       term owes to its nodes: the nodes, the expressions under them, and
       whatever lies between them and the term, the term included.
    3. Every ``"mediator"`` term reads at least one of its own declared nodes
       without passing through another term, and spend does not reach the term
       once those nodes and the other mediators' terms are cut out.

    What lies upstream of a mediator's nodes is a transform of spend, and a
    direct effect may share it.  Another mediator's term counts as a cut because
    it is held too.

    Parameters
    ----------
    model : MMM
        The model whose graph is read.
    effects : sequence of ChannelDependentEffect
        Its channel-dependent effects.
    registry : sequence of ResolvedMediator
        What :func:`resolve_mediators` returned.  Nodes with a ``problem`` are
        not used, and are named when an effect is left without a usable node.
    specs : mapping
        The validated spec of each effect in *effects*, by contribution
        variable.  An effect without a role is left alone here.

    Raises
    ------
    ValueError
        If any of the three checks fails, naming each term and the variables
        involved.
    """
    graph = _RoleGraph.of(model, effects, registry, specs)
    found: list[str] = []
    for name, spec in specs.items():
        if spec.role == "direct":
            found.extend(_direct_term_contradictions(graph, name))
        elif spec.role == "mediator":
            found.extend(_mediator_term_contradictions(graph, name))
    if found:
        raise _contradiction_error(found)


def _validated_roles(
    effects: Sequence[ChannelDependentEffect],
) -> dict[str, IncrementalitySpec]:
    """Validate the spec of every effect, and refuse one that declares no role.

    Parameters
    ----------
    effects : sequence of ChannelDependentEffect
        The model's channel-dependent effects.

    Returns
    -------
    dict
        The validated spec of each effect, by contribution variable.

    Raises
    ------
    NotImplementedError
        If an effect declares no role.
    ValueError
        If an effect's spec is not valid.
    """
    specs = {
        effect.contribution_var: (
            None if effect.spec is None else _validated_spec(effect.spec, effect.label)
        )
        for effect in effects
    }
    undeclared = [
        effect
        for effect in effects
        if (spec := specs[effect.contribution_var]) is None or spec.role is None
    ]
    if undeclared:
        listing = ", ".join(
            f"{effect.label!r} ({effect.contribution_var!r})" for effect in undeclared
        )
        raise NotImplementedError(
            "A request for the direct or the mediated path needs a role from "
            "every mu_effect that depends on channel spend, and these declare "
            f"none: {listing}.  Return IncrementalitySpec(role='mediator', "
            "mediators=...) for a term spend reaches only through nodes the "
            "effect declares, or IncrementalitySpec(role='direct') for a term "
            "that is one more direct function of spend.  Totals do not need a "
            "role."
        )
    return {name: spec for name, spec in specs.items() if spec is not None}


def resolve_path_plan(
    *,
    model: MMM,
    effects: Sequence[ChannelDependentEffect],
    registry: Sequence[ResolvedMediator],
) -> PathPlan:
    """Decide what a direct effect holds fixed, after checking the declarations.

    Parameters
    ----------
    model : MMM
        The model whose graph is read.
    effects : sequence of ChannelDependentEffect
        Its channel-dependent effects.
    registry : sequence of ResolvedMediator
        What :func:`resolve_mediators` returned.

    Returns
    -------
    PathPlan
        The terms of every ``"mediator"`` effect.

    Raises
    ------
    NotImplementedError
        If a channel-dependent effect declares no role.
    ValueError
        If the graph contradicts the declared roles; see :func:`check_roles`.
    """
    roles = _validated_roles(effects)
    check_roles(model, effects, registry, roles)
    return PathPlan(
        held_terms=tuple(
            sorted(name for name, spec in roles.items() if spec.role == "mediator")
        )
    )


# ==================== The mediators a request reads ====================


def _every_readable_mediator(
    registry: Sequence[ResolvedMediator],
) -> tuple[ResolvedMediator, ...]:
    """Return every declared node whose movement can be reported.

    Parameters
    ----------
    registry : sequence of ResolvedMediator
        Every node the model's effects declare.

    Returns
    -------
    tuple of ResolvedMediator
        The readable nodes, in the order of *registry*.

    Raises
    ------
    ValueError
        If the model declares no mediator, or none can be read.

    Warns
    -----
    UserWarning
        If some declared node is left out, naming why.
    """
    if not registry:
        raise ValueError(
            "This model declares no mediator, so there is no movement "
            "to report.  A mu_effect declares the nodes spend moves in "
            "the 'mediators' of its IncrementalitySpec."
        )
    # A name declared for two variables has an entry for each, with one reason
    # between them.
    left_out = list(
        dict.fromkeys(
            f"{entry.name!r}: {entry.problem or entry.unreadable}"
            for entry in registry
            if not entry.readable
        )
    )
    selected = tuple(entry for entry in registry if entry.readable)
    if not selected:
        raise ValueError(
            "None of the declared mediators can be read.  "
            + "  ".join(f"{reason}." for reason in left_out)
        )
    if left_out:
        warnings.warn(
            "Some declared mediators are left out because they cannot "
            "be read.  " + "  ".join(f"{reason}." for reason in left_out),
            UserWarning,
            skip_file_prefixes=(_PKG_PREFIX,),
        )
    return selected


def _named_mediators(
    registry: Sequence[ResolvedMediator], names: Sequence[str]
) -> tuple[ResolvedMediator, ...]:
    """Return the declared nodes a request names, each of which has to be readable.

    Parameters
    ----------
    registry : sequence of ResolvedMediator
        Every node the model's effects declare.
    names : sequence of str
        The declared names asked for.

    Returns
    -------
    tuple of ResolvedMediator
        The nodes, in the order named.

    Raises
    ------
    ValueError
        If a name is not declared, or its node cannot be read.
    """
    declared = {entry.name: entry for entry in registry}
    unknown = [name for name in names if name not in declared]
    if unknown:
        raise ValueError(
            f"The names {unknown} are not declared mediators of this "
            f"model.  The declared mediators are {sorted(declared)}; see "
            "Incrementality.mediators."
        )
    selected = tuple(declared[name] for name in names)
    for entry in selected:
        if not entry.readable:
            raise ValueError(
                f"The mediator {entry.name!r} cannot be read: "
                f"{entry.problem or entry.unreadable}."
            )
    return selected


def _refuse_unfit_induced_spend(
    selected: Sequence[ResolvedMediator],
    *,
    into: InducedInto,
    model_dims: Sequence[str],
) -> None:
    """Refuse nodes whose movement cannot be counted as spend a counterfactual induces.

    Parameters
    ----------
    selected : sequence of ResolvedMediator
        The named nodes.
    into : {"ratio", "decomposition"}
        A ratio adds the movement to each cell's spend, so a node has to have
        the model's dimensions.  A decomposition returns it in the nodes' own
        dimensions, which they have to share.
    model_dims : sequence of str
        The model's dimensions besides ``date``.

    Raises
    ------
    ValueError
        If a node is not declared as spend, feeds no term of the linear
        predictor, differs in units from another, or does not have the
        dimensions *into* needs.
    """
    not_spend = [entry.name for entry in selected if not entry.is_spend]
    if not_spend:
        raise ValueError(
            f"The mediators {not_spend} are not declared with "
            "is_spend=True, so their movement is not money to add to "
            "channel spend."
        )
    for entry in selected:
        if not entry.feeds_term:
            raise ValueError(
                f"The spend mediator {entry.name!r} feeds no term of the "
                "linear predictor, so what it buys is in no increment and "
                "the result would count its cost alone.  If it is a copy "
                "of a node in other units, declare that node with a "
                "'scale' instead."
            )
    units = sorted({str(entry.units) for entry in selected})
    if len(units) > 1:
        raise ValueError(
            "The named mediators are added together and are declared in "
            f"different units: {units}."
        )
    _refuse_mismatched_dims(selected, into=into, model_dims=model_dims)


def _refuse_mismatched_dims(
    selected: Sequence[ResolvedMediator],
    *,
    into: InducedInto,
    model_dims: Sequence[str],
) -> None:
    """Refuse spend mediators whose dimensions the result they enter cannot hold.

    Parameters
    ----------
    selected : sequence of ResolvedMediator
        The named spend nodes.
    into : {"ratio", "decomposition"}
        A ratio adds the movement to each cell's spend, so a node has to have
        the model's dimensions: a national node in a panel model would be
        counted once per geo.  A decomposition returns the summed movement in
        the nodes' own dimensions, which they have to share.
    model_dims : sequence of str
        The model's dimensions besides ``date``.

    Raises
    ------
    ValueError
        If a node does not have the dimensions *into* needs.
    """
    if into == "ratio":
        for entry in selected:
            if entry.dims != tuple(model_dims):
                raise ValueError(
                    f"The mediator {entry.name!r} has dimensions {entry.dims} besides "
                    f"'date' and the model has {tuple(model_dims)}.  "
                    "Added to each cell's spend, its movement would be "
                    "counted once per cell.  "
                    "decompose_incremental_contribution(induced_spend=...) "
                    "returns it in its own dimensions."
                )
    elif len({entry.dims for entry in selected}) > 1:
        dims_of = {entry.name: entry.dims for entry in selected}
        raise ValueError(
            "The named mediators are added into one 'induced_spend' and do "
            f"not share their dimensions: {dims_of}.  Ask for them in "
            "separate calls."
        )


def select_mediators(
    registry: Sequence[ResolvedMediator],
    names: Sequence[str] | None,
    *,
    induced_into: InducedInto | None = None,
    model_dims: Sequence[str],
) -> tuple[ResolvedMediator, ...]:
    """Pick the mediator nodes a request reads, refusing what it cannot read.

    Parameters
    ----------
    registry : sequence of ResolvedMediator
        What :func:`resolve_mediators` returned.
    names : sequence of str or None
        The declared names the request asks for, or ``None`` for every node
        that can be read.
    induced_into : {"ratio", "decomposition"}, optional
        Set when the named nodes are read as the spend a counterfactual
        induces, added to a ratio's denominator or returned next to the spend
        change of a decomposition.  Each has its own rules for which nodes
        qualify.
    model_dims : sequence of str
        The model's dimensions besides ``date``, which a ratio's induced spend
        has to have.  Required, so that a caller cannot leave out the
        dimensions a per-geo ratio is checked against.

    Returns
    -------
    tuple of ResolvedMediator
        The nodes to read, in the order named, or in the order of *registry*.

    Raises
    ------
    ValueError
        If the model declares no mediator; if a name is not declared or its
        node cannot be read; or, for induced spend, if a node is not declared
        as spend, feeds no term of the linear predictor, differs in units from
        another, or does not have the dimensions the result it is added to has.

    Warns
    -----
    UserWarning
        If every readable node was asked for and some declared node is left
        out.
    """
    if names is None:
        return _every_readable_mediator(registry)
    selected = _named_mediators(registry, names)
    if induced_into is not None:
        _refuse_unfit_induced_spend(selected, into=induced_into, model_dims=model_dims)
    return selected
