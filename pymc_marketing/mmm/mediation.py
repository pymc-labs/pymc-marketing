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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError
from pytensor.compile.ops import ViewOp
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors, toposort
from pytensor.xtensor.basic import TensorFromXTensor, XTensorFromTensor
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.additive_effect import IncrementalitySpec
from pymc_marketing.mmm.spend_reach import (
    CHANNEL_CONTRIBUTION,
    ChannelDependentEffect,
    _find_linear_predictor,
)
from pymc_marketing.names.mmm import Dim, Var
from pymc_marketing.pytensor_utils import blocked_between

if TYPE_CHECKING:
    from pymc_marketing.mmm.mmm import MMM

__all__ = [
    "PathPlan",
    "ResolvedMediator",
    "check_roles",
    "resolve_mediators",
    "resolve_path_plan",
]

_VALUE_KEEPING_OPS = (ViewOp, TensorFromXTensor, XTensorFromTensor)
"""Ops whose output is their input: a registration, or a change of tensor kind."""

_DEFAULT_REPORTING = (None, 1.0, False)
"""``(units, scale, is_spend)`` of a node nobody said how to report."""


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


def _inside(terms: dict[str, Variable], name: str) -> set[Variable]:
    """Return a term and whatever it reads without passing through another term."""
    others = [term for other, term in terms.items() if other != name]
    return set(ancestors([terms[name]], blockers=others))


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
    It matters only to the request that needs the node.

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
        One entry per declared name, sorted by name.

    Raises
    ------
    ValueError
        If an effect returns something that is not a valid
        :class:`~pymc_marketing.mmm.additive_effect.IncrementalitySpec`.
    """
    pymc_model = model.model
    declared: dict[str, dict] = {}
    for effect in model.mu_effects:
        spec = getattr(effect, "incrementality_spec", lambda: None)()
        if spec is None:
            continue
        label = type(effect).__name__
        for mediator in _validated_spec(spec, label).mediators:
            entry = declared.setdefault(
                mediator.name,
                {
                    "variable": mediator.variable,
                    "reporting": _DEFAULT_REPORTING,
                    "effects": [],
                    "fault": None,
                },
            )
            entry["effects"].append(label)
            reporting = (mediator.units, mediator.scale, mediator.is_spend)
            if entry["variable"] != mediator.variable:
                entry["fault"] = (
                    f"the name {mediator.name!r} is declared for "
                    f"{entry['variable']!r} and for {mediator.variable!r}"
                )
            elif reporting != _DEFAULT_REPORTING:
                if entry["reporting"] not in (_DEFAULT_REPORTING, reporting):
                    entry["fault"] = (
                        f"{mediator.variable!r} is declared with two sets of "
                        "reporting fields (units, scale, is_spend)"
                    )
                entry["reporting"] = reporting

    names_of: dict[str, list[str]] = {}
    for name, entry in declared.items():
        names_of.setdefault(entry["variable"], []).append(name)
    for variable, names in names_of.items():
        if len(names) > 1:
            for name in names:
                declared[name]["fault"] = (
                    f"{variable!r} is declared under the names {sorted(names)}"
                )

    channel_data = pymc_model[Var.CHANNEL_DATA]
    terms = _terms(model, effects)
    inside = [_inside(terms, name) for name in terms]
    direct_path = set(ancestors([terms[CHANNEL_CONTRIBUTION]]))
    predictor = _find_linear_predictor(model)
    frozen = model.frozen_deterministics

    def problem(variable: str) -> str | None:
        if variable not in pymc_model.named_vars:
            return f"{variable!r} is not a variable of the model"
        node = pymc_model[variable]
        if node not in pymc_model.deterministics:
            return f"{variable!r} is not a registered Deterministic"
        if variable in frozen:
            return f"{variable!r} is frozen at its posterior values"
        upstream = set(ancestors([node]))
        if channel_data not in upstream:
            return f"spend does not reach {variable!r}"
        if node in direct_path:
            return (
                f"{variable!r} is on the direct path: it is "
                f"{CHANNEL_CONTRIBUTION!r} or upstream of it"
            )
        if variable in terms:
            return f"{variable!r} is the term an effect adds to the linear predictor"
        if predictor is not None and predictor in upstream:
            return f"{variable!r} is the linear predictor or downstream of it"
        blocked = blocked_between(
            pymc_model, channel_data, [node], frozen_deterministics=frozen
        )
        if blocked:
            held = ", ".join(repr(var.name) for var in blocked)
            return (
                f"{held} lies between spend and {variable!r} and is held at its "
                "posterior values, so a counterfactual does not move the node"
            )
        return None

    def unreadable(name: str, entry: dict) -> str | None:
        if entry["fault"] is not None:
            return entry["fault"]
        variable = entry["variable"]
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

    resolved = []
    for name in sorted(declared):
        entry = declared[name]
        variable = entry["variable"]
        node = pymc_model.named_vars.get(variable)
        units, scale, is_spend = entry["reporting"]
        resolved.append(
            ResolvedMediator(
                name=name,
                variable=variable,
                effects=tuple(entry["effects"]),
                units=units,
                scale=scale,
                is_spend=is_spend,
                feeds_term=any(node in reads for reads in inside),
                problem=problem(variable),
                unreadable=unreadable(name, entry),
            )
        )
    return tuple(resolved)


def check_roles(
    model: MMM,
    effects: Sequence[ChannelDependentEffect],
    registry: Sequence[ResolvedMediator],
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
        Its channel-dependent effects.  One without a role is left alone here.
    registry : sequence of ResolvedMediator
        What :func:`resolve_mediators` returned.  Nodes with a ``problem`` are
        not used, and are named when an effect is left without a usable node.

    Raises
    ------
    ValueError
        If any of the three checks fails, naming each term and the variables
        involved.
    """
    pymc_model = model.model
    channel_data = pymc_model[Var.CHANNEL_DATA]
    terms = _terms(model, effects)
    specs = {
        effect.contribution_var: _validated_spec(effect.spec, effect.label)
        for effect in effects
        if effect.spec is not None
    }
    usable = {
        entry.variable: pymc_model[entry.variable]
        for entry in registry
        if entry.problem is None
    }
    problems = {
        entry.variable: entry.problem for entry in registry if entry.problem is not None
    }
    mediator_terms = [name for name, spec in specs.items() if spec.role == "mediator"]
    own_nodes = {
        name: [mediator.variable for mediator in specs[name].mediators]
        for name in mediator_terms
    }

    feeding: dict[str, list[Variable]] = {}
    owed: dict[str, set[Variable]] = {}
    for name in mediator_terms:
        within = _inside(terms, name)
        feeding[name] = [
            usable[variable]
            for variable in own_nodes[name]
            if variable in usable and usable[variable] in within
        ]
        # One pass in topological order marks what depends on the feeding nodes.
        downstream = set(feeding[name])
        for apply in toposort([terms[name]]):
            if not downstream.isdisjoint(apply.inputs):
                downstream.update(apply.outputs)
        owed[name] = downstream & within
        for node in feeding[name]:
            owed[name].update(_expression_under(node))

    named = {variable: name for name, variable in pymc_model.named_vars.items()}
    found: list[str] = []
    for name, spec in specs.items():
        term = terms[name]
        if spec.role == "direct":
            reads = _inside(terms, name)
            for variable, node in usable.items():
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
            upstream = set(ancestors([term]))
            for mediator in mediator_terms:
                shared = sorted(
                    named[var] for var in owed[mediator] & upstream if var in named
                )
                if owed[mediator] & upstream:
                    found.append(
                        f"the direct term {name!r} is built from what the "
                        f"mediator term {mediator!r} owes to its nodes"
                        + (f" ({', '.join(map(repr, shared))})" if shared else "")
                    )
        elif spec.role == "mediator":
            own = own_nodes[name]
            others = [terms[other] for other in mediator_terms if other != name]
            bypassed = [
                variable
                for variable in own
                if variable in usable
                and not set(
                    ancestors([term], blockers=[usable[variable], *others])
                ).isdisjoint(_expression_under(usable[variable]))
            ]
            hint = (
                f"; it reads the expression under {bypassed}, not the "
                "registered variable"
                if bypassed
                else ""
            )
            if not feeding[name]:
                faults = "; ".join(
                    problems[variable] for variable in own if variable in problems
                )
                upstream = set(ancestors([term]))
                through = [
                    other
                    for other in mediator_terms
                    if other != name and terms[other] in upstream
                ]
                via = (
                    f"; spend reaches it through the mediator term {through}, and "
                    "it needs a node of its own between that term and itself: "
                    "register what it takes from the term as a Deterministic and "
                    "declare that"
                    if through
                    else ""
                )
                found.append(
                    f"the mediator term {name!r} reads none of its declared "
                    f"nodes {own}" + (f" ({faults})" if faults else "") + hint + via
                )
            elif channel_data in set(
                ancestors([term], blockers=[*feeding[name], *others])
            ):
                found.append(
                    f"spend reaches the mediator term {name!r} past its "
                    f"declared nodes {own}{hint}"
                )

    if found:
        raise ValueError(
            "The declared roles contradict the model's graph: "
            + "; ".join(dict.fromkeys(found))
            + ".  A direct effect leaves the mediator terms out, which equals "
            "holding the mediators only if spend reaches each mediator term "
            "through its declared nodes alone and no direct term reads them.  "
            "Declare the nodes the term is built from, build the term on the "
            "registered variables, or change the role."
        )


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
    undeclared = [
        effect
        for effect in effects
        if effect.spec is None
        or _validated_spec(effect.spec, effect.label).role is None
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
    check_roles(model, effects, registry)
    return PathPlan(
        held_terms=tuple(
            sorted(
                effect.contribution_var
                for effect in effects
                if _validated_spec(effect.spec, effect.label).role == "mediator"
            )
        )
    )
