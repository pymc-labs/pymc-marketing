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

"""Lazy stochastic equations composed with the shared model-term lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from copy import copy, deepcopy
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import Any

import numpy as np
import pymc as pm
import pymc.dims as pmd
import xarray as xr
from pymc_extras.prior import Prior, VariableFactory
from pytensor.graph.basic import Variable

from pymc_marketing.mmm.link import LinkFunction, get_link_spec
from pymc_marketing.terms import (
    Dot,
    ModelTerm,
    Parameter,
    Product,
    Sum,
    build_param,
    get_coords,
    register_data,
)

_ACTIVE_CONTEXT: ContextVar[BuildContext | None] = ContextVar(
    "experimental_mmm_build_context", default=None
)


def _dimensions(dims: str | Sequence[str]) -> tuple[str, ...]:
    result = (dims,) if isinstance(dims, str) else tuple(dims)
    if any(not isinstance(dim, str) or not dim for dim in result):
        raise ValueError("Dimensions must be nonempty strings.")
    if len(result) != len(set(result)):
        raise ValueError("Dimensions must be unique.")
    return result


def _name(value: Any, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string.")


def _copy_recipe_value(value: Any) -> Any:
    if isinstance(value, Prior):
        return copy_prior(value)
    if isinstance(value, VariableFactory):
        clone = copy(value)
        for name, item in _attributes(value).items():
            object.__setattr__(clone, name, _copy_recipe_value(item))
        return clone
    if isinstance(value, dict):
        return {key: _copy_recipe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_recipe_value(item) for item in value)
    if isinstance(value, list):
        return [_copy_recipe_value(item) for item in value]
    if isinstance(value, Variable):
        return value
    return deepcopy(value)


def copy_prior(prior: Prior) -> Prior:
    """Copy a distribution recipe without losing its named core dimensions.

    Parameters
    ----------
    prior : Prior
        Distribution recipe, possibly containing nested priors.

    Returns
    -------
    Prior
        Independent recipe preserving distribution, parameters, dimensions, core dimensions, centering, and transform.

    Notes
    -----
    ``Prior.deepcopy`` currently serializes through a representation without ``core_dims``.
    Reconstructing the semantic fields also deliberately excludes the runtime ``dim_handler``.
    """
    if not isinstance(prior, Prior):
        raise TypeError("Expected a Prior distribution recipe.")
    return Prior(
        prior.distribution,
        dims=prior.dims,
        core_dims=prior.core_dims,
        centered=prior.centered,
        transform=prior.transform,
        **{key: _copy_recipe_value(value) for key, value in prior.parameters.items()},
    )


class GraphTerm(ModelTerm):
    """A lazy graph node using the shared ``ModelTerm`` algebra.

    Subclasses declare symbolic dependencies and implement ``_build(context)``.
    Build-time values belong to the context, never to the reusable specification.
    """

    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def dependencies(self) -> tuple[Any, ...]:
        """Return the symbolic expressions on which this node depends."""
        return ()

    def _build(self, context: BuildContext) -> Any:
        raise NotImplementedError

    def _specification_state(self) -> Any:
        """Return semantic configuration for structural mutation detection."""
        return vars(self)

    def create_variable(self) -> Any:
        """Resolve this node through the active identity-aware build context."""
        context = _ACTIVE_CONTEXT.get()
        if context is None:
            raise RuntimeError(
                "Graph terms must be built through BuildContext.build()."
            )
        return context.build(self)


class Data(GraphTerm):
    """Reference a raw, mutable named variable in the model dataset.

    Parameters
    ----------
    var_name : str
        Dataset variable name; no scaling is applied by this term.
    """

    def __init__(self, var_name: str) -> None:
        _name(var_name, "var_name")
        self.var_name = var_name

    def _build(self, context: BuildContext) -> Any:
        return context.data(self.var_name)


@dataclass(frozen=True)
class Binding:
    """Effective name, observations, dimensions, and divisor for an equation.

    Parameters
    ----------
    name : str
        Unique model variable name.
    observed : str, optional
        Dataset observation variable, or ``None`` for a latent equation.
    dims : tuple of str
        Named dimensions of the equation output.
    scale : xarray.DataArray or float, default 1
        Finite, nonzero divisor applied to observations; channel equations use one.
    """

    name: str
    observed: str | None
    dims: tuple[str, ...]
    scale: xr.DataArray | float = 1.0

    def __post_init__(self) -> None:
        _name(self.name, "name")
        _name(self.observed, "observed", optional=True)
        object.__setattr__(self, "dims", _dimensions(self.dims))
        if isinstance(self.scale, xr.DataArray):
            if not set(self.scale.dims).issubset(self.dims):
                raise ValueError("Scale dimensions must be equation dimensions.")
            values = self.scale.values
            object.__setattr__(self, "scale", self.scale.copy(deep=True))
        else:
            values = np.asarray(self.scale)
            if values.ndim:
                raise TypeError("A dimensional scale must be an xarray.DataArray.")
        try:
            valid = np.isfinite(values).all() and np.all(values != 0)
        except TypeError as error:
            raise ValueError("Equation scales must be finite and nonzero.") from error
        if not valid:
            raise ValueError("Equation scales must be finite and nonzero.")


class Equation(GraphTerm):
    """Declare a stochastic equation without constructing any PyMC variables.

    Parameters
    ----------
    mu : object, optional
        Convenience expression for the likelihood's ``mu`` parameter.
    likelihood : Prior, optional
        Distribution recipe, defaulting to the identity-link Normal likelihood.
    name : str, optional
        Model variable name, overridden by an effective binding when supplied.
    observed : str, optional
        Dataset observation variable; ``None`` leaves this to a workflow binding.
        Without a binding, ``None`` declares a latent equation.
    dims : tuple of str, optional
        Output dimensions, inferred from observations or the context when omitted.
    parameters : mapping of str to object, optional
        Arbitrary symbolic distribution parameters, including non-``mu`` parameterizations.
        A parameter cannot also be specified in ``likelihood`` or through ``mu``.

    Notes
    -----
    Prediction conditioning holds supplied intermediate observations fixed without updating posterior parameter draws.
    It is a forward-prediction policy, not posterior inference with new evidence.
    Missing training observations are rejected rather than automatically imputed.
    MMM outcome and channel slots supply their own observation bindings.
    An outcome uses ``target_column`` unless ``observed`` overrides it; channel slots use the raw channel column.
    """

    def __init__(
        self,
        *,
        mu: Any = None,
        likelihood: Prior | None = None,
        name: str | None = None,
        observed: str | None = None,
        dims: str | Sequence[str] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if likelihood is None:
            likelihood = get_link_spec(LinkFunction.IDENTITY).default_likelihood(())
            likelihood.dims = None
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping of distribution parameters.")
        self.likelihood = likelihood
        self.name = name
        self.observed = observed
        self.dims = None if dims is None else _dimensions(dims)
        self.parameters = dict(parameters or {})
        if mu is not None:
            if "mu" in self.parameters:
                raise ValueError("mu is specified both directly and in parameters.")
            self.parameters["mu"] = mu
        self._validate()

    @property
    def mu(self) -> Any:
        """Return the convenience mean expression, when present."""
        return self.parameters.get("mu")

    @mu.setter
    def mu(self, value: Any) -> None:
        if "mu" in self.likelihood.parameters:
            raise ValueError("mu is already specified in the likelihood recipe.")
        self.parameters["mu"] = value

    def _validate(self) -> None:
        _name(self.name, "name", optional=True)
        _name(self.observed, "observed", optional=True)
        if not isinstance(self.likelihood, Prior):
            raise TypeError("likelihood must be a Prior distribution recipe.")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping of distribution parameters.")
        for key in self.parameters:
            _name(key, "Parameter names")
        overlap = self.parameters.keys() & self.likelihood.parameters.keys()
        if overlap:
            raise ValueError(
                f"Parameters are specified in both equation and likelihood: {sorted(overlap)}."
            )
        if "observed" in self.parameters or "observed" in self.likelihood.parameters:
            raise ValueError(
                "Specify observations through Equation.observed or Binding, not parameters."
            )

    def dependencies(self) -> tuple[Any, ...]:
        """Return the symbolic distribution-parameter expressions."""
        return tuple(self.parameters.values())

    def _build(self, context: BuildContext) -> Any:
        self._validate()
        binding = context.binding(self)
        context.equations[id(self)] = self
        context.equation_names[id(self)] = binding.name
        context._claim_name(binding.name, self)
        context._ensure_dims(binding.dims)
        if context.prediction and binding.name in context.condition_on:
            if binding.observed is None:
                raise ValueError(
                    f"Cannot condition on latent equation {binding.name!r}."
                )
            observed = context._observations(binding)
            return pmd.Deterministic(binding.name, observed, dims=binding.dims)

        prior = copy_prior(self.likelihood)
        if binding.observed is not None and (prior.transform or not prior.centered):
            raise ValueError(
                "Observed equations do not support transformed or noncentered likelihood recipes."
            )
        prior.parameters.update(
            {
                parameter: context.build(expression, name=f"{binding.name}_{parameter}")
                for parameter, expression in self.parameters.items()
            }
        )
        prior.dims = binding.dims
        if not context.prediction and binding.observed is not None:
            prior.parameters["observed"] = context._observations(binding)
        value = prior.create_variable(binding.name, xdist=True)
        if context.prediction and context.history_length and "date" in binding.dims:
            if binding.observed is None:
                raise ValueError(
                    f"Observation-indexed latent equation {binding.name!r} cannot regenerate an unobserved history."
                )
            history = context._observation_array(binding).isel(
                date=slice(0, context.history_length)
            )
            context._finite(history, binding.observed)
            history = context._scaled_array(history, binding)
            prefix = pmd.as_xtensor(history)
            future = value.isel(date=slice(context.history_length, None))
            return pmd.concat([prefix, future], dim="date")
        return value


def _attributes(value: Any) -> dict[str, Any]:
    result = dict(vars(value)) if hasattr(value, "__dict__") else {}
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            result.setdefault(field.name, getattr(value, field.name))
    return result


def _children(value: Any) -> tuple[Any, ...]:
    if isinstance(value, GraphTerm):
        return value.dependencies()
    if isinstance(value, (ModelTerm, Sum, Product)):
        return tuple(_attributes(value).values())
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return ()


def walk(root: Any, stop: Callable[[Any], bool] | None = None) -> Iterator[Any]:
    """Yield each unique symbolic node, rejecting true dependency cycles.

    Parameters
    ----------
    root : object
        Graph term, shared term composition, or container of expressions.
    stop : callable, optional
        Return true to yield a node without traversing its dependencies.

    Yields
    ------
    object
        Each ``ModelTerm``, ``Sum``, or ``Product`` once, in dependency-first order.

    Raises
    ------
    ValueError
        If an expression depends on itself, directly or through other nodes.
    """
    yield from _walk(root, stop=stop)


def _walk(root: Any, stop: Callable[[Any], bool] | None = None) -> Iterator[Any]:
    visited: set[int] = set()
    active: set[int] = set()

    def visit(value: Any) -> Iterator[Any]:
        children = () if stop is not None and stop(value) else _children(value)
        symbolic = isinstance(value, (ModelTerm, Sum, Product))
        if not symbolic and not children:
            return
        key = id(value)
        if key in active:
            raise ValueError("Cycle detected in equation dependencies.")
        if key in visited:
            return
        active.add(key)
        for child in children:
            yield from visit(child)
        active.remove(key)
        visited.add(key)
        if symbolic:
            yield value

    yield from visit(root)


class _Reference(ModelTerm):
    """Route a cloned lifecycle's child operations through its owning context."""

    def __init__(self, context: BuildContext, original: Any) -> None:
        self.context = context
        self.original = original

    def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
        return self.context._coords(self.original)

    def add_coords(self, ds: xr.Dataset) -> None:
        self.context._add_coords(self.original)

    def register_data(self, ds: xr.Dataset) -> None:
        self.context._register(self.original)

    def create_variable(self) -> Any:
        return self.context.build(self.original)


class BuildContext:
    """Build a shared-term equation graph once per symbolic identity.

    Parameters
    ----------
    ds : xarray.Dataset
        Normalized model data, including any fixed prediction-history prefix.
    bindings : mapping, optional
        Equation identities (``id(equation)`` or equation objects) mapped to effective bindings.
    prediction : bool, default False
        Build generated or held-fixed outputs instead of observed likelihoods.
    condition_on : sequence of str, default ()
        Effective equation names whose observations are held fixed during prediction.
    history_length : int, default 0
        Number of measured historical rows prepended to a prediction dataset.
    default_dims : tuple of str, default ("date",)
        Dimensions for equations without explicit or observable dimensions.

    Attributes
    ----------
    variables : dict
        Symbolic identity to built value; an equation may store its anonymous history-spliced expression.
    equations : dict
        Built equation identities to original equation specifications.
    equation_names : dict
        Equation identities to actual named PyMC outputs, not anonymous downstream history expressions.
    data_variables : dict
        Raw dataset variable names to uniquely named model data containers.

    Notes
    -----
    Operates inside an active ``pm.Model``.
    Shared terms retain their lifecycle on one shallow clone per identity.
    Child references bind to this context.
    Priors are independent distribution recipes; only a shared ``Parameter`` object establishes parameter identity.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        *,
        bindings: Mapping[Any, Binding] | None = None,
        prediction: bool = False,
        condition_on: Sequence[str] = (),
        history_length: int = 0,
        default_dims: tuple[str, ...] = ("date",),
    ) -> None:
        if not isinstance(ds, xr.Dataset):
            raise TypeError("BuildContext requires an xarray.Dataset.")
        if (
            isinstance(history_length, bool)
            or not isinstance(history_length, int)
            or history_length < 0
        ):
            raise ValueError("history_length must be a nonnegative integer.")
        if history_length and (
            "date" not in ds.dims or history_length >= ds.sizes["date"]
        ):
            raise ValueError("Prediction history must leave at least one future date.")
        if isinstance(condition_on, str) or any(
            not isinstance(name, str) for name in condition_on
        ):
            raise TypeError("condition_on must be a sequence of equation names.")
        self.ds = ds
        self.prediction = prediction
        self.condition_on = frozenset(condition_on)
        self.history_length = history_length
        self.default_dims = _dimensions(default_dims)
        self.model = pm.modelcontext(None)
        self.variables: dict[int, Any] = {}
        self.equations: dict[int, Equation] = {}
        self.equation_names: dict[int, str] = {}
        self.data_variables: dict[str, str] = {}
        self._shared_data_aliases: set[str] = set()
        self._bindings: dict[int, Binding] = {}
        for equation, binding in (bindings or {}).items():
            if not isinstance(binding, Binding):
                raise TypeError("Each effective binding must be a Binding.")
            key = equation if isinstance(equation, int) else id(equation)
            if key in self._bindings and specification_key(
                self._bindings[key]
            ) != specification_key(binding):
                raise ValueError(
                    "One equation cannot have conflicting effective bindings."
                )
            self._bindings[key] = binding
        self._clones: dict[int, Any] = {}
        self._references: dict[int, _Reference] = {}
        self._building: set[int] = set()
        self._phase_active: set[tuple[str, int]] = set()
        self._phase_done: set[tuple[str, int]] = set()
        self._coordinate_cache: dict[int, dict[str, Any]] = {}
        self._names: dict[str, Any] = {}
        self._suggested_names: dict[int, str] = {}

    def binding(self, equation: Equation) -> Binding:
        """Return the effective metadata for an equation without building it."""
        key = id(equation)
        if key not in self._bindings:
            dims = equation.dims
            if dims is None and equation.observed in self.ds:
                dims = tuple(self.ds[equation.observed].dims)
            if dims is None:
                dims = equation.likelihood.dims
            if dims is None:
                dims = self.default_dims
            self._bindings[key] = Binding(
                name=equation.name
                or equation.observed
                or self._suggested_names.get(key, "equation"),
                observed=equation.observed,
                dims=_dimensions(dims),
            )
        return self._bindings[key]

    def _claim_name(self, name: str, owner: Any) -> None:
        _name(name, "Model variable name")
        previous = self._names.get(name)
        if previous is not None and previous is not owner:
            raise ValueError(
                f"Model variable name {name!r} is owned by two different graph nodes."
            )
        if previous is None and name in self.model.named_vars:
            raise ValueError(
                f"Model variable name {name!r} already exists in the model."
            )
        self._names[name] = owner

    def _add_model_coords(self, coords: Mapping[str, Any]) -> None:
        for dim, labels in coords.items():
            existing = self.model.coords.get(dim)
            if existing is None:
                self.model.add_coord(dim, values=labels)
            elif not np.array_equal(np.asarray(existing), np.asarray(labels)):
                raise ValueError(
                    f"Coordinate labels for dimension {dim!r} do not match the model."
                )

    def _ensure_dims(self, dims: Sequence[str]) -> None:
        self._add_model_coords(
            {dim: self.ds.coords[dim].values for dim in dims if dim in self.ds.coords}
        )
        missing = set(dims).difference(self.model.coords)
        if missing:
            raise ValueError(f"Missing coordinates for dimensions {sorted(missing)}.")

    @staticmethod
    def _finite(value: xr.DataArray, name: str) -> None:
        try:
            finite = np.isfinite(value.values).all()
        except TypeError as error:
            raise ValueError(
                f"Data variable {name!r} must contain finite numeric values."
            ) from error
        if not finite:
            raise ValueError(
                f"Data variable {name!r} contains missing or nonfinite observations."
            )

    def data(self, var_name: str) -> Any:
        """Register and return a finite, raw mutable dataset variable."""
        if var_name not in self.ds:
            raise ValueError(f"Required data variable {var_name!r} is missing.")
        if var_name in self.data_variables:
            return self.model[self.data_variables[var_name]]
        array = self.ds[var_name]
        self._finite(array, var_name)
        self._ensure_dims(array.dims)
        index = len(self.data_variables)
        name = f"_experimental_data_{index}"
        while name in self.model.named_vars or name in self._names:
            index += 1
            name = f"_experimental_data_{index}"
        variable = pmd.Data(name, array)
        self.data_variables[var_name] = name
        return variable

    def _observation_array(self, binding: Binding) -> xr.DataArray:
        if binding.observed not in self.ds:
            raise ValueError(
                f"Required observations {binding.observed!r} for equation {binding.name!r} are missing."
            )
        array = self.ds[binding.observed]
        if set(array.dims) != set(binding.dims):
            raise ValueError(
                f"Observation dimensions for {binding.name!r} must be {binding.dims}."
            )
        return array.transpose(*binding.dims)

    def _scaled_array(self, array: xr.DataArray, binding: Binding) -> xr.DataArray:
        scale = binding.scale
        if isinstance(scale, xr.DataArray):
            array, scale = xr.align(array, scale, join="exact", copy=False)
        return array / scale

    def _observations(self, binding: Binding) -> Any:
        observed = binding.observed
        if observed is None:
            raise ValueError("Cannot read observations for a latent equation.")
        array = self._observation_array(binding)
        self._finite(array, observed)
        scaled = self._scaled_array(array, binding)
        # Keep the raw mutable data container and apply the fixed divisor symbolically.
        scale = (
            pmd.as_xtensor(binding.scale)
            if isinstance(binding.scale, xr.DataArray)
            else binding.scale
        )
        self._finite(scaled, observed)
        return self.data(observed) / scale

    def _bound_value(self, value: Any) -> Any:
        if isinstance(value, (ModelTerm, Sum, Product)):
            key = id(value)
            if key not in self._references:
                self._references[key] = _Reference(self, value)
            return self._references[key]
        if isinstance(value, VariableFactory):
            return _copy_recipe_value(value)
        if isinstance(value, dict):
            return {key: self._bound_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._bound_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._bound_value(item) for item in value)
        return value

    def _clone(self, term: Any) -> Any:
        key = id(term)
        if key not in self._clones:
            clone = copy(term)
            self._clones[key] = clone
            for name, value in _attributes(term).items():
                object.__setattr__(clone, name, self._bound_value(value))
        return self._clones[key]

    def _start_phase(self, phase: str, term: Any) -> bool:
        key = (phase, id(term))
        if key in self._phase_active:
            raise ValueError("Cycle detected in model-term lifecycle dependencies.")
        if key in self._phase_done:
            return False
        self._phase_active.add(key)
        return True

    def _coords(self, term: Any) -> dict[str, Any]:
        if isinstance(term, GraphTerm):
            return {}
        key = id(term)
        if self._start_phase("coords", term):
            try:
                self._coordinate_cache[key] = get_coords(self._clone(term), self.ds)
            finally:
                self._phase_active.remove(("coords", key))
            self._phase_done.add(("coords", key))
        return self._coordinate_cache[key]

    def _add_coords(self, term: Any) -> None:
        if isinstance(term, GraphTerm) or not isinstance(term, ModelTerm):
            return
        if self._start_phase("add_coords", term):
            try:
                self._clone(term).add_coords(self.ds)
            finally:
                self._phase_active.remove(("add_coords", id(term)))
            self._phase_done.add(("add_coords", id(term)))

    def _register(self, term: Any) -> None:
        if isinstance(term, GraphTerm):
            return
        if self._start_phase("register", term):
            try:
                if isinstance(term, Dot):
                    if term.var_name not in self.ds:
                        raise ValueError(
                            f"Required data variable {term.var_name!r} is missing."
                        )
                    self._finite(self.ds[term.var_name], term.var_name)
                    if term.var_name in self._names or (
                        term.var_name in self.model.named_vars
                        and term.var_name not in self._shared_data_aliases
                    ):
                        raise ValueError(
                            f"Data variable name {term.var_name!r} is already owned by another model variable."
                        )
                self._add_model_coords(self._coords(term))
                if isinstance(term, ModelTerm):
                    self._add_coords(term)
                    self._clone(term).register_data(self.ds)
                else:
                    register_data(self._clone(term), ds=self.ds)
                if isinstance(term, Dot):
                    self._shared_data_aliases.add(term.var_name)
            finally:
                self._phase_active.remove(("register", id(term)))
            self._phase_done.add(("register", id(term)))

    def _conditioned(self, value: Any) -> bool:
        if not self.prediction or not isinstance(value, Equation):
            return False
        binding = self._bindings.get(id(value))
        name = binding.name if binding is not None else value.name or value.observed
        return name in self.condition_on

    def build(self, expression: Any, name: str = "param") -> Any:
        """Build an expression once by identity while activating recursive shared terms."""
        if pm.modelcontext(None) is not self.model:
            raise RuntimeError(
                "BuildContext must be used in the model where it was created."
            )
        if isinstance(expression, Variable):
            return expression
        if isinstance(expression, np.generic):
            expression = expression.item()
        if not isinstance(expression, (ModelTerm, Sum, Product)):
            recipe = (
                _copy_recipe_value(expression)
                if isinstance(expression, VariableFactory)
                else expression
            )
            if isinstance(recipe, VariableFactory):
                self._claim_name(name, object())
            token = _ACTIVE_CONTEXT.set(self)
            try:
                return build_param(recipe, name=name)
            finally:
                _ACTIVE_CONTEXT.reset(token)
        key = id(expression)
        if key in self._building:
            raise ValueError("Cycle detected in equation dependencies.")
        if key in self.variables:
            return self.variables[key]
        self._suggested_names.setdefault(key, name)
        if not self._building:
            for _ in _walk(expression, stop=self._conditioned):
                pass
        self._building.add(key)
        token = _ACTIVE_CONTEXT.set(self)
        try:
            if isinstance(expression, GraphTerm):
                value = expression._build(self)
            else:
                if isinstance(expression, (Parameter, Dot)):
                    variable_name = expression.name
                    if variable_name is None:
                        raise ValueError("Parameters and coefficients must have names.")
                    self._claim_name(variable_name, expression)
                if isinstance(expression, Dot):
                    if expression.var_name not in self.ds:
                        raise ValueError(
                            f"Required data variable {expression.var_name!r} is missing."
                        )
                    self._finite(self.ds[expression.var_name], expression.var_name)
                self._register(expression)
                value = build_param(self._clone(expression), name=name)
            self.variables[key] = value
            return value
        finally:
            _ACTIVE_CONTEXT.reset(token)
            self._building.remove(key)


def specification_key(*objects: Any) -> Any:
    """Fingerprint semantic configuration and symbolic sharing topology.

    Parameters
    ----------
    *objects : object
        Equation graphs and their configuration objects.

    Returns
    -------
    tuple
        Comparable structural representation suitable for detecting deep specification mutations.

    Notes
    -----
    Includes Prior core dimensions and semantic callable identity (for example ``Transform.func``).
    Graph terms can override ``_specification_state`` to omit callbacks, model references, and build-time caches.
    """
    seen: dict[int, int] = {}
    retained: list[Any] = []

    def freeze(value: Any) -> Any:
        if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
            return (type(value).__qualname__, value)
        if isinstance(value, np.generic):
            return (str(value.dtype), value.tobytes())
        if isinstance(
            value, (FunctionType, BuiltinFunctionType, MethodType, type, np.ufunc)
        ):
            return ("callable", id(value))
        if isinstance(value, Variable):
            return ("tensor", id(value))
        key = id(value)
        if key in seen:
            return ("reference", seen[key])
        seen[key] = len(seen)
        # Hooks and xarray accessors may create temporary containers; retain them so ids cannot be recycled.
        retained.append(value)
        kind = (type(value).__module__, type(value).__qualname__)
        if isinstance(value, partial):
            return (
                kind,
                freeze(value.func),
                freeze(value.args),
                freeze(value.keywords),
            )
        if isinstance(value, Prior):
            return (
                kind,
                freeze(value.distribution),
                freeze(value.dims),
                freeze(value.core_dims),
                value.centered,
                freeze(value.transform),
                freeze(value.parameters),
            )
        if isinstance(value, GraphTerm):
            return (kind, freeze(value._specification_state()))
        if isinstance(value, xr.Dataset):
            return (kind, freeze(dict(value.data_vars)), freeze(dict(value.coords)))
        if isinstance(value, xr.DataArray):
            return (
                kind,
                value.dims,
                freeze(value.values),
                tuple(
                    (str(name), freeze(coord.values))
                    for name, coord in value.coords.items()
                ),
            )
        if isinstance(value, np.ndarray):
            payload = (
                freeze(value.tolist()) if value.dtype.hasobject else value.tobytes()
            )
            return (kind, value.shape, str(value.dtype), payload)
        if isinstance(value, Mapping):
            return (
                kind,
                tuple((freeze(key), freeze(item)) for key, item in value.items()),
            )
        if isinstance(value, (tuple, list)):
            return (kind, tuple(freeze(item) for item in value))
        if isinstance(value, (set, frozenset)):
            return (kind, tuple(sorted((freeze(item) for item in value), key=repr)))
        state = _attributes(value)
        if state:
            return (
                kind,
                freeze(
                    {
                        key: item
                        for key, item in state.items()
                        if key not in {"on_change", "dim_handler"}
                    }
                ),
            )
        if callable(value):
            return (kind, id(value))
        return (kind, repr(value))

    return tuple(freeze(value) for value in objects)
