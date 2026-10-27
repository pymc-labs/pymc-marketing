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
"""Reusable labeled decision problems with fixed per-solve numerical preconditioning."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import xarray as xr
from pytensor.gradient import jacobian
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting import rewrite_graph
from pytensor.graph.rewriting.basic import in2out, node_rewriter
from pytensor.graph.traversal import ancestors
from pytensor.tensor.elemwise import CAReduce, DimShuffle
from pytensor.tensor.shape import Shape, Shape_i
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    Subtensor,
    as_index_constant,
    indices_from_subtensor,
)
from pytensor.tensor.variable import TensorConstant
from pytensor.xtensor.type import XTensorVariable, as_xtensor
from scipy.optimize import Bounds, OptimizeResult, minimize

from pymc_marketing.experimental.terms._data import (
    _align_labels,
    _dates,
)
from pymc_marketing.experimental.terms._graph import BuildContext
from pymc_marketing.experimental.terms._optimizer_terms import (
    _bind_terms,
    _coordinates,
    _data_dependencies,
    _fitted_inputs,
    _tensor_date_axis,
)
from pymc_marketing.experimental.terms._scm import SCM
from pymc_marketing.terms import ModelTerm, Product, Sum

type _Term = ModelTerm | Product | Sum
type _Callback = Callable[[Any, Mapping[str, Any]], Any]
type _Array = float | xr.DataArray
_LOWER = ("lower_xtensor", "canonicalize", "stabilize")


def _align(value: Any, template: xr.DataArray, what: str) -> xr.DataArray:
    """Broadcast scalars or exactly labeled subdimensions onto a decision array."""
    if np.iscomplexobj(value):
        raise ValueError(f"{what} must contain real numeric values.")
    if isinstance(value, xr.DataArray):
        if value.dtype.kind == "O" and any(
            np.iscomplexobj(element) for element in value.values.flat
        ):
            raise ValueError(f"{what} must contain real numeric values.")
        for dim in value.dims:
            if dim not in value.coords or value.coords[dim].dims != (dim,):
                raise ValueError(f"{what} needs one-dimensional labels for {dim!r}.")
        numeric = value.astype(float, copy=False)
        if value.dtype.kind in "iuO" and value.size:
            if (
                value.dtype.kind == "O"
                or int(value.values.min()) < -(2**53)
                or int(value.values.max()) > 2**53
            ) and any(
                isinstance(original, Integral)
                and (not np.isfinite(converted) or int(original) != int(converted))
                for original, converted in zip(
                    value.values.flat, numeric.values.flat, strict=True
                )
            ):
                raise ValueError(f"{what} must be exactly representable as float64.")
        return (
            _align_labels(numeric, template)
            .broadcast_like(template)
            .transpose(*template.dims)
        )
    if np.ndim(value) != 0:
        raise TypeError(f"{what} must be a scalar or a labeled xarray.DataArray.")
    scalar_value = float(value)
    if isinstance(value, Integral) and (
        not np.isfinite(scalar_value) or int(scalar_value) != int(value)
    ):
        raise ValueError(f"{what} must be exactly representable as float64.")
    return xr.full_like(template, scalar_value, dtype=float)


def _positive(value: Any, what: str, *, scalar: bool = False) -> None:
    """Reject scales and tolerances that cannot define an invertible transform."""
    if scalar and np.ndim(value) != 0:
        raise ValueError(f"{what} must be scalar.")
    if not np.isfinite(value).all() or not (np.asarray(value) > 0).all():
        raise ValueError(f"{what} must be finite and strictly positive.")


def _quotient(value: Any, scale: Any, *, allow_underflow: Any = False) -> Any:
    """Reject normalization that overflows or loses a nonzero finite value."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result = np.divide(value, scale)
    if not np.isfinite(result).all():
        raise ValueError(
            "Scaled values and derivatives must be finite; choose representable scales."
        )
    if np.any((np.asarray(value) != 0) & (result == 0) & ~np.asarray(allow_underflow)):
        raise ValueError(
            "Scaled nonzero values and derivatives must not underflow to zero; choose representable scales."
        )
    return result


@node_rewriter([Subtensor])
def _rewrite_constant_subtensor_indices(_fgraph: Any, node: Any) -> list[Any] | None:
    """Give equal integer indices equal types in the structural dependency proof."""
    base, *indices = node.inputs
    # XTensor scalar selections can use narrow integers, unlike per-row NumPy indices.
    # Shared decoded expressions prevent index lifting, so different widths can hide x - x.
    # Change only exactly representable scalar index constants, never arithmetic casts.
    normalized = [
        as_index_constant(int(index.data))
        if isinstance(index, Constant)
        and index.dtype != "int64"
        and np.ndim(index.data) == 0
        and np.asarray(index.data).dtype.kind in "iu"
        and -(2**63) <= int(index.data) < 2**63
        else index
        for index in indices
    ]
    if all(
        index is replacement
        for index, replacement in zip(indices, normalized, strict=True)
    ):
        return None
    return [node.op(base, *normalized)]


@node_rewriter([AdvancedIncSubtensor])
def _rewrite_full_slice_scatter(_fgraph: Any, node: Any) -> list[Any] | None:
    """Expose complete identity scatters to PyTensor's full-slice rewrites."""
    base, update, *index_variables = node.inputs
    shape = base.type.shape
    if any(size is None for size in shape) or shape != update.type.shape:
        return None
    # Advanced increments add in the promoted dtype before casting back to the base.
    # Basic-slice increments can cast the update first, so mixed dtypes are not equivalent.
    if not node.op.set_instead_of_inc and base.dtype != update.dtype:
        return None
    indices = indices_from_subtensor(index_variables, node.op.idx_list)
    if (
        not indices
        or not isinstance(indices[0], TensorConstant)
        or indices[0].ndim != 1
        or indices[0].data.dtype.kind not in "iu"
        or any(
            not isinstance(index, slice) or index != slice(None)
            for index in indices[1:]
        )
        or not np.array_equal(indices[0].data, np.arange(shape[0]))
    ):
        return None
    # Keep PyTensor responsible for the write's casting and broadcasting semantics.
    return [
        pt.inc_subtensor(base[:], update, set_instead_of_inc=node.op.set_instead_of_inc)
    ]


@dataclass(frozen=True)
class OptimizationResult:
    """Original-unit solution and unmodified scaled-solver diagnostics.

    Attributes
    ----------
    allocation : xarray.Dataset
        Decided inputs only, with their original units, dimensions, and coordinate labels.
    objective : float
        Maximized callback value in its original units.
    constraints : list of xarray.DataArray
        Original-unit residuals, in declaration order. Equalities require zero;
        inequalities require nonnegative values.
    scipy : scipy.optimize.OptimizeResult
        Unmodified SLSQP result. ``x``, ``fun``, derivatives, and multipliers use
        scaled solver coordinates, not the units of ``allocation``.
    decision_centers, decision_scales : xarray.Dataset
        Fixed affine transform for movable entries after unpacking ``scipy.x``;
        equal-bound entries are projected exactly to their bound values.
    objective_scale : float
        Fixed positive divisor applied to the objective during minimization.
    constraint_scales : list of xarray.DataArray
        Fixed positive divisors applied to constraint rows, with residual labels.
    feasible : bool
        Whether decoded bounds and residuals satisfy ``feasibility_tol`` after
        division by their reported scales, independently of ``scipy.success``.
    max_bound_violation, max_constraint_violation : float
        Maximum dimensionless violations using decision and constraint scales,
        respectively. Residuals themselves remain in original units.

    Notes
    -----
    Solver success is not a guarantee of feasibility or global optimality.
    A flat or nonconvex objective can have multiple local optima.
    """

    allocation: xr.Dataset
    objective: float
    constraints: list[xr.DataArray]
    scipy: OptimizeResult
    decision_centers: xr.Dataset
    decision_scales: xr.Dataset
    objective_scale: float
    constraint_scales: list[xr.DataArray]
    feasible: bool
    max_bound_violation: float
    max_constraint_violation: float


@dataclass
class _Layout:
    templates: dict[str, xr.DataArray]
    names: tuple[str, ...] = field(init=False)
    slices: dict[str, slice] = field(init=False)
    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.names = tuple(sorted(self.templates))
        self.slices = {}
        self.size = 0
        for name in self.names:
            self.slices[name] = slice(self.size, self.size + self.templates[name].size)
            self.size += self.templates[name].size

    def pack(self, arrays: Mapping[str, xr.DataArray]) -> np.ndarray:
        return np.concatenate([arrays[name].values.ravel() for name in self.names])

    def unpack(self, vector: np.ndarray) -> dict[str, xr.DataArray]:
        return {
            name: self.templates[name].copy(
                data=vector[self.slices[name]].reshape(self.templates[name].shape)
            )
            for name in self.names
        }

    def symbolic(self, vector: Any) -> dict[str, Any]:
        return {
            name: as_xtensor(
                vector[self.slices[name]].reshape(self.templates[name].shape),
                dims=self.templates[name].dims,
            )
            for name in self.names
        }


@dataclass
class _Constraint:
    kind: str
    template: xr.DataArray
    explicit_scale: Any
    expression: Any
    solver_rows: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    shape_index: int | None = None
    dependent: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool_))


@dataclass
class _SolveState:
    evaluator: Any
    layout: _Layout
    initial: dict[str, xr.DataArray]
    centers: dict[str, xr.DataArray]
    scales: dict[str, xr.DataArray]
    lower: dict[str, xr.DataArray]
    upper: dict[str, xr.DataArray]
    z_lower: np.ndarray
    z_upper: np.ndarray
    function: Any
    constraints: list[_Constraint]
    report_function: Any
    function_args: tuple[np.ndarray, ...]
    parameter_values: tuple[np.ndarray, ...]
    objective_scale: float = 1.0
    constraint_scales: list[xr.DataArray] = field(default_factory=list)

    def to_z(self, arrays: Mapping[str, xr.DataArray]) -> np.ndarray:
        return np.concatenate(
            [
                _quotient(
                    (arrays[name] - self.centers[name]).values,
                    self.scales[name].values,
                    allow_underflow=(self.lower[name] == self.upper[name]).values,
                ).ravel()
                for name in self.layout.names
            ]
        )

    def to_u(self, z: np.ndarray) -> dict[str, xr.DataArray]:
        blocks = self.layout.unpack(np.asarray(z, dtype=float))
        return {
            name: xr.where(
                self.lower[name] == self.upper[name],
                self.lower[name],
                self.centers[name] + self.scales[name] * blocks[name],
            )
            for name in self.layout.names
        }

    def raw(self, z: np.ndarray) -> list[np.ndarray]:
        z = np.asarray(z, dtype=float)
        if not np.isfinite(z).all():
            raise ValueError("Scaled decision coordinates must be finite.")
        values = self.function(z, *self.function_args)
        if any(not np.isfinite(value).all() for value in values):
            raise ValueError(
                "Objective, constraints, and their derivatives must be finite at every evaluated point."
            )
        for spec in self.constraints:
            shape = (
                tuple(values[spec.shape_index])
                if spec.shape_index is not None
                else spec.expression.type.shape
            )
            if shape != spec.template.shape:
                raise ValueError(
                    "Constraint shape must match its labeled dimensions; reduce sliced windows explicitly."
                )
        del values[2 + 2 * len(self.constraints) :]
        return values

    def labeled_constraints(self, values: list[np.ndarray]) -> list[xr.DataArray]:
        return [
            spec.template.copy(data=values[2 + 2 * index].reshape(spec.template.shape))
            for index, spec in enumerate(self.constraints)
        ]

    def report(
        self, allocation: Mapping[str, xr.DataArray]
    ) -> tuple[float, list[xr.DataArray]]:
        # Replay the written arithmetic at the decoded floats, without algebraic
        # reassociation through z hiding representational rounding in original units.
        values = self.report_function(
            *(allocation[name].values for name in self.layout.names),
            *self.parameter_values,
        )
        if any(not np.isfinite(value).all() for value in values):
            raise ValueError("Reported objective and constraints must be finite.")
        _quotient(values[0], self.objective_scale)
        residuals = []
        for spec, value in zip(self.constraints, values[1:], strict=True):
            if value.shape != spec.template.shape:
                raise ValueError("Constraint shape must match its labeled dimensions.")
            residuals.append(spec.template.copy(data=value))
        return float(values[0]), residuals

    def _violations(
        self, allocation: Mapping[str, xr.DataArray], residuals: list[xr.DataArray]
    ) -> tuple[float, float]:
        # Compare each element with its own scale, including heterogeneous units.
        bounds = max(
            float(
                (
                    np.maximum(
                        self.lower[name] - allocation[name],
                        allocation[name] - self.upper[name],
                    ).clip(min=0)
                    / self.scales[name]
                ).max()
            )
            for name in self.layout.names
        )
        constraints = 0.0
        for spec, value, scale in zip(
            self.constraints, residuals, self.constraint_scales, strict=True
        ):
            normalized = _quotient(value.values, scale.values)
            violation = (
                np.abs(normalized) if spec.kind == "eq" else np.maximum(-normalized, 0)
            )
            constraints = max(constraints, float(np.max(violation, initial=0.0)))
        if not np.isfinite((bounds, constraints)).all():
            raise ValueError("Scaled bound and constraint violations must be finite.")
        return bounds, constraints

    def solve(
        self, *, options: Mapping[str, Any] | None = None, feasibility_tol: float = 1e-6
    ) -> OptimizationResult:
        _positive(feasibility_tol, "feasibility_tol", scalar=True)
        feasibility_tol = float(feasibility_tol)
        z0 = self.to_z(self.initial)
        cached_z: np.ndarray | None = None
        cached_values: list[np.ndarray] = []
        normalisers: list[Any] = [self.objective_scale, self.objective_scale]
        for scale in self.constraint_scales:
            normalisers.extend((scale.values.ravel(), scale.values.reshape(-1, 1)))

        def values(z: np.ndarray) -> list[np.ndarray]:
            nonlocal cached_z, cached_values
            if cached_z is None or not np.array_equal(cached_z, z):
                cached_values = [
                    _quotient(value, scale)
                    for value, scale in zip(self.raw(z), normalisers, strict=True)
                ]
                cached_z = z.copy()
            return cached_values

        def loss(z: np.ndarray) -> tuple[float, np.ndarray]:
            current = values(z)
            return -float(current[0]), -current[1]

        values(z0)
        scipy_constraints = []
        for index, spec in enumerate(self.constraints):
            rows = spec.solver_rows
            if not rows.size:
                continue
            selection = slice(None) if rows.size == spec.template.size else rows
            scipy_constraints.append(
                {
                    "type": spec.kind,
                    "fun": lambda z, i=index, r=selection: values(z)[2 + 2 * i][r],
                    "jac": lambda z, i=index, r=selection: values(z)[3 + 2 * i][r],
                }
            )
        if np.array_equal(self.z_lower, self.z_upper):
            allocation = self.to_u(z0)
            final_objective, residuals = self.report(allocation)
            violation = max(self._violations(allocation, residuals))
            result = OptimizeResult(
                x=z0,
                fun=-float(_quotient(final_objective, self.objective_scale)),
                jac=loss(z0)[1],
                success=violation <= feasibility_tol,
                message="All decisions are fixed by bounds; feasibility checked without a solver.",
                nit=0,
                nfev=1,
                njev=1,
            )
        else:
            result = minimize(
                loss,
                z0,
                jac=True,
                method="SLSQP",
                bounds=Bounds(self.z_lower, self.z_upper),
                constraints=scipy_constraints,
                options={"maxiter": 500, "ftol": 1e-9, **(options or {})},
            )
            allocation = self.to_u(result.x)
            final_objective, residuals = self.report(allocation)
        bound_violation, constraint_violation = self._violations(allocation, residuals)
        feasible = max(bound_violation, constraint_violation) <= feasibility_tol
        if result.success and not feasible:
            warnings.warn(
                "SciPy reported success but the decoded allocation is infeasible at feasibility_tol; "
                "inspect the original-unit residuals and reported scales.",
                RuntimeWarning,
                stacklevel=2,
            )
        return OptimizationResult(
            allocation=xr.Dataset(allocation),
            objective=final_objective,
            constraints=residuals,
            scipy=result,
            decision_centers=xr.Dataset(self.centers),
            decision_scales=xr.Dataset(self.scales),
            objective_scale=self.objective_scale,
            constraint_scales=self.constraint_scales,
            feasible=feasible,
            max_bound_violation=bound_violation,
            max_constraint_violation=constraint_violation,
        )


def _construction_data(
    model: SCM, required: set[str], dates: pd.DatetimeIndex
) -> xr.Dataset:
    """Recover fitted raw input shapes without evaluating selected value graphs."""
    training = model._training_data
    context = model._context
    if context is None:
        raise ValueError("The model must be fitted before constructing a problem.")
    fitted = _fitted_inputs(model)
    arrays = {}
    for name in sorted(required):
        if training is not None and name in training:
            value = training[name]
        else:
            candidates = [variable for variable, raw in fitted.items() if raw == name]
            if not candidates:
                raise ValueError(f"Cannot recover fitted raw input {name!r}.")
            variable = candidates[0]
            dims = tuple(variable.dims)
            value = xr.DataArray(
                variable.get_value(),
                dims=dims,
                coords={dim: context.model.coords[dim] for dim in dims},
            )
        if "date" in value.dims:
            value = value.isel(date=np.arange(len(dates)) % value.sizes["date"])
            value = value.assign_coords(date=dates)
        arrays[name] = BuildContext._canonical_input(value, name, copy=True)
    return xr.Dataset(arrays, coords={"date": dates})


class _CallbackEvaluator:
    """Expose bound terms, labeled constants, and explicit symbolic parameters."""

    def __init__(
        self,
        evaluator: Any,
        parameters: Mapping[str, XTensorVariable],
        decisions: Mapping[str, XTensorVariable],
        parameter_templates: Mapping[str, xr.DataArray],
    ) -> None:
        self._evaluator = evaluator
        self._parameters = parameters
        self._decided_length = len(evaluator.coords["date"])
        self._has_carryover = (
            len(evaluator.output_coords["date"]) > self._decided_length
        )
        self._date_windows: dict[Variable, str] = {}
        for value in decisions.values():
            self._register(value, "decided")
        for name, value in parameters.items():
            if "date" in value.dims:
                size = parameter_templates[name].sizes["date"]
                self._register(
                    value, "decided" if size == self._decided_length else "scored"
                )

    def _register(self, value: XTensorVariable, window: str) -> None:
        if "date" in value.dims:
            self._date_windows[value] = window

    def check_dates(self, value: Variable) -> frozenset[str]:
        """Reject implicit alignment of decided and scored axes, even singletons."""
        if not self._has_carryover:
            return frozenset()

        @cache
        def date_axis(node: Variable) -> int | None:
            return _tensor_date_axis(node, date_axis)

        @cache
        def windows(node: Variable) -> frozenset[str]:
            if node in self._date_windows:
                return frozenset({self._date_windows[node]})
            if isinstance(node, XTensorVariable) and "date" not in node.dims:
                return frozenset()
            owner = node.owner
            if (
                owner is None
                or getattr(node, "ndim", 0) == 0
                or isinstance(owner.op, (Shape, Shape_i))
            ):
                return frozenset()
            if isinstance(owner.op, CAReduce):
                original = owner.inputs[0]
                axis = date_axis(original)
                reduced = owner.op.axis
                if reduced is None or (
                    axis is not None
                    and axis in {item % original.ndim for item in reduced}
                ):
                    return frozenset()
            elif isinstance(owner.op, DimShuffle):
                axis = date_axis(owner.inputs[0])
                if axis is not None and axis not in owner.op.new_order:
                    return frozenset()
            return frozenset(
                window for item in owner.inputs for window in windows(item)
            )

        for node in ancestors([value], blockers=set(self._date_windows)):
            if node in self._date_windows:
                continue
            input_windows = frozenset(
                window
                for item in (() if node.owner is None else node.owner.inputs)
                for window in windows(item)
            )
            if len(windows(node)) > 1 or len(input_windows) > 1:
                raise ValueError(
                    "Cannot combine decided dates with scored carryover dates; "
                    "use evaluate.decided(output) before combining them week by week."
                )
        return windows(value)

    def __call__(self, u: Mapping[str, Any]) -> dict[str, Any]:
        for value in u.values():
            if isinstance(value, Variable) and "scored" in self.check_dates(value):
                raise ValueError(
                    "Evaluator inputs must contain only decided dates; "
                    "use evaluate.decided(output) before reusing scored outputs as inputs."
                )
        outputs = self._evaluator(u)
        for value in outputs.values():
            self._register(value, "scored" if self._has_carryover else "decided")
        return outputs

    def constant(self, value: xr.DataArray) -> Any:
        result = self._evaluator.constant(value)
        if "date" in value.dims:
            self._register(
                result,
                "decided" if value.sizes["date"] == self._decided_length else "scored",
            )
        return result

    def decided(self, value: Any) -> Any:
        result = self._evaluator.decided(value)
        self._register(result, "decided")
        return result

    def parameter(self, name: str) -> Any:
        if name not in self._parameters:
            raise ValueError(f"Unknown problem parameter {name!r}.")
        return self._parameters[name]


class Problem:
    """Bind a reusable original-unit decision problem to fitted deterministic terms.

    Parameters
    ----------
    model : SCM
        Fitted source of parameter bindings and joint posterior draws, never mutated.
    terms : term or mapping of str to term
        One deterministic expression, available as ``evaluate(u)["term"]``, or
        separately named expressions retaining their fitted parameter identities.
        New deterministic wrappers and reductions are allowed; stochastic equations are not.
    dates : pandas.DatetimeIndex
        Nonempty, sorted, unique dates for the rows to decide.
        Temporal terms require consecutive dates at the fitted cadence and phase.
        Static inputs retain their fitted dimensions and nondate coordinate labels.
    objective : callable
        ``objective(evaluate, u)`` returns a scalar symbolic value to maximize.
        ``u`` contains only decided rows in original units.
        ``evaluate(u)`` returns selected terms with their joint posterior ``sample``
        draws and includes carryover rows, but excludes fixed history.
        Write every reduction, trade-off, and conversion explicitly.
        ``evaluate.constant(array)`` aligns labeled constants;
        ``evaluate.parameter(name)`` accesses a declared symbolic parameter.
        Use ``evaluate.decided(output)`` before combining a dated output with ``u``
        week by week; it removes carryover and leaves scalar or undated outputs unchanged.
    bounds : mapping
        Exactly every inferred decision name mapped to ``(lower, upper)`` in original units.
        Each endpoint is a scalar or exactly labeled DataArray over a subset of its dimensions.
        Unbounded decisions require an explicit ``(-np.inf, np.inf)`` entry.
        Equal bounds fix individual entries exactly.
    constraints : sequence of mappings, default ()
        Each has ``type`` (``"eq"`` or ``"ineq"``) and ``fun(evaluate, u)`` returning
        original-unit residuals; inequalities require residual >= 0.
        Vector dimensions retain full decided labels; reduce sliced windows explicitly.
        Optional ``scale`` is a positive scalar or labeled DataArray divisor.
    carryover : int or xarray.Dataset, default 0
        Fixed rows appended after the decided dates at the fitted cadence.
        An integer appends that many zero rows to every dated input, not to static inputs.
        To preserve nonzero controls, supply a Dataset containing exactly all dated
        inputs and consecutive tail dates, for example zero spend and reference price.
        Tail values are snapshotted, never optimized, and excluded from allocations.
    history : xarray.Dataset, optional
        Explicit measured, fixed history immediately preceding the decided dates.
        History is snapshotted and excluded from both outputs and allocations.
        Temporal history, decided rows, and carryover must follow the fitted cadence.
        Date-reduced history targets and static decisions upstream of temporal terms are rejected.
    parameters : mapping of str to float or xarray.DataArray, optional
        Named, finite callback values on fitted, decided, or output coordinates.
        Defaults are snapshotted; solve-time overrides must have the same dimensions
        and labels and cannot change defaults or previous solutions.
    scaling : 'auto', mapping, or None, default 'auto'
        Preconditioning is recomputed once per solve from its initial plan and then fixed.
        Decision scales use finite bound ranges, then initial magnitudes.
        Unbounded zero starts need an explicit decision scale.
        A mapping accepts ``decisions={name: scalar_or_DataArray}`` and
        ``objective=positive_scalar``; constraints can declare their own scales.
        Automatic objective and residual scales use initial movable-decision
        derivatives, falling back to nonzero initial values.
        Connected zero values with zero derivatives need explicit scales.
        ``None`` disables automatic scaling, but not explicit constraint scales.

    Notes
    -----
    Construct once, fill ``decisions`` to create a plan, score it with ``evaluate``,
    then call ``solve`` with that plan.
    Binding and numeric compilation happen once, without evaluating placeholder inputs.
    Reuse the same problem for multiple initial plans and declared-parameter overrides.
    Solver callbacks never sample the posterior.
    SLSQP success does not establish feasibility, causal identification, or global optimality.
    """

    def __init__(
        self,
        *,
        model: SCM,
        terms: _Term | Mapping[str, _Term],
        dates: pd.DatetimeIndex,
        objective: _Callback,
        bounds: Mapping[str, tuple[_Array, _Array]],
        constraints: Sequence[Mapping[str, Any]] = (),
        carryover: int | xr.Dataset = 0,
        history: xr.Dataset | None = None,
        parameters: Mapping[str, _Array] | None = None,
        scaling: str | Mapping[str, Any] | None = "auto",
    ) -> None:
        if isinstance(terms, (ModelTerm, Product, Sum)):
            terms = {"term": terms}
        if not isinstance(terms, Mapping) or not terms:
            raise TypeError(
                "terms must be one deterministic term or a nonempty named mapping of terms."
            )
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(term, (ModelTerm, Product, Sum))
            for name, term in terms.items()
        ):
            raise TypeError(
                "terms must map nonempty string names to deterministic term expressions."
            )
        if not isinstance(dates, pd.DatetimeIndex):
            raise TypeError("dates must be a pandas.DatetimeIndex.")
        dates = _dates(dates).as_unit("us")
        if not len(dates) or not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError("dates must be nonempty, unique, and in increasing order.")
        required = _data_dependencies(terms, model)
        if not required:
            raise ValueError(
                "Selected terms must depend on at least one Data input to optimize."
            )
        if not isinstance(bounds, Mapping):
            raise TypeError("bounds must map every inferred decision to a bound pair.")
        if unknown := set(bounds) - required:
            raise ValueError(f"Bounds for unknown inputs {sorted(unknown)}.")
        if missing := required - set(bounds):
            raise ValueError(
                f"Bounds are required for every decision: missing {sorted(missing)}."
            )
        self._evaluator = _bind_terms(
            model,
            terms,
            _construction_data(model, required, dates),
            history=history,
            carryover=carryover,
        )
        self._layout = _Layout(self._evaluator.templates)
        self._lower: dict[str, xr.DataArray] = {}
        self._upper: dict[str, xr.DataArray] = {}
        for name, template in self._layout.templates.items():
            pair = bounds[name]
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError(f"Bounds for {name!r} must be a (lower, upper) pair.")
            low = _align(pair[0], template, f"lower bound for {name!r}").copy(deep=True)
            high = _align(pair[1], template, f"upper bound for {name!r}").copy(
                deep=True
            )
            if bool(
                np.isnan(low).any()
                or np.isnan(high).any()
                or (low > high).any()
                or np.isposinf(low).any()
                or np.isneginf(high).any()
            ):
                raise ValueError(
                    f"Bounds for {name!r} must be ordered and cannot contain NaN or fixed infinities."
                )
            self._lower[name], self._upper[name] = low, high
        if scaling is None or (isinstance(scaling, str) and scaling == "auto"):
            config: Mapping[str, Any] = {}
        elif isinstance(scaling, Mapping):
            config = scaling
        else:
            raise ValueError(
                "scaling must be 'auto', None, or a mapping of decisions and objective scales."
            )
        if unknown := set(config) - {"decisions", "objective"}:
            raise ValueError(f"Unknown scaling keys {sorted(unknown)}.")
        explicit = config.get("decisions", {})
        if not isinstance(explicit, Mapping):
            raise TypeError("scaling['decisions'] must map input names to scales.")
        if unknown := set(explicit) - required:
            raise ValueError(f"Scales for unknown inputs {sorted(unknown)}.")
        self._automatic_scaling = scaling is not None
        self._decision_scales = {}
        for name, value in explicit.items():
            scale = _align(
                value, self._layout.templates[name], f"decision scale for {name!r}"
            ).copy(deep=True)
            _positive(scale.values, f"Decision scale for {name!r}")
            self._decision_scales[name] = scale
        self._objective_scale = config.get("objective")
        if self._objective_scale is not None:
            _positive(self._objective_scale, "Objective scale", scalar=True)
            self._objective_scale = float(self._objective_scale)
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping of named callback values.")
        self._parameters = {}
        for name, value in parameters.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Parameter names must be nonempty strings.")
            if isinstance(value, xr.DataArray):
                reference = self._evaluator.coords
                if "date" in value.dims and value.sizes["date"] != len(
                    reference["date"]
                ):
                    reference = self._evaluator.output_coords
                value = _align_labels(value, xr.Dataset(coords=_coordinates(reference)))
            elif np.ndim(value) == 0:
                value = xr.DataArray(value)
            else:
                raise TypeError(
                    f"Parameter {name!r} must be a scalar or labeled DataArray."
                )
            self._parameters[name] = BuildContext._canonical_input(
                value, f"parameter {name!r}", copy=True
            )
        try:
            self._compile(objective, constraints)
        except ValueError as error:
            if len(self._evaluator.output_coords.get("date", ())) != len(dates) and any(
                word in str(error).lower()
                for word in ("shape", "dimension", "broadcast", "incompatible")
            ):
                raise ValueError(
                    f"{error} Outputs include fixed carryover rows, whereas decisions contain only "
                    "decided rows; use evaluate.decided(output) to combine them week by week."
                ) from error
            raise
        self._evaluator.compile()

    @property
    def decisions(self) -> xr.Dataset:
        """Return a labeled NaN plan template containing only inferred decisions.

        Returns
        -------
        xarray.Dataset
            A fresh dataset in fitted dimension and nondate label order, with only
            decided dates and no posterior ``sample`` dimension.
        """
        return xr.Dataset(
            {
                name: xr.full_like(template, np.nan, dtype=float)
                for name, template in self._layout.templates.items()
            }
        )

    def evaluate(self, plan: xr.Dataset | Mapping[str, xr.DataArray]) -> xr.Dataset:
        """Score an exactly labeled plan without solving or rebinding.

        Parameters
        ----------
        plan : xarray.Dataset or mapping of str to xarray.DataArray
            Exactly all inferred decisions, in original units, with the template's
            dimensions and coordinate labels, without broadcasting.
            Dimension and label order can differ.
            Values must be finite; bounds are enforced by ``solve``, not by scoring.

        Returns
        -------
        xarray.Dataset
            Selected deterministic terms on decided and carryover rows, excluding
            history and retaining the same joint posterior draws across evaluations.
            Posterior-independent outputs do not acquire a ``sample`` dimension.
        """
        return self._evaluator.evaluate(plan)

    def solve(
        self,
        x0: xr.Dataset | Mapping[str, xr.DataArray],
        *,
        parameters: Mapping[str, _Array] | None = None,
        options: Mapping[str, Any] | None = None,
        feasibility_tol: float = 1e-6,
    ) -> OptimizationResult:
        """Maximize the objective from a required, exactly labeled initial plan.

        Parameters
        ----------
        x0 : xarray.Dataset or mapping of str to xarray.DataArray
            Exactly all finite decided inputs with the template's dimensions and
            labels, inside their original-unit bounds.
        parameters : mapping, optional
            Overrides of declared callback parameters, retaining their dimensions
            and labels; omitted parameters use snapshotted defaults.
        options : mapping, optional
            SLSQP options overriding ``maxiter=500`` and ``ftol=1e-9``.
        feasibility_tol : float, default 1e-6
            Positive tolerance for decoded violations divided by reported scales,
            independent of solver success and stopping tolerance.

        Returns
        -------
        OptimizationResult
            Only decided allocations, original-unit objective and labeled residuals,
            fixed per-solve scales, independent feasibility, and untouched SciPy diagnostics.
            Failure is returned rather than hidden.

        Notes
        -----
        Initial-plan scaling and parameter values are private to this solve.
        Neither this call nor subsequent solves repeat binding or compilation.
        """
        _positive(feasibility_tol, "feasibility_tol", scalar=True)
        return self._prepare(x0, parameters=parameters).solve(
            options=options, feasibility_tol=feasibility_tol
        )

    def _compile(
        self, objective: _Callback, constraints: Sequence[Mapping[str, Any]]
    ) -> None:
        layout = self._layout
        original = {
            name: pt.tensor(
                name=f"original_{name}",
                shape=layout.templates[name].shape,
                dtype="float64",
            )
            for name in layout.names
        }
        u = {
            name: as_xtensor(value, dims=layout.templates[name].dims)
            for name, value in original.items()
        }
        parameters = {
            name: pt.tensor(
                name=f"parameter_{name}", shape=value.shape, dtype="float64"
            )
            for name, value in self._parameters.items()
        }
        evaluate = _CallbackEvaluator(
            self._evaluator,
            {
                name: as_xtensor(value, dims=self._parameters[name].dims)
                for name, value in parameters.items()
            },
            decisions=u,
            parameter_templates=self._parameters,
        )
        z = pt.vector("decisions", shape=(layout.size,), dtype="float64")
        centers = pt.vector("centers", shape=(layout.size,), dtype="float64")
        scales = pt.vector("scales", shape=(layout.size,), dtype="float64")
        center_blocks, scale_blocks = layout.symbolic(centers), layout.symbolic(scales)
        # Fixed projections remain structural constants even though each solve
        # supplies its own centers, scales, and callback parameter values.
        decoded = {
            original[name]: pt.where(
                (self._lower[name] == self._upper[name]).values,
                self._lower[name].values,
                center_blocks[name].values + scale_blocks[name].values * block.values,
            )
            for name, block in layout.symbolic(z).items()
        }
        utility = as_xtensor(objective(evaluate, u))
        evaluate.check_dates(utility)
        if utility.type.ndim != 0:
            raise ValueError(
                f"objective must be scalar; explicitly reduce dimensions {utility.type.dims}."
            )
        report_graphs = [utility.values]
        tensor = rewrite_graph(
            clone_replace(utility.values, replace=decoded),
            include=(*_LOWER, "specialize"),
        )
        self._objective_connected = z in ancestors([tensor])
        outputs = [tensor, pt.grad(tensor, z, disconnected_inputs="ignore")]
        specifications = []
        for spec in constraints:
            if spec.get("type") not in {"eq", "ineq"} or not callable(spec.get("fun")):
                raise ValueError(
                    "Each constraint needs type 'eq' or 'ineq' and a callable fun."
                )
            residual = as_xtensor(spec["fun"](evaluate, u))
            evaluate.check_dates(residual)
            dims = tuple(residual.type.dims)
            if unknown := set(dims) - set(self._evaluator.coords):
                raise ValueError(
                    f"Constraint dimensions {sorted(unknown)} are not labeled."
                )
            coords = {dim: self._evaluator.coords[dim] for dim in dims}
            template = xr.DataArray(
                np.empty(tuple(len(coords[dim]) for dim in dims)),
                dims=dims,
                coords=coords,
            )
            report_graphs.append(residual.values)
            expression = rewrite_graph(
                clone_replace(residual.values, replace=decoded), include=_LOWER
            )
            if any(
                size is not None and size != expected
                for size, expected in zip(
                    expression.type.shape, template.shape, strict=True
                )
            ):
                raise ValueError(
                    "Constraint shape must match its labeled dimensions; reduce sliced windows explicitly."
                )
            flat = expression.flatten()
            outputs += [flat, jacobian(flat, z, disconnected_inputs="ignore")]
            explicit_scale = spec.get("scale")
            if explicit_scale is not None:
                explicit_scale = _align(
                    explicit_scale, template, "Constraint scale"
                ).copy(deep=True)
                _positive(explicit_scale.values, "Constraint scale")
            specifications.append(
                _Constraint(spec["type"], template, explicit_scale, expression)
            )
        for constraint in specifications:
            if any(size is None for size in constraint.expression.type.shape):
                constraint.shape_index = len(outputs)
                outputs.append(constraint.expression.shape)
            # Normalize only the dependency proof; reports preserve written arithmetic.
            projected = rewrite_graph(
                constraint.expression,
                include=(),
                custom_rewrite=in2out(
                    _rewrite_full_slice_scatter, _rewrite_constant_subtensor_indices
                ),
                clone=True,
            )
            constraint.dependent = np.array(
                [
                    z
                    in ancestors(
                        [
                            rewrite_graph(
                                projected[
                                    np.unravel_index(row, constraint.template.shape)
                                ],
                                include=(*_LOWER, "specialize"),
                                clone=True,
                            )
                        ]
                    )
                    for row in range(constraint.template.size)
                ],
                dtype=np.bool_,
            )
        self._function = pytensor.function(
            [z, centers, scales, *parameters.values()],
            outputs,
            on_unused_input="ignore",
        )
        self._report_function = pytensor.function(
            [*original.values(), *parameters.values()],
            rewrite_graph(report_graphs, include=("lower_xtensor",)),
            mode=pytensor.compile.mode.Mode(linker="py", optimizer=None),
            on_unused_input="ignore",
        )
        self._constraints = specifications

    def _parameter_values(
        self, overrides: Mapping[str, _Array] | None
    ) -> tuple[np.ndarray, ...]:
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, Mapping):
            raise TypeError("Parameter overrides must be a mapping.")
        if unknown := set(overrides) - set(self._parameters):
            raise ValueError(f"Overrides for unknown parameters {sorted(unknown)}.")
        values = []
        for name, template in self._parameters.items():
            value = overrides.get(name, template)
            if isinstance(value, xr.DataArray):
                if set(value.dims) != set(template.dims):
                    raise ValueError(
                        f"Parameter {name!r} must have dimensions {template.dims}."
                    )
                value = _align_labels(value, template).transpose(*template.dims)
            elif not template.dims and np.ndim(value) == 0:
                value = xr.DataArray(value)
            else:
                raise ValueError(
                    f"Parameter {name!r} must retain its declared dimensions and labels."
                )
            value = BuildContext._canonical_input(
                value, f"parameter {name!r}", copy=True
            )
            values.append(value.values)
        return tuple(values)

    def _prepare(
        self,
        x0: xr.Dataset | Mapping[str, xr.DataArray],
        *,
        parameters: Mapping[str, _Array] | None = None,
    ) -> _SolveState:
        layout = self._layout
        arrays = self._evaluator.validate_plan(x0)
        initial = {
            name: template.copy(data=arrays[name].copy())
            for name, template in layout.templates.items()
        }
        parameter_values = self._parameter_values(parameters)
        lower, upper = self._lower, self._upper
        centers, scales = {}, {}
        z_lower: dict[str, xr.DataArray] = {}
        z_upper: dict[str, xr.DataArray] = {}
        for name, template in layout.templates.items():
            low, high = lower[name], upper[name]
            if bool((initial[name] < low).any() or (initial[name] > high).any()):
                raise ValueError(
                    f"Initial input {name!r} must be finite and within its bounds."
                )
            fixed = low == high
            if not self._automatic_scaling:
                center, scale = xr.zeros_like(template), xr.ones_like(template)
            elif name in self._decision_scales:
                center, scale = (
                    xr.zeros_like(template),
                    self._decision_scales[name].copy(deep=True),
                )
            else:
                ranged = np.isfinite(low) & np.isfinite(high) & (high > low)
                with np.errstate(over="ignore", invalid="ignore"):
                    scale = xr.where(ranged, high - low, abs(initial[name]))
                scale = xr.where(fixed & (scale == 0), 1.0, scale)
                center = xr.where(ranged | fixed, low, 0.0)
            _positive(
                scale.values,
                f"Decision scale for {name!r}; supply scaling['decisions'] for unbounded zero starts",
            )
            for endpoint, coordinates in ((low, z_lower), (high, z_upper)):
                with np.errstate(
                    over="ignore", under="ignore", invalid="ignore", divide="ignore"
                ):
                    delta = endpoint.values - center.values
                    encoded = delta / scale.values
                finite = np.isfinite(endpoint.values)
                if np.any(finite & ~np.isfinite(encoded)):
                    raise ValueError(
                        "Finite decision bounds must have finite scaled coordinates."
                    )
                if np.any(finite & ~fixed.values & (delta != 0) & (encoded == 0)):
                    raise ValueError(
                        f"Decision bounds for {name!r} underflow to zero in scaled coordinates; "
                        "choose a representable decision scale."
                    )
                coordinates[name] = template.copy(data=encoded)
            if bool(((low < high) & (z_lower[name] >= z_upper[name])).any()):
                raise ValueError(
                    f"Decision scale for {name!r} collapses a nondegenerate bound interval; "
                    "choose a representable decision scale."
                )
            centers[name], scales[name] = center, scale
        specifications = [replace(spec) for spec in self._constraints]
        problem = _SolveState(
            evaluator=self._evaluator,
            layout=layout,
            initial=initial,
            centers=centers,
            scales=scales,
            lower=lower,
            upper=upper,
            z_lower=layout.pack(z_lower),
            z_upper=layout.pack(z_upper),
            function=self._function,
            constraints=specifications,
            report_function=self._report_function,
            function_args=(
                layout.pack(centers),
                layout.pack(scales),
                *parameter_values,
            ),
            parameter_values=parameter_values,
        )
        values = problem.raw(problem.to_z(initial))
        movable = problem.z_lower < problem.z_upper
        if self._automatic_scaling:
            magnitude = float(np.max(abs(values[1][movable]), initial=0.0))
            if (
                not magnitude
                and not float(values[0])
                and self._objective_scale is None
                and movable.any()
                and self._objective_connected
            ):
                raise ValueError(
                    "A connected objective with zero initial value and gradient needs an explicit objective scale."
                )
            problem.objective_scale = (
                self._objective_scale
                if self._objective_scale is not None
                else magnitude or abs(float(values[0])) or 1.0
            )
            _positive(problem.objective_scale, "Objective scale")
        for index, constraint in enumerate(specifications):
            dependent = constraint.dependent
            active = (
                ~(~dependent & (values[2 + 2 * index] == 0))
                if constraint.kind == "eq"
                else np.ones(constraint.template.size, dtype=np.bool_)
            )
            constraint.solver_rows = np.flatnonzero(active)
            if constraint.explicit_scale is not None:
                scale = constraint.explicit_scale.copy(deep=True)
            elif not self._automatic_scaling:
                scale = xr.ones_like(constraint.template)
            else:
                rows = np.max(
                    abs(values[3 + 2 * index][:, movable]), axis=1, initial=0.0
                )
                fallback = np.abs(np.asarray(values[2 + 2 * index], dtype=float))
                if movable.any() and np.any(dependent & (rows == 0) & (fallback == 0)):
                    raise ValueError(
                        "A connected constraint with zero initial residual and Jacobian needs an explicit scale."
                    )
                rows = np.where(rows > 0, rows, np.where(fallback > 0, fallback, 1.0))
                scale = constraint.template.copy(
                    data=rows.reshape(constraint.template.shape)
                )
                _positive(scale.values, "Constraint scale")
            problem.constraint_scales.append(scale)
        return problem
