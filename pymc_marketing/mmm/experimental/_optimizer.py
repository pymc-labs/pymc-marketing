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
"""Original-unit, term-first optimization with fixed numerical preconditioning."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
import pytensor
import pytensor.tensor as pt
import xarray as xr
from pytensor.gradient import jacobian
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting import rewrite_graph
from pytensor.graph.rewriting.basic import in2out, node_rewriter
from pytensor.graph.traversal import ancestors
from pytensor.tensor.subtensor import AdvancedIncSubtensor, indices_from_subtensor
from pytensor.tensor.variable import TensorConstant
from pytensor.xtensor.type import as_xtensor
from scipy.optimize import Bounds, OptimizeResult, minimize

from pymc_marketing.mmm.experimental._data import _align_labels
from pymc_marketing.mmm.experimental._gam import GAM
from pymc_marketing.mmm.experimental._graph import Data
from pymc_marketing.mmm.experimental._optimizer_terms import (
    _bind_terms,
    _data_dependencies,
)
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
        Decision arrays with their original units, dimensions, and coordinate labels.
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


@dataclass
class _Problem:
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
        values = self.function(z)
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
            *(allocation[name].values for name in self.layout.names)
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


def _build_problem(
    *,
    model: GAM,
    terms: _Term | Mapping[str, _Term],
    inputs: Sequence[Data],
    data: xr.Dataset,
    objective: _Callback,
    bounds: Mapping[str, tuple[_Array, _Array]] | None = None,
    constraints: Sequence[Mapping[str, Any]] = (),
    scaling: str | Mapping[str, Any] | None = "auto",
    history: xr.Dataset | None = None,
) -> _Problem:
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
    if not isinstance(inputs, Sequence) or any(
        not isinstance(term, Data) for term in inputs
    ):
        raise TypeError("inputs must be a sequence of Data terms.")
    names = [term.var_name for term in inputs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate inputs: {names}.")
    required = _data_dependencies(terms, model)
    if set(names) != required:
        raise ValueError(
            f"inputs must be exactly the Data dependencies: required {sorted(required)}, declared {sorted(names)}."
        )
    if not names:
        raise ValueError(
            "Selected terms must depend on at least one Data input to optimize."
        )
    if not isinstance(data, xr.Dataset):
        raise TypeError("data must be a labeled xarray.Dataset.")
    if missing := set(names) - set(data.data_vars):
        raise ValueError(f"data is missing selected inputs {sorted(missing)}.")
    selected_data = xr.Dataset(
        {name: data[name] for name in sorted(names)}, coords=data.coords
    )
    evaluator = _bind_terms(model, terms, selected_data, history=history)
    layout = _Layout(evaluator.templates)
    initial = {name: array.astype(float) for name, array in evaluator.templates.items()}
    bounds = bounds or {}
    if unknown := set(bounds) - set(names):
        raise ValueError(f"Bounds for unknown inputs {sorted(unknown)}.")
    lower, upper = {}, {}
    for name, template in layout.templates.items():
        pair = bounds.get(name, (-np.inf, np.inf))
        if len(pair) != 2:
            raise ValueError(f"Bounds for {name!r} must be a (lower, upper) pair.")
        lower[name] = _align(pair[0], template, f"lower bound for {name!r}")
        upper[name] = _align(pair[1], template, f"upper bound for {name!r}")
        low, high = lower[name], upper[name]
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
        if not bool(np.isfinite(initial[name]).all()) or bool(
            (initial[name] < low).any() or (initial[name] > high).any()
        ):
            raise ValueError(
                f"Initial input {name!r} must be finite and within its bounds."
            )
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
    if unknown := set(explicit) - set(names):
        raise ValueError(f"Scales for unknown inputs {sorted(unknown)}.")
    centers, scales = {}, {}
    z_lower: dict[str, xr.DataArray] = {}
    z_upper: dict[str, xr.DataArray] = {}
    for name, template in layout.templates.items():
        low, high = lower[name], upper[name]
        fixed = low == high
        if scaling is None:
            center, scale = xr.zeros_like(template), xr.ones_like(template)
        elif name in explicit:
            center, scale = (
                xr.zeros_like(template),
                _align(explicit[name], template, f"decision scale for {name!r}"),
            )
        else:
            ranged = np.isfinite(low) & np.isfinite(high) & (high > low)
            with np.errstate(over="ignore", invalid="ignore"):
                scale = xr.where(ranged, high - low, abs(initial[name]))
            # A fixed zero has no units-sensitive movement, and is excluded from derivative scaling.
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
    original = {
        name: pt.tensor(
            name=f"original_{name}", shape=layout.templates[name].shape, dtype="float64"
        )
        for name in layout.names
    }
    u = {
        name: as_xtensor(value, dims=layout.templates[name].dims)
        for name, value in original.items()
    }
    z = pt.vector("decisions", shape=(layout.size,), dtype="float64")
    # Preserve each input's shape across fixed projections so scalar row analysis
    # can prove that fixed tensor entries are constants, without crossing a reshape.
    decoded = {
        original[name]: pt.where(
            (lower[name] == upper[name]).values,
            lower[name].values,
            centers[name].values + scales[name].values * block.values,
        )
        for name, block in layout.symbolic(z).items()
    }
    utility = as_xtensor(objective(evaluator, u))
    if utility.type.ndim != 0:
        raise ValueError(
            f"objective must be scalar; explicitly reduce dimensions {utility.type.dims}."
        )
    report_graphs = [utility.values]
    tensor = rewrite_graph(
        clone_replace(utility.values, replace=decoded),
        include=(*_LOWER, "specialize"),
    )
    outputs = [tensor, pt.grad(tensor, z, disconnected_inputs="ignore")]
    specifications = []
    for spec in constraints:
        if spec.get("type") not in {"eq", "ineq"} or not callable(spec.get("fun")):
            raise ValueError(
                "Each constraint needs type 'eq' or 'ineq' and a callable fun."
            )
        residual = as_xtensor(spec["fun"](evaluator, u))
        dims = tuple(residual.type.dims)
        if unknown := set(dims) - set(evaluator.coords):
            raise ValueError(
                f"Constraint dimensions {sorted(unknown)} are not labeled."
            )
        coords = {dim: evaluator.coords[dim] for dim in dims}
        template = xr.DataArray(
            np.empty(tuple(len(coords[dim]) for dim in dims)), dims=dims, coords=coords
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
        specifications.append(
            _Constraint(spec["type"], template, spec.get("scale"), expression)
        )
    for constraint in specifications:
        if any(size is None for size in constraint.expression.type.shape):
            constraint.shape_index = len(outputs)
            outputs.append(constraint.expression.shape)
    function = pytensor.function([z], outputs, on_unused_input="ignore")
    report_function = pytensor.function(
        list(original.values()),
        rewrite_graph(report_graphs, include=("lower_xtensor",)),
        mode=pytensor.compile.mode.Mode(linker="py", optimizer=None),
        on_unused_input="ignore",
    )
    problem = _Problem(
        evaluator=evaluator,
        layout=layout,
        initial=initial,
        centers=centers,
        scales=scales,
        lower=lower,
        upper=upper,
        z_lower=layout.pack(z_lower),
        z_upper=layout.pack(z_upper),
        function=function,
        constraints=specifications,
        report_function=report_function,
    )
    values = problem.raw(problem.to_z(initial))
    movable = problem.z_lower < problem.z_upper
    if scaling is not None:
        magnitude = float(np.max(abs(values[1][movable]), initial=0.0))
        if (
            not magnitude
            and not float(values[0])
            and "objective" not in config
            and movable.any()
            and z in ancestors([tensor])
        ):
            raise ValueError(
                "A connected objective with zero initial value and gradient needs an explicit objective scale."
            )
        scale = config.get("objective", magnitude or abs(float(values[0])) or 1.0)
        if np.ndim(scale) != 0:
            raise ValueError("Objective scale must be scalar.")
        _positive(scale, "Objective scale")
        problem.objective_scale = float(scale)
    for index, constraint in enumerate(specifications):
        dependent = np.any(values[3 + 2 * index] != 0, axis=1)
        if np.any(~dependent):
            # Only the dependency-proof graph is normalized; reports retain every written row.
            projected = rewrite_graph(
                constraint.expression,
                include=(),
                custom_rewrite=in2out(_rewrite_full_slice_scatter),
                clone=True,
            )
            for row in np.flatnonzero(~dependent):
                expression = rewrite_graph(
                    projected[np.unravel_index(row, constraint.template.shape)],
                    include=(*_LOWER, "specialize"),
                    clone=True,
                )
                dependent[row] = z in ancestors([expression])
        # A zero Jacobian at the start is not evidence of a constant row.
        active = (
            ~(~dependent & (values[2 + 2 * index] == 0))
            if constraint.kind == "eq"
            else np.ones(constraint.template.size, dtype=np.bool_)
        )
        constraint.solver_rows = np.flatnonzero(active)
        if constraint.explicit_scale is not None:
            scale = _align(
                constraint.explicit_scale, constraint.template, "Constraint scale"
            )
            _positive(scale.values, "Constraint scale")
        elif scaling is None:
            scale = xr.ones_like(constraint.template)
        else:
            rows = np.max(abs(values[3 + 2 * index][:, movable]), axis=1, initial=0.0)
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


def optimize(
    *,
    model: GAM,
    terms: _Term | Mapping[str, _Term],
    inputs: Sequence[Data],
    data: xr.Dataset,
    objective: _Callback,
    bounds: Mapping[str, tuple[_Array, _Array]] | None = None,
    constraints: Sequence[Mapping[str, Any]] = (),
    scaling: str | Mapping[str, Any] | None = "auto",
    history: xr.Dataset | None = None,
    options: Mapping[str, Any] | None = None,
    feasibility_tol: float = 1e-6,
) -> OptimizationResult:
    """Maximize a user-written scalar objective of fitted deterministic terms.

    Parameters
    ----------
    model : GAM
        Fitted source of parameter bindings and joint posterior draws, never mutated.
    terms : term or mapping of str to term
        One deterministic expression (available as ``evaluate(u)['term']``), or
        named expressions evaluated separately. Existing fitted parameter identities
        must be retained; new deterministic wrappers and reductions are allowed.
        Stochastic ``Equation`` targets are not expected outcomes and are rejected.
    inputs : sequence of Data
        Exactly every data dependency of the selected expressions, without duplicates.
        Shared inputs are optimized once; unrelated equations are not rebuilt.
    data : xarray.Dataset
        Initial decision arrays in original units with labeled dimensions. Non-date
        labels must match fitting. Dimension and declaration order do not affect packing.
    objective : callable
        ``objective(evaluate, u)`` returns a scalar symbolic value to maximize.
        ``u`` maps input names to original-unit named PyTensor arrays; ``evaluate(u)``
        returns each selected term, retaining joint posterior ``sample`` draws.
        The user explicitly writes every reduction, trade-off, and unit conversion.
        ``evaluate.constant(DataArray)`` aligns labeled constants to the scenario.
    bounds : mapping, optional
        Input names to ``(lower, upper)`` in original units; each endpoint is a scalar
        or exactly labeled DataArray over a subset of input dimensions. Omitted bounds
        are infinite. Equal bounds hold individual entries fixed.
    constraints : sequence of mappings, default ()
        Each has ``type`` (``'eq'`` or ``'ineq'``) and ``fun(evaluate, u)`` returning
        original-unit residuals. Inequalities mean residual >= 0. Vector dimensions
        must retain their full scenario labels; reduce sliced windows explicitly.
        Optional ``scale`` is a positive scalar or labeled DataArray divisor.
    scaling : 'auto', mapping, or None, default 'auto'
        Fixed internal preconditioning uses finite bound ranges, then initial magnitudes.
        Unbounded zero starts need an explicit decision scale.
        A mapping accepts ``decisions={name: scalar_or_DataArray}`` and ``objective=positive_scalar``.
        Automatic divisors use initial derivatives with respect to movable scaled decisions.
        Zero derivatives fall back to nonzero initial value magnitudes.
        Connected expressions with zero initial value and derivatives need an explicit scale.
        Structurally constant zeros use one.
        ``None`` disables automatic scaling; explicit constraint scales still apply.
    history : xarray.Dataset, optional
        Explicit measured, fixed history for date-indexed inputs, immediately before scenario dates.
        Temporal history and scenario rows must follow the fitted cadence without gaps.
        Historical rows are never optimized or returned.
        Select date-indexed terms and write scoring windows in callbacks.
        Date-reduced history targets and static decisions upstream of temporal terms are rejected.
        To score carryover, explicitly add tail rows fixed by equal bounds.
    options : mapping, optional
        SLSQP options, overriding ``maxiter=500`` and ``ftol=1e-9`` in scaled coordinates.
    feasibility_tol : float, default 1e-6
        Positive tolerance on decoded residuals divided by their reported scales.
        Separate from solver stopping tolerance; success without feasibility warns.

    Returns
    -------
    OptimizationResult
        Original-unit allocation, objective, labeled residuals, fixed scales, and
        the untouched SciPy diagnostics. Failure is returned rather than hidden.

    Notes
    -----
    Terms and derivatives are compiled once, with no posterior predictive sampling
    inside the solver. Fixed scaling preserves the mathematical problem, not a
    guarantee of unit-independent convergence for every nonconvex or flat objective.
    This interface is experimental and supports SLSQP only.

    Examples
    --------
    .. code-block:: python

        result = optimize(
            model=gam,
            terms={"sales": media},
            inputs=[spend],
            data=initial,
            objective=lambda evaluate, u: (
                evaluate(u)["sales"].sum("date").sum("channel").mean("sample")
            ),
            bounds={"spend": (0.0, caps)},
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda evaluate, u: (
                        u["spend"].sum("date").sum("channel") - budget
                    ),
                }
            ],
        )
        # result.allocation and result.constraints remain in original units.
    """
    _positive(feasibility_tol, "feasibility_tol", scalar=True)
    return _build_problem(
        model=model,
        terms=terms,
        inputs=inputs,
        data=data,
        objective=objective,
        bounds=bounds,
        constraints=constraints,
        scaling=scaling,
        history=history,
    ).solve(options=options, feasibility_tol=feasibility_tol)
