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
"""Bind selected fitted terms to deterministic, labeled optimization inputs."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from functools import cache
from numbers import Integral
from typing import Any, cast

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import xarray as xr
from pymc.pytensorf import rvs_in_graph
from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant, Variable, clone_get_equiv
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting import rewrite_graph
from pytensor.graph.traversal import ancestors, truncated_graph_inputs
from pytensor.tensor.basic import Nonzero
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.shape import Shape, Shape_i, SpecifyShape
from pytensor.tensor.type_other import MakeSlice
from pytensor.xtensor.basic import Rename, TensorFromXTensor, XTensorFromTensor
from pytensor.xtensor.indexing import Index
from pytensor.xtensor.reduction import XReduce
from pytensor.xtensor.shape import Transpose
from pytensor.xtensor.type import XTensorSharedVariable, XTensorVariable, as_xtensor
from pytensor.xtensor.vectorization import XBlockwise, XElemwise

from pymc_marketing.experimental.terms._data import (
    _align_labels,
    _dates,
    validate_dataset,
)
from pymc_marketing.experimental.terms._graph import (
    BuildContext,
    Data,
    Equation,
    walk,
)
from pymc_marketing.experimental.terms._scm import SCM
from pymc_marketing.experimental.terms._terms import MediaTransform
from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
    BinomialAdstock,
    DelayedAdstock,
    GeometricAdstock,
    NoAdstock,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
)
from pymc_marketing.pytensor_utils import (
    _posterior_sample_major,
    extract_response_distribution,
)
from pymc_marketing.terms import (
    Dot,
    ModelTerm,
    Parameter,
    Ref,
)

_LOWER = ("lower_xtensor", "canonicalize", "stabilize")


def _input_provenance(
    context: BuildContext, declared: Collection[str]
) -> dict[Variable, str]:
    """Keep authoritative container ownership before inferring declared shared aliases."""
    model = context.model
    result = {model[raw]: raw for raw in context.data_variables}
    for variable, raw in context._registered_inputs.items():
        result.setdefault(variable, raw)
    for raw in declared:
        if raw in model.named_vars and isinstance(model[raw], XTensorSharedVariable):
            result.setdefault(model[raw], raw)
    return result


def _fitted_inputs(scm: SCM) -> dict[Variable, str]:
    """Recover fitted raw input provenance, including shared-term data aliases."""
    return _input_provenance(
        cast(BuildContext, scm._context),
        _data_dependencies({"equations": scm.equations}),
    )


def _fitted_graphs(terms: Mapping[str, Any], scm: SCM) -> list[Variable]:
    """Resolve fitted identities and explicit references without rebuilding equations."""
    context = cast(BuildContext, scm._context)
    result = []
    for node in walk(
        terms, stop=lambda node: id(node) in context.variables or isinstance(node, Ref)
    ):
        if id(node) in context.variables:
            value = context.variables[id(node)]
        elif isinstance(node, Ref):
            if node.name not in context.model.named_vars:
                raise ValueError(
                    f"Ref({node.name!r}) must reference a variable in the fitted model."
                )
            value = context.model[node.name]
        else:
            continue
        if isinstance(value, Variable):
            result.append(value)
    return result


def _data_dependencies(terms: Mapping[str, Any], scm: SCM | None = None) -> set[str]:
    """Find every declared input and, when fitted, inputs hidden behind named references."""
    context = None
    if scm is not None:
        scm._check_fitted()
        context = cast(BuildContext, scm._context)
    required: set[str] = set()
    for node in walk(
        terms,
        stop=lambda node: (
            context is not None
            and (id(node) in context.variables or isinstance(node, Ref))
        ),
    ):
        # A fitted boundary reads its actual graph, not the prior or lifecycle recipe.
        if context is not None and (
            id(node) in context.variables or isinstance(node, Ref)
        ):
            continue
        if isinstance(node, Data):
            required.add(node.var_name)
        elif isinstance(node, ModelTerm):
            required.update(node.data_vars)
    if scm is not None and context is not None:
        inputs = _fitted_inputs(scm)
        blockers = set(context.model.free_RVs) | set(context.model.observed_RVs)
        required.update(
            inputs[node]
            for node in ancestors(_fitted_graphs(terms, scm), blockers=blockers)
            if node in inputs
        )
    return required


def _scenario_data(
    scm: SCM, required: set[str], data: xr.Dataset
) -> tuple[xr.Dataset, dict[str, xr.DataArray]]:
    """Align scenario inputs, retaining fitted dimension and nondate label order."""
    if not isinstance(data, xr.Dataset):
        raise TypeError("Optimizer data must be an xarray.Dataset.")
    missing = required.difference(data.data_vars)
    if missing:
        raise ValueError(f"Required optimizer data are missing: {sorted(missing)}.")
    data = scm._prediction_data(
        xr.Dataset({name: data[name] for name in sorted(required)}, coords=data.coords)
    )
    training = cast(xr.Dataset, scm._training_data)
    templates = {}
    for name in sorted(required):
        value = data[name]
        dims = (
            tuple(training[name].dims)
            if name in training
            else tuple(sorted(value.dims))
        )
        if set(value.dims) != set(dims):
            raise ValueError(
                f"Input {name!r} must have fitted dimensions {dims}, got {value.dims}."
            )
        if "sample" in dims:
            raise ValueError(
                "Input dimension 'sample' is reserved for posterior draws."
            )
        templates[name] = BuildContext._canonical_input(value.transpose(*dims), name)
    return xr.Dataset(templates, coords=data.coords), templates


def _fitted_cadence(scm: SCM) -> str | pd.Timedelta:
    """Infer a regular cadence from the fitted dates, including a two-date fit."""
    training = cast(xr.Dataset, scm._training_data)
    if "date" not in training.dims or training.sizes["date"] < 2:
        raise ValueError(
            "Temporal scenarios require at least two training dates to infer the fitted cadence."
        )
    dates = _dates(training.get_index("date")).as_unit("us")
    frequency = pd.infer_freq(dates) if len(dates) >= 3 else dates[1] - dates[0]
    if frequency is None:
        raise ValueError("Temporal scenarios require regularly spaced training dates.")
    return frequency


def _check_cadence(scm: SCM, data: xr.Dataset, frequency: str | pd.Timedelta) -> None:
    """Require consecutive scenario rows in the fitted cadence's original phase."""
    if "date" not in data.dims:
        raise ValueError("Temporal scenarios require labeled date coordinates.")
    dates = _dates(data.get_index("date")).as_unit("us")
    expected = pd.date_range(dates[0], periods=len(dates), freq=frequency).as_unit("us")
    if not dates.equals(expected):
        raise ValueError(
            "Temporal history, decided, and carryover dates must be consecutive at the fitted cadence."
        )
    training = cast(xr.Dataset, scm._training_data)
    anchor = _dates(training.get_index("date")).as_unit("us")[0]
    start, end = sorted((anchor, dates[0]))
    phase = pd.date_range(start, end, freq=frequency).as_unit("us")
    if not len(phase) or phase[0] != start or phase[-1] != end:
        raise ValueError("Scenario dates must match the fitted cadence's phase.")


def _append_carryover(
    scm: SCM,
    scenario: xr.Dataset,
    templates: Mapping[str, xr.DataArray],
    carryover: int | xr.Dataset,
) -> tuple[xr.Dataset, dict[str, XTensorVariable]]:
    """Append owned fixed tail rows without assigning roles to dated input names."""
    dated = {name for name, value in templates.items() if "date" in value.dims}
    if isinstance(carryover, xr.Dataset):
        missing = dated.difference(carryover.data_vars)
        extra = set(carryover.data_vars).difference(dated)
        if missing or extra:
            raise ValueError(
                f"Carryover requires exactly the date-indexed inputs {sorted(dated)}; "
                f"missing {sorted(missing)}, extra {sorted(extra)}."
            )
        tail = validate_dataset(carryover)
        if "date" not in tail.dims:
            raise ValueError("Carryover data must contain labeled tail dates.")
        frequency = _fitted_cadence(scm)
    else:
        if (
            isinstance(carryover, bool)
            or not isinstance(carryover, Integral)
            or carryover < 0
        ):
            raise ValueError(
                "Carryover must be a nonnegative integer or an xarray.Dataset."
            )
        if not carryover:
            return scenario, {}
        frequency = _fitted_cadence(scm)
        if "date" not in scenario.dims:
            raise ValueError("Carryover requires labeled decided dates.")
        dates = pd.date_range(
            scenario.get_index("date")[-1], periods=int(carryover) + 1, freq=frequency
        )[1:].as_unit("us")
        tail = xr.Dataset(
            {
                name: xr.DataArray(
                    np.zeros(
                        tuple(
                            len(dates) if dim == "date" else value.sizes[dim]
                            for dim in value.dims
                        ),
                        dtype="float64",
                    ),
                    dims=value.dims,
                    coords={
                        dim: dates if dim == "date" else value.coords[dim]
                        for dim in value.dims
                    },
                )
                for name, value in templates.items()
                if name in dated
            },
            coords={"date": dates},
        )
    if "date" not in scenario.dims:
        raise ValueError("Carryover requires labeled decided dates.")
    if tail.get_index("date")[0] <= scenario.get_index("date")[-1]:
        raise ValueError("Carryover dates must strictly follow decided dates.")
    arrays = dict(templates)
    fixed = {}
    for name in sorted(dated):
        template = templates[name]
        value = tail[name]
        if set(value.dims) != set(template.dims):
            raise ValueError(
                f"Carryover input {name!r} must have dimensions {template.dims}."
            )
        value = _align_labels(
            value, template, dims=[dim for dim in template.dims if dim != "date"]
        ).transpose(*template.dims)
        value = BuildContext._canonical_input(value, name, copy=True)
        fixed[name] = as_xtensor(value)
        arrays[name] = xr.concat(
            [template, value],
            dim="date",
            coords="minimal",
            compat="override",
            join="exact",
        )
    coords = dict(scenario.coords)
    coords["date"] = np.concatenate(
        [scenario.coords["date"].values, tail.coords["date"].values]
    )
    combined = xr.Dataset(arrays, coords=coords)
    _check_cadence(scm, combined, frequency)
    return combined, fixed


def _prepend_history(
    scenario: xr.Dataset,
    templates: Mapping[str, xr.DataArray],
    history: xr.Dataset | None,
) -> tuple[xr.Dataset, dict[str, XTensorVariable], int]:
    """Prepend fixed history dates, adding values only to date-indexed inputs."""
    if history is None:
        return scenario, {}, 0
    if not isinstance(history, xr.Dataset):
        raise TypeError("History must be an xarray.Dataset.")
    dated = [name for name, value in templates.items() if "date" in value.dims]
    missing = set(dated).difference(history.data_vars)
    if missing:
        raise ValueError(f"History is missing date-indexed inputs: {sorted(missing)}.")
    history_variables = [*dated, "date"] if "date" in history.coords else dated
    history = validate_dataset(history[history_variables])
    if "date" not in history.dims:
        raise ValueError("History must contain labeled dates.")
    if history.get_index("date")[-1] >= scenario.get_index("date")[0]:
        raise ValueError("History dates must strictly precede scenario dates.")
    arrays = dict(templates)
    fixed = {}
    for name in dated:
        template = templates[name]
        value = history[name]
        if set(value.dims) != set(template.dims):
            raise ValueError(
                f"History input {name!r} must have dimensions {template.dims}."
            )
        value = _align_labels(
            value, template, dims=[dim for dim in template.dims if dim != "date"]
        ).transpose(*template.dims)
        # Own the historical values so caller mutations cannot change the bound history.
        value = BuildContext._canonical_input(value, name, copy=True)
        fixed[name] = as_xtensor(value)
        arrays[name] = xr.concat(
            [value, template],
            dim="date",
            coords="minimal",
            compat="override",
            join="exact",
        )
    coords = dict(scenario.coords)
    coords["date"] = np.concatenate(
        [history.coords["date"].values, scenario.coords["date"].values]
    )
    return xr.Dataset(arrays, coords=coords), fixed, history.sizes["date"]


def _dated_constant_replacements(
    context: BuildContext,
    built: Mapping[str, XTensorVariable],
    blockers: set[Variable],
) -> dict[Variable, Variable]:
    """Validate selected deterministic constant labels, selecting covered scenario dates."""
    selected = set(ancestors(list(built.values()), blockers=blockers))
    needed = {
        variable: value
        for variable, value in context._labeled_constants.items()
        if variable in selected
    }
    if not needed:
        return {}
    reference = xr.Dataset(
        coords=_coordinates(
            {
                dim: pd.Index(labels, name=dim, tupleize_cols=False)
                for dim, labels in context.model.coords.items()
                if labels is not None
            }
        )
    )
    replacements = {}
    for variable, value in needed.items():
        for dim in value.dims:
            if dim not in value.coords or value.coords[dim].dims != (dim,):
                raise ValueError(
                    f"Deterministic constant dimension {dim!r} must label its own dimension."
                )
        value = _align_labels(
            value, reference, dims=[dim for dim in value.dims if dim != "date"]
        )
        if "date" in value.dims:
            replacements[variable] = as_xtensor(context._select_constant_dates(value))
    return replacements


def _constant_index_value(index: Variable) -> Any:
    """Read index constants through only the known conversion and Boolean nonzero ops."""
    if isinstance(index, Constant):
        return index.data
    owner = index.owner
    if owner is None:
        return None
    if isinstance(owner.op, MakeSlice) and all(
        isinstance(item, Constant) for item in owner.inputs
    ):
        return slice(*(cast(Constant, item).data for item in owner.inputs))
    if isinstance(owner.op, (XTensorFromTensor, TensorFromXTensor)):
        return _constant_index_value(owner.inputs[0])
    if isinstance(owner.op, Nonzero):
        value = _constant_index_value(owner.inputs[0])
        if value is not None:
            value = np.asarray(value)
            if value.ndim == 1 and value.dtype.kind == "b":
                return np.flatnonzero(value)
    return None


def _full_date_index(index: Variable, length: int) -> bool:
    """Prove identity using the index's Boolean, integer, or slice semantics."""
    value = _constant_index_value(index)
    if isinstance(value, slice):
        return value.indices(length) == (0, length, 1)
    if value is None:
        return False
    value = np.asarray(value)
    if value.shape != (length,):
        return False
    if value.dtype.kind == "b":
        return bool(value.all())
    positions = np.arange(length)
    if value.dtype.kind == "i":
        return bool(np.all((value == positions) | (value == positions - length)))
    if value.dtype.kind == "u":
        return np.array_equal(value, positions)
    return False


def _fixed_shape_proof(boundaries: Collection[Variable]) -> Callable[[Variable], bool]:
    """Share the conservative shape-only exemption across both history proofs."""

    @cache
    def has_fixed_shape(value: Variable) -> bool:
        if value in boundaries:
            return True
        owner = value.owner
        if owner is not None and isinstance(owner.op, Index):
            original, *indexers = owner.inputs
            return has_fixed_shape(original) and all(
                _constant_index_value(index) is not None for index in indexers
            )
        shape = getattr(value.type, "shape", None)
        if shape is not None and all(size is not None for size in shape):
            return True
        if value.owner is None or not isinstance(
            value.owner.op,
            (
                DeepCopyOp,
                ViewOp,
                Elemwise,
                DimShuffle,
                XElemwise,
                XReduce,
                Rename,
                Transpose,
                TensorFromXTensor,
                XTensorFromTensor,
            ),
        ):
            return False
        return all(has_fixed_shape(item) for item in value.owner.inputs)

    return has_fixed_shape


def _tensor_date_axis(
    value: Variable, axis_of: Callable[[Variable], int | None]
) -> int | None:
    """Infer one operation's date axis without caching or proving source provenance."""
    if isinstance(value, XTensorVariable):
        return value.dims.index("date") if "date" in value.dims else None
    owner = value.owner
    if owner is None:
        return None
    op = owner.op
    if isinstance(op, (TensorFromXTensor, SpecifyShape, ViewOp, DeepCopyOp)):
        return axis_of(owner.inputs[0])
    if isinstance(op, Elemwise):
        axes = {
            axis + value.ndim - item.ndim
            for item in owner.inputs
            if (axis := axis_of(item)) is not None
        }
        return axes.pop() if len(axes) == 1 else None
    if isinstance(op, DimShuffle):
        axis = axis_of(owner.inputs[0])
        if axis is not None and axis in op.new_order:
            return op.new_order.index(axis)
        return None
    if isinstance(op, CAReduce):
        original = owner.inputs[0]
        axis = axis_of(original)
        if axis is None or op.axis is None:
            return None
        reduced = {item % original.ndim for item in op.axis}
        return None if axis in reduced else axis - sum(item < axis for item in reduced)
    return None


def _date_axis_proof(
    sources: Collection[Variable],
    blockers: Collection[Variable],
    boundaries: Collection[Variable],
    length: int,
    *,
    adstock_inputs: Mapping[Variable, Variable] | None = None,
    history: bool = False,
) -> tuple[Callable[[Variable], bool], Callable[[Variable], bool]]:
    """Track full ordered date axes, optionally forbidding date-reduced history factors."""
    has_fixed_shape = _fixed_shape_proof(boundaries)

    @cache
    def depends_on_dates(value: Variable) -> bool:
        if value in sources:
            return True
        owner = value.owner
        if value in blockers or owner is None:
            return False
        # Only value-independent shapes cut dated ancestry.
        if isinstance(owner.op, (Shape, Shape_i)) and has_fixed_shape(owner.inputs[0]):
            return False
        return any(depends_on_dates(item) for item in owner.inputs)

    @cache
    def tensor_date_axis(value: Variable) -> int | None:
        return _tensor_date_axis(value, tensor_date_axis)

    @cache
    def tensor_preserves_dates(value: Variable, axis: int) -> bool:
        if tensor_date_axis(value) != axis or value.owner is None:
            return False
        owner = value.owner
        op = owner.op
        if isinstance(op, TensorFromXTensor):
            return preserves_dates(owner.inputs[0])
        if isinstance(op, (DimShuffle, CAReduce, SpecifyShape, ViewOp, DeepCopyOp)):
            original = owner.inputs[0]
            original_axis = tensor_date_axis(original)
            return original_axis is not None and tensor_preserves_dates(
                original, original_axis
            )
        if isinstance(op, Elemwise):
            ordered = False
            for item in owner.inputs:
                original_axis = axis - value.ndim + item.ndim
                if original_axis < 0:
                    continue
                if tensor_preserves_dates(item, original_axis):
                    ordered = True
                elif getattr(item.type, "shape", ())[original_axis] != 1:
                    return False
            return ordered
        return False

    @cache
    def preserves_dates(value: Variable) -> bool:
        if value in sources:
            return True
        if not isinstance(value, XTensorVariable) or "date" not in value.dims:
            return False
        if adstock_inputs is not None and value in adstock_inputs:
            return preserves_dates(adstock_inputs[value])
        owner = value.owner
        if owner is None:
            return False
        if not history and isinstance(owner.op, XTensorFromTensor):
            return tensor_preserves_dates(owner.inputs[0], value.dims.index("date"))
        dated = [
            item
            for item in owner.inputs
            if isinstance(item, XTensorVariable) and "date" in item.dims
        ]
        if not dated:
            return False
        op = owner.op
        if isinstance(op, Index):
            original, *indexers = owner.inputs
            original = cast(XTensorVariable, original)
            for dim, index in zip(original.dims, indexers, strict=False):
                if dim == "date":
                    if not _full_date_index(index, length):
                        return False
                elif "date" in getattr(index, "dims", ()) or depends_on_dates(index):
                    return False
            return preserves_dates(original)
        if isinstance(op, XReduce):
            safe = "date" not in op.dims
        elif isinstance(op, Rename):
            original = cast(XTensorVariable, owner.inputs[0])
            safe = original.dims.index("date") == value.dims.index("date")
        elif isinstance(op, XBlockwise):
            safe = all("date" not in dims for group in op.core_dims for dims in group)
        else:
            safe = isinstance(op, (XElemwise, Transpose, ViewOp, DeepCopyOp))
        return (
            safe
            and all(preserves_dates(item) for item in dated)
            and (
                not history
                or all(
                    not depends_on_dates(item)
                    for item in owner.inputs
                    if not isinstance(item, XTensorVariable) or "date" not in item.dims
                )
            )
        )

    return depends_on_dates, preserves_dates


def _check_history_dates(
    scm: SCM,
    terms: Mapping[str, Any],
    context: BuildContext,
    built: Mapping[str, XTensorVariable],
    inputs: Mapping[Variable, str],
    templates: Mapping[str, xr.DataArray],
    dated_constants: Mapping[Variable, Variable],
    *,
    history: bool = True,
) -> None:
    """Prove ordered combined axes, retaining stricter measured-history semantics."""
    fitted_model = cast(BuildContext, scm._context).model
    blockers = set(fitted_model.free_RVs) | set(fitted_model.observed_RVs)
    sources = {
        value
        for value, raw in inputs.items()
        if raw in templates and "date" in templates[raw].dims
    }
    sources.update(
        rv
        for rv in fitted_model.free_RVs
        if "date"
        in getattr(rv, "dims", fitted_model.named_vars_to_dims.get(rv.name, ()))
    )
    sources.update(dated_constants)
    fourier_outputs = {
        value
        for _, term in context._fourier_terms.values()
        if isinstance(value := context.variables.get(id(term)), XTensorVariable)
    }
    sources.update(fourier_outputs)
    adstock_inputs: dict[Variable, Variable] = {}
    for node in walk([terms, scm.equations]):
        if not isinstance(node, MediaTransform) or type(node.transformation) not in (
            BinomialAdstock,
            DelayedAdstock,
            GeometricAdstock,
            WeibullCDFAdstock,
            WeibullPDFAdstock,
        ):
            continue
        value = context.variables.get(id(node))
        media = context.variables.get(id(node.media), node.media)
        if isinstance(value, XTensorVariable) and isinstance(media, XTensorVariable):
            # Built-in adstock retains the input's date axis after its internal padding/convolution.
            adstock_inputs[value] = media
        elif (
            isinstance(value, XTensorVariable)
            and isinstance(media, xr.DataArray)
            and "date" in media.dims
        ):
            # Literal dated media have already passed immutable constant-label validation.
            sources.add(value)

    depends_on_dates, preserves_dates = _date_axis_proof(
        sources,
        blockers,
        set(inputs) | blockers | fourier_outputs,
        context.ds.sizes.get("date", 0),
        adstock_inputs=adstock_inputs,
        history=history,
    )

    for name, output in built.items():
        if "date" not in output.dims:
            if history and depends_on_dates(output):
                raise ValueError(
                    f"Selected term {name!r} reduces date before history can be removed; "
                    "select a date-indexed term and reduce scenario dates in the callback instead."
                )
        elif not preserves_dates(output):
            if not history:
                raise ValueError(
                    f"Selected term {name!r} must retain the full date axis in its original order; "
                    "perform date slicing and scoring inside the callback instead."
                )
            raise ValueError(
                f"Selected term {name!r} must retain the full combined date axis in its original order "
                "without a dependency that reduces date before history can be removed; "
                "perform date slicing and scoring inside the callback instead."
            )


def _check_temporal_dates(
    scm: SCM,
    terms: Mapping[str, Any],
    context: BuildContext,
    built: Mapping[str, XTensorVariable],
) -> None:
    """Validate selected temporal cadence even when history and carryover are absent."""
    fitted = cast(BuildContext, scm._context)
    blockers = set(fitted.model.free_RVs) | set(fitted.model.observed_RVs)
    selected = set(ancestors(list(built.values()), blockers=blockers))
    temporal = False
    for node in walk([terms, scm.equations]):
        value = context.variables.get(id(node))
        if not isinstance(value, Variable) or value not in selected:
            continue
        if isinstance(node, MediaTransform) and isinstance(
            node.transformation, AdstockTransformation
        ):
            # Cadence is independent of causal history support, even for a one-row lag.
            temporal |= not isinstance(node.transformation, NoAdstock)
            continue
        lookback = getattr(node, "required_history", 0)
        if (
            isinstance(lookback, bool)
            or not isinstance(lookback, Integral)
            or lookback < 0
        ):
            raise ValueError("Term required_history must be a nonnegative integer.")
        temporal |= bool(lookback)
    if temporal:
        _check_cadence(scm, context.ds, _fitted_cadence(scm))


def _check_history_inputs(
    scm: SCM,
    terms: Mapping[str, Any],
    context: BuildContext,
    built: Mapping[str, XTensorVariable],
    templates: Mapping[str, xr.DataArray],
    dated_constants: Mapping[Variable, Variable],
) -> None:
    """Validate full date provenance and immutable temporal history."""
    fitted = cast(BuildContext, scm._context)
    blockers = set(fitted.model.free_RVs) | set(fitted.model.observed_RVs)
    selected = set(ancestors(list(built.values()), blockers=blockers))
    inputs = _fitted_inputs(scm)
    inputs.update(_input_provenance(context, templates))

    fourier_outputs = {
        value
        for _, term in context._fourier_terms.values()
        if isinstance(value := context.variables.get(id(term)), XTensorVariable)
    }
    has_fixed_shape = _fixed_shape_proof(set(inputs) | blockers | fourier_outputs)

    # A fixed input shape is harmless; a data-dependent cardinality is not.
    value_blockers = blockers | {
        value
        for value in selected
        if value.owner is not None
        and isinstance(value.owner.op, (Shape, Shape_i))
        and has_fixed_shape(value.owner.inputs[0])
    }
    _check_history_dates(scm, terms, context, built, inputs, templates, dated_constants)
    dated_inputs = {
        value
        for value, raw in inputs.items()
        if raw in templates and "date" in templates[raw].dims
    }

    @cache
    def depends_on_dated_decisions(value: Variable) -> bool:
        if value in dated_inputs:
            return True
        if value in value_blockers or value.owner is None:
            return False
        return any(depends_on_dated_decisions(item) for item in value.owner.inputs)

    for node in walk([terms, scm.equations]):
        value = context.variables.get(id(node))
        if not isinstance(value, Variable) or value not in selected:
            continue
        lookback = getattr(node, "required_history", 0)
        if not lookback:
            continue
        upstream_nodes = set(ancestors([value], blockers=value_blockers))
        upstream = {
            inputs[ancestor] for ancestor in upstream_nodes if ancestor in inputs
        }
        static = sorted(
            raw
            for raw in upstream
            if raw in templates and "date" not in templates[raw].dims
        )
        if static:
            raise ValueError(
                f"Static inputs {static} occur upstream of a history-dependent term; "
                "scenario decisions must not change measured history. "
                "Place static decision inputs downstream of temporal terms instead."
            )
        if any(
            isinstance(ancestor, XTensorVariable)
            and "date" not in ancestor.dims
            and depends_on_dated_decisions(ancestor)
            for ancestor in upstream_nodes
        ):
            raise ValueError(
                "A decision-derived intermediate reduces date upstream of a history-dependent term; "
                "scenario decisions must not change measured history. "
                "Place date reductions downstream of temporal terms or score them inside the callback instead."
            )


def _freeze_posterior_shapes(
    scm: SCM,
    context: BuildContext,
    built: dict[str, XTensorVariable],
) -> dict[str, XTensorVariable]:
    """Freeze proven shape metadata before adding a posterior sample axis."""
    fitted = cast(BuildContext, scm._context)
    free = set(fitted.model.free_RVs)
    blockers = free | set(fitted.model.observed_RVs)
    nodes = set(ancestors(list(built.values()), blockers=blockers))
    inputs = _fitted_inputs(scm)
    inputs.update(_input_provenance(context, context.data_variables))
    fourier = {
        value
        for _, term in context._fourier_terms.values()
        if isinstance(value := context.variables.get(id(term)), XTensorVariable)
    }
    fixed_shape = _fixed_shape_proof(set(inputs) | blockers | fourier)
    shapes = [
        node
        for node in nodes
        if node.owner is not None
        and isinstance(node.owner.op, (Shape, Shape_i))
        and fixed_shape(node.owner.inputs[0])
        and free.intersection(ancestors([node], blockers=blockers))
    ]
    if not shapes:
        return built
    shape_nodes = set(ancestors(shapes, blockers=blockers))
    replacements: dict[Variable, Variable] = {
        original: context.model[raw]
        for original, raw in inputs.items()
        if raw in context.data_variables
    }
    # These are value-free shape stand-ins, never substitute posterior values or priors.
    for value in shape_nodes.intersection(free | fourier):
        dims = getattr(
            value, "dims", fitted.model.named_vars_to_dims.get(value.name, ())
        )
        sizes = []
        for dim in dims:
            if dim in context.ds.sizes:
                sizes.append(context.ds.sizes[dim])
            else:
                labels = context.model.coords.get(dim)
                if labels is None:
                    raise ValueError(
                        f"Posterior shape metadata requires labeled coordinates for {dim!r}."
                    )
                sizes.append(len(labels))
        replacements[value] = as_xtensor(pt.zeros(sizes, dtype=value.dtype), dims=dims)
    rewritten = clone_replace(shapes, replace=replacements, rebuild_strict=False)
    values = pytensor.function([], rewritten)()
    outputs = list(built.values())
    boundaries = {
        **{
            node: pt.as_tensor_variable(value)
            for node, value in zip(shapes, values, strict=True)
        },
        **{rv: rv for rv in blockers},
    }
    memo = clone_get_equiv(
        list(truncated_graph_inputs(outputs, boundaries)),
        outputs,
        copy_inputs=False,
        copy_orphans=False,
        memo=boundaries,
        strict=False,
    )
    return {name: cast(XTensorVariable, memo[value]) for name, value in built.items()}


def _condition_outputs(
    scm: SCM,
    built: dict[str, XTensorVariable],
    data: xr.Dataset,
    shape_context: BuildContext,
) -> tuple[dict[str, XTensorVariable], pd.Index | None]:
    """Condition actual fitted RV identities, rejecting stochastic outcomes and foreign RVs."""
    context = cast(BuildContext, scm._context)
    model = context.model
    free = set(model.free_RVs)
    blockers = free | set(model.observed_RVs)
    nodes = set(ancestors(list(built.values()), blockers=blockers))
    equations = {model[name] for name in context.equation_names.values()}
    if nodes.intersection(equations):
        raise ValueError(
            "Optimizer targets and dependencies must be deterministic terms, not stochastic Equation outputs."
        )
    placeholders = clone_replace(
        list(built.values()), replace={rv: rv.clone() for rv in free}
    )
    if rvs_in_graph(placeholders):
        raise ValueError(
            "Selected terms contain stochastic variables not bound to this fitted model; "
            "use its actual fitted parameter objects, not new parameters with matching names."
        )
    built = _freeze_posterior_shapes(scm, shape_context, built)
    nodes = set(ancestors(list(built.values()), blockers=blockers))
    needed = nodes.intersection(free)
    if not needed:
        return built, None
    posterior = cast(xr.DataTree, scm.idata)["posterior"].to_dataset()
    indexers = {}
    for rv in needed:
        if rv.name not in posterior:
            raise ValueError(f"The fitted posterior is missing parameter {rv.name!r}.")
        value = posterior[rv.name]
        dims = getattr(rv, "dims", model.named_vars_to_dims.get(rv.name, ()))
        for dim in dims:
            if dim == "sample":
                raise ValueError(
                    "Parameter dimension 'sample' is reserved for posterior draws."
                )
            if dim not in data.coords:
                raise ValueError(
                    f"Posterior parameter {rv.name!r} requires scenario labels for {dim!r}."
                )
            labels = data.get_index(dim)
            given = value.get_index(dim)
            if dim == "date":
                matches = given.is_unique and labels.isin(given).all()
            else:
                matches = (
                    given.is_unique
                    and len(labels) == len(given)
                    and labels.isin(given).all()
                )
            if not matches:
                raise ValueError(
                    f"Posterior parameter {rv.name!r} has no fitted draws for the requested {dim!r} labels."
                )
            indexers[dim] = data.coords[dim]
    # This is a labeled, read-only view of the actual joint draws, not an assignment to the SCM.
    posterior = posterior[sorted(cast(str, rv.name) for rv in needed)].sel(indexers)
    idata = xr.DataTree.from_dict({"posterior": posterior})
    sampled = [
        name
        for name, value in built.items()
        if free.intersection(ancestors([value], blockers=free))
    ]
    conditioned = extract_response_distribution(
        model, idata, [built[name] for name in sampled]
    )
    result = dict(built)
    result.update(zip(sampled, cast(list[XTensorVariable], conditioned), strict=True))
    return result, _posterior_sample_major(idata).get_index("sample")


def _coordinates(coords: Mapping[str, pd.Index]) -> xr.Coordinates:
    """Construct labeled coordinates without implicit MultiIndex promotion."""
    result = xr.Coordinates(
        {
            dim: labels
            for dim, labels in coords.items()
            if not isinstance(labels, pd.MultiIndex)
        }
    )
    for dim, labels in coords.items():
        if isinstance(labels, pd.MultiIndex):
            result.update(xr.Coordinates.from_pandas_multiindex(labels, dim))
    return result


class _Evaluator:
    """Evaluate selected deterministic terms on original-unit scenario decisions."""

    def __init__(
        self,
        outputs: dict[str, XTensorVariable],
        containers: dict[str, XTensorSharedVariable],
        templates: dict[str, xr.DataArray],
        coords: dict[str, pd.Index],
        output_coords: dict[str, pd.Index],
        history: dict[str, XTensorVariable],
        carryover: dict[str, XTensorVariable],
    ) -> None:
        self.outputs = outputs
        self.containers = containers
        self.templates = templates
        self.coords = coords
        self.output_coords = output_coords
        self._history = history
        self._carryover = carryover
        self._full_date_sources: set[Variable] = set()
        self._numeric: Callable[..., list[np.ndarray]] | None = None

    def __call__(self, u: Mapping[str, XTensorVariable]) -> dict[str, XTensorVariable]:
        """Substitute decided inputs once, surrounding them with fixed history and tail."""
        missing = set(self.containers).difference(u)
        extra = set(u).difference(self.containers)
        if missing or extra:
            raise ValueError(
                f"evaluate requires exactly {sorted(self.containers)}; "
                f"missing {sorted(missing)}, extra {sorted(extra)}."
            )
        replacements = {}
        for name, container in self.containers.items():
            value = u[name]
            if not isinstance(value, XTensorVariable):
                raise TypeError(f"Input {name!r} must be an XTensorVariable.")
            if set(value.dims) != set(container.dims):
                raise ValueError(
                    f"Input {name!r} must have dimensions {container.dims}, got {value.dims}."
                )
            value = value.transpose(*container.dims)
            if name in self._history:
                value = ptx.concat([self._history[name], value], dim="date")
            if name in self._carryover:
                value = ptx.concat([value, self._carryover[name]], dim="date")
            replacements[container] = value
        values = clone_replace(
            list(self.outputs.values()), replace=replacements, rebuild_strict=False
        )
        outputs = {
            name: as_xtensor(
                pt.specify_shape(
                    value.values,
                    tuple(len(self.output_coords[dim]) for dim in value.dims),
                ),
                dims=value.dims,
            )
            for name, value in zip(
                self.outputs, cast(list[XTensorVariable], values), strict=True
            )
        }
        self._full_date_sources.update(
            value for value in outputs.values() if "date" in value.dims
        )
        return outputs

    def constant(self, value: xr.DataArray) -> XTensorVariable:
        """Align a labeled original-unit constant to the evaluator's coordinates.

        Parameters
        ----------
        value : xarray.DataArray
            Scalar or labeled array over decided or full output dates and posterior dimensions.

        Returns
        -------
        XTensorVariable
            Constant with each dimension's labels in evaluator order.
        """
        if not isinstance(value, xr.DataArray):
            raise TypeError("constant expects a labeled xarray.DataArray.")
        for dim in value.dims:
            if dim not in value.coords or value.coords[dim].dims != (dim,):
                raise ValueError(
                    f"Constant dimension {dim!r} must label its own dimension."
                )
        coords = self.coords
        if "date" in value.dims:
            labels = _dates(value.get_index("date")).as_unit("us")
            for candidate in (self.coords, self.output_coords):
                if "date" not in candidate:
                    continue
                target = _dates(candidate["date"]).as_unit("us")
                if (
                    labels.is_unique
                    and len(labels) == len(target)
                    and labels.isin(target).all()
                ):
                    coords = candidate
                    break
            else:
                raise ValueError(
                    "Constant 'date' labels must match exactly the decided dates or the full "
                    "decided-plus-carryover output dates."
                )
        reference = xr.Dataset(coords=_coordinates(coords))
        result = as_xtensor(_align_labels(value, reference).astype(float))
        if "date" in result.dims and len(coords["date"]) == len(
            self.output_coords["date"]
        ):
            self._full_date_sources.add(result)
        return result

    def decided(self, value: XTensorVariable) -> XTensorVariable:
        """Select decided rows only after proving a full ordered output date axis."""
        if not isinstance(value, XTensorVariable):
            raise TypeError("decided expects an XTensorVariable output.")
        if "date" not in value.dims:
            return value
        _, preserves_dates = _date_axis_proof(
            self._full_date_sources,
            self._full_date_sources,
            self._full_date_sources,
            len(self.output_coords["date"]),
        )
        if not preserves_dates(value):
            raise ValueError(
                "decided expects the full output date axis in its original order; "
                "call evaluate.decided on the full ordered output before slicing or permuting dates."
            )
        return value.isel(date=slice(0, len(self.coords["date"])))

    def validate_plan(
        self, plan: xr.Dataset | Mapping[str, xr.DataArray]
    ) -> dict[str, np.ndarray]:
        """Require exactly the decided inputs and labels, returning canonical numeric arrays."""
        if not isinstance(plan, (xr.Dataset, Mapping)):
            raise TypeError(
                "An evaluation plan must be an xarray.Dataset or a mapping of DataArrays."
            )
        names = set(plan.data_vars) if isinstance(plan, xr.Dataset) else set(plan)
        missing = set(self.templates).difference(names)
        extra = names.difference(self.templates)
        if missing or extra:
            raise ValueError(
                f"An evaluation plan requires exactly {sorted(self.templates)}; "
                f"missing {sorted(missing)}, extra {sorted(extra)}."
            )
        arrays = {}
        for name, template in self.templates.items():
            value = plan[name]
            if not isinstance(value, xr.DataArray):
                raise TypeError(
                    f"Plan input {name!r} must be a labeled xarray.DataArray."
                )
            if set(value.dims) != set(template.dims):
                raise ValueError(
                    f"Input {name!r} must have dimensions {template.dims}, got {value.dims}."
                )
            for dim in value.dims:
                if dim not in value.coords or value.coords[dim].dims != (dim,):
                    raise ValueError(
                        f"Plan input {name!r} dimension {dim!r} must label its own dimension."
                    )
            value = _align_labels(value, template).transpose(*template.dims)
            value = validate_dataset(value.to_dataset(name=name))[name]
            value = BuildContext._canonical_input(value, name)
            arrays[name] = value.values
        return arrays

    def compile(self) -> None:
        """Compile numeric evaluation once without evaluating any placeholder plan."""
        if self._numeric is not None:
            return
        tensors = {
            name: pt.tensor(name=name, dtype="float64", shape=template.shape)
            for name, template in self.templates.items()
        }
        values = self(
            {
                name: as_xtensor(tensors[name], dims=template.dims)
                for name, template in self.templates.items()
            }
        )
        outputs = rewrite_graph(
            [value.values for value in values.values()], include=_LOWER
        )
        self._numeric = pytensor.function(
            list(tensors.values()),
            outputs,
            on_unused_input="ignore",
        )

    def evaluate(self, plan: xr.Dataset | Mapping[str, xr.DataArray]) -> xr.Dataset:
        """Evaluate a decided plan on the already bound joint posterior draws.

        Parameters
        ----------
        plan : xarray.Dataset or mapping of str to xarray.DataArray
            Exactly the decided inputs with the bound dimensions and coordinate labels.

        Returns
        -------
        xarray.Dataset
            Selected original-unit outputs over decided and carryover dates, excluding history.
            Only posterior-dependent outputs carry ``sample``.
        """
        arrays = self.validate_plan(plan)
        self.compile()
        numeric = cast(Callable[..., list[np.ndarray]], self._numeric)(*arrays.values())
        return xr.Dataset(
            {
                name: xr.DataArray(
                    value,
                    dims=output.dims,
                    coords=_coordinates(
                        {dim: self.output_coords[dim] for dim in output.dims}
                    ),
                )
                for (name, output), value in zip(
                    self.outputs.items(), numeric, strict=True
                )
            }
        )


def _bind_terms(
    scm: SCM,
    terms: Mapping[str, Any],
    data: xr.Dataset,
    history: xr.Dataset | None = None,
    carryover: int | xr.Dataset = 0,
) -> _Evaluator:
    """Reuse fitted term provenance, freeze the joint posterior, and bind only selected inputs."""
    scm._check_fitted()
    fitted = cast(BuildContext, scm._context)
    admitted = tuple(walk(terms, stop=lambda node: id(node) in fitted.variables))
    if any(isinstance(node, Equation) for node in admitted):
        raise ValueError(
            "Optimizer targets and dependencies must be deterministic terms, not stochastic Equations."
        )
    required = _data_dependencies(terms, scm)
    scenario, templates = _scenario_data(scm, required, data)
    output_scenario, tail = _append_carryover(scm, scenario, templates, carryover)
    combined, fixed, history_length = _prepend_history(
        output_scenario,
        {name: output_scenario[name] for name in templates},
        history,
    )
    for node in admitted:
        if isinstance(node, (Parameter, Dot)) and id(node) not in fitted.variables:
            raise ValueError(
                "Selected terms contain a stochastic parameter not bound to this fitted model; "
                "use its actual fitted parameter objects, not new parameters with matching names."
            )
    with pm.Model(
        coords={
            dim: coord.values
            for dim, coord in combined.coords.items()
            if dim in combined.dims
        }
    ) as model:
        context = BuildContext(
            combined,
            bindings=scm._bindings,
            prediction=True,
            allow_constant_date_subset=True,
        )
        # Reuse the fitted deterministic graph and its actual random-variable identities.
        context.variables.update(fitted.variables)
        for node in admitted:
            if isinstance(node, Ref):
                context.variables[id(node)] = fitted.model[node.name]
        context._labeled_constants.update(fitted._labeled_constants)
        context._suggested_names.update(fitted._suggested_names)
        context._fourier_terms.update(fitted._fourier_terms)
        context._names.update(fitted._names)
        containers = {
            name: cast(XTensorSharedVariable, context.data(name)) for name in templates
        }
        built = {name: as_xtensor(context.build(term)) for name, term in terms.items()}
    blockers = set(fitted.model.free_RVs) | set(fitted.model.observed_RVs)
    selected = set(ancestors(list(built.values()), blockers=blockers))
    legacy_inputs = sorted(
        f"{raw!r} ({getattr(value, 'dtype', None)})"
        for value, raw in _fitted_inputs(scm).items()
        if value in selected and getattr(value, "dtype", None) != "float64"
    )
    if legacy_inputs:
        raise ValueError(
            f"Selected fitted inputs {legacy_inputs} do not have canonical float64 graph dtype. "
            "Rebuild and refit the model, or save/reload it and verify that the reloaded fitted input "
            "graphs use float64 before optimizing."
        )
    dated_constants = _dated_constant_replacements(context, built, blockers)
    _check_temporal_dates(scm, terms, context, built)
    if history_length or (history is not None and history.sizes.get("date", 0) > 0):
        _check_history_inputs(scm, terms, context, built, templates, dated_constants)
    else:
        inputs = _fitted_inputs(scm)
        inputs.update(_input_provenance(context, templates))
        _check_history_dates(
            scm,
            terms,
            context,
            built,
            inputs,
            templates,
            dated_constants,
            history=False,
        )
    if dated_constants:
        # Rebind before posterior graph rewrites can fold or replace a recorded leaf.
        # Stop at actual RV identities; clone_replace would clone their owned graphs again.
        constant_outputs = list(built.values())
        dated_replacements = {**dated_constants, **{rv: rv for rv in blockers}}
        memo = clone_get_equiv(
            list(truncated_graph_inputs(constant_outputs, dated_replacements)),
            constant_outputs,
            copy_inputs=False,
            copy_orphans=False,
            memo=dated_replacements,
            strict=False,
        )
        built = {
            name: cast(XTensorVariable, memo[value]) for name, value in built.items()
        }
        # Shape proofs must follow the Fourier output that actually survives rebinding.
        for _, term in context._fourier_terms.values():
            key = id(term)
            value = context.variables.get(key)
            if isinstance(value, XTensorVariable) and value in memo:
                context.variables[key] = memo[value]
    built, samples = _condition_outputs(scm, built, combined, context)
    replacements: dict[Variable, Variable] = {
        original: containers[raw]
        for original, raw in _fitted_inputs(scm).items()
        if raw in containers
    }
    replacements.update(
        {
            original: containers[raw]
            for original, raw in _input_provenance(context, templates).items()
        }
    )
    for dim, length in fitted.model.dim_lengths.items():
        if dim in combined.dims:
            replacements[length] = pt.as_tensor_variable(np.int64(combined.sizes[dim]))
    graph_nodes = set(ancestors(list(built.values())))
    for fourier, _ in fitted._fourier_terms.values():
        name = f"_{fourier.prefix}_dayofperiod"
        original = fitted.model[name]
        if original in graph_nodes:
            if "date" not in combined.dims:
                raise ValueError(
                    "Selected Fourier terms require scenario date coordinates."
                )
            days = fourier._get_days_in_period(_dates(combined.coords["date"].values))
            replacements[original] = as_xtensor(days.to_numpy(), dims=("date",))
    values = clone_replace(
        list(built.values()), replace=replacements, rebuild_strict=False
    )
    outputs = dict(zip(built, cast(list[XTensorVariable], values), strict=True))
    if history_length:
        outputs = {
            name: output.isel(date=slice(history_length, None))
            if "date" in output.dims
            else output
            for name, output in outputs.items()
        }
    if rvs_in_graph(list(outputs.values())):
        raise ValueError(
            "Selected outputs still contain stochastic variables after posterior conditioning."
        )
    allowed = set(containers.values())
    unresolved = {
        node.name
        for node in ancestors(list(outputs.values()))
        if isinstance(node, SharedVariable) and node not in allowed
    }
    if unresolved:
        raise ValueError(
            f"Selected terms contain unresolved shared inputs {sorted(unresolved, key=str)}; "
            "declare their raw data dependencies through Data or ModelTerm.data_vars."
        )
    coords = {
        dim: scenario.get_index(dim)
        if dim in scenario.dims
        else pd.Index(labels, name=dim, tupleize_cols=False)
        for dim, labels in model.coords.items()
        if labels is not None
    }
    if samples is not None:
        if "sample" in coords and not coords["sample"].equals(samples):
            raise ValueError(
                "Selected term sample labels must match the fitted joint posterior sample labels."
            )
        coords["sample"] = samples
    output_coords = dict(coords)
    if "date" in output_scenario.dims:
        output_coords["date"] = output_scenario.get_index("date")
    missing = {
        dim
        for output in outputs.values()
        for dim in output.dims
        if dim not in output_coords
    }
    if missing:
        raise ValueError(
            f"Selected outputs require labeled coordinates for dimensions {sorted(missing)}."
        )
    return _Evaluator(
        outputs, containers, templates, coords, output_coords, fixed, tail
    )
